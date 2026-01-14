#!/usr/bin/env python3
"""
ColabFold Master Service
Manages ColabFold job distribution and result collection using Celery
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import redis
from celery import Celery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery('colabfold_master')
celery_app.config_from_object('celeryconfig')

# Redis client for job tracking
redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

class ColabFoldMaster:
    def __init__(self, input_dir: str, output_dir: str, processed_dir: str = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.processed_dir = Path(processed_dir) if processed_dir else self.input_dir / "processed"
        
        # Create directories if they don't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Track processed files
        self.processed_files = set()
        self.running_jobs = {}
        
        logger.info(f"ColabFold Master initialized:")
        logger.info(f"  Input directory: {self.input_dir}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Processed directory: {self.processed_dir}")
    
    def is_fasta_file(self, file_path: Path) -> bool:
        """Check if file is a FASTA file"""
        fasta_extensions = ['.fasta', '.fa', '.fas', '.fna', '.ffn', '.faa', '.frn']
        return any(str(file_path).lower().endswith(ext) for ext in fasta_extensions)
    
    def submit_colabfold_job(self, input_file: Path) -> str:
        """Submit ColabFold job to worker container"""
        job_name = input_file.stem
        job_output_dir = self.output_dir / job_name
        job_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default ColabFold arguments
        colabfold_args = "--num-models 1 --model-type alphafold2_ptm --no-templates"
        
        # Find available worker container
        worker_container = self.get_available_worker()
        if not worker_container:
            raise Exception("No available ColabFold worker containers")
        
        # Create job metadata
        job_id = f"colabfold_{int(datetime.now().timestamp())}_{job_name}"
        
        # Get worker ID for temp directory
        worker_id = worker_container.split('-')[-1]  # Extract number from container name
        
        # Prepare paths using shared volumes
        shared_input_file = f"/colabfold/input/{input_file.name}"
        worker_temp_dir = f"/colabfold/temp/{worker_id}/{job_id}"
        worker_temp_input = f"{worker_temp_dir}/{input_file.name}"
        worker_temp_output = f"{worker_temp_dir}/output"
        worker_result_file = f"{worker_temp_dir}/result.json"
        
        job_metadata = {
            'job_id': job_id,
            'input_file': str(input_file),
            'output_dir': str(job_output_dir),
            'status': 'submitted',
            'submitted_at': datetime.now().isoformat(),
            'job_name': job_name,
            'worker_container': worker_container,
            'worker_id': worker_id,
            'colabfold_args': colabfold_args,
            'shared_input_file': shared_input_file,
            'worker_temp_dir': worker_temp_dir,
            'worker_temp_input': worker_temp_input,
            'worker_temp_output': worker_temp_output,
            'worker_result_file': worker_result_file
        }
        
        redis_client.hset(f"colabfold_job:{job_id}", mapping=job_metadata)
        redis_client.hset(f"colabfold_job:{job_id}", "master_tracking", "true")
        
        # Start job in worker container
        self.start_worker_job(worker_container, job_id, job_metadata)
        
        self.running_jobs[str(input_file)] = {
            'job_id': job_id,
            'worker_container': worker_container,
            'worker_id': worker_id,
            'status': 'submitted',
            'submitted_at': datetime.now()
        }
        
        logger.info(f"Submitted ColabFold job {job_id} to worker {worker_container}")
        return job_id
    
    def get_available_worker(self) -> str:
        """Find available worker container"""
        import subprocess
        
        try:
            # Get running worker containers
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=avnn-colabfold-worker", "--format", "{{.Names}}"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                workers = result.stdout.strip().split('\n')
                return workers[0] if workers and workers[0] else None
            
        except Exception as e:
            logger.error(f"Error finding workers: {str(e)}")
        
        return None
    
    def start_worker_job(self, worker_container: str, job_id: str, job_metadata: Dict):
        """Start ColabFold job in worker container using shared volumes"""
        import subprocess
        
        worker_temp_input = job_metadata['worker_temp_input']
        worker_temp_output = job_metadata['worker_temp_output']
        worker_result_file = job_metadata['worker_result_file']
        shared_input_file = job_metadata['shared_input_file']
        colabfold_args = job_metadata['colabfold_args']
        
        try:
            # Create worker temp directory
            mkdir_cmd = ["docker", "exec", worker_container, "mkdir", "-p", job_metadata['worker_temp_dir']]
            subprocess.run(mkdir_cmd, check=True)
            
            # Move input file from shared input to worker temp (atomic operation)
            mv_cmd = ["docker", "exec", worker_container, "mv", shared_input_file, worker_temp_input]
            subprocess.run(mv_cmd, check=True)
            
            # Run ColabFold job in worker container
            cmd = [
                "docker", "exec", worker_container,
                "python", "/colabfold/worker/colabfold_worker.py",
                "--input-file", worker_temp_input,
                "--output-dir", worker_temp_output,
                "--extra-args", colabfold_args,
                "--result-file", worker_result_file
            ]
            
            # Start job in background
            subprocess.Popen(cmd)
            
            # Update job status
            redis_client.hset(f"colabfold_job:{job_id}", "status", "running")
            
            logger.info(f"Started job {job_id} in worker {worker_container}")
            
        except Exception as e:
            logger.error(f"Error starting job in worker {worker_container}: {str(e)}")
            redis_client.hset(f"colabfold_job:{job_id}", "status", "error")
            redis_client.hset(f"colabfold_job:{job_id}", "error", str(e))
    
    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """Check status of a ColabFold job"""
        import subprocess
        
        try:
            # Get job data from Redis
            job_data = redis_client.hgetall(f"colabfold_job:{job_id}")
            if not job_data:
                return {'status': 'not_found'}
            
            worker_container = job_data.get('worker_container')
            worker_result_file = job_data.get('worker_result_file')
            worker_temp_output = job_data.get('worker_temp_output')
            
            if not worker_container or not worker_result_file:
                return {'status': 'error', 'error': 'Missing job metadata'}
            
            # Check if result file exists in worker container
            result_cmd = ["docker", "exec", worker_container, "test", "-f", worker_result_file]
            result_check = subprocess.run(result_cmd, capture_output=True)
            
            if result_check.returncode == 0:
                # Job completed - get result
                cat_cmd = ["docker", "exec", worker_container, "cat", worker_result_file]
                cat_result = subprocess.run(cat_cmd, capture_output=True, text=True)
                
                if cat_result.returncode == 0:
                    try:
                        result_data = json.loads(cat_result.stdout)
                        
                        if result_data.get('success'):
                            # Move results from worker temp to shared output
                            self.move_results_to_output(worker_container, job_id, job_data, result_data)
                            
                            return {
                                'status': 'completed',
                                'result': result_data,
                                'completed_at': datetime.now().isoformat()
                            }
                        else:
                            return {
                                'status': 'failed',
                                'error': result_data.get('error', 'Unknown error'),
                                'failed_at': datetime.now().isoformat()
                            }
                    except json.JSONDecodeError:
                        return {'status': 'error', 'error': 'Invalid result JSON'}
                else:
                    return {'status': 'error', 'error': 'Could not read result file'}
            else:
                # Check if process is still running
                ps_cmd = ["docker", "exec", worker_container, "pgrep", "-f", "colabfold_worker"]
                ps_check = subprocess.run(ps_cmd, capture_output=True)
                
                if ps_check.returncode == 0:
                    return {'status': 'running'}
                else:
                    return {'status': 'error', 'error': 'Worker process not found'}
                    
        except Exception as e:
            logger.error(f"Error checking job {job_id}: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def move_results_to_output(self, worker_container: str, job_id: str, job_data: Dict, result_data: Dict):
        """Move results from worker temp to shared output directory"""
        import subprocess
        
        try:
            worker_temp_output = job_data.get('worker_temp_output')
            shared_output_dir = job_data.get('output_dir')
            job_name = job_data.get('job_name')
            
            # Move results from worker temp to shared output
            shared_output_path = f"/colabfold/output/{job_name}"
            mv_cmd = ["docker", "exec", worker_container, "mv", worker_temp_output, shared_output_path]
            subprocess.run(mv_cmd, check=True)
            
            # Clean up worker temp directory
            worker_temp_dir = job_data.get('worker_temp_dir')
            cleanup_cmd = ["docker", "exec", worker_container, "rm", "-rf", worker_temp_dir]
            subprocess.run(cleanup_cmd)
            
            logger.info(f"Moved results from {worker_temp_output} to {shared_output_path}")
            
        except Exception as e:
            logger.error(f"Error moving results: {str(e)}")
    
    def process_completed_job(self, task_id: str):
        """Process a completed ColabFold job"""
        try:
            job_data = redis_client.hgetall(f"colabfold_job:{task_id}")
            if not job_data:
                logger.warning(f"No job data found for task {task_id}")
                return
            
            input_file = Path(job_data['input_file'])
            output_dir = Path(job_data['output_dir'])
            
            # Check job status
            status = self.check_job_status(task_id)
            
            if status['status'] == 'completed':
                logger.info(f"Job {task_id} completed successfully")
                
                # Find output PDB file
                pdb_file = None
                if output_dir.exists():
                    for file in output_dir.glob("*_rank_1_*.pdb"):
                        pdb_file = file
                        break
                
                # Update job metadata
                job_data.update({
                    'status': 'completed',
                    'completed_at': status['completed_at'],
                    'pdb_file': str(pdb_file) if pdb_file else None,
                    'result': status.get('result', '')
                })
                
                redis_client.hset(f"colabfold_job:{task_id}", mapping=job_data)
                
                # Move input file to processed directory
                processed_file = self.processed_dir / input_file.name
                if input_file.exists():
                    shutil.move(str(input_file), str(processed_file))
                
                logger.info(f"Processed {input_file} -> {processed_file}")
                
            elif status['status'] == 'failed':
                logger.error(f"Job {task_id} failed: {status.get('error', 'Unknown error')}")
                
                # Move to failed directory
                failed_dir = self.output_dir / "failed" / job_data['job_name']
                failed_dir.mkdir(parents=True, exist_ok=True)
                
                if input_file.exists():
                    shutil.move(str(input_file), str(failed_dir / input_file.name))
                
                # Save error info
                error_file = failed_dir / "error.txt"
                with open(error_file, 'w') as f:
                    f.write(f"ColabFold job {task_id} failed\n\n")
                    f.write(f"Error: {status.get('error', 'Unknown error')}\n")
                
                # Update job metadata
                job_data.update({
                    'status': 'failed',
                    'failed_at': status['failed_at'],
                    'error': status.get('error', '')
                })
                
                redis_client.hset(f"colabfold_job:{task_id}", mapping=job_data)
            
            # Clean up running jobs tracking
            if 'input_file' in job_data and job_data['input_file'] in self.running_jobs:
                del self.running_jobs[job_data['input_file']]
            
        except Exception as e:
            logger.error(f"Error processing completed job {task_id}: {str(e)}")
    
    def scan_input_directory(self):
        """Scan input directory for new files"""
        if not self.input_dir.exists():
            logger.warning(f"Input directory {self.input_dir} does not exist")
            return
        
        for file_path in self.input_dir.iterdir():
            if file_path.is_file() and file_path not in self.processed_files:
                if self.is_fasta_file(file_path):
                    self.submit_colabfold_job(file_path)
                    self.processed_files.add(file_path)
    
    def check_running_jobs(self):
        """Check status of running jobs"""
        completed_jobs = []
        
        for input_file, job_info in list(self.running_jobs.items()):
            task_id = job_info['task_id']
            status = self.check_job_status(task_id)
            
            if status['status'] in ['completed', 'failed', 'error']:
                completed_jobs.append(task_id)
                self.process_completed_job(task_id)
        
        return completed_jobs
    
    def get_job_statistics(self) -> Dict[str, Any]:
        """Get statistics about ColabFold jobs"""
        # Count jobs by status from Redis
        all_jobs = []
        for key in redis_client.scan_iter(match="colabfold_job:*"):
            job_data = redis_client.hgetall(key)
            if job_data.get('master_tracking') == 'true':
                all_jobs.append(job_data)
        
        stats = {
            'total_jobs': len(all_jobs),
            'submitted': len([j for j in all_jobs if j['status'] == 'submitted']),
            'running': len([j for j in all_jobs if j['status'] == 'running']),
            'completed': len([j for j in all_jobs if j['status'] == 'completed']),
            'failed': len([j for j in all_jobs if j['status'] == 'failed']),
            'currently_running': len(self.running_jobs)
        }
        
        return stats
    
    def monitor(self, interval: int = 10):
        """Main monitoring loop"""
        logger.info(f"Starting ColabFold Master with {interval}s interval")
        
        try:
            while True:
                # Scan for new files
                self.scan_input_directory()
                
                # Check running jobs
                self.check_running_jobs()
                
                # Log statistics
                stats = self.get_job_statistics()
                logger.info(f"Job stats: {stats}")
                
                time.sleep(interval)
        
        except KeyboardInterrupt:
            logger.info("ColabFold Master stopped by user")
        except Exception as e:
            logger.error(f"ColabFold Master error: {str(e)}")
            raise

def main():
    """Main entry point"""
    import argparse
    import shutil
    
    parser = argparse.ArgumentParser(description='ColabFold Master Service')
    parser.add_argument('--input-dir', required=True, help='Input directory for FASTA files')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--processed-dir', help='Directory for processed input files')
    parser.add_argument('--interval', type=int, default=10, help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    # Create and start master
    master = ColabFoldMaster(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        processed_dir=args.processed_dir
    )
    
    master.monitor(interval=args.interval)

if __name__ == '__main__':
    main()
