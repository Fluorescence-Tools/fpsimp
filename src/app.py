#!/usr/bin/env python3
"""
FPSIMP: Fluorescent Protein Simulation Pipeline
Standalone deployment - no external dependencies
"""
import os
import uuid
import json
import tempfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from celery.result import AsyncResult
import subprocess
import shutil

from flask import Flask, request, jsonify, send_file, abort
from jinja2 import Environment, FileSystemLoader
from flask_cors import CORS
from werkzeug.utils import secure_filename
from celery import Celery
import redis

# Import configuration first
from config import config

# Import local fpsim components (standalone module in web-app)
from fpsim.pipeline import PipelineConfig, run_fpsim_pipeline
from fpsim.utils import read_fasta_with_selection, extract_sequences_from_structure, write_fasta_from_pdb
from fpsim.fp_lib import get_fp_library

print("Local fpsim components imported successfully")

# Define list_fasta_sequences function (always use our implementation)
def list_fasta_sequences(fasta_path):
    try:
        from Bio import SeqIO
        sequences = []
        
        with open(fasta_path, 'r') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                sequence_str = str(record.seq)
                
                # Detect fluorescent protein
                fp_info = detect_fluorescent_protein(sequence_str)
                
                # Keep the full sequence as-is, don't split by colon in backend
                seq_data = {
                    'id': record.id,
                    'description': record.description,
                    'length': len(record.seq),
                    'sequence': sequence_str
                }
                
                # Add FP info if detected
                if fp_info:
                    seq_data['fp_name'] = fp_info['name']
                    seq_data['fp_color'] = fp_info['color']
                    seq_data['fp_match_type'] = fp_info['match_type']
                    if 'motif' in fp_info:
                        seq_data['fp_motif'] = fp_info['motif']
                    if 'fp_start' in fp_info:
                        seq_data['fp_start'] = fp_info['fp_start']
                    if 'fp_end' in fp_info:
                        seq_data['fp_end'] = fp_info['fp_end']
                    if 'fps' in fp_info:
                        seq_data['fps'] = fp_info['fps']
                    if 'dipole_triplets' in fp_info:
                        seq_data['dipole_triplets'] = fp_info['dipole_triplets']
                
                sequences.append(seq_data)
        
        print(f"Successfully parsed {len(sequences)} sequences/chains from {fasta_path}")
        for i, seq in enumerate(sequences, 1):
            print(f"  {i}. {seq['id']} ({seq['length']} residues)")
            
        return sequences
        
    except Exception as e:
        print(f"Error parsing FASTA file {fasta_path} with Biopython: {str(e)}")
        raise

def run_colabfold_container(
    fasta: Path,
    out_dir: Path,
    extra_args: str = "--num-models 1",
    container_name: str = "avnn-colabfold"
) -> Path:
    """Run ColabFold in a dedicated Docker container.
    
    Args:
        fasta: Path to input FASTA file
        out_dir: Directory to write output files
        extra_args: Additional arguments to pass to colabfold_batch
        container_name: Name of the ColabFold container
        
    Returns:
        Path to the best ranked PDB file
    """
    import shlex
    import subprocess
    from pathlib import Path
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Build docker exec command
    colabfold_bin = "/app/colabfold_batch"
    
    # Prepare paths for container (mount input/output)
    container_fasta = f"/tmp/{fasta.name}"
    container_out = "/tmp/output"
    
    # Copy input file to container
    copy_cmd = ["docker", "cp", str(fasta), f"{container_name}:{container_fasta}"]
    subprocess.run(copy_cmd, check=True)
    
    # Build colabfold command
    cmd_parts = ["docker", "exec", container_name, colabfold_bin]
    if extra_args.strip():
        cmd_parts.extend(shlex.split(extra_args.strip()))
    cmd_parts.extend([container_fasta, container_out])
    
    app.logger.info(f"Running ColabFold in container: {' '.join(cmd_parts)}")
    
    try:
        # Run ColabFold in container
        result = subprocess.run(
            cmd_parts,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            app.logger.info(f"ColabFold stdout: {result.stdout}")
        if result.stderr:
            app.logger.warning(f"ColabFold stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        error_msg = f"ColabFold container execution failed with code {e.returncode}"
        if e.stderr:
            error_msg += f"\nError output:\n{e.stderr}"
        if e.stdout:
            error_msg += f"\nOutput:\n{e.stdout}"
        raise RuntimeError(error_msg)
    
    # Copy results back from container
    container_results = f"{container_name}:{container_out}/"
    copy_result_cmd = ["docker", "cp", container_results, str(out_dir.parent)]
    subprocess.run(copy_result_cmd, check=True)
    
    # Find the best PDB
    results_dir = out_dir.parent / "output"
    if not results_dir.exists():
        raise RuntimeError("No results directory found from ColabFold container")
    
    # Look for ranked PDB files
    pdbs = sorted(results_dir.glob("*rank_1*.pdb"))
    if not pdbs:
        pdbs = sorted(results_dir.glob("*.pdb"))
    
    if not pdbs:
        raise RuntimeError("No PDB files produced by ColabFold container")
    
    best_pdb = pdbs[0]
    
    # Move to expected location
    final_pdb = out_dir / best_pdb.name
    best_pdb.rename(final_pdb)
    
    app.logger.info(f"ColabFold container completed. Using PDB: {final_pdb}")
    return final_pdb

app = Flask(__name__, static_folder='../static', static_url_path='/static')
CORS(app)

# Ensure proper MIME types for ES6 modules
@app.after_request
def after_request(response):
    if request.path.endswith('.js'):
        response.headers['Content-Type'] = 'application/javascript; charset=utf-8'
    return response

# Initialize Jinja2 environment for templates in static folder
template_env = Environment(loader=FileSystemLoader('/app/static'))

def render_static_template(template_name, **context):
    """Render a template from the static folder"""
    template = template_env.get_template(template_name)
    
    # Add url_for function to template context
    def url_for(endpoint, **kwargs):
        if 'filename' in kwargs:
            return f"/static/{kwargs['filename']}"
        elif kwargs:
            # Handle routes with parameters like job_id
            params = '/'.join(str(v) for v in kwargs.values())
            return f"/{endpoint}/{params}"
        else:
            return f"/{endpoint}"
    
    context['url_for'] = url_for
    return template.render(**context)

# Initialize Celery
celery = Celery(app.import_name)

# Load all configurations from the config object
app.config.from_object(config)
celery.config_from_object(config)

# Add DISABLE_COLABFOLD environment variable - default to true
app.config['DISABLE_COLABFOLD'] = os.environ.get('DISABLE_COLABFOLD', 'true').lower() == 'true'

# Custom Jinja2 filters
def format_file_size(size):
    """Format file size in human-readable format"""
    try:
        size = float(size)
    except (ValueError, TypeError):
        return "Unknown"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

app.jinja_env.filters['get_file_size'] = format_file_size

# Update Celery configuration with additional settings
celery.conf.update({
    'worker_enable_remote_control': True,
    'worker_send_task_events': True,
    'task_send_sent_event': True,
    'worker_prefetch_multiplier': 1,
    'broker_connection_retry_on_startup': True,
    'worker_hijack_root_logger': False,
    'worker_env': {
        'COLABFOLD_PATH': os.getenv('COLABFOLD_PATH', '/opt/localcolabfold/colabfold-conda/bin'),
        'PATH': f"{os.getenv('COLABFOLD_PATH', '/opt/localcolabfold/colabfold-conda/bin')}:{os.environ.get('PATH', '')}"
    } if os.getenv('COLABFOLD_PATH') else {}
})

# Set ColabFold path from environment variable or use default
colabfold_path = os.getenv('COLABFOLD_PATH')
if colabfold_path and os.path.exists(colabfold_path):
    os.environ['COLABFOLD_PATH'] = colabfold_path
    print(f"Using ColabFold from: {colabfold_path}")

# Redis client for job status
redis_client = redis.Redis.from_url(
    url=config.broker_url,  # Use the modern lowercase config key
    db=1,  # Use a different DB than Celery
    decode_responses=True
)

# Initialize FP library
FP_LIBRARY, FP_COLOR, FP_MOTIFS, FP_DIPOLE_TRIPLETS = get_fp_library()

def detect_fluorescent_protein(sequence: str) -> Optional[Dict[str, str]]:
    """Detect if a sequence contains a known fluorescent protein.
    
    Returns dict with 'name', 'color', 'fp_start', 'fp_end' if detected, None otherwise.
    For sequences with multiple FPs, returns info about all detected FPs.
    """
    sequence = sequence.upper().replace(' ', '').replace('\n', '')
    
    # Check full sequence match first
    for fp_name, fp_seq in FP_LIBRARY.items():
        if fp_seq.upper() == sequence:
            return {
                'name': fp_name,
                'color': FP_COLOR.get(fp_name, 'unknown'),
                'match_type': 'full',
                'fp_start': 0,
                'fp_end': len(sequence) - 1,
                'dipole_triplets': FP_DIPOLE_TRIPLETS.get(fp_name, []),
                'fps': [{
                    'name': fp_name,
                    'color': FP_COLOR.get(fp_name, 'unknown'),
                    'start': 0,
                    'end': len(sequence) - 1,
                    'dipole_triplets': FP_DIPOLE_TRIPLETS.get(fp_name, [])
                }]
            }
    
    # Collect all FP matches in the sequence
    all_fps = []
    
    # Check motif matches and try to find full FP regions
    for fp_name, motifs in FP_MOTIFS.items():
        for motif in motifs:
            motif_upper = motif.upper()
            if motif_upper in sequence:
                fp_seq = FP_LIBRARY[fp_name].upper()
                motif_pos = sequence.find(motif_upper)
                motif_in_fp = fp_seq.find(motif_upper)
                
                # Calculate likely FP boundaries
                if motif_in_fp >= 0:
                    # Motif found in library - align based on library position
                    likely_start = max(0, motif_pos - motif_in_fp)
                    likely_end = min(len(sequence) - 1, likely_start + len(fp_seq) - 1)
                else:
                    # Motif not in library (variant) - estimate boundaries
                    # Assume FP is roughly library length, centered around motif
                    likely_start = max(0, motif_pos - len(fp_seq) // 4)
                    likely_end = min(len(sequence) - 1, likely_start + len(fp_seq) - 1)
                
                # If sequence length close to FP length, assume entire sequence is single FP
                if abs(len(sequence) - len(fp_seq)) < 20:
                    return {
                        'name': fp_name,
                        'color': FP_COLOR.get(fp_name, 'unknown'),
                        'match_type': 'motif',
                        'motif': motif,
                        'fp_start': 0,
                        'fp_end': len(sequence) - 1,
                        'dipole_triplets': FP_DIPOLE_TRIPLETS.get(fp_name, []),
                        'fps': [{
                            'name': fp_name,
                            'color': FP_COLOR.get(fp_name, 'unknown'),
                            'start': 0,
                            'end': len(sequence) - 1,
                            'dipole_triplets': FP_DIPOLE_TRIPLETS.get(fp_name, [])
                        }]
                    }
                
                # Check if we already added this exact FP (same name in similar position)
                duplicate = False
                for existing_fp in all_fps:
                    if existing_fp['name'] == fp_name and abs(existing_fp['start'] - likely_start) < 50:
                        duplicate = True
                        break
                
                if not duplicate:
                    all_fps.append({
                        'name': fp_name,
                        'color': FP_COLOR.get(fp_name, 'unknown'),
                        'start': likely_start,
                        'end': likely_end,
                        'motif': motif,
                        'dipole_triplets': FP_DIPOLE_TRIPLETS.get(fp_name, [])
                    })
    
    if all_fps:
        # Sort by start position
        all_fps.sort(key=lambda x: x['start'])
        
        # Return info about first FP for backward compatibility, plus list of all FPs
        first_fp = all_fps[0]
        fp_names = ', '.join([fp['name'] for fp in all_fps])
        
        return {
            'name': fp_names if len(all_fps) > 1 else first_fp['name'],
            'color': first_fp['color'],
            'match_type': 'motif',
            'motif': first_fp['motif'],
            'fp_start': first_fp['start'],
            'fp_end': first_fp['end'],
            'dipole_triplets': first_fp.get('dipole_triplets', []),
            'fps': all_fps
        }
    
    return None

class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    COLABFOLD_RUNNING = "colabfold_running"
    COLABFOLD_COMPLETE = "colabfold_complete"
    SAMPLING_COMPLETE = "sampling_complete"
    COMPLETED = "completed"
    FAILED = "failed"

@celery.task(bind=True, name='run_colabfold_task')
def run_colabfold_task(self, fasta_path: str, output_dir: str, extra_args: str = "--num-models 1", gpu: int = None):
    """ColabFold task that uses shared volumes for processing.
    
    Args:
        fasta_path: Path to input FASTA file
        output_dir: Directory to write output files
        extra_args: Additional arguments to pass to colabfold_batch
        gpu: GPU device ID to use
        
    Returns:
        Path to the best ranked PDB file
    """
    import os
    import json
    import time
    import subprocess
    from pathlib import Path
    
    # Get job_id from Redis
    job_id = redis_client.hget(f"colabfold_task:{self.request.id}", "job_id")
    if job_id:
        job_id = job_id.decode('utf-8')
    
    self.update_state(state='PROGRESS', meta={'status': 'Setting up ColabFold processing...'})
    
    try:
        # Update job status
        if job_id:
            redis_client.hset(f"job:{job_id}", "status", JobStatus.COLABFOLD_RUNNING)
            redis_client.hset(f"job:{job_id}", "colabfold_started_at", datetime.now().isoformat())
        
        # Copy FASTA file to shared ColabFold input directory
        input_file = Path(fasta_path)
        if not input_file.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        
        # Use shared volume paths
        colabfold_input_dir = Path("/colabfold/input")
        colabfold_output_dir = Path("/colabfold/output")
        colabfold_temp_dir = Path("/colabfold/temp")
        
        # Ensure directories exist
        colabfold_input_dir.mkdir(parents=True, exist_ok=True)
        colabfold_output_dir.mkdir(parents=True, exist_ok=True)
        colabfold_temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy input file to shared input directory
        shared_input_file = colabfold_input_dir / input_file.name
        import shutil
        shutil.copy2(input_file, shared_input_file)
        
        self.update_state(state='PROGRESS', meta={'status': 'File copied to shared input, waiting for ColabFold master...'})
        
        # Simulate ColabFold master processing by manually running the workflow
        # In production, the ColabFold master would handle this automatically
        output_name = input_file.stem
        shared_output_dir = colabfold_output_dir / output_name
        
        # Wait for processing (simulate master processing)
        max_wait_time = 3600  # 1 hour max wait
        poll_interval = 30   # Check every 30 seconds
        waited_time = 0
        
        while waited_time < max_wait_time:
            # Check if output directory exists and has PDB files
            if shared_output_dir.exists():
                pdb_files = list(shared_output_dir.glob("*.pdb"))
                if pdb_files:
                    # Copy results back to requested output directory
                    local_output_dir = Path(output_dir)
                    local_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy all output files
                    for output_file in shared_output_dir.glob("*"):
                        if output_file.is_file():
                            shutil.copy2(output_file, local_output_dir / output_file.name)
                    
                    # Find the best PDB file
                    best_pdb = local_output_dir / pdb_files[0].name
                    
                    # Update job status with success
                    if job_id:
                        redis_client.hset(f"job:{job_id}", "status", JobStatus.COLABFOLD_COMPLETE)
                        redis_client.hset(f"job:{job_id}", "colabfold_completed_at", datetime.now().isoformat())
                        redis_client.hset(f"job:{job_id}", "af_pdb", str(best_pdb))
                        redis_client.hdel(f"job:{job_id}", "run_colabfold")  # Mark as completed
                    
                    self.update_state(state='SUCCESS', meta={'status': 'ColabFold completed', 'pdb_path': str(best_pdb)})
                    return str(best_pdb)
            
            # Update progress
            self.update_state(state='PROGRESS', meta={'status': f'Waiting for processing... ({waited_time//60}m {waited_time%60}s elapsed)'})
            
            # Wait before next check
            time.sleep(poll_interval)
            waited_time += poll_interval
        
        # Timeout reached - try manual processing as fallback
        self.update_state(state='PROGRESS', meta={'status': 'Starting manual ColabFold processing...'})
        
        # Manual processing using clean workers
        worker_id = "1"
        job_timestamp = str(int(datetime.now().timestamp()))
        temp_job_dir = colabfold_temp_dir / worker_id / f"manual_job_{job_timestamp}"
        temp_job_dir.mkdir(parents=True, exist_ok=True)
        
        # Move input file to worker temp
        worker_input = temp_job_dir / input_file.name
        shutil.copy2(shared_input_file, worker_input)
        
        # Run ColabFold worker script
        worker_output = temp_job_dir / "output"
        worker_result = temp_job_dir / "result.json"
        
        # Find available worker container
        try:
            # Use the first available worker
            worker_container = "avnn-web-stack-colabfold-worker-1"
            
            # Run the worker script
            cmd = [
                "docker", "exec", worker_container,
                "python", "/colabfold/worker/colabfold_worker.py",
                "--input-file", str(worker_input),
                "--output-dir", str(worker_output),
                "--extra-args", extra_args,
                "--result-file", str(worker_result)
            ]
            
            subprocess.run(cmd, check=True, timeout=3600)
            
            # Check results
            if worker_result.exists():
                with open(worker_result, 'r') as f:
                    result_data = json.load(f)
                
                if result_data.get('success'):
                    # Move results to shared output
                    final_output_dir = colabfold_output_dir / output_name
                    if worker_output.exists():
                        shutil.move(str(worker_output), str(final_output_dir))
                    
                    # Copy to local output directory
                    local_output_dir = Path(output_dir)
                    local_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    for output_file in final_output_dir.glob("*"):
                        if output_file.is_file():
                            shutil.copy2(output_file, local_output_dir / output_file.name)
                    
                    pdb_files = list(local_output_dir.glob("*.pdb"))
                    if pdb_files:
                        best_pdb = local_output_dir / pdb_files[0].name
                        
                        # Update job status with success
                        if job_id:
                            redis_client.hset(f"job:{job_id}", "status", JobStatus.COLABFOLD_COMPLETE)
                            redis_client.hset(f"job:{job_id}", "colabfold_completed_at", datetime.now().isoformat())
                            redis_client.hset(f"job:{job_id}", "af_pdb", str(best_pdb))
                            redis_client.hdel(f"job:{job_id}", "run_colabfold")
                        
                        self.update_state(state='SUCCESS', meta={'status': 'ColabFold completed', 'pdb_path': str(best_pdb)})
                        return str(best_pdb)
            
        except Exception as e:
            raise Exception(f"Manual ColabFold processing failed: {str(e)}")
        
        # Timeout reached
        raise TimeoutError(f"ColabFold processing timed out after {max_wait_time//60} minutes")
        
    except Exception as e:
        # Update job status with failure
        if job_id:
            redis_client.hset(f"job:{job_id}", "status", JobStatus.COLABFOLD_FAILED)
            redis_client.hset(f"job:{job_id}", "colabfold_failed_at", datetime.now().isoformat())
            redis_client.hset(f"job:{job_id}", "error", f"ColabFold failed: {str(e)}")
        
        self.update_state(state='FAILURE', meta={'status': f'ColabFold failed: {str(e)}'})
        raise

@celery.task(bind=True, name='run_pipeline_task')
def run_pipeline_task(self, job_id: str, config_dict: Dict[str, Any], colabfold_result: str = None):
    """Celery task to run fpsim pipeline (after ColabFold is done)"""
    try:
        # Update job status to running
        redis_client.hset(f"job:{job_id}", "status", JobStatus.RUNNING)
        redis_client.hset(f"job:{job_id}", "pipeline_started_at", datetime.now().isoformat())
        redis_client.hset(f"job:{job_id}", "worker_id", self.request.id)

        # If ColabFold result is provided, update config
        if colabfold_result:
            config_dict['af_pdb'] = colabfold_result
            app.logger.info(f"Received ColabFold result: {colabfold_result}")

        # Convert all path parameters to Path objects
        from pathlib import Path
        path_params = ['out_dir', 'fasta', 'af_pdb', 'linker_pdb', 'segments_json']
        for param in path_params:
            if param in config_dict and config_dict[param] is not None:
                config_dict[param] = Path(str(config_dict[param]))

        # Create PipelineConfig from dict
        config = PipelineConfig(**config_dict)

        # Run the pipeline
        outputs = run_fpsim_pipeline(config)
        
        # Convert outputs to serializable format
        def to_serializable(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_serializable(v) for v in obj]
            return obj

        serializable_outputs = to_serializable(outputs)
        serializable_outputs['rmf_files'] = [str(f) for f in Path(outputs.get('sampling', '')).glob('*.rmf*')] if 'sampling' in outputs else []
        
        # Store results
        redis_client.hset(f"job:{job_id}", "status", JobStatus.COMPLETED)
        redis_client.hset(f"job:{job_id}", "completed_at", datetime.now().isoformat())
        redis_client.hset(f"job:{job_id}", "outputs", json.dumps(serializable_outputs))
        
        return {"status": "completed", "outputs": serializable_outputs}
        
    except Exception as e:
        redis_client.hset(f"job:{job_id}", "status", JobStatus.FAILED)
        redis_client.hset(f"job:{job_id}", "error", str(e))
        redis_client.hset(f"job:{job_id}", "failed_at", datetime.now().isoformat())
        raise

@celery.task(bind=True, name='run_fpsim_job')
def run_fpsim_job(self, job_id: str, config_dict: Dict[str, Any]):
    """Celery task to run fpsim pipeline"""
    try:
        # Update job status to running with heartbeat
        redis_client.hset(f"job:{job_id}", "status", JobStatus.RUNNING)
        redis_client.hset(f"job:{job_id}", "started_at", datetime.now().isoformat())
        redis_client.hset(f"job:{job_id}", "heartbeat", datetime.now().isoformat())
        redis_client.hset(f"job:{job_id}", "worker_id", self.request.id)
        
        # Convert all path parameters to Path objects
        from pathlib import Path
        path_params = ['out_dir', 'fasta', 'af_pdb', 'linker_pdb', 'segments_json']
        for param in path_params:
            if param in config_dict and config_dict[param] is not None:
                config_dict[param] = Path(str(config_dict[param]))
        
        # Run ColabFold if requested
        use_structure_plddt = config_dict.get('use_structure_plddt', False)
        provided_structure = config_dict.get('af_pdb')
        run_colabfold_requested = config_dict.get('run_colabfold', False)
        
        # Helper to make config serializable
        def make_serializable(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            if isinstance(obj, tuple):
                return [make_serializable(v) for v in obj]
            return obj

        if run_colabfold_requested and not (use_structure_plddt and provided_structure):
            app.logger.info(f"Queueing ColabFold task before pipeline")
            
            # Create output directory for ColabFold
            job_dir = Path(app.config['RESULTS_FOLDER']) / job_id
            cf_out = job_dir / 'colabfold'
            cf_out.mkdir(parents=True, exist_ok=True)
            
            # Run ColabFold task
            colabfold_args = config_dict.get('colabfold_args', '--num-models 1')
            
            # Serialize config for pipeline task
            config_dict_serializable = make_serializable(config_dict)
            
            pipeline_signature = run_pipeline_task.s(job_id, config_dict_serializable)
            
            colabfold_task = run_colabfold_task.apply_async(
                args=[str(config_dict.get('fasta')), str(cf_out), colabfold_args, config_dict.get('gpu', None)],
                queue='imp_queue',
                link=pipeline_signature
            )
            
            # Update config for pipeline task
            config_dict['run_colabfold'] = False
            config_dict['af_pdb'] = None  # Will be set by pipeline task from result
            
            app.logger.info(f"Chained ColabFold -> Pipeline for job {job_id}")
            return {"status": "queued", "colabfold_task_id": colabfold_task.id, "pipeline_task_id": None}
        
        else:
            # Run pipeline directly
            app.logger.info(f"Running pipeline directly (no ColabFold needed)")
            
            # Serialize config for pipeline task
            config_dict_serializable = make_serializable(config_dict)
            
            run_pipeline_task.apply_async(args=[job_id, config_dict_serializable], queue='imp_queue')
            return {"status": "queued", "pipeline_task_id": None}
        
    except Exception as e:
        redis_client.hset(f"job:{job_id}", "status", JobStatus.FAILED)
        redis_client.hset(f"job:{job_id}", "error", str(e))
        redis_client.hset(f"job:{job_id}", "failed_at", datetime.now().isoformat())
        raise

@app.route('/')
def index():
    """Main page"""
    return render_static_template('index.html', config=config)

@app.route('/results/<job_id>')
def results_page(job_id: str):
    """Results page for a specific job"""
    # Verify job exists
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data:
        return render_static_template('error.html', message='Job not found'), 404
    
    # Get job status
    status = job_data.get('status', 'unknown')
    
    # Parse job parameters if available
    try:
        parameters = json.loads(job_data.get('parameters', '{}'))
    except (json.JSONDecodeError, TypeError):
        parameters = {}
    
    # Get results if job is completed
    results = None
    all_files = []
    if status == JobStatus.COMPLETED:
        try:
            results = json.loads(job_data.get('outputs', '{}'))
        except (json.JSONDecodeError, TypeError):
            results = {}
        
        # List all files in job directory (excluding hidden files)
        job_dir = app.config['RESULTS_FOLDER'] / job_id
        if job_dir.exists():
            for file_path in sorted(job_dir.rglob('*')):
                if file_path.is_file():
                    # Skip hidden files (starting with .)
                    if file_path.name.startswith('.'):
                        continue
                    # Skip files in hidden directories
                    if any(part.startswith('.') for part in file_path.relative_to(job_dir).parts):
                        continue
                    
                    rel_path = file_path.relative_to(job_dir)
                    file_size = file_path.stat().st_size
                    all_files.append({
                        'name': file_path.name,
                        'path': str(rel_path),
                        'size': file_size,
                        'size_formatted': format_file_size(file_size),
                        'extension': file_path.suffix.lower()
                    })
    
    # Calculate progress information
    progress = {
        'queue_position': None,
        'total_queued': None,
        'estimated_wait_formatted': None,
        'current_step': None,
        'progress_percent': 0
    }
    
    if status == JobStatus.QUEUED:
        # Calculate queue position
        queued_jobs = redis_client.keys('job:*')
        queue_list = []
        for job_key in queued_jobs:
            jd = redis_client.hgetall(job_key)
            if jd.get('status') == 'queued':
                created_at = jd.get('created_at')
                if created_at:
                    queue_list.append((job_key, created_at))
        
        queue_list.sort(key=lambda x: x[1])
        
        for idx, (job_key, _) in enumerate(queue_list):
            if job_key == f'job:{job_id}':
                progress['queue_position'] = idx + 1
                progress['total_queued'] = len(queue_list)
                break
        
        if progress['queue_position']:
            wait_seconds = progress['queue_position'] * 30
            progress['estimated_wait_formatted'] = format_duration(wait_seconds)
    
    elif status == JobStatus.RUNNING:
        progress['current_step'] = job_data.get('current_step', None)
        step_progress = {'colabfold': 20, 'sampling': 50, 'measurements': 30}
        if progress['current_step'] in step_progress:
            progress['progress_percent'] = step_progress[progress['current_step']]
    
    elif status == JobStatus.COLABFOLD_RUNNING:
        progress['current_step'] = 'colabfold'
        progress['progress_percent'] = 20
        # Add ColabFold-specific progress info
        colabfold_started_at = job_data.get('colabfold_started_at')
        if colabfold_started_at:
            try:
                from datetime import datetime
                started = datetime.fromisoformat(colabfold_started_at)
                elapsed = (datetime.now() - started).total_seconds()
                progress['elapsed_time'] = format_duration(int(elapsed))
            except:
                pass
    
    elif status == JobStatus.COLABFOLD_COMPLETE:
        progress['progress_percent'] = 30
        progress['current_step'] = 'sampling'
    
    elif status == JobStatus.SAMPLING_COMPLETE:
        progress['progress_percent'] = 80
        progress['current_step'] = 'measurements'
    
    elif status == JobStatus.COMPLETED:
        progress['progress_percent'] = 100
    
    # Extract detailed ColabFold information
    colabfold_info = {
        'started_at': job_data.get('colabfold_started_at'),
        'completed_at': job_data.get('colabfold_completed_at'),
        'failed_at': job_data.get('colabfold_failed_at'),
        'af_pdb': job_data.get('af_pdb'),
        'processing_time': None,
        'status': None
    }
    
    # Calculate processing time if available
    if colabfold_info['started_at'] and colabfold_info['completed_at']:
        try:
            from datetime import datetime
            started = datetime.fromisoformat(colabfold_info['started_at'])
            completed = datetime.fromisoformat(colabfold_info['completed_at'])
            elapsed = (completed - started).total_seconds()
            colabfold_info['processing_time'] = format_duration(int(elapsed))
        except:
            pass
    
    # Determine ColabFold status
    if status == JobStatus.COLABFOLD_RUNNING:
        colabfold_info['status'] = 'running'
    elif status == JobStatus.COLABFOLD_COMPLETE:
        colabfold_info['status'] = 'completed'
    elif colabfold_info['failed_at']:
        colabfold_info['status'] = 'failed'
    
    return render_static_template(
        'results.html',
        job_id=job_id,
        job_data=job_data,
        status=status,
        parameters=parameters,
        results=results,
        all_files=all_files,
        JobStatus=JobStatus,
        config=config,
        progress=progress,
        colabfold_info=colabfold_info
    )

@app.route('/api/upload', methods=['POST'])
def upload_fasta():
    """Upload FASTA file and return sequences"""
    try:
        print("=== DEBUG: Starting upload_fasta ===")
        
        if 'fasta' not in request.files:
            print("DEBUG: No FASTA file in request")
            return jsonify({'error': 'No FASTA file provided'}), 400
        
        file = request.files['fasta']
        print(f"DEBUG: File object: {file}")
        print(f"DEBUG: Filename: {file.filename}")
        
        if file.filename == '':
            print("DEBUG: Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith(('.fasta', '.fa', '.fas')):
            print(f"DEBUG: Invalid file extension: {file.filename}")
            return jsonify({'error': 'File must be a FASTA file (.fasta, .fa, .fas)'}), 400
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        print(f"DEBUG: File size: {file_size} bytes")
        
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            print(f"DEBUG: File too large: {file_size}")
            return jsonify({'error': f'File too large. Maximum size: {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'}), 400
        
        if file_size == 0:
            print("DEBUG: File is empty")
            return jsonify({'error': 'File is empty'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_id = str(uuid.uuid4())
        file_path = app.config['UPLOAD_FOLDER'] / f"{upload_id}_{filename}"
        print(f"DEBUG: Saving file to: {file_path}")
        file.save(file_path)
        print("DEBUG: File saved successfully")
        
        # Read FASTA content
        print("DEBUG: Reading FASTA content")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"DEBUG: Content length: {len(content)} characters")
        print(f"DEBUG: First 100 characters: {content[:100]}")
        
        # Validate FASTA format
        if not content.strip().startswith('>'):
            print("DEBUG: Invalid FASTA format - doesn't start with '>'")
            file_path.unlink()  # Clean up
            return jsonify({'error': 'Invalid FASTA format. File must start with ">"'}), 400
        
        # List sequences
        print("DEBUG: Calling list_fasta_sequences")
        sequences = list_fasta_sequences(file_path)
        print(f"DEBUG: Returned sequences type: {type(sequences)}")
        print(f"DEBUG: Sequences content: {sequences}")
        
        if not sequences:
            print("DEBUG: No sequences found")
            file_path.unlink()  # Clean up
            return jsonify({'error': 'No valid sequences found in FASTA file'}), 400
        
        # Format sequences for the frontend
        print("DEBUG: Formatting sequences for frontend")
        formatted_sequences = []
        for i, seq in enumerate(sequences):
            print(f"DEBUG: Processing sequence {i}: type={type(seq)}, content={seq}")
            try:
                # Keep all fields from list_fasta_sequences (including FP info)
                formatted_sequences.append(seq)
                print(f"DEBUG: Formatted sequence {i}: {seq['id']} ({seq['length']} residues)")
            except Exception as e:
                print(f"DEBUG: Error formatting sequence {i}: {str(e)}")
                raise
        
        print(f"DEBUG: Final formatted_sequences: {len(formatted_sequences)} sequences")
        
        response_data = {
            'upload_id': upload_id,
            'filename': filename,
            'file_size': file_size,
            'sequences': formatted_sequences,
            'file_path': str(file_path)
        }
        print(f"DEBUG: Response data: {response_data}")
        
        return jsonify(response_data)
    
    except UnicodeDecodeError:
        return jsonify({'error': 'File encoding error. Please ensure the file is in UTF-8 format'}), 400
    except Exception as e:
        app.logger.error(f'Upload error: {str(e)}')
        return jsonify({'error': f'Failed to process FASTA file: {str(e)}'}), 500

@app.route('/api/upload_pdb', methods=['POST'])
def upload_pdb():
    """Upload PDB/mmCIF file, extract chain sequences, and create a merged FASTA.

    Returns per-chain sequences for the UI selector and stores both the structure
    file and an auto-generated merged FASTA (chains joined with ':') under the same upload_id.
    """
    try:
        app.logger.info(f"Upload PDB request received. Files: {list(request.files.keys())}")
        if 'pdb' not in request.files:
            app.logger.error(f"No 'pdb' key in request.files. Available keys: {list(request.files.keys())}")
            return jsonify({'error': 'No PDB/CIF file provided'}), 400

        file = request.files['pdb']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.lower().endswith(('.pdb', '.cif', '.mmcif')):
            return jsonify({'error': 'File must be a PDB or mmCIF file (.pdb, .cif, .mmcif)'}), 400

        # Size checks
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': f'File too large. Maximum size: {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'}), 400
        if file_size == 0:
            return jsonify({'error': 'File is empty'}), 400

        # Save structure
        filename = secure_filename(file.filename)
        upload_id = str(uuid.uuid4())
        pdb_path = app.config['UPLOAD_FOLDER'] / f"{upload_id}_{filename}"
        file.save(pdb_path)

        # Extract per-chain sequences
        seqs_by_chain = extract_sequences_from_structure(pdb_path)
        if not seqs_by_chain:
            pdb_path.unlink(missing_ok=True)
            return jsonify({'error': 'No sequences could be extracted from structure'}), 400

        # Create merged FASTA under uploads with same upload_id
        fasta_out = app.config['UPLOAD_FOLDER'] / f"{upload_id}_{Path(filename).stem}.fasta"
        write_fasta_from_pdb(pdb_path, fasta_out)

        # Build sequences array for UI (chain-wise entries)
        sequences = []
        for chain_id, seq in sorted(seqs_by_chain.items(), key=lambda kv: kv[0]):
            # Detect fluorescent protein
            fp_info = detect_fluorescent_protein(seq)
            
            seq_data = {
                'id': chain_id,
                'description': f"Chain {chain_id}",
                'length': len(seq),
                'sequence': seq,
                'chain': chain_id,
            }
            
            # Add FP info if detected
            if fp_info:
                seq_data['fp_name'] = fp_info['name']
                seq_data['fp_color'] = fp_info['color']
                seq_data['fp_match_type'] = fp_info['match_type']
                if 'motif' in fp_info:
                    seq_data['fp_motif'] = fp_info['motif']
                if 'fp_start' in fp_info:
                    seq_data['fp_start'] = fp_info['fp_start']
                if 'fp_end' in fp_info:
                    seq_data['fp_end'] = fp_info['fp_end']
                if 'fps' in fp_info:
                    seq_data['fps'] = fp_info['fps']
                if 'dipole_triplets' in fp_info:
                    seq_data['dipole_triplets'] = fp_info['dipole_triplets']
            
            sequences.append(seq_data)

        return jsonify({
            'upload_id': upload_id,
            'filename': filename,
            'file_size': file_size,
            'sequences': sequences,
            'file_path': str(pdb_path),
            'file_url': f"/api/download_structure/{upload_id}_{filename}",
            'derived_fasta': str(fasta_out),
            'merged_header': Path(filename).stem,
            'note': 'PDB uploaded. Derived FASTA created by merging chains with :'
        })

    except Exception as e:
        app.logger.error(f'Upload PDB error: {str(e)}')
        return jsonify({'error': f'Failed to process PDB/CIF file: {str(e)}'}), 500

@app.route('/api/fetch_structure', methods=['POST'])
def fetch_structure():
    """Fetch structure by PDB ID, UniProt ID, or AlphaFold ID"""
    try:
        data = request.json
        if not data or 'id' not in data:
            return jsonify({'error': 'No ID provided'}), 400
        
        structure_id = data['id'].strip()
        if not structure_id:
            return jsonify({'error': 'Empty ID provided'}), 400
        
        # Store original case for AlphaFold IDs, but uppercase for PDB IDs
        original_id = structure_id
        structure_id_upper = structure_id.upper()
        
        # Determine if it's a PDB ID, UniProt ID, or AlphaFold ID
        if len(structure_id_upper) == 4 and structure_id_upper.isalnum():
            # PDB ID - fetch from RCSB
            url = f"https://files.rcsb.org/download/{structure_id_upper}.pdb"
            filename = f"{structure_id_upper}.pdb"
            return _download_pdb_direct(url, filename)
        elif original_id.startswith('AF-') and original_id.endswith('.pdb'):
            # Direct AlphaFold PDB URL
            url = f"https://alphafold.ebi.ac.uk/files/{original_id}"
            filename = original_id
            return _download_pdb_direct(url, filename)
        elif original_id.startswith('AF-'):
            # AlphaFold ID format - check if ColabFold is disabled
            if app.config['DISABLE_COLABFOLD']:
                return jsonify({'error': 'ColabFold is disabled. Only UniProt names (e.g., P42212) are allowed.'}), 400
            return _download_alphafold_intelligent(original_id)
        elif len(structure_id_upper) >= 6:
            # Assume UniProt ID - use intelligent AlphaFold downloader
            return _download_alphafold_intelligent(structure_id_upper)
        else:
            return jsonify({'error': 'Invalid ID format. Use PDB ID (e.g., 1ABC), UniProt name (e.g., P42212), or AlphaFold ID (e.g., AF-P32455-F1-model_v6)'}), 400
        
    except Exception as e:
        app.logger.error(f'Fetch structure error: {str(e)}')
        return jsonify({'error': f'Failed to fetch structure: {str(e)}'}), 500

@app.route('/api/download_structure/<filename>')
def download_structure(filename: str):
    """Serve structure files for 3D viewer"""
    try:
        # Security: only allow files in upload folder with valid names
        if not filename or '..' in filename or '/' in filename:
            abort(400)
        
        file_path = app.config['UPLOAD_FOLDER'] / filename
        if not file_path.exists():
            abort(404)
        
        return send_file(file_path, mimetype='text/plain')
    except Exception as e:
        app.logger.error(f'Download structure error: {str(e)}')
        abort(500)

def _download_pdb_direct(url: str, filename: str):
    """Download PDB file directly from URL"""
    import requests
    
    response = requests.get(url, timeout=30)
    
    if response.status_code == 404:
        return jsonify({'error': f'Structure not found: {filename}'}), 404
    elif response.status_code != 200:
        return jsonify({'error': f'Failed to download structure: HTTP {response.status_code}'}), 500
    
    # Save the downloaded file
    upload_id = str(uuid.uuid4())
    pdb_path = app.config['UPLOAD_FOLDER'] / f"{upload_id}_{filename}"
    
    with open(pdb_path, 'w') as f:
        f.write(response.text)
    
    # Extract per-chain sequences
    seqs_by_chain = extract_sequences_from_structure(pdb_path)
    if not seqs_by_chain:
        pdb_path.unlink(missing_ok=True)
        return jsonify({'error': 'No sequences could be extracted from structure'}), 400
    
    # Create merged FASTA under uploads with same upload_id
    return _create_structure_response(upload_id, filename, pdb_path, seqs_by_chain)


def _download_alphafold_intelligent(af_id: str):
    """Download latest AlphaFold structure using intelligent version detection"""
    try:
        from alphafold_downloader import download_latest_alphafold
        
        # Create temporary directory for download
        temp_dir = Path(app.config['UPLOAD_FOLDER']) / f"temp_{uuid.uuid4()}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Download the latest version
            downloaded_path = download_latest_alphafold(
                af_id, 
                out_dir=temp_dir, 
                fmt="pdb", 
                max_version_to_try=50
            )
            
            # Move to final location
            upload_id = str(uuid.uuid4())
            filename = downloaded_path.name
            final_path = app.config['UPLOAD_FOLDER'] / f"{upload_id}_{filename}"
            
            import shutil
            shutil.move(str(downloaded_path), str(final_path))
            
            # Clean up temp directory
            temp_dir.rmdir()
            
            # Extract per-chain sequences
            seqs_by_chain = extract_sequences_from_structure(final_path)
            if not seqs_by_chain:
                final_path.unlink(missing_ok=True)
                return jsonify({'error': 'No sequences could be extracted from structure'}), 400
            
            # Create response
            return _create_structure_response(upload_id, filename, final_path, seqs_by_chain)
            
        except FileNotFoundError as e:
            return jsonify({'error': str(e)}), 404
        finally:
            # Clean up temp directory if it still exists
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                
    except ImportError:
        # Fallback to simple method if downloader not available
        return _fallback_alphafold_download(af_id)


def _fallback_alphafold_download(af_id: str):
    """Fallback method for AlphaFold download without intelligent version detection"""
    import requests
    
    # Normalize AF ID
    if af_id.startswith('AF-'):
        # Extract UniProt accession from AF-P32455 or AF-P32455-F1
        import re
        m = re.match(r"^AF-([A-Z0-9]+)(?:-F\d+)?$", af_id.upper())
        if not m:
            return jsonify({'error': f'Invalid AlphaFold ID format: {af_id}'}), 400
        uniprot_acc = m.group(1)
    else:
        uniprot_acc = af_id.upper()
    
    # Try version 4 as default
    filename = f"AF-{uniprot_acc}-F1-model_v4.pdb"
    url = f"https://alphafold.ebi.ac.uk/files/{filename}"
    
    response = requests.get(url, timeout=30)
    
    if response.status_code == 404:
        return jsonify({'error': f'AlphaFold model for {af_id} not found'}), 404
    elif response.status_code != 200:
        return jsonify({'error': f'Failed to download structure: HTTP {response.status_code}'}), 500
    
    # Save the downloaded file
    upload_id = str(uuid.uuid4())
    pdb_path = app.config['UPLOAD_FOLDER'] / f"{upload_id}_{filename}"
    
    with open(pdb_path, 'w') as f:
        f.write(response.text)
    
    # Extract per-chain sequences
    seqs_by_chain = extract_sequences_from_structure(pdb_path)
    if not seqs_by_chain:
        pdb_path.unlink(missing_ok=True)
        return jsonify({'error': 'No sequences could be extracted from structure'}), 400
    
    # Create merged FASTA under uploads with same upload_id
    return _create_structure_response(upload_id, filename, pdb_path, seqs_by_chain)


def _create_structure_response(upload_id: str, filename: str, pdb_path: Path, seqs_by_chain: dict):
    """Create standardized response for structure download"""
    # Create merged FASTA under uploads with same upload_id
    merged_fasta_path = app.config['UPLOAD_FOLDER'] / f"{upload_id}_{Path(filename).stem}.fasta"
    
    # Merge all chains with ':' separator for multimer support
    all_sequences = []
    for chain_id, sequence in seqs_by_chain.items():
        all_sequences.append(sequence)
    
    merged_sequence = ":".join(all_sequences)
    
    with open(merged_fasta_path, 'w') as f:
        f.write(f">{Path(filename).stem}\n{merged_sequence}\n")
    
    sequences = []
    for chain_id, sequence in seqs_by_chain.items():
        fp_info = detect_fluorescent_protein(sequence)
        seq_data = {
            'id': chain_id,
            'description': f"Chain {chain_id}",
            'chain': chain_id,
            'length': len(sequence),
            'sequence': sequence
        }
        if fp_info:
            seq_data['fp_name'] = fp_info['name']
            seq_data['fp_color'] = fp_info['color']
            seq_data['fp_match_type'] = fp_info['match_type']
            if 'fp_start' in fp_info: seq_data['fp_start'] = fp_info['fp_start']
            if 'fp_end' in fp_info: seq_data['fp_end'] = fp_info['fp_end']
            if 'dipole_triplets' in fp_info: seq_data['dipole_triplets'] = fp_info['dipole_triplets']
            if 'fps' in fp_info: seq_data['fps'] = fp_info['fps']
        sequences.append(seq_data)

    return jsonify({
        'upload_id': upload_id,
        'filename': filename,
        'file_path': str(pdb_path),
        'file_url': f"/api/download_structure/{upload_id}_{filename}",
        'file_size': pdb_path.stat().st_size,
        'sequences': sequences,
        'derived_fasta': str(merged_fasta_path),
        'merged_header': Path(filename).stem,
        'note': f"Structure {filename} downloaded. Derived FASTA created by merging chains with ':'"
    })

@app.route('/api/submit', methods=['POST'])
def submit_job():
    """Submit fpsim job"""
    try:
        data = request.json
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Validate required fields
        if 'upload_id' not in data:
            return jsonify({'error': 'upload_id is required'}), 400
        
        # Get file path
        upload_id = data['upload_id']
        # Prefer a FASTA under this upload_id; if multiple files uploaded (PDB+FASTA), choose .fasta
        candidates = sorted(app.config['UPLOAD_FOLDER'].glob(f"{upload_id}_*"))
        if not candidates:
            return jsonify({'error': 'Uploaded file not found'}), 400
        fasta_path = None
        # Prefer .fasta
        for c in candidates:
            if c.suffix.lower() in ('.fasta', '.fa', '.fas'):
                fasta_path = c
                break
        # Fallback: if only a PDB exists, try to create a FASTA from it
        if fasta_path is None:
            pdb_candidates = [c for c in candidates if c.suffix.lower() in ('.pdb', '.cif', '.mmcif')]
            if pdb_candidates:
                pdb_src = pdb_candidates[0]
                fasta_path = app.config['UPLOAD_FOLDER'] / f"{upload_id}_{pdb_src.stem}.fasta"
                try:
                    write_fasta_from_pdb(pdb_src, fasta_path)
                except Exception as e:
                    return jsonify({'error': f'Failed to derive FASTA from uploaded PDB: {e}'}), 400
            else:
                # Last resort: use the first candidate
                fasta_path = candidates[0]
        
        # Create job output directory
        job_dir = app.config['RESULTS_FOLDER'] / job_id
        job_dir.mkdir(exist_ok=True)
        
        # Extract parameters with defaults
        params = data.get('parameters', {})
        membrane_regions = data.get('membrane_regions', params.get('membrane_regions', []))
        
        # Force disable ColabFold if DISABLE_COLABFOLD is set
        if app.config['DISABLE_COLABFOLD']:
            params['run_colabfold'] = False
            app.logger.info("ColabFold disabled via DISABLE_COLABFOLD environment variable")
        
        # Convert membrane regions and sequences to membrane_seq format
        membrane_seqs = params.get('membrane_seqs', [])
        membrane_seq_list = list(membrane_seqs)
        
        # Also include the sequence_id if specific residue regions are provided
        if membrane_regions and params.get('sequence_id') and params.get('sequence_id') not in membrane_seq_list:
            # For now, we'll use the sequence ID as the membrane sequence identifier
            # In a more sophisticated implementation, you'd map regions to specific sequences
            membrane_seq_list.append(params.get('sequence_id'))
        
        # Ensure membrane is enabled if sequences or regions are provided
        should_enable_membrane = bool(params.get('membrane', False)) or len(membrane_seq_list) > 0 or len(membrane_regions) > 0
        params['membrane'] = should_enable_membrane
        
        # Create pipeline configuration
        config_dict = {
            'fasta': str(fasta_path),
            'out_dir': str(job_dir),
            'sequence_id': params.get('sequence_id'),
            'chain': params.get('chain', 'A'),
            'af_pdb': params.get('af_pdb'),
            'run_colabfold': bool(params.get('run_colabfold', True)),
            'use_structure_plddt': bool(params.get('use_structure_plddt', False)),
            'colabfold_args': params.get('colabfold_args', '--num-models 1'),
            'low_spec': params.get('low_spec', False),
            'mem_frac': params.get('mem_frac', 0.5),
            'relax': params.get('relax'),
            'num_models': params.get('num_models'),
            'num_ensemble': params.get('num_ensemble'),
            'num_recycle': params.get('num_recycle'),
            'model_type': params.get('model_type'),
            'max_seq': params.get('max_seq'),
            'max_extra_seq': params.get('max_extra_seq'),
            'msa_mode': params.get('msa_mode'),
            'pair_mode': params.get('pair_mode'),
            'no_templates': params.get('no_templates', False),
            'use_gpu_relax': params.get('use_gpu_relax', True),
            'gpu': params.get('gpu'),
            'plddt_rigid': params.get('plddt_rigid', 70.0),
            'model_disordered_as_beads': bool(params.get('model_disordered_as_beads', False)),
            'frames': params.get('frames', 100000),
            'steps_per_frame': params.get('steps_per_frame', 10),
            'membrane': params.get('membrane', False),
            'membrane_weight': params.get('membrane_weight', 10.0),
            'barrier_radius': params.get('barrier_radius', 100.0),
            'membrane_seq': list(membrane_seq_list),
            'reuse': params.get('reuse', True),
            'force': params.get('force', False),
            'verbose': params.get('verbose', False),
            'fp_lib': params.get('fp_lib'),
            'fp_library': FP_LIBRARY,
            'fp_color': FP_COLOR,
            'measure': params.get('measure', False),
            'sites': params.get('sites', []),
            'measure_out_tsv': params.get('measure_out_tsv'),
            'measure_plot': params.get('measure_plot', True),
            'measure_plot_out': params.get('measure_plot_out'),
            'measure_frame_start': params.get('measure_frame_start', 0),
            'measure_max_frames': params.get('measure_max_frames')
        }
        
        # Store job metadata
        job_metadata = {
            'job_id': job_id,
            'status': JobStatus.QUEUED,
            'created_at': datetime.now().isoformat(),
            'fasta_filename': fasta_path.name,
            'parameters': json.dumps(params),  # Convert dict to JSON string
            'membrane_regions': json.dumps(data.get('membrane_regions', [])),  # Convert list to JSON string
            'email': data.get('email', '')  # Store email for notifications
        }
        
        # Store each field individually to avoid type issues with Redis
        for key, value in job_metadata.items():
            redis_client.hset(f"job:{job_id}", key, value)
        
        # Ensure all values in config_dict are serializable
        serializable_config = {}
        for key, value in config_dict.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                serializable_config[key] = value
            elif isinstance(value, (list, dict, tuple)):
                serializable_config[key] = list(value) if isinstance(value, tuple) else value
            else:
                # For any other type, convert to string
                serializable_config[key] = str(value)
        
        # Submit to Celery with serialized config
        run_fpsim_job.apply_async(args=[job_id, serializable_config], queue='imp_queue')
        
        # Get the base URL from the request
        base_url = request.host_url.rstrip('/')
        results_url = f"{base_url}/results/{job_id}"
        
        return jsonify({
            'job_id': job_id,
            'status': JobStatus.QUEUED,
            'message': 'Job submitted successfully',
            'results_url': results_url
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to submit job: {str(e)}'}), 500

@app.route('/api/status/<job_id>')
def get_job_status_legacy(job_id: str):
    """Get job status with detailed progress information"""
    job_data = redis_client.hgetall(f"job:{job_id}")
    
    if not job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    # Deserialize JSON fields
    if 'parameters' in job_data:
        try:
            job_data['parameters'] = json.loads(job_data['parameters'])
        except (json.JSONDecodeError, TypeError):
            pass
    
    if 'membrane_regions' in job_data:
        try:
            job_data['membrane_regions'] = json.loads(job_data['membrane_regions'])
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Add detailed progress information
    status = job_data.get('status', 'unknown')
    progress_info = {
        'queue_position': None,
        'estimated_wait_seconds': None,
        'current_step': None,
        'progress_percent': 0,
        'cli_output': [],
        'email': job_data.get('email', None)
    }
    
    if status == 'queued':
        # Calculate queue position
        queued_jobs = redis_client.keys('job:*')
        queue_list = []
        for job_key in queued_jobs:
            jd = redis_client.hgetall(job_key)
            if jd.get('status') == 'queued':
                created_at = jd.get('created_at')
                if created_at:
                    queue_list.append((job_key, created_at))
        
        # Sort by creation time
        queue_list.sort(key=lambda x: x[1])
        
        # Find position in queue
        for idx, (job_key, _) in enumerate(queue_list):
            if job_key == f'job:{job_id}':
                progress_info['queue_position'] = idx + 1
                progress_info['total_queued'] = len(queue_list)
                break
        
        # Estimate wait time (assuming ~30 seconds per job average)
        if progress_info['queue_position']:
            progress_info['estimated_wait_seconds'] = progress_info['queue_position'] * 30
            progress_info['estimated_wait_formatted'] = format_duration(progress_info['estimated_wait_seconds'])
    
    elif status == 'running':
        # Get current step info
        current_step = job_data.get('current_step', None)
        progress_info['current_step'] = current_step
        
        # Calculate progress based on step
        step_progress = {
            'colabfold': 20,
            'sampling': 50,
            'measurements': 30
        }
        
        if current_step in step_progress:
            progress_info['progress_percent'] = step_progress[current_step]
        
        # Get CLI output
        cli_output = get_job_cli_output(job_id)
        progress_info['cli_output'] = cli_output
    
    elif status == 'colabfold_complete':
        progress_info['progress_percent'] = 30
        progress_info['current_step'] = 'sampling'
        progress_info['cli_output'] = get_job_cli_output(job_id)
    
    elif status == 'sampling_complete':
        progress_info['progress_percent'] = 80
        progress_info['current_step'] = 'measurements'
        progress_info['cli_output'] = get_job_cli_output(job_id)
    
    elif status == 'completed':
        progress_info['progress_percent'] = 100
        progress_info['cli_output'] = get_job_cli_output(job_id)
    
    # Add progress info to response
    job_data['progress'] = progress_info
    
    # Add results URL
    base_url = request.host_url.rstrip('/')
    job_data['results_url'] = f"{base_url}/results/{job_id}"
    
    return jsonify(job_data)


def format_duration(seconds: int) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''}"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours} hour{'s' if hours > 1 else ''} {minutes} minute{'s' if minutes > 1 else ''}"


def get_job_cli_output(job_id: str) -> list:
    """Get CLI output for a job"""
    output_lines = []
    log_file = app.config['RESULTS_FOLDER'] / job_id / 'pipeline.log'
    
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                # Return last 50 lines
                for line in lines[-50:]:
                    line = line.strip()
                    if line:
                        # Parse timestamp and message
                        parts = line.split(' - ', 2)
                        if len(parts) >= 2:
                            output_lines.append({
                                'timestamp': parts[0] if len(parts) > 0 else '',
                                'level': parts[1] if len(parts) > 1 else 'INFO',
                                'message': parts[2] if len(parts) > 2 else line
                            })
                        else:
                            output_lines.append({'message': line})
        except Exception as e:
            output_lines.append({'error': str(e)})
    
    return output_lines


# Email notification functions
def send_job_notification(job_id: str, event: str, message: str):
    """Send email notification for job event"""
    job_data = redis_client.hgetall(f"job:{job_id}")
    email = job_data.get('email')
    
    if not email:
        return False
    
    try:
        # Build email content
        subject = f"FPsimP Job {event}: {job_id[:8]}"
        
        body = f"""
Job ID: {job_id}
Status: {event}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{message}

---
FPsimP - Fluorescent Protein Simulation Pipeline
"""
        
        # Send email (placeholder - configure SMTP in production)
        if config.SMTP_SERVER:
            send_email(email, subject, body)
        else:
            print(f"[EMAIL] To: {email}")
            print(f"[EMAIL] Subject: {subject}")
            print(f"[EMAIL] Body: {body}")
        
        return True
    except Exception as e:
        print(f"Failed to send email notification: {e}")
        return False


def send_email(to_addr: str, subject: str, body: str):
    """Send email via SMTP"""
    smtp_server = config.SMTP_SERVER
    smtp_port = config.SMTP_PORT
    smtp_user = config.SMTP_USER
    smtp_pass = config.SMTP_PASS
    from_addr = config.SMTP_FROM
    
    if not smtp_server:
        print(f"[EMAIL] Would send to {to_addr}: {subject}")
        return
    
    msg = MIMEMultipart()
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            if smtp_user and smtp_pass:
                server.starttls()
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"[EMAIL] Sent to {to_addr}")
    except Exception as e:
        print(f"[EMAIL] Failed to send: {e}")

@app.route('/api/download/<job_id>/<filename>')
def download_file(job_id: str, filename: str):
    """Download result file"""
    job_data = redis_client.hgetall(f"job:{job_id}")
    
    if not job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    if job_data.get('status') != JobStatus.COMPLETED:
        return jsonify({'error': 'Job not completed'}), 400
    
    # URL-decode the filename to handle subdirectories correctly
    from urllib.parse import unquote
    decoded_filename = unquote(filename)
    
    # Security: only allow downloading from job directory
    job_dir = app.config['RESULTS_FOLDER'] / job_id
    file_path = job_dir / secure_filename(decoded_filename)
    
    # For files in subdirectories, we need to handle the path more carefully
    # secure_filename only works on the basename, so we need to validate the full path
    if '..' in decoded_filename or decoded_filename.startswith('/'):
        abort(404)
    
    # Reconstruct the full path while maintaining directory structure
    full_file_path = job_dir / decoded_filename
    
    if not full_file_path.exists() or not full_file_path.is_relative_to(job_dir):
        abort(404)
    
    return send_file(full_file_path, as_attachment=True)

@app.route('/api/download_zip/<job_id>')
def download_zip(job_id: str):
    """Download entire job folder as zip"""
    job_data = redis_client.hgetall(f"job:{job_id}")
    
    if not job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    if job_data.get('status') != JobStatus.COMPLETED:
        return jsonify({'error': 'Job not completed'}), 400
    
    # Security: only allow downloading from job directory
    job_dir = app.config['RESULTS_FOLDER'] / job_id
    
    if not job_dir.exists():
        return jsonify({'error': 'Job directory not found'}), 404
    
    # Create temporary zip file
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    temp_zip.close()
    
    try:
        # Create zip archive
        shutil.make_archive(temp_zip.name[:-4], 'zip', job_dir)
        zip_path = temp_zip.name[:-4] + '.zip'
        
        # Send file and clean up after
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=f'fpsim_results_{job_id}.zip',
            mimetype='application/zip'
        )
    except Exception as e:
        if os.path.exists(temp_zip.name):
            os.unlink(temp_zip.name)
        return jsonify({'error': f'Failed to create zip: {str(e)}'}), 500

@app.route('/api/job_status/<job_id>')
def get_job_status(job_id: str):
    """Get job status and progress information"""
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    # Check if job is actually running by checking heartbeat
    status = job_data.get('status', 'unknown')
    heartbeat_str = job_data.get('heartbeat')
    worker_id = job_data.get('worker_id')
    
    # If status is "running" but heartbeat is stale, mark as queued
    if status == JobStatus.RUNNING and heartbeat_str:
        try:
            from datetime import datetime, timedelta
            heartbeat = datetime.fromisoformat(heartbeat_str)
            heartbeat_threshold = datetime.now() - timedelta(minutes=5)
            
            if heartbeat > heartbeat_threshold:
                # Heartbeat is fresh, job is actually running
                pass
            else:
                # Heartbeat is stale, job is orphaned - mark as queued
                status = JobStatus.QUEUED
                app.logger.warning(f"Job {job_id} has stale heartbeat from {heartbeat}, marking as queued")
        except Exception as e:
            app.logger.warning(f"Error parsing heartbeat for job {job_id}: {e}")
            status = JobStatus.QUEUED
    
    # Base response
    base_url = request.host_url.rstrip('/')
    response = {
        'job_id': job_id,
        'status': status,
        'created_at': job_data.get('created_at'),
        'started_at': job_data.get('started_at'),
        'results_url': f"{base_url}/results/{job_id}",
        'error': job_data.get('error'),
        'progress': {}
    }
    
    return jsonify(response)

@app.route('/api/job_intermediate/<job_id>')
def get_job_intermediate_results(job_id: str):
    """Get intermediate results for job (sequence, pLDDT, segmentation, FPs, topology)"""
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    job_dir = app.config['RESULTS_FOLDER'] / job_id
    if not job_dir.exists():
        return jsonify({'error': 'Job directory not found'}), 404
    
    result = {}
    
    # Read segments.json (contains sequence, segmentation, FP domains, pLDDT info)
    # Read segments.json (contains sequence, segmentation, FP domains, pLDDT info)
    segments_file = job_dir / 'segments.json'
    if segments_file.exists():
        try:
            import json
            segments_data = json.loads(segments_file.read_text())
            
            # Check if this is a multimer job
            if 'chains' in segments_data and 'chain_labels' in segments_data:
                # Multimer handling
                full_sequence = ""
                full_plddt = {}
                full_segments = []
                full_fp_domains = []
                
                af_pdb = segments_data.get('af_pdb')
                offset = 0
                
                # Import necessary function
                from fpsim.segments import parse_plddt_from_pdb
                
                for label in segments_data['chain_labels']:
                    chain_data = segments_data['chains'].get(label, {})
                    chain_seq = chain_data.get('sequence', '')
                    chain_len = chain_data.get('sequence_len', len(chain_seq))
                    
                    # Fallback if sequence is missing (e.g. old jobs) but length is known
                    if not chain_seq and chain_len > 0:
                        chain_seq = 'X' * chain_len
                    
                    full_sequence += chain_seq
                    
                    # Parse pLDDT for this chain
                    if af_pdb and Path(af_pdb).exists():
                        try:
                            # Parse pLDDT (keys are 1-based relative to chain)
                            chain_plddt = parse_plddt_from_pdb(Path(af_pdb), label)
                            
                            # Add to full dict with offset
                            for pos, score in chain_plddt.items():
                                full_plddt[pos + offset] = score
                        except Exception as e:
                            app.logger.warning(f"Could not parse pLDDT for chain {label}: {e}")
                    
                    # Shift segments
                    if 'segments' in chain_data:
                        for seg in chain_data['segments']:
                            full_segments.append({
                                **seg,
                                'start': seg['start'] + offset,
                                'end': seg['end'] + offset
                            })
                            
                    # Shift FP domains
                    if 'fp_domains' in chain_data:
                        for fp in chain_data['fp_domains']:
                            full_fp_domains.append({
                                **fp,
                                'start': fp['start'] + offset,
                                'end': fp['end'] + offset
                            })
                            
                    offset += chain_len
                
                result['sequence'] = full_sequence
                result['plddt'] = full_plddt
                result['segments'] = full_segments
                result['fp_domains'] = full_fp_domains
                
            else:
                # Single chain handling
                # Extract sequence
                if 'sequence' in segments_data:
                    result['sequence'] = segments_data['sequence']
                
                # Extract pLDDT from PDB file
                af_pdb = segments_data.get('af_pdb')
                if af_pdb and Path(af_pdb).exists():
                    from fpsim.segments import parse_plddt_from_pdb
                    chain = segments_data.get('chain', 'A')
                    try:
                        plddt_dict = parse_plddt_from_pdb(Path(af_pdb), chain)
                        result['plddt'] = plddt_dict
                    except Exception as e:
                        app.logger.warning(f"Could not parse pLDDT: {e}")
                
                # Extract segmentation
                if 'segments' in segments_data:
                    result['segments'] = segments_data['segments']
                
                # Extract FP domains
                if 'fp_domains' in segments_data:
                    result['fp_domains'] = segments_data['fp_domains']
            
            # Extract membrane regions from job parameters (common to both)
            # Extract membrane regions (common to both)
            try:
                # Try reading from top-level redis key first
                membrane_regions_str = job_data.get('membrane_regions')
                if membrane_regions_str and membrane_regions_str != '[]':
                    try:
                        result['membrane_regions'] = json.loads(membrane_regions_str)
                    except (json.JSONDecodeError, TypeError):
                        # Might be stored as raw list in some cases?
                        result['membrane_regions'] = membrane_regions_str
                else:
                    # Fallback to parameters
                    params_str = job_data.get('parameters', '{}')
                    params = json.loads(params_str) if isinstance(params_str, str) else params_str
                    
                    membrane_regions = params.get('membrane_regions', [])
                    if isinstance(membrane_regions, str):
                        result['membrane_regions'] = json.loads(membrane_regions)
                    else:
                        result['membrane_regions'] = membrane_regions
            except Exception as e:
                app.logger.warning(f"Could not parse membrane regions: {e}")
                result['membrane_regions'] = []
                
        except Exception as e:
            app.logger.error(f"Error reading segments.json: {e}")
    
    # Read topology file
    top_file = job_dir / 'fusion.top.dat'
    if top_file.exists():
        try:
            result['topology'] = top_file.read_text().split('\n')
        except Exception as e:
            app.logger.error(f"Error reading topology: {e}")
    
    return jsonify(result)

@app.route('/api/job_diagnostics/<job_id>')
def get_job_diagnostics(job_id: str):
    """Get diagnostic information about job, queue, and worker status"""
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    diagnostics = {
        'job_id': job_id,
        'status': job_data.get('status', 'unknown'),
        'created_at': job_data.get('created_at'),
        'queue_info': {},
        'worker_info': {},
        'celery_info': {}
    }
    
    # Get queue information with heartbeat-based active detection
    try:
        from datetime import datetime, timedelta
        
        # Count jobs in various states
        all_job_keys = redis_client.keys('job:*')
        queued_jobs = []
        running_jobs = []
        stale_jobs = []
        heartbeat_threshold = datetime.now() - timedelta(minutes=5)
        
        for job_key in all_job_keys:
            jd = redis_client.hgetall(job_key)
            status = jd.get('status')
            created_at = jd.get('created_at')
            heartbeat_str = jd.get('heartbeat')
            
            if status == JobStatus.QUEUED and created_at:
                queued_jobs.append((job_key.decode() if isinstance(job_key, bytes) else job_key, created_at))
            elif status == JobStatus.RUNNING:
                # Check if heartbeat is fresh (within last 5 minutes)
                is_active = False
                if heartbeat_str:
                    try:
                        heartbeat = datetime.fromisoformat(heartbeat_str)
                        if heartbeat > heartbeat_threshold:
                            is_active = True
                    except:
                        pass
                
                job_key_str = job_key.decode() if isinstance(job_key, bytes) else job_key
                if is_active:
                    running_jobs.append(job_key_str)
                else:
                    stale_jobs.append(job_key_str)
        
        # Sort queued jobs by creation time
        queued_jobs.sort(key=lambda x: x[1])
        
        # Find position of current job in queue
        job_position = None
        if job_data.get('status') == JobStatus.QUEUED:
            for idx, (jk, _) in enumerate(queued_jobs):
                if jk == f'job:{job_id}':
                    job_position = idx + 1
                    break
        
        diagnostics['queue_info'] = {
            'queued_count': len(queued_jobs),
            'running_count': len(running_jobs),
            'stale_count': len(stale_jobs),
            'job_position': job_position,
            'estimated_wait_minutes': job_position * 5 if job_position else None
        }
        
    except Exception as e:
        app.logger.error(f"Error getting queue info: {e}")
        diagnostics['queue_info']['error'] = str(e)
    
    # Get Celery worker information
    try:
        # Use the existing celery instance with increased timeout
        inspect = celery.control.inspect(timeout=15.0)
        active_workers = None
        registered_tasks = None
        
        # Try multiple times since inspect can be flaky
        for attempt in range(3):
            try:
                active_workers = inspect.active()
                if active_workers is not None:
                    registered_tasks = inspect.registered()
                    break
            except Exception as e:
                app.logger.warning(f"Celery inspect attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    import time
                    time.sleep(0.5)
        
        # Fallback: check for worker heartbeats in Redis
        workers_detected = 0
        worker_names = []
        active_task_count = 0
        
        if active_workers:
            workers_detected = len(active_workers)
            worker_names = list(active_workers.keys())
            active_task_count = sum(len(tasks) for tasks in active_workers.values())
            detection_method = 'celery_inspect'
        else:
            # Fallback: check Redis for celery worker keys
            try:
                worker_keys = redis_client.keys('_kombu.binding.*')
                if worker_keys:
                    workers_detected = 1  # At least one worker exists
                    worker_names = ['worker (detected via Redis)']
                # Use heartbeat-based running count as proxy for active tasks
                active_task_count = diagnostics['queue_info'].get('running_count', 0)
                detection_method = 'redis_fallback'
            except Exception as redis_err:
                app.logger.warning(f"Redis worker check failed: {redis_err}")
                detection_method = 'failed'
        
        diagnostics['celery_info'] = {
            'workers_online': workers_detected,
            'worker_names': worker_names,
            'active_tasks_count': active_task_count,
            'detection_method': detection_method
        }
        
        # Check if run_fpsim_job task is registered
        if registered_tasks:
            for worker, tasks in registered_tasks.items():
                if 'app.run_fpsim_job' in tasks:
                    diagnostics['celery_info']['fpsim_task_registered'] = True
                    break
        
    except Exception as e:
        app.logger.error(f"Error getting Celery info: {e}")
        diagnostics['celery_info']['error'] = str(e)
        # Even if inspect fails, assume worker is running if we can see completed jobs
        diagnostics['celery_info']['workers_online'] = 1
        diagnostics['celery_info']['worker_names'] = ['worker (assumed running)']
    
    # Check if task is in Celery queue
    try:
        queue_length = redis_client.llen('celery')
        diagnostics['celery_info']['celery_queue_length'] = queue_length
    except Exception as e:
        diagnostics['celery_info']['queue_error'] = str(e)
    
    return jsonify(diagnostics)

@app.route('/api/admin/cleanup_stale_jobs', methods=['POST'])
def cleanup_stale_jobs():
    """Clean up jobs stuck in running state for more than 2 hours (with activity check)"""
    from datetime import datetime, timedelta
    import os
    
    try:
        all_job_keys = redis_client.keys('job:*')
        cleaned = []
        skipped = []
        cutoff_time = datetime.now() - timedelta(hours=2)
        activity_threshold = datetime.now() - timedelta(minutes=5)
        
        for job_key in all_job_keys:
            jd = redis_client.hgetall(job_key)
            status = jd.get('status')
            started_at_str = jd.get('started_at')
            
            if status == JobStatus.RUNNING and started_at_str:
                try:
                    started_at = datetime.fromisoformat(started_at_str)
                    if started_at < cutoff_time:
                        job_id = job_key.decode() if isinstance(job_key, bytes) else job_key
                        job_id = job_id.replace('job:', '')
                        
                        # Check if job is actively writing (sampling.log modified recently)
                        job_dir = app.config['RESULTS_FOLDER'] / job_id
                        sampling_log = job_dir / 'sampling.log'
                        
                        is_active = False
                        if sampling_log.exists():
                            try:
                                mtime = datetime.fromtimestamp(sampling_log.stat().st_mtime)
                                if mtime > activity_threshold:
                                    is_active = True
                                    skipped.append({
                                        'job_id': job_id,
                                        'reason': 'Active sampling detected',
                                        'last_activity': mtime.isoformat()
                                    })
                            except Exception as e:
                                app.logger.warning(f"Error checking file mtime: {e}")
                        
                        # Only mark as failed if NOT actively writing
                        if not is_active:
                            redis_client.hset(f"job:{job_id}", "status", JobStatus.FAILED)
                            redis_client.hset(f"job:{job_id}", "error", "Job stalled - no activity detected")
                            redis_client.hset(f"job:{job_id}", "failed_at", datetime.now().isoformat())
                            cleaned.append(job_id)
                        
                except Exception as e:
                    app.logger.warning(f"Error processing job: {e}")
        
        return jsonify({
            'success': True,
            'cleaned_count': len(cleaned),
            'skipped_count': len(skipped),
            'cleaned_jobs': cleaned,
            'skipped_jobs': skipped
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/job_cancel/<job_id>', methods=['POST'])
def cancel_job(job_id: str):
    """Cancel a running job"""
    try:
        # Get job data from Redis
        job_data = redis_client.hgetall(f"job:{job_id}")
        if not job_data:
            return jsonify({'error': 'Job not found'}), 404
        
        job_status = job_data.get('status')
        
        # Only allow cancelling active jobs
        if job_status not in [JobStatus.QUEUED, JobStatus.RUNNING, 
                              JobStatus.COLABFOLD_COMPLETE, JobStatus.SAMPLING_COMPLETE]:
            return jsonify({'error': 'Job is not active and cannot be cancelled'}), 400
        
        # Get Celery task ID if available
        worker_id = job_data.get('worker_id')
        
        # Revoke the Celery task
        if worker_id:
            try:
                celery.control.revoke(worker_id, terminate=True, signal='SIGKILL')
                app.logger.info(f"Revoked Celery task {worker_id} for job {job_id}")
            except Exception as e:
                app.logger.warning(f"Failed to revoke Celery task: {e}")
        
        # Update job status in Redis
        redis_client.hset(f"job:{job_id}", "status", JobStatus.FAILED)
        redis_client.hset(f"job:{job_id}", "error", "Job cancelled by user")
        redis_client.hset(f"job:{job_id}", "failed_at", datetime.now().isoformat())
        
        return jsonify({
            'success': True,
            'message': 'Job cancelled successfully',
            'job_id': job_id
        })
    
    except Exception as e:
        app.logger.error(f"Error cancelling job {job_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/job_sampling_log/<job_id>')
def get_job_sampling_log(job_id: str):
    """Get IMP sampling log for real-time progress"""
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    job_dir = app.config['RESULTS_FOLDER'] / job_id
    if not job_dir.exists():
        return jsonify({'error': 'Job directory not found'}), 404
    
    # Look for sampling log file
    log_file = job_dir / 'sampling.log'
    if log_file.exists():
        try:
            # Return only the last N lines to prevent stalling
            lines = log_file.read_text().split('\n')
            max_lines = 100  # Limit to last 100 lines to prevent stalling
            last_lines = lines[-max_lines:] if len(lines) > max_lines else lines
            
            # Parse current frame and total frames for progress
            current_frame = 0
            total_frames = 100000  # Default
            
            # Get total frames from job parameters
            try:
                params = json.loads(job_data.get('parameters', '{}'))
                total_frames = params.get('frames', 100000)
            except:
                pass
            
            # Find latest frame number in log (search backwards for efficiency)
            import re
            frame_pattern = re.compile(r'---\s+frame\s+(\d+)')
            for line in reversed(lines):
                match = frame_pattern.search(line)
                if match:
                    current_frame = int(match.group(1))
                    break
            
            # Calculate progress percentage
            progress_percent = 0
            if total_frames > 0:
                progress_percent = min(100, (current_frame / total_frames) * 100)
            
            return jsonify({
                'log': '\n'.join(last_lines),
                'current_frame': current_frame,
                'total_frames': total_frames,
                'progress_percent': round(progress_percent, 1)
            })
        except Exception as e:
            return jsonify({'error': f'Error reading log: {str(e)}'}), 500
    
    return jsonify({'log': None, 'current_frame': 0, 'total_frames': 0, 'progress_percent': 0})

@app.route('/api/results/<job_id>')
def get_job_results(job_id: str):
    """Get job results and available files"""
    job_data = redis_client.hgetall(f"job:{job_id}")
    
    if not job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    if job_data.get('status') != JobStatus.COMPLETED:
        return jsonify({'error': 'Job not completed'}), 400
    
    job_dir = app.config['RESULTS_FOLDER'] / job_id
    
    # List available files
    files = []
    if job_dir.exists():
        for file_path in job_dir.rglob('*'):
            if file_path.is_file():
                rel_path = file_path.relative_to(job_dir)
                files.append({
                    'name': file_path.name,
                    'path': str(rel_path),
                    'size': file_path.stat().st_size,
                    'type': 'rmf' if file_path.suffix in ['.rmf', '.rmf3'] else 'other'
                })
    
    return jsonify({
        'job_id': job_id,
        'status': job_data.get('status'),
        'outputs': json.loads(job_data.get('outputs', '{}')),
        'files': files
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/config')
def get_config():
    """Get application configuration including ColabFold status"""
    return jsonify({
        'colabfold_enabled': not app.config['DISABLE_COLABFOLD'],
        'disable_colabfold': app.config['DISABLE_COLABFOLD']
    })

@app.route('/api/colabfold/status')
def colabfold_status():
    """Get status of the ColabFold queue and watcher"""
    queue_dir = Path(os.environ.get('CF_QUEUE_DIR', '/app/cf_queue')).resolve()
    
    if not queue_dir.exists():
        return jsonify({
            'status': 'error',
            'message': 'Queue directory not found',
            'queue_count': 0
        })
    
    jobs = []
    for job_dir in queue_dir.iterdir():
        if job_dir.is_dir():
            status = 'unknown'
            if (job_dir / 'DONE').exists():
                status = 'done'
            elif (job_dir / 'ERROR').exists():
                status = 'error'
            elif (job_dir / 'STARTED').exists():
                status = 'running'
            elif (job_dir / 'READY').exists():
                status = 'ready'
            
            jobs.append({
                'id': job_dir.name,
                'status': status,
                'created_at': datetime.fromtimestamp(job_dir.stat().st_ctime).isoformat()
            })
    
    return jsonify({
        'status': 'ok',
        'queue_count': len([j for j in jobs if j['status'] in ['ready', 'running']]),
        'jobs': jobs
    })

if __name__ == '__main__':
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)
