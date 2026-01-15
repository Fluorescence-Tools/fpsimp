"""
Celery tasks for FPSIMP.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# We'll import these from app to avoid complex re-initialization
# The app will import this module AFTER initializing celery
from app import app, celery, redis_client
from models import JobStatus
from fpsim.pipeline import PipelineConfig, run_fpsim_pipeline
from services.email_service import send_job_notification

@celery.task(bind=True, name='run_colabfold_task')
def run_colabfold_task(self, fasta_path: str, output_dir: str, extra_args: str = "--num-models 1", gpu: int = None):
    """
    ColabFold task that uses shared volumes for processing.
    
    Args:
        fasta_path: Path to input FASTA file
        output_dir: Directory to write output files
        extra_args: Additional arguments to pass to colabfold_batch
        gpu: GPU device ID to use
        
    Returns:
        Path to the best ranked PDB file
    """
    try:
        from utils.structure import run_colabfold_container
        
        # Determine job ID from output_dir (it's results/<job_id>/colabfold)
        job_id = Path(output_dir).parent.name
        
        # Update job status
        redis_client.hset(f"job:{job_id}", "status", JobStatus.COLABFOLD_RUNNING)
        redis_client.hset(f"job:{job_id}", "current_step", "colabfold")
        redis_client.hset(f"job:{job_id}", "colabfold_started_at", datetime.now().isoformat())
        
        # Run ColabFold container
        pdb_result = run_colabfold_container(
            fasta=Path(fasta_path),
            out_dir=Path(output_dir),
            extra_args=extra_args,
            container_name=f"fpsimp-cf-{job_id[:8]}"
        )
        
        # Update job status
        redis_client.hset(f"job:{job_id}", "status", JobStatus.COLABFOLD_COMPLETE)
        redis_client.hset(f"job:{job_id}", "colabfold_completed_at", datetime.now().isoformat())
        redis_client.hset(f"job:{job_id}", "af_pdb", str(pdb_result))
        
        return str(pdb_result)
        
    except Exception as e:
        # Update job status with failure
        # We need to find job_id again if it failed before reaching the line above
        try:
            job_id = Path(output_dir).parent.name
            redis_client.hset(f"job:{job_id}", "status", JobStatus.COLABFOLD_FAILED)
            redis_client.hset(f"job:{job_id}", "colabfold_failed_at", datetime.now().isoformat())
            redis_client.hset(f"job:{job_id}", "error", f"ColabFold failed: {str(e)}")
            
            # Send notification
            send_job_notification(job_id, "FAILED", f"ColabFold generation failed: {str(e)}", redis_client)
        except:
            pass
            
        self.update_state(state='FAILURE', meta={'status': f'ColabFold failed: {str(e)}'})
        raise

@celery.task(bind=True, name='run_pipeline_task')
def run_pipeline_task(self, job_id: str, config_dict: Dict[str, Any], colabfold_result: str = None):
    """Celery task to run fpsim pipeline (after ColabFold is done)"""
    try:
        # Update job status to running
        redis_client.hset(f"job:{job_id}", "status", JobStatus.RUNNING)
        redis_client.hset(f"job:{job_id}", "pipeline_started_at", datetime.now().isoformat())
        redis_client.hset(f"job:{job_id}", "worker_id", self.request.id or 'manual')

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
        def update_step_status(step_name):
            try:
                redis_client.hset(f"job:{job_id}", "current_step", step_name)
            except Exception as e:
                print(f"Failed to update step status: {e}")

        config_obj = PipelineConfig(**config_dict, status_callback=update_step_status)

        # Run the pipeline
        outputs = run_fpsim_pipeline(config_obj)
        
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
        
        # Send notification
        send_job_notification(job_id, "COMPLETED", "Your FPsimP job has finished successfully.", redis_client)
        
        return {"status": "completed", "outputs": serializable_outputs}
        
    except Exception as e:
        redis_client.hset(f"job:{job_id}", "status", JobStatus.FAILED)
        redis_client.hset(f"job:{job_id}", "error", str(e))
        redis_client.hset(f"job:{job_id}", "failed_at", datetime.now().isoformat())
        
        # Send notification
        send_job_notification(job_id, "FAILED", f"Your FPsimP job failed with error: {str(e)}", redis_client)
        
        raise

@celery.task(bind=True, name='run_fpsim_job')
def run_fpsim_job(self, job_id: str, config_dict: Dict[str, Any]):
    """Celery task to run fpsim pipeline"""
    try:
        # Update job status to running with heartbeat
        redis_client.hset(f"job:{job_id}", "status", JobStatus.RUNNING)
        redis_client.hset(f"job:{job_id}", "started_at", datetime.now().isoformat())
        redis_client.hset(f"job:{job_id}", "heartbeat", datetime.now().isoformat())
        redis_client.hset(f"job:{job_id}", "worker_id", self.request.id or 'manual')
        
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
            from config import config
            job_dir = Path(config.RESULTS_FOLDER) / job_id
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
            
            # Run pipeline directly
            try:
                result = run_pipeline_task(job_id, config_dict_serializable)
                return result
            except Exception as e:
                app.logger.error(f"Error running pipeline: {e}")
                raise
                
    except Exception as e:
        app.logger.error(f"Error in run_fpsim_job: {e}")
        redis_client.hset(f"job:{job_id}", "status", JobStatus.FAILED)
        redis_client.hset(f"job:{job_id}", "error", str(e))
        redis_client.hset(f"job:{job_id}", "failed_at", datetime.now().isoformat())
        
        # Send notification
        send_job_notification(job_id, "FAILED", f"Job submission failed: {str(e)}", redis_client)
        raise
