"""
Job management routes for FPSIMP.
"""
import json
import uuid
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from flask import Blueprint, jsonify, request, current_app, abort

from models import JobStatus
from utils.general import format_duration
from utils.bio import FP_LIBRARY, FP_COLOR
from services.log_service import get_job_cli_output

jobs_bp = Blueprint('jobs', __name__)

@jobs_bp.route('/api/submit', methods=['POST'])
def submit_job():
    """Submit fpsim job"""
    from app import redis_client
    from tasks import run_fpsim_job
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
        candidates = sorted(current_app.config['UPLOAD_FOLDER'].glob(f"{upload_id}_*"))
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
                from fpsim.utils import write_fasta_from_pdb
                pdb_src = pdb_candidates[0]
                fasta_path = current_app.config['UPLOAD_FOLDER'] / f"{upload_id}_{pdb_src.stem}.fasta"
                try:
                    write_fasta_from_pdb(pdb_src, fasta_path)
                except Exception as e:
                    return jsonify({'error': f'Failed to derive FASTA from uploaded PDB: {e}'}), 400
            else:
                # Last resort: use the first candidate
                fasta_path = candidates[0]
        
        # Create job output directory
        job_dir = current_app.config['RESULTS_FOLDER'] / job_id
        job_dir.mkdir(exist_ok=True)
        
        # Create inputs directory within job results
        inputs_dir = job_dir / "inputs"
        inputs_dir.mkdir(exist_ok=True)
        
        # Copy input files to results directory and rename them to original names
        import shutil
        for c in candidates:
            # Original name is after the first underscore in 'uuid_filename'
            original_name = c.name.split('_', 1)[-1] if '_' in c.name else c.name
            shutil.copy2(c, inputs_dir / original_name)
        
        # Determine the local FASTA path within the results inputs folder
        local_fasta_path = None
        for c in inputs_dir.iterdir():
            if c.suffix.lower() in ('.fasta', '.fa', '.fas'):
                local_fasta_path = c
                break
        
        # If no FASTA, try to derive from PDB in inputs_dir
        if local_fasta_path is None:
            pdb_candidates = [c for c in inputs_dir.iterdir() if c.suffix.lower() in ('.pdb', '.cif', '.mmcif')]
            if pdb_candidates:
                from fpsim.utils import write_fasta_from_pdb
                pdb_src = pdb_candidates[0]
                local_fasta_path = inputs_dir / f"{pdb_src.stem}.fasta"
                try:
                    write_fasta_from_pdb(pdb_src, local_fasta_path)
                except Exception as e:
                    return jsonify({'error': f'Failed to derive FASTA from uploaded PDB: {e}'}), 400
            else:
                # Fallback to the first file in inputs_dir if any
                all_inputs = list(inputs_dir.iterdir())
                if all_inputs:
                    local_fasta_path = all_inputs[0]
        
        if not local_fasta_path:
            return jsonify({'error': 'No input files found in job directory'}), 400

        # Extract parameters with defaults
        params = data.get('parameters', {})
        membrane_regions = data.get('membrane_regions', params.get('membrane_regions', []))
        
        # Force disable ColabFold if DISABLE_COLABFOLD is set
        if current_app.config['DISABLE_COLABFOLD']:
            params['run_colabfold'] = False
            current_app.logger.info("ColabFold disabled via DISABLE_COLABFOLD environment variable")
        
        # Convert membrane regions and sequences to membrane_seq format
        membrane_seqs = params.get('membrane_seqs', [])
        membrane_seq_list = list(membrane_seqs)
        
        # Also include the sequence_id if specific residue regions are provided
        if membrane_regions and params.get('sequence_id') and params.get('sequence_id') not in membrane_seq_list:
            membrane_seq_list.append(params.get('sequence_id'))
        
        # Ensure membrane is enabled if sequences or regions are provided
        should_enable_membrane = bool(params.get('membrane', False)) or len(membrane_seq_list) > 0 or len(membrane_regions) > 0
        params['membrane'] = should_enable_membrane
        
        # Create pipeline configuration using local input files
        config_dict = {
            'fasta': str(local_fasta_path),
            'out_dir': str(job_dir),
            'sequence_id': params.get('sequence_id'),
            'chain': params.get('chain', 'A'),
            'af_pdb': params.get('af_pdb'), # If user provided a path, it might need translation or it was uploaded
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
        
        # If af_pdb was an uploaded file, point to the local copy
        if config_dict['af_pdb']:
            af_pdb_path = Path(config_dict['af_pdb'])
            # Check if this looks like an uploaded file (uuid prefix)
            if '_' in af_pdb_path.name:
                af_original_name = af_pdb_path.name.split('_', 1)[-1]
                local_af_pdb = inputs_dir / af_original_name
                if local_af_pdb.exists():
                    config_dict['af_pdb'] = str(local_af_pdb)
        
        # Store job metadata
        job_metadata = {
            'job_id': job_id,
            'status': JobStatus.QUEUED,
            'created_at': datetime.now().isoformat(),
            'fasta_filename': fasta_path.name,
            'parameters': json.dumps(params),
            'membrane_regions': json.dumps(data.get('membrane_regions', [])),
            'email': data.get('email') or '' # Handle None if email: null is sent
        }
        
        for key, value in job_metadata.items():
            if value is None: value = '' # Extra safety
            redis_client.hset(f"job:{job_id}", key, value)
        
        serializable_config = {}
        for key, value in config_dict.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                serializable_config[key] = value
            elif isinstance(value, (list, dict, tuple)):
                serializable_config[key] = list(value) if isinstance(value, tuple) else value
            else:
                serializable_config[key] = str(value)
        
        run_fpsim_job.apply_async(args=[job_id, serializable_config], queue='imp_queue')
        
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

@jobs_bp.route('/api/status/<job_id>')
def get_job_status_legacy(job_id: str):
    """Get job status with detailed progress information"""
    from app import redis_client
    job_data = redis_client.hgetall(f"job:{job_id}")
    
    if not job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    if 'parameters' in job_data:
        try:
            job_data['parameters'] = json.loads(job_data['parameters'])
        except: pass
    
    if 'membrane_regions' in job_data:
        try:
            job_data['membrane_regions'] = json.loads(job_data['membrane_regions'])
        except: pass
    
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
                progress_info['queue_position'] = idx + 1
                progress_info['total_queued'] = len(queue_list)
                break
        
        if progress_info['queue_position']:
            progress_info['estimated_wait_seconds'] = progress_info['queue_position'] * 30
            progress_info['estimated_wait_formatted'] = format_duration(progress_info['estimated_wait_seconds'])
    
    elif status == 'running':
        current_step = job_data.get('current_step', 'unknown')
        progress_info['current_step'] = current_step
        
        # Calculate weighted progress
        # ColabFold: 0-10%
        # Sampling: 10-90%
        # Measurements/Analysis: 90-100%
        
        base_progress = 0
        step_progress = 0
        
        if current_step == 'colabfold':
            base_progress = 0
            # We don't have detailed progress for ColabFold, so we fake it or check logs
            # For now, just say 5%
            step_progress = 5
        elif current_step == 'sampling':
            base_progress = 10
            # Get actual sampling progress
            sampling_progress = get_job_cli_output(job_id) # Using this just to check logs if needed, but we used get_sampling_progress
            from services.log_service import get_sampling_progress
            sp = get_sampling_progress(job_id)
            if sp:
                # Scale 0-100% of sampling to 0-80% of total
                step_progress = int(sp.get('progress_percent', 0) * 0.8)
        elif current_step == 'measurements':
            base_progress = 90
            step_progress = 5
            
        progress_info['progress_percent'] = base_progress + step_progress
        progress_info['cli_output'] = get_job_cli_output(job_id)
    
    elif status in ['colabfold_complete', 'sampling_complete', 'completed']:
        progress_info['progress_percent'] = {'colabfold_complete': 10, 'sampling_complete': 90, 'completed': 100}.get(status, 100)
        progress_info['cli_output'] = get_job_cli_output(job_id)
    
    job_data['progress'] = progress_info
    base_url = request.host_url.rstrip('/')
    job_data['results_url'] = f"{base_url}/results/{job_id}"
    
    return jsonify(job_data)

@jobs_bp.route('/api/job_status/<job_id>')
def get_job_status(job_id: str):
    """Get job status and progress information"""
    from app import redis_client
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    status = job_data.get('status', 'unknown')
    heartbeat_str = job_data.get('heartbeat')
    
    if status == JobStatus.RUNNING and heartbeat_str:
        try:
            heartbeat = datetime.fromisoformat(heartbeat_str)
            if heartbeat <= datetime.now() - timedelta(minutes=5):
                status = JobStatus.QUEUED
                current_app.logger.warning(f"Job {job_id} stale heartbeat, marking as queued")
        except:
            status = JobStatus.QUEUED
    
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

@jobs_bp.route('/api/job_intermediate/<job_id>')
def get_job_intermediate_results(job_id: str):
    """Get intermediate results for job"""
    from app import redis_client
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    job_dir = current_app.config['RESULTS_FOLDER'] / job_id
    if not job_dir.exists():
        return jsonify({'error': 'Job directory not found'}), 404
    
    result = {}
    segments_file = job_dir / 'segments.json'
    if segments_file.exists():
        try:
            segments_data = json.loads(segments_file.read_text())
            from fpsim.segments import parse_plddt_from_pdb
            
            if 'chains' in segments_data and 'chain_labels' in segments_data:
                full_sequence = ""
                full_plddt = {}
                full_segments = []
                full_fp_domains = []
                af_pdb = segments_data.get('af_pdb')
                offset = 0
                
                for label in segments_data['chain_labels']:
                    chain_data = segments_data['chains'].get(label, {})
                    chain_seq = chain_data.get('sequence', '')
                    chain_len = chain_data.get('sequence_len', len(chain_seq))
                    if not chain_seq and chain_len > 0: chain_seq = 'X' * chain_len
                    full_sequence += chain_seq
                    
                    if af_pdb and Path(af_pdb).exists():
                        try:
                            chain_plddt = parse_plddt_from_pdb(Path(af_pdb), label)
                            for pos, score in chain_plddt.items(): full_plddt[pos + offset] = score
                        except: pass
                    
                    if 'segments' in chain_data:
                        for seg in chain_data['segments']:
                            full_segments.append({**seg, 'start': seg['start'] + offset, 'end': seg['end'] + offset})
                    if 'fp_domains' in chain_data:
                        for fp in chain_data['fp_domains']:
                            full_fp_domains.append({**fp, 'start': fp['start'] + offset, 'end': fp['end'] + offset})
                    offset += chain_len
                
                result.update({'sequence': full_sequence, 'plddt': full_plddt, 'segments': full_segments, 'fp_domains': full_fp_domains})
            else:
                if 'sequence' in segments_data: result['sequence'] = segments_data['sequence']
                af_pdb = segments_data.get('af_pdb')
                if af_pdb and Path(af_pdb).exists():
                    try: result['plddt'] = parse_plddt_from_pdb(Path(af_pdb), segments_data.get('chain', 'A'))
                    except: pass
                if 'segments' in segments_data: result['segments'] = segments_data['segments']
                if 'fp_domains' in segments_data: result['fp_domains'] = segments_data['fp_domains']
            
            try:
                membrane_regions_str = job_data.get('membrane_regions')
                if membrane_regions_str and membrane_regions_str != '[]':
                    result['membrane_regions'] = json.loads(membrane_regions_str)
                else:
                    params = json.loads(job_data.get('parameters', '{}'))
                    result['membrane_regions'] = params.get('membrane_regions', [])
            except: result['membrane_regions'] = []
                
        except Exception as e: current_app.logger.error(f"Error reading segments.json: {e}")
    
    top_file = job_dir / 'fusion.top.dat'
    if top_file.exists():
        try: result['topology'] = top_file.read_text().split('\n')
        except: pass
        
    return jsonify(result)

@jobs_bp.route('/api/job_diagnostics/<job_id>')
def get_job_diagnostics(job_id: str):
    """Get diagnostic information about job"""
    from app import redis_client, celery
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data:
        return jsonify({'error': 'Job not found'}), 404
    
    diagnostics = {
        'job_id': job_id,
        'status': job_data.get('status', 'unknown'),
        'created_at': job_data.get('created_at'),
        'queue_info': {}, 'worker_info': {}, 'celery_info': {}
    }
    
    try:
        all_job_keys = redis_client.keys('job:*')
        queued_jobs, running_jobs, stale_jobs = [], [], []
        threshold = datetime.now() - timedelta(minutes=5)
        
        for job_key in all_job_keys:
            jd = redis_client.hgetall(job_key)
            status, hb = jd.get('status'), jd.get('heartbeat')
            key_str = job_key.decode() if isinstance(job_key, bytes) else job_key
            if status == JobStatus.QUEUED: queued_jobs.append((key_str, jd.get('created_at')))
            elif status == JobStatus.RUNNING:
                active = False
                if hb:
                    try:
                        if datetime.fromisoformat(hb) > threshold: active = True
                    except: pass
                if active: running_jobs.append(key_str)
                else: stale_jobs.append(key_str)
        
        queued_jobs.sort(key=lambda x: x[1])
        job_pos = next((i + 1 for i, (jk, _) in enumerate(queued_jobs) if jk == f'job:{job_id}'), None)
        diagnostics['queue_info'] = {'queued_count': len(queued_jobs), 'running_count': len(running_jobs), 'stale_count': len(stale_jobs), 'job_position': job_pos}
    except Exception as e: diagnostics['queue_info']['error'] = str(e)
    
    try:
        inspect = celery.control.inspect(timeout=5.0)
        # Check active tasks
        active = inspect.active()
        # Check online workers (ping)
        ping = inspect.ping()
        
        workers_detected = 0
        worker_names = []
        
        if ping:
            worker_names = list(ping.keys())
            workers_detected = len(worker_names)
        elif active:
            # Fallback to active if ping fails but active returns something
            worker_names = list(active.keys())
            workers_detected = len(worker_names)
        
        # Fallback: If no workers detected via inspect, but we have RUNNING jobs with recent heartbeat,
        # then we know at least one worker is alive.
        if workers_detected == 0 and diagnostics['counts']['RUNNING'] > 0:
            # Check for recent heartbeats in the running jobs
            import time
            current_ts = time.time()
            recent_activity = False
            
            for job_id in diagnostics['jobs']['running']:
                job_key = f"job:{job_id}"
                hb = redis_client.hget(job_key, "heartbeat")
                if hb:
                    try:
                        # Heartbeat is usually isoformat string
                        hb_ts = datetime.datetime.fromisoformat(hb).timestamp()
                        if current_ts - hb_ts < 120:  # Within last 2 minutes
                            recent_activity = True
                            break
                    except: pass
            
            if recent_activity:
                workers_detected = 1
                worker_names = ['worker-inferred-from-activity']
            
        active_tasks_count = 0
        if active:
            active_tasks_count = sum(len(t) for t in active.values())
        # If we inferred a worker, we assume it's running the number of running jobs
        elif workers_detected > 0 and active_tasks_count == 0:
             active_tasks_count = diagnostics['counts']['RUNNING']
            
        if workers_detected > 0:
            diagnostics['worker_info'] = {
                'workers_detected': workers_detected, 
                'worker_names': worker_names, 
                'active_tasks': active_tasks_count
            }
    except: pass
    
    return jsonify(diagnostics)

@jobs_bp.route('/api/admin/cleanup_stale_jobs', methods=['POST'])
def cleanup_stale_jobs():
    """Clean up jobs stuck in running state"""
    from app import redis_client
    try:
        all_job_keys = redis_client.keys('job:*')
        cleaned = []
        cutoff = datetime.now() - timedelta(hours=2)
        threshold = datetime.now() - timedelta(minutes=5)
        
        for job_key in all_job_keys:
            jd = redis_client.hgetall(job_key)
            if jd.get('status') == JobStatus.RUNNING and jd.get('started_at'):
                try:
                    if datetime.fromisoformat(jd.get('started_at')) < cutoff:
                        job_id = (job_key.decode() if isinstance(job_key, bytes) else job_key).replace('job:', '')
                        log = current_app.config['RESULTS_FOLDER'] / job_id / 'sampling.log'
                        if not log.exists() or datetime.fromtimestamp(log.stat().st_mtime) < threshold:
                            redis_client.hset(f"job:{job_id}", "status", JobStatus.FAILED)
                            redis_client.hset(f"job:{job_id}", "error", "Job stalled")
                            cleaned.append(job_id)
                except: pass
        return jsonify({'success': True, 'cleaned_count': len(cleaned)})
    except Exception as e: return jsonify({'error': str(e)}), 500

@jobs_bp.route('/api/job_cancel/<job_id>', methods=['POST'])
def cancel_job(job_id: str):
    """Cancel a running job"""
    from app import redis_client, celery
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data: return jsonify({'error': 'Job not found'}), 404
    
    worker_id = job_data.get('worker_id')
    if worker_id: celery.control.revoke(worker_id, terminate=True)
    
    redis_client.hset(f"job:{job_id}", "status", "cancelled")
    redis_client.hset(f"job:{job_id}", "cancelled_at", datetime.now().isoformat())
    return jsonify({'success': True, 'message': 'Job cancellation requested'})

@jobs_bp.route('/api/job_sampling_log/<job_id>')
def get_job_sampling_log(job_id: str):
    """Get IMP sampling log with parsing for progress"""
    log_file = current_app.config['RESULTS_FOLDER'] / job_id / 'sampling.log'
    if not log_file.exists(): return jsonify({'error': 'Sampling log not found'}), 404
    
    try:
        from services.log_service import get_sampling_progress
        progress_data = get_sampling_progress(job_id)
        
        return jsonify({
            'log': progress_data.get('log_subset', ''),
            'current_frame': progress_data.get('current_frame', 0),
            'total_frames': progress_data.get('total_frames', 100000),
            'progress_percent': progress_data.get('progress_percent', 0)
        })
    except Exception as e: return jsonify({'error': str(e)}), 500

@jobs_bp.route('/api/results/<job_id>')
def get_job_results(job_id: str):
    """Get job results and available files"""
    from app import redis_client
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data: return jsonify({'error': 'Job not found'}), 404
    
    job_dir = current_app.config['RESULTS_FOLDER'] / job_id
    if not job_dir.exists(): return jsonify({'error': 'Job directory not found'}), 404
    
    files = []
    for f in job_dir.rglob('*'):
        if f.is_file():
            files.append({'name': str(f.relative_to(job_dir)), 'size': f.stat().st_size, 'url': f"/api/download/{job_id}/{f.relative_to(job_dir)}"})
    
    return jsonify({'job_id': job_id, 'status': job_data.get('status'), 'files': files})
