#!/usr/bin/env python3
"""
Simple ColabFold Job Watcher - CPU Mode
"""

import os
import sys
import time
import json
import signal
from pathlib import Path
from datetime import datetime

# Configuration
QUEUE_DIR = Path(os.environ.get('CF_QUEUE_DIR', '/queue'))
POLL_INTERVAL = float(os.environ.get('CF_POLL_INTERVAL', '2.0'))
CLEANUP_DAYS = int(os.environ.get('CLEANUP_JOBS_AFTER_DAYS', '7'))
JOB_TIMEOUT = int(os.environ.get('JOB_TIMEOUT', str(3600 * 6)))

def setup_logging():
    """Setup simple logging to stdout"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("ColabFold Job Watcher - CPU Mode")
    logger.info("=" * 60)
    logger.info(f"Queue directory: {QUEUE_DIR}")
    logger.info(f"Poll interval: {POLL_INTERVAL}s")
    logger.info(f"Job timeout: {JOB_TIMEOUT}s")
    logger.info(f"Cleanup after: {CLEANUP_DAYS} days")
    
    # Ensure queue directory exists
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Watching for jobs in {QUEUE_DIR}")
    
    # Setup signal handlers
    shutdown = False
    def signal_handler(signum, frame):
        nonlocal shutdown
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown = True
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Main loop
    while not shutdown:
        try:
            # Check for jobs
            for job_dir in sorted(QUEUE_DIR.iterdir()):
                if not job_dir.is_dir():
                    continue
                
                # Skip if already processed
                if (job_dir / 'DONE').exists() or (job_dir / 'ERROR').exists():
                    continue
                
                # Check if job is ready
                ready_file = job_dir / 'READY'
                if not ready_file.exists():
                    continue
                
                logger.info(f"Found job: {job_dir.name}")
                
                # Process the job
                try:
                    process_job(job_dir, logger)
                except Exception as e:
                    logger.error(f"Error processing job {job_dir.name}: {e}")
                    (job_dir / 'ERROR').write_text(str(e))
                
                # Check if we should shutdown
                if shutdown:
                    break
            
            # Cleanup old jobs
            cleanup_old_jobs(logger)
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        
        # Sleep before next poll
        if not shutdown:
            time.sleep(POLL_INTERVAL)
    
    logger.info("ColabFold watcher shutdown complete")

def process_job(job_dir: Path, logger):
    """Process a single ColabFold job"""
    logger.info(f"Processing job: {job_dir.name}")
    
    input_fasta = job_dir / 'input.fasta'
    if not input_fasta.exists():
        logger.error(f"Missing input.fasta for job {job_dir.name}")
        (job_dir / 'ERROR').write_text('Missing input.fasta')
        return
    
    out_dir = job_dir / 'out'
    out_dir.mkdir(exist_ok=True)
    
    # Build command - CPU mode by default
    cmd = [
        'colabfold_batch',
        '--num-models', '1',
        '--msa-mode', 'mmseqs2_unpaired',
        '--model-type', 'auto',
        str(input_fasta),
        str(out_dir)
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    # Run ColabFold
    import subprocess
    log_file = job_dir / 'log.txt'
    
    try:
        with open(log_file, 'w') as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT
            )
            
            # Wait for completion with timeout
            try:
                proc.wait(timeout=JOB_TIMEOUT)
            except subprocess.TimeoutExpired:
                logger.warning(f"Job {job_dir.name} timed out")
                proc.kill()
                (job_dir / 'TIMEOUT').write_text(f"Timeout after {JOB_TIMEOUT}s")
                (job_dir / 'ERROR').write_text('Job timed out')
                return
            
            if proc.returncode != 0:
                logger.error(f"ColabFold failed with code {proc.returncode}")
                (job_dir / 'ERROR').write_text(f'Exit code: {proc.returncode}')
                return
    
    except Exception as e:
        logger.error(f"Error running ColabFold: {e}")
        (job_dir / 'ERROR').write_text(str(e))
        return
    
    # Find output PDB
    pdbs = list(out_dir.glob('*.pdb'))
    if not pdbs:
        logger.error(f"No PDB produced for job {job_dir.name}")
        (job_dir / 'ERROR').write_text('No PDB produced')
        return
    
    # Write result
    result = {"af_pdb": str(pdbs[0].relative_to(job_dir))}
    (job_dir / 'result.json').write_text(json.dumps(result))
    (job_dir / 'DONE').write_text('ok')
    
    logger.info(f"Job {job_dir.name} completed successfully")

def cleanup_old_jobs(logger):
    """Clean up old completed jobs"""
    if CLEANUP_DAYS <= 0:
        return
    
    cutoff = datetime.now().timestamp() - (CLEANUP_DAYS * 24 * 3600)
    
    for job_dir in QUEUE_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        
        # Check if job is old and completed
        if (job_dir / 'DONE').exists() or (job_dir / 'ERROR').exists():
            try:
                mtime = job_dir.stat().st_mtime
                if mtime < cutoff:
                    logger.info(f"Cleaning up old job: {job_dir.name}")
                    import shutil
                    shutil.rmtree(job_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup {job_dir.name}: {e}")

if __name__ == '__main__':
    main()
