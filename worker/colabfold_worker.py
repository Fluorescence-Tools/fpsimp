#!/usr/bin/env python3
"""
Simple ColabFold Worker Script
Minimal worker that only processes ColabFold jobs
No Celery, no complex dependencies - just ColabFold
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

def run_colabfold_job(input_file: str, output_dir: str, extra_args: str = "--num-models 1") -> dict:
    """Run ColabFold job and return results"""
    
    # Find colabfold_batch
    colabfold_path = os.getenv('COLABFOLD_PATH', '/colabfold/localcolabfold/colabfold-conda/bin')
    colabfold_batch = os.path.join(colabfold_path, 'colabfold_batch')
    
    if not os.path.exists(colabfold_batch):
        return {
            'success': False,
            'error': f'ColabFold not found at {colabfold_batch}',
            'input_file': input_file,
            'output_dir': output_dir
        }
    
    # Parse extra arguments
    if extra_args:
        args = extra_args.split()
    else:
        args = ["--num-models", "1"]
    
    # Build command
    cmd = [colabfold_batch] + args + [input_file, output_dir]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run ColabFold
    start_time = datetime.now()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Find output PDB file
        pdb_file = None
        if result.returncode == 0:
            output_path = Path(output_dir)
            for file in output_path.glob("*_rank_1_*.pdb"):
                pdb_file = str(file)
                break
        
        return {
            'success': result.returncode == 0,
            'input_file': input_file,
            'output_dir': output_dir,
            'pdb_file': pdb_file,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'return_code': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'ColabFold job timed out (1 hour)',
            'input_file': input_file,
            'output_dir': output_dir,
            'duration': 3600
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'input_file': input_file,
            'output_dir': output_dir
        }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Simple ColabFold Worker')
    parser.add_argument('--input-file', required=True, help='Input FASTA file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--extra-args', default='--num-models 1', help='Extra ColabFold arguments')
    parser.add_argument('--result-file', help='File to write JSON result to')
    
    args = parser.parse_args()
    
    # Run ColabFold job
    result = run_colabfold_job(
        input_file=args.input_file,
        output_dir=args.output_dir,
        extra_args=args.extra_args
    )
    
    # Print result for capture
    print(json.dumps(result, indent=2))
    
    # Write to file if specified
    if args.result_file:
        with open(args.result_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)

if __name__ == '__main__':
    main()
