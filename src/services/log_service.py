"""
Logging service for FPSIMP.
"""
from config import config

def get_job_cli_output(job_id: str):
    """Get CLI output for a job"""
    output_lines = []
    log_file = config.RESULTS_FOLDER / job_id / 'pipeline.log'
    
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
