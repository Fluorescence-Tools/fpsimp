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

def get_sampling_progress(job_id: str):
    """
    Parse sampling log to get progress information.
    Returns dict with current_frame, total_frames, progress_percent.
    """
    log_file = config.RESULTS_FOLDER / job_id / 'sampling.log'
    result = {
        'current_frame': 0,
        'total_frames': 100000, # Default, will be updated from log/redis
        'progress_percent': 0,
        'log_subset': ''
    }
    
    if not log_file.exists():
        return result
        
    try:
        # Try to get total_frames from Redis parameters if available
        # This is optional here as the caller might prefer to pass it, 
        # but good for self-containedness.
        try:
            from app import redis_client
            import json
            job_data = redis_client.hgetall(f"job:{job_id}")
            if job_data and 'parameters' in job_data:
                params = json.loads(job_data['parameters'])
                if 'frames' in params:
                    result['total_frames'] = int(params['frames'])
        except:
            pass
            
        # Read file
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        result['log_subset'] = "".join(lines[-200:])
            
        # Parse total frames from beginning if not already set or to confirm
        import re
        for line in lines[:500]: 
            m = re.search(r'Starting sampling:\s*(\d+)\s*frames', line)
            if m:
                result['total_frames'] = int(m.group(1))
                break
        
        # Parse current frame from end (scan backwards)
        for line in reversed(lines[-200:]):
            m = re.search(r'(?:Frame|frame)[:\s]+(\d+)', line)
            if m:
                result['current_frame'] = int(m.group(1))
                break
        
        if result['total_frames'] > 0:
            result['progress_percent'] = min(100, int((result['current_frame'] / result['total_frames']) * 100))
            
    except Exception as e:
        # Just log error but return default result
        print(f"Error parsing sampling progress: {e}")
        
    return result
