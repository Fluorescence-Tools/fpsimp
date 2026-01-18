"""
View routes for FPSIMP.
"""
import json
from flask import Blueprint, render_template, current_app, request
from models import JobStatus

views_bp = Blueprint('views', __name__)

@views_bp.route('/')
def index():
    """Main page"""
    from app import render_static_template
    return render_static_template('index.html')

@views_bp.route('/results/<job_id>')
def results_page(job_id: str):
    """Results page for a specific job"""
    from app import redis_client, render_static_template
    job_data = redis_client.hgetall(f"job:{job_id}")
    
    if not job_data:
        return "Job not found", 404
    
    # Get results if job is completed
    status = job_data.get('status', 'unknown')
    results = None
    all_files = []
    
    if status == JobStatus.COMPLETED:
        try:
            results = json.loads(job_data.get('outputs', '{}'))
        except:
            results = {}
            
        # List all files in job directory for display
        job_dir = current_app.config['RESULTS_FOLDER'] / job_id
        if job_dir.exists():
            for f in job_dir.rglob('*'):
                if f.is_file():
                    rel_path = f.relative_to(job_dir)
                    
                    # Format size
                    size_bytes = f.stat().st_size
                    for unit in ['B', 'KB', 'MB', 'GB']:
                        if size_bytes < 1024:
                            size_formatted = f"{size_bytes:.1f} {unit}"
                            break
                        size_bytes /= 1024
                    else:
                        size_formatted = f"{size_bytes:.1f} TB"
                        
                    all_files.append({
                        'name': f.name,
                        'path': str(rel_path),
                        'extension': f.suffix.lower(),
                        'size_formatted': size_formatted,
                        'size': f.stat().st_size
                    })
            
            # Sort files by name (directories/groups typically handled by table sorting if implemented, but simple sort here)
            all_files.sort(key=lambda x: x['name'])
            
    # Parse parameters if present
    parameters = {}
    if 'parameters' in job_data:
        try:
            parameters = json.loads(job_data['parameters'])
        except:
            pass

    return render_static_template(
        'results.html',
        job_id=job_id,
        status=status,
        job_data=job_data,
        parameters=parameters,
        results=results,
        all_files=all_files,
        config=current_app.config,
        JobStatus=JobStatus
    )
