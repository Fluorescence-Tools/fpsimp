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
    if status == JobStatus.COMPLETED:
        try:
            results = json.loads(job_data.get('outputs', '{}'))
        except:
            results = {}
            
    return render_static_template(
        'results.html',
        job_id=job_id,
        status=status,
        job_data=job_data,
        results=results,
        config=current_app.config
    )
