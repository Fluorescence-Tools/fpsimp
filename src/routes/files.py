"""
File handling routes (uploads, downloads) for FPSIMP.
"""
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from flask import Blueprint, jsonify, request, send_file, current_app, abort
from werkzeug.utils import secure_filename

from utils.bio import list_fasta_sequences
from utils.structure import (
    _download_pdb_direct, 
    _download_alphafold_intelligent, 
    _fallback_alphafold_download,
    _create_structure_response
)

files_bp = Blueprint('files', __name__)

@files_bp.route('/api/upload_fasta', methods=['POST'])
def upload_fasta():
    """Upload FASTA file and return sequences"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        upload_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = current_app.config['UPLOAD_FOLDER'] / f"{upload_id}_{filename}"
        file.save(str(file_path))
        
        sequences = list_fasta_sequences(file_path)
        
        return jsonify({
            'upload_id': upload_id,
            'filename': filename,
            'sequences': sequences
        })

@files_bp.route('/api/upload_pdb', methods=['POST'])
def upload_pdb():
    """Upload PDB/mmCIF file"""
    # Support both 'file' (original) and 'pdb' (used in frontend) keys
    file = request.files.get('file') or request.files.get('pdb')
    
    if not file:
        return jsonify({'error': 'No file part (expected "file" or "pdb" key)'}), 400
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        from fpsim.utils import extract_sequences_from_structure
        upload_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        pdb_path = current_app.config['UPLOAD_FOLDER'] / f"{upload_id}_{filename}"
        file.save(str(pdb_path))
        
        seqs_by_chain = extract_sequences_from_structure(pdb_path)
        if not seqs_by_chain:
            pdb_path.unlink(missing_ok=True)
            return jsonify({'error': 'No sequences could be extracted from structure'}), 400
            
        return _create_structure_response(upload_id, filename, pdb_path, seqs_by_chain)

@files_bp.route('/api/fetch_structure', methods=['POST'])
def fetch_structure():
    """Fetch structure by ID"""
    data = request.json
    source = data.get('source', 'pdb')
    identifier = data.get('id', '').strip().upper()
    
    if not identifier:
        return jsonify({'error': 'No identifier provided'}), 400
        
    if source == 'pdb':
        # Prefer CIF as it's the modern standard, fallback to PDB
        url = f"https://files.rcsb.org/download/{identifier}.cif"
        filename = f"{identifier}.cif"
        
        # Check if CIF exists, if not try PDB
        import requests
        try:
            head = requests.head(url)
            if head.status_code != 200:
                url = f"https://files.rcsb.org/download/{identifier}.pdb"
                filename = f"{identifier}.pdb"
        except:
            url = f"https://files.rcsb.org/download/{identifier}.pdb"
            filename = f"{identifier}.pdb"
            
        return _download_pdb_direct(url, filename)
    elif source == 'alphafold':
        return _download_alphafold_intelligent(identifier)
    else:
        return jsonify({'error': 'Invalid source'}), 400

@files_bp.route('/api/download_structure/<filename>')
def download_structure(filename: str):
    """Serve structure files for 3D viewer"""
    from app import redis_client
    # Security: only allow files from upload folder
    safe_filename = secure_filename(filename)
    file_path = current_app.config['UPLOAD_FOLDER'] / safe_filename
    
    if not file_path.exists():
        abort(404)
        
    return send_file(file_path)

@files_bp.route('/api/download/<job_id>/<filename>')
def download_file(job_id: str, filename: str):
    """Download result file"""
    from app import redis_client
    from models import JobStatus
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data: return jsonify({'error': 'Job not found'}), 404
    if job_data.get('status') != JobStatus.COMPLETED: return jsonify({'error': 'Job not completed'}), 400
    
    from urllib.parse import unquote
    decoded_filename = unquote(filename)
    if '..' in decoded_filename or decoded_filename.startswith('/'): abort(404)
    
    job_dir = current_app.config['RESULTS_FOLDER'] / job_id
    full_file_path = job_dir / decoded_filename
    if not full_file_path.exists() or not full_file_path.is_relative_to(job_dir): abort(404)
    
    return send_file(full_file_path, as_attachment=True)

@files_bp.route('/api/download_zip/<job_id>')
def download_zip(job_id: str):
    """Download entire job folder as zip"""
    from app import redis_client
    from models import JobStatus
    job_data = redis_client.hgetall(f"job:{job_id}")
    if not job_data: return jsonify({'error': 'Job not found'}), 404
    if job_data.get('status') != JobStatus.COMPLETED: return jsonify({'error': 'Job not completed'}), 400
    
    job_dir = current_app.config['RESULTS_FOLDER'] / job_id
    if not job_dir.exists(): return jsonify({'error': 'Job directory not found'}), 404
    
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    temp_zip.close()
    try:
        shutil.make_archive(temp_zip.name[:-4], 'zip', job_dir)
        return send_file(temp_zip.name[:-4] + '.zip', as_attachment=True, download_name=f'fpsim_results_{job_id}.zip', mimetype='application/zip')
    except Exception as e:
        if os.path.exists(temp_zip.name): os.unlink(temp_zip.name)
        return jsonify({'error': f'Failed to create zip: {str(e)}'}), 500

@files_bp.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@files_bp.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

@files_bp.route('/api/config')
def get_config_endpoint():
    """Get application configuration"""
    return jsonify({
        'disable_colabfold': current_app.config.get('DISABLE_COLABFOLD', True),
        'gpu_enabled': current_app.config.get('GPU_ENABLED', False),
        'max_content_length': current_app.config.get('MAX_CONTENT_LENGTH')
    })

@files_bp.route('/api/colabfold_status')
def colabfold_status():
    """Get status of the ColabFold queue and watcher"""
    queue_dir = Path(os.environ.get('CF_QUEUE_DIR', '/tmp/colabfold_queue'))
    active_jobs = [d.name for d in queue_dir.iterdir() if d.is_dir() and not (d / 'DONE').exists() and not (d / 'ERROR').exists()]
    return jsonify({'active_jobs_count': len(active_jobs), 'active_jobs': active_jobs, 'enabled': not current_app.config['DISABLE_COLABFOLD']})
