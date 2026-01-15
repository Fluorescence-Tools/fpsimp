"""
Structure-related utility functions for FPSIMP.
"""
import requests
from pathlib import Path
from flask import jsonify
from config import config
from fpsim.utils import extract_sequences_from_structure, write_fasta_from_pdb
from .bio import detect_fluorescent_protein

def run_colabfold_container(
    fasta: Path,
    out_dir: Path,
    extra_args: str = "--num-models 1",
    container_name: str = "avnn-colabfold"
):
    """Run ColabFold in a dedicated Docker container."""
    import subprocess
    
    # Ensure out_dir exists
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Map paths for Docker
    # We assume the current results and uploads folders are mounted in the container
    container_fasta = f"/inputs/{fasta.name}"
    container_out = "/outputs"
    
    # Pull image if not present (optional, but good for first run)
    # subprocess.run(["docker", "pull", "mindenas/colabfold:latest"], check=True)
    
    # Construct Docker command
    cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "--gpus", "all",
        "-v", f"{fasta.parent}:/inputs:ro",
        "-v", f"{out_dir}:/outputs",
        "mindenas/colabfold:latest",
        "colabfold_batch",
        container_fasta,
        container_out
    ]
    
    # Add extra arguments if provided
    if extra_args:
        # Split extra_args while respecting quotes (simple split for now)
        cmd.extend(extra_args.split())
    
    print(f"Running ColabFold container: {' '.join(cmd)}")
    
    # Run container
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        print(f"ColabFold container failed: {process.stderr}")
        raise RuntimeError(f"ColabFold container failed: {process.stderr}")
    
    # Copy results back from container
    # (Actually they are already in out_dir because of the volume mount)
    
    # Look for the best ranked structure
    results_dir = out_dir
    exts = [".pdb", ".cif", ".mmcif"]
    structures = []
    
    for ex in exts:
        structures.extend(list(results_dir.glob(f"*rank_1*{ex}")))
        
    if not structures:
        for ex in exts:
            structures.extend(list(results_dir.glob(f"*{ex}")))
    
    structures.sort() # simple deterministic choice
    
    if not structures:
        raise RuntimeError("No structure files (.pdb or .cif) found in ColabFold output")
    
    return structures[0]

def _create_structure_response(upload_id: str, filename: str, pdb_path: Path, seqs_by_chain: dict):
    """Create standardized response for structure download"""
    # Create merged FASTA under uploads with same upload_id
    merged_fasta_path = config.UPLOAD_FOLDER / f"{upload_id}_{Path(filename).stem}.fasta"
    
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

def _download_pdb_direct(url: str, filename: str):
    """Download structure file directly from URL"""
    import uuid
    upload_id = str(uuid.uuid4())
    pdb_path = config.UPLOAD_FOLDER / f"{upload_id}_{filename}"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return jsonify({'error': f'Failed to download structure: {response.status_code}'}), 400
        
        # Save as binary for CIF/PDB
        with open(pdb_path, 'wb') as f:
            f.write(response.content)
    except Exception as e:
        return jsonify({'error': f'Error downloading structure: {str(e)}'}), 500
    
    # Extract per-chain sequences
    seqs_by_chain = extract_sequences_from_structure(pdb_path)
    if not seqs_by_chain:
        pdb_path.unlink(missing_ok=True)
        return jsonify({'error': 'No sequences could be extracted from structure'}), 400
    
    return _create_structure_response(upload_id, filename, pdb_path, seqs_by_chain)

def _download_alphafold_intelligent(af_id: str):
    """Download latest AlphaFold structure using intelligent version detection (Preferring CIF)"""
    import uuid
    # Try v4 first (latest), fallback to v3, v2, v1
    # For each version, try CIF first, then PDB
    for version in range(4, 0, -1):
        for ext in ['.cif', '.pdb']:
            filename = f"AF-{af_id}-F1-model_v{version}{ext}"
            url = f"https://alphafold.ebi.ac.uk/files/{filename}"
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    upload_id = str(uuid.uuid4())
                    pdb_path = config.UPLOAD_FOLDER / f"{upload_id}_{filename}"
                    with open(pdb_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Extract per-chain sequences
                    seqs_by_chain = extract_sequences_from_structure(pdb_path)
                    if not seqs_by_chain:
                        pdb_path.unlink(missing_ok=True)
                        continue # Try next format/version
                        
                    return _create_structure_response(upload_id, filename, pdb_path, seqs_by_chain)
            except:
                continue
            
    return jsonify({'error': f'Could not find AlphaFold structure for {af_id}'}), 404

def _fallback_alphafold_download(af_id: str):
    """Fallback method for AlphaFold download without intelligent version detection"""
    import uuid
    filename = f"{af_id}.pdb"
    url = f"https://alphafold.ebi.ac.uk/files/AF-{af_id}-F1-model_v4.pdb"
    
    response = requests.get(url)
    if response.status_code != 200:
        # Try v3 as a common fallback
        url = f"https://alphafold.ebi.ac.uk/files/AF-{af_id}-F1-model_v3.pdb"
        response = requests.get(url)
        
    if response.status_code != 200:
        return jsonify({'error': f'Failed to download AlphaFold structure for {af_id}'}), 400
    
    upload_id = str(uuid.uuid4())
    pdb_path = config.UPLOAD_FOLDER / f"{upload_id}_{filename}"
    
    with open(pdb_path, 'w') as f:
        f.write(response.text)
    
    # Extract per-chain sequences
    seqs_by_chain = extract_sequences_from_structure(pdb_path)
    if not seqs_by_chain:
        pdb_path.unlink(missing_ok=True)
        return jsonify({'error': 'No sequences could be extracted from structure'}), 400
    
    return _create_structure_response(upload_id, filename, pdb_path, seqs_by_chain)
