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

def _create_structure_response(
    upload_id: str, 
    filename: str, 
    pdb_path: Path, 
    seqs_by_chain: dict,
    rigid_threshold: float = 70.0,
    min_rb_len: int = 12,
    min_linker_len: int = 10,
    model_disordered_as_beads: bool = False
):
    """Create standardized response for structure download"""
    from fpsim.segments import parse_plddt_from_pdb, segments_from_plddt
    
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
    plddt_by_chain = {}
    segments_by_chain = {}

    for chain_id, sequence in seqs_by_chain.items():
        # Detect FP
        fp_info = detect_fluorescent_protein(sequence)
        
        # Parse pLDDT
        try:
            raw_plddt = parse_plddt_from_pdb(pdb_path, chain_id)
            # Keep native integer keys for calculation
            plddt = raw_plddt
            
            # For JSON output only
            plddt_for_json = {str(k): float(v) for k, v in raw_plddt.items()}
            plddt_by_chain[chain_id] = plddt_for_json
        except Exception as e:
            print(f"Warning: Could not parse pLDDT for chain {chain_id}: {e}")
            plddt = {}

        # Generate Segments
        # segments_from_plddt expects list of (name, start, end, identity)
        fp_domains_for_seg = []
        if fp_info and 'fps' in fp_info:
             # If multiple FPs were detected (e.g. via motif search/alignment)
             for fp in fp_info['fps']:
                 fp_domains_for_seg.append((fp['name'], fp['start'], fp['end'], 0.99)) # Dummy identity if not available
        elif fp_info and 'fp_start' in fp_info and 'fp_end' in fp_info:
            fp_domains_for_seg.append((fp_info['name'], fp_info['fp_start'], fp_info['fp_end'], 0.99))

        if plddt:
            sorted_residues = sorted(plddt.keys())
            segments = segments_from_plddt(
                len(sequence),
                plddt,
                fp_domains_for_seg,
                residue_numbers=sorted_residues,
                rigid_threshold=rigid_threshold,
                min_rb_len=min_rb_len,
                min_linker_len=min_linker_len,
                model_disordered_as_beads=model_disordered_as_beads
            )
            # Add chain_id to segments for frontend convenience
            for seg in segments:
                seg['chain_id'] = chain_id
            segments_by_chain[chain_id] = segments
        else:
             segments_by_chain[chain_id] = []

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
            if 'fps' in fp_info: 
                seq_data['fps'] = fp_info['fps']
            else:
                # If no explicit 'fps' list but we have start/end (from alignment/motif), 
                # create a single entry list for compatibility with viewer.js
                # Note: viewer.js expects 0-based indices for start/end in the fps list
                if 'fp_start' in fp_info and 'fp_end' in fp_info:
                    seq_data['fps'] = [{
                        'name': fp_info['name'],
                        'start': fp_info['fp_start'] - 1, # Convert 1-based to 0-based
                        'end': fp_info['fp_end'] - 1,     # Convert 1-based to 0-based
                        'color': fp_info['color'],
                        'dipole_triplets': fp_info.get('dipole_triplets', [])
                    }]
        sequences.append(seq_data)

    # Debug logging
    with open("/app/fpsim_debug.log", "a") as f:
        f.write(f"DEBUG: Response prep - plddt_by_chain keys: {list(plddt_by_chain.keys())}\n")
        if plddt_by_chain:
             first_chain = list(plddt_by_chain.keys())[0]
             f.write(f"DEBUG: First chain ({first_chain}) sample: {str(list(plddt_by_chain[first_chain].items())[:3])}\n")
             f.write(f"DEBUG: Segments keys: {list(segments_by_chain.keys())}\n")

    return jsonify({
        'upload_id': upload_id,
        'filename': filename,
        'file_path': str(pdb_path),
        'file_url': f"/api/download_structure/{upload_id}_{filename}",
        'file_size': pdb_path.stat().st_size,
        'sequences': sequences,
        'plddt': plddt_by_chain,
        'segments': segments_by_chain,
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
