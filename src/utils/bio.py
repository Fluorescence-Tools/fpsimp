"""
Bio-related utility functions for FPSIMP.
"""
import re
from Bio import SeqIO
from fpsim.fp_lib import get_fp_library

# Initialize FP library once when module is loaded
FP_LIBRARY, FP_COLOR, FP_MOTIFS, FP_DIPOLE_TRIPLETS = get_fp_library()

def detect_fluorescent_protein(sequence: str):
    """
    Detect if a sequence contains a known fluorescent protein.
    Returns dict with 'name', 'color', 'fp_start', 'fp_end' if detected, None otherwise.
    For sequences with multiple FPs, returns info about all detected FPs.
    """
    results = []
    
    # 0. Try alignment-based detection (most robust for full domain)
    try:
        from fpsim.segments import find_fp_domains
        # find_fp_domains returns list of (name, start, end, identity)
        hits = find_fp_domains(sequence, FP_LIBRARY, min_identity=0.35)
        for name, start, end, identity in hits:
            # Inject dipole triplets so frontend can auto-populate
            dt = FP_DIPOLE_TRIPLETS.get(name, [])
            results.append({
                'name': name,
                'color': FP_COLOR.get(name, 'gray'),
                'match_type': 'alignment',
                'fp_start': start,
                'fp_end': end,
                'identity': identity,
                'dipole_triplets': dt
            })
    except Exception as e:
        print(f"Alignment detection failed: {e}")
        # Proceed to fallbacks
    
    # 1. First try exact name match if the sequence is short (the whole sequence is an FP)
    # This is a bit inefficient for long sequences, so we only do it if the sequence is < 300 aa
    if len(sequence) < 300:
        for name, lib_seq in FP_LIBRARY.items():
            if sequence == lib_seq:
                results.append({
                    'name': name,
                    'color': FP_COLOR.get(name, 'gray'),
                    'match_type': 'exact',
                    'fp_start': 1,
                    'fp_end': len(sequence),
                    'dipole_triplets': FP_DIPOLE_TRIPLETS.get(name, [])
                })
    
    # 2. Try motif matching for more robust detection
    for name, motif_list in FP_MOTIFS.items():
        # Clean motif for regex (some might have brackets etc.)
        # The library motifs are usually just short sequences
        for motif in motif_list:
            match = re.search(motif, sequence)
            if match:
                # Found a match, now estimate the start/end of the FP domain
                # Most FPs are ~230-240 aa long. The motif is typically in the middle.
                # For simplicity, we'll use the pre-calculated offsets if available
                # or just return the motif position
                results.append({
                    'name': name,
                    'color': FP_COLOR.get(name, 'gray'),
                    'match_type': 'motif',
                    'fp_start': match.start() + 1,
                    'fp_end': match.end(),
                    'motif_start': match.start() + 1,
                    'motif_end': match.end(),
                    'dipole_triplets': FP_DIPOLE_TRIPLETS.get(name, [])
                })
                break # Only need one motif match per name
            
    # 3. Handle dipole triplets if available
    for name, triplets in FP_DIPOLE_TRIPLETS.items():
        # Check if all triplets match
        matches = []
        normalized_triplets = [] # Store simple strings for frontend
        
        for triplet in triplets:
            if isinstance(triplet, list):
                # Handle cases where one triplet is a list of alternatives
                for alt_triplet in triplet:
                    m = re.search(alt_triplet, sequence)
                    if m:
                        matches.append(m)
                        normalized_triplets.append(alt_triplet) # Use matched alternative
                        break
            else:
                m = re.search(triplet, sequence)
                if m:
                    matches.append(m)
                    normalized_triplets.append(triplet)
        
        if len(matches) == len(triplets):
            # All triplets matched!
            results.append({
                'name': name,
                'color': FP_COLOR.get(name, 'gray'),
                'match_type': 'dipole_triplets',
                'fps': [m.start() + 1 for m in matches],
                'dipole_triplets': normalized_triplets
            })

    if not results:
        return None
        
    # Pick the "best" match (alignment > exact > motif > triplets)
    alignment = [r for r in results if r['match_type'] == 'alignment']
    if alignment: return alignment[0]

    exact = [r for r in results if r['match_type'] == 'exact']
    if exact: return exact[0]
    
    motif = [r for r in results if r['match_type'] == 'motif']
    if motif: return motif[0]
    
    triplets = [r for r in results if r['match_type'] == 'dipole_triplets']
    if triplets: return triplets[0]
    
    return results[0]

def list_fasta_sequences(fasta_path):
    """Read FASTA and return list of sequences with FP info"""
    try:
        from Bio import SeqIO
        sequences = []
        
        with open(fasta_path, 'r') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                sequence_str = str(record.seq)
                
                # Detect fluorescent protein
                fp_info = detect_fluorescent_protein(sequence_str)
                
                seq_data = {
                    'id': record.id,
                    'description': record.description,
                    'length': len(sequence_str),
                    'sequence': sequence_str
                }
                
                if fp_info:
                    seq_data['fp_name'] = fp_info['name']
                    seq_data['fp_color'] = fp_info['color']
                    seq_data['fp_match_type'] = fp_info['match_type']
                    if 'fp_start' in fp_info: seq_data['fp_start'] = fp_info['fp_start']
                    if 'fp_end' in fp_info: seq_data['fp_end'] = fp_info['fp_end']
                
                sequences.append(seq_data)
        return sequences
    except Exception as e:
        print(f"Error reading FASTA: {e}")
        return []
