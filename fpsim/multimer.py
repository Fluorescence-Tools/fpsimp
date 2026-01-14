"""Multimer processing utilities for fpsim.

Handles chain processing and segmentation for multimer sequences.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json
import string as _string
from .segments import find_fp_domains, parse_plddt_from_pdb, segments_from_plddt
from .fp_lib import get_fp_library


def process_multimer_chains(
    seq_raw: str,
    seq_id_raw: str,
    af_pdb: Path,
    fp_library: Dict[str, str],
    plddt_rigid: float = 70.0,
    min_rb_len: int = 12,
    min_linker_len: int = 10,
    verbose_func=None,
    verbose: bool = False
) -> Tuple[List[str], Dict[str, List[Dict]], Dict[str, List[Tuple[str, int, int, float]]], Dict[str, Dict]]:
    """Process multimer sequence into per-chain segments and metadata.
    
    Returns:
        - labels: Chain labels (A, B, C, ...)
        - segs_by_chain: Segments for each chain
        - fp_domains_by_chain: FP domains for each chain
        - chains_meta: Metadata for each chain
    """
    def vprint(msg: str):
        if verbose_func:
            verbose_func(msg)
    
    parts = seq_raw.split(":")
    labels = [_string.ascii_uppercase[i] if i < 26 else str(i + 1) for i in range(len(parts))]
    vprint(f"Detected multimer: {len(parts)} chains -> {', '.join(labels)}")

    # Build per-chain segments
    segs_by_chain: Dict[str, List[Dict]] = {}
    fp_domains_by_chain: Dict[str, List[Tuple[str, int, int, float]]] = {}
    chains_meta: Dict[str, Dict] = {}
    
    for i, label in enumerate(labels):
        seq_for_seg = parts[i]
        vprint(f"[{label}] sequence length: {len(seq_for_seg)}")
        
        domains = find_fp_domains(seq_for_seg, fp_library, min_identity=0.40, verbose=verbose)
        vprint(f"[{label}] FP domains detected: {[(n,s,e,round(idy,2)) for (n,s,e,idy) in domains]}")
        
        plddt = parse_plddt_from_pdb(af_pdb, label)
        vprint(f"[{label}] pLDDT residues parsed: {len(plddt)}")
        
        segs = segments_from_plddt(
            len(seq_for_seg), plddt, domains, 
            rigid_threshold=plddt_rigid, min_rb_len=min_rb_len, min_linker_len=min_linker_len
        )
        vprint(f"[{label}] segments: {len(segs)} -> {[(seg['kind'], seg['start'], seg['end']) for seg in segs]}")
        
        segs_by_chain[label] = segs
        _, colors, _, _ = get_fp_library()
        fp_domains_by_chain[label] = domains
        chains_meta[label] = {
            "sequence_id": f"{seq_id_raw}_{label}",
            "sequence": seq_for_seg,
            "sequence_len": int(len(seq_for_seg)),
            "chain": label,
            "fp_domains": [{"name": str(n), "start": int(s), "end": int(e), "identity": float(idy), "color": colors.get(n, "green")} for (n, s, e, idy) in domains],
            "segments": [{**seg, "start": int(seg["start"]), "end": int(seg["end"])} for seg in segs],
        }

    return labels, segs_by_chain, fp_domains_by_chain, chains_meta


def create_multimer_metadata(
    seq_id_raw: str,
    af_pdb: Path,
    plddt_rigid: float,
    min_rb_len: int,
    bead_res_per_bead: int,
    labels: List[str],
    chains_meta: Dict[str, Dict]
) -> Dict:
    """Create metadata dictionary for multimer segmentation."""
    return {
        "sequence_id_prefix": seq_id_raw,
        "af_pdb": str(af_pdb),
        "plddt_rigid": float(plddt_rigid),
        "min_rb_len": int(min_rb_len),
        "bead_res_per_bead": int(bead_res_per_bead),
        "chain_labels": labels,
        "chains": chains_meta,
    }
