"""FP library helper module.

Implements FP library data loading and management.
Extracted from cli.py to improve modularity.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Global FP library data
FP_LIBRARY: Dict[str, str] = {}
FP_COLOR: Dict[str, str] = {}
FP_MOTIFS: Dict[str, List[str]] = {}
FP_DIPOLE_TRIPLETS: Dict[str, List[str]] = {}

def load_fp_library_json(fp_json: Path) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]], Dict[str, List[str]]]:
    """Load FP library from JSON file."""
    seqs: Dict[str, str] = {}
    colors: Dict[str, str] = {}
    motifs: Dict[str, List[str]] = {}
    dipole_triplets: Dict[str, List[str]] = {}
    with open(fp_json, "r") as fh:
        data = json.load(fh)
    for name, info in data.items():
        if "sequence" in info:
            seqs[name] = str(info["sequence"])
        if "color" in info:
            colors[name] = str(info["color"])
        if "motifs" in info:
            mlist = info["motifs"]
            if isinstance(mlist, list):
                motifs[name] = [str(x) for x in mlist]
        if "dipole_triplets" in info:
            triplets = info["dipole_triplets"]
            if isinstance(triplets, list):
                dipole_triplets[name] = [str(x) for x in triplets]
    return seqs, colors, motifs, dipole_triplets

def init_default_fp_library():
    """Initialize FP_LIBRARY and FP_COLOR from packaged json if available; else fallback to built-ins."""
    global FP_LIBRARY, FP_COLOR, FP_MOTIFS, FP_DIPOLE_TRIPLETS
    packaged = Path(__file__).parent / "fp_library.json"
    if packaged.exists():
        try:
            FP_LIBRARY, FP_COLOR, FP_MOTIFS, FP_DIPOLE_TRIPLETS = load_fp_library_json(packaged)
            return
        except Exception as e:
            raise RuntimeError(f"Failed to load FP library from {packaged}: {e}")
    else:
        raise FileNotFoundError(f"FP library file not found at {packaged}")

def get_fp_library() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]], Dict[str, List[str]]]:
    """Get the FP library data (sequences, colors, motifs, dipole_triplets)."""
    if not FP_LIBRARY:  # Initialize if not already done
        init_default_fp_library()
    return FP_LIBRARY, FP_COLOR, FP_MOTIFS, FP_DIPOLE_TRIPLETS
