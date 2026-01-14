"""Segmentation helpers: FP domain detection and pLDDT parsing.

These are extracted from cli.py to reduce its size and improve testability.
Interfaces are kept identical to preserve behavior.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import logging


# ----------------------
# Local alignment helper
# ----------------------
try:  # Biopython may not be present in all environments
    from Bio.Align import PairwiseAligner  # type: ignore
except Exception as e:  # pragma: no cover
    PairwiseAligner = None  # type: ignore


# Setup logger for this module
logger = logging.getLogger(__name__)


def _align_best_window(query: str, templ: str) -> tuple[int, int, float]:
    """Local alignment; return (q_start_1based, q_end_1based, identity)."""
    if PairwiseAligner is None:
        raise RuntimeError("Biopython (PairwiseAligner) is required for alignment")
    al = PairwiseAligner()
    # Similar scoring as pairwise2.localms(2,-1,-5,-1)
    al.mode = "local"
    al.match_score = 2.0
    al.mismatch_score = -1.0
    al.open_gap_score = -5.0
    al.extend_gap_score = -1.0

    aln = al.align(query, templ)
    if not aln:
        return (0, 0, 0.0)
    a0 = aln[0]

    # a0.aligned gives arrays of matched blocks without gaps
    # shape: (nblocks, 2) for query and target
    q_blocks = a0.aligned[0]
    t_blocks = a0.aligned[1]
    if len(q_blocks) == 0:
        return (0, 0, 0.0)

    # Cast to native Python ints to avoid numpy int leakage
    q_start = int(q_blocks[0][0])
    q_end = int(q_blocks[-1][1])

    # Identity over aligned, counting only positions where both are aligned (no gaps)
    matches = 0
    aligned_len = 0
    for (qs, qe), (ts, te) in zip(q_blocks, t_blocks):
        qseg = query[qs:qe]
        tseg = templ[ts:te]
        aligned_len += len(qseg)
        for qc, tc in zip(qseg, tseg):
            if qc == tc:
                matches += 1
    identity = (matches / aligned_len) if aligned_len else 0.0

    # Convert to 1-based inclusive indices
    return (int(q_start + 1), int(q_end), float(identity))


def find_fp_domains(seq: str, lib: Dict[str, str], min_identity: float = 0.35, verbose: bool = False) -> List[Tuple[str, int, int, float]]:
    """
    Detect FP domains in a fusion protein robustly.
    Returns list of (name, start_1based, end_1based, identity).
    """
    # Lazy import to avoid tight coupling
    from .fp_lib import get_fp_library

    try:
        _, _, FP_MOTIFS, _ = get_fp_library()
    except Exception as e:
        raise RuntimeError(f"Failed to load FP library: {e}")
    
    if verbose:
        logger.info("=== FP Library Debug Info ===")
        logger.info(f"Library contains {len(lib)} FP sequences:")
        for name, sequence in lib.items():
            motifs = FP_MOTIFS.get(name, []) if isinstance(FP_MOTIFS, dict) else []
            logger.info(f"  - {name}: {len(sequence)} aa, motifs: {motifs}")
        logger.info(f"Target sequence length: {len(seq)} aa")
        logger.info(f"Minimum identity threshold: {min_identity}")
        logger.info("=============================")
    
    hits: List[Tuple[str, int, int, float]] = []

    # 1) Try quick motif seeds for fast/robust localization
    for name, templ in lib.items():
        motifs = FP_MOTIFS.get(name, []) if isinstance(FP_MOTIFS, dict) else []
        best_seed = (-1, -1.0)  # (index, score)
        for m in motifs:
            i = seq.find(m)
            if i >= 0:
                sc = len(m)
                if sc > best_seed[1]:
                    best_seed = (i, sc)

        if best_seed[0] >= 0:
            # Expand a window around the motif and align
            i = best_seed[0]
            wnd = 300  # large enough to cover full FP (~230 aa) with slack
            start = max(0, i - 50)
            end = min(len(seq), i + wnd)
            qs, qe, idy = _align_best_window(seq[start:end], templ)
            if idy >= min_identity and qs > 0:
                hits.append((name, int(start + qs), int(start + qe), float(idy)))
            continue

        # 2) Fallback: full local alignment if no motif was found
        qs, qe, idy = _align_best_window(seq, templ)
        if idy >= min_identity and qs > 0:
            hits.append((name, int(qs), int(qe), float(idy)))

    # Non-overlapping selection by highest identity then shortest span
    hits.sort(key=lambda x: (-x[3], x[2] - x[1], x[1]))
    covered = set()
    non_overlap: List[Tuple[str, int, int, float]] = []
    for nm, s, e, idy in hits:
        if any(i in covered for i in range(s, e + 1)):
            continue
        for i in range(s, e + 1):
            covered.add(i)
        non_overlap.append((str(nm), int(s), int(e), float(idy)))
    return non_overlap


def parse_plddt_from_pdb(pdb_path, chain_id: str) -> Dict[int, float]:
    """Parse pLDDT from AlphaFold/ColabFold PDB B-factor column for a given chain.

    Returns a mapping: residue index -> average pLDDT across atoms.
    """
    from pathlib import Path as _Path
    from typing import Dict as _Dict

    if not isinstance(pdb_path, _Path):
        pdb_path = _Path(pdb_path)
    plddt_sum: _Dict[int, float] = {}
    plddt_cnt: _Dict[int, int] = {}
    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 66:
                continue
            ch = line[21].strip() if len(line) > 21 else ""
            if chain_id and ch and ch != chain_id:
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            try:
                b = float(line[60:66])
            except ValueError:
                continue
            plddt_sum[resseq] = plddt_sum.get(resseq, 0.0) + b
            plddt_cnt[resseq] = plddt_cnt.get(resseq, 0) + 1
    if not plddt_sum:
        raise ValueError(f"No pLDDT data found for chain '{chain_id}' in {pdb_path}")
    return {r: plddt_sum[r] / plddt_cnt[r] for r in sorted(plddt_sum.keys())}


def segments_from_plddt(
    seq_len: int,
    plddt: Dict[int, float],
    fp_domains: List[Tuple[str, int, int, float]] | None,
    rigid_threshold: float = 70.0,
    min_rb_len: int = 12,
    min_linker_len: int = 10,
) -> List[Dict]:
    """Create core/linker segments from pLDDT profile with FP-bridge restriction.

    - Residues with pLDDT >= rigid_threshold start as core (R), otherwise linker (L)
    - Short rigid islands (< min_rb_len) are converted to linker
    - FP intervals are forced rigid
    - High-pLDDT domains are grouped into rigid bodies
    - Low-pLDDT linkers:
        1. Always eligible if adjacent to FP boundaries
        2. Eligible if the stretch of low pLDDT >= min_linker_len
    """
    lab = ['R'] * seq_len
    for i in range(seq_len):
        score = plddt.get(i + 1, 0.0)
        lab[i] = 'R' if score >= rigid_threshold else 'L'

    def runs_from_labels(labels: List[str]) -> List[Tuple[str, int, int]]:
        runs: List[Tuple[str, int, int]] = []
        if not labels:
            return runs
        cur = labels[0]
        s = 1
        for idx in range(2, len(labels) + 1):
            if labels[idx - 1] != cur:
                runs.append((cur, s, idx - 1))
                cur = labels[idx - 1]
                s = idx
        runs.append((cur, s, len(labels)))
        return runs

    # convert small rigid islands to L
    runs = runs_from_labels(lab)
    for lbl, s, e in runs:
        if lbl == 'R' and (e - s + 1) < min_rb_len:
            for i in range(s - 1, e):
                lab[i] = 'L'

    # Force FP to rigid
    if fp_domains:
        for _nm, s, e, _idy in fp_domains:
            for i in range(max(1, s) - 1, min(seq_len, e)):
                lab[i] = 'R'

    # Eligible linkers: 
    # 1) low-pLDDT adjacent to FP boundaries
    # 2) large stretches of low-pLDDT
    eligible = [False] * seq_len
    
    # Track runs of L to check for length >= min_linker_len
    runs = runs_from_labels(lab)
    for lbl, s, e in runs:
        if lbl == 'L' and (e - s + 1) >= min_linker_len:
            for i in range(s - 1, e):
                eligible[i] = True
    
    # Also include linkers adjacent to FP domains
    if fp_domains:
        for _nm, s, e, _idy in fp_domains:
            i = s - 2
            while i >= 0 and lab[i] == 'L':
                eligible[i] = True
                i -= 1
            i = e
            while i < seq_len and lab[i] == 'L':
                eligible[i] = True
                i += 1

    for i in range(seq_len):
        if lab[i] == 'L' and not eligible[i]:
            lab[i] = 'R'

    merged = runs_from_labels(lab)
    segs: List[Dict] = []
    for lbl, s, e in merged:
        if lbl == 'R':
            segs.append(dict(kind="core", name="core", start=s, end=e))
        else:
            segs.append(dict(kind="linker", name="linker", start=s, end=e))
    return segs
