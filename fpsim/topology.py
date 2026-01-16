"""Topology writer module.

Implements PMI topology writing for single- and multi-chain cases.
Extracted from cli.py to reduce its size and improve testability.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Tuple

from .fp_lib import get_fp_library


def write_top(
    top_path: Path,
    fasta_path: Path,
    pdb_path: Path,
    chain_id: str,
    segs,
    bead_size: int = 10,
    em_res_per_gaussian: int = 10,
    pdb_offset: int = 0,
    pdb_dir: str | None = None,
    fasta_dir: str | None = None,
    gmm_dir: str | None = None,
    molecule_name: str = "Fusion",
    fasta_id: str = "Fusion",
    core_color: str = "gray",
    linker_color: str = "dark gray",
    fp_domains: List[Tuple[str,int,int,float]] | None = None,
):
    # Defaults
    if fasta_dir is None: fasta_dir = "."
    if pdb_dir   is None: pdb_dir   = "."
    if gmm_dir   is None:
        default_gmm = (top_path.parent / "gmm_files").resolve()
        gmm_dir = os.path.relpath(str(default_gmm), start=str(Path.cwd()))

    fasta_fn = str(Path(fasta_path).resolve())
    pdb_fn   = str(Path(pdb_path).resolve())

    # Normalize FP domains
    fp_intervals: List[Tuple[int,int,str]] = []
    if fp_domains:
        for nm, s, e, _idy in fp_domains:
            fp_intervals.append((int(s), int(e), str(nm)))
    fp_intervals.sort(key=lambda x: (x[0], x[1]))

    def _fp_name_if_inside(a: int, b: int) -> str | None:
        for s, e, nm in fp_intervals:
            if a >= s and b <= e:
                return nm
        return None

    # Split a core segment at all FP boundaries it overlaps
    def _explode_core_segment(a: int, b: int) -> List[Tuple[str,int,int,str|None]]:
        if not fp_intervals:
            return [("core", a, b, None)]
        # Collect cut points that fall inside [a,b]
        cuts = {a, b + 1}
        for s, e, _ in fp_intervals:
            if s >= a and s <= b: cuts.add(s)
            if (e + 1) >= a and (e + 1) <= (b + 1): cuts.add(e + 1)
        xs = sorted(cuts)
        pieces: List[Tuple[str,int,int,str|None]] = []
        for i in range(len(xs) - 1):
            s = xs[i]
            e = xs[i+1] - 1
            if s > e:
                continue
            nm = _fp_name_if_inside(s, e)
            if nm is None:
                pieces.append(("core", s, e, None))
            else:
                pieces.append(("fp", s, e, nm))
        return pieces

    # RB policy: RB 1 = protein core; RB>=2 for FP cores & linkers
    protein_rb = 1
    next_rb = 2

    lines: list[str] = []
    lines += [
        "|directories|",
        f"|pdb_dir|{pdb_dir if pdb_dir.endswith('/') else pdb_dir + '/'}|",
        f"|fasta_dir|{fasta_dir if fasta_dir.endswith('/') else fasta_dir + '/'}|",
        f"|gmm_dir|{gmm_dir if gmm_dir.endswith('/') else gmm_dir + '/'}|",
        "",
        "|topology_dictionary|\n"
        "|molecule_name|color|fasta_fn|fasta_id|pdb_fn|chain|residue_range|pdb_offset|"
        "bead_size|em_residues_per_gaussian|rigid_body|super_rigid_body|chain_of_super_rigid_bodies|",
    ]

    # Access FP colors via library
    _, FP_COLOR, _, _ = get_fp_library()
    fp_color_map = {k.lower(): v for k, v in FP_COLOR.items()}

    # Emit a line per atomic piece (core split at FP edges; each linker its own)
    for seg in segs:
        a, b = int(seg["start"]), int(seg["end"])
        if seg["kind"] == "linker":
            rr = f"{a},{b}"
            use_rb = next_rb; next_rb += 1
            line = (
                f"|{molecule_name}|{linker_color}|{fasta_fn}|{fasta_id}|BEADS|{chain_id}|{rr}|{pdb_offset}|"
                f"{bead_size}|{em_res_per_gaussian}|{use_rb}|1||"
            )
            lines.append(line)
        else:
            for kind, s, e, fp_nm in _explode_core_segment(a, b):
                rr = f"{s},{e}"
                if kind == "fp":
                    use_rb = next_rb; next_rb += 1
                    color = fp_color_map.get(fp_nm.lower(), core_color) if fp_nm else core_color
                    line = (
                        f"|{molecule_name}|{color}|{fasta_fn}|{fasta_id}|{pdb_fn}|{chain_id}|{rr}|{pdb_offset}|"
                        f"1|{em_res_per_gaussian}|{use_rb}|1||"
                    )
                else:
                    use_rb = protein_rb
                    line = (
                        f"|{molecule_name}|{core_color}|{fasta_fn}|{fasta_id}|{pdb_fn}|{chain_id}|{rr}|{pdb_offset}|"
                        f"1|{em_res_per_gaussian}|{use_rb}|1||"
                    )
                lines.append(line)

    top_path.write_text("\n".join(lines) + "\n")


def write_top_multi(
    top_path: Path,
    fasta_path: Path,
    pdb_path: Path,
    chain_labels: List[str],
    segs_by_chain: Dict[str, List[Dict]],
    *,
    bead_size: int = 10,
    em_res_per_gaussian: int = 10,
    pdb_offset: int = 0,
    pdb_dir: str | None = None,
    fasta_dir: str | None = None,
    gmm_dir: str | None = None,
    molecule_name: str = "Fusion",
    fasta_id_prefix: str = "Fusion",
    core_color: str = "gray",
    linker_color: str = "dark gray",
    fp_domains_by_chain: Dict[str, List[Tuple[str,int,int,float]]] | None = None,
):
    # Defaults
    if fasta_dir is None: fasta_dir = "."
    if pdb_dir   is None: pdb_dir   = "."
    if gmm_dir   is None:
        default_gmm = (top_path.parent / "gmm_files").resolve()
        gmm_dir = os.path.relpath(str(default_gmm), start=str(Path.cwd()))

    fasta_fn = str(Path(fasta_path).resolve())
    pdb_fn   = str(Path(pdb_path).resolve())

    lines: list[str] = []
    lines += [
        "|directories|",
        f"|pdb_dir|{pdb_dir if pdb_dir.endswith('/') else pdb_dir + '/'}|",
        f"|fasta_dir|{fasta_dir if fasta_dir.endswith('/') else fasta_dir + '/'}|",
        f"|gmm_dir|{gmm_dir if gmm_dir.endswith('/') else gmm_dir + '/'}|",
        "",
        "|topology_dictionary|\n"
        "|molecule_name|color|fasta_fn|fasta_id|pdb_fn|chain|residue_range|pdb_offset|"
        "bead_size|em_residues_per_gaussian|rigid_body|super_rigid_body|chain_of_super_rigid_bodies|",
    ]

    # GLOBAL RB counter across chains:
    # RB 1 is reserved for protein core everywhere; FP/linkers get RB>=2 globally
    protein_rb = 1
    next_rb_global = 2

    # Access FP colors
    _, FP_COLOR, _, _ = get_fp_library()
    fp_color_map = {k.lower(): v for k, v in FP_COLOR.items()}

    for label in chain_labels:
        segs = segs_by_chain.get(label, [])
        fasta_id = f"{fasta_id_prefix}_{label}" if not str(fasta_id_prefix).endswith(f"_{label}") else str(fasta_id_prefix)
        mol_name = f"{molecule_name}_{label}" if not str(molecule_name).endswith(f"_{label}") else str(molecule_name)

        # FP domains for this chain
        fp_intervals: List[Tuple[int,int,str]] = []
        if fp_domains_by_chain and label in fp_domains_by_chain:
            for nm, s, e, _idy in fp_domains_by_chain[label]:
                fp_intervals.append((int(s), int(e), str(nm)))
        fp_intervals.sort(key=lambda x: (x[0], x[1]))

        def _fp_name_if_inside(a: int, b: int) -> str | None:
            for s, e, nm in fp_intervals:
                if a >= s and b <= e:
                    return nm
            return None

        def _explode_core_segment(a: int, b: int) -> List[Tuple[str,int,int,str|None]]:
            if not fp_intervals:
                return [("core", a, b, None)]
            cuts = {a, b + 1}
            for s, e, _ in fp_intervals:
                if a <= s <= b: cuts.add(s)
                if a <= (e + 1) <= (b + 1): cuts.add(e + 1)
            xs = sorted(cuts)
            pieces: List[Tuple[str,int,int,str|None]] = []
            for i in range(len(xs) - 1):
                s0 = xs[i]
                e0 = xs[i+1] - 1
                if s0 > e0:
                    continue
                nm = _fp_name_if_inside(s0, e0)
                if nm is None:
                    pieces.append(("core", s0, e0, None))
                else:
                    pieces.append(("fp", s0, e0, nm))
            return pieces

        # Emit rows; RB 1 for non-FP core, global RBs for FP/linkers
        for seg in segs:
            a, b = int(seg["start"]), int(seg["end"])
            
            # Determine representation (linker=beads, core=rigid)
            # Default to logic based on kind if representation not present
            kind = seg.get("kind", "core")
            rep = seg.get("representation", "bead" if kind == "linker" else "rigid_body")
            
            # Helper to get color
            user_color = seg.get("color")
            
            if rep == "bead": # Treated as linker/flexible
                rr = f"{a},{b}"
                use_rb = next_rb_global; next_rb_global += 1
                # Use user color if provided, else linker default
                seg_color = user_color if user_color else linker_color
                lines.append(
                    f"|{mol_name}|{seg_color}|{fasta_fn}|{fasta_id}|BEADS|{label}|{rr}|{pdb_offset}|"
                    f"{bead_size}|{em_res_per_gaussian}|{use_rb}|1||"
                )
            else: # Rigid Body
                # We still might want to explode it by FPs if it's auto-generated,
                # but if it's an override, we trust the segments.
                # If kind is 'fp', it might be a rigid body with specific color.
                
                # Check if we should explode (legacy logic for auto-segmentation)
                # If we have an override, we assume segments are final.
                # But `_explode_core_segment` is local.
                # Let's assume if "representation" is present, it's an override or explicit set.
                
                is_override = "representation" in seg
                
                if is_override:
                    rr = f"{a},{b}"
                    # Rigid bodies need RBs.
                    # Standard logic: RB 1 for core protein, distinct RBs for FPs.
                    # If user set, we might need new RBs?
                    # The original logic used global RB 1 for all 'core' unless it was FP.
                    # If we let user set "Rigid Body", they might want it to be part of the main rigid body (RB 1) or a new one.
                    # Typically, "Core" implies THE rigid body.
                    
                    if kind == "fp" or user_color:
                         # If it's FP or has custom color, give it a unique RB?
                         # Or just use the color?
                         # PMI topology file format: |rigid_body| column.
                         # If 1, it's the main RB. If distinct, it moves separately.
                         # If user selects "Rigid Body", they likely want it rigid relative to the core?
                         # Actually usually "Rigid Body" means "Part of the main rigid structure" OR "A rigid body".
                         # Let's assume RB 1 for "core" kind without custom behavior, and new RB for others?
                         # For now, let's stick to: if it's "core" kind, use protein_rb. If "fp", distinct.
                         
                         use_rb = next_rb_global; next_rb_global += 1
                         seg_color = user_color if user_color else (fp_color_map.get(seg.get("name","").lower(), core_color))
                         
                         if kind == "core" and not user_color:
                             use_rb = protein_rb
                             seg_color = core_color
                    else:
                         use_rb = protein_rb
                         seg_color = core_color
                         
                    lines.append(
                        f"|{mol_name}|{seg_color}|{fasta_fn}|{fasta_id}|{pdb_fn}|{label}|{rr}|{pdb_offset}|"
                        f"1|{em_res_per_gaussian}|{use_rb}|1||"
                    )
                else:
                    # Legacy fallback
                    if kind == "linker": # Should have been caught by rep
                         pass 
                    else:
                        for sub_kind, s, e, fp_nm in _explode_core_segment(a, b):
                            rr = f"{s},{e}"
                            if sub_kind == "fp":
                                use_rb = next_rb_global; next_rb_global += 1
                                color = fp_color_map.get(fp_nm.lower(), core_color) if fp_nm else core_color
                                lines.append(
                                    f"|{mol_name}|{color}|{fasta_fn}|{fasta_id}|{pdb_fn}|{label}|{rr}|{pdb_offset}|"
                                    f"1|{em_res_per_gaussian}|{use_rb}|1||"
                                )
                            else:
                                # non-FP protein core
                                lines.append(
                                    f"|{mol_name}|{core_color}|{fasta_fn}|{fasta_id}|{pdb_fn}|{label}|{rr}|{pdb_offset}|"
                                    f"1|{em_res_per_gaussian}|{protein_rb}|1||"
                                )

    top_path.write_text("\n".join(lines) + "\n")
