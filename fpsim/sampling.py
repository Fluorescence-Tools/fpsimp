"""Sampling helpers (IMP/PMI).

Implementations extracted from cli.py to reduce its size and improve testability.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import click

from .fp_lib import get_fp_library



class LogFilter:
    """Filter stdout to reduce verbosity of IMP sampling output."""
    def __init__(self, stream, loop_interval: int = 200):
        self.stream = stream
        self.loop_interval = loop_interval
        self.counter = 0
        self.buffer = ""

    def write(self, data):
        self.buffer += data
        if "\n" in self.buffer:
            lines = self.buffer.split("\n")
            # Process all complete lines
            for line in lines[:-1]:
                self._process_line(line + "\n")
            # Keep the remainder
            self.buffer = lines[-1]

    def _process_line(self, line):
        line_lower = line.lower()
        
        # 1. Critical lines (always print)
        if "starting sampling" in line_lower:
             self.stream.write(line)
             self.stream.flush()
             return

        # 2. Drop spam lines entirely
        if "writing coordinates" in line_lower or "energy" in line_lower:
            return

        # 3. Throttled lines (frames)
        if "frame" in line_lower or "score" in line_lower:
            self.counter += 1
            if self.counter % self.loop_interval == 0:
                self.stream.write(line)
                self.stream.flush()
            return
            
        # 4. All other lines
        self.stream.write(line)
        self.stream.flush()

    def flush(self):
        if self.buffer:
            self._process_line(self.buffer)
            self.buffer = ""
        self.stream.flush()


def _read_top_directories(top_path: Path) -> tuple[str | None, str | None, str | None]:
    pdb_dir = fasta_dir = gmm_dir = None
    with open(top_path, "r") as fh:
        lines = [ln.strip() for ln in fh]
    try:
        i = lines.index("|directories|")
    except ValueError:
        return pdb_dir, fasta_dir, gmm_dir
    j = i + 1
    while j < len(lines) and lines[j].startswith("|") and "topology_dictionary" not in lines[j]:
        parts = [p for p in lines[j].split("|") if p]
        if len(parts) >= 2:
            key, val = parts[0], parts[1]
            if key == "pdb_dir":
                pdb_dir = val
            elif key == "fasta_dir":
                fasta_dir = val
            elif key == "gmm_dir":
                gmm_dir = val
        j += 1
    return pdb_dir, fasta_dir, gmm_dir


def _parse_topology_entries(top_path: Path) -> list[dict]:
    """Parse topology_dictionary rows from a PMI .top.dat file into dicts."""
    rows: list[dict] = []
    with open(top_path, "r") as fh:
        lines = [ln.strip() for ln in fh]
    try:
        i = lines.index("|topology_dictionary|")
    except ValueError:
        return rows
    # Skip the column header line immediately following the marker
    j = i + 2
    while j < len(lines) and lines[j].startswith("|"):
        raw = lines[j]
        parts_all = raw.split("|")
        if len(parts_all) >= 3 and parts_all[0] == "" and parts_all[-1] == "":
            parts = parts_all[1:-1]
        else:
            parts = [p for p in parts_all if p is not None]
        if len(parts) < 13:
            j += 1
            continue
        if parts[0] == "molecule_name" or parts[2] == "fasta_fn":
            j += 1
            continue
        rr = parts[6]
        try:
            s, e = rr.split(',')
            s_i, e_i = int(s), int(e)
        except Exception:
            s_i = e_i = 0
        rows.append({
            "molecule_name": parts[0],
            "color": parts[1],
            "fasta_fn": parts[2],
            "fasta_id": parts[3],
            "pdb_fn": parts[4],
            "chain": parts[5],
            "start": s_i,
            "end": e_i,
            "pdb_offset": parts[7],
            "bead_size": parts[8],
            "em_res_per_gaussian": parts[9],
            "rigid_body": parts[10],
        })
        j += 1
    return rows


def run_imp_sampling(
    top_path: Path,
    pdb_dir: Path,
    out_dir: Path,
    steps_per_frame: int = 10,
    frames: int = 100000,
    k_center: float = 0.05,          # weak harmonic tether strength
    center_initial: bool = True,     # center once before sampling
    membrane: bool = False,
    membrane_weight: float = 10.0,
    barrier_radius: float = 800.0,
    membrane_seqs: list[str] | tuple[str, ...] = (),
):
    # Local imports to keep function self-contained
    import IMP
    import IMP.atom, IMP.core, IMP.algebra
    import IMP.pmi.topology
    import IMP.pmi.restraints
    import IMP.pmi.restraints.basic
    import IMP.pmi.restraints.stereochemistry
    import IMP.pmi.macros
    from Bio import SeqIO
    import sys
    
    # Redirect stdout/stderr to sampling.log for web display
    log_file = out_dir / 'sampling.log'
    log_handle = open(log_file, 'w', buffering=1)  # Line buffered
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # Use LogFilter to throttle output
    # target ~100 log entries total (1% progress steps)
    loop_interval = max(1, int(frames / 100))
    filtered_log = LogFilter(log_handle, loop_interval=loop_interval)
    sys.stdout = filtered_log
    sys.stderr = filtered_log
    
    try:
        mdl = IMP.Model()

        # Ensure GMM directory exists (PMI writes there during representation build)
        t_pdb_dir, t_fasta_dir, t_gmm_dir = _read_top_directories(top_path)
        from pathlib import Path as _Path
        if t_gmm_dir:
            gmm_out_dir = _Path(t_gmm_dir)
            if not gmm_out_dir.is_absolute():
                # PMI resolves relative gmm_dir against topology file directory
                gmm_out_dir = (_Path(top_path).parent / gmm_out_dir).resolve()
        else:
            gmm_out_dir = (out_dir / "gmm_files").resolve()
        gmm_out_dir.mkdir(parents=True, exist_ok=True)

        # Prefer passing directories explicitly (new-style)
        abs_pdb_dir = _Path(pdb_dir).resolve()
        reader = IMP.pmi.topology.TopologyReader(
            str(top_path),
            pdb_dir=str(abs_pdb_dir),
            fasta_dir=".",
            gmm_dir=str(gmm_out_dir),
        )

        bs = IMP.pmi.macros.BuildSystem(mdl)
        bs.add_state(reader)
        hier, dof = bs.execute_macro()

        output_objects: List[object] = []

        # Connectivity per molecule
        print("\n" + "="*80)
        print("POTENTIALS AND RESTRAINTS APPLIED:")
        print("="*80)
        
        crs = []
        moldict = bs.get_molecules()[0]
        mols = []
        print("\n1. CONNECTIVITY RESTRAINTS:")
        for molname in moldict:
            for mol in moldict[molname]:
                cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol)
                cr.add_to_model()
                output_objects.append(cr)
                crs.append(cr)
                mols.append(mol)
                print(f"   - Applied to molecule: {molname}")

        # Excluded volume
        print("\n2. EXCLUDED VOLUME POTENTIAL:")
        print(f"   - Type: ExcludedVolumeSphere")
        print(f"   - Applied to {len(mols)} molecule(s)")
        evr = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(included_objects=mols)
        evr.add_to_model()
        output_objects.append(evr)

        # --- Membrane restraint (optional) ---
        if membrane:
            # Classify from topology
            _, FP_COLOR, _, _ = get_fp_library()
            fp_colors = {str(c).lower() for c in FP_COLOR.values()}
            core_inside: list[tuple[int, int, str]] = []
            objs_above: list[tuple[int, int, str]] = []
            for row in _parse_topology_entries(top_path):
                nm = str(row.get("molecule_name", "")).strip()
                s = int(row.get("start", 0))
                e = int(row.get("end", 0))
                if not (nm and s > 0 and e >= s):
                    continue
                pdb_fn = str(row.get("pdb_fn", ""))
                color = str(row.get("color", "")).lower()
                if pdb_fn == "BEADS" or color in fp_colors:
                    objs_above.append((s, e, nm))
                else:
                    core_inside.append((s, e, nm))

            # If user provided membrane sequences, override inside with matches only
            if membrane_seqs:
                print(f"Restraining to membrane sequences: {membrane_seqs}")
                rows = _parse_topology_entries(top_path)
                id_to_names: dict[str, set[str]] = {}
                id_to_fasta: dict[str, str] = {}
                path_to_records: dict[str, dict[str, str]] = {}
                for r in rows:
                    fid = str(r.get("fasta_id", ""))
                    fn = str(r.get("fasta_fn", ""))
                    name = str(r.get("molecule_name", ""))
                    if fid:
                        id_to_names.setdefault(fid, set()).add(name)
                    if fid and fn and fid not in id_to_fasta:
                        if fn not in path_to_records:
                            recs: dict[str, str] = {}
                            for rec in SeqIO.parse(fn, "fasta"):
                                recs[str(rec.id)] = str(rec.seq).replace("\n", "").replace("\r", "")
                            path_to_records[fn] = recs
                        seqmap = path_to_records.get(fn, {})
                        if fid in seqmap:
                            id_to_fasta[fid] = seqmap[fid]
                forced_inside: set[tuple[int, int, str]] = set()
                for q in membrane_seqs:
                    q_clean = str(q).strip().upper()
                    if not q_clean:
                        continue
                    for fid, seq in id_to_fasta.items():
                        s_up = seq.upper()
                        start_idx = 0
                        while True:
                            k = s_up.find(q_clean, start_idx)
                            if k < 0:
                                break
                            a = k + 1
                            b = k + len(q_clean)
                            for nm in id_to_names.get(fid, {"Fusion"}):
                                forced_inside.add((a, b, nm))
                            start_idx = k + 1
                core_inside = list(forced_inside)
            print("\n3. MEMBRANE POTENTIAL:")
            print(f"   - Enabled: True")
            print(f"   - Weight: {membrane_weight}")
            print(f"   - Barrier radius: {barrier_radius} Å")
            if membrane_seqs:
                print(f"   - Membrane sequences specified: {list(membrane_seqs)}")
            print(f"   - Residues restrained inside membrane:")
            for s, e, nm in core_inside:
                print(f"     * {nm}: residues {s}-{e}")
            if objs_above:
                print(f"   - Objects above membrane (beads/FP):")
                for s, e, nm in objs_above:
                    print(f"     * {nm}: residues {s}-{e}")

            if core_inside:
                mr = IMP.pmi.restraints.basic.MembraneRestraint(
                    hier,
                    objects_inside=[(s, e, nm) for (s, e, nm) in core_inside],
                    objects_above=[(s, e, nm) for (s, e, nm) in objs_above] if objs_above else None,
                    weight=float(membrane_weight),
                )
                mr.create_membrane_density()
                mr.add_to_model()
                eb = IMP.pmi.restraints.basic.ExternalBarrier(hierarchies=hier, radius=barrier_radius)
                eb.add_to_model()
        else:
            print("\n3. MEMBRANE POTENTIAL:")
            print("   - Enabled: False")

        print("\n4. BEAD MODEL INFORMATION:")
        # Print bead information from topology
        for row in _parse_topology_entries(top_path):
            nm = str(row.get("molecule_name", ""))
            s = int(row.get("start", 0))
            e = int(row.get("end", 0))
            bead_size = row.get("bead_size", "N/A")
            pdb_fn = str(row.get("pdb_fn", ""))
            if s > 0 and e >= s:
                if pdb_fn == "BEADS":
                    print(f"   - {nm} (res {s}-{e}): Bead representation, bead_size={bead_size}")
                else:
                    print(f"   - {nm} (res {s}-{e}): Atomic/coarse-grained from PDB, bead_size={bead_size}")
        
        print("\n" + "="*80)
        print("STARTING SAMPLING...")
        print("="*80 + "\n")
        
        # Setup periodic progress reporting
        import time
        last_progress_time = time.time()
        progress_interval = 30  # Report progress every 30 seconds
        frame_count = 0
        
        # Movers & sampling
        IMP.pmi.tools.shuffle_configuration(hier)

        # Quickly move all flexible beads into place
        dof.optimize_flexible_beads(100)

        all_movers = dof.get_movers()

        # Backwards compatibility: older IMP/PMI used ReplicaExchange0
        RexClass = getattr(IMP.pmi.macros, "ReplicaExchange", None)
        if RexClass is None:
            RexClass = getattr(IMP.pmi.macros, "ReplicaExchange0", None)
        if RexClass is None:
            raise click.ClickException("Neither IMP.pmi.macros.ReplicaExchange nor ReplicaExchange0 found. Update IMP/PMI.")

        # Create a custom sampling function with reduced output
        def limited_output_sampling():
            nonlocal last_progress_time, frame_count
            
            print(f"Starting sampling: {frames} frames, {steps_per_frame} steps per frame")
            print("Progress will be reported every 30 seconds...")
            
            # Run the sampling with periodic progress updates
            rex = RexClass(
                mdl,
                root_hier=hier,
                monte_carlo_sample_objects=all_movers,
                global_output_directory=str(out_dir / "out_linkers"),
                monte_carlo_steps=steps_per_frame,
                number_of_frames=frames,
                monte_carlo_temperature=1.0,
                number_of_best_scoring_models=0,
            )
            
            # Execute with reduced output
            start_time = time.time()
            rex.execute_macro()
            
            print(f"Sampling completed in {time.time() - start_time:.1f} seconds")
        
        limited_output_sampling()

    finally:
        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_handle.close()
