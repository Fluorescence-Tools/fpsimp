"""
Pipeline orchestration for fpsim end-to-end workflow.

This module contains the core business logic extracted from the CLI,
separating concerns between argument parsing and workflow execution.
"""
from __future__ import annotations
import json
import string as _string
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .segments import find_fp_domains, parse_plddt_from_pdb, segments_from_plddt
from .fp_lib import get_fp_library
from .topology import write_top, write_top_multi
from .sampling import run_imp_sampling
from .utils import (
    read_single_fasta, read_fasta_with_selection, sanitize_fasta_for_pmi, existing_ranked_pdb, 
    has_segments_and_top, has_sampling_outputs, run_colabfold as run_colabfold_exec
)
from .multimer import process_multimer_chains, create_multimer_metadata
from .colabfold_utils import build_colabfold_args
from .measure import (
    get_particles_by_segment, xyz_of, match_particles_to_sites, 
    compute_statistics, write_measurements_tsv, create_distance_kappa_plot,
    parse_site, setup_rmf_hierarchy, measure_frames, measure_trajectory
)


class PipelineConfig:
    """Configuration container for pipeline execution."""
    
    def __init__(
        self,
        fasta: Path,
        out_dir: Path,
        sequence_id: Optional[str] = None,
        chain: str = "A",
        af_pdb: Optional[Path] = None,
        run_colabfold: bool = True,
        colabfold_external: bool = False,  # ColabFold handled externally (webapp)
        use_structure_plddt: bool = False,  # Override to use pLDDT from provided structure
        colabfold_args: str = "--num-models 1",
        low_spec: bool = False,
        mem_frac: float = 0.5,
        relax: Optional[bool] = None,
        num_models: Optional[int] = None,
        num_ensemble: Optional[int] = None,
        num_recycle: Optional[int] = None,
        model_type: Optional[str] = None,
        max_seq: Optional[int] = None,
        max_extra_seq: Optional[int] = None,
        msa_mode: Optional[str] = None,
        pair_mode: Optional[str] = None,
        no_templates: bool = False,
        use_gpu_relax: Optional[bool] = None,
        gpu: Optional[int] = None,
        plddt_rigid: float = 70.0,
        min_rb_len: int = 12,
        min_linker_len: int = 10,
        model_disordered_as_beads: bool = False,
        bead_res_per_bead: int = 10,
        frames: int = 100000,
        steps_per_frame: int = 10,
        membrane: bool = False,
        membrane_weight: float = 10.0,
        barrier_radius: float = 100.0,
        membrane_seq: Tuple[str, ...] = (),
        reuse: bool = True,
        force: bool = False,
        verbose: bool = False,
        fp_lib: Optional[Path] = None,
        fp_library: Optional[Dict[str, str]] = None,
        fp_color: Optional[Dict[str, str]] = None,
        vprint_func = None,
        # Measurement parameters
        measure: bool = False,
        sites: Tuple[str, ...] = (),
        measure_out_tsv: Optional[Path] = None,
        measure_plot: bool = True,
        measure_plot_out: Optional[Path] = None,
        measure_frame_start: int = 0,
        measure_max_frames: Optional[int] = None,
        status_callback = None,
        segments_override: Optional[Dict[str, List[Dict]]] = None
    ):
        self.fasta = fasta
        self.out_dir = out_dir
        self.sequence_id = sequence_id
        self.chain = chain
        self.af_pdb = af_pdb
        self.run_colabfold = run_colabfold
        self.use_structure_plddt = use_structure_plddt
        self.colabfold_args = colabfold_args
        self.low_spec = low_spec
        self.mem_frac = mem_frac
        self.relax = relax
        self.num_models = num_models
        self.num_ensemble = num_ensemble
        self.num_recycle = num_recycle
        self.model_type = model_type
        self.max_seq = max_seq
        self.max_extra_seq = max_extra_seq
        self.msa_mode = msa_mode
        self.pair_mode = pair_mode
        self.no_templates = no_templates
        self.use_gpu_relax = use_gpu_relax
        self.gpu = gpu
        self.plddt_rigid = plddt_rigid
        self.min_rb_len = min_rb_len
        # If model_disordered_as_beads is False, effectively disable length-based linkers
        self.min_linker_len = min_linker_len if model_disordered_as_beads else 1000000
        self.bead_res_per_bead = bead_res_per_bead
        self.frames = frames
        self.steps_per_frame = steps_per_frame
        self.membrane = membrane
        self.membrane_weight = membrane_weight
        self.barrier_radius = barrier_radius
        self.membrane_seq = membrane_seq
        self.reuse = reuse
        self.force = force
        self.verbose = verbose
        self.fp_lib = fp_lib
        self.fp_library = fp_library or {}
        self.fp_color = fp_color or {}
        self.vprint = vprint_func or (lambda x: None)
        # Measurement parameters
        self.measure = measure
        self.sites = sites
        self.measure_out_tsv = measure_out_tsv
        self.measure_plot = measure_plot
        self.measure_plot_out = measure_plot_out
        self.measure_frame_start = measure_frame_start
        self.measure_max_frames = measure_max_frames
        self.status_callback = status_callback or (lambda x: None)
        self.segments_override = segments_override


def ensure_af_pdb(config: PipelineConfig) -> Path:
    """Ensure we have an AlphaFold structure, either existing or newly predicted."""
    if config.af_pdb is None:
        if config.colabfold_external:
            # Create a dummy structure file for webapp context
            cf_out = config.out_dir / "colabfold"
            cf_out.mkdir(parents=True, exist_ok=True)
            dummy_pdb = cf_out / "dummy_webapp_structure.pdb"
            
            # Create a minimal PDB file with proper format
            if not dummy_pdb.exists():
                with open(dummy_pdb, 'w') as f:
                    f.write("ATOM      1  N   MET A   1      -0.000   0.000   0.000  1.00 50.00           N\n")
                    f.write("TER\n")
                    f.write("END\n")
            
            config.vprint(f"[webapp] Using dummy structure for externally handled ColabFold: {dummy_pdb}")
            return dummy_pdb
        else:
            if not config.run_colabfold:
                raise ValueError("Provide --af-structure or set --run-colabfold to predict one.")
            
            cf_out = config.out_dir / "af_models"
            
            # Reuse existing structure if allowed
            if config.reuse and not config.force:
                existing = existing_ranked_pdb(cf_out)
                if existing is not None:
                    config.vprint(f"[reuse] Using existing structure: {existing}")
                    return existing
            
            # Compose args from explicit flags
            combined_args = build_colabfold_args(
                config.colabfold_args, config.num_models, config.num_ensemble, 
                config.num_recycle, config.model_type, config.max_seq, 
                config.max_extra_seq, config.msa_mode, config.pair_mode, 
                config.no_templates, config.use_gpu_relax
            )
            
            try:
                return run_colabfold_exec(config.fasta, cf_out, combined_args, config.gpu)
            except Exception as e:
                raise RuntimeError(f"ColabFold failed: {e}")
    
    elif config.af_pdb.is_dir():
        cf_out = config.af_pdb
        # Reuse existing structure if allowed
        if config.run_colabfold: # Only try to reuse if colabfold is meant to run
            existing = existing_ranked_pdb(cf_out)
            if existing:
                config.vprint(f"[reuse] Using existing structure: {existing}")
                return existing
        
        # Compose args from explicit flags
        combined_args = build_colabfold_args(
            config.colabfold_args, config.num_models, config.num_ensemble, 
            config.num_recycle, config.model_type, config.max_seq, 
            config.max_extra_seq, config.msa_mode, config.pair_mode, 
            config.no_templates, config.use_gpu_relax
        )
        
        try:
            return run_colabfold_exec(config.fasta, cf_out, combined_args, config.gpu)
        except Exception as e:
            raise RuntimeError(f"ColabFold failed: {e}")
    
    return config.af_pdb


def process_multimer_workflow(config: PipelineConfig, af_pdb: Path, seq_id_raw: str, seq_raw: str) -> Path:
    """Process multimer workflow: segmentation, topology, and sampling."""
    # Check for reuse early
    if config.reuse and not config.force and has_segments_and_top(config.out_dir):
        config.vprint("[reuse] segments.json and fusion.top.dat found. Skipping segmentation and topology.")
        top_path = config.out_dir / "fusion.top.dat"
        pdb_dir = af_pdb.parent
        
        if config.reuse and has_sampling_outputs(config.out_dir):
            config.vprint("[reuse] Sampling outputs found. Skipping IMP sampling.")
            return top_path
        
        run_imp_sampling(
            top_path, pdb_dir, config.out_dir, 
            steps_per_frame=config.steps_per_frame, frames=config.frames,
            membrane=config.membrane, membrane_weight=config.membrane_weight, 
            barrier_radius=config.barrier_radius, membrane_seqs=config.membrane_seq
        )
        return top_path
    
    # Process multimer chains
    if config.segments_override:
        config.vprint("[override] Using provided segmentation override.")
        segs_by_chain = config.segments_override
        # Logic to reconstitute labels/fp_domains, and crucially SEQUENCE info
        # We need `labels` for the next steps
        labels = sorted(list(segs_by_chain.keys()))
        sequences = seq_raw.split(':')
        
        chains_meta = {}
        fp_domains_by_chain = {}
        
        for idx, label in enumerate(labels):
            # Assumes generic chain labeling A, B, C matching sequence order
            seq = sequences[idx] if idx < len(sequences) else ""
            segments = segs_by_chain[label]
            
            # Reconstruct fp_domains from segments if kind == 'fp'
            fp_domains_dicts = []
            fp_domains_tuples = []
            
            for s in segments:
                if s.get('kind') == 'fp':
                    name = s.get('name', 'FP')
                    start = s['start']
                    end = s['end']
                    identity = s.get('identity', 1.0)
                    color = s.get('color', 'green')
                    
                    fp_domains_dicts.append({
                        "name": name,
                        "start": start,
                        "end": end,
                        "identity": identity,
                        "color": color
                    })
                    fp_domains_tuples.append((name, start, end, identity))
            
            fp_domains_by_chain[label] = fp_domains_tuples
            
            chains_meta[label] = {
                "sequence": seq,
                "sequence_len": len(seq),
                "segments": segments,
                "fp_domains": fp_domains_dicts
            }
    else:
        labels, segs_by_chain, fp_domains_by_chain, chains_meta = process_multimer_chains(
            seq_raw, seq_id_raw, af_pdb, config.fp_library, 
            config.plddt_rigid, config.min_rb_len, config.min_linker_len, config.vprint, config.verbose
        )
    
    # Create metadata
    if config.segments_override:
        meta = {
           "sequence_id": seq_id_raw,
           "af_pdb": str(af_pdb),
           "plddt_rigid": config.plddt_rigid,
           "chains": chains_meta,
           "chain_labels": labels,
           "override": True
        }
    else:
        meta = create_multimer_metadata(
            seq_id_raw, af_pdb, config.plddt_rigid, config.min_rb_len, 
            config.bead_res_per_bead, labels, chains_meta
        )
    (config.out_dir / "segments.json").write_text(json.dumps(meta, indent=2))
    
    # Sanitize FASTA and write multi-topology
    sani_fasta = config.out_dir / "sanitized.fasta"
    ids, nrep = sanitize_fasta_for_pmi(config.fasta, sani_fasta)
    if nrep:
        config.vprint(f"[warning] Replaced {nrep} non-standard residues with 'X' in sanitized FASTA for PMI")
    
    config.vprint(f"Sanitized FASTA written: {sani_fasta} with records: {', '.join(ids)}")
    
    top_path = config.out_dir / "fusion.top.dat"
    write_top_multi(
        top_path, sani_fasta, af_pdb, labels, segs_by_chain, 
        bead_size=config.bead_res_per_bead, fasta_id_prefix=seq_id_raw, 
        fp_domains_by_chain=fp_domains_by_chain
    )
    
    # Run sampling
    pdb_dir = af_pdb.parent
    if config.reuse and not config.force and has_sampling_outputs(config.out_dir):
        config.vprint("[reuse] Sampling outputs found. Skipping IMP sampling.")
    else:
        config.status_callback("sampling")
        run_imp_sampling(
            top_path, pdb_dir, config.out_dir, 
            steps_per_frame=config.steps_per_frame, frames=config.frames,
            membrane=config.membrane, membrane_weight=config.membrane_weight, 
            barrier_radius=config.barrier_radius, membrane_seqs=config.membrane_seq
        )
    
    # Generate sampling script for users
    generate_sampling_script(
        top_path, config.out_dir, membrane=config.membrane,
        membrane_weight=config.membrane_weight, barrier_radius=config.barrier_radius,
        membrane_seqs=config.membrane_seq
    )
    config.vprint(f"Generated sampling script: {config.out_dir / 'sampling_script.py'}")
    
    return top_path


def generate_sampling_script(top_path: Path, out_dir: Path, membrane: bool = False, 
                           membrane_weight: float = 1.0, barrier_radius: float = 1000.0,
                           membrane_seqs: list[str] = None) -> Path:
    """Generate a stripped-down sampling script for users to run manually."""
    
    script_content = f'''#!/usr/bin/env python3
"""
Stripped-down IMP sampling script for FPSIMP results.

Generated from topology: {top_path.name}
To run: python sampling_script.py
"""

import sys
from pathlib import Path

# Add IMP to Python path (adjust if needed)
sys.path.insert(0, "/opt/conda/envs/fpsimp/lib/python3.11/site-packages")

try:
    import IMP
    import IMP.atom, IMP.core, IMP.algebra
    import IMP.pmi.topology
    import IMP.pmi.restraints
    import IMP.pmi.restraints.basic
    import IMP.pmi.restraints.stereochemistry
    import IMP.pmi.macros
except ImportError as e:
    print(f"Error importing IMP modules: {{e}}")
    print("Please ensure IMP is properly installed and accessible.")
    sys.exit(1)

def main():
    """Main sampling function."""
    # Paths
    top_path = Path("{top_path}")
    out_dir = Path("{out_dir}")
    
    if not top_path.exists():
        print(f"Topology file not found: {{top_path}}")
        sys.exit(1)
    
    print("Starting IMP sampling...")
    print(f"Topology file: {{top_path}}")
    print(f"Output directory: {{out_dir}}")
    
    # Create model
    mdl = IMP.Model()
    
    # Load topology
    print("Loading topology...")
    reader = IMP.pmi.topology.TopologyReader(
        str(top_path),
        pdb_dir=str(top_path.parent),
        fasta_dir=".",
        gmm_dir=str(out_dir / "gmm_files"),
    )
    
    # Build system
    print("Building system...")
    bs = IMP.pmi.macros.BuildSystem(mdl)
    bs.add_state(reader)
    hier, dof = bs.execute_macro()
    
    # Get molecules
    mols = [hier.get_child(i) for i in range(hier.get_number_of_children())]
    
    # Add restraints
    print("Adding restraints...")
    
    # 1. Connectivity restraints
    print("   - Connectivity restraints")
    for molname in set([mol.get_name() for mol in mols]):
        mol_instances = [mol for mol in mols if mol.get_name() == molname]
        if mol_instances:
            cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol_instances[0])
            cr.add_to_model()
    
    # 2. Excluded volume
    print("   - Excluded volume")
    evr = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(included_objects=mols)
    evr.add_to_model()
    
    # 3. Membrane restraints (if enabled)
    membrane_config = {membrane}
    if membrane_config:
        print("   - Membrane restraints")
        # Parse topology for membrane regions
        membrane_seqs_config = {membrane_seqs or []}
        
        # Add membrane restraint
        mr = IMP.pmi.restraints.basic.MembraneRestraint(hier)
        mr.create_membrane_density()
        mr.add_to_model()
        
        # Add external barrier
        eb = IMP.pmi.restraints.basic.ExternalBarrier(hierarchies=hier, radius={barrier_radius})
        eb.add_to_model()
    
    # Setup sampling
    print("Setting up sampling...")
    IMP.pmi.tools.shuffle_configuration(hier)
    dof.optimize_flexible_beads(100)
    
    all_movers = dof.get_movers()
    
    # Replica exchange
    print("Starting replica exchange sampling...")
    RexClass = getattr(IMP.pmi.macros, "ReplicaExchange", None)
    if RexClass is None:
        RexClass = getattr(IMP.pmi.macros, "ReplicaExchange0", None)
    
    if RexClass is None:
        print("Error: Neither ReplicaExchange nor ReplicaExchange0 found")
        sys.exit(1)
    
    rex = RexClass(
        mdl,
        root_hier=hier,
        monte_carlo_sample_objects=all_movers,
        global_output_directory=str(out_dir / "out_linkers"),
        monte_carlo_steps=1000,  # Adjust as needed
        number_of_frames=1000,   # Adjust as needed
        monte_carlo_temperature=1.0,
        number_of_best_scoring_models=5,
    )
    
    print("Running sampling...")
    rex.execute_macro()
    
    print("Sampling completed!")
    print(f"Results saved to: {{out_dir / 'out_linkers'}}")

if __name__ == "__main__":
    main()
'''
    
    script_path = out_dir / "sampling_script.py"
    script_path.write_text(script_content)
    return script_path


def process_single_chain_workflow(config: PipelineConfig, af_pdb: Path, seq_id_raw: str, seq_raw: str) -> Path:
    """Process single-chain workflow: segmentation, topology, and sampling."""
    # Check for reuse early
    if config.reuse and not config.force and has_segments_and_top(config.out_dir):
        config.vprint("[reuse] segments.json and fusion.top.dat found. Skipping segmentation and topology.")
        top_path = config.out_dir / "fusion.top.dat"
        pdb_dir = af_pdb.parent
        
        if config.reuse and has_sampling_outputs(config.out_dir):
            config.vprint("[reuse] Sampling outputs found. Skipping IMP sampling.")
            return top_path
        
        run_imp_sampling(
            top_path, pdb_dir, config.out_dir, 
            steps_per_frame=config.steps_per_frame, frames=config.frames,
            membrane=config.membrane, membrane_weight=config.membrane_weight, 
            barrier_radius=config.barrier_radius, membrane_seqs=config.membrane_seq
        )
        return top_path
    
    # Single-chain processing
    seq_id = seq_id_raw
    seq_for_seg = seq_raw
    
    
    # Check for segments override
    # Try to find override for current chain
    override_segs = None
    if config.segments_override:
        if config.chain in config.segments_override:
            override_segs = config.segments_override[config.chain]
        elif len(config.segments_override) == 1:
            # Fallback for single-chain jobs where key might differ
            override_segs = list(config.segments_override.values())[0]
            config.vprint(f"[override] Using provided segments (inferred chain match)")

    if override_segs:
        config.vprint(f"[override] Using provided segmentation override for chain {config.chain}")
        segs = override_segs
        
        # Reconstruct FP domains from segments
        domains = []
        for s in segs:
            if s.get('kind') == 'fp':
                domains.append((
                    s.get('name', 'FP'), 
                    int(s['start']), 
                    int(s['end']), 
                    s.get('identity', 1.0)
                ))
    else:
        # Find FP domains and create segments
        domains = find_fp_domains(seq_for_seg, config.fp_library, min_identity=0.40, verbose=config.verbose)
        config.vprint(f"[run {config.chain}] FP domains: {[(n,s,e,round(idy,2)) for (n,s,e,idy) in domains]}")
        
        plddt = parse_plddt_from_pdb(af_pdb, config.chain)
        config.vprint(f"[run {config.chain}] pLDDT residues parsed: {len(plddt)}")
        
        sorted_residues = sorted(plddt.keys())
        segs = segments_from_plddt(
            len(seq_for_seg), plddt, domains,
            residue_numbers=sorted_residues, 
            rigid_threshold=config.plddt_rigid, min_rb_len=config.min_rb_len,
            min_linker_len=config.min_linker_len
        )
        config.vprint(f"[run {config.chain}] segments: {len(segs)} -> {[(seg['kind'], seg['start'], seg['end']) for seg in segs]}")
    
    # Create metadata
    _, colors, _, _ = get_fp_library()
    segs_json = [{**seg, "start": int(seg["start"]), "end": int(seg["end"])} for seg in segs]
    meta = {
        "sequence_id": seq_id,
        "sequence": seq_for_seg,
        "sequence_len": len(seq_for_seg),
        "af_pdb": str(af_pdb),
        "chain": str(config.chain),
        "plddt_rigid": float(config.plddt_rigid),
        "min_rb_len": int(config.min_rb_len),
        "bead_res_per_bead": int(config.bead_res_per_bead),
        "fp_domains": [{"name": str(n), "start": int(s), "end": int(e), "identity": float(idy), "color": colors.get(n, "green")} for (n, s, e, idy) in domains],
        "segments": segs_json,
    }
    (config.out_dir / "segments.json").write_text(json.dumps(meta, indent=2))
    
    # Sanitize FASTA and write topology
    sani_fasta = config.out_dir / "sanitized.fasta"
    ids, nrep = sanitize_fasta_for_pmi(config.fasta, sani_fasta)
    if nrep:
        config.vprint(f"[warning] Replaced {nrep} non-standard residues with 'X' in sanitized FASTA for PMI")
    
    config.vprint(f"Sanitized FASTA written: {sani_fasta} with records: {', '.join(ids)}")
    
    top_path = config.out_dir / "fusion.top.dat"
    write_top(
        top_path, sani_fasta, af_pdb, config.chain, segs, 
        bead_size=config.bead_res_per_bead, fasta_id=seq_id, fp_domains=domains
    )
    
    # Run sampling
    pdb_dir = af_pdb.parent
    config.status_callback("sampling")
    run_imp_sampling(
        top_path, pdb_dir, config.out_dir, 
        steps_per_frame=config.steps_per_frame, frames=config.frames,
        membrane=config.membrane, membrane_weight=config.membrane_weight, 
        barrier_radius=config.barrier_radius, membrane_seqs=config.membrane_seq
    )
    
    # Generate sampling script for users
    generate_sampling_script(
        top_path, config.out_dir, membrane=config.membrane,
        membrane_weight=config.membrane_weight, barrier_radius=config.barrier_radius,
        membrane_seqs=config.membrane_seq
    )
    config.vprint(f"Generated sampling script: {config.out_dir / 'sampling_script.py'}")
    
    return top_path


def run_measurement_step(config: PipelineConfig, out_dir: Path) -> Optional[Dict[str, any]]:
    """Execute measurement step if enabled."""
    if not config.measure:
        return None
    
    if not config.sites:
        config.vprint("[measure] No sites specified, skipping measurement")
        return None
    
    if len(config.sites) not in (2, 4):
        raise ValueError(f"Provide either 2 sites (distance) or 4 sites (distance + kappa). Got {len(config.sites)} sites.")
    
    # Find RMF files in sampling output
    sampling_dir = out_dir / "out_linkers"
    rmf_dir = sampling_dir / "rmfs"
    
    rmf_files = []
    if rmf_dir.exists():
        rmf_files = list(rmf_dir.glob("*.rmf3")) + list(rmf_dir.glob("*.rmf"))
    
    if not rmf_files:
        # Fallback to top-level if no rmfs folder (for non-standard output)
        rmf_files = list(sampling_dir.glob("*.rmf3")) + list(sampling_dir.glob("*.rmf"))
    
    if not rmf_files:
        config.vprint("[measure] No RMF files found in sampling output, skipping measurement")
        return None
    
    # Sort RMF files numerically if possible
    def _rmf_sort_key(p: Path):
        stem = p.stem
        try:
            return int(stem)
        except ValueError:
            return stem
            
    rmf_files.sort(key=_rmf_sort_key)
    config.vprint(f"[measure] Using {len(rmf_files)} RMF files: {rmf_files[0].name} ... {rmf_files[-1].name}")
    
    # Parse sites
    try:
        parsed_sites = [parse_site(site) for site in config.sites]
        config.vprint(f"[measure] Parsed sites: {parsed_sites}")
    except Exception as e:
        raise ValueError(f"Failed to parse sites: {e}")
    
    # Measure trajectory
    try:
        rows = measure_trajectory(rmf_files, parsed_sites, config.verbose)
        config.vprint(f"[measure] Measured {len(rows)} frames total")
    except Exception as e:
        raise RuntimeError(f"Failed to measure trajectory: {e}")
    
    # Compute statistics
    stats = compute_statistics(rows)
    config.vprint(f"[measure] Statistics computed: distance_mean={stats.get('distance_mean', 'N/A'):.2f}")
    
    # Write TSV if requested
    tsv_path = None
    if config.measure_out_tsv:
        tsv_path = config.measure_out_tsv
    else:
        tsv_path = out_dir / "measurements.tsv"
    
    write_measurements_tsv(rows, tsv_path)
    config.vprint(f"[measure] Results written to: {tsv_path}")
    
    # Generate plot if enabled and we have kappa data
    plot_path = None
    if config.measure_plot and len(parsed_sites) == 4:
        plot_path = config.measure_plot_out or (tsv_path.with_suffix(".png"))
        create_distance_kappa_plot(rows, rmf_files[0], tsv_path, plot_path)
        config.vprint(f"[measure] Plot saved to: {plot_path}")
    
    return {
        "stats": stats,
        "tsv_path": tsv_path,
        "plot_path": plot_path,
        "frames_measured": len(rows)
    }


def run_fpsim_pipeline(config: PipelineConfig) -> Dict[str, Path]:
    """
    Execute the complete fpsim pipeline: predict → segment → sample → measure.
    
    Returns:
        Dictionary with paths to key outputs: topology, segments, sampling, measurements
    """
    config.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Ensure we have an AF PDB
    af_pdb = ensure_af_pdb(config)
    
    # Step 2: Read sequence and determine workflow type
    seq_id_raw, seq_raw = read_fasta_with_selection(config.fasta, config.sequence_id)
    
    # Step 3: Execute appropriate workflow
    if ":" in seq_raw:
        # Multimer workflow
        config.vprint(f"[run] Detected multimer: {len(seq_raw.split(':'))} chains")
        top_path = process_multimer_workflow(config, af_pdb, seq_id_raw, seq_raw)
    else:
        # Single-chain workflow
        config.vprint(f"[run] Single-chain workflow for chain {config.chain}")
        top_path = process_single_chain_workflow(config, af_pdb, seq_id_raw, seq_raw)
    
    # Step 4: Execute measurement step if enabled
    config.status_callback("measurements")
    measurement_results = run_measurement_step(config, config.out_dir)
    
    results = {
        "topology": top_path,
        "segments": config.out_dir / "segments.json",
        "sampling": config.out_dir / "out_linkers"
    }
    
    if measurement_results:
        results["measurements"] = measurement_results
    
    return results
