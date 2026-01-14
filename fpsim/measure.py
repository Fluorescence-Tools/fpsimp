"""Measurement and plotting utilities for fpsim.

During the initial refactor, we re-export from cli.py to preserve behavior.
Future commits can move the plotting and measurement logic here.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import numpy as np
from pathlib import Path
import IMP
import IMP.algebra
import IMP.core
import IMP.atom
import IMP.rmf
import RMF
import re


# Setup logger for this module
logger = logging.getLogger(__name__)


def setup_rmf_hierarchy(rmf_path: Path, verbose: bool = False):
    """Setup RMF hierarchy and return model, rmf, hierarchy, and leaves."""
    
    mdl = IMP.Model()
    rmf = RMF.open_rmf_file_read_only(str(rmf_path))
    hs = IMP.rmf.create_hierarchies(rmf, mdl)
    if not hs:
        raise ValueError("Failed to create hierarchy from RMF")
    hier = hs[0]
    
    try:
        leaves = list(IMP.atom.get_leaves(hier))
    except Exception as e:
        leaves = []
        if verbose:
            logger.warning(f"Failed to enumerate leaves: {e}")
    
    if verbose:
        logger.info(f"Leaves={len(leaves)}")
        for i, lp in enumerate(leaves[:10]):
            nm = lp.get_name() if hasattr(lp, "get_name") else str(lp)
            logger.info(f"  leaf[{i}] name={nm} pid={lp.get_particle_index()}")
    
    return mdl, rmf, hier, leaves


def dump_leaves_to_tsv(leaves, dump_path: Path):
    """Write leaf particle info to TSV file."""
    
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dump_path, "w") as fh:
        fh.write("pid\tname\tchain\tresseq\tatom\n")
        for lp in leaves:
            pid = lp.get_particle_index()
            nm = lp.get_name() if hasattr(lp, "get_name") else ""
            ch = ""; ri = ""; at = ""
            
            try:
                if IMP.atom.Atom.get_is_setup(lp):
                    at = IMP.atom.Atom(lp).get_atom_type().get_string()
            except Exception:
                pass
            
            # Try to get chain/res via residue or ancestors
            try:
                if IMP.atom.Residue.get_is_setup(lp):
                    res = IMP.atom.Residue(lp)
                    ri = int(res.get_index())
                    ch = IMP.atom.Chain(IMP.atom.get_chain(res)).get_id()
                else:
                    cur = IMP.atom.Hierarchy(lp)
                    while cur.get_parent().get_particle_index() != cur.get_particle_index():
                        par = cur.get_parent()
                        if IMP.atom.Residue.get_is_setup(par.get_particle()):
                            res = IMP.atom.Residue(par.get_particle())
                            ri = int(res.get_index())
                            ch = IMP.atom.Chain(IMP.atom.get_chain(res)).get_id()
                            break
                        cur = par
            except Exception:
                pass
            
            # Fallback parse from name
            if (not ch or not ri) and nm:
                m = re.search(r"([A-Za-z0-9]):(\d+)", nm)
                if m:
                    ch = ch or m.group(1)
                    ri = ri or int(m.group(2))
            
            fh.write(f"{pid}\t{nm}\t{ch}\t{ri}\t{at}\n")


def match_particles_to_sites(hier, sites, verbose: bool = False):
    """Match parsed sites to particles in hierarchy."""
    
    parts = [
        match_atom_particle(hier, chain, resi, atom)
        for (chain, resi, atom) in sites
    ]
    
    if verbose:
        for i, p in enumerate(parts):
            if p is None:
                logger.warning(f"Match[{i}] NOT FOUND for {sites[i]}")
            else:
                nm = p.get_name() if hasattr(p, "get_name") else str(p)
                logger.info(f"Match[{i}] pid={p} name={nm}")
    
    if any(p is None for p in parts):
        # Collect a few example leaves to help the user debug
        examples = []
        try:
            for i, lp in enumerate(IMP.atom.get_leaves(hier)):
                nm = lp.get_name() if hasattr(lp, "get_name") else str(lp)
                examples.append(nm)
                if len(examples) >= 10:
                    break
        except Exception:
            pass
        idx_bad = [i for i, p in enumerate(parts) if p is None]
        hint = ("; examples: " + ", ".join(examples)) if examples else ""
        raise ValueError(f"Could not locate site(s) in RMF hierarchy: indices {idx_bad} of {len(parts)}{hint}")
    
    return parts


def compute_kappa(coords: list[np.ndarray]) -> tuple[float, float, float]:
    """Compute distance, kappa, and kappa-squared from four 3D coordinates."""
    d1, d2, a1, a2 = coords
    dM = 0.5 * (d1 + d2)
    aM = 0.5 * (a1 + a2)
    rvec = aM - dM
    r = float(np.linalg.norm(rvec))
    
    if r > 0:
        rhat = rvec / r
        muD = d2 - d1
        muA = a2 - a1
        muD /= (np.linalg.norm(muD) + 1e-12)
        muA /= (np.linalg.norm(muA) + 1e-12)
        k = float(np.dot(muD, muA) - 3.0 * np.dot(muD, rhat) * np.dot(muA, rhat))
        k2 = k * k
    else:
        k = float('nan')
        k2 = float('nan')
        
    return r, k, k2


def compute_statistics(rows):
    """Compute summary statistics from measurement rows."""
    r_arr = np.array([r for _, r, _, _ in rows], dtype=np.float64)
    out = {
        "frames": len(rows),
        "distance_mean": float(np.nanmean(r_arr)),
        "distance_std": float(np.nanstd(r_arr)),
    }
    
    # Check if we have kappa data (4 sites)
    if len(rows) > 0 and not np.isnan(rows[0][2]):
        k_arr = np.array([k for _, _, k, _ in rows], dtype=np.float64)
        k2_arr = np.array([k2 for _, _, _, k2 in rows], dtype=np.float64)
        out.update({
            "kappa_mean": float(np.nanmean(k_arr)),
            "kappa_std": float(np.nanstd(k_arr)),
            "kappa2_mean": float(np.nanmean(k2_arr)),
            "kappa2_std": float(np.nanstd(k2_arr)),
        })
    
    return out


def measure_frames(rmf, parts, frame_start: int = 0, max_frames: int | None = None, verbose: bool = False, progress: bool = False, global_frame_offset: int = 0):
    """Loop over RMF frames and compute distance (+ kappa if 4 sites)."""
    import RMF
    import IMP.rmf
    
    rows = []
    num_frames = rmf.get_number_of_frames()
    
    start = max(0, frame_start)
    end = num_frames
    if max_frames is not None:
        end = min(end, start + max_frames)
    
    if verbose:
        logger.info(f"Measuring frames {start} to {end} (total {num_frames})")
        
    for fi in range(start, end):
        try:
            IMP.rmf.load_frame(rmf, RMF.FrameID(fi))
            coords = [xyz_of(p) for p in parts]
            
            if len(coords) == 2:
                r = float(np.linalg.norm(coords[1] - coords[0]))
                k = float('nan')
                k2 = float('nan')
            elif len(coords) == 4:
                r, k, k2 = compute_kappa(coords)
            else:
                raise ValueError(f"Expected 2 or 4 points, got {len(coords)}")
                
            rows.append((global_frame_offset + fi, r, k, k2))
        except Exception as e:
            if verbose:
                logger.warning(f"Error measuring frame {fi}: {e}")
            continue
             
    return rows


def measure_trajectory(rmf_paths: list[Path], parts_info: list[tuple[str, int, str]], verbose: bool = False):
    """Process multiple RMF files and measure across all of them."""
    import RMF
    
    if not rmf_paths:
        return []

    # Setup hierarchy from first file
    mdl, rmf0, hier, _ = setup_rmf_hierarchy(rmf_paths[0], verbose)
    
    # Match particles once
    parts = match_particles_to_sites(hier, parts_info, verbose)
    
    all_rows = []
    global_offset = 0
    
    for rp in rmf_paths:
        try:
            rmf = RMF.open_rmf_file_read_only(str(rp))
            IMP.rmf.link_hierarchies(rmf, [hier])
            rows = measure_frames(rmf, parts, verbose=verbose, global_frame_offset=global_offset)
            all_rows.extend(rows)
            global_offset += rmf.get_number_of_frames()
        except Exception as e:
            if verbose:
                logger.warning(f"Failed to process RMF {rp}: {e}")
            continue
            
    return all_rows


def write_measurements_tsv(rows, out_tsv: Path):
    """Write measurement results to TSV file."""
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w") as fh:
        # Check if we have kappa data (4 sites)
        has_kappa = len(rows) > 0 and not np.isnan(rows[0][2])
        
        if not has_kappa:
            fh.write("frame\tdistance_Ang\n")
            for fi, r, _, _ in rows:
                fh.write(f"{fi}\t{r:.3f}\n")
        else:
            fh.write("frame\tdistance_Ang\tkappa\tkappa2\n")
            for fi, r, k, k2 in rows:
                fh.write(f"{fi}\t{r:.3f}\t{'' if np.isnan(k) else f'{k:.6f}'}\t{'' if np.isnan(k2) else f'{k2:.6f}'}\n")


def create_distance_kappa_plot(rows, rmf_path: Path, out_tsv: Path | None = None, plot_out: Path | None = None):
    """Create 2D histogram plot of distance vs kappa^2."""
    # Distance (Å) and kappa^2 arrays
    D = np.array([r for _, r, _, _ in rows], dtype=np.float64)
    K2 = np.array([k2 for _, _, _, k2 in rows], dtype=np.float64)
    m = np.isfinite(D) & np.isfinite(K2)
    D = D[m]; K2 = K2[m]
    
    # Binning ranges
    xb = np.linspace(0.0, 150.0, 151)
    yb = np.logspace(-2, np.log10(4), 81)
    H, xe, ye = np.histogram2d(D, K2, bins=[xb, yb])

    # Determine output path
    if plot_out is None:
        if out_tsv is not None:
            plot_out = out_tsv.with_suffix(".png")
        else:
            plot_out = rmf_path.with_suffix(".measure.png")

    save_2d_marginals_png(
        img_path=plot_out,
        H=H,
        x_edges=xe,
        y_edges=ye,
        title="Distance vs $\\kappa^2$",
        xlabel="Distance (Å)",
        ylabel="$\\kappa^2$",
    )


def save_2d_marginals_png(img_path, H, x_edges, y_edges, title="", xlabel="", ylabel="", cmap_2d: str = "viridis"):
    """Render a 2D histogram (x vs y) with top/right marginals and save as PNG.

    Assumes y_edges may be log-spaced; uses pcolormesh aligned to edges.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    H = np.asarray(H, dtype=float)
    x_edges = np.asarray(x_edges, dtype=float)
    y_edges = np.asarray(y_edges, dtype=float)

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4], wspace=0.05, hspace=0.05)
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_x = fig.add_subplot(gs[0, 0], sharex=ax_hist)
    ax_y = fig.add_subplot(gs[1, 1], sharey=ax_hist)

    # 2D heatmap aligned to edges
    im = ax_hist.pcolormesh(x_edges, y_edges, H.T, cmap=cmap_2d, shading="auto")
    ax_hist.set_yscale("log")
    ax_hist.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    ax_hist.set_ylim(float(y_edges[0]), float(y_edges[-1]))
    ax_hist.set_xlabel(xlabel)
    ax_hist.set_ylabel(ylabel)

    # Marginals
    ax_x.hist(
        np.repeat((x_edges[:-1] + x_edges[1:]) / 2.0, H.sum(axis=1).astype(int), axis=0)
        if H.size and H.sum() > 0 else [],
        bins=x_edges,
        color="#6666aa",
    )
    ax_x.axis("off")
    ax_y.hist(
        np.repeat((y_edges[:-1] + y_edges[1:]) / 2.0, H.sum(axis=0).astype(int), axis=0)
        if H.size and H.sum() > 0 else [],
        bins=y_edges,
        orientation="horizontal",
        color="#66aa66",
    )
    ax_y.axis("off")

    # Colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cax, label="Counts")

    fig.suptitle(title)
    img_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(img_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


import re

def parse_site(s: str):
    """Parse site string like 'CHAIN:RESSEQ:ATOM', 'PDB:CHAIN:RESSEQ:ATOM', or 'A481CA'."""
    # New format: single string like A481CA
    m = re.match(r'^([A-Za-z])([0-9]+)([A-Za-z_]+)$', s)
    if m:
        chain, resseq_str, atom = m.groups()
        try:
            resseq = int(resseq_str)
            return chain.strip(), resseq, atom.strip()
        except ValueError:
            raise ValueError(f"Invalid residue number in site '{s}'")

    # Old format: CHAIN:RESSEQ:ATOM or CHAIN:RESSEQ
    parts = [p for p in s.split(":") if p]
    if len(parts) < 2:
        raise ValueError(f"Invalid site '{s}'. Expected CHAIN:RESSEQ or CHAIN:RESSEQ:ATOM.")
    
    if len(parts) == 2:
        chain, resseq_str = parts
        atom = "CA"
    elif len(parts) == 3:
        chain, resseq_str, atom = parts
    else: # PDB:CHAIN:RESSEQ:ATOM format
        chain, resseq_str, atom = parts[-3:]
    
    try:
        resseq = int(resseq_str)
    except ValueError:
        raise ValueError(f"Invalid residue index in site '{s}'")
        
    return chain.strip(), resseq, atom.strip()


def match_atom_particle(h, chain_id: str, resseq: int, atom_name: str):
    """Select particle using IMP.atom.Selection on the hierarchy.
    1) Try atom-level selection (chain, residue_index, atom_type)
    2) If none found or no XYZ, try residue-level selection (chain, residue_index) and
       pick a particle with XYZ (the residue itself or a child).
    """
    
    # 1) Atom-level selection
    try:
        def _get_selection(h, cid, rs, an):
            sel = IMP.atom.Selection(h)
            sel.set_chain_id(cid)
            sel.set_residue_index(int(rs))
            if an:
                sel.set_atom_type(IMP.atom.AtomType(an))
            return sel.get_selected_particles()

        # Try exact match first
        cand = _get_selection(h, chain_id, resseq, atom_name)
        
        # If no match, try common variations for long chain IDs (e.g., from webapp)
        if not cand:
            # Try last character if it's a single letter (e.g. "path_A" -> "A")
            if len(chain_id) > 1 and chain_id[-1].isupper():
                cand = _get_selection(h, chain_id[-1], resseq, atom_name)
            
            # Try part after last underscore (e.g. "pdb_A" -> "A")
            if not cand and "_" in chain_id:
                last_part = chain_id.split("_")[-1]
                if last_part != chain_id:
                    cand = _get_selection(h, last_part, resseq, atom_name)

        for p in cand:
            if IMP.core.XYZ.get_is_setup(p):
                return p
    except Exception:
        pass
    # 2) Residue-level selection (no atom)
    try:
        sel2 = IMP.atom.Selection(h)
        sel2.set_chain_id(chain_id)
        sel2.set_residue_index(int(resseq))
        cand2 = sel2.get_selected_particles()
        # Prefer ones with XYZ directly
        for p in cand2:
            if IMP.core.XYZ.get_is_setup(p):
                return p
        # Otherwise try leaves under these particles
        for p in cand2:
            try:
                hp = IMP.atom.Hierarchy(p)
                for leaf in IMP.atom.get_leaves(hp):
                    if IMP.core.XYZ.get_is_setup(leaf):
                        return leaf
            except Exception:
                continue
    except Exception:
        pass
    return None


def xyz_of(p):
    """Extract XYZ coordinates from an IMP particle."""
    import numpy as np
    
    if not IMP.core.XYZ.get_is_setup(p):
        raise RuntimeError("Particle lacks XYZ decorator")
    xyz = IMP.core.XYZ(p)
    return np.array([xyz.get_x(), xyz.get_y(), xyz.get_z()], dtype=np.float64)

def get_particles_by_segment(hier: IMP.atom.Hierarchy, segments: dict, segment_kind: str) -> list[IMP.Particle]:
    """Select particles corresponding to a specific segment kind (e.g., 'core' or 'fp')."""
    particles = []
    for chain_id, chain_info in segments.get('chains', {}).items():
        if segment_kind == 'fp':
            target_segments = chain_info.get('fp_domains', [])
        else:
            target_segments = [s for s in chain_info.get('segments', []) if s['kind'] == segment_kind]

        for segment in target_segments:
            start_res = segment['start']
            end_res = segment['end']
            sel = IMP.atom.Selection(
                hier,
                chain_id=chain_id,
                residue_indexes=range(start_res, end_res + 1)
            )
            particles.extend(sel.get_selected_particles())
    
    # Return unique particles
    return list(dict.fromkeys(particles))

def get_rigid_bodies(hier: IMP.atom.Hierarchy) -> list[IMP.Particle]:
    """Traverse the hierarchy and return all particles that are RigidBody objects."""

    # To get all particles, simply initialize Selection with the hierarchy and no filters.
    all_particles = IMP.atom.Selection(hier).get_selected_particles()
    rigid_bodies = [p for p in all_particles if IMP.core.RigidBody.get_is_setup(p)]
    return rigid_bodies

def align_rigid_bodies(rigid_bodies_to_transform: list[IMP.Particle], ref_particles_current: list[IMP.Particle], target_coords: list[IMP.algebra.Vector3D]):
    """Align rigid bodies using IMP's built-in Kabsch algorithm implementation."""

    if len(ref_particles_current) != len(target_coords):
        raise ValueError("The number of current reference particles must match the number of target coordinates.")

    # This function requires a list of Vector3D coordinate objects.
    source_coords = [IMP.core.XYZ(p).get_coordinates() for p in ref_particles_current]

    # Get the transformation that aligns the current reference particles to the target reference particles.
    # This computes both the optimal rotation and translation.
    transformation = IMP.algebra.get_transformation_aligning_first_to_second(
        source_coords, target_coords
    )

    # Apply the transformation to each rigid body by directly updating its reference frame.
    for p in rigid_bodies_to_transform:
        rb = IMP.core.RigidBody(p)
        current_transform = rb.get_reference_frame().get_transformation_to()
        new_transform = IMP.algebra.compose(transformation, current_transform)
        rb.set_reference_frame(IMP.algebra.ReferenceFrame3D(new_transform))
