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
    # Use index access to handle variable row length (2, 4, or 5 columns)
    r_arr = np.array([row[1] for row in rows], dtype=np.float64)
    out = {
        "frames": len(rows),
        "distance_mean": float(np.nanmean(r_arr)),
        "distance_std": float(np.nanstd(r_arr)),
    }
    
    # Check if we have kappa data (4 sites)
    if len(rows) > 0 and len(rows[0]) >= 4 and not np.isnan(rows[0][2]):
        k_arr = np.array([row[2] for row in rows], dtype=np.float64)
        k2_arr = np.array([row[3] for row in rows], dtype=np.float64)
        out.update({
            "kappa_mean": float(np.nanmean(k_arr)),
            "kappa_std": float(np.nanstd(k_arr)),
            "kappa_std": float(np.nanstd(k_arr)),
            "kappa2_mean": float(np.nanmean(k2_arr)),
            "kappa2_std": float(np.nanstd(k2_arr)),
        })
        
    # Check if we have Rapp (5 items)
    if len(rows) > 0 and len(rows[0]) >= 5:
        rapp_arr = np.array([row[4] for row in rows], dtype=np.float64)
        out.update({
             "rapp_mean": float(np.nanmean(rapp_arr)),
             "rapp_std": float(np.nanstd(rapp_arr)),
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
                
                
            # Calculate Rapp per frame: Rapp = R * ((2/3) / k2)^(1/6)
            # If k2 is invalid or <=0 (should handle carefully), rapp is nan
            rapp = float('nan')
            if not np.isnan(k2) and k2 > 0:
                rapp = r * np.power((2.0/3.0) / k2, 1.0/6.0)

            rows.append((global_frame_offset + fi, r, k, k2, rapp))
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
            fh.write("frame\tRDA\n")
            for row in rows:
                # row is (fi, r) or (fi, r, nan, nan, nan) depending on legacy
                fi, r = row[0], row[1]
                fh.write(f"{fi}\t{r:.3f}\n")
        else:
            fh.write("frame\tRDA\tkappa\tkappa2\tRapp\n")
            for fi, r, k, k2, rapp in rows:
                k_str = "" if np.isnan(k) else f"{k:.6f}"
                k2_str = "" if np.isnan(k2) else f"{k2:.6f}"
                rapp_str = "" if np.isnan(rapp) else f"{rapp:.3f}"
                
                fh.write(f"{fi}\t{r:.3f}\t{k_str}\t{k2_str}\t{rapp_str}\n")


def create_distance_kappa_plot(rows, rmf_path: Path, out_tsv: Path | None = None, plot_out: Path | None = None, plot_title: str | None = None):
    """Create dual-panel 2D histogram plot: Rapp vs k2 and Distance vs k2."""
    # Distance (Å) and kappa^2 arrays
    # Distance (Å) and kappa^2 arrays
    D = np.array([r for _, r, _, _, _ in rows], dtype=np.float64)
    K2 = np.array([k2 for _, _, _, k2, _ in rows], dtype=np.float64)
    R_app = np.array([rapp for _, _, _, _, rapp in rows], dtype=np.float64)
    
    m = np.isfinite(D) & np.isfinite(K2)
    D = D[m]; K2 = K2[m]
    # Filter Rapp using same mask if needed, but it might have its own NaNs
    # Actually simpler: Rapp comes pre-calculated.
    # Note: R_app might be NaN even if D and K2 are finite (e.g. K2=0, though usually filtered)
    
    # We use the valid indices for D/K2 to filter Rapp for the plots?
    # Or just mask all.
    # Let's realign.
    
    # Re-extract clean arrays for plotting using boolean masking on ALL
    # Actually, R_app calculation depends on K2 > 0.
    
    # Let's blindly trust the pre-calculated Rapp for the Rapp-histogram.
    # But for D vs K2 histogram, we use valid D and K2.
    
    # Update masking
    mask_dist = np.isfinite(D) & np.isfinite(K2)
    D = D[mask_dist]
    K2 = K2[mask_dist] # For Dist plot
    
    # For Rapp plot
    # We want valid Rapp and valid K2
    mask_rapp = np.isfinite(R_app) & np.isfinite(K2) # Wait, K2 is already filtered? No, K2 is raw array here.
    # Let's act on original arrays
    K2_orig = np.array([k2 for _, _, _, k2, _ in rows], dtype=np.float64)
    
    mask_rapp_valid = np.isfinite(R_app) & np.isfinite(K2_orig)
    R_app_clean = R_app[mask_rapp_valid]
    K2_clean_for_rapp = K2_orig[mask_rapp_valid]
    
    # Filter valid R_app for stats and plotting
    # We need strictly valid R_app for its own histogram
    valid_rapp_mask = np.isfinite(R_app)
    R_app_clean = R_app[valid_rapp_mask]
    K2_clean_for_rapp = K2[valid_rapp_mask]
    
    # Calculate statistics
    d_mean, d_std = np.mean(D), np.std(D)
    k2_mean, k2_std = np.mean(K2), np.std(K2)
    
    if len(R_app_clean) > 0:
        r_app_mean, r_app_std = np.mean(R_app_clean), np.std(R_app_clean)
    else:
        r_app_mean, r_app_std = np.nan, np.nan

    # Binning ranges
    # Distance and Rapp share the same X binning logic (0-150 A)
    xb = np.linspace(0.0, 150.0, 151)
    # Log bins for K2, extended to 0.001
    yb = np.logspace(-3, np.log10(4), 81)
    
    # Histogram 1: Distance vs K2
    H_dist, xe_dist, ye = np.histogram2d(D, K2, bins=[xb, yb])
    
    # Histogram 2: Rapp vs K2 (using clean data)
    H_rapp, xe_rapp, _ = np.histogram2d(R_app_clean, K2_clean_for_rapp, bins=[xb, yb])
    
    # Calculate Modes (Stabilized)
    def get_mode(hist, edges, is_log=False):
        if hist.sum() == 0: return np.nan
        # 1. Find the peak bin
        idx = np.argmax(hist)
        
        # 2. Define a small window around peak (e.g. +/- 1 bin) to centroid
        # This stabilizes the mode against single-bin noise
        start = max(0, idx - 1)
        end = min(len(hist), idx + 2) # Slice is exclusive at end
        
        subset_counts = hist[start:end]
        
        if is_log:
            # For log bins, working in log-space (geometric mean) matches the linear centroid logic
            # Bin centers in log space: sqrt(edge[i]*edge[i+1])
            # Actually simpler: log_centers = (log(e[i]) + log(e[i+1])) / 2
            # Log edges
            le = np.log(edges)
            subset_log_centers = 0.5 * (le[start:end] + le[start+1:end+1])
            
            # Weighted mean of log centers
            total_w = subset_counts.sum()
            if total_w == 0: return np.nan
            w_mean_log = np.sum(subset_counts * subset_log_centers) / total_w
            return np.exp(w_mean_log)
        else:
            # Linear bin centers
            frame_centers = 0.5 * (edges[start:end] + edges[start+1:end+1])
            total_w = subset_counts.sum()
            if total_w == 0: return np.nan
            return np.sum(subset_counts * frame_centers) / total_w
            
    x_mode = get_mode(H_dist.sum(axis=1), xb)
    y_mode = get_mode(H_dist.sum(axis=0), yb, is_log=True)
    r_app_mode = get_mode(H_rapp.sum(axis=1), xb)

    # Determine output path
    if plot_out is None:
        if out_tsv is not None:
            plot_out = out_tsv.with_suffix(".png")
        else:
            plot_out = rmf_path.with_suffix(".measure.png")

    # Use explicit title if provided, else fallback to filename
    final_title = plot_title if plot_title is not None else rmf_path.name

    save_dual_2d_marginals_png(
        img_path=plot_out,
        H_dist=H_dist,
        H_rapp=H_rapp,
        x_edges=xb,
        y_edges=yb,
        title=final_title,
        stats={
            "x_mean": d_mean, "x_std": d_std,
            "y_mean": k2_mean, "y_std": k2_std,
            "r_app_mean": r_app_mean, "r_app_std": r_app_std,
            "x_mode": x_mode, "y_mode": y_mode, "r_app_mode": r_app_mode
        }
    )


def save_dual_2d_marginals_png(img_path, H_dist, H_rapp, x_edges, y_edges, title="", cmap_2d: str = "viridis", stats: Dict[str, float] = None):
    """Render a dual-panel 2D histogram (Rapp vs K2 | Dist vs K2) with marginals."""
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.ticker as ticker

    # Data prep
    H_dist = np.asarray(H_dist, dtype=float)
    H_rapp = np.asarray(H_rapp, dtype=float)
    x_edges = np.asarray(x_edges, dtype=float)
    y_edges = np.asarray(y_edges, dtype=float)
    
    H_dist_m = np.ma.masked_where(H_dist == 0, H_dist)
    H_rapp_m = np.ma.masked_where(H_rapp == 0, H_rapp)
    
    cmap = cm.get_cmap(cmap_2d).copy()
    cmap.set_bad(color='white')

    # Figure Layout
    fig = plt.figure(figsize=(16, 7))
    
    # Dimensions
    b_main = 0.10
    h_main = 0.60
    w_main = 0.35 
    h_marg = 0.15
    w_marg_y = 0.08
    gap_x = 0.0
    
    # Panel 1 (Left): R_app vs K2
    l1 = 0.08
    rect_rapp = [l1, b_main, w_main, h_main]
    rect_rapp_x = [l1, b_main + h_main, w_main, h_marg]
    
    # Panel 2 (Right): Dist vs K2
    l2 = l1 + w_main + gap_x
    rect_dist = [l2, b_main, w_main, h_main]
    rect_dist_x = [l2, b_main + h_main, w_main, h_marg]
    
    # Shared Y Marginal (Far Right)
    l_y = l2 + w_main
    rect_y = [l_y, b_main, w_marg_y, h_main]
    rect_text = [l_y, b_main + h_main, w_marg_y*2, h_marg] 
    
    # Create Axes (NO SHARING to avoid auto-hide bugs)
    ax_rapp = fig.add_axes(rect_rapp)
    ax_rapp_x = fig.add_axes(rect_rapp_x)
    ax_dist = fig.add_axes(rect_dist)
    ax_dist_x = fig.add_axes(rect_dist_x)
    ax_y = fig.add_axes(rect_y)
    ax_text = fig.add_axes(rect_text)
    ax_text.axis("off") 
    
    axes_main = [ax_rapp, ax_dist]
    axes_x = [ax_rapp_x, ax_dist_x]
    
    # Common Ticks Definition
    x_ticks_maj = np.arange(0, 160, 20)
    # Y Ticks: Log decades + 0.5 steps
    y_ticks_vals = [0.01, 0.1, 0.5, 1.0, 2.0, 4.0]
    
    # Plotting Main 2D
    def plot_2d_main(ax, H):
        im = ax.pcolormesh(x_edges, y_edges, H.T, cmap=cmap, shading="auto", rasterized=True)
        ax.set_yscale("log")
        ax.set_xlim(0, 150)
        ax.set_ylim(0.005, 4.5)
        
        # Grid settings
        # ax.grid(False) 

        # Ticks Configuration
        ax.xaxis.set_major_locator(ticker.FixedLocator(x_ticks_maj))
        ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks_vals))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_yticklabels([f"{t:g}" for t in y_ticks_vals])
        
        # Turn off minor ticks to align with user requested explicit ticks
        ax.minorticks_off()
        
        # Tick Params: Enable ALL sides (Top, Right, Left, Bottom)
        # Direction IN. Labels mostly controlled separately.
        ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.5,
                       top=True, right=True, bottom=True, left=True,
                       labelbottom=True, labelleft=True, labelsize=12)
                       
        # Bold labels
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontweight('bold')
            label.set_fontsize(12)
            
        # Spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)
        return im

    img1 = plot_2d_main(ax_rapp, H_rapp_m)
    img2 = plot_2d_main(ax_dist, H_dist_m)
    
    # Axis Labels
    ax_rapp.set_xlabel("Apparent Distance (Å)", fontsize=14, fontweight='bold')
    ax_rapp.set_ylabel("$\\kappa^2$", fontsize=14, fontweight='bold')
    ax_dist.set_xlabel("Distance (Å)", fontsize=14, fontweight='bold')
    ax_dist.set_ylabel("") # No label for second plot
    
    # Fix specific visibility for side-by-side layout
    # Ax Rapp: Left labels ON.
    # Ax Dist: Left labels OFF (but ticks ON).
    ax_dist.tick_params(axis='y', labelleft=False)
    
    # Marginals
    def plot_marginal_x(ax, H, color):
        counts = H.sum(axis=1)
        # Using raw counts for marginals to match visual density (especially for log axes)
        ax.bar(x_edges[:-1], counts, width=np.diff(x_edges), color=color, alpha=0.9, align='edge', edgecolor=color)
        ax.set_xlim(0, 150)
        ax.set_xticks([]) # Hide marginal ticks
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    
    plot_marginal_x(ax_rapp_x, H_rapp, "#6666aa")
    plot_marginal_x(ax_dist_x, H_dist, "#6666aa")
    
    # Y Marginal
    y_counts = H_dist.sum(axis=0)
    y_widths = np.diff(y_edges)
    # Use raw counts for Y marginal on log axis to represent density per log-interval
    ax_y.barh(y_edges[:-1], y_counts, height=y_widths, color="#66aa66", alpha=0.9, align='edge', edgecolor="#66aa66")
    ax_y.set_yscale("log")
    ax_y.set_ylim(0.005, 4.5)
    ax_y.set_xticks([])
    ax_y.set_yticks([])
    for spine in ax_y.spines.values():
        spine.set_linewidth(1.5)
        
    # Stats Lines (Mean - Dashed)
    if stats:
        if "r_app_mean" in stats:
            m = stats["r_app_mean"]
            ax_rapp.axvline(m, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            ax_rapp_x.axvline(m, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        if "y_mean" in stats:
            m = stats["y_mean"]
            for ax in [ax_rapp, ax_dist, ax_y]:
                ax.axhline(m, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        if "x_mean" in stats:
            m = stats["x_mean"]
            ax_dist.axvline(m, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            ax_dist_x.axvline(m, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    # Mode Lines (Mode - Dotted)
    if stats:
        if "r_app_mode" in stats and not np.isnan(stats["r_app_mode"]):
            m = stats["r_app_mode"]
            ax_rapp.axvline(m, color='red', linestyle=':', linewidth=1.5, alpha=0.9)
            ax_rapp_x.axvline(m, color='red', linestyle=':', linewidth=1.5, alpha=0.9)
        if "y_mode" in stats and not np.isnan(stats["y_mode"]):
            m = stats["y_mode"]
            for ax in [ax_rapp, ax_dist, ax_y]:
                ax.axhline(m, color='red', linestyle=':', linewidth=1.5, alpha=0.9)
        if "x_mode" in stats and not np.isnan(stats["x_mode"]):
            m = stats["x_mode"]
            ax_dist.axvline(m, color='red', linestyle=':', linewidth=1.5, alpha=0.9)
            ax_dist_x.axvline(m, color='red', linestyle=':', linewidth=1.5, alpha=0.9)

    # Legend
    if stats:
        # 1. Means (Bottom Legend)
        txt = []
        if "r_app_mean" in stats:
             txt.append(f"$R_{{app}}(2/3)$: {stats['r_app_mean']:.1f} ± {stats['r_app_std']:.1f} Å")
        if "x_mean" in stats:
            txt.append(f"Dist: {stats['x_mean']:.1f} ± {stats['x_std']:.1f} Å")
        if "y_mean" in stats:
            txt.append(f"$\\kappa^2$: {stats['y_mean']:.2f} ± {stats['y_std']:.2f}")

        # 2. Modes (Top Legend)
        txt_m = []
        if "r_app_mode" in stats and not np.isnan(stats["r_app_mode"]):
             txt_m.append(f"$R_{{app}}$ Mode: {stats['r_app_mode']:.1f} Å")
        if "x_mode" in stats and not np.isnan(stats["x_mode"]):
             txt_m.append(f"Dist Mode: {stats['x_mode']:.1f} Å")
        if "y_mode" in stats and not np.isnan(stats["y_mode"]):
             txt_m.append(f"$\\kappa^2$ Mode: {stats['y_mode']:.2f}")

        # Display Stats (Means) - Lower
        if txt:
            ax_text.text(0.2, 0.45, "\n".join(txt), transform=ax_text.transAxes,
                         horizontalalignment='left', verticalalignment='center',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, linewidth=1.5, edgecolor='black'),
                         fontsize=11, fontweight='bold')
                         
        # Display Modes - Upper
        if txt_m:
            ax_text.text(0.2, 1.2, "\n".join(txt_m), transform=ax_text.transAxes,
                         horizontalalignment='left', verticalalignment='center',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, linewidth=1.5, edgecolor='black'),
                         fontsize=11, fontweight='bold')

    # Colorbar
    cax = fig.add_axes([0.88, b_main, 0.015, h_main])
    cbar = fig.colorbar(img2, cax=cax)
    cbar.set_label("Counts", fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    if title:
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        
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
