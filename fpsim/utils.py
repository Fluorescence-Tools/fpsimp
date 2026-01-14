"""Utility functions for fpsim.

Common helpers extracted from cli.py to improve modularity.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import click
from Bio import SeqIO
from Bio.PDB import PDBParser, MMCIFParser, PPBuilder


def read_single_fasta(fasta_path: Path) -> Tuple[str, str]:
    """Read exactly one sequence from a FASTA file."""
    recs = list(SeqIO.parse(str(fasta_path), "fasta"))
    if len(recs) != 1:
        raise click.ClickException("FASTA must contain exactly one sequence")
    rec = recs[0]
    seq = str(rec.seq).replace("\n", "").replace("\r", "")
    return rec.id, seq


def list_fasta_sequences(fasta_path: Path) -> List[Tuple[str, str]]:
    """List all sequences in a FASTA file with their IDs."""
    recs = list(SeqIO.parse(str(fasta_path), "fasta"))
    return [(rec.id, str(rec.seq).replace("\n", "").replace("\r", "")) for rec in recs]


def read_fasta_sequence_by_id(fasta_path: Path, sequence_id: str) -> Tuple[str, str]:
    """Read a specific sequence from a FASTA file by its ID.
    
    Args:
        fasta_path: Path to the FASTA file
        sequence_id: ID of the sequence to extract (can be partial match)
        
    Returns:
        Tuple of (sequence_id, sequence)
        
    Raises:
        click.ClickException: If sequence not found or multiple matches
    """
    recs = list(SeqIO.parse(str(fasta_path), "fasta"))
    
    if not recs:
        raise click.ClickException(f"No sequences found in {fasta_path}")
    
    # Try exact match first
    exact_matches = [rec for rec in recs if rec.id == sequence_id]
    if len(exact_matches) == 1:
        rec = exact_matches[0]
        seq = str(rec.seq).replace("\n", "").replace("\r", "")
        return rec.id, seq
    elif len(exact_matches) > 1:
        raise click.ClickException(f"Multiple exact matches for sequence ID '{sequence_id}'")
    
    # Try partial match (case-insensitive)
    partial_matches = [rec for rec in recs if sequence_id.lower() in rec.id.lower()]
    if len(partial_matches) == 1:
        rec = partial_matches[0]
        seq = str(rec.seq).replace("\n", "").replace("\r", "")
        return rec.id, seq
    elif len(partial_matches) > 1:
        match_ids = [rec.id for rec in partial_matches]
        raise click.ClickException(
            f"Multiple partial matches for '{sequence_id}': {', '.join(match_ids)}. "
            f"Please be more specific."
        )
    
    # No matches found
    available_ids = [rec.id for rec in recs]
    raise click.ClickException(
        f"Sequence ID '{sequence_id}' not found. Available sequences: {', '.join(available_ids)}"
    )


def parse_sequence_ids(sequence_id_str: str = None) -> List[str]:
    """Parse comma-separated sequence IDs into a list."""
    if sequence_id_str is None:
        return []
    return [s.strip() for s in sequence_id_str.split(',') if s.strip()]


def sanitize_folder_name(name: str) -> str:
    """Sanitize a sequence ID to be a valid folder name."""
    import re
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    # Ensure it's not empty
    if not sanitized:
        sanitized = "sequence"
    return sanitized


def write_single_sequence_fasta(sequence_id: str, sequence: str, output_path: Path) -> Path:
    """Write a single sequence to a FASTA file.
    
    Args:
        sequence_id: The sequence identifier
        sequence: The amino acid sequence
        output_path: Path where to write the FASTA file
        
    Returns:
        Path to the written FASTA file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f">{sequence_id}\n")
        # Write sequence in 80-character lines (standard FASTA format)
        for i in range(0, len(sequence), 80):
            f.write(sequence[i:i+80] + "\n")
    
    return output_path


def read_multiple_fasta_sequences(fasta_path: Path, sequence_ids: List[str]) -> List[Tuple[str, str]]:
    """Read multiple sequences from a FASTA file by their IDs.
    
    Args:
        fasta_path: Path to the FASTA file
        sequence_ids: List of sequence IDs to extract
        
    Returns:
        List of tuples (sequence_id, sequence)
    """
    results = []
    for seq_id in sequence_ids:
        try:
            result = read_fasta_sequence_by_id(fasta_path, seq_id)
            results.append(result)
        except Exception as e:
            raise click.ClickException(f"Failed to find sequence '{seq_id}': {e}")
    return results


def read_fasta_with_selection(fasta_path: Path, sequence_id: str = None) -> Tuple[str, str]:
    """Read a FASTA file, either single sequence or select by ID from multi-sequence.
    
    Args:
        fasta_path: Path to the FASTA file
        sequence_id: Optional sequence ID to select from multi-sequence file
        
    Returns:
        Tuple of (sequence_id, sequence)
    """
    recs = list(SeqIO.parse(str(fasta_path), "fasta"))
    
    if not recs:
        raise click.ClickException(f"No sequences found in {fasta_path}")
    
    if len(recs) == 1:
        # Single sequence file
        if sequence_id is not None:
            click.echo(f"[info] Single sequence file, ignoring --sequence-id '{sequence_id}'")
        rec = recs[0]
        seq = str(rec.seq).replace("\n", "").replace("\r", "")
        return rec.id, seq
    
    # Multi-sequence file
    if sequence_id is None:
        available_ids = [rec.id for rec in recs]
        raise click.ClickException(
            f"Multi-sequence FASTA detected ({len(recs)} sequences). "
            f"Please specify --sequence-id. Available sequences: {', '.join(available_ids)}"
        )
    
    return read_fasta_sequence_by_id(fasta_path, sequence_id)


def sanitize_fasta_for_pmi(fasta_in: Path, fasta_out: Path) -> Tuple[List[str], int]:
    """Write a sanitized FASTA for PMI.
    - Replaces any non-standard AA with 'X' (except ':' which is treated as a chain separator).
    - If ':' is present in the input sequence, split into two sequences (homodimer),
      naming them with '_A' and '_B' suffixes.
    Returns: (list of record IDs written, number of replacements made)
    """
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    
    recs = list(SeqIO.parse(str(fasta_in), "fasta"))
    if len(recs) != 1:
        raise click.ClickException("FASTA must contain exactly one sequence for sanitization")
    
    rec = recs[0]
    seq = str(rec.seq).replace("\n", "").replace("\r", "")
    base_id = str(rec.id)

    # Guard: if base_id already ends with a chain suffix (_A or _B), strip it
    # to avoid generating IDs like "<id>_A_A" when splitting multimers.
    if base_id.endswith("_A") or base_id.endswith("_B"):
        base_id = base_id[:-2]
    
    # Check for chain separator
    if ":" in seq:
        # Split into two chains
        parts = seq.split(":", 1)
        if len(parts) != 2:
            raise click.ClickException("Expected exactly one ':' separator for homodimer")
        seq_a, seq_b = parts
        
        # Sanitize each chain
        clean_a, n_rep_a = "", 0
        for c in seq_a.upper():
            if c in standard_aa:
                clean_a += c
            else:
                clean_a += "X"
                n_rep_a += 1
        
        clean_b, n_rep_b = "", 0
        for c in seq_b.upper():
            if c in standard_aa:
                clean_b += c
            else:
                clean_b += "X"
                n_rep_b += 1
        
        # Write two records
        with open(fasta_out, "w") as fh:
            fh.write(f">{base_id}_A\n{clean_a}\n")
            fh.write(f">{base_id}_B\n{clean_b}\n")
        
        return [f"{base_id}_A", f"{base_id}_B"], n_rep_a + n_rep_b
    
    else:
        # Single chain
        clean_seq, n_rep = "", 0
        for c in seq.upper():
            if c in standard_aa:
                clean_seq += c
            else:
                clean_seq += "X"
                n_rep += 1
        
        with open(fasta_out, "w") as fh:
            fh.write(f">{base_id}\n{clean_seq}\n")
        
        return [base_id], n_rep


def extract_sequences_from_structure(structure_path: Path) -> dict:
    """Extract per-chain amino acid sequences from a PDB or mmCIF file.

    Returns a dict mapping chain ID -> sequence (string of one-letter AAs).
    """
    parser = None
    sp = str(structure_path)
    if sp.lower().endswith(('.cif', '.mmcif')):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure("struct", sp)
    ppb = PPBuilder()
    seqs: dict[str, str] = {}
    # Iterate over models and chains; concatenate polypeptides per chain
    for model in structure:
        for chain in model:
            polypeptides = ppb.build_peptides(chain)
            if not polypeptides:
                continue
            seq = "".join(str(pp.get_sequence()) for pp in polypeptides)
            if seq:
                seqs[str(chain.id)] = seq
        break  # only first model
    return seqs


def write_fasta_from_pdb(af_pdb: Path, out_fasta: Path) -> Path:
    """Create a single-record FASTA from a PDB/MMCIF by merging chains with ':'.

    Header is taken from the PDB stem (without extension). If multiple chains are
    present, their sequences are joined in alphabetical chain order with ':' as a separator.
    """
    seqs_by_chain = extract_sequences_from_structure(af_pdb)
    if not seqs_by_chain:
        raise click.ClickException(f"No sequences could be extracted from structure: {af_pdb}")
    header = af_pdb.stem
    merged_seq = ":".join(seqs_by_chain[k] for k in sorted(seqs_by_chain.keys()))
    write_single_sequence_fasta(header, merged_seq, out_fasta)
    return out_fasta

def existing_ranked_pdb(models_dir: Path) -> Path | None:
    """Find the best ranked PDB file in a directory."""
    if not models_dir.exists():
        return None
    cands = sorted(list(models_dir.glob("*rank_1*.pdb")))
    if not cands:
        cands = sorted(list(models_dir.glob("*.pdb")))
    return cands[0] if cands else None


def has_segments_and_top(out_dir: Path) -> bool:
    """Check if segmentation outputs exist."""
    return (out_dir / "segments.json").exists() and (out_dir / "fusion.top.dat").exists()


def has_sampling_outputs(out_dir: Path) -> bool:
    """Check if sampling outputs exist."""
    d = out_dir / "out_linkers"
    return d.exists() and any(d.iterdir())


def run_colabfold(
    fasta: Path,
    out_dir: Path,
    extra_args: str = "--num-models 1",
    gpu: int | None = None,
) -> Path:
    """Run ColabFold and return the path to the best PDB.
    
    Looks for colabfold_batch in the following order:
    1. COLABFOLD_PATH environment variable
    2. System PATH
    
    Args:
        fasta: Path to input FASTA file
        out_dir: Directory to write output files
        extra_args: Additional arguments to pass to colabfold_batch
        gpu: GPU device ID to use (sets CUDA_VISIBLE_DEVICES)
        
    Returns:
        Path to the best ranked PDB file
    """
    import shlex
    import subprocess
    import os
    from pathlib import Path
    
    # Diagnostic logging
    import traceback
    print(f"[DEBUG] run_colabfold called from:")
    traceback.print_stack()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get colabfold_batch path from environment or use default
    colabfold_path = os.environ.get('COLABFOLD_PATH')
    if colabfold_path:
        colabfold_bin = str(Path(colabfold_path) / 'colabfold_batch')
        if not os.path.exists(colabfold_bin):
            raise click.ClickException(
                f"colabfold_batch not found at {colabfold_bin}. "
                f"Please check COLABFOLD_PATH environment variable."
            )
        click.echo(f"[colabfold] Using ColabFold from: {colabfold_path}")
    else:
        colabfold_bin = "colabfold_batch"
        click.echo("[colabfold] COLABFOLD_PATH not set, looking for colabfold_batch in PATH")
    
    # Build command
    cmd_parts = [colabfold_bin]
    if extra_args.strip():
        cmd_parts.extend(shlex.split(extra_args.strip()))
    cmd_parts.extend([str(fasta), str(out_dir)])
    
    # Set up environment
    env = os.environ.copy()
    
    # Set up GPU selection if specified
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        click.echo(f"[colabfold] Using GPU {gpu} (CUDA_VISIBLE_DEVICES={gpu})")
    
    # Add ColabFold path to PATH if specified
    if colabfold_path:
        env_path = env.get('PATH', '')
        if colabfold_path not in env_path:
            env['PATH'] = f"{colabfold_path}:{env_path}"
    
    click.echo(f"[colabfold] Running: {' '.join(cmd_parts)}")
    
    try:
        result = subprocess.run(
            cmd_parts,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(Path.cwd()),
            env=env
        )
        if result.stdout:
            click.echo(f"[colabfold] stdout: {result.stdout}")
        if result.stderr:
            click.echo(f"[colabfold] stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        error_msg = f"colabfold_batch failed with code {e.returncode}"
        if e.stderr:
            error_msg += f"\nError output:\n{e.stderr}"
        if e.stdout:
            error_msg += f"\nOutput:\n{e.stdout}"
        raise click.ClickException(error_msg)
    except FileNotFoundError:
        raise click.ClickException(
            "colabfold_batch not found. Either:\n"
            "1. Install ColabFold and add it to your PATH, or\n"
            "2. Set the COLABFOLD_PATH environment variable to the directory containing colabfold_batch"
        )
    
    # Find the best PDB
    candidates = existing_ranked_pdb(out_dir)
    if candidates is None:
        raise FileNotFoundError("No PDB produced by colabfold_batch")
    
    click.echo(f"[colabfold] Using PDB: {candidates}")
    return candidates
