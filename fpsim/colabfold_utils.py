"""ColabFold argument processing utilities for fpsim.

Handles complex ColabFold flag composition and argument building.
"""
from __future__ import annotations
from typing import List
import shlex


def build_colabfold_args(
    base_args: str,
    num_models: int | None = None,
    num_ensemble: int | None = None,
    num_recycle: int | None = None,
    model_type: str | None = None,
    max_seq: int | None = None,
    max_extra_seq: int | None = None,
    msa_mode: str | None = None,
    pair_mode: str | None = None,
    no_templates: bool = False,
    use_gpu_relax: bool | None = None,
) -> str:
    """Build ColabFold arguments from base args and individual flags."""
    parts: List[str] = shlex.split(base_args) if base_args else []
    
    def add_flag(flag: str, val):
        if val is None:
            return
        if isinstance(val, bool):
            if flag in ["--no-templates", "--use-gpu-relax"]:
                # These are flags without values - only add if True
                if val:
                    parts.append(flag)
            else:
                # Other boolean flags that need True/False values
                parts.append(flag)
                parts.append("True" if val else "False")
        else:
            parts.append(flag)
            parts.append(str(val))
    
    add_flag("--num-models", num_models)
    add_flag("--num-ensemble", num_ensemble)
    add_flag("--num-recycle", num_recycle)
    add_flag("--model-type", model_type)
    add_flag("--max-seq", max_seq)
    add_flag("--max-extra-seq", max_extra_seq)
    add_flag("--msa-mode", msa_mode)
    add_flag("--pair-mode", pair_mode)
    add_flag("--no-templates", no_templates)
    add_flag("--use-gpu-relax", use_gpu_relax)
    
    return " ".join(shlex.quote(p) for p in parts)
