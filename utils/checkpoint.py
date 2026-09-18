"""Checkpoint compatibility helpers for the unified target-modality table."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re

import torch


def dit_milestone_paths(directory: Path) -> list[Path]:
    """Numbered training bundles only; exclude VAE, EMA exports, best, and temp files."""
    pattern = re.compile(r"(?:dit_)?ckpt_step_(\d+)\.pt\Z")
    paths = [p for p in directory.glob("*ckpt_step_*.pt")
             if p.is_file() and pattern.fullmatch(p.name)]
    return sorted(paths, key=lambda p: (int(p.stem.rsplit("_", 1)[-1]),
                                       p.name.startswith("dit_")), reverse=True)


def dit_resume_candidates(directory: Path, step="latest") -> list[Path]:
    """Prefer new DiT names; preserve legacy resume and corrupt-file fallback."""
    prefixes = ("dit_ckpt_step_", "ckpt_step_")
    if step not in ("", "latest", None):
        paths = [directory / f"{prefix}{int(step)}.pt" for prefix in prefixes]
        paths = [p for p in paths if p.is_file()]
    else:
        milestones = dit_milestone_paths(directory)
        paths = []
        # Once a run writes prefixed bundles, its legacy latest can be stale. Try
        # new latest and new milestones before falling back to the old family.
        for prefix in prefixes:
            latest = directory / f"{prefix}latest.pt"
            if latest.is_file():
                paths.append(latest)
            paths.extend(p for p in milestones if p.name.startswith(prefix))
    if not paths:
        raise FileNotFoundError(f"no DiT resume checkpoint for step={step!r} under {directory}")
    return list(dict.fromkeys(p.resolve() for p in paths))


def has_dit_checkpoint(directory: Path) -> bool:
    return any((directory / name).is_file() for name in
               ("dit_ckpt_step_latest.pt", "ckpt_step_latest.pt")) or bool(dit_milestone_paths(directory))


def adapt_unified_modality_embedding(
    state_dict: Mapping[str, torch.Tensor],
    target_state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], bool]:
    """Map legacy [T1,T1c,T2,FLAIR,null] embeddings to six unified rows.

    Unified order is [T1,T1c,T2,FLAIR,seg,null]. The new seg row uses the
    target model's initialization (zero in the migration contract); every other tensor
    remains strict-loadable.
    """
    out = dict(state_dict)
    key = "modality_emb.weight"
    if key not in out or key not in target_state_dict:
        return out, False
    source = out[key]
    target = target_state_dict[key]
    if tuple(source.shape) == tuple(target.shape):
        return out, False
    if source.ndim != 2 or target.ndim != 2 or source.shape[0] != 5 or target.shape[0] != 6:
        raise ValueError(
            f"unsupported modality embedding migration {tuple(source.shape)} -> {tuple(target.shape)}"
        )
    if source.shape[1] != target.shape[1]:
        raise ValueError("modality embedding width changed; semantic migration is unsafe")
    migrated = target.detach().cpu().clone().to(dtype=source.dtype)
    migrated.zero_()
    migrated[:4].copy_(source[:4])
    migrated[5].copy_(source[4])
    out[key] = migrated
    return out, True
