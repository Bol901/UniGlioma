from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping

CANONICAL_TASKS = (
    "modality_only",
    "mask_guide",
    "inpaint",
    "sr",
    "missing",
    "whole_brain",
    "deblur",
    "dealias",
    "motion",
    "seg",
)
CANONICAL_TASK_SET = frozenset(CANONICAL_TASKS)
LEGACY_TASK_NAMES = frozenset(("gen", "text", "text_gen", "mask", "single"))
LABEL_REQUIRED_TASKS = frozenset(("mask_guide", "inpaint", "seg"))
ONLINE_VAE_TASKS = frozenset(
    ("inpaint", "sr", "whole_brain", "deblur", "dealias", "motion")
)
IMAGE_TARGET_TASKS = frozenset(CANONICAL_TASK_SET - {"seg"})
NO_REF_TASKS = frozenset(("modality_only",))
SEG_TARGET_MODALITY_ID = 4
NULL_MODALITY_ID = 5
NUM_TARGET_MODALITIES = 5


def validate_task_weights(task_weights: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(task_weights, Mapping):
        raise TypeError("task_weights must be a mapping")
    unknown = sorted(set(map(str, task_weights)) - CANONICAL_TASK_SET)
    if unknown:
        hint = (
            "; legacy aliases are not accepted"
            if set(unknown) & LEGACY_TASK_NAMES
            else ""
        )
        raise ValueError(f"unknown task_weights keys: {unknown}{hint}")
    out = {str(k): float(v) for k, v in task_weights.items() if float(v) > 0.0}
    if not out:
        raise ValueError(
            "task_weights must contain at least one positive canonical task"
        )
    return out


def apply_curriculum_preset(
    cfg: Mapping[str, Any], preset: str | None = None
) -> dict[str, Any]:
    out = deepcopy(dict(cfg))
    curriculum = out.get("curriculum") or {}
    if not curriculum:
        out["task_weights"] = validate_task_weights(out.get("task_weights") or {})
        return out
    presets = curriculum.get("presets") or {}
    selected = str(preset or curriculum.get("default_preset") or "full")
    if selected not in presets:
        raise ValueError(
            f"unknown curriculum preset {selected!r}; available={sorted(presets)}"
        )
    spec = presets[selected]
    if not isinstance(spec, Mapping):
        raise TypeError(f"curriculum.presets.{selected} must be a mapping")
    out["task_weights"] = validate_task_weights(spec.get("task_weights") or {})
    states = spec.get("cfg_train_states") or {}
    missing = sorted(set(out["task_weights"]) - set(states))
    if missing:
        raise ValueError(
            f"curriculum preset {selected!r} lacks cfg_train_states for {missing}"
        )
    out["cfg_train_states"] = deepcopy(dict(states))
    if "aux_seg_supervision" in spec:
        out["aux_seg_supervision"] = bool(spec["aux_seg_supervision"])
    out["curriculum_preset"] = selected
    return out
