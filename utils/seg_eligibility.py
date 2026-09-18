from __future__ import annotations

SEG_HEAD_ELIGIBILITY: dict[str, tuple[str, str]] = {
    "modality_only": ("train_aux", "target label exists → aux supervision OK"),
    "mask_guide": (
        "train_aux",
        "target MRI label OK; CAUTION: head sees the fed mask — adherence needs a mask-free evaluator, not the head",
    ),
    "inpaint": (
        "train_aux",
        "original-image target label OK; health-hole vs lesion scored separately",
    ),
    "sr": (
        "train_aux",
        "target label reliable & space-matched (restricted by target quality)",
    ),
    "missing": (
        "train_aux",
        "target is re-selected; check label ↔ target identity first",
    ),
    "whole_brain": (
        "train_aux",
        "complete-target label OK; score observed vs completed regions separately",
    ),
    "deblur": ("train_aux", "pre-degradation target label OK & co-spatial"),
    "dealias": ("train_aux", "same as deblur"),
    "motion": ("train_aux", "same as deblur"),
    "seg": (
        "evaluation_forbidden",
        "generative seg IS the output; never aux-supervise / evaluate via the head — Dice uses the decoded label vs full-res GT",
    ),
}
DERIVED_MODE_USE: dict[str, tuple[str, str]] = {
    "inpaint_tumor": (
        "sampler_diagnostic",
        "pseudo-healthy: no paired healthy-tissue GT — original-lesion Dice is NOT a success metric",
    ),
    "inpaint_lesion": (
        "train_aux",
        "original lesion GT pairs with the regenerated lesion",
    ),
    "sr_native_thick": ("sampler_diagnostic", "native thick input, no paired thin GT"),
}


def seg_head_use(task: str, mode: str | None = None) -> tuple[str, str]:
    if task in {"seg_1mod", "seg_2mod", "seg_3mod", "seg_4mod"}:
        task = "seg"
    base = SEG_HEAD_ELIGIBILITY.get(task, ("evaluation_forbidden", "unknown task"))
    if base[0] == "evaluation_forbidden":
        return base
    return DERIVED_MODE_USE.get(mode, base)


def seg_head_readout_allowed(task: str, mode: str | None = None) -> bool:
    return seg_head_use(task, mode)[0] != "evaluation_forbidden"
