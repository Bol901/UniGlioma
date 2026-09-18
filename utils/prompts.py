"""Shared task instruction composition for training, caching and inference."""

from __future__ import annotations
import json
import os
from pathlib import Path
from functools import lru_cache
from utils.prompt_variants import (
    IMPERATIVE_SHELLS,
    INPAINT_TEMPLATES,
    MASK_GUIDE_TEMPLATES,
    MISSING_TEMPLATES,
    MODALITY_DESC_BUCKETS,
    SR_TEMPLATES,
)
from utils.tasks import CANONICAL_TASK_SET


def acq_stem(path) -> str:
    return os.path.splitext(os.path.basename(str(path)))[0]


QUALITY_PHRASE = {
    "thin": "thin-slice isotropic",
    "medium": "standard-resolution",
    "thick": "thick-slice",
    "unknown": "",
}
QUALITIES = tuple(QUALITY_PHRASE)
VIEWS = ("axial", "sagittal", "coronal", "unknown")
MODALITY_NAME = {0: "T1", 1: "T1c", 2: "T2", 3: "FLAIR"}
DEFAULT_QUALITY = "thin"
DEFAULT_VIEW = "axial"
N_DESC_VARIANTS = 5
WHOLE_BRAIN_TEMPLATES = tuple(INPAINT_TEMPLATES["whole_brain"])
MOTION_TEMPLATES = (
    "Generate the {desc} with motion artifacts corrected from the degraded input.",
    "Reconstruct the {desc} by removing patient-motion artifacts from the reference.",
    "Produce the {desc} from the motion-corrupted acquisition.",
    "Restore the {desc} with rigid-motion ghosting and blur removed.",
    "Create a motion-corrected {desc} from the degraded MRI input.",
    "Recover the {desc} from the reference affected by acquisition motion.",
)
_SR_SOURCE_TPL = {
    "sr": (
        "Generate the {desc} from the {input_desc}.",
        "Reconstruct the {desc} from the {input_desc}.",
        "Produce the {desc} using the {input_desc} as reference.",
    ),
    "deblur": (
        "Generate the {desc} with blur removed from the {input_desc}.",
        "Reconstruct the {desc} by deblurring the {input_desc}.",
        "Produce the {desc} with sharp detail restored from the blurred {input_desc}.",
    ),
    "dealias": (
        "Generate the {desc} free of aliasing from the {input_desc}.",
        "Reconstruct the {desc} by de-aliasing the {input_desc}.",
        "Produce the {desc} without fold-over artifacts from the {input_desc}.",
    ),
}
SYNTHETIC_THICK_INPUT_TEMPLATES = _SR_SOURCE_TPL
NATIVE_THICK_INPUT_TEMPLATES = _SR_SOURCE_TPL
N_TASK_VARIANTS = max(
    len(IMPERATIVE_SHELLS),
    len(MASK_GUIDE_TEMPLATES),
    len(SR_TEMPLATES["default"]),
    len(MISSING_TEMPLATES),
    len(WHOLE_BRAIN_TEMPLATES),
    len(MOTION_TEMPLATES),
)
TASK_TEMPLATES = {
    "mask_guide": MASK_GUIDE_TEMPLATES,
    "sr": SR_TEMPLATES["default"],
    "missing": MISSING_TEMPLATES,
    "whole_brain": WHOLE_BRAIN_TEMPLATES,
    "deblur": SR_TEMPLATES["deblur"],
    "dealias": SR_TEMPLATES["dealias"],
    "motion": MOTION_TEMPLATES,
}


def _pool_v0(pool):
    return pool["default"][0] if isinstance(pool, dict) else pool[0]


PARAM_SR_TASKS = ("sr", "deblur", "dealias")


def slotted_variant(task: str) -> int:
    for i, t in enumerate(TASK_TEMPLATES.get(str(task), ())):
        if "{in_mm}" in t or "{out_mm}" in t:
            return i
    return 0


TASK_TEMPLATE = {t: _pool_v0(pool) for t, pool in TASK_TEMPLATES.items()}
TASK_TEMPLATE["inpaint"] = INPAINT_TEMPLATES["healthy"][0]
_TASKLESS = ("modality_only",)
INPAINT_CONTENT_TEMPLATES = {
    key: INPAINT_TEMPLATES[key] for key in ("healthy", "tumor", "cavity")
}
INPAINT_CONTENTS = tuple(INPAINT_CONTENT_TEMPLATES)
DEFAULT_SEG_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "assets/seg_prompts.json"
)


def get_seg_prompt_variants(path=None) -> dict[tuple[int, ...], tuple[str, ...]]:
    return _load_seg_prompt_variants(
        str(Path(path or DEFAULT_SEG_PROMPT_PATH).resolve())
    )


@lru_cache(maxsize=8)
def _load_seg_prompt_variants(filename):
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(
            f"seg prompt catalog missing: {path}. Set seg_prompt_variants_path in YAML to the original 15-combination catalog used to build the text cache."
        )
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    out = {}
    for row in payload.get("combinations", []):
        key = tuple(sorted((int(x) for x in row["modality_ids"])))
        prompts = tuple((str(x) for x in row["prompts"]))
        if (
            not 1 <= len(key) <= 4
            or len(set(key)) != len(key)
            or (not set(key) <= {0, 1, 2, 3})
            or (len(prompts) != 6)
            or any((not x.strip() for x in prompts))
            or (key in out)
        ):
            raise ValueError(f"invalid seg prompt entry for {key} in {path}")
        out[key] = prompts
    if len(out) != 15:
        raise ValueError(f"expected 15 non-empty seg modality combinations in {path}")
    return out


def seg_prompt(modality_ids, variant: int = 0, *, catalog_path=None) -> str:
    key = tuple(sorted({int(x) for x in modality_ids}))
    catalog = get_seg_prompt_variants(catalog_path)
    if key not in catalog:
        raise ValueError(
            f"seg requires a non-empty subset of image modalities 0..3; got {key}"
        )
    prompts = catalog[key]
    return prompts[int(variant) % len(prompts)]


def n_inpaint_variants(content: str = "healthy") -> int:
    return len(
        INPAINT_CONTENT_TEMPLATES.get(
            str(content), INPAINT_CONTENT_TEMPLATES["healthy"]
        )
    )


def _norm_quality(quality) -> str:
    q = str(quality).lower()
    return q if q in QUALITY_PHRASE else "unknown"


def _norm_view(view) -> str:
    v = str(view).lower()
    return v if v in VIEWS else "unknown"


def _clean_mm(value):
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not x > 0:
        return None
    return int(x) if x.is_integer() else f"{x:g}"


def descriptor(
    modality, quality, view, variant: int = 0, *, target_mm=None, thickness_mm=None
) -> str:
    name = MODALITY_NAME.get(int(modality), MODALITY_NAME[3])
    q = _norm_quality(quality)
    v = _norm_view(view)
    mm = _clean_mm(target_mm)
    view_phrase = v if v != "unknown" else ""
    if q == "thin" and mm == 1:
        return " ".join(
            (x for x in ("1mm thin-slice isotropic", view_phrase, name, "MRI") if x)
        )
    if q == "thick" and mm in SR_THICK_GRID:
        return " ".join(
            (x for x in (f"{mm} mm thick-slice", view_phrase, name, "MRI") if x)
        )
    bucket = MODALITY_DESC_BUCKETS.get(name, {}).get(f"{q}:{v}")
    if bucket:
        return bucket[int(variant) % len(bucket)]
    return " ".join(
        (x for x in (QUALITY_PHRASE[q], v if v != "unknown" else "", name, "MRI") if x)
    )


def compose(
    task,
    modality,
    quality,
    view,
    *,
    inpaint_content=None,
    variant: int = 0,
    desc_variant: int = 0,
    shell: int | None = None,
    sr_in_mm=None,
    sr_out_mm=None,
    target_mm=None,
    thickness_mm=None,
    synthetic_input_mm=None,
    synthetic_input_generic: bool = False,
    native_input_mm=None,
    native_input_generic: bool = False,
    seg_modalities=None,
    seg_catalog_path=None,
    sr_method: str = "nearest-neighbor",
) -> str:
    task = str(task)
    if task not in CANONICAL_TASK_SET:
        raise ValueError(f"unknown canonical task {task!r}")
    if task == "seg":
        return seg_prompt(
            seg_modalities or (), variant=variant, catalog_path=seg_catalog_path
        )
    base = descriptor(
        modality,
        quality,
        view,
        variant=desc_variant,
        target_mm=target_mm,
        thickness_mm=thickness_mm,
    )
    if task == "inpaint":
        content = str(inpaint_content or "healthy")
        tmpls = (
            INPAINT_CONTENT_TEMPLATES.get(content)
            or INPAINT_CONTENT_TEMPLATES["healthy"]
        )
        return tmpls[int(variant) % len(tmpls)].format(desc=base)
    if task in PARAM_SR_TASKS and (
        synthetic_input_mm is not None or synthetic_input_generic
    ):
        mm = _clean_mm(synthetic_input_mm)
        input_desc = (
            f"{mm} mm thick-slice input" if mm is not None else "thick-slice input"
        )
        pool = SYNTHETIC_THICK_INPUT_TEMPLATES[task]
        return pool[int(variant) % len(pool)].format(desc=base, input_desc=input_desc)
    if task in PARAM_SR_TASKS and (native_input_mm is not None or native_input_generic):
        mm = _clean_mm(native_input_mm)
        input_desc = (
            f"{mm} mm thick-slice input" if mm is not None else "thick-slice input"
        )
        pool = NATIVE_THICK_INPUT_TEMPLATES[task]
        return pool[int(variant) % len(pool)].format(desc=base, input_desc=input_desc)
    tmpl_pool = TASK_TEMPLATES.get(task)
    if tmpl_pool:
        if task in PARAM_SR_TASKS:
            t = tmpl_pool[int(variant) % len(tmpl_pool)]
            fill = {"desc": base}
            if "{in_mm}" in t or "{out_mm}" in t:
                fill["in_mm"] = sr_in_mm if sr_in_mm is not None else ""
                fill["out_mm"] = sr_out_mm if sr_out_mm is not None else ""
            if "{method_appearance}" in t:
                fill["method_appearance"] = method_appearance(sr_method)
            return t.format(**fill)
        return tmpl_pool[int(variant) % len(tmpl_pool)].format(desc=base)
    if shell is not None:
        return IMPERATIVE_SHELLS[int(shell) % len(IMPERATIVE_SHELLS)].format(desc=base)
    return base


SR_THICK_GRID = (2, 3, 4, 5, 6, 7, 8, 9, 10)
SR_MM_GRID = tuple(((out, in_mm) for in_mm in range(2, 11) for out in range(1, in_mm)))
SR_METHODS = (
    "nearest-neighbor",
    "bilinear-interpolated",
    "bicubic-interpolated",
    "random-affine",
)


def method_appearance(method: str = "nearest-neighbor") -> str:
    m = str(method or "nearest-neighbor").strip().lower()
    return {
        "nearest": "with a nearest-neighbor resampled appearance",
        "nearest-neighbor": "with a nearest-neighbor resampled appearance",
        "nn": "with a nearest-neighbor resampled appearance",
        "bilinear-interpolated": "with a bilinear-interpolated resampled appearance",
        "bicubic-interpolated": "with a bicubic-interpolated resampled appearance",
        "bilinear": "with a bilinear-interpolated resampled appearance",
        "bicubic": "with a bicubic-interpolated resampled appearance",
        "gaussian-blur": "with a gaussian-smoothed appearance",
        "gaussian": "with a gaussian-smoothed appearance",
        "affine": "with an obliquely thick-sliced appearance",
        "random-affine": "with an obliquely thick-sliced appearance",
    }.get(m, f"with a {m} resampled appearance")


def n_desc_variants() -> int:
    return N_DESC_VARIANTS


def n_task_variants(task: str, content: str | None = None) -> int:
    if str(task) == "inpaint":
        return n_inpaint_variants(content or "healthy")
    pool = TASK_TEMPLATES.get(str(task))
    if pool:
        return len(pool)
    return 1


def n_synthetic_input_variants(task: str) -> int:
    return len(SYNTHETIC_THICK_INPUT_TEMPLATES.get(str(task), ()))


def n_native_input_variants(task: str) -> int:
    return len(NATIVE_THICK_INPUT_TEMPLATES.get(str(task), ()))


def enumerate_taskless_prompts(*, seg_catalog_path=None, include_seg=True):
    seen = set()
    tasks = ["modality_only", *TASK_TEMPLATES]
    for mod in MODALITY_NAME:
        for q in QUALITIES + ("unknown",):
            for v in VIEWS:
                for dv in range(N_DESC_VARIANTS):
                    for t in tasks:
                        if t == "modality_only":
                            s = compose(t, mod, q, v, desc_variant=dv)
                            if s not in seen:
                                seen.add(s)
                                yield s
                            for sh in range(len(IMPERATIVE_SHELLS)):
                                s = compose(t, mod, q, v, desc_variant=dv, shell=sh)
                                if s not in seen:
                                    seen.add(s)
                                    yield s
                        elif t in PARAM_SR_TASKS:
                            for tv, tmpl in enumerate(TASK_TEMPLATES[t]):
                                if "{in_mm}" in tmpl or "{out_mm}" in tmpl:
                                    if "{method_appearance}" in tmpl:
                                        for out_mm, in_mm in SR_MM_GRID:
                                            for m in SR_METHODS:
                                                s = compose(
                                                    t,
                                                    mod,
                                                    q,
                                                    v,
                                                    desc_variant=dv,
                                                    variant=tv,
                                                    sr_in_mm=in_mm,
                                                    sr_out_mm=out_mm,
                                                    sr_method=m,
                                                )
                                                if s not in seen:
                                                    seen.add(s)
                                                    yield s
                                    else:
                                        for out_mm, in_mm in SR_MM_GRID:
                                            s = compose(
                                                t,
                                                mod,
                                                q,
                                                v,
                                                desc_variant=dv,
                                                variant=tv,
                                                sr_in_mm=in_mm,
                                                sr_out_mm=out_mm,
                                            )
                                            if s not in seen:
                                                seen.add(s)
                                                yield s
                                else:
                                    s = compose(
                                        t, mod, q, v, desc_variant=dv, variant=tv
                                    )
                                    if s not in seen:
                                        seen.add(s)
                                        yield s
                        else:
                            for tv in range(len(TASK_TEMPLATES[t])):
                                s = compose(t, mod, q, v, desc_variant=dv, variant=tv)
                                if s not in seen:
                                    seen.add(s)
                                    yield s
                    for content in INPAINT_CONTENTS:
                        for var in range(len(INPAINT_CONTENT_TEMPLATES[content])):
                            s = compose(
                                "inpaint",
                                mod,
                                q,
                                v,
                                inpaint_content=content,
                                variant=var,
                                desc_variant=dv,
                            )
                            if s not in seen:
                                seen.add(s)
                                yield s
    for mod in MODALITY_NAME:
        for v in VIEWS:
            for t in tasks:
                variants = (
                    range(len(TASK_TEMPLATES[t])) if t in TASK_TEMPLATES else (0,)
                )
                for tv in variants:
                    if t == "modality_only":
                        candidates = [compose(t, mod, "thin", v, target_mm=1)]
                        candidates.extend(
                            (
                                compose(t, mod, "thin", v, target_mm=1, shell=sh)
                                for sh in range(len(IMPERATIVE_SHELLS))
                            )
                        )
                    else:
                        candidates = [
                            compose(t, mod, "thin", v, target_mm=1, variant=tv)
                        ]
                    for s in candidates:
                        if s not in seen:
                            seen.add(s)
                            yield s
            for content in INPAINT_CONTENTS:
                for var in range(len(INPAINT_CONTENT_TEMPLATES[content])):
                    s = compose(
                        "inpaint",
                        mod,
                        "thin",
                        v,
                        target_mm=1,
                        inpaint_content=content,
                        variant=var,
                    )
                    if s not in seen:
                        seen.add(s)
                        yield s
            for task in PARAM_SR_TASKS:
                for target_mm in (None, 1):
                    for q_syn in QUALITIES + ("unknown",):
                        for generic in (False, True):
                            for dv in range(N_DESC_VARIANTS):
                                for in_mm in SR_THICK_GRID if not generic else (None,):
                                    for tv in range(n_synthetic_input_variants(task)):
                                        s = compose(
                                            task,
                                            mod,
                                            q_syn,
                                            v,
                                            desc_variant=dv,
                                            target_mm=target_mm,
                                            synthetic_input_mm=in_mm,
                                            synthetic_input_generic=generic,
                                            variant=tv,
                                        )
                                        if s not in seen:
                                            seen.add(s)
                                            yield s
                    for generic in (False, True):
                        for q_syn in QUALITIES + ("unknown",):
                            for dv in range(N_DESC_VARIANTS):
                                for in_mm in SR_THICK_GRID if not generic else (None,):
                                    for tv in range(n_native_input_variants(task)):
                                        s = compose(
                                            task,
                                            mod,
                                            q_syn,
                                            v,
                                            desc_variant=dv,
                                            target_mm=target_mm,
                                            native_input_mm=in_mm,
                                            native_input_generic=generic,
                                            variant=tv,
                                        )
                                        if s not in seen:
                                            seen.add(s)
                                            yield s
            for out_mm, in_mm in ((2, 3), (2, 5), (2, 7), (3, 5), (3, 7), (5, 7)):
                for dv in range(N_DESC_VARIANTS):
                    for tv in range(n_synthetic_input_variants("sr")):
                        s = compose(
                            "sr",
                            mod,
                            "thick",
                            v,
                            desc_variant=dv,
                            target_mm=out_mm,
                            synthetic_input_mm=in_mm,
                            variant=tv,
                        )
                        if s not in seen:
                            seen.add(s)
                            yield s
    for prompts in (
        get_seg_prompt_variants(seg_catalog_path).values() if include_seg else ()
    ):
        for s in prompts:
            if s not in seen:
                seen.add(s)
                yield s
