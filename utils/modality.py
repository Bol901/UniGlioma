from __future__ import annotations
import os
import re
from typing import Optional

DEFAULT_MODALITY_SUFFIX_MAP: dict[str, int] = {
    "_t1c": 1,
    "_t1ce": 1,
    "_t2": 2,
    "_fl": 3,
    "_flair": 3,
    "_t1": 0,
}
UNCOND_MODALITY_ID = -1
_SUFFIX_RE = re.compile("(_[a-z0-9]+)(?:\\.h5)?$", re.IGNORECASE)


def derive_modality_id(path: str, suffix_map: Optional[dict[str, int]] = None) -> int:
    suffix_map = suffix_map if suffix_map is not None else DEFAULT_MODALITY_SUFFIX_MAP
    name = os.path.basename(path).lower()
    name = re.sub("\\.h5$", "", name)
    keys_sorted = sorted(suffix_map.keys(), key=len, reverse=True)
    for k in keys_sorted:
        kl = k.lower()
        if name.endswith(kl):
            return int(suffix_map[k])
    m = _SUFFIX_RE.search(os.path.basename(path).lower())
    raise ValueError(
        f"Cannot infer modality from path={path!r} (detected suffix={(m.group(1) if m else None)}); known suffixes={list(suffix_map.keys())}"
    )


def case_key_from_path(path: str, suffix_map: Optional[dict[str, int]] = None) -> str:
    suffix_map = suffix_map if suffix_map is not None else DEFAULT_MODALITY_SUFFIX_MAP
    parent = os.path.dirname(path)
    base = os.path.basename(path)
    stem = re.sub("\\.h5$", "", base, flags=re.IGNORECASE)
    keys_sorted = sorted(suffix_map.keys(), key=len, reverse=True)
    stem_l = stem.lower()
    for k in keys_sorted:
        kl = k.lower()
        if stem_l.endswith(kl):
            stem = stem[: len(stem) - len(kl)]
            break
    return os.path.join(parent, stem) if parent else stem
