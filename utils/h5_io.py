from __future__ import annotations
import os
import hashlib
from typing import Any
import h5py
import numpy as np

DHW_RAS_AFFINE = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def require_h5_path(path) -> str:
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError(f"Expected an .h5 data path, got {path!r}")
    raw = os.fspath(path).strip()
    if not raw.lower().endswith(".h5"):
        raise ValueError(
            f"Only .h5 data inputs are supported: {raw!r}. Regenerate the manifest from the prepared H5 tree."
        )
    return os.path.normpath(raw)


def resolve_sample_h5_path(item: Any, *, h5_root: str | None = None) -> str:
    path = item.get("path", item.get("image")) if isinstance(item, dict) else item
    path = require_h5_path(path)
    if os.path.isabs(path) or not h5_root:
        return path
    return os.path.normpath(os.path.join(os.fspath(h5_root), path))


def center_crop_pad(arr: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    if arr.ndim != 3 or len(target) != 3 or any((int(t) <= 0 for t in target)):
        raise ValueError(
            f"Expected a 3D volume and positive DHW target: {arr.shape}, {target}"
        )
    out = np.zeros(target, dtype=arr.dtype)
    src, dst = ([], [])
    for a, t in zip(arr.shape, target):
        if a >= t:
            s = (a - t) // 2
            src.append(slice(s, s + t))
            dst.append(slice(0, t))
        else:
            d = (t - a) // 2
            src.append(slice(0, a))
            dst.append(slice(d, d + a))
    out[tuple(dst)] = arr[tuple(src)]
    return out


def read_h5_volume(path, *, mode: str = "iso1mm", spatial_size=None) -> np.ndarray:
    path = require_h5_path(path)
    if mode not in ("iso1mm", "native"):
        raise ValueError(f"Expected iso1mm or native H5 mode, got {mode!r}")
    key = f"image/{mode}"
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"{path}: missing {key}")
        arr = np.asarray(f[key])
    if arr.ndim != 3:
        raise ValueError(f"{path}: {key} must be DHW, got {arr.shape}")
    return (
        center_crop_pad(arr, tuple(spatial_size)) if spatial_size is not None else arr
    )


def read_h5_brain_mask(
    path, *, mode: str = "iso1mm", spatial_size=None, require_nonempty: bool = True
) -> np.ndarray:
    path = require_h5_path(path)
    if mode not in ("iso1mm", "native"):
        raise ValueError(f"Expected iso1mm or native H5 mode, got {mode!r}")
    image_key = f"image/{mode}"
    mask_key = f"brain_mask/{mode}"
    with h5py.File(path, "r") as f:
        if image_key not in f:
            raise KeyError(f"{path}: missing {image_key}")
        if mask_key not in f:
            raise KeyError(f"{path}: missing {mask_key}")
        image_shape = tuple(f[image_key].shape)
        mask_ds = f[mask_key]
        if len(image_shape) != 3:
            raise ValueError(f"{path}: {image_key} must be DHW, got {image_shape}")
        if tuple(mask_ds.shape) != image_shape:
            raise ValueError(
                f"{path}: {mask_key} shape {tuple(mask_ds.shape)} does not match {image_key} shape {image_shape}"
            )
        mask = np.asarray(mask_ds)
    if mask.ndim != 3:
        raise ValueError(f"{path}: {mask_key} must be DHW, got {mask.shape}")
    if not (
        np.issubdtype(mask.dtype, np.bool_) or np.issubdtype(mask.dtype, np.integer)
    ):
        raise ValueError(
            f"{path}: {mask_key} must be a binary integer array, got {mask.dtype}"
        )
    values = np.unique(mask)
    if not np.isin(values, (0, 1)).all():
        raise ValueError(
            f"{path}: {mask_key} must contain only 0/1, got {values.tolist()}"
        )
    mask = mask.astype(bool, copy=False)
    if require_nonempty and (not mask.any()):
        raise ValueError(f"{path}: {mask_key} is empty")
    if spatial_size is not None:
        mask = center_crop_pad(mask, tuple(spatial_size))
        if require_nonempty and (not mask.any()):
            raise ValueError(f"{path}: {mask_key} is empty after center crop/pad")
    return mask


def write_h5_volume(path, volume: np.ndarray, *, affine=None, mode="iso1mm") -> None:
    path = require_h5_path(path)
    volume = np.asarray(volume)
    if volume.ndim != 3 or mode not in ("iso1mm", "native"):
        raise ValueError(
            f"Expected DHW volume and iso1mm/native mode: {volume.shape}, {mode}"
        )
    if affine is not None:
        affine = np.asarray(affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError("affine must be a finite 4x4 matrix")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(f"image/{mode}", data=volume)
        f.attrs["array_order"] = "DHW"
        if affine is not None:
            f.create_dataset(f"meta/{mode}/affine", data=affine)


LATENT_CACHE_VERSION = 1
LATENT_GROUP = "latent"
LATENT_MODES = ("iso1mm", "native")
MASK_LATENT_GROUP = "latent/mask"


def latent_size_tag(spatial_size) -> str:
    d, h, w = (int(x) for x in spatial_size)
    return f"{d}x{h}x{w}"


def resolve_latent_paths(f, mode: str, expected_spatial_size):
    if LATENT_GROUP not in f:
        return (None, None)
    tag = latent_size_tag(expected_spatial_size)
    sized = f"{LATENT_GROUP}/{mode}/{tag}"
    if f"{sized}/mean" in f:
        return (sized, sized)
    legacy = f"{LATENT_GROUP}/{mode}"
    if f"{legacy}/mean" in f:
        return (legacy, LATENT_GROUP)
    return (None, None)


def resolve_mask_latent_paths(f, expected_spatial_size):
    tag = latent_size_tag(expected_spatial_size)
    sized = f"{MASK_LATENT_GROUP}/{tag}"
    if f"{sized}/mean" in f:
        return sized
    return None


def vae_ckpt_fingerprint(ckpt_path: str) -> str:
    sidecar = ckpt_path + ".fingerprint"
    if os.path.isfile(sidecar):
        with open(sidecar, "r", encoding="ascii") as f:
            value = f.read().strip()
        if len(value) == 40 and all((c in "0123456789abcdef" for c in value.lower())):
            return value.lower()
        raise ValueError(f"invalid VAE fingerprint sidecar: {sidecar}")
    st = os.stat(ckpt_path)
    raw = f"{os.path.abspath(ckpt_path)}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def validate_latent_attrs(
    f: "h5py.File",
    h5_path: str,
    *,
    attrs_path: str,
    expected_fingerprint: str,
    expected_spatial_size,
    expected_latent_shape,
) -> None:
    if attrs_path not in f:
        raise KeyError(f"{h5_path}: missing latent group '{attrs_path}'.")
    g = f[attrs_path]
    ver = int(g.attrs.get("cache_version", -1))
    if ver != LATENT_CACHE_VERSION:
        raise ValueError(
            f"{h5_path}: latent cache_version={ver} != {LATENT_CACHE_VERSION}"
        )
    fp = str(g.attrs.get("vae_ckpt_fingerprint", ""))
    if fp != expected_fingerprint:
        raise ValueError(
            f"{h5_path}: latent VAE fingerprint mismatch ({fp!r} != {expected_fingerprint!r})"
        )
    sp = tuple((int(x) for x in g.attrs.get("spatial_size", ())))
    if sp != tuple((int(x) for x in expected_spatial_size)):
        raise ValueError(f"{h5_path}: spatial_size={sp} != {expected_spatial_size}")
    ls = tuple((int(x) for x in g.attrs.get("latent_shape", ())))
    if ls != tuple((int(x) for x in expected_latent_shape)):
        raise ValueError(f"{h5_path}: latent_shape={ls} != {expected_latent_shape}")


def h5_axis_spacing(f, mode: str, axis: int):
    if int(axis) not in (0, 1, 2):
        raise ValueError(f"axis must be in 0..2, got {axis}")
    if f"meta/{mode}/spacing" in f:
        value = float(f[f"meta/{mode}/spacing"][axis])
    elif f"meta/{mode}/affine" in f:
        value = float(np.linalg.norm(f[f"meta/{mode}/affine"][:3, axis]))
    else:
        return 1.0 if mode == "iso1mm" else None
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {mode} spacing on axis {axis}: {value}")
    return value
