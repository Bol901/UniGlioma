from __future__ import annotations
import math
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.ndimage import (
    affine_transform,
    binary_dilation,
    uniform_filter1d,
    zoom,
)


def thick_slice_lowres(
    vol: np.ndarray, factor: float, axis: int, *, up_order: int = 0, on_low_grid=None
) -> np.ndarray:
    assert vol.ndim == 3, f"expected (D,H,W), got {vol.shape}"
    f = float(factor)
    if f <= 1.0:
        if on_low_grid is not None:
            lo0 = np.clip(vol.astype(np.float64), 0.0, 1.0)
            return np.clip(on_low_grid(lo0).astype(np.float64), 0.0, 1.0)
        return np.clip(vol.astype(np.float64), 0.0, 1.0)
    if int(axis) not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
    if int(up_order) not in (0, 1, 3):
        raise ValueError(f"up_order must be 0, 1, or 3; got {up_order}")
    k = max(1, int(round(f)))
    blurred = uniform_filter1d(
        vol.astype(np.float64), size=k, axis=axis, mode="nearest"
    )
    zd = [1.0, 1.0, 1.0]
    zd[axis] = 1.0 / f
    lo = zoom(blurred, zd, order=0)
    if on_low_grid is not None:
        lo = on_low_grid(lo)
    zu = [vol.shape[a] / lo.shape[a] for a in range(3)]
    up = zoom(lo, zu, order=int(up_order))
    if up.shape != vol.shape:
        fix = [1.0, 1.0, 1.0]
        for a in range(3):
            fix[a] = vol.shape[a] / up.shape[a]
        up = zoom(up, fix, order=int(up_order))
    return np.clip(up, 0.0, 1.0)


def _euler_rotation(ax: float, ay: float, az: float) -> np.ndarray:
    cx, sx = (np.cos(ax), np.sin(ax))
    cy, sy = (np.cos(ay), np.sin(ay))
    cz, sz = (np.cos(az), np.sin(az))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def random_affine_thick_lowres(
    vol: np.ndarray,
    factor: float,
    axis: int,
    *,
    max_angle_deg=15.0,
    rng: "np.random.Generator | None" = None,
    order: int = 1,
    up_order: int = 0,
    return_params: bool = False,
    on_low_grid=None,
):
    assert vol.ndim == 3, f"expected (D,H,W), got {vol.shape}"
    rng = rng if rng is not None else np.random.default_rng()
    if np.isscalar(max_angle_deg):
        ranges = [float(max_angle_deg)] * 3
    else:
        ranges = [float(x) for x in max_angle_deg]
        if len(ranges) != 3:
            raise ValueError("max_angle_deg must be a scalar or length-3 sequence")
    angles_deg = np.array([rng.uniform(-r, r) if r > 0 else 0.0 for r in ranges])
    if not np.any(np.abs(angles_deg) > 0):
        lr = thick_slice_lowres(
            vol, factor, axis, up_order=up_order, on_low_grid=on_low_grid
        )
        return (lr, angles_deg) if return_params else lr
    R = _euler_rotation(*np.deg2rad(angles_deg))
    c = (np.asarray(vol.shape, dtype=np.float64) - 1.0) / 2.0
    fwd = affine_transform(
        vol.astype(np.float64),
        R,
        offset=c - R @ c,
        order=order,
        mode="constant",
        cval=0.0,
    )
    lr_rot = thick_slice_lowres(
        fwd, factor, axis, up_order=up_order, on_low_grid=on_low_grid
    )
    Rinv = R.T
    lr = affine_transform(
        lr_rot, Rinv, offset=c - Rinv @ c, order=order, mode="constant", cval=0.0
    )
    lr = np.clip(lr, 0.0, 1.0)
    return (lr, angles_deg) if return_params else lr


def kspace_lowpass(vol: np.ndarray, factor: float, axis: int) -> np.ndarray:
    assert vol.ndim == 3, f"expected (D,H,W), got {vol.shape}"
    n = vol.shape[axis]
    keep = max(1, int(round(n / float(factor))))
    lo = (n - keep) // 2
    hi = lo + keep
    F = np.fft.fftshift(np.fft.fftn(vol))
    out_k = np.zeros_like(F)
    sl = [slice(None)] * 3
    sl[axis] = slice(lo, hi)
    out_k[tuple(sl)] = F[tuple(sl)]
    out = np.fft.ifftn(np.fft.ifftshift(out_k)).real
    return np.clip(out, 0.0, 1.0)


def gaussian_blur(
    vol: np.ndarray, sigma: float = 2.0, axis: int | None = None
) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    assert vol.ndim == 3
    if float(sigma) <= 0.0:
        raise ValueError(f"sigma must be > 0; got {sigma}")
    if axis is not None and int(axis) not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
    sigmas = (
        float(sigma)
        if axis is None
        else [float(sigma) if a == int(axis) else 0.0 for a in range(3)]
    )
    out = gaussian_filter(vol.astype(np.float64), sigma=sigmas)
    return np.clip(out, 0.0, 1.0)


def inplane_gaussian_blur_low(
    low: np.ndarray, sigma_voxel: float, axis: int
) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    if float(sigma_voxel) <= 0:
        return np.clip(low.astype(np.float64), 0.0, 1.0)
    sigmas = [0.0 if a == int(axis) else float(sigma_voxel) for a in range(3)]
    out = gaussian_filter(low.astype(np.float64), sigma=sigmas)
    return np.clip(out, 0.0, 1.0)


def uniform_undersample(vol: np.ndarray, factor: float, axis: int) -> np.ndarray:
    assert vol.ndim == 3
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
    if float(factor) < 2.0:
        raise ValueError(f"factor must be >= 2; got {factor}")
    n = vol.shape[axis]
    F = np.fft.fftshift(np.fft.fftn(vol))
    out_k = np.zeros_like(F)
    idx = np.arange(n // 2, n, max(1, int(factor)))
    idx = np.unique(np.concatenate([idx, np.arange(n // 2, -1, -max(1, int(factor)))]))
    idx = idx[(idx >= 0) & (idx < n)]
    sl = [slice(None)] * 3
    sl[axis] = idx
    out_k[tuple(sl)] = F[tuple(sl)]
    out = np.abs(np.fft.ifftn(np.fft.ifftshift(out_k)))
    return np.clip(out, 0.0, 1.0)


def single_axis_fov_crop(
    vol: np.ndarray,
    axis: int,
    keep_frac: float,
    *,
    side: str = "both",
    fill: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(vol)
    if arr.ndim != 3:
        raise ValueError(f"expected (D,H,W), got {arr.shape}")
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
    side = str(side).lower()
    if side not in ("both", "start", "end"):
        raise ValueError(f"side must be 'both', 'start' or 'end'; got {side!r}")
    keep_frac = float(keep_frac)
    if not 0.0 < keep_frac <= 1.0:
        raise ValueError(f"keep_frac must be in (0,1], got {keep_frac}")
    n = arr.shape[axis]
    keep = max(1, min(n, int(round(n * keep_frac))))
    if side == "both":
        lo = (n - keep) // 2
    elif side == "start":
        lo = 0
    else:
        lo = n - keep
    hi = lo + keep
    missing = np.ones(arr.shape, dtype=bool)
    slab = [slice(None)] * 3
    slab[axis] = slice(lo, hi)
    missing[tuple(slab)] = False
    out = np.array(arr, dtype=np.float64, copy=True)
    out[missing] = float(fill)
    return (np.clip(out, 0.0, 1.0), missing)


def bbox_fov_crop(
    vol: np.ndarray,
    axis: int,
    *,
    bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    side: str = "both",
    keep_frac: float = 0.5,
    start: int | None = None,
    center_on_image: bool = False,
    fill: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(vol)
    if arr.ndim != 3:
        raise ValueError(f"expected (D,H,W), got {arr.shape}")
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
    if len(bbox) != 3 or any((len(b) != 2 for b in bbox)):
        raise ValueError(f"bbox must be per-axis (lo, hi) pairs; got {bbox!r}")
    side = str(side).lower()
    if side not in ("both", "start", "end", "random"):
        raise ValueError(
            f"side must be 'both', 'start', 'end' or 'random'; got {side!r}"
        )
    keep_frac = float(keep_frac)
    if not 0.0 < keep_frac <= 1.0:
        raise ValueError(f"keep_frac must be in (0,1], got {keep_frac}")
    lo0, hi0 = (int(x) for x in bbox[axis])
    n = arr.shape[axis]
    extent = max(1, hi0 - lo0 + 1)
    lo0, hi0 = (max(0, lo0), min(n - 1, hi0))
    keep = max(1, min(extent, int(round(extent * keep_frac))))
    if side == "both":
        lo = (n - keep) // 2 if center_on_image else lo0 + (extent - keep) // 2
    elif side == "start":
        lo = lo0
    elif side == "end":
        lo = hi0 - keep + 1
    else:
        if start is None:
            raise ValueError("side='random' requires an explicit start index")
        lo_min, lo_max = (lo0, hi0 - keep + 1)
        if not lo_min <= int(start) <= lo_max:
            raise ValueError(
                f"random start {start} outside valid bbox interval [{lo_min}, {lo_max}]"
            )
        lo = int(start)
    lo = max(0, min(n - keep, int(lo)))
    hi = lo + keep
    missing = np.ones(arr.shape, dtype=bool)
    slab = [slice(None)] * 3
    slab[axis] = slice(lo, hi)
    missing[tuple(slab)] = False
    out = np.array(arr, dtype=np.float64, copy=True)
    out[missing] = float(fill)
    return (np.clip(out, 0.0, 1.0), missing)


def _euler_rotation_axis2(theta: float) -> np.ndarray:
    c, s = (math.cos(theta), math.sin(theta))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def periodic_nod_motion(
    vol: np.ndarray,
    *,
    spacing_mm=(1.0, 1.0, 1.0),
    n_poses: int = 3,
    planes_per_pose: int = 6,
    max_angle_deg: float = 3.0,
    trans_ap_mm: float = 2.5,
    trans_si_mm: float = 0.75,
    dc_anchor: int = 4,
    corrupt_fraction: float = 0.375,
    order: int = 1,
    workers: int = 8,
    active_poses: int = 0,
    phase_min_gap_deg: float = 40.0,
    rng: "np.random.Generator | None" = None,
    return_params: bool = False,
):
    import math as _m

    assert vol.ndim == 3, f"expected (D,H,W), got {vol.shape}"
    v = np.asarray(vol, np.float64)
    Dsp = np.diag([float(x) for x in spacing_mm])
    Din = np.diag(1.0 / np.asarray(spacing_mm, float))
    shp = v.shape
    rng = rng if rng is not None else np.random.default_rng()
    phase0 = float(rng.uniform(0.0, 2.0 * _m.pi))
    if int(active_poses) and 0 < int(active_poses) < n_poses:
        P = int(active_poses)
        gap = _m.radians(max(0.0, float(phase_min_gap_deg)))
        phases = []
        for _try in range(200):
            if len(phases) >= P:
                break
            ph = float(rng.uniform(0.0, 2.0 * _m.pi))
            if all((min(abs(ph - q), 2 * _m.pi - abs(ph - q)) >= gap for q in phases)):
                phases.append(ph)
        if len(phases) < P:
            for _ in range(P - len(phases)):
                phases.append(float(rng.uniform(0.0, 2.0 * _m.pi)))
        phases.sort()
        n_moved = P
        pose_phs = [phase0] + phases
    else:
        n_moved = n_poses - 1
        pose_phs = [phase0] + [
            phase0 + 2 * _m.pi * j / n_poses for j in range(1, n_poses)
        ]
    angles = [0.0] + [max_angle_deg * _m.sin(ph) for ph in pose_phs[1:]]
    pad = int(_m.ceil(max(shp[0], shp[1]) * _m.sin(_m.radians(max_angle_deg)))) + 2
    vp = np.pad(
        v, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=0.0
    )
    n0 = vp.shape[0]
    center = (np.asarray(vp.shape, np.float64) - 1.0) / 2.0
    dist = np.abs(np.arange(n0) - n0 // 2)
    pid = np.zeros(n0, dtype=int)
    outside = dist > dc_anchor
    pid[outside] = (dist[outside] - dc_anchor - 1) // planes_per_pose % n_moved + 1
    if corrupt_fraction < 1.0:
        pid[rng.random(n0) >= corrupt_fraction] = 0
    k_motion = np.fft.fftshift(np.fft.fftn(vp))
    params = {
        "n_poses": n_poses,
        "active_poses": n_moved,
        "planes_per_pose": planes_per_pose,
        "max_angle_deg": max_angle_deg,
        "trans_ap_mm": trans_ap_mm,
        "trans_si_mm": trans_si_mm,
        "dc_anchor": dc_anchor,
        "corrupt_fraction": corrupt_fraction,
        "phase0_rad": phase0,
        "angles_deg": angles,
        "poses": [],
    }
    params["poses"].append({"angle_deg": angles[0], "trans_mm": [0.0, 0.0, 0.0]})

    def _pose_k(j: int):
        ph = pose_phs[j]
        t_mm = [trans_si_mm * _m.cos(ph), trans_ap_mm * _m.sin(ph), 0.0]
        R = _euler_rotation_axis2(angles[j])
        A = Din @ R.T @ Dsp
        off = center - A @ center - Din @ R.T @ np.asarray(t_mm)
        moved = affine_transform(
            vp, matrix=A, offset=off, order=order, mode="constant", cval=0.0
        )
        return (j, t_mm, np.fft.fftshift(np.fft.fftn(moved)))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for j, t_mm, moved_k in ex.map(_pose_k, range(1, n_moved + 1)):
            params["poses"].append({"angle_deg": angles[j], "trans_mm": list(t_mm)})
            sel = np.nonzero(pid == j)[0]
            if len(sel):
                k_motion[sel] = moved_k[sel]
    out = np.abs(np.fft.ifftn(np.fft.ifftshift(k_motion)))
    out = out[pad : pad + shp[0], pad : pad + shp[1], :]
    out = np.clip(out, 0.0, 1.0)
    return (out, params) if return_params else out


def simulate_kspace_rigid_motion(
    vol: np.ndarray,
    *,
    rng: "np.random.Generator | None" = None,
    axis: int | None = None,
    events: int | None = None,
    max_rotation_deg: float = 5.0,
    max_translation_vox: float = 4.0,
    line_fraction_range=(0.08, 0.2),
    return_params: bool = False,
):
    arr = np.asarray(vol, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected (D,H,W), got {arr.shape}")
    rng = rng if rng is not None else np.random.default_rng()
    axis = int(rng.integers(0, 3)) if axis is None else int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
    events = int(rng.choice((1, 2), p=(0.8, 0.2))) if events is None else int(events)
    if events not in (1, 2):
        raise ValueError(f"events must be 1 or 2; got {events}")
    if float(max_rotation_deg) < 0.0 or float(max_translation_vox) < 0.0:
        raise ValueError("motion maxima must be non-negative")
    frac_lo, frac_hi = (float(x) for x in line_fraction_range)
    if not 0.0 < frac_lo <= frac_hi < 1.0:
        raise ValueError("line_fraction_range must satisfy 0 < low <= high < 1")
    out_k = np.fft.fftshift(np.fft.fftn(arr))
    center = (np.asarray(arr.shape, dtype=np.float64) - 1.0) / 2.0
    params = {"axis": axis, "events": []}
    for _ in range(events):
        angles = rng.uniform(-float(max_rotation_deg), float(max_rotation_deg), size=3)
        shifts = rng.uniform(
            -float(max_translation_vox), float(max_translation_vox), size=3
        )
        if np.linalg.norm(angles) < 0.25 and np.linalg.norm(shifts) < 0.25:
            angles[axis] = (
                0.5 * float(max_rotation_deg) if float(max_rotation_deg) > 0 else 0.0
            )
            shifts[axis] = (
                0.5 * float(max_translation_vox)
                if float(max_translation_vox) > 0
                else 0.0
            )
        if float(max_rotation_deg) == 0.0 and float(max_translation_vox) == 0.0:
            angles = np.zeros(3)
            shifts = np.zeros(3)
        R = _euler_rotation(*np.deg2rad(angles))
        inv = R.T
        offset = center - inv @ (center + shifts)
        moved = affine_transform(
            arr, inv, offset=offset, order=1, mode="constant", cval=0.0
        )
        moved_k = np.fft.fftshift(np.fft.fftn(moved))
        frac = float(rng.uniform(frac_lo, frac_hi))
        width = max(1, min(arr.shape[axis], int(round(arr.shape[axis] * frac))))
        start = int(rng.integers(0, arr.shape[axis] - width + 1))
        band = [slice(None)] * 3
        band[axis] = slice(start, start + width)
        out_k[tuple(band)] = moved_k[tuple(band)]
        params["events"].append(
            {
                "rotation_deg": angles.tolist(),
                "translation_vox": shifts.tolist(),
                "line_start": start,
                "line_count": width,
                "line_fraction": frac,
            }
        )
    out = np.abs(np.fft.ifftn(np.fft.ifftshift(out_k)))
    out = np.clip(out, 0.0, 1.0)
    return (out, params) if return_params else out


def tumor_inpaint_mask(label: np.ndarray, dilate_iter: int = 2) -> np.ndarray:
    m = np.asarray(label) >= 1
    if dilate_iter > 0 and m.any():
        m = binary_dilation(m, iterations=int(dilate_iter))
    return m


def healthy_inpaint_mask(
    label: np.ndarray,
    brain_mask: np.ndarray,
    dilate_iter: int = 2,
    *,
    margin: int = 2,
    min_brain_frac: float = 0.6,
    max_tries: int = 24,
    prefer_contralateral: bool = False,
    contralateral_axis: int = 2,
    return_info: bool = False,
    rng: "np.random.Generator | None" = None,
) -> "np.ndarray | tuple[np.ndarray, dict]":
    label = np.asarray(label)
    tumor = label >= 1
    tumor_d = (
        binary_dilation(tumor, iterations=int(dilate_iter))
        if dilate_iter > 0 and tumor.any()
        else tumor
    )
    brain = np.asarray(brain_mask)
    if brain.shape != label.shape:
        raise ValueError(
            f"brain_mask shape {brain.shape} does not match label shape {label.shape}"
        )
    brain = brain.astype(bool, copy=False)
    D, H, W = tumor_d.shape
    info = {"strategy": "empty", "attempts": 0}

    def _finish(mask: np.ndarray, strategy: str):
        info["strategy"] = strategy
        result = np.asarray(mask, dtype=bool)
        return (result, info) if return_info else result

    if not tumor_d.any() or not brain.any():
        return _finish(np.zeros((D, H, W), dtype=bool), "empty_input")
    coords = np.argwhere(tumor_d)
    lo = coords.min(0)
    hi = coords.max(0) + 1
    shape = tumor_d[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    forbidden = (
        binary_dilation(tumor, iterations=int(dilate_iter) + int(margin))
        if tumor.any()
        else tumor
    )
    rng = rng if rng is not None else np.random.default_rng()

    def _valid(cand: np.ndarray) -> bool:
        if not cand.any() or (cand & forbidden).any():
            return False
        return bool((cand & brain).sum() >= float(min_brain_frac) * cand.sum())

    if prefer_contralateral:
        axis = int(contralateral_axis)
        if axis not in (0, 1, 2):
            raise ValueError(f"contralateral_axis must be 0, 1, or 2; got {axis}")
        info["attempts"] += 1
        mirrored = np.flip(tumor_d, axis=axis).copy()
        if _valid(mirrored):
            return _finish(mirrored, "contralateral")
    for _ in range(int(max_tries)):
        info["attempts"] += 1
        s = shape
        for ax in range(3):
            if rng.random() < 0.5:
                s = np.flip(s, ax)
        sd, sh, sw = s.shape
        d0 = int(rng.integers(0, max(1, D - sd + 1)))
        h0 = int(rng.integers(0, max(1, H - sh + 1)))
        w0 = int(rng.integers(0, max(1, W - sw + 1)))
        cand = np.zeros((D, H, W), dtype=bool)
        cand[d0 : d0 + sd, h0 : h0 + sh, w0 : w0 + sw] = s
        if _valid(cand):
            return _finish(cand, "random_translate")
    return _finish(np.zeros((D, H, W), dtype=bool), "placement_failed")


def apply_mask(image: np.ndarray, mask: np.ndarray, fill: float = 0.0) -> np.ndarray:
    out = image.copy()
    out[mask] = fill
    return out


def downsample_mask_max(mask: np.ndarray, latent_shape) -> np.ndarray:
    D, H, W = mask.shape
    ld, lh, lw = (int(x) for x in latent_shape)
    assert D % ld == 0 and H % lh == 0 and (W % lw == 0), (
        f"mask {mask.shape} not divisible by latent grid {(ld, lh, lw)}"
    )
    bd, bh, bw = (D // ld, H // lh, W // lw)
    m = mask.astype(np.float32).reshape(ld, bd, lh, bh, lw, bw)
    return m.max(axis=(1, 3, 5))


def sr_sampling_geometry(shape, factor, axis, working_mm):
    if int(axis) not in (0, 1, 2) or not np.isfinite(factor) or factor <= 1:
        raise ValueError("synthetic SR requires a finite factor > 1 and axis in 0..2")
    if working_mm is None or not np.isfinite(working_mm) or working_mm <= 0:
        raise ValueError("synthetic SR requires positive working-grid spacing")
    n = int(shape[axis])
    low_n = int(round(n / factor))
    if low_n < 1:
        raise ValueError(f"SR factor {factor} produces an empty axis of length {n}")
    return dict(
        working_axis_mm=float(working_mm),
        slab_width_mm=max(1, int(round(factor))) * working_mm,
        lowres_axis_size=low_n,
        effective_sample_spacing_mm=(n - 1) * working_mm / (low_n - 1)
        if low_n > 1
        else None,
    )
