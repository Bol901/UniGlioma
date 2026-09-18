"""UniGlioma task-conditioned volumetric inference using caller-supplied local weights."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import h5py
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from models import MMDiTCfg, MMDiT3D, VAE3DConfig, ViTVAE3D
from utils import grid_to_tokens, make_ts, tokens_to_grid
from utils.checkpoint import adapt_unified_modality_embedding
from utils.config_loader import parse_yaml_args
from utils.tasks import CANONICAL_TASKS, SEG_TARGET_MODALITY_ID
from utils.h5_io import h5_axis_spacing
from utils.seg_eligibility import seg_head_readout_allowed
from utils.h5_io import (
    DHW_RAS_AFFINE as _DHW_RAS_AFFINE,
    center_crop_pad as _center_crop_pad,
    require_h5_path,
    read_h5_brain_mask,
    read_h5_volume,
    resolve_latent_paths,
    vae_ckpt_fingerprint,
    validate_latent_attrs,
)


def build_vae_from_cfg(cfg: dict, device: torch.device) -> ViTVAE3D:
    vc = cfg["vae"]
    vae_cfg = VAE3DConfig(
        in_channels=vc["in_channels"],
        out_channels=vc["out_channels"],
        patch_size=tuple(vc["patch_size"]),
        dim=vc["dim"],
        depth_enc=vc["depth_enc"],
        depth_dec=vc["depth_dec"],
        heads=vc["heads"],
        dim_head=vc["dim_head"],
        mlp_mult=vc["mlp_mult"],
        latent_dim=vc["latent_dim"],
        out_activation=vc["out_activation"],
    )
    vae = ViTVAE3D(vae_cfg, use_compile=False)
    payload = torch.load(cfg["vae_ckpt"], map_location="cpu", weights_only=True)
    vae.load_state_dict(payload["model"])
    return vae.to(device).eval()


def _normalize_ckpt_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        nk = k
        nk = nk.replace("module.", "", 1) if nk.startswith("module.") else nk
        nk = nk.replace("._fsdp_wrapped_module", "")
        nk = nk.replace("_fsdp_wrapped_module.", "")
        nk = nk.replace("._checkpoint_wrapped_module", "")
        nk = nk.replace("_checkpoint_wrapped_module.", "")
        out[nk] = v
    return out


def _infer_seg_head_cfg(m: dict) -> dict:
    s = m.get("seg_head", {}) or {}
    if not bool(s.get("enabled", False)):
        return {"seg_head_enabled": False}
    out = {"seg_head_enabled": True, "seg_num_classes": int(s.get("num_classes", 4))}
    if s.get("tap_layer") is not None:
        out["seg_tap_layer"] = int(s["tap_layer"])
    downsample = int(s.get("downsample", 1))
    if downsample < 1 or downsample & downsample - 1 != 0:
        raise ValueError("seg_head.downsample must be a positive power of two")
    drop_stages = downsample.bit_length() - 1
    if s.get("channels") is not None:
        channels = tuple((int(c) for c in s["channels"]))
        out["seg_channels"] = channels[:-drop_stages] if drop_stages else channels
    if s.get("full_shape") is not None:
        out["seg_full_shape"] = tuple((int(x) // downsample for x in s["full_shape"]))
    return out


def build_mmdit_from_cfg(cfg: dict, device: torch.device, dit_ckpt: str):
    m = cfg["mmdit"]
    mmdit_cfg = MMDiTCfg(
        in_channels=m["in_channels"],
        latent_shape=tuple(m["latent_shape"]),
        patch_size=tuple(m["patch_size"]),
        n_ref_types=int(m.get("n_ref_types", 3)),
        max_refs=int(m.get("max_refs", 3)),
        ref_fusion_heads=int(m.get("ref_fusion_heads", 0)),
        delta_time_dim=int(m.get("delta_time_dim", 256)),
        use_ref_slab=bool(m.get("use_ref_slab", True)),
        width=m["width"],
        text_hidden=m["text_hidden"],
        text_max_len=int(cfg.get("text_max_length", m.get("text_max_len", 512))),
        heads=m["heads"],
        mlp_ratio=m["mlp_ratio"],
        depth_double=m["depth_double"],
        depth_single=m["depth_single"],
        num_modalities=int(cfg.get("num_modalities", 4)),
        use_uncond_modality_slot=bool(cfg.get("use_uncond_modality_slot", True)),
        timestep_embed_dim=m["timestep_embed_dim"],
        pool_text_for_vec=bool(m.get("pool_text_for_vec", True)),
        rope_mode=m.get("rope_mode", "axial3d"),
        rope_theta_base=float(m.get("rope_theta_base", 10000.0)),
        dropout=0.0,
        use_qk_norm=bool(m.get("use_qk_norm", False)),
        mlp_variant=str(m.get("mlp_variant", "gelu")).lower(),
        grad_checkpoint=bool(m.get("grad_checkpoint", False)),
        arch_version=int(m.get("arch_version", 1)),
        **_infer_seg_head_cfg(m),
    )
    model = MMDiT3D(mmdit_cfg)
    payload = torch.load(dit_ckpt, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    state = _normalize_ckpt_state_dict(state)
    state, _ = adapt_unified_modality_embedding(state, model.state_dict())
    model.load_state_dict(state, strict=True)
    return (model.to(device).eval(), mmdit_cfg)


def _cropped_iso_affine(ref_path: str, cfg: dict, h5_mode: str):
    target = tuple((int(x) for x in cfg["spatial_size"]))
    ref_path = require_h5_path(ref_path)
    with h5py.File(ref_path, "r") as f:
        key = f"meta/{h5_mode}/affine"
        img_key = f"image/{h5_mode}"
        if key not in f or img_key not in f:
            return None
        A = np.asarray(f[key], dtype=np.float64)
        src_shape = f[img_key].shape
    starts = np.array(
        [
            (s - t) // 2 if s >= t else -((t - s) // 2)
            for s, t in zip(src_shape, target)
        ],
        dtype=np.float64,
    )
    A2 = A.copy()
    A2[:3, 3] = A[:3, :3] @ starts + A[:3, 3]
    return A2.astype(np.float32)


def save_canonical_nifti(vol_dhw: np.ndarray, out_path, affine=None):
    aff = _DHW_RAS_AFFINE if affine is None else np.asarray(affine, dtype=np.float32)
    img = nib.Nifti1Image(np.asarray(vol_dhw, dtype=np.float32), affine=aff)
    img = nib.as_closest_canonical(img)
    nib.save(img, out_path)


def load_ref_latent(
    ref_path: str, *, cfg: dict, device: torch.device, h5_mode: str = "iso1mm"
) -> torch.Tensor:
    ref_path = require_h5_path(ref_path)
    with h5py.File(ref_path, "r") as f:
        base, attrs_path = resolve_latent_paths(f, h5_mode, cfg["spatial_size"])
        if base is None:
            raise KeyError(
                f"{ref_path}: no latent/{h5_mode} at spatial_size={cfg['spatial_size']}; run scripts/precompute_latents.py for this mode and size."
            )
        validate_latent_attrs(
            f,
            ref_path,
            attrs_path=attrs_path,
            expected_fingerprint=vae_ckpt_fingerprint(cfg["vae_ckpt"]),
            expected_spatial_size=cfg["spatial_size"],
            expected_latent_shape=cfg["mmdit"]["latent_shape"],
        )
        mean = np.asarray(f[f"{base}/mean"])
    expected = (
        int(cfg["mmdit"]["in_channels"]),
        *map(int, cfg["mmdit"]["latent_shape"]),
    )
    if mean.shape != expected:
        raise ValueError(f"{ref_path}: latent mean shape {mean.shape} != {expected}")
    return torch.from_numpy(mean).float().unsqueeze(0).to(device)


def load_ref_image01(ref_path: str, cfg: dict, h5_mode: str = "iso1mm") -> np.ndarray:
    return read_h5_volume(
        ref_path, mode=h5_mode, spatial_size=cfg["spatial_size"]
    ).astype(np.float32)


def full_head_sibling(bet_h5: str) -> str:
    bet_h5 = require_h5_path(bet_h5)
    suffix = "_bet.h5"
    if not bet_h5.endswith(suffix):
        raise ValueError(f"motion --ref-image must be a *_bet.h5 file, got {bet_h5!r}")
    candidate = bet_h5[: -len(suffix)] + ".h5"
    if not os.path.isfile(candidate):
        raise FileNotFoundError(
            f"motion requires pre-BET full-head sibling {candidate!r} for {bet_h5!r}"
        )
    return candidate


def load_label_dhw(path: str, cfg: dict) -> np.ndarray:
    label = read_h5_volume(path, spatial_size=cfg["spatial_size"])
    classes = int((cfg["mmdit"].get("seg_head") or {}).get("num_classes", 4))
    if not np.isfinite(label).all() or not np.equal(label, np.round(label)).all():
        raise ValueError(f"{path}: image/iso1mm must contain finite integer label IDs")
    if label.min() < 0 or label.max() >= classes:
        raise ValueError(f"{path}: label IDs must be in 0..{classes - 1}")
    return label.astype(np.int16)


@torch.no_grad()
def _encode_image01_dhw(
    arr01: np.ndarray, vae: ViTVAE3D, cfg: dict, device: torch.device
) -> torch.Tensor:
    arr = _center_crop_pad(
        arr01.astype(np.float32), tuple((int(x) for x in cfg["spatial_size"]))
    )
    x = (
        torch.from_numpy(arr)
        .float()
        .clamp(0, 1)
        .mul_(2.0)
        .sub_(1.0)[None, None]
        .to(device)
    )
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        post = vae.encode(x).latent_dist
        mu = tokens_to_grid(post.mu, vae._last_grid_shape)
    return mu.float()


def _empty_refs(R: int, mmdit_cfg, ld, device):
    return (
        torch.zeros(1, R, mmdit_cfg.in_channels, *ld, device=device),
        torch.zeros(1, R, dtype=torch.long, device=device),
        torch.zeros(1, R, device=device),
        torch.zeros(1, R, dtype=torch.bool, device=device),
    )


@torch.no_grad()
def assemble_conditioning(task, args, *, cfg, mmdit_cfg, vae, device):
    ld = tuple(mmdit_cfg.latent_shape)
    R = int(mmdit_cfg.max_refs)
    tgt_mod = int(args.target_modality)
    ref_latent, ref_modality_id, ref_dt, ref_valid = _empty_refs(
        R, mmdit_cfg, ld, device
    )
    ref_type_id = torch.zeros(1, R, dtype=torch.long, device=device)
    uncond_modality = int(mmdit_cfg.num_modalities)

    def _need(x, name):
        if not x:
            raise ValueError(f"--task {task} requires {name}")
        return x

    refs = list(args.ref_image or [])
    rmods = list(args.ref_modality or [])
    rdts = list([])
    degradation = {}
    if len(refs) > R:
        raise ValueError(f"--task {task} accepts at most {R} --ref-image values")
    if task in ("modality_only",):
        tag = f"{task}_mod{tgt_mod}"
    elif task == "mask_guide":
        lbl = load_label_dhw(_need(args.label, "--label (segmentation H5)"), cfg)
        K = int(getattr(mmdit_cfg, "seg_num_classes", 4))
        lab01 = _center_crop_pad(
            np.clip(lbl.astype(np.float32), 0, K - 1) / float(max(K - 1, 1)),
            tuple((int(x) for x in cfg["spatial_size"])),
        )
        ref_latent[0, 0] = _encode_image01_dhw(lab01, vae, cfg, device)[0]
        ref_modality_id[0, 0] = uncond_modality
        ref_type_id[0, 0] = 1
        ref_valid[0, 0] = True
        tag = f"mask_guide_mod{tgt_mod}"
    elif task in ("sr", "deblur", "dealias", "motion"):
        from utils.ref_synth import (
            gaussian_blur,
            periodic_nod_motion,
            thick_slice_lowres,
            uniform_undersample,
            sr_sampling_geometry,
        )

        source = _need(refs, "one clean --ref-image")[0]
        if len(refs) != 1:
            raise ValueError(f"--task {task} requires exactly one --ref-image")
        h5_mode = args.h5_mode or cfg.get("h5_mode", "iso1mm")
        img01 = None if task == "motion" else load_ref_image01(source, cfg, h5_mode)
        native = bool(getattr(args, "native_input", False))
        if task == "sr":
            if native:
                native_mm = None
                with h5py.File(source, "r") as f:
                    if "meta/native/spacing" in f:
                        spacing = np.asarray(f["meta/native/spacing"], dtype=float)
                        if spacing.size == 3 and np.isfinite(spacing).all():
                            native_mm = float(spacing.max())
                degradation = {"sr_native_axis_mm": native_mm, "sr_native_input": True}
                lr = img01
                tag = f"sr_native_mod{tgt_mod}"
                if args.sr_in_mm is not None or args.sr_out_mm is not None:
                    raise ValueError(
                        "--sr-in-mm/--sr-out-mm cannot be claimed for --native-input (no synthetic degradation geometry)"
                    )
            else:
                with h5py.File(source, "r") as f:
                    working_mm = h5_axis_spacing(
                        f, args.h5_mode or cfg.get("h5_mode", "iso1mm"), args.sr_axis
                    )
                    native_mm = h5_axis_spacing(f, "native", args.sr_axis)
                degradation = sr_sampling_geometry(
                    img01.shape, float(args.sr_factor), int(args.sr_axis), working_mm
                )
                degradation["sr_native_axis_mm"] = native_mm
                degradation["sr_axis"] = int(args.sr_axis)
                degradation["sr_factor"] = float(args.sr_factor)
                degradation["sr_up_order"] = int(args.sr_up_order)
                lr = thick_slice_lowres(
                    img01,
                    float(args.sr_factor),
                    int(args.sr_axis),
                    up_order=int(args.sr_up_order),
                )
                tag = f"sr_x{args.sr_factor:g}_ax{args.sr_axis}_o{args.sr_up_order}_mod{tgt_mod}"
        elif task == "deblur":
            lr = gaussian_blur(img01, sigma=float(args.deblur_sigma))
            tag = f"deblur_s{args.deblur_sigma:g}_mod{tgt_mod}"
        elif task == "dealias":
            lr = uniform_undersample(
                img01,
                factor=max(2, int(args.dealias_factor)),
                axis=int(args.dealias_axis),
            )
            tag = f"dealias_x{args.dealias_factor}_ax{args.dealias_axis}_mod{tgt_mod}"
        else:
            brain_mask = read_h5_brain_mask(
                source, mode=h5_mode, spatial_size=cfg["spatial_size"]
            )
            full_head = load_ref_image01(full_head_sibling(source), cfg, h5_mode)
            lr = (
                periodic_nod_motion(
                    full_head,
                    rng=np.random.default_rng(int(args.seed)),
                    n_poses=int(cfg.get("motion_n_poses", 3)),
                    planes_per_pose=int(cfg.get("motion_planes_per_pose", 6)),
                    max_angle_deg=float(
                        args.motion_max_angle_deg
                        if args.motion_max_angle_deg is not None
                        else cfg.get("motion_max_angle_deg", 3.0)
                    ),
                    trans_ap_mm=float(
                        args.motion_trans_ap_mm
                        if args.motion_trans_ap_mm is not None
                        else cfg.get("motion_trans_ap_mm", 2.5)
                    ),
                    trans_si_mm=float(
                        args.motion_trans_si_mm
                        if args.motion_trans_si_mm is not None
                        else cfg.get("motion_trans_si_mm", 0.75)
                    ),
                    dc_anchor=int(cfg.get("motion_dc_anchor", 4)),
                    corrupt_fraction=float(
                        args.motion_corrupt_fraction
                        if args.motion_corrupt_fraction is not None
                        else cfg.get(
                            "motion_corrupt_fraction",
                            sum(cfg.get("motion_corrupt_fraction_range", (0.25, 0.5)))
                            / 2.0,
                        )
                    ),
                )
                * brain_mask
            )
            degradation = {
                "motion": "periodic_nod_fullhead",
                "brain_mask_key": f"brain_mask/{h5_mode}",
                "brain_voxels": int(brain_mask.sum()),
            }
            tag = f"motion_periodic_mod{tgt_mod}"
        ref_latent[0, 0] = _encode_image01_dhw(lr, vae, cfg, device)[0]
        ref_modality_id[0, 0] = tgt_mod
        ref_valid[0, 0] = True
    elif task == "inpaint":
        from utils.ref_synth import tumor_inpaint_mask, apply_mask

        img01 = load_ref_image01(
            _need(refs, "--ref-image")[0],
            cfg,
            args.h5_mode or cfg.get("h5_mode", "iso1mm"),
        )
        lab = load_label_dhw(_need(args.label, "--label (for the inpaint region)"), cfg)
        hole = tumor_inpaint_mask(lab, int(args.inpaint_dilate))
        masked = apply_mask(img01, hole, 0.0)
        ref_latent[0, 0] = _encode_image01_dhw(masked, vae, cfg, device)[0]
        ref_modality_id[0, 0] = tgt_mod
        ref_valid[0, 0] = True
        if R > 1:
            hole01 = _center_crop_pad(
                hole.astype(np.float32), tuple((int(x) for x in cfg["spatial_size"]))
            )
            ref_latent[0, 1] = _encode_image01_dhw(hole01, vae, cfg, device)[0]
            ref_modality_id[0, 1] = uncond_modality
            ref_type_id[0, 1] = 2
            ref_valid[0, 1] = True
        tag = f"inpaint_mod{tgt_mod}"
    elif task == "whole_brain":
        from utils.ref_synth import bbox_fov_crop

        source = _need(refs, "one clean --ref-image")[0]
        if len(refs) != 1 or R < 2:
            raise ValueError(
                "whole_brain requires exactly one --ref-image and max_refs>=2"
            )
        img01 = load_ref_image01(
            source, cfg, args.h5_mode or cfg.get("h5_mode", "iso1mm")
        )
        h5_mode = args.h5_mode or cfg.get("h5_mode", "iso1mm")
        brain_mask = read_h5_brain_mask(
            source, mode=h5_mode, spatial_size=cfg["spatial_size"]
        )
        brain_idx = np.argwhere(brain_mask)
        bbox = tuple(
            ((int(brain_idx[:, a].min()), int(brain_idx[:, a].max())) for a in range(3))
        )
        rng = np.random.default_rng(int(args.seed))
        view = str(args.view)
        axis = int(args.whole_brain_axis)
        if args.whole_brain_keep_frac is None:
            remove_range = (
                cfg.get("whole_brain_axial_remove_range", (0.2, 0.5))
                if view == "axial"
                else cfg.get("whole_brain_nonaxial_remove_range", (1.0 / 3.0, 0.5))
            )
            keep_frac = 1.0 - float(rng.uniform(*map(float, remove_range)))
        else:
            keep_frac = float(args.whole_brain_keep_frac)
        if view == "axial":
            lo0, hi0 = bbox[axis]
            extent = hi0 - lo0 + 1
            keep = max(1, min(extent, int(round(extent * keep_frac))))
            start = int(rng.integers(lo0, hi0 - keep + 2))
            partial, hole = bbox_fov_crop(
                img01, axis, bbox=bbox, side="random", keep_frac=keep_frac, start=start
            )
        else:
            partial, hole = bbox_fov_crop(
                img01,
                axis,
                bbox=bbox,
                side="both",
                keep_frac=keep_frac,
                center_on_image=True,
            )
        degradation = {
            "brain_mask_key": f"brain_mask/{h5_mode}",
            "brain_voxels": int(brain_mask.sum()),
            "removed_brain_frac": float((brain_mask & hole).sum() / brain_mask.sum()),
        }
        ref_latent[0, 0] = _encode_image01_dhw(partial, vae, cfg, device)[0]
        ref_modality_id[0, 0] = tgt_mod
        ref_valid[0, 0] = True
        ref_latent[0, 1] = _encode_image01_dhw(
            hole.astype(np.float32), vae, cfg, device
        )[0]
        ref_modality_id[0, 1] = uncond_modality
        ref_type_id[0, 1] = 2
        ref_valid[0, 1] = True
        tag = f"whole_brain_k{keep_frac:g}_ax{axis}_mod{tgt_mod}"
    elif task == "missing":
        _need(refs, "--ref-image (one or more sibling modalities)")
        if len(rmods) != len(refs):
            raise ValueError(
                "--ref-modality count must match --ref-image for missing-modality"
            )
        for k, (rp, rm) in enumerate(zip(refs[:R], rmods[:R])):
            ref_latent[0, k] = load_ref_latent(
                rp,
                cfg=cfg,
                device=device,
                h5_mode=args.h5_mode or cfg.get("h5_mode", "iso1mm"),
            )[0]
            ref_modality_id[0, k] = int(rm)
            ref_valid[0, k] = True
        if not 1 <= len(refs) <= R or any((int(m) == tgt_mod for m in rmods)):
            raise ValueError(
                "missing requires 1..max_refs sibling modalities excluding the target"
            )
        tag = f"missing_from{'+'.join(map(str, rmods))}_to{tgt_mod}"
    elif task == "seg":
        if not 1 <= len(refs) <= min(4, R) or len(rmods) != len(refs):
            raise ValueError(
                "seg requires 1..4 --ref-image values and matching --ref-modality ids"
            )
        if len(set(map(int, rmods))) != len(rmods) or any(
            (int(m) not in range(4) for m in rmods)
        ):
            raise ValueError("seg reference modalities must be unique ids in 0..3")
        tgt_mod = SEG_TARGET_MODALITY_ID
        for k, (rp, rm) in enumerate(zip(refs, rmods)):
            ref_latent[0, k] = load_ref_latent(
                rp,
                cfg=cfg,
                device=device,
                h5_mode=args.h5_mode or cfg.get("h5_mode", "iso1mm"),
            )[0]
            ref_modality_id[0, k] = int(rm)
            ref_valid[0, k] = True
        tag = f"seg_from{'+'.join(map(str, sorted(map(int, rmods))))}"
    else:
        raise ValueError(f"unknown --task {task}")
    return dict(
        ref_latent=ref_latent,
        ref_modality_id=ref_modality_id,
        ref_dt=ref_dt,
        ref_valid=ref_valid,
        ref_type_id=ref_type_id,
        task=task,
        degradation=degradation,
        target_modality=tgt_mod,
        tag=tag,
        seg_modalities=sorted(map(int, rmods)) if task == "seg" else None,
    )


@torch.no_grad()
def cfg_velocity(
    model,
    x,
    t,
    te,
    tm,
    mid,
    *,
    ref_latent: torch.Tensor,
    ref_modality_id: torch.Tensor,
    ref_dt: torch.Tensor,
    ref_valid: torch.Tensor,
    cfg_text: float,
    cfg_ref: float,
    ref_type_id: torch.Tensor = None,
) -> torch.Tensor:
    b = x.shape[0]
    if not model.use_ref_slab:
        x2 = x.repeat(2, 1, 1, 1, 1)
        t2 = t.repeat(2)
        te2 = te.repeat(2, 1, 1)
        tm2 = tm.repeat(2, 1)
        mid2 = mid.repeat(2)
        keep_text = torch.cat(
            [
                torch.zeros(b, dtype=torch.bool, device=x.device),
                torch.ones(b, dtype=torch.bool, device=x.device),
            ]
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            v_base, v_text = model(
                x2, t2, te2, tm2, mid2, text_drop_mask=~keep_text
            ).chunk(2, dim=0)
        return v_base + cfg_text * (v_text - v_base)
    K = 3
    device = x.device
    keep_text = torch.cat(
        [
            torch.zeros(b, dtype=torch.bool, device=device),
            torch.ones(b, dtype=torch.bool, device=device),
            torch.ones(b, dtype=torch.bool, device=device),
        ]
    )
    keep_refs = torch.cat(
        [
            torch.zeros(b, dtype=torch.bool, device=device),
            torch.zeros(b, dtype=torch.bool, device=device),
            torch.ones(b, dtype=torch.bool, device=device),
        ]
    )
    x3 = x.repeat(K, 1, 1, 1, 1)
    t3 = t.repeat(K)
    te3 = te.repeat(K, 1, 1)
    tm3 = tm.repeat(K, 1)
    mid3 = mid.repeat(K)
    rl3 = ref_latent.repeat(K, 1, 1, 1, 1, 1)
    rmod3 = ref_modality_id.repeat(K, 1)
    rdt3 = ref_dt.repeat(K, 1)
    rtid3 = ref_type_id.repeat(K, 1) if ref_type_id is not None else None
    rvalid3 = ref_valid.repeat(K, 1) & keep_refs.view(K * b, 1)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        v = model(
            x3,
            t3,
            te3,
            tm3,
            mid3,
            text_drop_mask=~keep_text,
            ref_latent=rl3,
            ref_modality_id=rmod3,
            ref_dt=rdt3,
            ref_valid=rvalid3,
            ref_type_id=rtid3,
        )
    v_base, v_text, v_full = v.chunk(K, dim=0)
    return v_base + cfg_text * (v_text - v_base) + cfg_ref * (v_full - v_text)


TASKS = CANONICAL_TASKS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Multi-task UniGlioma inference.")
    p.add_argument("--config", type=str, default="configs/infer.yaml")
    p.add_argument("--dit-ckpt", type=str, required=True)
    p.add_argument("--task", type=str, default="modality_only", choices=TASKS)
    p.add_argument(
        "--ref-image",
        type=str,
        nargs="*",
        default=[],
        help="Reference .h5 files; cached tasks require precomputed latents.",
    )
    p.add_argument(
        "--ref-modality",
        type=int,
        nargs="*",
        default=[],
        help="Modality id(s) of the ref image(s); required for missing and seg.",
    )
    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label .h5 containing discrete image/iso1mm for mask_guide / inpaint.",
    )
    p.add_argument("--inpaint-dilate", type=int, default=2)
    p.add_argument(
        "--inpaint-content", choices=("healthy", "tumor", "cavity"), default=None
    )
    p.add_argument("--sr-factor", type=float, default=3.0)
    p.add_argument("--sr-axis", type=int, default=0)
    p.add_argument("--sr-up-order", type=int, choices=(0, 1, 3), default=0)
    p.add_argument(
        "--native-input",
        action="store_true",
        help="SR: feed --ref-image as a real thick acquisition without synthetic re-degradation. Deblur/dealias always use the supplied native image directly and add only their named artifact.",
    )
    p.add_argument("--deblur-sigma", type=float, default=1.5)
    p.add_argument("--dealias-factor", type=int, default=3)
    p.add_argument("--dealias-axis", type=int, choices=(0, 1, 2), default=1)
    p.add_argument("--whole-brain-axis", type=int, choices=(0, 1, 2), default=0)
    p.add_argument(
        "--whole-brain-keep-frac",
        type=float,
        default=None,
        help="Optional deterministic retained fraction. Default samples the training policy: axial 0.5..0.8; sagittal/coronal 0.5..0.667.",
    )
    p.add_argument(
        "--motion-max-angle-deg",
        type=float,
        default=None,
        help="Override periodic-nod rotation cap (default config: 3 degrees).",
    )
    p.add_argument("--motion-trans-ap-mm", type=float, default=None)
    p.add_argument("--motion-trans-si-mm", type=float, default=None)
    p.add_argument(
        "--motion-corrupt-fraction",
        type=float,
        default=None,
        help="Override moved k-space fraction; default is the config-range midpoint.",
    )
    p.add_argument(
        "--sr-in-mm",
        type=float,
        default=None,
        help="Optional verified synthetic input spacing; omit for the generic SR prompt.",
    )
    p.add_argument(
        "--sr-out-mm",
        type=float,
        default=None,
        help="Optional target working-grid spacing; must match H5 geometry.",
    )
    p.add_argument(
        "--target-modality",
        type=int,
        default=3,
        help="Modality id of the target to generate (default 3=FLAIR).",
    )
    p.add_argument(
        "--quality",
        type=str,
        default="thin",
        choices=("thin", "medium", "thick"),
        help="Target acquisition quality baked into the prompt; 'thin' = isotropic (highest quality, the default you usually want).",
    )
    p.add_argument(
        "--view", type=str, default="axial", choices=("axial", "sagittal", "coronal")
    )
    p.add_argument(
        "--generic-thin-prompt",
        action="store_true",
        help="Use 'thin-slice' instead of the default explicit '1mm thin-slice' target wording.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Optional complete task instruction; overrides the automatically composed prompt.",
    )
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--cfg-text", type=float, default=None)
    p.add_argument("--cfg-ref", type=float, default=None)
    p.add_argument(
        "--h5-mode",
        choices=("iso1mm", "native"),
        default=None,
        help="H5 image/latent mode; labels always use iso1mm.",
    )
    p.add_argument("--schedule", type=str, default=None)
    p.add_argument("--t-min", type=float, default=None)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--out-dir", type=str, default="outputs/infer")
    p.add_argument(
        "--out-name",
        type=str,
        default=None,
        help="Override the prediction filename (default <task-tag>.nii.gz).",
    )
    p.add_argument(
        "--seg-steps",
        type=int,
        nargs="*",
        default=None,
        help="Sampler step index(es) at which to save the seg-head argmax mask. Pass two (e.g. --seg-steps 12 25) to keep two early/mid masks; negative counts from the end (-1 = last denoising step).",
    )
    p.add_argument(
        "--seg-out-name",
        type=str,
        default=None,
        help="Override seg-head mask filename stem (default <prediction-stem>); each step appends _seg<i>.",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    return parse_yaml_args(p, "inference", argv)


def validate_cli_contract(args) -> None:
    task = str(args.task)
    if args.seg_steps and (not seg_head_readout_allowed(task)):
        raise ValueError(
            f"--seg-steps is forbidden for task={task}; auxiliary head has no valid evaluation target"
        )
    for path in list(args.ref_image or []) + ([args.label] if args.label else []):
        require_h5_path(path)
    refs = list(args.ref_image or [])
    rmods = [int(x) for x in args.ref_modality or []]
    rdts = list([])
    if not 0 <= int(args.target_modality) <= 3:
        raise ValueError("--target-modality must be an image modality id in 0..3")

    def reject(*names):
        values = {
            "--ref-image": refs,
            "--ref-modality": rmods,
            "--ref-dt": rdts,
            "--label": args.label,
        }
        used = [name for name in names if values[name]]
        if used:
            raise ValueError(f"--task {task} does not accept {', '.join(used)}")

    if task in ("modality_only",):
        reject("--ref-image", "--ref-modality", "--ref-dt", "--label")
    elif task == "mask_guide":
        reject("--ref-image", "--ref-modality", "--ref-dt")
        if not args.label:
            raise ValueError("mask_guide requires --label")
    elif task in ("sr", "deblur", "dealias", "motion", "whole_brain"):
        reject("--ref-modality", "--ref-dt", "--label")
        if len(refs) != 1:
            raise ValueError(f"--task {task} requires exactly one --ref-image")
        if task == "whole_brain" and args.quality != "thin":
            raise ValueError("whole_brain only supports a thin-slice target/input")
    elif task == "inpaint":
        reject("--ref-modality", "--ref-dt")
        if len(refs) != 1 or not args.label:
            raise ValueError("inpaint requires exactly one --ref-image and --label")
        if args.inpaint_content is None:
            raise ValueError("inpaint requires --inpaint-content healthy|tumor|cavity")
    elif task == "missing":
        reject("--ref-dt", "--label")
        if not 1 <= len(refs) <= 4 or len(rmods) != len(refs):
            raise ValueError(
                "missing requires 1..4 refs and one --ref-modality per ref"
            )
        if len(set(rmods)) != len(rmods) or any((m not in range(4) for m in rmods)):
            raise ValueError("missing reference modalities must be unique ids in 0..3")
        if int(args.target_modality) in rmods:
            raise ValueError(
                "missing reference modalities must exclude --target-modality"
            )
    elif task == "seg":
        reject("--ref-dt", "--label")
        if not 1 <= len(refs) <= 4 or len(rmods) != len(refs):
            raise ValueError("seg requires 1..4 refs and one --ref-modality per ref")
        if len(set(rmods)) != len(rmods) or any((m not in range(4) for m in rmods)):
            raise ValueError("seg reference modalities must be unique ids in 0..3")
    if args.sr_factor <= 1:
        raise ValueError("--sr-factor must be > 1")
    if args.sr_factor <= 1:
        raise ValueError("--sr-factor must be > 1")
    if (
        args.sr_in_mm is not None
        and args.sr_out_mm is not None
        and (float(args.sr_in_mm) <= float(args.sr_out_mm))
    ):
        raise ValueError(
            "--sr-in-mm must be > --sr-out-mm (only low→high restoration is described; audit SR-02 rejects degenerate in=out slots)."
        )
    if args.deblur_sigma <= 0:
        raise ValueError("--deblur-sigma must be > 0")
    if args.dealias_factor < 2:
        raise ValueError("--dealias-factor must be >= 2")
    if args.whole_brain_keep_frac is not None and (
        not 0 < args.whole_brain_keep_frac < 1
    ):
        raise ValueError("--whole-brain-keep-frac must be in (0, 1)")
    for name in ("motion_max_angle_deg", "motion_trans_ap_mm", "motion_trans_si_mm"):
        value = getattr(args, name, None)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.motion_corrupt_fraction is not None and (
        not 0 <= args.motion_corrupt_fraction <= 1
    ):
        raise ValueError("--motion-corrupt-fraction must be in [0, 1]")


@torch.no_grad()
def _resolve_seg_steps(seg_steps, total_steps: int, seg_enabled: bool):
    if not seg_steps:
        return []
    if not seg_enabled:
        raise ValueError(
            "--seg-steps requested but this checkpoint/config has seg_head disabled"
        )
    out = []
    for s in seg_steps:
        s = total_steps + s if s < 0 else s
        if not 0 <= s < total_steps:
            raise ValueError(f"seg step {s} out of range [0,{total_steps - 1}]")
        out.append(s)
    return sorted(set(out))


def sample_and_decode(
    model,
    vae,
    mmdit_cfg,
    cfg,
    cond,
    text_emb,
    text_mask,
    *,
    steps,
    cfg_text,
    cfg_ref,
    schedule,
    t_min,
    rho,
    seg_steps,
    device,
):
    z_shape = (mmdit_cfg.in_channels,) + tuple(mmdit_cfg.latent_shape)
    ts = make_ts(steps, device=device, kind=schedule, t_min=t_min, rho=rho)
    mid_full = torch.tensor([cond["target_modality"]], device=device, dtype=torch.long)
    seg_set = set(
        _resolve_seg_steps(seg_steps, steps, getattr(model, "seg_head_enabled", False))
    )
    if not seg_head_readout_allowed(
        cond.get("task", "seg" if cond["target_modality"] == 4 else "modality_only")
    ):
        seg_set.clear()
    full_shape = tuple((int(v) for v in cfg["spatial_size"]))
    x = torch.randn((1,) + z_shape, device=device)
    seg_masks = {}
    for i in trange(steps, desc=cond["tag"], leave=False):
        t = ts[i].expand(1)
        if i in seg_set:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _v, seg_logits = model(
                    x,
                    t,
                    text_emb,
                    text_mask,
                    mid_full,
                    text_drop_mask=torch.zeros(1, dtype=torch.bool, device=device),
                    ref_latent=cond["ref_latent"],
                    ref_modality_id=cond["ref_modality_id"],
                    ref_dt=cond["ref_dt"],
                    ref_valid=cond["ref_valid"],
                    ref_type_id=cond["ref_type_id"],
                    return_seg=True,
                )
            seg_low = seg_logits.float().argmax(dim=1, keepdim=True).float()
            seg_masks[i] = (
                F.interpolate(seg_low, size=full_shape, mode="nearest")[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        v = cfg_velocity(
            model,
            x,
            t,
            text_emb,
            text_mask,
            mid_full,
            ref_latent=cond["ref_latent"],
            ref_modality_id=cond["ref_modality_id"],
            ref_dt=cond["ref_dt"],
            ref_valid=cond["ref_valid"],
            ref_type_id=cond["ref_type_id"],
            cfg_text=cfg_text,
            cfg_ref=cfg_ref,
        )
        x = x + v * (ts[i + 1] - ts[i])
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        x_img = vae.decode(grid_to_tokens(x), grid=tuple(mmdit_cfg.latent_shape))
    vol = np.clip((x_img.float().detach().cpu().numpy()[0, 0] + 1.0) * 0.5, 0.0, 1.0)
    return (vol, seg_masks)


def synthetic_sr_prompt(args, cond):
    from utils.prompts import compose

    target_mm = (
        1
        if args.quality == "thin" and (not getattr(args, "generic_thin_prompt", False))
        else None
    )
    if getattr(args, "native_input", False):
        native_mm = cond.get("degradation", {}).get("sr_native_axis_mm")
        if native_mm is not None and np.isfinite(native_mm) and (float(native_mm) > 1):
            return compose(
                "sr",
                cond["target_modality"],
                args.quality,
                args.view,
                target_mm=target_mm,
                native_input_mm=f"{float(native_mm):g}",
            )
        return compose(
            "sr",
            cond["target_modality"],
            args.quality,
            args.view,
            target_mm=target_mm,
            native_input_generic=True,
        )
    if args.sr_in_mm is not None or args.sr_out_mm is not None:
        geometry = cond["degradation"]
        native = geometry["sr_native_axis_mm"]
        working = geometry["working_axis_mm"]
        effective = geometry["effective_sample_spacing_mm"]
        slab = geometry["slab_width_mm"]
        if (
            args.sr_in_mm is None
            or args.sr_out_mm is None
            or native is None
            or (native > working)
            or (effective is None)
            or (not np.isclose(effective, slab, rtol=1e-06))
            or (not np.isclose(args.sr_in_mm, effective, rtol=1e-06))
            or (not np.isclose(args.sr_out_mm, working, rtol=1e-06))
        ):
            raise ValueError(
                "SR mm prompt values do not match the H5 working grid, native acquisition and synthetic sample spacing; omit --sr-in-mm/--sr-out-mm for a generic instruction"
            )
        return compose(
            "sr",
            cond["target_modality"],
            args.quality,
            args.view,
            target_mm=target_mm,
            synthetic_input_mm=f"{args.sr_in_mm:g}",
        )
    return compose(
        "sr",
        cond["target_modality"],
        args.quality,
        args.view,
        target_mm=target_mm,
        synthetic_input_generic=True,
    )


def input_acquisition(cfg: dict, path: str):
    from utils.prompts import _norm_quality, _norm_view, acq_stem

    lookup_path = cfg.get("acq_lookup")
    if not lookup_path or not Path(lookup_path).is_file():
        return ("unknown", "unknown")
    with Path(lookup_path).open("r", encoding="utf-8") as f:
        lookup = json.load(f)
    qv = lookup.get(acq_stem(path)) or (None, None)
    return (
        _norm_quality(qv[0] if len(qv) > 0 else None),
        _norm_view(qv[1] if len(qv) > 1 else None),
    )


def native_slice_thickness(path: str):
    try:
        with h5py.File(path, "r") as f:
            if "meta/native/spacing" not in f:
                return None
            spacing = np.asarray(f["meta/native/spacing"], dtype=float)
        if spacing.size != 3 or not np.isfinite(spacing).all():
            return None
        return float(spacing.max())
    except (OSError, KeyError, ValueError):
        return None


def main():
    args = parse_args()
    validate_cli_contract(args)
    cfg = args._config
    observed_q = observed_v = "unknown"
    if args.ref_image:
        observed_q, observed_v = input_acquisition(cfg, args.ref_image[0])
    if args.task == "whole_brain":
        if observed_q not in ("unknown", "thin"):
            raise ValueError(
                f"whole_brain requires a native thin-slice input; acquisition lookup says {observed_q}"
            )
        if observed_v != "unknown":
            args.view = observed_v
        lo, hi = (0.5, 0.8) if args.view == "axial" else (0.5, 2.0 / 3.0)
        if args.whole_brain_keep_frac is not None and (
            not lo <= args.whole_brain_keep_frac <= hi
        ):
            raise ValueError(
                f"whole_brain --whole-brain-keep-frac must be in [{lo:g}, {hi:g}] for {args.view} view"
            )
    if args.task in ("whole_brain", "motion"):
        source = args.ref_image[0]
        h5_mode = args.h5_mode or cfg.get("h5_mode", "iso1mm")
        read_h5_brain_mask(source, mode=h5_mode, spatial_size=cfg["spatial_size"])
        if args.task == "motion":
            full_head_sibling(source)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    infer = cfg.get("infer", {})
    steps = int(args.steps if args.steps is not None else infer.get("steps", 50))
    mode = args.task
    task_cfg = (infer.get("task_cfg") or {}).get(mode, {})
    cfg_text = float(
        args.cfg_text if args.cfg_text is not None else task_cfg.get("text", infer.get("cfg_text", 4.0))
    )
    cfg_ref = float(
        args.cfg_ref if args.cfg_ref is not None else task_cfg.get("ref", infer.get("cfg_ref", 1.0))
    )
    schedule = args.schedule or infer.get("schedule", "power")
    t_min = float(args.t_min if args.t_min is not None else infer.get("t_min", 0.001))
    rho = float(args.rho if args.rho is not None else infer.get("rho", 3.0))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vae = build_vae_from_cfg(cfg, device)
    model, mmdit_cfg = build_mmdit_from_cfg(cfg, device, args.dit_ckpt)
    if not cfg.get("qwen3vl_path"):
        raise ValueError(
            "infer.py requires `qwen3vl_path` in the YAML — text encoding must match the Qwen3-VL features the model was trained on."
        )
    from models.qwen3vl_text import Qwen3VLTextEncoder

    enc = Qwen3VLTextEncoder(
        model_path=cfg["qwen3vl_path"],
        device=device,
        dtype=torch.bfloat16,
        chat_template=bool(cfg.get("qwen3vl_chat_template", True)),
        add_generation_prompt=bool(cfg.get("qwen3vl_add_generation_prompt", True)),
        layer=int(cfg.get("qwen3vl_layer", -1)),
        max_length=int(cfg.get("text_max_length", 512)),
    )
    cond = assemble_conditioning(
        args.task, args, cfg=cfg, mmdit_cfg=mmdit_cfg, vae=vae, device=device
    )
    prompt_quality, prompt_view = (args.quality, args.view)
    if args.task in ("motion", "whole_brain"):
        if args.task == "motion":
            if observed_q != "unknown":
                prompt_quality = observed_q
            if observed_v != "unknown":
                prompt_view = observed_v
        elif observed_q not in ("unknown", "thin"):
            raise ValueError(
                f"whole_brain requires a native thin-slice input; acquisition lookup says {observed_q}"
            )
        else:
            prompt_quality, prompt_view = ("thin", "axial")
    prompt_target_mm = (
        1
        if prompt_quality == "thin"
        and (not getattr(args, "generic_thin_prompt", False))
        else None
    )
    from utils.prompts import compose as compose_prompt

    if args.task == "sr":
        prompt = synthetic_sr_prompt(args, cond)
    elif args.task in ("deblur", "dealias") and observed_q == "thick":
        native_mm = native_slice_thickness(args.ref_image[0])
        if native_mm is not None and native_mm > 1:
            prompt = compose_prompt(
                args.task,
                cond["target_modality"],
                prompt_quality,
                prompt_view,
                target_mm=prompt_target_mm,
                native_input_mm=f"{native_mm:g}",
            )
        else:
            prompt = compose_prompt(
                args.task,
                cond["target_modality"],
                prompt_quality,
                prompt_view,
                target_mm=prompt_target_mm,
                native_input_generic=True,
            )
    elif args.task == "inpaint":
        if args.inpaint_content is None:
            raise ValueError("inpaint requires --inpaint-content healthy|tumor|cavity")
        prompt = compose_prompt(
            args.task,
            cond["target_modality"],
            prompt_quality,
            prompt_view,
            inpaint_content=args.inpaint_content,
            target_mm=prompt_target_mm,
        )
    elif args.task == "seg":
        prompt = compose_prompt(
            args.task,
            cond["target_modality"],
            prompt_quality,
            prompt_view,
            seg_modalities=cond["seg_modalities"],
            seg_catalog_path=cfg.get("seg_prompt_variants_path"),
        )
    else:
        prompt = compose_prompt(
            args.task,
            cond["target_modality"],
            prompt_quality,
            prompt_view,
            target_mm=prompt_target_mm,
        )
    if args.prompt.strip():
        prompt = args.prompt.strip()
    print(f"[infer] prompt: {prompt!r}", flush=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        text_emb, text_mask = enc.encode([prompt])
    text_emb = text_emb.float()
    text_mask = text_mask.long()
    if tuple(cond["ref_latent"].shape[-3:]) != tuple(mmdit_cfg.latent_shape):
        raise ValueError(
            f"ref_latent grid {tuple(cond['ref_latent'].shape[-3:])} != {tuple(mmdit_cfg.latent_shape)} (check spatial_size)."
        )
    vol, seg_masks = sample_and_decode(
        model,
        vae,
        mmdit_cfg,
        cfg,
        cond,
        text_emb,
        text_mask,
        steps=steps,
        cfg_text=cfg_text,
        cfg_ref=cfg_ref,
        schedule=schedule,
        t_min=t_min,
        rho=rho,
        seg_steps=args.seg_steps,
        device=device,
    )
    h5_mode = args.h5_mode or cfg.get("h5_mode", "iso1mm")
    affine_source = args.ref_image[0] if args.ref_image else args.label
    affine_mode = h5_mode if args.ref_image else "iso1mm"
    aff = (
        _cropped_iso_affine(affine_source, cfg, affine_mode) if affine_source else None
    )
    out_path = out_dir / (args.out_name or f"{cond['tag']}.nii.gz")
    if args.task == "seg":
        num_classes = int(
            (cfg["mmdit"].get("seg_head", {}) or {}).get("num_classes", 4)
        )
        vol = np.rint(np.clip(vol, 0.0, 1.0) * float(max(num_classes - 1, 1))).astype(
            np.int16
        )
    save_canonical_nifti(vol, out_path, affine=aff)
    print(f"saved {out_path} (canonical RAS)", flush=True)
    if cond.get("degradation"):
        import json

        out_path.with_name(out_path.name + ".json").write_text(
            json.dumps(
                {
                    "task": args.task,
                    "prompt": prompt,
                    "degradation": cond["degradation"],
                },
                indent=2,
            )
        )
    stem = args.seg_out_name or (
        out_path.name[:-7] if out_path.name.endswith(".nii.gz") else out_path.stem
    )
    for s, seg_mask in sorted(seg_masks.items()):
        seg_path = out_dir / f"{stem}_seg{s}.nii.gz"
        save_canonical_nifti(seg_mask, seg_path, affine=aff)
        print(
            f"saved {seg_path} (seg head argmax @ step {s}, canonical RAS)", flush=True
        )


if __name__ == "__main__":
    main()
