from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py
import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("mask-latent")
from utils.h5_io import (
    MASK_LATENT_GROUP,
    center_crop_pad,
    latent_size_tag,
    resolve_sample_h5_path,
    vae_ckpt_fingerprint,
)
from models import VAE3DConfig, ViTVAE3D
from utils import load_train_config, tokens_to_grid


def build_vae(cfg: dict, device: torch.device) -> ViTVAE3D:
    vc = cfg["vae"]
    vae = ViTVAE3D(
        VAE3DConfig(
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
        ),
        use_compile=False,
    )
    ckpt = cfg["vae_ckpt"]
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"VAE checkpoint not found: {ckpt}")
    vae.load_state_dict(
        torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
    )
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae.to(device).eval()


def render_label_01(label: np.ndarray, num_classes: int) -> np.ndarray:
    K = max(int(num_classes), 2)
    lab = np.clip(label.astype(np.float32), 0, K - 1)
    return lab / float(K - 1)


def collect_label_h5s(json_paths: list[str], label_h5_root: str) -> list[str]:
    seen, out = (set(), [])

    def add_raw(p):
        if not p:
            return
        h5 = resolve_sample_h5_path(p, h5_root=label_h5_root)
        h5 = os.path.normpath(h5)
        if h5 not in seen:
            seen.add(h5)
            out.append(h5)

    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as f:
            items = json.load(f)
        for it in items:
            if isinstance(it, str):
                add_raw(it)
            elif isinstance(it, dict):
                add_raw(it.get("label_path") or it.get("label_h5"))
    return out


@torch.no_grad()
def encode_mean(vae: ViTVAE3D, x01: np.ndarray, device) -> np.ndarray:
    x = (
        torch.from_numpy(np.ascontiguousarray(x01))
        .float()
        .clamp(0, 1)
        .mul_(2.0)
        .sub_(1.0)
    )
    x = x[None, None].to(device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        post = vae.encode(x).latent_dist
        mu = tokens_to_grid(post.mu, vae._last_grid_shape)
    return mu.float().cpu().numpy()[0].astype(np.float16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument(
        "--json",
        nargs="+",
        required=True,
        help="label-H5 path list, or stage JSON(s) with label_path.",
    )
    ap.add_argument(
        "--h5-root",
        default=None,
        help="Prefix for relative paths (default cfg.label_h5_root).",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Default cfg seg num_classes or 4.",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg = load_train_config(args.config)
    seg_cfg = cfg["mmdit"].get("seg_head", {}) or {}
    label_h5_root = (
        args.h5_root
        if args.h5_root is not None
        else seg_cfg.get("label_h5_root", cfg.get("h5_root"))
    )
    num_classes = (
        args.num_classes
        if args.num_classes is not None
        else int(seg_cfg.get("num_classes", 4))
    )
    device = (
        torch.device(f"cuda:{local_rank}")
        if world > 1 and args.device.startswith("cuda")
        else torch.device(args.device)
    )
    fingerprint = vae_ckpt_fingerprint(cfg["vae_ckpt"])
    iso_target = tuple((int(x) for x in cfg["spatial_size"]))
    latent_shape = tuple((int(x) for x in cfg["mmdit"]["latent_shape"]))
    tag = latent_size_tag(iso_target)
    base = f"{MASK_LATENT_GROUP}/{tag}"
    pfx = f"rank{rank}"
    files = collect_label_h5s(args.json, label_h5_root)
    shard = files[rank::world]
    if args.limit > 0:
        shard = shard[: args.limit]
    log.info(
        "%s label_h5s_total=%d shard=%d K=%d tag=%s fp=%s world=%d",
        pfx,
        len(files),
        len(shard),
        num_classes,
        tag,
        fingerprint[:12],
        world,
    )
    vae = build_vae(cfg, device)
    stats = {"written": 0, "skipped": 0, "missing": 0}
    t0 = time.time()
    for fi, h5_path in enumerate(shard):
        if fi % args.log_every == 0:
            rate = fi / max(1e-06, time.time() - t0)
            log.info(
                "%s %d/%d (%.1f/s) written=%d skipped=%d missing=%d",
                pfx,
                fi,
                len(shard),
                rate,
                stats["written"],
                stats["skipped"],
                stats["missing"],
            )
        if not os.path.isfile(h5_path):
            stats["missing"] += 1
            continue
        try:
            with h5py.File(h5_path, "r") as f:
                if (
                    not args.force
                    and f"{base}/mean" in f
                    and (
                        str(f[base].attrs.get("vae_ckpt_fingerprint", ""))
                        == fingerprint
                    )
                    and (int(f[base].attrs.get("num_classes", -1)) == int(num_classes))
                ):
                    stats["skipped"] += 1
                    continue
                if "image/iso1mm" not in f:
                    stats["missing"] += 1
                    continue
                lab = np.asarray(f["image/iso1mm"])
        except (OSError, KeyError) as e:
            log.warning("%s skip %s: %s", pfx, h5_path, e)
            continue
        lab = center_crop_pad(lab.astype(np.int16), iso_target)
        mean = encode_mean(vae, render_label_01(lab, num_classes), device)
        with h5py.File(h5_path, "a") as f:
            g = f.require_group(base)
            g.attrs["cache_version"] = 1
            g.attrs["vae_ckpt_fingerprint"] = fingerprint
            g.attrs["spatial_size"] = np.asarray(iso_target, dtype=np.int64)
            g.attrs["latent_shape"] = np.asarray(latent_shape, dtype=np.int64)
            g.attrs["num_classes"] = int(num_classes)
            g.attrs["render"] = "label_id/(K-1)->[0,1]"
            dp = f"{base}/mean"
            if dp in f:
                del f[dp]
            f.create_dataset(dp, data=mean)
        stats["written"] += 1
    log.info(
        "%s DONE shard=%d written=%d skipped=%d missing=%d elapsed=%.1fs",
        pfx,
        len(shard),
        stats["written"],
        stats["skipped"],
        stats["missing"],
        time.time() - t0,
    )


if __name__ == "__main__":
    main()
