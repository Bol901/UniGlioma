from __future__ import annotations
import argparse
import json
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
from utils.h5_io import (
    center_crop_pad,
    resolve_sample_h5_path,
    LATENT_CACHE_VERSION,
    LATENT_GROUP,
    LATENT_MODES,
    latent_size_tag,
    vae_ckpt_fingerprint,
)


def _latent_data_base(mode: str, iso_target) -> str:
    if mode == "iso1mm":
        return f"{LATENT_GROUP}/{mode}/{latent_size_tag(iso_target)}"
    return f"{LATENT_GROUP}/{mode}"


from models import VAE3DConfig, ViTVAE3D
from utils import load_train_config, tokens_to_grid

PATCH = 8
ALIGN = 16


def _fmt_eta(sec: float) -> str:
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def build_vae(cfg: dict, device: torch.device, compile_vae: bool) -> ViTVAE3D:
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
    vae = ViTVAE3D(vae_cfg, use_compile=compile_vae)
    ckpt = cfg["vae_ckpt"]
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"VAE checkpoint not found: {ckpt}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    vae.load_state_dict(payload["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae.to(device).eval()


def _round_up(v: int, m: int) -> int:
    return (int(v) + m - 1) // m * m


def foreground_bbox(arr: np.ndarray) -> tuple[slice, slice, slice]:
    mask = arr > 0
    if not mask.any():
        return tuple((slice(0, s) for s in arr.shape))
    idx = np.where(mask)
    sl = []
    for ax in range(3):
        lo, hi = (int(idx[ax].min()), int(idx[ax].max()) + 1)
        sl.append(slice(lo, hi))
    return tuple(sl)


def pad_up_to_align(arr: np.ndarray, m: int = ALIGN) -> np.ndarray:
    target = tuple((_round_up(s, m) for s in arr.shape))
    if target == arr.shape:
        return arr
    return center_crop_pad(arr, target)


def native_preprocess(arr: np.ndarray, h5_basename: str) -> np.ndarray:
    if "bet" in h5_basename.lower():
        arr = arr[foreground_bbox(arr)]
    return pad_up_to_align(arr, ALIGN)


def to_vae_input(arr: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(arr)).float()
    x = x.clamp(0, 1).mul_(2.0).sub_(1.0)
    return x[None]


@torch.no_grad()
def encode_batch(vae: ViTVAE3D, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        post = vae.encode(x).latent_dist
        g = vae._last_grid_shape
        mu = tokens_to_grid(post.mu, g)
        logvar = tokens_to_grid(post.logvar, g)
    return (
        mu.float().cpu().numpy().astype(np.float16),
        logvar.float().cpu().numpy().astype(np.float16),
    )


def load_h5_list(json_paths: list[str], h5_root: str | None) -> list[str]:
    seen, out = (set(), [])
    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError(f"{jp}: top level must be a list of h5 path strings")
        for it in items:
            p = resolve_sample_h5_path(it, h5_root=h5_root)
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _stamp_iso_attrs(
    g: "h5py.Group", cfg: dict, fingerprint: str, latent_shape
) -> None:
    g.attrs["cache_version"] = int(LATENT_CACHE_VERSION)
    g.attrs["vae_ckpt_fingerprint"] = fingerprint
    g.attrs["spatial_size"] = np.asarray(
        [int(x) for x in cfg["spatial_size"]], dtype=np.int64
    )
    g.attrs["latent_shape"] = np.asarray([int(x) for x in latent_shape], dtype=np.int64)
    g.attrs["latent_dim"] = int(cfg["vae"]["latent_dim"])
    g.attrs["crop"] = "iso=center192;native=bbox-if-bet,pad16"


def _write_mode(
    f: "h5py.File", base: str, mean: np.ndarray, logvar: np.ndarray
) -> None:
    for name, arr in (("mean", mean), ("logvar", logvar)):
        dp = f"{base}/{name}"
        if dp in f:
            del f[dp]
        f.create_dataset(dp, data=arr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    ap.add_argument(
        "--json",
        nargs="+",
        required=True,
        help="JSON file(s); each a list of h5 path strings.",
    )
    ap.add_argument(
        "--h5-root",
        type=str,
        default=None,
        help="Prefix for relative h5 paths (default cfg.h5_root).",
    )
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["iso1mm", "native"],
        help=f"Subset of {LATENT_MODES} to encode.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="VAE encode batch. iso1mm batches across files; native batches files that share the padded shape.",
    )
    ap.add_argument(
        "--native-voxel-budget",
        type=int,
        default=48000000,
        help="Max summed input voxels per native batch. Large native volumes auto-drop toward batch 1 so peak activation memory stays bounded across shapes.",
    )
    ap.add_argument(
        "--max-native-voxels",
        type=int,
        default=0,
        help="Skip a native sample whose padded voxel count exceeds this (0 = use --native-voxel-budget).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if latent datasets already exist.",
    )
    ap.add_argument(
        "--huge-max-native",
        type=int,
        default=0,
        help="Hard ceiling for the deferred huge-native end pass. Native volumes over --max-native-voxels are no longer dropped: their paths are collected and encoded one-by-one (batch=1) AFTER the main pass. 0 = no ceiling (encode every deferred file); >0 = still genuinely skip any whose padded voxel count exceeds this (logged to a .huge_skipped manifest).",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--compile-vae", action="store_true")
    ap.add_argument(
        "--manifest-dir",
        type=str,
        default="logs/.latent_manifest",
        help="Dir for per-rank done manifests. On restart, paths listed here are skipped WITHOUT opening the H5 (the slow part). Empty string disables the manifest.",
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Log progress every N files (per rank).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug: process at most this many files (per rank).",
    )
    args = ap.parse_args()
    for m in args.modes:
        if m not in LATENT_MODES:
            raise ValueError(f"--modes got {m!r}; allowed: {LATENT_MODES}")
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = rank == 0
    cfg = load_train_config(args.config)
    h5_root = args.h5_root if args.h5_root is not None else cfg.get("h5_root")
    device = (
        torch.device(f"cuda:{local_rank}")
        if world > 1 and args.device.startswith("cuda")
        else torch.device(args.device)
    )
    fingerprint = vae_ckpt_fingerprint(cfg["vae_ckpt"])
    latent_shape = tuple((int(x) for x in cfg["mmdit"]["latent_shape"]))
    iso_target = tuple((int(x) for x in cfg["spatial_size"]))
    pfx = f"[precompute][rank{rank}]"
    print(f"{pfx} loading h5 list from {args.json} ...", flush=True)
    files = load_h5_list(args.json, h5_root)
    bs = max(1, args.batch_size)
    nat_budget = max(1, args.native_voxel_budget)
    max_native = args.max_native_voxels if args.max_native_voxels > 0 else nat_budget
    shard = files[rank::world]
    if args.limit > 0:
        shard = shard[: args.limit]
    print(
        f"{pfx} files_total={len(files)} shard={len(shard)} modes={args.modes} iso_target={iso_target} fingerprint={fingerprint[:12]} world={world} batch={bs} {('DRY-RUN' if args.dry_run else '')}",
        flush=True,
    )
    if args.dry_run:
        sample = shard[:5]
        exist = sum((os.path.isfile(p) for p in sample))
        print(
            f"{pfx} DRY-RUN sample_exists={exist}/{len(sample)} e.g. {sample[:2]}",
            flush=True,
        )
        return
    print(
        f"{pfx} building VAE (ckpt={os.path.basename(cfg['vae_ckpt'])}, compile={args.compile_vae}) ...",
        flush=True,
    )
    tb = time.time()
    vae = build_vae(cfg, device, compile_vae=args.compile_vae)
    print(
        f"{pfx} VAE ready in {time.time() - tb:.1f}s; starting single pass (open→decide→read→batch→encode→write)",
        flush=True,
    )
    stats = {
        "written": 0,
        "skipped_modes": 0,
        "skipped_files": 0,
        "missing": 0,
        "skipped_done": 0,
        "huge_deferred": 0,
        "huge_written": 0,
        "huge_skipped": 0,
    }
    huge_paths: list[str] = []
    t0 = time.time()
    done: set[str] = set()
    recorded: set[str] = set()
    pending: dict[str, set[str]] = {}
    manifest_fh = None
    mdir = args.manifest_dir.strip()
    if mdir:
        import glob

        sig = (
            "-".join(sorted(args.modes))
            + "-s"
            + "x".join((str(x) for x in iso_target))
            + "-v"
            + fingerprint[:12]
        )
        os.makedirs(mdir, exist_ok=True)
        if not args.force:
            for mf in glob.glob(os.path.join(mdir, f"latent.{sig}.rank*.done")):
                try:
                    with open(mf, "r", encoding="utf-8") as fh:
                        for line in fh:
                            p = line.strip()
                            if p:
                                done.add(p)
                except OSError:
                    pass
        mpath = os.path.join(mdir, f"latent.{sig}.rank{rank}.done")
        manifest_fh = open(mpath, "a", encoding="utf-8")
        print(
            f"{pfx} manifest {mpath} loaded_done={len(done)} sig={sig} force={args.force}",
            flush=True,
        )

    def mark_done(h5_path: str) -> None:
        if h5_path in recorded:
            return
        recorded.add(h5_path)
        if manifest_fh is not None:
            manifest_fh.write(h5_path + "\n")
            manifest_fh.flush()

    def write_one(
        h5_path: str, mode: str, mean: np.ndarray, logvar: np.ndarray
    ) -> None:
        base = _latent_data_base(mode, iso_target)
        with h5py.File(h5_path, "a") as f:
            if mode == "iso1mm":
                _stamp_iso_attrs(f.require_group(base), cfg, fingerprint, latent_shape)
            else:
                f.require_group(LATENT_GROUP)
            _write_mode(f, base, mean, logvar)
        stats["written"] += 1
        rem = pending.get(h5_path)
        if rem is not None:
            rem.discard(mode)
            if not rem:
                pending.pop(h5_path, None)
                mark_done(h5_path)

    def flush(mode: str, paths: list[str], tensors: list[torch.Tensor]) -> None:
        if not paths:
            return
        x = torch.stack(tensors, 0).to(device, non_blocking=True)
        mean, logvar = encode_batch(vae, x)
        for i, p in enumerate(paths):
            write_one(p, mode, mean[i], logvar[i])
        if mode == "native":
            del x, mean, logvar
            torch.cuda.empty_cache()

    iso_paths: list[str] = []
    iso_ts: list[torch.Tensor] = []
    nat_buckets: dict[tuple, tuple[list, list]] = {}
    for fi, h5_path in enumerate(shard):
        if fi % args.log_every == 0:
            rate = fi / max(1e-06, time.time() - t0)
            eta = _fmt_eta((len(shard) - fi) / max(1e-06, rate))
            print(
                f"{pfx} scan/encode {fi}/{len(shard)} ({rate:.1f} files/s) ETA={eta} written={stats['written']} skipped_files={stats['skipped_files']} skipped_modes={stats['skipped_modes']} missing={stats['missing']} huge_deferred={stats['huge_deferred']} skipped_done={stats['skipped_done']} iso_buf={len(iso_paths)} nat_buf={sum((len(v[0]) for v in nat_buckets.values()))}",
                flush=True,
            )
        if h5_path in done:
            stats["skipped_done"] += 1
            continue
        if not os.path.isfile(h5_path):
            stats["missing"] += 1
            continue
        base = os.path.basename(h5_path)
        try:
            with h5py.File(h5_path, "r") as f:
                todo = []
                for mode in args.modes:
                    mbase = _latent_data_base(mode, iso_target)
                    have = f"{mbase}/mean" in f and f"{mbase}/logvar" in f
                    if have and mode == "iso1mm":
                        fp = str(f[mbase].attrs.get("vae_ckpt_fingerprint", ""))
                        if fp != fingerprint:
                            have = False
                    if have and (not args.force) or f"image/{mode}" not in f:
                        stats["skipped_modes"] += 1
                    else:
                        todo.append(mode)
                if not todo:
                    stats["skipped_files"] += 1
                    mark_done(h5_path)
                    continue
                pending[h5_path] = set(todo)
                arrs = {m: np.asarray(f[f"image/{m}"], dtype=np.float32) for m in todo}
        except (OSError, KeyError) as e:
            print(f"{pfx} WARN skip {h5_path}: {e}", flush=True)
            continue
        if "iso1mm" in arrs:
            iso_ts.append(to_vae_input(center_crop_pad(arrs["iso1mm"], iso_target)))
            iso_paths.append(h5_path)
            if len(iso_paths) >= bs:
                flush("iso1mm", iso_paths, iso_ts)
                iso_paths, iso_ts = ([], [])
        if "native" in arrs:
            x = to_vae_input(native_preprocess(arrs["native"], base))
            vox = x.shape[-3] * x.shape[-2] * x.shape[-1]
            if vox > max_native:
                stats["huge_deferred"] += 1
                huge_paths.append(h5_path)
                del x
            else:
                eff_bs = max(1, min(bs, nat_budget // max(1, vox)))
                key = tuple(x.shape)
                bkt = nat_buckets.setdefault(key, ([], []))
                bkt[0].append(h5_path)
                bkt[1].append(x)
                if len(bkt[0]) >= eff_bs:
                    flush("native", bkt[0], bkt[1])
                    nat_buckets[key] = ([], [])
    flush("iso1mm", iso_paths, iso_ts)
    for key, (paths, tensors) in nat_buckets.items():
        flush("native", paths, tensors)
    if huge_paths:
        hcap = args.huge_max_native
        huge_skipped_fh = None
        if mdir and hcap > 0:
            sig = (
                "-".join(sorted(args.modes))
                + "-s"
                + "x".join((str(x) for x in iso_target))
                + "-v"
                + fingerprint[:12]
            )
            huge_skipped_fh = open(
                os.path.join(mdir, f"latent.{sig}.rank{rank}.huge_skipped"),
                "a",
                encoding="utf-8",
            )
        th = time.time()
        print(
            f"{pfx} huge end-pass: {len(huge_paths)} files, batch=1, hcap={hcap or 'none'}",
            flush=True,
        )
        for hi, h5_path in enumerate(huge_paths):
            try:
                with h5py.File(h5_path, "r") as f:
                    arr = np.asarray(f["image/native"], dtype=np.float32)
            except (OSError, KeyError) as e:
                print(f"{pfx} WARN huge skip {h5_path}: {e}", flush=True)
                continue
            x = to_vae_input(native_preprocess(arr, os.path.basename(h5_path)))
            vox = x.shape[-3] * x.shape[-2] * x.shape[-1]
            if hcap > 0 and vox > hcap:
                stats["huge_skipped"] += 1
                if huge_skipped_fh is not None:
                    huge_skipped_fh.write(h5_path + "\n")
                    huge_skipped_fh.flush()
                print(
                    f"{pfx} WARN huge over hcap, skip {h5_path} padded={tuple(x.shape[-3:])} vox={vox} > {hcap}",
                    flush=True,
                )
                continue
            flush("native", [h5_path], [x])
            stats["huge_written"] += 1
            if hi % 50 == 0:
                rate = (hi + 1) / max(1e-06, time.time() - th)
                print(
                    f"{pfx} huge {hi + 1}/{len(huge_paths)} ({rate:.2f} files/s) padded={tuple(x.shape[-3:])} vox={vox}",
                    flush=True,
                )
        if huge_skipped_fh is not None:
            huge_skipped_fh.close()
    if manifest_fh is not None:
        manifest_fh.close()
    print(
        f"{pfx} DONE shard={len(shard)} written={stats['written']} skipped_files={stats['skipped_files']} skipped_modes={stats['skipped_modes']} missing={stats['missing']} huge_deferred={stats['huge_deferred']} huge_written={stats['huge_written']} huge_skipped={stats['huge_skipped']} skipped_done={stats['skipped_done']} elapsed={time.time() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
