from __future__ import annotations
import hashlib
import os
import shutil
from typing import Optional, Sequence
import numpy as np
import torch

FID_WEIGHTS_FILENAME = "pt_inception-2015-12-05-6726825d.pth"


def ensure_fid_weights_cached(weights_path: Optional[str]) -> None:
    if not weights_path:
        raise ValueError(
            "Optional slice-FID requires explicit local fid_weights; automatic downloads are disabled."
        )
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"fid_weights not found: {weights_path}")
    hf_home = os.environ.setdefault("HF_HOME", "/working/huggingface_cache")
    torch.hub.set_dir(os.path.join(hf_home, "hub", "uniglioma-fid-cache"))
    ckpt_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    dst = os.path.join(ckpt_dir, FID_WEIGHTS_FILENAME)
    if not os.path.isfile(dst):
        shutil.copy(weights_path, dst)


try:
    import h5py
except Exception:
    h5py = None
try:
    from pytorch_fid.fid_score import calculate_frechet_distance
    from pytorch_fid.inception import InceptionV3

    _HAS_FID = True
except Exception:
    _HAS_FID = False


def center_crop_pad(arr: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(target, dtype=arr.dtype)
    src_sl, dst_sl = ([], [])
    for a, t in zip(arr.shape, target):
        if a >= t:
            s = (a - t) // 2
            src_sl.append(slice(s, s + t))
            dst_sl.append(slice(0, t))
        else:
            d = (t - a) // 2
            src_sl.append(slice(0, a))
            dst_sl.append(slice(d, d + a))
    out[tuple(dst_sl)] = arr[tuple(src_sl)]
    return out


def _minmax(s: np.ndarray) -> np.ndarray:
    lo, hi = (float(s.min()), float(s.max()))
    return (s - lo) / (hi - lo + 1e-08)


class SliceFID:
    def __init__(
        self,
        device: torch.device,
        *,
        dims: int = 2048,
        slice_axis: int = 2,
        slice_frac: tuple[float, float] = (0.2, 0.85),
        feat_batch: int = 64,
        weights_path: Optional[str] = None,
    ):
        if not _HAS_FID:
            raise RuntimeError(
                "pytorch-fid not installed. `pip install pytorch-fid` in the training venv, or set infer.fid_enable=false."
            )
        ensure_fid_weights_cached(weights_path)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device).eval()
        self.device = device
        self.slice_axis = slice_axis
        self.slice_frac = slice_frac
        self.feat_batch = feat_batch

    def _slice_lohi(self, n: int) -> tuple[int, int]:
        lo = int(self.slice_frac[0] * n)
        hi = int(self.slice_frac[1] * n)
        return (lo, max(hi, lo + 1))

    def _volume_slices(self, vol: np.ndarray) -> np.ndarray:
        lo, hi = self._slice_lohi(vol.shape[self.slice_axis])
        sl = [_minmax(np.take(vol, k, axis=self.slice_axis)) for k in range(lo, hi)]
        return np.stack(sl).astype(np.float32)

    @torch.no_grad()
    def _features(self, slices: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(slices), self.feat_batch):
            chunk = slices[i : i + self.feat_batch]
            t = torch.from_numpy(chunk).float().unsqueeze(1).repeat(1, 3, 1, 1)
            t = t.to(self.device)
            f = self.model(t)[0]
            out.append(f.squeeze(-1).squeeze(-1).cpu().numpy())
        return np.concatenate(out, 0)

    def stats_from_volumes(
        self, vols: Sequence[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, int]:
        slices = np.concatenate([self._volume_slices(v) for v in vols], 0)
        feats = self._features(slices)
        mu = feats.mean(0)
        sigma = np.cov(feats, rowvar=False)
        return (mu, sigma, len(slices))

    def _ref_cache_key(self, h5_paths: Sequence[str], spatial_size) -> str:
        raw = "|".join(sorted((os.path.basename(p) for p in h5_paths)))
        raw += f"|sp={tuple(spatial_size)}|ax={self.slice_axis}|fr={self.slice_frac}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def real_stats(
        self,
        h5_paths: Sequence[str],
        spatial_size,
        cache_dir: str,
        image_key: str = "image/iso1mm",
    ) -> tuple[np.ndarray, np.ndarray]:
        os.makedirs(cache_dir, exist_ok=True)
        key = self._ref_cache_key(h5_paths, spatial_size)
        cache_path = os.path.join(cache_dir, f"fid_real_{key}.npz")
        if os.path.isfile(cache_path):
            d = np.load(cache_path)
            return (d["mu"], d["sigma"])
        spatial_size = tuple((int(x) for x in spatial_size))
        vols = []
        for p in h5_paths:
            with h5py.File(p, "r") as f:
                if image_key not in f:
                    continue
                img = np.asarray(f[image_key], dtype=np.float32)
            vols.append(center_crop_pad(img, spatial_size))
        if not vols:
            raise RuntimeError(
                f"No '{image_key}' found in any of the {len(h5_paths)} ref H5s."
            )
        mu, sigma, n = self.stats_from_volumes(vols)
        np.savez(cache_path, mu=mu, sigma=sigma, n=n)
        return (mu, sigma)

    def fid_against_real(
        self,
        gen_vols: Sequence[np.ndarray],
        real_mu: np.ndarray,
        real_sigma: np.ndarray,
    ) -> float:
        mu, sigma, _ = self.stats_from_volumes(gen_vols)
        return float(calculate_frechet_distance(mu, sigma, real_mu, real_sigma))
