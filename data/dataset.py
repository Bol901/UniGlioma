"""H5-backed task sampling, structured conditioning and batch collation."""

from __future__ import annotations
import functools
import glob
import hashlib
import json
import os
import random
from typing import Any, Optional
import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.h5_io import (
    h5_axis_spacing,
    center_crop_pad,
    latent_size_tag,
    resolve_latent_paths,
    resolve_mask_latent_paths,
    read_h5_brain_mask,
    resolve_sample_h5_path,
    require_h5_path,
    vae_ckpt_fingerprint,
    validate_latent_attrs,
)
from utils.ref_synth import (
    apply_mask,
    downsample_mask_max,
    gaussian_blur,
    healthy_inpaint_mask,
    tumor_inpaint_mask,
    thick_slice_lowres,
    sr_sampling_geometry,
    random_affine_thick_lowres,
    inplane_gaussian_blur_low,
    uniform_undersample,
    bbox_fov_crop,
    periodic_nod_motion,
)
from utils.prompts import (
    _norm_quality,
    _norm_view,
    get_seg_prompt_variants,
    acq_stem,
    compose as compose_prompt,
    n_desc_variants,
    n_inpaint_variants,
    n_synthetic_input_variants,
    n_task_variants,
    seg_prompt,
    slotted_variant,
    PARAM_SR_TASKS,
    SR_MM_GRID,
)

SR_FACTORS_DEFAULT = (2, 3, 4, 5, 6, 7, 8, 9, 10)
SR_RELATIVE_DEFAULT = tuple(((i, o) for i in range(3, 11) for o in range(2, i)))
from utils.tasks import (
    CANONICAL_TASKS,
    LABEL_REQUIRED_TASKS,
    SEG_TARGET_MODALITY_ID,
    validate_task_weights,
)


def _raw_path_from_item(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("path", item.get("image", item))).strip()
    return str(item).strip()


def string_key(s: str) -> str:
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()


class StringFeatureStore:
    def __init__(self, h5_path: str, expected_fingerprint: str, expected_hidden: int):
        if os.path.isdir(h5_path):
            parts = sorted(glob.glob(os.path.join(h5_path, "part_*.h5")))
            if not parts:
                raise FileNotFoundError(
                    f"prompt_feature_h5 dir has no part_*.h5: {h5_path}. Run scripts/build_prompts.py + (torchrun) encode_prompts.py."
                )
        elif os.path.isfile(h5_path):
            parts = [h5_path]
        else:
            raise FileNotFoundError(
                f"prompt_feature_h5 not found: {h5_path}. Run scripts/build_prompts.py + scripts/encode_prompts.py."
            )
        self.h5_path = h5_path
        self.part_paths = parts
        keys_all, off_all, len_all, part_all = ([], [], [], [])
        for pi, p in enumerate(parts):
            with h5py.File(p, "r") as f:
                fp = str(f.attrs.get("fingerprint", ""))
                if fp != expected_fingerprint:
                    raise ValueError(
                        f"{p}: fingerprint {fp!r} != cfg {expected_fingerprint!r} (different Qwen3-VL ckpt / chat-template / layer); regenerate."
                    )
                hid = int(f.attrs.get("hidden_size", -1))
                if hid != int(expected_hidden):
                    raise ValueError(
                        f"{p}: hidden_size={hid} != cfg mmdit.text_hidden={expected_hidden}."
                    )
                k = f["keys"][()]
                keys_all.append(np.asarray(k, dtype="S40"))
                off_all.append(np.asarray(f["offsets"][()], dtype=np.int64))
                len_all.append(np.asarray(f["lengths"][()], dtype=np.int32))
                part_all.append(np.full(len(k), pi, dtype=np.int16))
        keys = np.concatenate(keys_all) if keys_all else np.empty(0, dtype="S40")
        offs = np.concatenate(off_all) if off_all else np.empty(0, dtype=np.int64)
        lens = np.concatenate(len_all) if len_all else np.empty(0, dtype=np.int32)
        prts = np.concatenate(part_all) if part_all else np.empty(0, dtype=np.int16)
        order = np.argsort(keys, kind="stable")
        self._keys = np.ascontiguousarray(keys[order])
        self._off = np.ascontiguousarray(offs[order])
        self._len = np.ascontiguousarray(lens[order])
        self._part = np.ascontiguousarray(prts[order])
        self._files: dict[int, "h5py.File"] = {}
        self._pid: Optional[int] = None

    def __len__(self) -> int:
        return int(self._keys.shape[0])

    def _lookup(self, key: bytes):
        i = int(np.searchsorted(self._keys, np.bytes_(key)))
        if i >= self._keys.shape[0] or self._keys[i] != key:
            return None
        return (int(self._part[i]), int(self._off[i]), int(self._len[i]))

    def _feat(self, part_idx: int) -> "h5py.Dataset":
        pid = os.getpid()
        if self._pid != pid:
            self._files = {}
            self._pid = pid
        f = self._files.get(part_idx)
        if f is None:
            f = h5py.File(self.part_paths[part_idx], "r")
            self._files[part_idx] = f
        return f["feat"]

    def get(self, text: str) -> torch.Tensor:
        ent = self._lookup(string_key(text).encode("ascii"))
        if ent is None:
            raise KeyError(
                f"prompt not in unified cache: {str(text)[:80]!r}. Re-run build_all_prompts.py + encode_prompts.py over current data."
            )
        part_idx, start, length = ent
        return torch.from_numpy(
            np.asarray(self._feat(part_idx)[start : start + length])
        )


class H5LatentPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_path: str,
        h5_root: str,
        train: bool = True,
        h5_mode: str = "iso1mm",
        h5_mix_prob: float = 0.5,
        skip_missing_h5: bool = False,
        *,
        expected_fingerprint: str,
        expected_spatial_size,
        expected_latent_shape,
        prompt_store: "StringFeatureStore",
        return_logvar: bool = False,
        return_label: bool = False,
        label_h5_root: Optional[str] = None,
        label_full_shape: Optional[tuple] = None,
        seg_num_classes: int = 4,
        num_refs: int = 0,
        cond_extra_channels: int = 0,
        modality_names: Optional[dict] = None,
        empty_text_uses_modality: bool = False,
    ):
        if prompt_store is None:
            raise ValueError("H5LatentPairDataset requires prompt_store")
        self.num_refs = int(num_refs)
        self.cond_extra_channels = int(cond_extra_channels)
        self.modality_names = {
            int(k): str(v) for k, v in (modality_names or {}).items()
        }
        self.empty_text_uses_modality = bool(empty_text_uses_modality)
        if return_label:
            if not label_h5_root:
                raise ValueError("return_label=True requires label_h5_root")
            if not label_full_shape:
                raise ValueError("return_label=True requires label_full_shape")
            if h5_mode != "iso1mm":
                raise ValueError(
                    f"seg head (return_label=True) requires h5_mode='iso1mm', got {h5_mode!r}; native/mix latents would misalign the label."
                )
        self.return_label = bool(return_label)
        self.h5_root = h5_root
        self.label_h5_root = label_h5_root
        self.label_full_shape = (
            tuple((int(x) for x in label_full_shape)) if label_full_shape else None
        )
        self.seg_num_classes = int(seg_num_classes)
        with open(json_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise ValueError(f"{json_path}: top level must be a list")
        self.train = train
        self.h5_mode = h5_mode
        self.h5_mix_prob = float(h5_mix_prob)
        self.return_logvar = bool(return_logvar)
        self.expected_fingerprint = expected_fingerprint
        self.expected_spatial_size = tuple((int(x) for x in expected_spatial_size))
        self.expected_latent_shape = tuple((int(x) for x in expected_latent_shape))
        self.prompt_store = prompt_store

        def _resolve(p):
            return resolve_sample_h5_path({"path": p}, h5_root=h5_root)

        records: list[dict[str, Any]] = []
        missing = 0
        for e in entries:
            if not isinstance(e, dict):
                raise ValueError(f"{json_path}: entry must be a dict, got {type(e)}")
            if "images" in e:
                imgs = e["images"]
                if not isinstance(imgs, list) or not imgs:
                    raise ValueError(
                        f"{json_path}: 'images' must be a non-empty list: {e!r}"
                    )
                images = [
                    {
                        "path": im.get("path") or im.get("image"),
                        "modality": int(im["modality"]),
                    }
                    for im in imgs
                ]
                texts = list(e.get("texts", []) or [])
            else:
                for k in ("tgt_path", "tgt_modality"):
                    if k not in e:
                        raise ValueError(f"{json_path}: entry missing {k!r}: {e!r}")
                images = [{"path": e["tgt_path"], "modality": int(e["tgt_modality"])}]
                if "ref_path" in e:
                    images.append(
                        {
                            "path": e["ref_path"],
                            "modality": int(e.get("ref_modality", e["tgt_modality"])),
                        }
                    )
                texts = []
                if e.get("prompt") is not None:
                    texts = [str(e["prompt"])]
                elif (
                    isinstance(e.get("prompt_variants"), list) and e["prompt_variants"]
                ):
                    texts = [str(p) for p in e["prompt_variants"]]
                    e = {**e, "_prompt_variants": True}
            for im in images:
                im["path"] = require_h5_path(im["path"])
                im["h5"] = _resolve(im["path"])
            label_path = e.get("label_path")
            label_h5 = (
                resolve_sample_h5_path(
                    {"path": label_path}, h5_root=label_h5_root or h5_root
                )
                if label_path
                else None
            )
            if skip_missing_h5:
                need = [images[0]["h5"]] + [
                    im["h5"] for im in images[1 : 1 + self.num_refs]
                ]
                if not all((os.path.isfile(p) for p in need)):
                    missing += 1
                    continue
                if label_h5 is not None and (not os.path.isfile(label_h5)):
                    label_h5 = None
            records.append(
                {
                    "images": images,
                    "texts": [str(t) for t in texts],
                    "prompt_variants": bool(e.get("_prompt_variants", False)),
                    "label_path": label_path,
                    "label_h5": label_h5,
                }
            )
        if not records:
            raise RuntimeError(f"No records loaded from {json_path}")
        self.records = records
        self.missing_h5 = missing
        self._validated: set[str] = set()

    def __len__(self) -> int:
        return len(self.records)

    def _select_mode(self) -> str:
        if self.h5_mode in ("iso1mm", "native"):
            return self.h5_mode
        if self.h5_mode == "mix":
            return "iso1mm" if random.random() < self.h5_mix_prob else "native"
        raise ValueError(f"Unknown h5_mode={self.h5_mode!r}")

    def _build_text(self, rec: dict[str, Any], target_mod: int):
        texts = rec["texts"]
        if texts:
            chosen = random.choice(texts) if self.train and len(texts) > 1 else texts[0]
            return (self.prompt_store.get(chosen), True, chosen)
        name = self.modality_names.get(int(target_mod))
        if name is None:
            raise KeyError(
                f"empty texts but no modality_names entry for modality {target_mod}; configs must provide modality_names as the empty-text fallback."
            )
        return (self.prompt_store.get(name), bool(self.empty_text_uses_modality), name)

    def _read_label_both(
        self, label_h5: str
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        with h5py.File(label_h5, "r") as f:
            key = "image/iso1mm"
            if key not in f:
                raise KeyError(
                    f"{label_h5}: missing {key}. Build the label H5 with the required discrete labels at image/iso1mm first."
                )
            raw = np.asarray(f[key])
        D, H, W = self.expected_spatial_size
        base = center_crop_pad(raw.astype(np.int16), self.expected_spatial_size)
        cond = None
        if self.cond_extra_channels > 0:
            C = self.cond_extra_channels
            clab = np.clip(base, 0, C - 1)
            ld, lh, lw = self.expected_latent_shape
            bd, bh, bw = (D // ld, H // lh, W // lw)
            soft = np.empty((C, ld, lh, lw), dtype=np.float32)
            for c in range(C):
                soft[c] = (
                    (clab == c).reshape(ld, bd, lh, bh, lw, bw).mean(axis=(1, 3, 5))
                )
            cond = torch.from_numpy(soft)
        lab = None
        if self.return_label:
            K = self.seg_num_classes
            ld, lh, lw = self.label_full_shape
            fac = (D // ld, H // lh, W // lw)
            clab = np.clip(base, 0, K - 1)
            if fac == (1, 1, 1):
                larr = clab.astype(np.int16)
            else:
                bd, bh, bw = fac
                frac = np.empty((K, ld, lh, lw), dtype=np.float32)
                for c in range(K):
                    frac[c] = (
                        (clab == c).reshape(ld, bd, lh, bh, lw, bw).mean(axis=(1, 3, 5))
                    )
                larr = frac.argmax(0).astype(np.int16)
            lab = torch.from_numpy(larr)
        return (cond, lab)

    def _read_label_full(self, label_h5: str) -> torch.Tensor:
        with h5py.File(label_h5, "r") as f:
            key = "image/iso1mm"
            if key not in f:
                raise KeyError(f"{label_h5}: missing {key}.")
            raw = np.asarray(f[key])
        D, H, W = self.expected_spatial_size
        base = center_crop_pad(raw.astype(np.int16), self.expected_spatial_size)
        K = self.seg_num_classes
        ld, lh, lw = self.label_full_shape
        clab = np.clip(base, 0, K - 1)
        if (D // ld, H // lh, W // lw) == (1, 1, 1):
            larr = clab.astype(np.int16)
        else:
            bd, bh, bw = (D // ld, H // lh, W // lw)
            frac = np.empty((K, ld, lh, lw), dtype=np.float32)
            for c in range(K):
                frac[c] = (
                    (clab == c).reshape(ld, bd, lh, bh, lw, bw).mean(axis=(1, 3, 5))
                )
            larr = frac.argmax(0).astype(np.int16)
        return torch.from_numpy(larr)

    def _read_latent(
        self, h5_path: str, mode: str
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        with h5py.File(h5_path, "r") as f:
            base, attrs_path = resolve_latent_paths(f, mode, self.expected_spatial_size)
            if base is None:
                raise KeyError(
                    f"{h5_path}: no latent for mode={mode!r} at size {latent_size_tag(self.expected_spatial_size)} (nor legacy); precompute it for this spatial_size."
                )
            if h5_path not in self._validated:
                validate_latent_attrs(
                    f,
                    h5_path,
                    attrs_path=attrs_path,
                    expected_fingerprint=self.expected_fingerprint,
                    expected_spatial_size=self.expected_spatial_size,
                    expected_latent_shape=self.expected_latent_shape,
                )
                self._validated.add(h5_path)
            mean = torch.from_numpy(np.asarray(f[f"{base}/mean"]))
            logvar = None
            if self.return_logvar:
                logvar = torch.from_numpy(np.asarray(f[f"{base}/logvar"]))
        return (mean, logvar)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        mode = self._select_mode()
        images = rec["images"]
        target = images[0]
        target_mod = int(target["modality"])
        tgt_mean, tgt_logvar = self._read_latent(target["h5"], mode)
        avail = images[1 : 1 + self.num_refs]
        has_ref = len(avail) > 0
        ref_latents, ref_modality_ids = ([], [])
        for j in range(self.num_refs):
            if j < len(avail):
                rm, _ = self._read_latent(avail[j]["h5"], mode)
                ref_latents.append(rm)
                ref_modality_ids.append(int(avail[j]["modality"]))
            else:
                ref_latents.append(torch.zeros_like(tgt_mean))
                ref_modality_ids.append(0)
        text_emb, has_text, prompt_str = self._build_text(rec, target_mod)
        has_label = rec["label_h5"] is not None
        out = {
            "latent_mean": tgt_mean,
            "modality_id": target_mod,
            "ref_latents": ref_latents,
            "ref_modality_ids": ref_modality_ids,
            "has_ref": has_ref,
            "note_idx": -1,
            "path": target["path"],
            "text_emb": text_emb,
            "has_text": has_text,
            "has_label": has_label,
            "prompt": prompt_str,
        }
        if tgt_logvar is not None:
            out["latent_logvar"] = tgt_logvar
        need_cond = self.cond_extra_channels > 0
        need_label = self.return_label
        cond_extra = label_full = None
        if has_label and (need_cond or need_label):
            cond_extra, label_full = self._read_label_both(rec["label_h5"])
        if need_cond:
            out["cond_extra"] = (
                cond_extra
                if cond_extra is not None
                else torch.zeros(self.cond_extra_channels, *self.expected_latent_shape)
            )
        if need_label:
            out["label_full"] = (
                label_full
                if label_full is not None
                else torch.zeros(self.label_full_shape, dtype=torch.int16)
            )
        return out


def pair_collate(
    batch: list[dict[str, Any]], *, fixed_text_len: Optional[int] = None
) -> dict[str, Any]:
    embs = [b["text_emb"] for b in batch]
    B = len(embs)
    H = embs[0].shape[1]
    L_target = (
        int(fixed_text_len)
        if fixed_text_len and fixed_text_len > 0
        else max((e.shape[0] for e in embs))
    )
    text_emb = embs[0].new_zeros(B, L_target, H)
    text_mask = torch.zeros(B, L_target, dtype=torch.long)
    for i, e in enumerate(embs):
        L = min(e.shape[0], L_target)
        text_emb[i, :L] = e[:L]
        text_mask[i, :L] = 1
    n_refs = len(batch[0]["ref_latents"])
    ref_latents = [
        torch.stack([b["ref_latents"][k] for b in batch], dim=0) for k in range(n_refs)
    ]
    ref_modality_ids = [
        torch.tensor([int(b["ref_modality_ids"][k]) for b in batch], dtype=torch.long)
        for k in range(n_refs)
    ]
    out = {
        "latent_mean": torch.stack([b["latent_mean"] for b in batch], dim=0),
        "modality_id": torch.tensor(
            [b["modality_id"] for b in batch], dtype=torch.long
        ),
        "note_idx": torch.tensor([b["note_idx"] for b in batch], dtype=torch.long),
        "path": [b["path"] for b in batch],
        "text_emb": text_emb,
        "text_mask": text_mask,
        "ref_latents": ref_latents,
        "ref_modality_ids": ref_modality_ids,
        "has_ref": torch.tensor([bool(b["has_ref"]) for b in batch], dtype=torch.bool),
        "has_text": torch.tensor(
            [bool(b["has_text"]) for b in batch], dtype=torch.bool
        ),
        "has_label": torch.tensor(
            [bool(b["has_label"]) for b in batch], dtype=torch.bool
        ),
    }
    if "latent_logvar" in batch[0]:
        out["latent_logvar"] = torch.stack([b["latent_logvar"] for b in batch], dim=0)
    if "cond_extra" in batch[0]:
        out["cond_extra"] = torch.stack([b["cond_extra"] for b in batch], dim=0)
    if "label_full" in batch[0]:
        out["label_full"] = torch.stack([b["label_full"] for b in batch], dim=0)
    return out


def _build_pair_loader(
    json_path: str,
    cfg: dict[str, Any],
    *,
    train: bool,
    world_size: int,
    rank: int,
    load_labels: bool = True,
) -> DataLoader:
    if not cfg.get("h5_root"):
        raise ValueError(
            "UniGlioma requires `h5_root` pointing at the precomputed latent tree."
        )
    if not cfg.get("prompt_feature_h5"):
        raise ValueError(
            "requires `prompt_feature_h5` (the unified StringFeatureStore — a dir of part_*.h5 or a single .h5; build via scripts/encode_prompts.py)."
        )
    from models.qwen3vl_text import qwen3vl_fingerprint

    _fp = qwen3vl_fingerprint(
        cfg["qwen3vl_path"],
        chat_template=bool(cfg.get("qwen3vl_chat_template", True)),
        add_gen=bool(cfg.get("qwen3vl_add_generation_prompt", True)),
        layer=int(cfg.get("qwen3vl_layer", -1)),
    )
    _hid = int(cfg["mmdit"]["text_hidden"])
    prompt_store = StringFeatureStore(cfg["prompt_feature_h5"], _fp, _hid)
    seg_cfg = cfg["mmdit"].get("seg_head", {}) or {}
    configured_tasks = set((cfg.get("task_weights") or {}).keys())
    return_label = bool(
        seg_cfg.get("enabled", False) or configured_tasks & LABEL_REQUIRED_TASKS
    )
    ds_factor = int(seg_cfg.get("downsample", 1))
    full_shape_cfg = tuple(seg_cfg.get("full_shape", cfg["spatial_size"]))
    label_full_shape = tuple((x // ds_factor for x in full_shape_cfg))
    if return_label:
        for s_ax, l_ax in zip(tuple(cfg["spatial_size"]), label_full_shape):
            if l_ax <= 0 or s_ax % l_ax != 0:
                raise ValueError(
                    f"seg label_full_shape={label_full_shape} must evenly divide spatial_size={tuple(cfg['spatial_size'])} (downsample={ds_factor})"
                )
    task_weights = cfg.get("task_weights")
    common = dict(
        json_path=json_path,
        h5_root=cfg["h5_root"],
        train=train,
        h5_mode=cfg.get("h5_mode", "iso1mm"),
        h5_mix_prob=float(cfg.get("h5_mix_prob", 0.5)),
        skip_missing_h5=bool(cfg.get("skip_missing_h5", False)),
        expected_fingerprint=vae_ckpt_fingerprint(cfg["vae_ckpt"]),
        expected_spatial_size=tuple(cfg["spatial_size"]),
        expected_latent_shape=tuple(cfg["mmdit"]["latent_shape"]),
        return_logvar=bool(cfg.get("latent_sample", False)),
        prompt_store=prompt_store,
        return_label=return_label and load_labels,
        label_h5_root=seg_cfg.get("label_h5_root"),
        label_full_shape=label_full_shape,
        seg_num_classes=int(seg_cfg.get("num_classes", 4)),
        cond_extra_channels=int(cfg["mmdit"].get("cond_extra_channels", 0)),
        modality_names={
            int(k): v for k, v in (cfg.get("modality_names", {}) or {}).items()
        },
        empty_text_uses_modality=bool(cfg.get("empty_text_uses_modality", False)),
    )
    if task_weights:
        acq_lookup = {}
        acq_json = cfg.get("acq_lookup", "local_data/acquisition.json")
        if acq_json and os.path.isfile(acq_json):
            with open(acq_json, "r", encoding="utf-8") as _af:
                acq_lookup = json.load(_af)
        else:
            print(
                f"[data] WARN: acq_lookup '{acq_json}' missing → observed metadata stays unknown. Run the data-format documentation.",
                flush=True,
            )
        ds = MultiTaskDataset(
            **common,
            num_refs=int(cfg["mmdit"].get("max_refs", cfg.get("num_refs", 0))),
            task_weights=dict(task_weights),
            seg_prompt_variants_path=cfg.get("seg_prompt_variants_path"),
            latent_channels=int(cfg["mmdit"]["in_channels"]),
            sr_factors=tuple(cfg.get("sr_factors", SR_FACTORS_DEFAULT)),
            sr_factor_weights=cfg.get("sr_factor_weights"),
            sr_axis_weights=tuple(cfg.get("sr_axis_weights", (2, 1, 1))),
            sr_affine_enabled=bool(cfg.get("sr_affine_enabled", True)),
            sr_affine_max_angle_deg=cfg.get("sr_affine_max_angle_deg", 15.0),
            sr_up_orders=tuple(cfg.get("sr_up_orders", (0, 1, 3))),
            sr_up_order_weights=tuple(cfg.get("sr_up_order_weights", (1, 1, 1))),
            sr_thin_prob=float(cfg.get("sr_thin_prob", 0.7)),
            sr_absolute_thin_prob=float(
                cfg.get("sr_absolute_thin_prob", cfg.get("sr_thin_prob", 0.7))
            ),
            sr_relative_pairs=tuple(
                (tuple(x) for x in cfg.get("sr_relative_pairs", SR_RELATIVE_DEFAULT))
            ),
            thin_target_one_mm_prob=float(cfg.get("thin_target_one_mm_prob", 0.5)),
            acq_thickness_drop_prob=float(cfg.get("acq_thickness_drop_prob", 0.15)),
            prompt_exact_input_mm_prob=float(
                cfg.get("prompt_exact_input_mm_prob", 0.7)
            ),
            inpaint_dilate=int(cfg.get("inpaint_dilate", 2)),
            inpaint_size_bin_edges=tuple(
                cfg.get("inpaint_size_bin_edges", (0.01, 0.025))
            ),
            inpaint_size_bin_weights=tuple(
                cfg.get("inpaint_size_bin_weights", (1, 1, 1))
            ),
            inpaint_size_bin_names=tuple(
                cfg.get("inpaint_size_bin_names", ("small", "medium", "large"))
            ),
            inpaint_large_oversample=float(cfg.get("inpaint_large_oversample", 2.0)),
            inpaint_case_max_tries=int(cfg.get("inpaint_case_max_tries", 64)),
            inpaint_place_max_tries=int(cfg.get("inpaint_place_max_tries", 48)),
            inpaint_min_brain_frac=float(cfg.get("inpaint_min_brain_frac", 0.6)),
            inpaint_margin=int(cfg.get("inpaint_margin", 2)),
            inpaint_contralateral_large=bool(
                cfg.get("inpaint_contralateral_large", True)
            ),
            inpaint_contralateral_axis=int(cfg.get("inpaint_contralateral_axis", 2)),
            inpaint_placement_fail_blacklist=int(
                cfg.get("inpaint_placement_fail_blacklist", 3)
            ),
            inpaint_pseudo_healthy_prob=float(
                cfg.get("inpaint_pseudo_healthy_prob", 0.0)
            ),
            inpaint_lesion_prob=float(cfg.get("inpaint_lesion_prob", 0.5)),
            inpaint_cavity_class_ids=tuple(cfg.get("inpaint_cavity_class_ids", (1,))),
            inpaint_tumor_class_ids=tuple(cfg.get("inpaint_tumor_class_ids", (2,))),
            inpaint_tumor_dominance_frac=float(
                cfg.get("inpaint_tumor_dominance_frac", 0.15)
            ),
            deblur_sigma_range=tuple(cfg.get("deblur_sigma_range", (0.8, 2.5))),
            restore_composite_prob=float(cfg.get("restore_composite_prob", 0.3)),
            dealias_factors=tuple(cfg.get("dealias_factors", (2, 3, 4))),
            dealias_factor_weights=tuple(cfg.get("dealias_factor_weights", (1, 1, 1))),
            dealias_axis_weights=tuple(cfg.get("dealias_axis_weights", (1, 1, 1))),
            whole_brain_axis_weights=tuple(
                cfg.get("whole_brain_axis_weights", (1, 1, 1))
            ),
            whole_brain_affine_max_angle_deg=float(
                cfg.get("whole_brain_affine_max_angle_deg", 10.0)
            ),
            whole_brain_nonaxial_remove_range=tuple(
                cfg.get("whole_brain_nonaxial_remove_range", (1.0 / 3.0, 0.5))
            ),
            whole_brain_axial_remove_range=tuple(
                cfg.get("whole_brain_axial_remove_range", (0.2, 0.5))
            ),
            motion_n_poses=int(cfg.get("motion_n_poses", 3)),
            motion_planes_per_pose=int(cfg.get("motion_planes_per_pose", 6)),
            motion_max_angle_deg=float(cfg.get("motion_max_angle_deg", 3.0)),
            motion_trans_ap_mm=float(cfg.get("motion_trans_ap_mm", 2.5)),
            motion_trans_si_mm=float(cfg.get("motion_trans_si_mm", 0.75)),
            motion_dc_anchor=int(cfg.get("motion_dc_anchor", 4)),
            motion_corrupt_fraction_range=tuple(
                cfg.get("motion_corrupt_fraction_range", (0.25, 0.5))
            ),
            motion_active_poses=int(cfg.get("motion_active_poses", 0)),
            seg_ref_count_weights=dict(
                cfg.get("seg_ref_count_weights", {1: 1, 2: 1, 3: 1, 4: 1})
            ),
            region_mask_channels=int(cfg["mmdit"].get("region_mask_channels", 0)),
            task_prompts=dict(cfg.get("task_prompts", {})),
            acq_lookup=acq_lookup,
            load_labels=load_labels,
            emit_spatial_conditions=bool(cfg["mmdit"].get("use_ref_slab", True)),
            num_modalities=int(cfg.get("num_modalities", 4)),
        )
        collate_fn = functools.partial(
            multitask_collate,
            fixed_text_len=int(
                cfg.get("text_fixed_length", cfg.get("text_max_length", 512))
            ),
        )
    else:
        ds = H5LatentPairDataset(**common, num_refs=int(cfg.get("num_refs", 0)))
        collate_fn = functools.partial(
            pair_collate,
            fixed_text_len=int(
                cfg.get("text_fixed_length", cfg.get("text_max_length", 512))
            ),
        )
    if getattr(ds, "missing_h5", 0) or rank == 0:
        print(
            f"[data][multi] split={os.path.basename(json_path)} records={len(ds)} missing_h5={ds.missing_h5} task_weights={task_weights}",
            flush=True,
        )
    use_ddp = world_size > 1 and dist.is_available() and dist.is_initialized()
    sampler = None
    if use_ddp:
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=train,
            drop_last=train,
            seed=cfg.get("seed", 42),
        )
    batch_size = int(cfg["batch_size"]) if train else int(cfg.get("val_batch_size", 1))
    num_workers = int(cfg.get("num_workers", 4))
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None and train,
        num_workers=num_workers,
        pin_memory=bool(cfg.get("pin_memory", True)),
        drop_last=train,
        persistent_workers=num_workers > 0,
        prefetch_factor=int(cfg.get("prefetch_factor", 2)) if num_workers > 0 else None,
        collate_fn=collate_fn,
    )


def build_train_pair_loader(
    cfg: dict[str, Any], world_size: int, rank: int, *, load_labels: bool = True
) -> DataLoader:
    return _build_pair_loader(
        cfg["train_json"],
        cfg,
        train=True,
        world_size=world_size,
        rank=rank,
        load_labels=load_labels,
    )


def build_val_pair_loader(
    cfg: dict[str, Any], world_size: int, rank: int, *, load_labels: bool = True
) -> DataLoader:
    return _build_pair_loader(
        cfg["val_json"],
        cfg,
        train=False,
        world_size=world_size,
        rank=rank,
        load_labels=load_labels,
    )


class IneligibleSampleError(RuntimeError):
    pass


MULTITASK_TASKS = CANONICAL_TASKS


class MultiTaskDataset(H5LatentPairDataset):
    def __init__(
        self,
        *args,
        task_weights: Optional[dict] = None,
        seg_prompt_variants_path: Optional[str] = None,
        latent_channels: Optional[int] = None,
        sr_factors=SR_FACTORS_DEFAULT,
        sr_factor_weights=None,
        sr_axis_weights=(2, 1, 1),
        sr_affine_enabled: bool = True,
        sr_affine_max_angle_deg=15.0,
        sr_up_orders=(0, 1, 3),
        sr_up_order_weights=(1, 1, 1),
        sr_thin_prob: float = 0.7,
        sr_absolute_thin_prob: Optional[float] = None,
        sr_relative_pairs=SR_RELATIVE_DEFAULT,
        thin_target_one_mm_prob: float = 0.5,
        acq_thickness_drop_prob: float = 0.15,
        prompt_exact_input_mm_prob: float = 0.7,
        inpaint_dilate: int = 2,
        inpaint_size_bin_edges=(0.01, 0.025),
        inpaint_size_bin_weights=(1, 1, 1),
        inpaint_size_bin_names=("small", "medium", "large"),
        inpaint_large_oversample: float = 2.0,
        inpaint_case_max_tries: int = 64,
        inpaint_place_max_tries: int = 48,
        inpaint_min_brain_frac: float = 0.6,
        inpaint_margin: int = 2,
        inpaint_contralateral_large: bool = True,
        inpaint_contralateral_axis: int = 2,
        inpaint_placement_fail_blacklist: int = 3,
        inpaint_pseudo_healthy_prob: float = 0.0,
        inpaint_lesion_prob: float = 0.5,
        inpaint_cavity_class_ids=(1,),
        inpaint_tumor_class_ids=(2,),
        inpaint_tumor_dominance_frac: float = 0.15,
        deblur_sigma_range=(0.8, 2.5),
        restore_composite_prob: float = 0.3,
        dealias_factors=(2, 3, 4),
        dealias_factor_weights=(1, 1, 1),
        dealias_axis_weights=(1, 1, 1),
        whole_brain_axis_weights=(1, 1, 1),
        whole_brain_affine_max_angle_deg: float = 10.0,
        whole_brain_nonaxial_remove_range=(1.0 / 3.0, 0.5),
        whole_brain_axial_remove_range=(0.2, 0.5),
        motion_n_poses: int = 3,
        motion_planes_per_pose: int = 6,
        motion_max_angle_deg: float = 3.0,
        motion_trans_ap_mm: float = 2.5,
        motion_trans_si_mm: float = 0.75,
        motion_dc_anchor: int = 4,
        motion_corrupt_fraction_range=(0.25, 0.5),
        motion_active_poses: int = 0,
        seg_ref_count_weights=None,
        region_mask_channels: int = 1,
        task_prompts: Optional[dict] = None,
        acq_lookup: Optional[dict] = None,
        load_labels: bool = True,
        emit_spatial_conditions: bool = True,
        num_modalities: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.latent_channels = latent_channels
        self.seg_prompt_variants_path = seg_prompt_variants_path
        if float((task_weights or {}).get("seg", 0)) > 0:
            get_seg_prompt_variants(seg_prompt_variants_path)
        self.load_labels = bool(load_labels)
        self.emit_spatial_conditions = bool(emit_spatial_conditions)
        self.num_modalities = int(num_modalities)
        tw = validate_task_weights(task_weights or {"modality_only": 1.0})
        tot = float(sum(tw.values())) or 1.0
        self.task_names = list(tw.keys())
        self.task_probs = [tw[t] / tot for t in self.task_names]
        if "seg" in self.task_names and (self.num_modalities != 5 or self.num_refs < 4):
            raise ValueError("seg requires num_modalities=5 and mmdit.max_refs>=4")
        if set(self.task_names) & {"inpaint", "whole_brain"} and self.num_refs < 2:
            raise ValueError("inpaint/whole_brain require mmdit.max_refs>=2")
        if not self.emit_spatial_conditions:
            unsupported = set(self.task_names) - {"modality_only"}
            if unsupported:
                raise ValueError(
                    f"mmdit.use_ref_slab=false supports only target-only gen/text_gen; got spatial tasks {sorted(unsupported)}"
                )
        self.sr_factors = tuple((int(f) for f in sr_factors))
        self.sr_factor_weights = (
            tuple((float(w) for w in sr_factor_weights))
            if sr_factor_weights
            else tuple((1.0 for _ in self.sr_factors))
        )
        if len(self.sr_factor_weights) != len(self.sr_factors):
            raise ValueError("sr_factor_weights must match sr_factors length")
        if any((f <= 1 for f in self.sr_factors)) or any(
            (w <= 0 for w in self.sr_factor_weights)
        ):
            raise ValueError(
                "sr factors must be >1 and all sampling weights must be positive"
            )
        self.sr_axis_weights = tuple((float(w) for w in sr_axis_weights or (1, 1, 1)))
        if len(self.sr_axis_weights) != 3:
            raise ValueError(
                "sr_axis_weights must have 3 entries (axis 0/1/2 = S-I/A-P/L-R)"
            )
        if any((w <= 0 for w in self.sr_axis_weights)):
            raise ValueError("sr_axis_weights must all be positive")
        self.sr_absolute_thin_prob = float(
            sr_thin_prob if sr_absolute_thin_prob is None else sr_absolute_thin_prob
        )
        self.sr_thin_prob = self.sr_absolute_thin_prob
        if not 0.0 <= self.sr_absolute_thin_prob <= 1.0:
            raise ValueError("sr_absolute_thin_prob must be in [0,1]")
        self.sr_relative_pairs = tuple(((int(i), int(o)) for i, o in sr_relative_pairs))
        allowed_relative = {(i, o) for i in range(3, 11) for o in range(2, i)}
        if not self.sr_relative_pairs or any(
            (p not in allowed_relative for p in self.sr_relative_pairs)
        ):
            raise ValueError(
                "sr_relative_pairs must be a non-empty subset of (in>out) pairs over 2..10 mm"
            )
        self.thin_target_one_mm_prob = float(thin_target_one_mm_prob)
        self.acq_thickness_drop_prob = float(acq_thickness_drop_prob)
        self.prompt_exact_input_mm_prob = float(prompt_exact_input_mm_prob)
        if not 0.0 <= self.thin_target_one_mm_prob <= 1.0:
            raise ValueError("thin_target_one_mm_prob must be in [0,1]")
        if not 0.0 <= self.acq_thickness_drop_prob <= 1.0:
            raise ValueError("acq_thickness_drop_prob must be in [0,1]")
        if not 0.0 <= self.prompt_exact_input_mm_prob <= 1.0:
            raise ValueError("prompt_exact_input_mm_prob must be in [0,1]")
        self.sr_affine_enabled = bool(sr_affine_enabled)
        self.sr_affine_max_angle_deg = (
            float(sr_affine_max_angle_deg)
            if np.isscalar(sr_affine_max_angle_deg)
            else [float(x) for x in sr_affine_max_angle_deg]
        )
        self.sr_up_orders = tuple((int(x) for x in sr_up_orders))
        self.sr_up_order_weights = tuple((float(x) for x in sr_up_order_weights))
        if len(self.sr_up_orders) != len(self.sr_up_order_weights) or any(
            (x not in (0, 1, 3) for x in self.sr_up_orders)
        ):
            raise ValueError("sr_up_orders must be 0/1/3 and match sr_up_order_weights")
        if any((w <= 0 for w in self.sr_up_order_weights)):
            raise ValueError("sr_up_order_weights must all be positive")
        self.deblur_sigma_range = tuple((float(x) for x in deblur_sigma_range))
        self.restore_composite_prob = float(restore_composite_prob)
        self.dealias_factors = tuple((int(x) for x in dealias_factors))
        self.dealias_factor_weights = tuple((float(x) for x in dealias_factor_weights))
        self.dealias_axis_weights = tuple((float(x) for x in dealias_axis_weights))
        self.whole_brain_axis_weights = tuple(
            (float(x) for x in whole_brain_axis_weights)
        )
        self.whole_brain_affine_max_angle_deg = float(whole_brain_affine_max_angle_deg)
        if not 0.0 <= self.whole_brain_affine_max_angle_deg <= 45.0:
            raise ValueError("whole_brain_affine_max_angle_deg must be in [0,45]")
        self.whole_brain_nonaxial_remove_range = tuple(
            (float(x) for x in whole_brain_nonaxial_remove_range)
        )
        self.whole_brain_axial_remove_range = tuple(
            (float(x) for x in whole_brain_axial_remove_range)
        )
        if len(self.whole_brain_axis_weights) != 3 or any(
            (w <= 0 for w in self.whole_brain_axis_weights)
        ):
            raise ValueError("whole_brain_axis_weights must have 3 positive entries")
        self.motion_n_poses = int(motion_n_poses)
        self.motion_planes_per_pose = int(motion_planes_per_pose)
        self.motion_max_angle_deg = float(motion_max_angle_deg)
        self.motion_trans_ap_mm = float(motion_trans_ap_mm)
        self.motion_trans_si_mm = float(motion_trans_si_mm)
        self.motion_dc_anchor = int(motion_dc_anchor)
        self.motion_corrupt_fraction_range = tuple(
            (float(x) for x in motion_corrupt_fraction_range)
        )
        self.motion_active_poses = int(motion_active_poses)
        if (
            self.motion_n_poses < 2
            or self.motion_planes_per_pose < 1
            or self.motion_dc_anchor < 0
        ):
            raise ValueError(
                "motion_n_poses>=2, motion_planes_per_pose>=1, motion_dc_anchor>=0"
            )
        if self.motion_active_poses and (
            not 0 < self.motion_active_poses < self.motion_n_poses
        ):
            raise ValueError(
                "motion_active_poses must satisfy 0 < active < n_poses (0 = full grid)"
            )
        if (
            len(self.motion_corrupt_fraction_range) != 2
            or not 0.0
            <= self.motion_corrupt_fraction_range[0]
            <= self.motion_corrupt_fraction_range[1]
            <= 1.0
        ):
            raise ValueError("motion_corrupt_fraction_range must lie in [0,1]")
        raw_seg_weights = seg_ref_count_weights or {1: 1, 2: 1, 3: 1, 4: 1}
        self.seg_ref_count_weights = {
            int(k): float(v) for k, v in raw_seg_weights.items() if float(v) > 0
        }
        if not self.seg_ref_count_weights or any(
            (k not in (1, 2, 3, 4) for k in self.seg_ref_count_weights)
        ):
            raise ValueError(
                "seg_ref_count_weights keys must be a non-empty subset of 1..4"
            )
        if (
            len(self.deblur_sigma_range) != 2
            or not 0 < self.deblur_sigma_range[0] <= self.deblur_sigma_range[1]
        ):
            raise ValueError("deblur_sigma_range must satisfy 0 < low <= high")
        if not 0.0 <= self.restore_composite_prob <= 1.0:
            raise ValueError("restore_composite_prob must be in [0, 1]")
        if (
            len(self.dealias_factors) != len(self.dealias_factor_weights)
            or any((x < 2 for x in self.dealias_factors))
            or any((w <= 0 for w in self.dealias_factor_weights))
        ):
            raise ValueError("dealias factors must be >=2 and match their weights")
        for name, weights in (
            ("dealias_axis_weights", self.dealias_axis_weights),
            ("whole_brain_axis_weights", self.whole_brain_axis_weights),
        ):
            if len(weights) != 3 or any((w <= 0 for w in weights)):
                raise ValueError(f"{name} must contain three positive weights")
        if (
            len(self.whole_brain_nonaxial_remove_range) != 2
            or not 0
            < self.whole_brain_nonaxial_remove_range[0]
            <= self.whole_brain_nonaxial_remove_range[1]
            < 1
        ):
            raise ValueError("whole_brain_nonaxial_remove_range must lie in (0,1)")
        if (
            len(self.whole_brain_axial_remove_range) != 2
            or not 0
            < self.whole_brain_axial_remove_range[0]
            <= self.whole_brain_axial_remove_range[1]
            < 1
        ):
            raise ValueError("whole_brain_axial_remove_range must lie in (0,1)")
        self.inpaint_dilate = int(inpaint_dilate)
        self.inpaint_size_bin_edges = tuple((float(x) for x in inpaint_size_bin_edges))
        if any((x <= 0 for x in self.inpaint_size_bin_edges)) or any(
            (
                b <= a
                for a, b in zip(
                    self.inpaint_size_bin_edges, self.inpaint_size_bin_edges[1:]
                )
            )
        ):
            raise ValueError(
                "inpaint_size_bin_edges must be positive and strictly increasing"
            )
        n_inpaint_bins = len(self.inpaint_size_bin_edges) + 1
        base_bin_weights = tuple((float(x) for x in inpaint_size_bin_weights))
        self.inpaint_size_bin_names = tuple((str(x) for x in inpaint_size_bin_names))
        if len(base_bin_weights) != n_inpaint_bins:
            raise ValueError(
                f"inpaint_size_bin_weights needs {n_inpaint_bins} values, got {len(base_bin_weights)}"
            )
        if len(self.inpaint_size_bin_names) != n_inpaint_bins:
            raise ValueError(
                f"inpaint_size_bin_names needs {n_inpaint_bins} values, got {len(self.inpaint_size_bin_names)}"
            )
        if any((x <= 0 for x in base_bin_weights)):
            raise ValueError("inpaint_size_bin_weights must all be positive")
        if float(inpaint_large_oversample) <= 0:
            raise ValueError("inpaint_large_oversample must be positive")
        actual_bin_weights = list(base_bin_weights)
        actual_bin_weights[-1] *= float(inpaint_large_oversample)
        self.inpaint_size_bin_weights = tuple(actual_bin_weights)
        self.inpaint_case_max_tries = int(inpaint_case_max_tries)
        self.inpaint_place_max_tries = int(inpaint_place_max_tries)
        self.inpaint_min_brain_frac = float(inpaint_min_brain_frac)
        self.inpaint_margin = int(inpaint_margin)
        self.inpaint_contralateral_large = bool(inpaint_contralateral_large)
        self.inpaint_contralateral_axis = int(inpaint_contralateral_axis)
        self.inpaint_placement_fail_blacklist = max(
            1, int(inpaint_placement_fail_blacklist)
        )
        self.inpaint_lesion_prob = float(inpaint_lesion_prob)
        if not 0.0 <= self.inpaint_lesion_prob <= 1.0:
            raise ValueError("inpaint_lesion_prob must be in [0, 1]")
        self.inpaint_pseudo_healthy_prob = float(inpaint_pseudo_healthy_prob)
        if not 0.0 <= self.inpaint_pseudo_healthy_prob <= 1.0:
            raise ValueError("inpaint_pseudo_healthy_prob must be in [0, 1]")
        self.inpaint_cavity_class_ids = tuple(
            (int(c) for c in inpaint_cavity_class_ids)
        )
        self.inpaint_tumor_class_ids = tuple((int(c) for c in inpaint_tumor_class_ids))
        self.inpaint_tumor_dominance_frac = float(inpaint_tumor_dominance_frac)
        if not 0.0 <= self.inpaint_tumor_dominance_frac <= 1.0:
            raise ValueError("inpaint_tumor_dominance_frac must be in [0, 1]")
        if self.inpaint_case_max_tries <= 0 or self.inpaint_place_max_tries <= 0:
            raise ValueError("inpaint case/place max tries must be positive")
        if not 0.0 < self.inpaint_min_brain_frac <= 1.0:
            raise ValueError("inpaint_min_brain_frac must be in (0, 1]")
        if self.inpaint_contralateral_axis not in (0, 1, 2):
            raise ValueError("inpaint_contralateral_axis must be 0, 1, or 2")
        self.region_mask_channels = int(region_mask_channels)
        self._labeled_idx = [
            i for i, r in enumerate(self.records) if r.get("label_h5") is not None
        ]
        self._labeled_idx_set = set(self._labeled_idx)
        self._inpaint_idx_by_bin = [[] for _ in range(n_inpaint_bins)]
        self._inpaint_record_bin: dict[int, int] = {}
        self._inpaint_unplaceable_idx: set[int] = set()
        self._inpaint_placement_fail: dict[int, int] = {}
        self._inpaint_too_small = 0
        self.task_prompts = dict(task_prompts or {})
        self._force_task = None
        self._inpaint_void_tumor = False
        self._sr_native_input = False
        self._force_missing_target_mod = None
        self._force_seg_ref_count = None
        self._force_seg_acc: Optional[str] = None
        self._force_seg_mods: Optional[list] = None
        self.acc_index: dict[str, dict[int, int]] = {}
        for i, rec in enumerate(self.records):
            acc = os.path.dirname(str(rec["images"][0]["path"])).strip("/")
            self.acc_index.setdefault(acc, {})[int(rec["images"][0]["modality"])] = i
        self._multimod_idx = [
            i
            for i, rec in enumerate(self.records)
            if len(
                self.acc_index.get(
                    os.path.dirname(str(rec["images"][0]["path"])).strip("/"), {}
                )
            )
            >= 2
        ]
        self._seg_acc_by_count: dict[int, list[str]] = {k: [] for k in range(1, 5)}
        _tw = task_weights or {}
        _needs_mask_scan = (
            float(_tw.get("seg", 0)) > 0 or float(_tw.get("mask_guide", 0)) > 0
        )
        _mask_ok_map: dict = {}
        _seg_target_idx: dict = {}
        if _needs_mask_scan:
            label_h5s = sorted(
                {
                    self.records[i].get("label_h5")
                    for mod_map in self.acc_index.values()
                    for i in mod_map.values()
                    if self.records[i].get("label_h5")
                }
            )
            from concurrent.futures import ThreadPoolExecutor as _TPE

            def _mask_ok(lh: str) -> bool:
                try:
                    with h5py.File(lh, "r") as f:
                        return self._mask_latent_group(f, lh) is not None
                except (OSError, ValueError, RuntimeError, KeyError):
                    return False

            if label_h5s:
                with _TPE(max_workers=24) as ex:
                    _mask_ok_map = {
                        lh: ok for lh, ok in zip(label_h5s, ex.map(_mask_ok, label_h5s))
                    }
        self._mask_ok_map = _mask_ok_map
        self._seg_target_idx = {}
        for acc, mod_map in self.acc_index.items():
            labelled = [
                i
                for i in mod_map.values()
                if _mask_ok_map.get(self.records[i].get("label_h5"), False)
            ]
            if not labelled:
                continue
            self._seg_target_idx[acc] = min(labelled)
            for count in range(1, min(4, len(mod_map)) + 1):
                self._seg_acc_by_count[count].append(acc)
        self._mask_guide_idx = [
            i
            for i, r in enumerate(self.records)
            if r.get("label_h5") and _mask_ok_map.get(r["label_h5"], False)
        ]
        self.acq_lookup = dict(acq_lookup or {})
        self._thin_idx = [i for i, r in enumerate(self.records) if self._rec_is_thin(r)]
        self._thick_idx = [
            i for i, r in enumerate(self.records) if self._rec_is_thick(r)
        ]
        if self.train:
            required_pools = {
                "mask_guide": self._mask_guide_idx,
                "inpaint": self._labeled_idx,
                "missing": self._multimod_idx,
                "seg": [
                    x for values in self._seg_acc_by_count.values() for x in values
                ],
            }
            for thin_task in ("sr", "deblur", "dealias", "whole_brain"):
                if thin_task in self.task_names:
                    required_pools[thin_task] = self._thin_idx
            empty = [
                task
                for task in self.task_names
                if task in required_pools and (not required_pools[task])
            ]
            if empty:
                raise RuntimeError(
                    f"configured tasks have no eligible training records: {empty}"
                )

    def _acq_entry(self, pathlike):
        stem = acq_stem(pathlike)
        if stem in self.acq_lookup:
            return self.acq_lookup[stem] or []
        if not stem.endswith("_bet"):
            return self.acq_lookup.get(stem + "_bet") or []
        return []

    def _acq(self, pathlike) -> tuple[str, str]:
        e = self._acq_entry(pathlike)
        q = e[0] if len(e) > 0 else None
        v = e[1] if len(e) > 1 else None
        return (_norm_quality(q), _norm_view(v))

    def _acq_full(self, pathlike) -> tuple[str, str, Optional[float]]:
        e = self._acq_entry(pathlike)
        q = _norm_quality(e[0] if len(e) > 0 else None)
        v = _norm_view(e[1] if len(e) > 1 else None)
        th = None
        if len(e) > 2:
            try:
                th = float(e[2]) if e[2] is not None else None
            except (TypeError, ValueError):
                th = None
        return (q, v, th)

    def _read_image_full(self, h5_path: str, mode: str) -> np.ndarray:
        with h5py.File(h5_path, "r") as f:
            key = f"image/{mode}"
            arr = np.asarray(f[key]).astype(np.float32)
        return center_crop_pad(arr, self.expected_spatial_size)

    def _read_brain_mask_full(self, h5_path: str, mode: str) -> np.ndarray:
        try:
            return read_h5_brain_mask(
                h5_path,
                mode=mode,
                spatial_size=self.expected_spatial_size,
                require_nonempty=True,
            )
        except (OSError, KeyError, ValueError) as exc:
            raise IneligibleSampleError(
                f"invalid brain mask for {h5_path}: {exc}"
            ) from exc

    def _full_head_path(self, bet_h5: str):
        if not str(bet_h5).endswith("_bet.h5"):
            return None
        cand = str(bet_h5)[: -len("_bet.h5")] + ".h5"
        return cand if os.path.exists(cand) else None

    def _prepare_motion_record(self, rec, mode: str):
        candidates = [rec]
        if self.train:
            candidates.extend(
                (self.records[random.randrange(len(self.records))] for _ in range(16))
            )
        last_reason = "no candidate examined"
        for candidate in candidates:
            target = candidate["images"][0]
            full_h5 = self._full_head_path(target["h5"])
            if full_h5 is None:
                last_reason = f"missing pre-BET sibling for {target['h5']}"
                continue
            try:
                brain_mask = self._read_brain_mask_full(target["h5"], mode)
            except IneligibleSampleError as exc:
                last_reason = str(exc)
                continue
            return (candidate, full_h5, brain_mask)
        raise IneligibleSampleError(f"motion has no eligible record: {last_reason}")

    def _read_label_full_raw(self, label_h5: str) -> np.ndarray:
        with h5py.File(label_h5, "r") as f:
            if "image/iso1mm" not in f:
                raise KeyError(f"{label_h5}: missing image/iso1mm (label H5)")
            raw = np.asarray(f["image/iso1mm"]).astype(np.int16)
        return center_crop_pad(raw, self.expected_spatial_size)

    def _inpaint_bin_for_fraction(self, hole_fraction: float) -> int:
        return int(
            np.searchsorted(
                np.asarray(self.inpaint_size_bin_edges, dtype=np.float64),
                float(hole_fraction),
                side="right",
            )
        )

    def _prepare_whole_brain(self, rec, mode):

        def _load(candidate):
            h5_path = candidate["images"][0]["h5"]
            return (
                self._read_image_full(h5_path, mode),
                self._read_brain_mask_full(h5_path, mode),
            )

        try:
            img, brain = _load(rec)
        except IneligibleSampleError:
            if not self.train:
                raise
            for _ in range(16):
                rec = self.records[random.choice(self._thin_idx)]
                try:
                    img, brain = _load(rec)
                    break
                except IneligibleSampleError:
                    continue
            else:
                raise IneligibleSampleError(
                    "whole_brain: no sampled thin record has a valid explicit brain mask"
                )
        total = int(brain.sum())
        from scipy.ndimage import affine_transform

        wb_rng = np.random.default_rng(random.getrandbits(64))
        wb_ang = self.whole_brain_affine_max_angle_deg
        wb_angles = (
            np.array([wb_rng.uniform(-wb_ang, wb_ang) for _ in range(3)])
            if wb_ang > 0
            else np.zeros(3)
        )
        pad = int(0.3 * max(img.shape)) if wb_ang > 0 else 0
        if pad > 0:
            imgP = np.pad(img, pad, mode="constant")
            brainP = np.pad(brain, pad, mode="constant")
            a = np.deg2rad(wb_angles)
            cx, sx = (np.cos(a[0]), np.sin(a[0]))
            cy, sy = (np.cos(a[1]), np.sin(a[1]))
            cz, sz = (np.cos(a[2]), np.sin(a[2]))
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            c = (np.asarray(imgP.shape, dtype=np.float64) - 1.0) / 2.0
            imgR = affine_transform(
                imgP.astype(np.float64),
                R,
                offset=c - R @ c,
                order=1,
                mode="constant",
                cval=0,
            )
            brainR = affine_transform(
                brainP.astype(np.float64),
                R,
                offset=c - R @ c,
                order=0,
                mode="constant",
                cval=0,
            )
            brainR = brainR > 0.5
        else:
            imgR, brainR, R = (img, brain, None)
            pad = 0
        bidx = np.argwhere(brainR)
        bbox = tuple(((int(bidx[:, a].min()), int(bidx[:, a].max())) for a in range(3)))
        axis = random.choices((0, 1, 2), weights=self.whole_brain_axis_weights, k=1)[0]
        view = self._acq(rec["images"][0]["path"])[1]
        if view == "axial":
            removed_frac = float(random.uniform(*self.whole_brain_axial_remove_range))
            keep_frac = 1.0 - removed_frac
            lo0, hi0 = bbox[axis]
            extent = hi0 - lo0 + 1
            keep = max(1, min(extent, int(round(extent * keep_frac))))
            start = random.randint(lo0, hi0 - keep + 1)
            side = "random"
            degraded, missing = bbox_fov_crop(
                imgR, axis, bbox=bbox, side=side, keep_frac=keep_frac, start=start
            )
        else:
            side = "both"
            removed_frac = float(
                random.uniform(*self.whole_brain_nonaxial_remove_range)
            )
            keep_frac = 1.0 - removed_frac
            start = None
            degraded, missing = bbox_fov_crop(
                imgR,
                axis,
                bbox=bbox,
                side=side,
                keep_frac=keep_frac,
                center_on_image=True,
            )
        if pad > 0:
            Rinv = R.T
            cv = np.asarray(degraded.shape, dtype=np.float64)
            c2 = (cv - 1.0) / 2.0
            go = c2 - Rinv @ c2
            degraded = affine_transform(
                degraded.astype(np.float64),
                Rinv,
                offset=go,
                order=1,
                mode="constant",
                cval=0,
            )
            missing = (
                affine_transform(
                    missing.astype(np.float64),
                    Rinv,
                    offset=go,
                    order=0,
                    mode="constant",
                    cval=0,
                )
                > 0.5
            )
            sl = tuple((slice(pad, -pad if pad else None) for _ in range(3)))
            degraded = np.ascontiguousarray(degraded[sl]).clip(0.0, 1.0)
            missing = np.ascontiguousarray(missing[sl])
        fraction = int((brain & missing).sum()) / max(1, total)
        return (
            rec,
            axis,
            keep_frac,
            side,
            start,
            degraded,
            missing,
            fraction,
            view,
            f"brain_mask/{mode}",
            wb_angles.tolist(),
        )

    def _prepare_training_inpaint(self, initial_idx: int, mode: str) -> dict:
        wanted_bin = random.choices(
            range(len(self.inpaint_size_bin_names)),
            weights=self.inpaint_size_bin_weights,
            k=1,
        )[0]
        placement_attempts = 0
        placement_failures = 0
        candidates_examined = 0
        for case_try in range(self.inpaint_case_max_tries):
            known_pool = self._inpaint_idx_by_bin[wanted_bin]
            known_candidate = None
            for _ in range(min(8, len(known_pool))):
                probed = int(random.choice(known_pool))
                if probed not in self._inpaint_unplaceable_idx:
                    known_candidate = probed
                    break
            if case_try == 0 and initial_idx in self._labeled_idx_set:
                candidate_idx = int(initial_idx)
            elif known_candidate is not None and random.random() < 0.9:
                candidate_idx = known_candidate
            else:
                candidate_idx = int(random.choice(self._labeled_idx))
            if candidate_idx in self._inpaint_unplaceable_idx:
                continue
            rec = self.records[candidate_idx]
            target = rec["images"][0]
            lab = self._read_label_full_raw(rec["label_h5"])
            if int((lab >= 1).sum()) <= 100:
                self._inpaint_too_small += 1
                self._inpaint_unplaceable_idx.add(candidate_idx)
                continue
            try:
                img = self._read_image_full(target["h5"], mode)
                brain = self._read_brain_mask_full(target["h5"], mode)
            except IneligibleSampleError:
                continue
            brain_voxels = int(brain.sum())
            source_hole = tumor_inpaint_mask(lab, self.inpaint_dilate)
            source_hole_voxels = int(source_hole.sum())
            hole_fraction = source_hole_voxels / max(1, brain_voxels)
            size_bin = self._inpaint_bin_for_fraction(hole_fraction)
            candidates_examined += 1
            if candidate_idx not in self._inpaint_record_bin:
                self._inpaint_record_bin[candidate_idx] = size_bin
                self._inpaint_idx_by_bin[size_bin].append(candidate_idx)
            if size_bin != wanted_bin:
                continue
            placement_attempts += 1
            hmask, placement = healthy_inpaint_mask(
                lab,
                brain,
                self.inpaint_dilate,
                margin=self.inpaint_margin,
                min_brain_frac=self.inpaint_min_brain_frac,
                max_tries=self.inpaint_place_max_tries,
                prefer_contralateral=self.inpaint_contralateral_large
                and size_bin == len(self.inpaint_size_bin_names) - 1,
                contralateral_axis=self.inpaint_contralateral_axis,
                return_info=True,
                rng=np.random.default_rng(random.getrandbits(64)),
            )
            if not hmask.any():
                placement_failures += 1
                if not source_hole.any() or brain_voxels <= 0:
                    self._inpaint_unplaceable_idx.add(candidate_idx)
                else:
                    fails = self._inpaint_placement_fail.get(candidate_idx, 0) + 1
                    self._inpaint_placement_fail[candidate_idx] = fails
                    if fails >= self.inpaint_placement_fail_blacklist:
                        self._inpaint_unplaceable_idx.add(candidate_idx)
                continue
            return {
                "rec": rec,
                "lab": lab,
                "img": img,
                "brain_mask": brain,
                "hmask": hmask,
                "hole_voxels": int(hmask.sum()),
                "brain_voxels": brain_voxels,
                "hole_fraction": float(hmask.sum()) / max(1, brain_voxels),
                "size_bin": size_bin,
                "placement_strategy": str(placement["strategy"]),
                "placement_attempts": placement_attempts,
                "placement_failures": placement_failures,
                "candidates_examined": candidates_examined,
            }
        name = self.inpaint_size_bin_names[wanted_bin]
        raise RuntimeError(
            f"failed to construct a non-empty inpaint hole after {self.inpaint_case_max_tries} candidate cases for bin={name!r}; edges={self.inpaint_size_bin_edges}, placement_attempts={placement_attempts}, placement_failures={placement_failures}. Increase inpaint_case_max_tries or revise the bin edges, but do not train on an empty mask."
        )

    def _prepare_training_inpaint_lesion(self, initial_idx: int, mode: str) -> dict:
        for attempt in range(self.inpaint_case_max_tries):
            idx = initial_idx if attempt == 0 else random.choice(self._labeled_idx)
            rec = self.records[idx]
            if rec.get("label_h5") is None:
                continue
            lab = self._read_label_full_raw(rec["label_h5"])
            if int((lab >= 1).sum()) <= 100:
                self._inpaint_too_small += 1
                continue
            hmask = tumor_inpaint_mask(lab, self.inpaint_dilate)
            if hmask.any():
                try:
                    img = self._read_image_full(rec["images"][0]["h5"], mode)
                    brain = self._read_brain_mask_full(rec["images"][0]["h5"], mode)
                except IneligibleSampleError:
                    continue
                return {
                    "rec": rec,
                    "lab": lab,
                    "img": img,
                    "hmask": hmask,
                    "brain_mask": brain,
                    "content": self._lesion_content(lab),
                }
        raise RuntimeError(
            f"no labelled case with a non-empty tumor and valid brain mask found for lesion inpaint after {self.inpaint_case_max_tries} tries"
        )

    def _prepare_training_inpaint_pseudo_healthy(
        self, initial_idx: int, mode: str
    ) -> dict:
        out = self._prepare_training_inpaint_lesion(initial_idx, mode)
        out["content"] = "healthy"
        out["pseudo_healthy"] = True
        return out

    def _mask_latent_group(self, f, path):
        base = resolve_mask_latent_paths(f, self.expected_spatial_size)
        if base is None:
            return None
        g = f[base]
        validate_latent_attrs(
            f,
            path,
            attrs_path=base,
            expected_fingerprint=self.expected_fingerprint,
            expected_spatial_size=self.expected_spatial_size,
            expected_latent_shape=self.expected_latent_shape,
        )
        if int(g.attrs.get("num_classes", -1)) != self.seg_num_classes:
            raise RuntimeError(f"mask-latent num_classes mismatch: {path}@{base}")
        shape = g["mean"].shape
        if (
            len(shape) != 4
            or tuple(shape[1:]) != tuple(self.expected_latent_shape)
            or (self.latent_channels is not None and shape[0] != self.latent_channels)
        ):
            raise RuntimeError(f"mask-latent shape mismatch: {path}@{base}: {shape}")
        return g

    def _read_mask_latent(self, label_h5: str) -> Optional[torch.Tensor]:
        if label_h5 is None:
            return None
        with h5py.File(label_h5, "r") as f:
            g = self._mask_latent_group(f, label_h5)
            return (
                None
                if g is None
                else torch.from_numpy(np.asarray(g["mean"], dtype=np.float32))
            )

    def _roll_task(self) -> str:
        if self._force_task is not None:
            return self._force_task
        return random.choices(self.task_names, weights=self.task_probs, k=1)[0]

    def _rec_is_thin(self, rec) -> bool:
        meta = self.acq_lookup.get(acq_stem(rec["images"][0]["path"]))
        return bool(meta) and str(meta[0]).lower() == "thin"

    def _rec_is_thick(self, rec) -> bool:
        meta = self.acq_lookup.get(acq_stem(rec["images"][0]["path"]))
        return bool(meta) and str(meta[0]).lower() == "thick"

    def _lesion_content(self, lab: np.ndarray) -> str:
        arr = np.asarray(lab)
        t = int(np.isin(arr, self.inpaint_tumor_class_ids).sum())
        c = int(np.isin(arr, self.inpaint_cavity_class_ids).sum())
        denom = t + c
        if denom <= 0:
            return "cavity"
        return "tumor" if t >= self.inpaint_tumor_dominance_frac * denom else "cavity"

    def _sr_native_mm(self, h5_path: str, axis: int):
        if axis is None or int(axis) not in (0, 1, 2):
            raise ValueError(f"_sr_native_mm: axis must be in 0..2, got {axis!r}")
        with h5py.File(h5_path, "r") as f:
            return h5_axis_spacing(f, "native", axis)

    def _snap_sr_mm(self, out_mm, in_mm):
        try:
            o, i = (int(round(float(out_mm))), int(round(float(in_mm))))
        except (TypeError, ValueError):
            return None
        if i > o and (o, i) in SR_MM_GRID:
            return (o, i)
        return None

    def _strict_text_lookup(self, s: str):
        try:
            return self.prompt_store.get(s)
        except KeyError as e:
            raise RuntimeError(
                f"STRICT prompt-cache miss (no fallback allowed): the composed prompt {s[:160]!r} is NOT in prompt_feature_h5. This is a coverage regression — fix the loader/enumeration or (for genuinely new strings) rebuild the cache via scripts/encode_prompts.py, then verify the newly built cache."
            ) from e

    def _task_text(
        self,
        rec: dict,
        task: str,
        target_mod: int,
        quality: str,
        view: str,
        inpaint_content: Optional[str] = None,
        sr_in_mm=None,
        sr_out_mm=None,
        seg_modalities=None,
        sr_method: str = "nearest-neighbor",
        non_mm_variants: bool = False,
        target_mm=None,
        synthetic_input_mm=None,
        synthetic_input_generic: bool = False,
    ):
        th = None
        try:
            _q, _v, th = self._acq_full(rec["images"][0]["path"])
        except Exception:
            th = None
        if task == "inpaint":
            content = str(inpaint_content or "healthy")
            if self.train:
                variant = random.randrange(n_inpaint_variants(content))
                dv = random.randrange(n_desc_variants())
            else:
                variant = 0
                dv = 0
            s = compose_prompt(
                "inpaint",
                target_mod,
                quality,
                view,
                inpaint_content=content,
                variant=variant,
                desc_variant=dv,
                target_mm=target_mm,
                thickness_mm=th,
            )
            return (self._strict_text_lookup(s), True, s)
        if task == "seg":
            variant = random.randrange(6) if self.train else 0
            s = seg_prompt(
                seg_modalities or (),
                variant=variant,
                catalog_path=self.seg_prompt_variants_path,
            )
            return (self._strict_text_lookup(s), True, s)
        ctask = task
        if self.train:
            dv = random.randrange(n_desc_variants())
            if ctask == "modality_only":
                if random.random() < 0.5:
                    s = compose_prompt(
                        ctask,
                        target_mod,
                        quality,
                        view,
                        desc_variant=dv,
                        target_mm=target_mm,
                        thickness_mm=th,
                    )
                else:
                    s = compose_prompt(
                        ctask,
                        target_mod,
                        quality,
                        view,
                        desc_variant=dv,
                        shell=random.randrange(6),
                        target_mm=target_mm,
                        thickness_mm=th,
                    )
            elif ctask in PARAM_SR_TASKS:
                if synthetic_input_mm is not None or synthetic_input_generic:
                    s = compose_prompt(
                        ctask,
                        target_mod,
                        quality,
                        view,
                        desc_variant=dv,
                        variant=random.randrange(n_synthetic_input_variants(ctask)),
                        target_mm=target_mm,
                        synthetic_input_mm=synthetic_input_mm,
                        synthetic_input_generic=synthetic_input_generic,
                        thickness_mm=th,
                    )
                elif non_mm_variants:
                    s = compose_prompt(
                        ctask,
                        target_mod,
                        quality,
                        view,
                        desc_variant=dv,
                        variant=random.randrange(6),
                        target_mm=target_mm,
                        thickness_mm=th,
                    )
                else:
                    s = compose_prompt(
                        ctask,
                        target_mod,
                        quality,
                        view,
                        desc_variant=dv,
                        variant=random.randrange(n_task_variants(ctask))
                        if sr_in_mm is not None and sr_out_mm is not None
                        else 0,
                        sr_in_mm=sr_in_mm,
                        sr_out_mm=sr_out_mm,
                        sr_method=sr_method,
                        target_mm=target_mm,
                        thickness_mm=th,
                    )
            else:
                s = compose_prompt(
                    ctask,
                    target_mod,
                    quality,
                    view,
                    desc_variant=dv,
                    variant=random.randrange(n_task_variants(ctask)),
                    target_mm=target_mm,
                    thickness_mm=th,
                )
            return (self._strict_text_lookup(s), True, s)
        if ctask in PARAM_SR_TASKS and (
            synthetic_input_mm is not None or synthetic_input_generic
        ):
            s = compose_prompt(
                ctask,
                target_mod,
                quality,
                view,
                target_mm=target_mm,
                synthetic_input_mm=synthetic_input_mm,
                synthetic_input_generic=synthetic_input_generic,
                thickness_mm=th,
            )
            return (self._strict_text_lookup(s), True, s)
        if ctask in PARAM_SR_TASKS:
            if (
                sr_in_mm is not None
                and sr_out_mm is not None
                and (int(sr_in_mm) > int(sr_out_mm))
            ):
                s = compose_prompt(
                    ctask,
                    target_mod,
                    quality,
                    view,
                    variant=slotted_variant(ctask),
                    sr_in_mm=int(sr_in_mm),
                    sr_out_mm=int(sr_out_mm),
                    sr_method=sr_method,
                    target_mm=target_mm,
                    thickness_mm=th,
                )
            else:
                s = compose_prompt(
                    ctask,
                    target_mod,
                    quality,
                    view,
                    target_mm=target_mm,
                    thickness_mm=th,
                )
            return (self._strict_text_lookup(s), True, s)
        s = compose_prompt(
            ctask, target_mod, quality, view, target_mm=target_mm, thickness_mm=th
        )
        return (self._strict_text_lookup(s), True, s)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        task = self._roll_task()
        seg_input_mods = None
        inpaint_prepared = None
        inpaint_lesion_prep = None
        if self.train:
            if (
                task == "mask_guide"
                and self._mask_guide_idx
                and (
                    rec.get("label_h5") is None
                    or not self._mask_ok_map.get(rec["label_h5"], False)
                )
            ):
                rec = self.records[random.choice(self._mask_guide_idx)]
            elif (
                task == "inpaint"
                and self._labeled_idx
                and (rec.get("label_h5") is None)
            ):
                rec = self.records[random.choice(self._labeled_idx)]
            elif task in ("sr", "deblur", "dealias", "whole_brain") and (
                not self._rec_is_thin(rec)
            ):
                rec = self.records[random.choice(self._thin_idx)]
            elif task == "missing" and self._multimod_idx:
                acc = os.path.dirname(str(rec["images"][0]["path"])).strip("/")
                if len(self.acc_index.get(acc, {})) < 2:
                    rec = self.records[random.choice(self._multimod_idx)]
        if task == "seg":
            desired = (
                int(self._force_seg_ref_count) if self._force_seg_ref_count else None
            )
            eligible_counts = [
                k
                for k, weight in self.seg_ref_count_weights.items()
                if self._seg_acc_by_count.get(k) and k <= int(self.num_refs)
            ]
            if not eligible_counts:
                raise RuntimeError(
                    "seg has no labelled studies with cached image modalities"
                )
            if desired is not None:
                if desired not in eligible_counts:
                    raise RuntimeError(
                        f"no labelled study can provide {desired} seg refs"
                    )
                count = desired
            else:
                count = random.choices(
                    eligible_counts,
                    weights=[self.seg_ref_count_weights[k] for k in eligible_counts],
                    k=1,
                )[0]
            if self._force_seg_acc is not None and self._force_seg_mods is not None:
                acc = self._force_seg_acc
                avail = self.acc_index[acc]
                seg_input_mods = list(self._force_seg_mods)
                count = len(seg_input_mods)
            else:
                acc_pool = self._seg_acc_by_count[count]
                acc = (
                    random.choice(acc_pool)
                    if self.train
                    else acc_pool[int(idx) % len(acc_pool)]
                )
                avail = self.acc_index[acc]
                mods = sorted(avail)
                if self.train:
                    seg_input_mods = sorted(random.sample(mods, count))
                else:
                    from itertools import combinations

                    combos = list(combinations(mods, count))
                    seg_input_mods = list(combos[int(idx) % len(combos)])
            rec = self.records[self._seg_target_idx[acc]]
        mode = self._select_mode()
        motion_prepared = None
        if task == "motion":
            rec, full_h5, brain_mask = self._prepare_motion_record(rec, mode)
            motion_prepared = {"full_h5": full_h5, "brain_mask": brain_mask}
        inpaint_lesion = False
        inpaint_pseudo_prep = None
        inpaint_pseudo = False
        if self.train and task == "inpaint" and self.load_labels:
            if (
                self.inpaint_pseudo_healthy_prob > 0
                and random.random() < self.inpaint_pseudo_healthy_prob
            ):
                inpaint_pseudo_prep = self._prepare_training_inpaint_pseudo_healthy(
                    idx, mode
                )
                rec = inpaint_pseudo_prep["rec"]
                inpaint_pseudo = True
            else:
                inpaint_lesion = random.random() < self.inpaint_lesion_prob
                if inpaint_lesion:
                    inpaint_lesion_prep = self._prepare_training_inpaint_lesion(
                        idx, mode
                    )
                    rec = inpaint_lesion_prep["rec"]
                else:
                    inpaint_prepared = self._prepare_training_inpaint(idx, mode)
                    rec = inpaint_prepared["rec"]
        fov_prepared = None
        if task == "whole_brain":
            if not self._rec_is_thin(rec):
                if not self._thin_idx:
                    raise IneligibleSampleError(
                        "whole_brain requires a native thin-slice target (no thin records available)"
                    )
                rec = self.records[random.choice(self._thin_idx)]
            fov_prepared = self._prepare_whole_brain(rec, mode)
            rec = fov_prepared[0]
        missing_inputs = []
        if task == "missing":
            acc = os.path.dirname(str(rec["images"][0]["path"])).strip("/")
            avail = self.acc_index.get(acc, {})
            if len(avail) >= 2:
                selected_mod = (
                    int(self._force_missing_target_mod)
                    if self._force_missing_target_mod is not None
                    else random.choice(list(avail))
                )
                if selected_mod not in avail:
                    raise IneligibleSampleError(
                        f"forced missing target modality {selected_mod} unavailable in {acc}"
                    )
                others = [m for m in avail if m != selected_mod]
                missing_inputs = random.sample(
                    others, random.randint(1, min(self.num_refs, len(others)))
                )
                rec = self.records[avail[selected_mod]]
        target = rec["images"][0]
        if task == "dealias":
            full_h5 = self._full_head_path(target["h5"])
            if full_h5 is None or not os.path.isfile(full_h5):
                raise IneligibleSampleError(
                    f"dealias full-head sibling missing for {target['h5']} (DA-01)"
                )
            target = {**target, "h5": full_h5, "path": full_h5}
        target_mod = int(target["modality"])
        tgt_mean, tgt_logvar = self._read_latent(target["h5"], mode)
        C_z = tgt_mean.shape[0]
        ld = tuple(self.expected_latent_shape)
        D = tuple(self.expected_spatial_size)
        R = int(self.num_refs) if self.emit_spatial_conditions else 0
        C_reg = int(self.region_mask_channels) if self.emit_spatial_conditions else 0
        has_label = rec["label_h5"] is not None and self.load_labels
        region_mask = torch.zeros(C_reg, *ld) if C_reg > 0 else torch.zeros(0, *ld)
        ref_latent = torch.zeros(R, C_z, *ld)
        ref_mod = torch.zeros(R, dtype=torch.long)
        ref_dt = torch.zeros(R, dtype=torch.float32)
        ref_valid = torch.zeros(R, dtype=torch.bool)
        ref_type = torch.zeros(R, dtype=torch.long)
        ref_image = (
            torch.zeros(R, *D, dtype=torch.float16)
            if self.emit_spatial_conditions and R > 0
            else torch.zeros(0, dtype=torch.float16)
        )
        ref_needs_encode = torch.zeros(R, dtype=torch.bool)
        ref_paths = [""] * R
        ref_roles = [""] * R
        label_full = None
        text_emb = None
        prompt_str = None
        target_needs_encode = False
        target_encode_slot = -1
        prompt_target_quality = None
        prompt_target_mm = None
        inpaint_content = None
        inpaint_hole_voxels = 0
        inpaint_brain_voxels = 0
        inpaint_hole_fraction = 0.0
        inpaint_size_bin = -1
        inpaint_placement_attempts = 0
        inpaint_placement_failures = 0
        inpaint_candidates_examined = 0
        inpaint_placement_strategy = "none"
        if task in ("inpaint", "mask_guide", "seg") and (not has_label):
            task = "modality_only"
        if (
            task == "missing"
            and len(
                self.acc_index.get(os.path.dirname(str(target["path"])).strip("/"), {})
            )
            < 2
        ):
            task = "modality_only"
        target_ref = target["path"]
        label_eval_full = None
        if has_label:
            label_full = self._read_label_full(rec["label_h5"])
            if not self.train and self.return_label:
                K = self.seg_num_classes
                label_eval_full = np.clip(
                    self._read_label_full_raw(rec["label_h5"]), 0, K - 1
                )
        cond_feed = task == "mask_guide"
        sr_params: dict = {}
        degradation_params: dict = {}
        factor, axis = (0, -1)
        if self.emit_spatial_conditions and has_label and cond_feed and (R > 0):
            mlat = self._read_mask_latent(rec["label_h5"])
            if mlat is None:
                raise RuntimeError(
                    f"mask_guide sample has no valid mask latent: {rec.get('label_h5')} (label exists but latent/mask missing or contract-violated — audit MG-02). Train resamples onto _mask_guide_idx; eval pool must filter too."
                )
            else:
                ref_latent[0] = mlat
                ref_mod[0] = self.num_modalities
                ref_type[0] = 1
                ref_valid[0] = True
                ref_paths[0] = str(rec.get("label_h5") or rec.get("label_path") or "")
                ref_roles[0] = "tumor_mask"
        if task == "inpaint":
            inpaint_content = "healthy"
            if inpaint_prepared is not None:
                lab = inpaint_prepared["lab"]
                img = inpaint_prepared["img"]
                brain = inpaint_prepared["brain_mask"]
                hmask = inpaint_prepared["hmask"]
                inpaint_placement_attempts = int(inpaint_prepared["placement_attempts"])
                inpaint_placement_failures = int(inpaint_prepared["placement_failures"])
                inpaint_candidates_examined = int(
                    inpaint_prepared["candidates_examined"]
                )
                inpaint_placement_strategy = str(inpaint_prepared["placement_strategy"])
            elif inpaint_pseudo_prep is not None:
                lab = inpaint_pseudo_prep["lab"]
                img = inpaint_pseudo_prep["img"]
                brain = inpaint_pseudo_prep["brain_mask"]
                hmask = inpaint_pseudo_prep["hmask"]
                inpaint_content = "healthy"
                inpaint_placement_attempts = 1
                inpaint_placement_strategy = "real_lesion_pseudo_healthy"
            elif inpaint_lesion_prep is not None:
                lab = inpaint_lesion_prep["lab"]
                img = inpaint_lesion_prep["img"]
                brain = inpaint_lesion_prep["brain_mask"]
                hmask = inpaint_lesion_prep["hmask"]
                inpaint_content = inpaint_lesion_prep["content"]
                inpaint_placement_attempts = 1
                inpaint_placement_strategy = "real_lesion"
            else:
                lab = self._read_label_full_raw(rec["label_h5"])
                img = self._read_image_full(target["h5"], mode)
                brain = self._read_brain_mask_full(target["h5"], mode)
            if (
                inpaint_prepared is not None
                or inpaint_pseudo_prep is not None
                or inpaint_lesion_prep is not None
            ):
                pass
            elif self._inpaint_void_tumor:
                hmask = tumor_inpaint_mask(lab, self.inpaint_dilate)
                inpaint_placement_attempts = 1
                inpaint_placement_strategy = "real_lesion"
            else:
                source_hole = tumor_inpaint_mask(lab, self.inpaint_dilate)
                brain_voxels = int(brain.sum())
                source_fraction = int(source_hole.sum()) / max(1, brain_voxels)
                source_bin = self._inpaint_bin_for_fraction(source_fraction)
                hmask, placement = healthy_inpaint_mask(
                    lab,
                    brain,
                    self.inpaint_dilate,
                    margin=self.inpaint_margin,
                    min_brain_frac=self.inpaint_min_brain_frac,
                    max_tries=self.inpaint_place_max_tries,
                    prefer_contralateral=self.inpaint_contralateral_large
                    and source_bin == len(self.inpaint_size_bin_names) - 1,
                    contralateral_axis=self.inpaint_contralateral_axis,
                    return_info=True,
                    rng=np.random.default_rng(random.getrandbits(64)),
                )
                inpaint_placement_attempts = 1
                inpaint_placement_failures = int(not hmask.any())
                inpaint_candidates_examined = 1
                inpaint_placement_strategy = str(placement["strategy"])
            if not hmask.any():
                raise IneligibleSampleError(
                    f"inpaint case {target['path']!r} produced an empty mask; empty/no-op inpaint samples are forbidden"
                )
            inpaint_hole_voxels = int(hmask.sum())
            inpaint_brain_voxels = int(brain.sum())
            inpaint_hole_fraction = inpaint_hole_voxels / max(1, inpaint_brain_voxels)
            inpaint_size_bin = self._inpaint_bin_for_fraction(inpaint_hole_fraction)
            if inpaint_pseudo and label_full is not None:
                hole_lab = torch.from_numpy(
                    downsample_mask_max(hmask, tuple(self.label_full_shape))
                )
                label_full = label_full.clone()
                label_full[hole_lab > 0] = 0
            masked = apply_mask(img, hmask, 0.0)
            ref_image[0] = torch.from_numpy(masked.astype(np.float16))
            ref_needs_encode[0] = True
            ref_mod[0] = target_mod
            ref_valid[0] = True
            ref_paths[0] = str(target["path"])
            ref_roles[0] = "masked_image"
            if R > 1:
                ref_image[1] = torch.from_numpy(hmask.astype(np.float16))
                ref_needs_encode[1] = True
                ref_mod[1] = self.num_modalities
                ref_type[1] = 2
                ref_valid[1] = True
                ref_paths[1] = str(rec.get("label_h5") or rec.get("label_path") or "")
                ref_roles[1] = "hole_mask"
            if C_reg > 0:
                region_mask = torch.from_numpy(downsample_mask_max(hmask, ld))[
                    None
                ].float()
        elif task == "sr":
            img = self._read_image_full(target["h5"], mode)
            sr_params: dict = {}
            degradation_params = {}
            if self._sr_native_input:
                factor = 1
                up_order = 0
                axis = 0
                try:
                    with h5py.File(target["h5"], "r") as f:
                        for grp in ("meta/native/spacing", "meta/iso1mm/spacing"):
                            if grp in f:
                                sp = f[grp][...]
                                if sp is not None and len(sp) == 3:
                                    axis = int(np.argmax([float(x) for x in sp]))
                                    break
                except Exception:
                    pass
                lr = img
                sr_method = None
                try:
                    native_mm = self._sr_native_mm(target["h5"], axis)
                except Exception:
                    native_mm = None
                degradation_params = {
                    "sr_native_input": True,
                    "sr_native_axis": int(axis),
                    "sr_native_axis_mm": native_mm,
                }
            else:
                relative_target = (
                    self.train and random.random() >= self.sr_absolute_thin_prob
                )
                if relative_target:
                    input_mm, output_mm = random.choice(self.sr_relative_pairs)
                else:
                    input_mm = random.choices(
                        self.sr_factors, weights=self.sr_factor_weights, k=1
                    )[0]
                    output_mm = 1
                axis = random.choices((0, 1, 2), weights=self.sr_axis_weights, k=1)[0]
                up_order = random.choices(
                    self.sr_up_orders, weights=self.sr_up_order_weights, k=1
                )[0]
                with h5py.File(target["h5"], "r") as f:
                    working_mm = h5_axis_spacing(f, mode, axis)
                if working_mm is None or not np.isfinite(working_mm) or working_mm <= 0:
                    raise ValueError(
                        f"SR requires working-grid spacing for {target['h5']} mode={mode}"
                    )
                factor_eff = float(input_mm) / float(working_mm)
                output_factor = float(output_mm) / float(working_mm)
                if factor_eff <= 1.0 or (relative_target and output_factor <= 1.0):
                    raise ValueError(
                        f"SR requested physical thickness incompatible with working grid: working={working_mm:g}mm input={input_mm}mm output={output_mm}mm"
                    )
                factor = factor_eff
                native_out_mm = self._sr_native_mm(target["h5"], axis)
                rotation_seed = random.getrandbits(64)
                angles = np.zeros(3)

                def _synthetic_thickness(thickness_factor, *, return_angles=False):
                    if self.sr_affine_enabled:
                        return random_affine_thick_lowres(
                            img,
                            thickness_factor,
                            axis,
                            max_angle_deg=self.sr_affine_max_angle_deg,
                            rng=np.random.default_rng(rotation_seed),
                            up_order=up_order,
                            return_params=return_angles,
                        )
                    result = thick_slice_lowres(
                        img, thickness_factor, axis, up_order=up_order
                    )
                    return (result, np.zeros(3)) if return_angles else result

                if self.sr_affine_enabled:
                    lr, angles = _synthetic_thickness(factor_eff, return_angles=True)
                    sr_method = "random-affine"
                else:
                    lr = _synthetic_thickness(factor_eff)
                    sr_method = {
                        0: "nearest-neighbor",
                        1: "bilinear-interpolated",
                        3: "bicubic-interpolated",
                    }[up_order]
                geometry = sr_sampling_geometry(img.shape, factor_eff, axis, working_mm)
                exact_input = (
                    relative_target or random.random() < self.prompt_exact_input_mm_prob
                )
                sr_params = dict(
                    sr_method=sr_method,
                    synthetic_input_mm=int(input_mm) if exact_input else None,
                    synthetic_input_generic=not exact_input,
                )
                if relative_target:
                    if R < 2:
                        raise RuntimeError(
                            "relative SR requires mmdit.max_refs>=2 for online target encoding"
                        )
                    synthetic_target = _synthetic_thickness(output_factor)
                    ref_image[1] = torch.from_numpy(synthetic_target.astype(np.float16))
                    target_needs_encode = True
                    target_encode_slot = 1
                    tgt_logvar = None
                    prompt_target_quality = (
                        "thick"
                        if int(output_mm) >= 4
                        else "medium"
                        if int(output_mm) >= 2
                        else "thin"
                    )
                    prompt_target_mm = None
                degradation_params = {
                    "sr_branch": "relative_thick"
                    if relative_target
                    else "absolute_thin",
                    "sr_requested_input_mm": int(input_mm),
                    "sr_requested_output_mm": int(output_mm),
                    "sr_factor": factor_eff,
                    "sr_axis": int(axis),
                    "sr_up_order": int(up_order),
                    "sr_method": sr_method,
                    **geometry,
                    "sr_native_axis_mm": native_out_mm,
                    "sr_in_mm": int(input_mm) if exact_input else None,
                    "sr_out_mm": int(output_mm),
                    "prompt_exact_input_mm": bool(exact_input),
                    "rotation_seed": rotation_seed,
                    "rotation_angles_deg": angles.tolist(),
                }
            ref_image[0] = torch.from_numpy(lr.astype(np.float16))
            ref_needs_encode[0] = True
            ref_mod[0] = target_mod
            ref_valid[0] = True
            ref_paths[0] = str(target["path"])
            ref_roles[0] = (
                "native_thick_input" if self._sr_native_input else "synthetic_lr"
            )
        elif task in ("deblur", "dealias", "motion"):
            img = self._read_image_full(target["h5"], mode)
            rng = np.random.default_rng(random.getrandbits(64))
            if task in ("deblur", "dealias"):
                if self._sr_native_input:
                    s_axis = 0
                    with h5py.File(target["h5"], "r") as f:
                        for grp in ("meta/native/spacing", "meta/iso1mm/spacing"):
                            if grp in f:
                                sp = f[grp][...]
                                if sp is not None and len(sp) == 3:
                                    s_axis = int(np.argmax([float(x) for x in sp]))
                                    break
                    try:
                        native_mm = self._sr_native_mm(target["h5"], s_axis)
                    except Exception:
                        native_mm = None
                    if task == "deblur":
                        sigma = float(rng.uniform(*self.deblur_sigma_range))
                        degraded = gaussian_blur(img, sigma=sigma)
                        degradation_params = {
                            "native_thick_input": True,
                            "sr_native_axis": int(s_axis),
                            "sr_native_axis_mm": native_mm,
                            "sigma": sigma,
                        }
                    else:
                        a_factor = random.choices(
                            self.dealias_factors,
                            weights=self.dealias_factor_weights,
                            k=1,
                        )[0]
                        a_axis = random.choices(
                            (0, 1, 2), weights=self.dealias_axis_weights, k=1
                        )[0]
                        degraded = uniform_undersample(
                            img, factor=a_factor, axis=a_axis
                        )
                        degradation_params = {
                            "native_thick_input": True,
                            "sr_native_axis": int(s_axis),
                            "sr_native_axis_mm": native_mm,
                            "alias_factor": int(a_factor),
                            "alias_axis": int(a_axis),
                        }
                    native_degraded = True
                else:
                    composite = (
                        self.train and random.random() < self.restore_composite_prob
                    )
                    if composite:
                        input_mm = random.choices(
                            self.sr_factors, weights=self.sr_factor_weights, k=1
                        )[0]
                        s_axis = random.choices(
                            (0, 1, 2), weights=self.sr_axis_weights, k=1
                        )[0]
                        s_up = random.choices(
                            self.sr_up_orders, weights=self.sr_up_order_weights, k=1
                        )[0]
                        with h5py.File(target["h5"], "r") as f:
                            working_mm = h5_axis_spacing(f, mode, s_axis)
                        if (
                            working_mm is None
                            or not np.isfinite(working_mm)
                            or working_mm <= 0
                        ):
                            raise ValueError(
                                f"{task} composite requires working-grid spacing for {target['h5']}"
                            )
                        s_factor = float(input_mm) / float(working_mm)
                        if s_factor <= 1.0:
                            raise ValueError(
                                f"{task} input {input_mm}mm is not thicker than {working_mm:g}mm grid"
                            )
                        exact_input = random.random() < self.prompt_exact_input_mm_prob
                        if task == "deblur":
                            rot_seed = random.getrandbits(64)
                            rot_rng = np.random.default_rng(rot_seed)
                            affine_max = (
                                self.sr_affine_max_angle_deg
                                if self.sr_affine_enabled
                                else 0.0
                            )
                            sigma_mm = float(rng.uniform(*self.deblur_sigma_range))
                            sigma_low_vox = sigma_mm / max(1e-09, float(working_mm))
                            _on_low = lambda lo: inplane_gaussian_blur_low(
                                lo, sigma_low_vox, s_axis
                            )
                            lr, angles = random_affine_thick_lowres(
                                img,
                                s_factor,
                                s_axis,
                                up_order=s_up,
                                max_angle_deg=affine_max,
                                rng=rot_rng,
                                return_params=True,
                                on_low_grid=_on_low,
                            )
                            degradation_params = {
                                "composite": "sr+deblur",
                                "sr_factor": float(s_factor),
                                "sr_axis": int(s_axis),
                                "sr_up_order": int(s_up),
                                "blur_axis_inplane": int(s_axis),
                                "sigma_mm": round(sigma_mm, 4),
                                "rotation_seed": rot_seed,
                                "rotation_angles_deg": angles.tolist(),
                                "sr_in_mm": int(input_mm) if exact_input else None,
                                "sr_out_mm": 1,
                                "prompt_exact_input_mm": bool(exact_input),
                            }
                        else:
                            a_factor = random.choices(
                                self.dealias_factors,
                                weights=self.dealias_factor_weights,
                                k=1,
                            )[0]
                            a_axis = random.choices(
                                (0, 1, 2), weights=self.dealias_axis_weights, k=1
                            )[0]
                            _on_low = lambda lo: uniform_undersample(
                                lo, a_factor, a_axis
                            )
                            lr = thick_slice_lowres(
                                img,
                                s_factor,
                                s_axis,
                                up_order=s_up,
                                on_low_grid=_on_low,
                            )
                            degradation_params = {
                                "composite": "sr+dealias",
                                "sr_factor": float(s_factor),
                                "sr_axis": int(s_axis),
                                "sr_up_order": int(s_up),
                                "alias_factor": int(a_factor),
                                "alias_axis": int(a_axis),
                                "affine_enabled": False,
                                "sr_in_mm": int(input_mm) if exact_input else None,
                                "sr_out_mm": 1,
                                "prompt_exact_input_mm": bool(exact_input),
                            }
                        sr_params = dict(
                            sr_method="random-affine"
                            if task == "deblur" and self.sr_affine_enabled
                            else {
                                0: "nearest-neighbor",
                                1: "bilinear-interpolated",
                                3: "bicubic-interpolated",
                            }[s_up],
                            synthetic_input_mm=int(input_mm) if exact_input else None,
                            synthetic_input_generic=not exact_input,
                        )
                        degraded = lr
                        native_degraded = False
                    else:
                        sr_params = dict(non_mm_variants=True)
                        if task == "deblur":
                            b_axis = random.choice((0, 1, 2))
                            s_lo, s_hi = self.deblur_sigma_range
                            sigma = float(
                                np.exp(rng.uniform(np.log(s_lo), np.log(s_hi)))
                            )
                            degraded = gaussian_blur(img, sigma=sigma, axis=b_axis)
                            degradation_params = {
                                "composite": "blur",
                                "sigma": round(sigma, 4),
                                "blur_axis": int(b_axis),
                                "sr_in_mm": None,
                                "sr_out_mm": None,
                            }
                        else:
                            a_factor = random.choices(
                                self.dealias_factors,
                                weights=self.dealias_factor_weights,
                                k=1,
                            )[0]
                            a_axis = random.choices(
                                (0, 1, 2), weights=self.dealias_axis_weights, k=1
                            )[0]
                            degraded = uniform_undersample(
                                img, factor=a_factor, axis=a_axis
                            )
                            degradation_params = {
                                "composite": "alias",
                                "alias_factor": int(a_factor),
                                "alias_axis": int(a_axis),
                                "sr_in_mm": None,
                                "sr_out_mm": None,
                            }
                        native_degraded = False
            else:
                if motion_prepared is None:
                    raise RuntimeError(
                        "motion record was not prepared before target loading"
                    )
                full_h5 = motion_prepared["full_h5"]
                full = self._read_image_full(full_h5, mode)
                corrupt_fraction = float(
                    rng.uniform(*self.motion_corrupt_fraction_range)
                )
                degraded, _mparams = periodic_nod_motion(
                    full,
                    rng=rng,
                    n_poses=self.motion_n_poses,
                    planes_per_pose=self.motion_planes_per_pose,
                    max_angle_deg=self.motion_max_angle_deg,
                    trans_ap_mm=self.motion_trans_ap_mm,
                    trans_si_mm=self.motion_trans_si_mm,
                    dc_anchor=self.motion_dc_anchor,
                    corrupt_fraction=corrupt_fraction,
                    active_poses=self.motion_active_poses,
                    return_params=True,
                )
                brain_mask = motion_prepared["brain_mask"]
                degraded = degraded * brain_mask
                degradation_params = {
                    "motion": "periodic_nod_fullhead",
                    "corrupt_fraction": round(corrupt_fraction, 4),
                    "brain_mask_key": f"brain_mask/{mode}",
                    "brain_voxels": int(brain_mask.sum()),
                    **{
                        k: v
                        for k, v in _mparams.items()
                        if k
                        in (
                            "n_poses",
                            "planes_per_pose",
                            "max_angle_deg",
                            "trans_ap_mm",
                            "trans_si_mm",
                            "dc_anchor",
                            "corrupt_fraction",
                            "phase0_rad",
                        )
                    },
                }
                native_degraded = False
            ref_image[0] = torch.from_numpy(degraded.astype(np.float16))
            ref_needs_encode[0] = True
            ref_mod[0] = target_mod
            ref_valid[0] = True
            ref_paths[0] = str(target["path"])
            ref_roles[0] = f"{task}_degraded_input"
            if native_degraded:
                ref_roles[0] += "_native_thick"
        elif task == "whole_brain":
            (
                _,
                axis,
                keep_frac,
                side,
                crop_start,
                degraded,
                missing_region,
                removed_frac,
                crop_view,
                brain_mask_key,
                wb_angles,
            ) = fov_prepared
            ref_image[0] = torch.from_numpy(degraded.astype(np.float16))
            ref_needs_encode[0] = True
            ref_mod[0] = target_mod
            ref_valid[0] = True
            ref_paths[0] = str(target["path"])
            ref_roles[0] = "partial_fov_image"
            if R > 1:
                ref_image[1] = torch.from_numpy(missing_region.astype(np.float16))
                ref_needs_encode[1] = True
                ref_mod[1] = self.num_modalities
                ref_type[1] = 2
                ref_valid[1] = True
                ref_roles[1] = "missing_fov_mask"
            if C_reg > 0:
                region_mask = torch.from_numpy(downsample_mask_max(missing_region, ld))[
                    None
                ].float()
            degradation_params = {
                "axis": int(axis),
                "keep_frac": float(keep_frac),
                "side": str(side),
                "view": str(crop_view),
                "crop_start": int(crop_start) if crop_start is not None else None,
                "brain_mask_key": str(brain_mask_key),
                "removed_brain_frac": round(float(removed_frac), 6),
                "only_background_crop": False,
                "rotation_angles_deg": [round(a, 3) for a in wb_angles],
            }
        elif task == "seg":
            if int(self.num_modalities) <= SEG_TARGET_MODALITY_ID:
                raise ValueError(
                    "seg requires num_modalities=5 (image ids 0..3 plus seg id 4)"
                )
            mask_latent = self._read_mask_latent(rec["label_h5"])
            if mask_latent is None:
                raise RuntimeError(f"seg target mask latent missing: {rec['label_h5']}")
            target_mod = SEG_TARGET_MODALITY_ID
            target_ref = rec["label_h5"]
            tgt_mean, tgt_logvar = (mask_latent, None)
            acc = os.path.dirname(str(rec["images"][0]["path"])).strip("/")
            avail = self.acc_index[acc]
            for slot, mod in enumerate(seg_input_mods or []):
                src = self.records[avail[int(mod)]]["images"][0]
                src_mean, _ = self._read_latent(src["h5"], mode)
                ref_latent[slot] = src_mean
                ref_mod[slot] = int(mod)
                ref_valid[slot] = True
                ref_paths[slot] = str(src["path"])
                ref_roles[slot] = "seg_modality_input"
            if label_full is None:
                label_full = self._read_label_full(rec["label_h5"])
            degradation_params = {"seg_modalities": list(seg_input_mods or [])}
        elif task == "missing":
            acc = os.path.dirname(str(target["path"])).strip("/")
            avail = dict(self.acc_index.get(acc, {}))
            inputs = missing_inputs
            for slot, m in enumerate(inputs):
                s_mean, _ = self._read_latent(
                    self.records[avail[m]]["images"][0]["h5"], mode
                )
                ref_latent[slot] = s_mean
                ref_mod[slot] = int(m)
                ref_valid[slot] = True
                ref_paths[slot] = str(self.records[avail[m]]["images"][0]["path"])
                ref_roles[slot] = "modality_input"
        tq, tv = self._acq(target_ref)
        t_th = None
        try:
            _q, _v, t_th = self._acq_full(target_ref)
        except Exception:
            t_th = None
        if self.train and random.random() < self.acq_thickness_drop_prob:
            tq = "unknown"
        if prompt_target_quality is not None:
            tq = prompt_target_quality
        if (
            prompt_target_mm is None
            and tq == "thin"
            and self.train
            and (random.random() < self.thin_target_one_mm_prob)
        ):
            prompt_target_mm = 1
        text_emb, has_text, prompt_str = self._task_text(
            rec,
            task,
            target_mod,
            tq,
            tv,
            inpaint_content=inpaint_content,
            target_mm=prompt_target_mm,
            seg_modalities=seg_input_mods,
            **sr_params,
        )
        if label_full is None:
            label_full = (
                torch.zeros(self.label_full_shape, dtype=torch.int16)
                if self.return_label
                else torch.zeros(0, dtype=torch.int16)
            )
        if text_emb is None:
            raise RuntimeError(
                f"no text embedding produced for task={task} prompt={prompt_str!r} — cache coverage regression (see _task_text STRICT lookup)."
            )
        out = {
            "latent_mean": tgt_mean,
            "modality_id": target_mod,
            "text_emb": text_emb,
            "has_text": bool(has_text),
            "has_label": bool(has_label),
            "path": target_ref,
            "task": task,
            "pseudo_healthy": bool(inpaint_pseudo),
            "prompt": prompt_str if prompt_str is not None else "",
            "target_quality": tq,
            "target_view": tv,
            "target_mm": int(prompt_target_mm)
            if prompt_target_mm is not None
            else None,
            "target_needs_encode": bool(target_needs_encode),
            "target_encode_slot": int(target_encode_slot),
            "sr_factor": int(factor) if task == "sr" else 0,
            "sr_axis": int(axis) if task == "sr" else -1,
            "sr_synthetic": bool(task == "sr" and (not self._sr_native_input)),
            "inpaint_hole_voxels": inpaint_hole_voxels,
            "inpaint_brain_voxels": inpaint_brain_voxels,
            "inpaint_hole_fraction": inpaint_hole_fraction,
            "inpaint_size_bin": inpaint_size_bin,
            "inpaint_size_bin_name": self.inpaint_size_bin_names[inpaint_size_bin]
            if inpaint_size_bin >= 0
            else "none",
            "inpaint_placement_attempts": inpaint_placement_attempts,
            "inpaint_placement_failures": inpaint_placement_failures,
            "inpaint_candidates_examined": inpaint_candidates_examined,
            "inpaint_placement_strategy": inpaint_placement_strategy,
            "inpaint_content": inpaint_content or "none",
            "seg_ref_count": len(seg_input_mods or ()),
            "degradation_params": json.dumps(degradation_params, sort_keys=True),
        }
        if self.emit_spatial_conditions:
            out.update(
                {
                    "region_mask": region_mask,
                    "ref_latent": ref_latent,
                    "ref_modality_id": ref_mod,
                    "ref_dt": ref_dt,
                    "ref_valid": ref_valid,
                    "ref_type_id": ref_type,
                    "ref_image": ref_image,
                    "ref_needs_encode": ref_needs_encode,
                    "ref_paths": ref_paths,
                    "ref_roles": ref_roles,
                    "cond_feed": bool(cond_feed),
                    "label_full": label_full,
                }
            )
        if label_eval_full is not None:
            out["label_eval_full"] = torch.from_numpy(label_eval_full)
        if tgt_logvar is not None:
            out["latent_logvar"] = tgt_logvar
        return out


def multitask_collate(
    batch: list[dict], *, fixed_text_len: Optional[int] = None
) -> dict:
    embs = [b["text_emb"] for b in batch]
    if any((e is None for e in embs)):
        non_null = [e for e in embs if e is not None]
        H = non_null[0].shape[1] if non_null else 0
        if H == 0:
            raise RuntimeError(
                "multitask_collate: batch has NO text embeddings at all (all None) — this is a cache-coverage regression; __getitem__ should have raised already (STRICT)."
            )
        embs = [
            e
            if e is not None
            else torch.zeros(0, H, dtype=non_null[0].dtype, device=non_null[0].device)
            for e in embs
        ]
    B = len(embs)
    H = embs[0].shape[1]
    L = (
        int(fixed_text_len)
        if fixed_text_len and fixed_text_len > 0
        else max((e.shape[0] for e in embs))
    )
    text_emb = embs[0].new_zeros(B, L, H)
    text_mask = torch.zeros(B, L, dtype=torch.long)
    for i, e in enumerate(embs):
        n = min(e.shape[0], L)
        text_emb[i, :n] = e[:n]
        text_mask[i, :n] = 1
        if e.shape[0] == 0:
            batch[i]["has_text"] = False
    out = {
        "latent_mean": torch.stack([b["latent_mean"] for b in batch]),
        "modality_id": torch.tensor(
            [b["modality_id"] for b in batch], dtype=torch.long
        ),
        "text_emb": text_emb,
        "text_mask": text_mask,
        "has_text": torch.tensor([b["has_text"] for b in batch], dtype=torch.bool),
        "has_label": torch.tensor([b["has_label"] for b in batch], dtype=torch.bool),
        "path": [b["path"] for b in batch],
        "task": [b["task"] for b in batch],
        "prompt": [b.get("prompt", "") for b in batch],
        "target_quality": [b.get("target_quality", "") for b in batch],
        "target_view": [b.get("target_view", "") for b in batch],
        "target_mm": [b.get("target_mm") for b in batch],
        "target_needs_encode": torch.tensor(
            [bool(b.get("target_needs_encode", False)) for b in batch], dtype=torch.bool
        ),
        "target_encode_slot": torch.tensor(
            [int(b.get("target_encode_slot", -1)) for b in batch], dtype=torch.long
        ),
        "sr_factor": torch.tensor(
            [int(b.get("sr_factor", 0)) for b in batch], dtype=torch.long
        ),
        "sr_axis": torch.tensor(
            [int(b.get("sr_axis", -1)) for b in batch], dtype=torch.long
        ),
        "sr_synthetic": torch.tensor(
            [bool(b.get("sr_synthetic", False)) for b in batch], dtype=torch.bool
        ),
        "inpaint_hole_voxels": torch.tensor(
            [int(b.get("inpaint_hole_voxels", 0)) for b in batch], dtype=torch.long
        ),
        "inpaint_brain_voxels": torch.tensor(
            [int(b.get("inpaint_brain_voxels", 0)) for b in batch], dtype=torch.long
        ),
        "inpaint_hole_fraction": torch.tensor(
            [float(b.get("inpaint_hole_fraction", 0.0)) for b in batch],
            dtype=torch.float32,
        ),
        "inpaint_size_bin": torch.tensor(
            [int(b.get("inpaint_size_bin", -1)) for b in batch], dtype=torch.long
        ),
        "inpaint_size_bin_name": [
            b.get("inpaint_size_bin_name", "none") for b in batch
        ],
        "inpaint_placement_attempts": torch.tensor(
            [int(b.get("inpaint_placement_attempts", 0)) for b in batch],
            dtype=torch.long,
        ),
        "inpaint_placement_failures": torch.tensor(
            [int(b.get("inpaint_placement_failures", 0)) for b in batch],
            dtype=torch.long,
        ),
        "inpaint_candidates_examined": torch.tensor(
            [int(b.get("inpaint_candidates_examined", 0)) for b in batch],
            dtype=torch.long,
        ),
        "inpaint_content": [b.get("inpaint_content", "none") for b in batch],
        "pseudo_healthy": torch.tensor(
            [bool(b.get("pseudo_healthy", False)) for b in batch], dtype=torch.bool
        ),
        "inpaint_placement_strategy": [
            b.get("inpaint_placement_strategy", "none") for b in batch
        ],
        "seg_ref_count": torch.tensor(
            [int(b.get("seg_ref_count", 0)) for b in batch], dtype=torch.long
        ),
        "degradation_params": [b.get("degradation_params", "{}") for b in batch],
    }
    if "region_mask" in batch[0]:
        out.update(
            {
                "region_mask": torch.stack([b["region_mask"] for b in batch]),
                "ref_latent": torch.stack([b["ref_latent"] for b in batch]),
                "ref_modality_id": torch.stack([b["ref_modality_id"] for b in batch]),
                "ref_dt": torch.stack([b["ref_dt"] for b in batch]),
                "ref_valid": torch.stack([b["ref_valid"] for b in batch]),
                "ref_type_id": torch.stack([b["ref_type_id"] for b in batch]),
                "ref_image": torch.stack([b["ref_image"] for b in batch]),
                "ref_needs_encode": torch.stack([b["ref_needs_encode"] for b in batch]),
                "ref_paths": [b.get("ref_paths", []) for b in batch],
                "ref_roles": [b.get("ref_roles", []) for b in batch],
                "cond_feed": torch.tensor(
                    [b["cond_feed"] for b in batch], dtype=torch.bool
                ),
                "label_full": torch.stack([b["label_full"] for b in batch]),
            }
        )
    if any(("label_eval_full" in b for b in batch)):
        template = next((b["label_eval_full"] for b in batch if "label_eval_full" in b))
        out["label_eval_full"] = torch.stack(
            [b.get("label_eval_full", torch.zeros_like(template)) for b in batch]
        )
    if any(("latent_logvar" in b for b in batch)):
        out["latent_logvar"] = torch.stack(
            [b.get("latent_logvar", torch.zeros_like(b["latent_mean"])) for b in batch]
        )
        out["latent_sample_mask"] = torch.tensor(
            ["latent_logvar" in b and b["task"] != "seg" for b in batch],
            dtype=torch.bool,
        )
    return out
