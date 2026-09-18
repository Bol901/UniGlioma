"""UniGlioma DiT training with cached targets and online reference encoding."""

from __future__ import annotations
import argparse
import datetime
import json
import logging
import math
import os
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional
from utils.h5_io import DHW_RAS_AFFINE as _DHW_RAS_AFFINE
import numpy as np
import nibabel as nib
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    get_model_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

try:
    import wandb
except ImportError:
    wandb = None
DEBUG_LOGGING = False
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("train")


class _DropCompileNoise(logging.Filter):
    _NOISY = (
        "torch._inductor",
        "torch._dynamo",
        "torch._functorch",
        "torch.fx",
        "torch._subclasses",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        return not str(record.name).startswith(self._NOISY)


def _silence_compile_logs():
    _noisy = (
        "torch._inductor",
        "torch._dynamo",
        "torch._functorch",
        "torch.fx",
        "torch._inductor.select_algorithm",
    )
    try:
        import torch._logging

        torch._logging.set_logs(
            inductor=logging.ERROR, dynamo=logging.ERROR, aot=logging.ERROR
        )
    except Exception:
        pass
    _f = _DropCompileNoise()
    seen = set()
    for _lname in ("", "torch", "torch._inductor", "torch._dynamo"):
        for _h in logging.getLogger(_lname).handlers:
            if id(_h) not in seen:
                _h.addFilter(_f)
                seen.add(id(_h))
    for _n in _noisy:
        logging.getLogger(_n).setLevel(logging.ERROR)


_silence_compile_logs()
from data import build_train_pair_loader as build_train_loader
from models import MMDiTCfg, MMDiT3D, VAE3DConfig, ViTVAE3D
from models import fsdp_config_from_dict, wrap_fsdp_mmdit
from utils import (
    EMA,
    barrier,
    grid_to_tokens,
    tokens_to_grid,
    init_distributed,
    is_main_process,
    load_train_config,
    make_ts,
    set_seed,
)
from utils.tasks import (
    LABEL_REQUIRED_TASKS,
    ONLINE_VAE_TASKS,
    SEG_TARGET_MODALITY_ID,
    apply_curriculum_preset,
)
from utils.seg_eligibility import seg_head_use, seg_head_readout_allowed
from utils.eval_outputs import (
    build_eval_case_ids,
    study_key,
    eval_file_base,
    eval_ref_suffix,
)
from utils.checkpoint import (adapt_unified_modality_embedding, dit_milestone_paths,
                              dit_resume_candidates, has_dit_checkpoint)


def log_rank(
    message: str, rank: int | None = None, device: torch.device | None = None
) -> None:
    if not DEBUG_LOGGING:
        return
    if rank is None:
        rank = int(os.environ.get("RANK", -1))
    prefix = f"[rank {rank}]"
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        prefix += f"[mem {alloc:.2f}G/{reserved:.2f}G]"
    log.info(f"{prefix} {message}")


def build_vae(cfg: dict) -> ViTVAE3D:
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
    vae = ViTVAE3D(vae_cfg, use_compile=bool(cfg.get("compile_vae", True)))
    ckpt = cfg["vae_ckpt"]
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"VAE checkpoint not found: {ckpt}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    vae.load_state_dict(payload["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def build_mmdit_cfg(cfg: dict) -> MMDiTCfg:
    m = cfg["mmdit"]
    return MMDiTCfg(
        n_ref_types=int(m.get("n_ref_types", 3)),
        in_channels=m["in_channels"],
        latent_shape=tuple(m["latent_shape"]),
        patch_size=tuple(m["patch_size"]),
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
        dropout=float(m.get("dropout", 0.0)),
        use_qk_norm=bool(m.get("use_qk_norm", False)),
        mlp_variant=str(m.get("mlp_variant", "gelu")).lower(),
        attn_impl=str(m.get("attn_impl", "sdpa")).lower(),
        grad_checkpoint=bool(m.get("grad_checkpoint", False)),
        arch_version=int(m.get("arch_version", 1)),
        **_seg_head_cfg(m),
    )


EMBEDDER_PREFIXES = ("ref_type_emb",)
GATE_NAMES = ()


def _clean_name(n: str) -> str:
    return (
        n.replace("module.", "")
        .replace("_fsdp_wrapped_module.", "")
        .replace("_checkpoint_wrapped_module.", "")
    )


def apply_freeze(raw_model, *, freeze_backbone: bool, extra_trainable=()):
    extra = tuple(extra_trainable or ())
    n_train = 0
    for n, p in raw_model.named_parameters():
        if not freeze_backbone:
            p.requires_grad_(True)
        else:
            cn = _clean_name(n)
            trainable = (
                cn.startswith(EMBEDDER_PREFIXES)
                or cn in GATE_NAMES
                or (extra and cn.startswith(extra))
            )
            p.requires_grad_(bool(trainable))
        n_train += int(p.requires_grad)
    return n_train


def build_param_groups(
    model, *, base_wd: float, embedder_lr_scale: float, backbone_lr_scale: float
):
    embed, backbone, nd_embed, nd_back = ([], [], [], [])
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        cn = _clean_name(n)
        is_embed = cn.startswith(EMBEDDER_PREFIXES) or cn in GATE_NAMES
        if cn in GATE_NAMES or p.ndim <= 1:
            (nd_embed if is_embed else nd_back).append(p)
        elif is_embed:
            embed.append(p)
        else:
            backbone.append(p)
    groups = []
    if embed:
        groups.append(
            {"params": embed, "weight_decay": base_wd, "lr_scale": embedder_lr_scale}
        )
    if backbone:
        groups.append(
            {"params": backbone, "weight_decay": base_wd, "lr_scale": backbone_lr_scale}
        )
    if nd_embed:
        groups.append(
            {"params": nd_embed, "weight_decay": 0.0, "lr_scale": embedder_lr_scale}
        )
    if nd_back:
        groups.append(
            {"params": nd_back, "weight_decay": 0.0, "lr_scale": backbone_lr_scale}
        )
    return groups


def _seg_head_cfg(m: dict) -> dict:
    s = m.get("seg_head", {}) or {}
    if not bool(s.get("enabled", False)):
        return {"seg_head_enabled": False}
    out = {"seg_head_enabled": True, "seg_num_classes": int(s.get("num_classes", 4))}
    if s.get("tap_layer") is not None:
        out["seg_tap_layer"] = int(s["tap_layer"])
    ds_factor = int(s.get("downsample", 1))
    if ds_factor < 1 or ds_factor & ds_factor - 1 != 0:
        raise ValueError(
            f"seg_head.downsample must be a power of 2 ≥1; got {ds_factor}"
        )
    n_drop = ds_factor.bit_length() - 1
    if s.get("channels") is not None:
        channels = tuple((int(c) for c in s["channels"]))
        if n_drop >= len(channels):
            raise ValueError(
                f"seg_head.downsample={ds_factor} would drop {n_drop} deconv stage(s) but only {len(channels) - 1} exist (channels={channels})"
            )
        out["seg_channels"] = channels[: len(channels) - n_drop] if n_drop else channels
    if s.get("full_shape") is not None:
        full = tuple((int(x) for x in s["full_shape"]))
        out["seg_full_shape"] = tuple((x // ds_factor for x in full))
    return out


def _broadcast_str(s: str) -> str:
    obj = [s]
    dist.broadcast_object_list(obj, src=0)
    return obj[0]


def resolve_exp_dir(cfg: dict, args_config: str) -> Path:
    exp_root = Path(cfg["ckpt_dir"])
    mode = str(cfg.get("resume", "")).strip()
    resuming = mode in ("latest", "auto")
    exp_dir_str = ""
    if is_main_process():
        if resuming:
            resume_exp = str(cfg.get("resume_exp", "")).strip()
            if not resume_exp:
                raise ValueError(
                    f"resume: {mode!r} requires `resume_exp` (experiment dir name under ckpt_dir, or an absolute path)."
                )
            raw = Path(resume_exp)
            candidates = [raw] if raw.is_absolute() else [raw, exp_root / resume_exp]
            exp_dir = next((c for c in candidates if c.is_dir()), None)
            if exp_dir is None:
                if mode == "auto":
                    exp_dir = raw if raw.is_absolute() else exp_root / resume_exp
                else:
                    tried = ", ".join((str(c) for c in candidates))
                    raise FileNotFoundError(f"resume_exp not found (tried: {tried})")
        else:
            run_name = str(cfg.get("run_name", "run"))
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            sid = uuid.uuid4().hex[:6]
            exp_dir = exp_root / f"{run_name}_{ts}_{sid}"
        (exp_dir / "ckpts").mkdir(parents=True, exist_ok=True)
        (exp_dir / "infer").mkdir(parents=True, exist_ok=True)
        try:
            snap = "config_resume.yaml" if resuming else "config.yaml"
            shutil.copy(args_config, exp_dir / snap)
        except OSError:
            pass
        exp_dir_str = str(exp_dir)
        log.info(f"[exp] experiment dir: {exp_dir}")
    return Path(_broadcast_str(exp_dir_str))


def dump_resolved_config(exp_dir: Path, cfg: dict, mmdit_cfg: MMDiTCfg) -> None:
    if not is_main_process():
        return
    import dataclasses
    import subprocess

    try:
        import yaml as _yaml
    except Exception:
        return
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        sha = ""
    clean = {k: v for k, v in cfg.items() if not str(k).startswith("__")}
    payload = {
        "git_commit": sha,
        "config_path": cfg.get("__config_path__", ""),
        "cfg": clean,
        "mmdit_cfg_resolved": dataclasses.asdict(mmdit_cfg),
    }
    try:
        with open(exp_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
            _yaml.safe_dump(payload, f, sort_keys=True, default_flow_style=False)
        log.info(f"[exp] wrote {exp_dir / 'config_resolved.yaml'} (git {sha[:8]})")
    except OSError:
        pass


def _relink(link_path: Path, target_name: str) -> None:
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    try:
        os.symlink(target_name, link_path)
    except OSError:
        target_path = link_path.with_name(target_name)
        try:
            os.link(target_path, link_path)
        except OSError:
            shutil.copy2(target_path, link_path)


def _prune_old_ckpts(ck: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    milestones = dit_milestone_paths(ck)
    retained_steps = set(sorted({int(p.stem.rsplit("_", 1)[-1]) for p in milestones},
                                reverse=True)[:keep_last])
    for path in milestones:
        if int(path.stem.rsplit("_", 1)[-1]) in retained_steps:
            continue
        for old in (path, path.with_name(path.stem + "_ema.pt")):
            try:
                old.unlink()
            except FileNotFoundError:
                pass


def _full_fsdp_state_dict(train_model: nn.Module, optim=None):
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    if optim is not None:
        model_sd, optim_sd = get_state_dict(train_model, optim, options=opts)
    else:
        model_sd = get_model_state_dict(train_model, options=opts)
        optim_sd = None
    return (model_sd, optim_sd)


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_ckpt(
    raw_model: nn.Module,
    optim,
    step: int,
    exp_dir: Path,
    ema: EMA | None = None,
    *,
    train_model: nn.Module | None = None,
    use_fsdp: bool = False,
    keep_last: int = 0,
    milestone: bool = True,
) -> None:
    model_sd = None
    optim_sd = None
    if use_fsdp:
        if train_model is None:
            raise ValueError("FSDP checkpointing requires train_model")
        model_sd, optim_sd = _full_fsdp_state_dict(train_model, optim)
    elif is_main_process():
        model_sd = raw_model.state_dict()
        optim_sd = optim.state_dict()
    if is_main_process():
        ck = exp_dir / "ckpts"
        ck.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model_sd,
            "optimizer": optim_sd,
            "step": int(step),
            "ema": ema.state_dict() if ema is not None else None,
        }
        ema_payload = (
            {"model": ema.state_dict(), "step": int(step)} if ema is not None else None
        )
        _atomic_save(payload, ck / "dit_ckpt_step_latest.pt")
        if ema_payload is not None:
            _atomic_save(ema_payload, ck / "dit_ckpt_step_latest_ema.pt")
        if milestone:
            _atomic_save(payload, ck / f"dit_ckpt_step_{step}.pt")
            if ema_payload is not None:
                _atomic_save(ema_payload, ck / f"dit_ckpt_step_{step}_ema.pt")
            _prune_old_ckpts(ck, keep_last)
    barrier()


def save_best_ckpt(
    raw_model: nn.Module,
    exp_dir: Path,
    fid: float,
    *,
    ema: EMA | None = None,
    train_model: nn.Module | None = None,
    use_fsdp: bool = False,
) -> None:
    model_sd = None
    if use_fsdp:
        model_sd, _ = _full_fsdp_state_dict(train_model, None)
    elif is_main_process():
        model_sd = raw_model.state_dict()
    if is_main_process():
        ck = exp_dir / "ckpts"
        ck.mkdir(parents=True, exist_ok=True)
        _atomic_save({"model": model_sd, "fid": float(fid)}, ck / "dit_ckpt_step_best.pt")
        if ema is not None:
            _atomic_save(
                {"model": ema.state_dict(), "fid": float(fid)},
                ck / "dit_ckpt_step_best_ema.pt",
            )
    barrier()


def _resume_candidates(exp_dir: Path, resume_step) -> list[Path]:
    return dit_resume_candidates(exp_dir / "ckpts", resume_step)


def _load_first_valid(candidates: list[Path]):
    last_exc = None
    for p in candidates:
        try:
            payload = torch.load(p, map_location="cpu", weights_only=False)
            if last_exc is not None and is_main_process():
                log.info(
                    f"[resume] recovered: loaded {p.name} after skipping corrupt ckpt(s)"
                )
            return (p, payload)
        except Exception as exc:
            last_exc = exc
            if is_main_process():
                log.info(
                    f"[resume] WARN: {p.name} failed to load ({type(exc).__name__}: {exc}); trying next candidate"
                )
    raise RuntimeError(
        f"no loadable checkpoint among {[c.name for c in candidates]}"
    ) from last_exc


def maybe_resume(
    raw_model: nn.Module,
    optim,
    exp_dir: Path,
    cfg: dict,
    ema: EMA | None = None,
    *,
    train_model: nn.Module | None = None,
    use_fsdp: bool = False,
) -> int:
    p, payload = _load_first_valid(
        _resume_candidates(exp_dir, cfg.get("resume_step", "latest"))
    )
    step = int(payload.get("step", 0))
    if step < 0 or "model" not in payload:
        raise ValueError(f"invalid resume checkpoint: {p}")
    optimizer_sd = payload.get("optimizer")
    if optimizer_sd is None:
        raise RuntimeError(
            f"{p}: resume requires optimizer state. Use init_checkpoint for weight-only initialization (step=0), including old seed checkpoints."
        )
    if step > 0 and (not optimizer_sd.get("state")):
        raise RuntimeError(f"{p}: nonzero-step resume has empty optimizer state")
    ema_sd = None
    if ema is not None:
        ema_sd = payload.get("ema")
        if ema_sd is None and "ema" not in payload:
            ema_path = p.with_name(p.stem + "_ema.pt")
            if not ema_path.is_file():
                raise RuntimeError(f"{p}: missing companion EMA for strict resume")
            companion = torch.load(ema_path, map_location="cpu", weights_only=False)
            if companion.get("step") != step:
                raise RuntimeError(
                    f"EMA step mismatch: model={step}, EMA={companion.get('step')}"
                )
            ema_sd = companion["model"]
        if ema_sd is None:
            raise RuntimeError(
                f"{p}: EMA enabled but checkpoint contains no EMA; use init_checkpoint"
            )
        ema_sd = _normalize_eval_state_dict(ema_sd)
        expected = ema.state_dict()
        if set(ema_sd) != set(expected) or any(
            (ema_sd[k].shape != expected[k].shape for k in expected)
        ):
            raise RuntimeError(
                f"{p}: EMA parameter keys/shapes do not match the current model"
            )
    if use_fsdp:
        if train_model is None:
            raise ValueError("FSDP resume requires train_model")
        target = getattr(train_model, "_orig_mod", train_model)
        options = StateDictOptions(
            full_state_dict=True, broadcast_from_rank0=False, strict=True
        )
        set_model_state_dict(
            target,
            model_state_dict=_normalize_eval_state_dict(payload["model"]),
            options=options,
        )
        if optimizer_sd.get("state"):
            set_optimizer_state_dict(
                target, optim, optim_state_dict=optimizer_sd, options=options
            )
    else:
        raw_model.load_state_dict(
            _normalize_eval_state_dict(payload["model"]), strict=True
        )
        optim.load_state_dict(optimizer_sd)
    sharded = (
        use_fsdp
        and str(getattr(target, "sharding_strategy", "")).split(".")[-1] != "NO_SHARD"
    )
    _validate_restored_optimizer(
        optim, optimizer_sd, step, use_fsdp=use_fsdp, sharded=sharded
    )
    if ema_sd is not None:
        ema.load_state_dict(ema_sd)
    if is_main_process():
        log.info(f"[resume] restored model/optimizer/EMA from {p} at step={step}")
    barrier()
    return step


def _validate_restored_optimizer(optim, saved, step, *, use_fsdp=False, sharded=True):
    if not isinstance(optim, (torch.optim.Adam, torch.optim.AdamW)):
        return
    error = None
    expected_elements = have = 0
    try:
        expected_elements = sum(
            (
                int(st["exp_avg"].numel())
                for st in saved["state"].values()
                if "exp_avg" in st
            )
        )
        for st in saved["state"].values():
            if not all((k in st for k in ("step", "exp_avg", "exp_avg_sq"))):
                raise RuntimeError("checkpoint has incomplete Adam state")
            if st["exp_avg"].shape != st["exp_avg_sq"].shape:
                raise RuntimeError("checkpoint Adam moment shapes disagree")
        have = 0
        for group in optim.param_groups:
            for param in group["params"]:
                state = optim.state.get(param)
                if not state or param.numel() == 0:
                    continue
                if not all((k in state for k in ("step", "exp_avg", "exp_avg_sq"))):
                    raise RuntimeError("restored Adam state is incomplete")
                if any(
                    (state[k].shape != param.shape for k in ("exp_avg", "exp_avg_sq"))
                ):
                    raise RuntimeError(
                        "restored Adam moment shape does not match parameter"
                    )
                have += param.numel()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    reports = [(have, expected_elements, error)]
    if use_fsdp and dist.is_available() and dist.is_initialized():
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, (have, expected_elements, error))
    errors = [r[2] for r in reports if r[2]]
    if errors:
        raise RuntimeError("optimizer restore validation failed: " + "; ".join(errors))
    if len({r[1] for r in reports}) != 1:
        raise RuntimeError("ranks loaded different optimizer checkpoint sizes")
    restored_counts = (
        [sum((r[0] for r in reports))]
        if use_fsdp and sharded
        else [r[0] for r in reports]
    )
    if any((n != expected_elements or (step > 0 and n == 0) for n in restored_counts)):
        raise RuntimeError(
            f"optimizer restore coverage mismatch: {restored_counts}/{expected_elements} moment elements"
        )


def sample_target_latents(batch, device, enabled):
    z = batch["latent_mean"].to(device, non_blocking=True).float()
    if enabled and "latent_logvar" in batch:
        logvar = batch["latent_logvar"].to(device, non_blocking=True).float()
        mask = batch.get(
            "latent_sample_mask", batch["modality_id"] != SEG_TARGET_MODALITY_ID
        )
        mask = mask.to(device=device, dtype=torch.bool).view(-1, *[1] * (z.ndim - 1))
        z = torch.where(mask, z + (0.5 * logvar).exp() * torch.randn_like(z), z)
    return z


def lr_at(
    step: int,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    lr_min: float,
    schedule: str,
    decay_steps: int | None = None,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if schedule == "constant":
        return base_lr
    horizon = int(decay_steps) if decay_steps else total_steps
    if step >= horizon:
        return lr_min
    progress = float(step - warmup_steps) / float(max(1, horizon - warmup_steps))
    return lr_min + 0.5 * (base_lr - lr_min) * (1.0 + math.cos(math.pi * progress))


def p_label_cond_at(step: int, base: float, schedule: Optional[dict]) -> float:
    if not schedule:
        return base
    s0 = int(schedule.get("start_step", 0))
    s1 = int(schedule.get("end_step", s0))
    v1 = float(schedule.get("end_value", base))
    if step <= s0:
        return base
    if s1 <= s0 or step >= s1:
        return v1
    frac = float(step - s0) / float(s1 - s0)
    return base + (v1 - base) * frac


_CFG_STATE_ORDER = ("B", "T", "R", "F")


def sample_cfg_train_states(
    tasks: list[str], policies: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not policies:
        raise ValueError("cfg_train_states must be a non-empty mapping")
    state_ids = torch.empty(len(tasks), dtype=torch.long, device=device)
    for task in dict.fromkeys((str(t) for t in tasks)):
        policy = policies.get(task)
        if not isinstance(policy, dict):
            raise KeyError(f"cfg_train_states has no policy for task={task!r}")
        probs = torch.tensor(
            [float(policy.get(s, 0.0)) for s in _CFG_STATE_ORDER],
            device=device,
            dtype=torch.float32,
        )
        if bool((probs < 0).any()) or float(probs.sum()) <= 0:
            raise ValueError(f"invalid cfg_train_states[{task!r}]={policy!r}")
        probs = probs / probs.sum()
        idx = [i for i, t in enumerate(tasks) if str(t) == task]
        picks = torch.multinomial(probs, len(idx), replacement=True)
        state_ids[torch.tensor(idx, device=device)] = picks
    keep_text = (state_ids == 1) | (state_ids == 3)
    keep_refs = (state_ids == 2) | (state_ids == 3)
    return (keep_text, keep_refs, state_ids)


def sample_train_timesteps(
    batch_size: int, device: torch.device, cfg: dict
) -> torch.Tensor:
    kind = str(cfg.get("timestep_sampling", "uniform")).lower()
    eps = float(cfg.get("timestep_eps", 1e-05))
    if kind == "uniform":
        t = torch.rand(batch_size, device=device)
    elif kind == "logit_normal":
        mean = float(cfg.get("timestep_logit_mean", 0.0))
        std = float(cfg.get("timestep_logit_std", 1.0))
        t = torch.sigmoid(torch.randn(batch_size, device=device) * std + mean)
    elif kind == "beta":
        alpha = float(cfg.get("timestep_beta_alpha", 2.0))
        beta = float(cfg.get("timestep_beta_beta", 2.0))
        dist_beta = torch.distributions.Beta(alpha, beta)
        t = dist_beta.sample((batch_size,)).to(device)
    else:
        raise ValueError(f"unknown timestep_sampling={kind!r}")
    return t.clamp(eps, 1.0 - eps)


def all_labelled_seg_supervision(
    has_label: torch.Tensor, tasks: list[str] | tuple[str, ...] | None = None
) -> torch.Tensor:
    out = has_label.to(dtype=torch.bool)
    if tasks is not None:
        if len(tasks) != out.numel():
            raise ValueError("tasks length must match has_label")
        eligible = torch.tensor(
            [seg_head_use(str(task))[0] == "train_aux" for task in tasks],
            dtype=torch.bool,
            device=out.device,
        )
        out = out & eligible
    return out


def quantize_generative_segmentation(vol01: np.ndarray, num_classes: int) -> np.ndarray:
    k = max(2, int(num_classes))
    return np.rint(np.clip(vol01, 0.0, 1.0) * float(k - 1)).astype(np.int16)


def segmentation_dice_scores(
    pred: np.ndarray, target: np.ndarray, num_classes: int
) -> dict[str, float]:
    p = np.asarray(pred)
    y = np.asarray(target)

    def dice(a, b):
        denom = int(a.sum()) + int(b.sum())
        return 1.0 if denom == 0 else 2.0 * float((a & b).sum()) / denom

    out = {"foreground": dice(p > 0, y > 0)}
    per_class = []
    for cls in range(1, int(num_classes)):
        value = dice(p == cls, y == cls)
        out[f"class_{cls}"] = value
        per_class.append(value)
    out["macro"] = float(np.mean(per_class)) if per_class else out["foreground"]
    return out


def seg_aux_loss(
    seg_logits: torch.Tensor,
    label: torch.Tensor,
    t: torch.Tensor,
    *,
    num_classes: int,
    class_weights: Optional[torch.Tensor],
    t_weight: str,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = seg_logits.float()
    ce = F.cross_entropy(logits, label, weight=class_weights, reduction="none")
    ce = ce.flatten(1).mean(dim=1)
    probs = logits.softmax(dim=1)
    oh = F.one_hot(label, num_classes).permute(0, 4, 1, 2, 3).to(probs.dtype)
    dims = (2, 3, 4)
    inter = (probs * oh).sum(dims)
    denom = probs.sum(dims) + oh.sum(dims)
    dice_pc = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    dice = dice_pc.mean(dim=1)
    per_sample = ce + dice
    if t_weight == "linear":
        w = t
    elif t_weight in ("quad", "square"):
        w = t * t
    elif t_weight in ("none", "uniform", ""):
        w = torch.ones_like(t)
    else:
        raise ValueError(f"unknown seg t_weight={t_weight!r}")
    if mask is None:
        mask = torch.ones_like(t)
    m = mask.to(per_sample.dtype)
    denom = m.sum().clamp_min(1.0)
    loss = (w * per_sample * m).sum() / denom
    with torch.no_grad():
        coeff_pc = 1.0 - dice_pc
        fg = coeff_pc[:, 1:].mean(dim=1) if num_classes > 1 else coeff_pc.mean(dim=1)
        parts = {
            "seg_ce": (ce * m).sum().detach() / denom,
            "seg_dice_loss": (dice * m).sum().detach() / denom,
            "fg_dice": (fg * m).sum().detach() / denom,
            "seg_n": m.sum().detach(),
        }
    return (loss, parts)


@torch.no_grad()
def sync_or_update_ema(
    ema: EMA,
    raw_model: nn.Module,
    train_model: nn.Module,
    *,
    use_fsdp: bool,
    update: bool,
) -> None:
    if use_fsdp:
        with FSDP.summon_full_params(train_model, writeback=False):
            if update:
                ema.update(raw_model)
            else:
                ema.sync(raw_model)
    elif update:
        ema.update(raw_model)
    else:
        ema.sync(raw_model)


def _normalize_eval_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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


def _save_eval_h5(vol: np.ndarray, path: Path) -> Path:
    if str(path).endswith(".h5"):
        path = path.with_suffix(".nii.gz")
    img = nib.Nifti1Image(
        np.asarray(vol, dtype=np.float32),
        affine=np.asarray(_DHW_RAS_AFFINE, dtype=np.float32),
    )
    img = nib.as_closest_canonical(img)
    nib.save(img, path)
    return path


@torch.no_grad()
def encode_ref_images(
    vae: ViTVAE3D, imgs: torch.Tensor, mmdit_cfg: MMDiTCfg, device: torch.device
) -> torch.Tensor:
    x = imgs.to(device, non_blocking=True).float().clamp(0, 1).mul(2.0).sub(1.0)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        post = vae.encode(x).latent_dist
        mu = tokens_to_grid(post.mu, vae._last_grid_shape)
    return mu.float()


def _decode_latent_to_vol01(
    vae: ViTVAE3D, z: torch.Tensor, mmdit_cfg: MMDiTCfg, device: torch.device
) -> np.ndarray:
    z = z.to(device, non_blocking=True).float()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        img = vae.decode(grid_to_tokens(z), grid=tuple(mmdit_cfg.latent_shape))
    vols = img.float().detach().cpu().numpy()[:, 0]
    return np.clip((vols + 1.0) * 0.5, 0.0, 1.0)


@torch.no_grad()
def _cfg_velocity(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    te: torch.Tensor,
    tm: torch.Tensor,
    mid: torch.Tensor,
    *,
    cfg_text: float,
    cfg_ref: float,
    ref_latent: Optional[torch.Tensor],
    ref_modality_id: Optional[torch.Tensor],
    ref_dt: Optional[torch.Tensor],
    ref_valid: Optional[torch.Tensor],
    ref_type_id: Optional[torch.Tensor] = None,
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
    x3 = x.repeat(K, 1, 1, 1, 1)
    t3 = t.repeat(K)
    te3 = te.repeat(K, 1, 1)
    tm3 = tm.repeat(K, 1)
    mid3 = mid.repeat(K)
    rl3 = ref_latent.repeat(K, 1, 1, 1, 1, 1)
    rmod3 = ref_modality_id.repeat(K, 1)
    rdt3 = ref_dt.repeat(K, 1)
    rtid3 = ref_type_id.repeat(K, 1) if ref_type_id is not None else None
    keep_refs = torch.cat(
        [
            torch.zeros(b, dtype=torch.bool, device=device),
            torch.zeros(b, dtype=torch.bool, device=device),
            torch.ones(b, dtype=torch.bool, device=device),
        ]
    )
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


@torch.no_grad()
def _sample_latents_from_cached_batch(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    cfg: dict,
    mmdit_cfg: MMDiTCfg,
    vae: Optional[ViTVAE3D] = None,
    *,
    force_no_refs: bool = False,
    seg_steps: Optional[list] = None,
    eval_mode: Optional[str] = None,
) -> tuple[torch.Tensor, dict]:
    infer = cfg.get("infer", {})
    steps = int(infer.get("steps", 50))
    task_cfg = (infer.get("task_cfg") or {}).get(str(eval_mode), {})
    cfg_text = float(task_cfg.get("text", infer.get("cfg_text", 4.0)))
    cfg_ref = float(task_cfg.get("ref", infer.get("cfg_ref", 1.0)))
    ts = make_ts(
        steps,
        device=device,
        kind=infer.get("schedule", "power"),
        t_min=float(infer.get("t_min", 0.001)),
        rho=float(infer.get("rho", 3.0)),
    )
    te = batch["text_emb"].to(device, dtype=torch.bfloat16, non_blocking=True)
    tm = batch["text_mask"].to(device, non_blocking=True).long()
    mid = batch["modality_id"].to(device, non_blocking=True).long()
    b = te.shape[0]
    R = int(mmdit_cfg.max_refs)
    grid = tuple(mmdit_cfg.latent_shape)
    z_shape = (mmdit_cfg.in_channels,) + grid
    spatial_enabled = bool(mmdit_cfg.use_ref_slab)
    ref_latent = (
        torch.zeros(b, R, mmdit_cfg.in_channels, *grid, device=device)
        if spatial_enabled
        else None
    )
    ref_modality_id = (
        torch.zeros(b, R, dtype=torch.long, device=device) if spatial_enabled else None
    )
    ref_dt = torch.zeros(b, R, device=device) if spatial_enabled else None
    ref_valid = (
        torch.zeros(b, R, dtype=torch.bool, device=device) if spatial_enabled else None
    )
    ref_type_id = (
        torch.zeros(b, R, dtype=torch.long, device=device) if spatial_enabled else None
    )
    if spatial_enabled and R > 0 and ("ref_latent" in batch):
        ref_latent = batch["ref_latent"].to(device, non_blocking=True).float()
        ref_modality_id = batch["ref_modality_id"].to(device, non_blocking=True).long()
        ref_dt = batch["ref_dt"].to(device, non_blocking=True).float()
        ref_valid = batch["ref_valid"].to(device, non_blocking=True).bool()
        rti = batch.get("ref_type_id")
        if rti is not None:
            ref_type_id = rti.to(device, non_blocking=True).long()
        needs = batch.get("ref_needs_encode")
        if (
            vae is not None
            and needs is not None
            and bool(needs.any())
            and ("ref_image" in batch)
        ):
            needs = needs.to(device).bool()
            ref_img = batch["ref_image"].to(device, non_blocking=True)
            for s in range(R):
                if not bool(needs[:, s].any()):
                    continue
                enc = encode_ref_images(
                    vae, ref_img[:, s].unsqueeze(1), mmdit_cfg, device
                )
                ref_latent[:, s] = torch.where(
                    needs[:, s].view(-1, 1, 1, 1, 1), enc, ref_latent[:, s]
                )
        if force_no_refs:
            ref_valid = torch.zeros_like(ref_valid)
    seg_set = (
        set((int(s) for s in seg_steps or [] if 0 <= int(s) < steps))
        if getattr(model, "seg_head_enabled", False)
        else set()
    )
    if any(
        (
            not seg_head_readout_allowed(task, eval_mode)
            for task in batch.get("task", ["modality_only"])
        )
    ):
        seg_set.clear()
    full_shape = tuple((int(v) for v in cfg["spatial_size"]))

    def _fwd_kwargs():
        return dict(
            ref_latent=ref_latent,
            ref_modality_id=ref_modality_id,
            ref_dt=ref_dt,
            ref_valid=ref_valid,
            ref_type_id=ref_type_id,
        )

    x = torch.randn((b,) + z_shape, device=device)
    seg_masks: dict = {}
    for i in range(steps):
        t = ts[i].expand(b)
        if i in seg_set:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _v, seg_logits = model(
                    x,
                    t,
                    te,
                    tm,
                    mid,
                    text_drop_mask=torch.zeros(b, dtype=torch.bool, device=device),
                    return_seg=True,
                    **_fwd_kwargs(),
                )
            seg_low = seg_logits.float().argmax(dim=1, keepdim=True).float()
            seg_masks[i] = (
                F.interpolate(seg_low, size=full_shape, mode="nearest")[:, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        v = _cfg_velocity(
            model,
            x,
            t,
            te,
            tm,
            mid,
            cfg_text=cfg_text,
            cfg_ref=cfg_ref,
            ref_latent=ref_latent,
            ref_modality_id=ref_modality_id,
            ref_dt=ref_dt,
            ref_valid=ref_valid,
            ref_type_id=ref_type_id,
        )
        x = x + v * (ts[i + 1] - ts[i])
    return (x, seg_masks)


@torch.no_grad()
def run_validation_eval(
    cfg: dict,
    raw_model: nn.Module,
    train_model: nn.Module,
    device: torch.device,
    exp_dir: Path,
    step: int,
    mmdit_cfg: MMDiTCfg,
    *,
    ema: EMA | None = None,
    use_fsdp: bool = False,
) -> float | None:
    infer = cfg.get("infer", {})
    primary_fid_mode = str(infer.get("best_fid_mode", "modality_only"))
    fid_enable = bool(infer.get("fid_enable", False))
    save_n = int(infer.get("save_num_volumes", 5))
    fid_n = int(infer.get("fid_num_volumes", 0)) if fid_enable else 0
    seg_start = int((cfg["mmdit"].get("seg_head", {}) or {}).get("start_step", 0))
    mask_eval_enabled = bool(infer.get("mask_enable", True))
    mask_eval_start = int(infer.get("mask_start_step", seg_start))
    task_weights = {
        str(k): float(v)
        for k, v in (cfg.get("task_weights", {}) or {}).items()
        if float(v) > 0
    }
    mask_eval_active = (
        "mask_guide" in task_weights
        and mmdit_cfg.use_ref_slab
        and mask_eval_enabled
        and (step >= mask_eval_start)
    )
    label_eval_active = bool(set(task_weights) & LABEL_REQUIRED_TASKS)
    n = max(fid_n, save_n)
    if n <= 0:
        return None
    world = (
        dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    )
    rank = dist.get_rank() if world > 1 else 0
    per_rank_full = ema is not None or not use_fsdp
    distributed = world > 1 and per_rank_full
    if distributed:
        eval_sd = (
            ema.state_dict()
            if ema is not None
            else {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
        )
    else:
        eval_sd = None
        if ema is not None:
            if is_main_process():
                eval_sd = ema.state_dict()
        elif use_fsdp:
            eval_sd, _ = _full_fsdp_state_dict(train_model, optim=None)
        elif is_main_process():
            eval_sd = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
        if not is_main_process():
            barrier()
            return None
    from data import build_val_pair_loader as build_val_loader
    from data.dataset import multitask_collate as pair_collate, IneligibleSampleError

    setup_error = None
    try:
        rng = random.Random(int(cfg.get("seed", 42)))
        val_ds = build_val_loader(
            cfg, world_size=1, rank=0, load_labels=label_eval_active
        ).dataset
        if not val_ds.records:
            raise ValueError("validation dataset is empty")
        insp_mods = list(infer.get("infer_modalities", [0, 1, 2, 3]))
        seg_steps_insp = list(infer.get("infer_seg_steps", [10]))
        all_by_mod: dict[int, list[int]] = {}
        labeled_by_mod: dict[int, list[int]] = {}
        thin_by_mod: dict[int, list[int]] = {}
        thin_axial_by_mod: dict[int, list[int]] = {}
        thin_axial_labeled_by_mod: dict[int, list[int]] = {}
        thick_by_mod: dict[int, list[int]] = {}
        for i, r in enumerate(val_ds.records):
            mod = int(r["images"][0]["modality"])
            all_by_mod.setdefault(mod, []).append(i)
            if r.get("label_h5"):
                labeled_by_mod.setdefault(mod, []).append(i)
            if val_ds._rec_is_thin(r):
                thin_by_mod.setdefault(mod, []).append(i)
                if val_ds._acq(r["images"][0]["path"])[1] == "axial":
                    thin_axial_by_mod.setdefault(mod, []).append(i)
                    if r.get("label_h5"):
                        thin_axial_labeled_by_mod.setdefault(mod, []).append(i)
            if val_ds._rec_is_thick(r):
                thick_by_mod.setdefault(mod, []).append(i)

        def _pool_by_mod(cand_by_mod):
            selected: dict[int, list[int]] = {}
            active_mods = [int(m) for m in insp_mods if cand_by_mod.get(int(m))]
            per_mod = max(
                1,
                -(-max(fid_n, 1) // max(1, len(active_mods))),
                -(-max(save_n, 1) // max(1, len(active_mods))),
            )
            for mod in insp_mods:
                cand = cand_by_mod.get(int(mod), [])
                selected[int(mod)] = rng.sample(cand, min(per_mod, len(cand)))
            p = []
            max_len = max((len(v) for v in selected.values()), default=0)
            for j in range(max_len):
                for mod in insp_mods:
                    rows = selected.get(int(mod), [])
                    if j >= len(rows):
                        continue
                    idx = rows[j]
                    p.append(
                        {
                            "idx": idx,
                            "mod": int(mod),
                            "inspect": len(p) < save_n,
                            "save_id": f"val{idx}",
                        }
                    )
            return p

        source_pools = {
            "any": _pool_by_mod(all_by_mod),
            "labeled": _pool_by_mod(labeled_by_mod),
            "thin": _pool_by_mod(thin_by_mod),
            "thin_axial": _pool_by_mod(thin_axial_by_mod),
            "thin_axial_labeled": _pool_by_mod(thin_axial_labeled_by_mod),
            "thick": _pool_by_mod(thick_by_mod),
        }
        _acc_index = getattr(val_ds, "acc_index", {})
        _mm_by_mod: dict[int, list[int]] = {}
        for i, r in enumerate(val_ds.records):
            acc = os.path.dirname(str(r["images"][0]["path"])).strip("/")
            if len(_acc_index.get(acc, {})) >= 2:
                _mm_by_mod.setdefault(int(r["images"][0]["modality"]), []).append(i)
        source_pools["multimod"] = _pool_by_mod(_mm_by_mod)
        _mm_thin_axial_by_mod: dict[int, list[int]] = {}
        for mod, rows in thin_axial_by_mod.items():
            _mm_thin_axial_by_mod[mod] = [
                i
                for i in rows
                if len(
                    _acc_index.get(
                        os.path.dirname(
                            str(val_ds.records[i]["images"][0]["path"])
                        ).strip("/"),
                        {},
                    )
                )
                >= 2
            ]
        source_pools["multimod_thin_axial"] = _pool_by_mod(_mm_thin_axial_by_mod)
        for item in source_pools["multimod_thin_axial"]:
            item["missing_target_mod"] = int(item["mod"])
        _seg4 = list(getattr(val_ds, "_seg_acc_by_count", {}).get(4, []))
        _roster = [
            (a, sorted(_acc_index.get(a, {})))
            for a in _seg4
            if len(_acc_index.get(a, {})) >= 4
        ]
        _roster = rng.sample(_roster, min(save_n, len(_roster))) if _roster else []
        source_pools["seg_roster"] = [
            {
                "idx": k,
                "mod": 4,
                "inspect": k < save_n,
                "seg_acc": a,
                "seg_mods": ms,
                "save_id": f"seg{k}",
            }
            for k, (a, ms) in enumerate(_roster)
        ]
        image_h5_paths = {
            im["path"]: im["h5"] for record in val_ds.records for im in record["images"]
        }
        case_ids = build_eval_case_ids(list(image_h5_paths.values()))
        text_max = int(cfg.get("text_fixed_length", cfg.get("text_max_length", 256)))
        spatial = tuple(cfg["spatial_size"])
        out_dir = exp_dir / "infer" / f"step_{step}"
        out_dir.mkdir(parents=True, exist_ok=True)
        fid = None
        vae = None
        eval_model = None
        eval_qwen = None
        EVAL_SPEC = {
            "modality_only": {
                "task": "modality_only",
                "keep_refs": False,
                "source": "thin_axial",
                "target": "same",
            },
            "mask_guide": {
                "task": "mask_guide",
                "keep_refs": True,
                "source": "thin_axial_labeled",
                "target": "same",
                "save_refs": True,
            },
            "inpaint": {
                "task": "inpaint",
                "keep_refs": True,
                "source": "thin_axial_labeled",
                "target": "same",
                "save_refs": True,
                "prompt_task": "inpaint",
            },
            "sr": {
                "task": "sr",
                "keep_refs": True,
                "source": "thin_axial",
                "target": "same",
                "save_refs": True,
            },
            "missing": {
                "task": "missing",
                "keep_refs": True,
                "source": "multimod_thin_axial",
                "target": "vary",
                "save_refs": True,
            },
            "whole_brain": {
                "task": "whole_brain",
                "keep_refs": True,
                "source": "thin_axial",
                "target": "same",
                "save_refs": True,
            },
            "deblur": {
                "task": "deblur",
                "keep_refs": True,
                "source": "thin_axial",
                "target": "same",
                "save_refs": True,
            },
            "dealias": {
                "task": "dealias",
                "keep_refs": True,
                "source": "thin_axial",
                "target": "same",
                "save_refs": True,
            },
            "motion": {
                "task": "motion",
                "keep_refs": True,
                "source": "any",
                "target": "same",
                "save_refs": True,
                "match_quality": True,
            },
            "seg_1mod": {
                "task": "seg",
                "keep_refs": True,
                "source": "seg_roster",
                "target": "vary",
                "save_refs": True,
                "fid": False,
                "seg_ref_count": 1,
            },
            "seg_2mod": {
                "task": "seg",
                "keep_refs": True,
                "source": "seg_roster",
                "target": "vary",
                "save_refs": True,
                "fid": False,
                "seg_ref_count": 2,
                "derived_from": "seg_1mod",
            },
            "seg_3mod": {
                "task": "seg",
                "keep_refs": True,
                "source": "seg_roster",
                "target": "vary",
                "save_refs": True,
                "fid": False,
                "seg_ref_count": 3,
                "derived_from": "seg_1mod",
            },
            "seg_4mod": {
                "task": "seg",
                "keep_refs": True,
                "source": "seg_roster",
                "target": "vary",
                "save_refs": True,
                "fid": False,
                "seg_ref_count": 4,
                "derived_from": "seg_1mod",
            },
            "inpaint_tumor": {
                "task": "inpaint",
                "keep_refs": True,
                "source": "thin_axial_labeled",
                "target": "same",
                "void_tumor": True,
                "save_refs": True,
                "fid": False,
                "derived_from": "inpaint",
                "gt_role": "original_target",
                "prompt_task": "inpaint",
            },
            "inpaint_lesion": {
                "task": "inpaint",
                "keep_refs": True,
                "source": "thin_axial_labeled",
                "target": "same",
                "void_tumor": True,
                "save_refs": True,
                "fid": False,
                "derived_from": "inpaint",
                "gt_role": "original_target",
                "inpaint_content": "auto",
            },
            "sr_native_thick": {
                "task": "sr",
                "keep_refs": True,
                "source": "thick",
                "target": "same",
                "native_sr": True,
                "save_refs": True,
                "fid": False,
                "force_thin_prompt": True,
                "derived_from": "sr",
                "gt_role": "source_input",
            },
            "deblur_native_thick": {
                "task": "deblur",
                "keep_refs": True,
                "source": "thick",
                "target": "same",
                "native_sr": True,
                "save_refs": True,
                "fid": False,
                "force_thin_prompt": True,
                "derived_from": "deblur",
                "gt_role": "source_input",
            },
            "dealias_native_thick": {
                "task": "dealias",
                "keep_refs": True,
                "source": "thick",
                "target": "same",
                "native_sr": True,
                "save_refs": True,
                "fid": False,
                "force_thin_prompt": True,
                "derived_from": "dealias",
                "gt_role": "source_input",
            },
        }
        eval_task_cfg = infer.get("eval_tasks") or {}
        for mode, override in eval_task_cfg.items():
            if mode in EVAL_SPEC and isinstance(override, dict):
                EVAL_SPEC[mode].update(override)
        _TASK2MODE = {
            v["task"]: k for k, v in EVAL_SPEC.items() if "derived_from" not in v
        }
        _trained = [
            t
            for t, w in (cfg.get("task_weights") or {"modality_only": 1.0}).items()
            if float(w) > 0
        ]
        modes_to_eval: list[str] = []
        for t in _trained:
            m = _TASK2MODE.get(t, t)
            if m in EVAL_SPEC and m not in modes_to_eval:
                modes_to_eval.append(m)
        if "modality_only" not in modes_to_eval:
            modes_to_eval = ["modality_only"] + modes_to_eval
        for m, spec in EVAL_SPEC.items():
            base = spec.get("derived_from")
            if base and base in modes_to_eval and (m not in modes_to_eval):
                modes_to_eval.append(m)
        modes_to_eval = [
            m
            for m in modes_to_eval
            if EVAL_SPEC[m].get("enabled", True)
            and (m != "mask_guide" or mask_eval_active)
        ]
        if rank == 0:
            log.info(f"[val] eval modes (task-driven): {modes_to_eval}")
        local_fid_by_mode: dict[str, list] = {m: [] for m in modes_to_eval}
        local_reals_by_mode: dict[str, list] = {m: [] for m in modes_to_eval}
        local_prompts: list[dict] = []
        local_dice: dict[int, list] = {}
        local_pair_metrics: list[dict] = []
        local_generative_seg_metrics: list[dict] = []
    except Exception as exc:
        setup_error = f"rank{rank}: {type(exc).__name__}: {exc}"
    setup_errors = [setup_error]
    if distributed:
        setup_errors = [None] * world
        dist.all_gather_object(setup_errors, setup_error)
    if any(setup_errors):
        if is_main_process():
            failure_dir = exp_dir / "infer" / f"step_{step}"
            try:
                failure_dir.mkdir(parents=True, exist_ok=True)
                (failure_dir / "validation_status.json").write_text(
                    json.dumps(
                        {
                            "step": step,
                            "complete": False,
                            "stage": "setup",
                            "errors": setup_errors,
                        },
                        indent=2,
                    )
                )
            except OSError as exc:
                log.error(f"[val] cannot save setup failure: {exc}")
            log.error(f"[val] setup failed: {setup_errors}")
        barrier()
        return None
    mode_stats = {}
    for mode in modes_to_eval:
        size = len(source_pools.get(EVAL_SPEC[mode]["source"], []))
        assigned = len(range(rank, size, world)) if distributed else size
        mode_stats[mode] = dict(
            requested=assigned,
            eligible=0,
            generated=0,
            saved=0,
            ok=0,
            skipped=0,
            failed=0,
            ineligible=0,
            not_run=0,
            empty_pool=int(size == 0 and rank == 0),
        )
    eval_errors = []
    try:
        eval_model = MMDiT3D(mmdit_cfg).to(device).eval()
        eval_model.load_state_dict(_normalize_eval_state_dict(eval_sd), strict=True)
        log_rank(
            f"building VAE for validation begin ckpt={cfg['vae_ckpt']}", device=device
        )
        vae = build_vae(cfg).to(device).eval()
        log_rank("building VAE for validation done", device=device)
        from utils.prompts import compose as compose_prompt

        seed0 = int(cfg.get("seed", 42))
        eval_one_mm_prob = float(infer.get("eval_one_mm_prompt_fraction", 0.5))

        def _eval_target_mm(mode, item, ordinal=None):
            if eval_one_mm_prob == 0.5 and ordinal is not None:
                return 1 if int(ordinal) % 2 == 0 else None
            key = f"{seed0}|{mode}|{item.get('save_id', item.get('idx', 0))}"
            return 1 if random.Random(key).random() < eval_one_mm_prob else None

        def _prompt_for(mode, tgt, idx, item, sample, ordinal):
            target_mm = _eval_target_mm(mode, item, ordinal)
            ic = EVAL_SPEC[mode].get("inpaint_content")
            if ic is not None:
                content = ic
                if ic == "auto":
                    content = "healthy"
                    rec_i = val_ds.records[idx]
                    if rec_i.get("label_h5"):
                        try:
                            content = val_ds._lesion_content(
                                val_ds._read_label_full_raw(rec_i["label_h5"])
                            )
                        except Exception:
                            content = "healthy"
                return compose_prompt(
                    "inpaint",
                    tgt,
                    "thin",
                    "axial",
                    inpaint_content=content,
                    target_mm=target_mm,
                )
            ctask = EVAL_SPEC[mode].get("prompt_task", EVAL_SPEC[mode]["task"])
            if ctask in ("sr", "deblur", "dealias"):
                try:
                    dp = json.loads(sample.get("degradation_params") or "{}")
                except Exception:
                    dp = {}
                composite = ctask == "sr" or str(dp.get("composite", "")).startswith(
                    "sr+"
                )
                if composite:
                    exact = bool(dp.get("prompt_exact_input_mm", False))
                    input_mm = dp.get("sr_requested_input_mm", dp.get("sr_in_mm"))
                    return compose_prompt(
                        ctask,
                        tgt,
                        "thin",
                        "axial",
                        target_mm=target_mm,
                        synthetic_input_mm=input_mm if exact else None,
                        synthetic_input_generic=not exact,
                    )
            return compose_prompt(ctask, tgt, "thin", "axial", target_mm=target_mm)

        def _get_text_emb(prompt_str):
            nonlocal eval_qwen
            try:
                return val_ds.prompt_store.get(prompt_str)
            except KeyError:
                if eval_qwen is None:
                    from models.qwen3vl_text import Qwen3VLTextEncoder

                    eval_qwen = Qwen3VLTextEncoder(
                        cfg["qwen3vl_path"],
                        device=device,
                        chat_template=bool(cfg.get("qwen3vl_chat_template", True)),
                        add_generation_prompt=bool(
                            cfg.get("qwen3vl_add_generation_prompt", True)
                        ),
                        layer=int(cfg.get("qwen3vl_layer", -1)),
                        max_length=int(cfg.get("text_max_length", 256)),
                    ).eval()
                return eval_qwen.encode_valid_list([prompt_str])[0]

        written: set = set()
        records: dict = {}
        in_mask_by: dict = {}
        pair_masks: dict = {}
        pair_segs: dict = {}

        def _save_refs(batch, file_base, mode, rec):
            rv = batch.get("ref_valid")
            rne = batch.get("ref_needs_encode")
            ri = batch.get("ref_image")
            rl = batch.get("ref_latent")
            if rv is None:
                return
            valid_slots = [k for k in range(int(rv.shape[1])) if bool(rv[0, k])]
            roles = (batch.get("ref_roles") or [[]])[0]
            paths = (batch.get("ref_paths") or [[]])[0]
            rmods = batch.get("ref_modality_id")
            rec.setdefault("refs", [])
            for pos, k in enumerate(valid_slots, start=1):
                if (
                    rne is not None
                    and bool(rne[0, k])
                    and (ri is not None)
                    and (ri.shape[1] > k)
                ):
                    vol = np.clip(ri[0, k].float().cpu().numpy(), 0.0, 1.0)
                    decoded = False
                elif rl is not None and rl.shape[1] > k:
                    vol = _decode_latent_to_vol01(
                        vae, rl[:, k].float(), mmdit_cfg, device
                    )[0]
                    decoded = True
                else:
                    continue
                role = str(roles[k] if k < len(roles) and roles[k] else "reference")
                rmod = int(rmods[0, k]) if rmods is not None else -1
                rn = str(
                    _save_eval_h5(
                        vol,
                        out_dir
                        / (
                            file_base
                            + eval_ref_suffix(
                                pos, len(valid_slots), role, rmod, decoded=decoded
                            )
                        ),
                    ).relative_to(out_dir)
                )
                rec["files"][f"{mode}_ref_{pos}-{len(valid_slots)}"] = rn
                rec["refs"].append(
                    {
                        "position": pos,
                        "count": len(valid_slots),
                        "slot": k,
                        "role": role,
                        "modality_id": rmod,
                        "source_path": str(paths[k] if k < len(paths) else ""),
                        "file": rn,
                    }
                )

        for mode in modes_to_eval:
            spec = EVAL_SPEC[mode]
            src = source_pools.get(spec["source"], [])
            stats = mode_stats[mode]
            if not src:
                continue
            my = (
                list(range(len(src)))[rank::world]
                if distributed
                else list(range(len(src)))
            )
            val_ds._force_task = spec["task"]
            val_ds._inpaint_void_tumor = bool(spec.get("void_tumor", False))
            val_ds._sr_native_input = bool(spec.get("native_sr", False))
            val_ds._force_seg_ref_count = spec.get("seg_ref_count")
            for pe in my:
                item = src[pe]
                idx = item["idx"]
                inspect = bool(item["inspect"] and spec.get("save", True))
                mode_slot = modes_to_eval.index(mode)
                random.seed(
                    seed0 + 100000 * mode_slot + 1000 * int(item.get("seed_slot", pe))
                )
                val_ds._force_missing_target_mod = item.get("missing_target_mod")
                val_ds._force_seg_acc = item.get("seg_acc")
                _full_mods = item.get("seg_mods") or []
                val_ds._force_seg_mods = (
                    list(_full_mods)[: int(spec.get("seg_ref_count", 1))]
                    if spec["source"] == "seg_roster"
                    else None
                )
                try:
                    sample = val_ds[idx]
                except IneligibleSampleError as e:
                    stats["skipped"] += 1
                    if rank == 0:
                        log.warning(f"[val] skip {mode} idx={idx}: {e}")
                    continue
                except Exception as exc:
                    stats["failed"] += 1
                    eval_errors.append(
                        dict(
                            mode=mode,
                            idx=idx,
                            stage="sample",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    continue
                if sample.get("task") != spec["task"]:
                    stats["ineligible"] += 1
                    continue
                try:
                    stats["eligible"] += 1
                    tgt = int(sample["modality_id"])
                    force_thin = bool(spec.get("force_thin_prompt", False))
                    if force_thin:
                        _target_mm = _eval_target_mm(mode, item, pe)
                        _dp = {}
                        try:
                            _dp = json.loads(sample.get("degradation_params") or "{}")
                        except Exception:
                            _dp = {}
                        _nat_mm = _dp.get("sr_native_axis_mm")
                        _mm = (
                            val_ds._snap_sr_mm(1.0, _nat_mm)
                            if _nat_mm is not None
                            else None
                        )
                        if _mm is not None:
                            prompt_str = compose_prompt(
                                spec["task"],
                                tgt,
                                "thin",
                                "axial",
                                native_input_mm=_mm[1],
                                target_mm=_target_mm,
                            )
                        else:
                            prompt_str = compose_prompt(
                                spec["task"],
                                tgt,
                                "thin",
                                "axial",
                                native_input_generic=True,
                                target_mm=_target_mm,
                            )
                    elif spec["task"] in ("seg", "motion") or bool(
                        spec.get("match_quality", False)
                    ):
                        prompt_str = sample.get("prompt")
                    else:
                        prompt_str = _prompt_for(mode, tgt, idx, item, sample, pe)
                    if prompt_str is None:
                        stats["skipped"] += 1
                        continue
                    matched_prompt = (
                        spec.get("match_quality") or spec["task"] in ("seg", "motion")
                    ) and (not force_thin)
                    requested_q = (
                        str(sample.get("target_quality", ""))
                        if matched_prompt
                        else "thin"
                    )
                    requested_v = (
                        str(sample.get("target_view", ""))
                        if matched_prompt
                        else "axial"
                    )
                    sample["text_emb"] = _get_text_emb(prompt_str)
                    sample["prompt"] = prompt_str
                    batch = pair_collate([sample], fixed_text_len=text_max)
                    torch.manual_seed(seed0 + pe)
                    latents, seg_masks = _sample_latents_from_cached_batch(
                        eval_model,
                        batch,
                        device,
                        cfg,
                        mmdit_cfg,
                        vae=vae,
                        force_no_refs=not spec["keep_refs"],
                        seg_steps=seg_steps_insp
                        if inspect and seg_head_readout_allowed(spec["task"], mode)
                        else None,
                        eval_mode=mode,
                    )
                    gen01 = _decode_latent_to_vol01(vae, latents, mmdit_cfg, device)
                    stats["generated"] += 1
                    seg_pred = seg_gt = None
                    real01 = None
                    do_fid = fid_enable and bool(spec.get("fid", True))
                    if do_fid or inspect:
                        real01 = _decode_latent_to_vol01(
                            vae, batch["latent_mean"], mmdit_cfg, device
                        )
                    if spec["task"] == "seg":
                        num_classes = int(
                            (cfg["mmdit"].get("seg_head", {}) or {}).get(
                                "num_classes", 4
                            )
                        )
                        seg_pred = quantize_generative_segmentation(
                            gen01[0], num_classes
                        )
                        gt = batch["label_eval_full"][0].float()[None, None]
                        if tuple(gt.shape[2:]) != tuple(spatial):
                            gt = F.interpolate(gt, size=tuple(spatial), mode="nearest")
                        seg_gt = gt[0, 0].cpu().numpy().astype(np.int16)
                        scores = segmentation_dice_scores(seg_pred, seg_gt, num_classes)
                        local_generative_seg_metrics.append(
                            {"step": int(step), "mode": mode, "idx": int(idx), **scores}
                        )
                    if do_fid:
                        local_fid_by_mode.setdefault(mode, []).append(
                            gen01[0].astype(np.float16)
                        )
                        local_reals_by_mode.setdefault(mode, []).append(
                            real01[0].astype(np.float16)
                        )
                    if not inspect:
                        stats["ok"] += 1
                        continue
                    save_id = str(item.get("save_id", f"val{idx}"))
                    key = (save_id, tgt, mode)
                    rec = records.setdefault(
                        key,
                        {
                            "idx": idx,
                            "save_id": save_id,
                            "mode": mode,
                            "tgt_modality": tgt,
                            "files": {},
                            "prompts": {},
                            "seg_steps": sorted(seg_masks),
                        },
                    )
                    rec["target_path"] = str(sample.get("path", ""))
                    rec["target_quality"] = str(sample.get("target_quality", ""))
                    rec["target_view"] = str(sample.get("target_view", ""))
                    rec["requested_quality"] = requested_q
                    rec["requested_view"] = requested_v
                    _su = seg_head_use(spec["task"], mode)
                    rec["seg_head_use"] = f"{_su[0]}: {_su[1]}"
                    rec["cfg"] = (infer.get("task_cfg") or {}).get(mode, {})
                    if spec["task"] == "sr":
                        rec["sr"] = {
                            "synthetic_downsample": bool(
                                sample.get("sr_synthetic", False)
                            ),
                            "factor": int(sample.get("sr_factor", 0)),
                            "axis": int(sample.get("sr_axis", -1)),
                        }
                    dp = sample.get("degradation_params")
                    if dp:
                        rec["degradation"] = dp
                    rec["prompts"][mode] = prompt_str
                    case_source = (
                        next((p for p in sample["ref_paths"] if p))
                        if spec["task"] == "seg"
                        else sample["path"]
                    )
                    case_source = image_h5_paths.get(case_source, case_source)
                    case_id = case_ids[study_key(case_source)]
                    file_base = eval_file_base(
                        spec["task"], mode, case_id, tgt, save_id
                    )
                    rec.update(
                        case_id=case_id,
                        case_source=str(case_source),
                        file_prefix=file_base,
                        output_layout_version=3,
                    )
                    (out_dir / file_base).parent.mkdir(parents=True, exist_ok=True)
                    name = f"{file_base}_00_generated.h5"
                    name = str(
                        _save_eval_h5(
                            seg_pred if seg_pred is not None else gen01[0],
                            out_dir / name,
                        ).relative_to(out_dir)
                    )
                    rec["files"][mode] = name
                    prompt_name = f"{file_base}_90_prompt.txt"
                    (out_dir / prompt_name).write_text(
                        prompt_str + "\n", encoding="utf-8"
                    )
                    rec["files"]["prompt"] = prompt_name
                    pair_id = (save_id, tgt)
                    if mode in ("modality_only", "mask_guide"):
                        lf = batch.get("label_eval_full")
                        if lf is None:
                            lf = batch.get("label_full")
                        if lf is not None and lf[0].ndim == 3 and (lf[0].numel() > 0):
                            lv = lf[0].float()[None, None]
                            if tuple(lv.shape[2:]) != tuple(spatial):
                                lv = F.interpolate(
                                    lv, size=tuple(spatial), mode="nearest"
                                )
                            pair_masks[pair_id] = (lv[0, 0] > 0).cpu()
                            if mode == "mask_guide":
                                in_mask_by[key] = pair_masks[pair_id]
                    for st, sm in sorted(seg_masks.items()):
                        sname = f"{file_base}_20_seg_head_step{st:03d}.h5"
                        sname = str(
                            _save_eval_h5(sm[0], out_dir / sname).relative_to(out_dir)
                        )
                        rec["files"][f"{mode}_seg{st}"] = sname
                        im = in_mask_by.get(key)
                        if mode in ("modality_only", "mask_guide"):
                            pair_segs[save_id, tgt, mode, int(st)] = (
                                torch.from_numpy(sm[0]) > 0
                            )
                        if mode == "mask_guide" and im is not None:
                            seg_bin = torch.from_numpy(sm[0]).to(im.device) > 0
                            inter = float((im & seg_bin).sum())
                            denom = float(im.sum()) + float(seg_bin.sum())
                            local_dice.setdefault(int(st), []).append(
                                2.0 * inter / denom if denom > 0 else 0.0
                            )
                    gt_role = str(spec.get("gt_role", "gt"))
                    target_role = (
                        "label_gt"
                        if seg_gt is not None
                        else "original_target_vae_recon"
                        if gt_role == "original_target"
                        else "source_input_vae_recon"
                        if gt_role == "source_input"
                        else "target_vae_recon"
                    )
                    gt_name = f"{file_base}_01_{target_role}.h5"
                    if gt_name not in written and (
                        real01 is not None or seg_gt is not None
                    ):
                        _save_eval_h5(
                            seg_gt if seg_gt is not None else real01[0],
                            out_dir / gt_name,
                        )
                        written.add(gt_name)
                    rec["files"][gt_role] = gt_name[:-3] + ".nii.gz"
                    lbl = batch.get("label_eval_full")
                    if lbl is None:
                        lbl = batch.get("label_full")
                    if (
                        bool(batch["has_label"][0])
                        and lbl is not None
                        and (lbl[0].ndim == 3)
                        and (lbl[0].numel() > 0)
                    ):
                        lbl_name = f"{file_base}_02_label_gt.h5"
                        if lbl_name not in written:
                            lv = lbl[0].float()[None, None]
                            if tuple(lv.shape[2:]) != tuple(spatial):
                                lv = F.interpolate(
                                    lv, size=tuple(spatial), mode="nearest"
                                )
                            _save_eval_h5(lv[0, 0].cpu().numpy(), out_dir / lbl_name)
                            written.add(lbl_name)
                        rec["files"]["label"] = lbl_name[:-3] + ".nii.gz"
                    if spec.get("save_refs", False):
                        _save_refs(batch, file_base, mode, rec)
                    stats["saved"] += 1
                    stats["ok"] += 1
                except Exception as exc:
                    stats["failed"] += 1
                    eval_errors.append(
                        dict(
                            mode=mode,
                            idx=idx,
                            stage="generation_or_save",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
        if rank == 0:
            log.info(
                "[infer] eval denominators (audit SH-04): "
                + "; ".join((f"{m}={s}" for m, s in sorted(mode_stats.items())))
            )
        val_ds._force_task = None
        val_ds._inpaint_void_tumor = False
        val_ds._sr_native_input = False
        val_ds._force_missing_target_mod = None
        val_ds._force_seg_ref_count = None
        val_ds._force_seg_acc = None
        val_ds._force_seg_mods = None
        for pair_id, cond_mask in pair_masks.items():
            save_id, tgt = pair_id
            for st in seg_steps_insp:
                modality_only_seg = pair_segs.get((save_id, tgt, "modality_only", int(st)))
                mask_seg = pair_segs.get((save_id, tgt, "mask_guide", int(st)))
                if modality_only_seg is None or mask_seg is None:
                    continue
                cond = cond_mask.bool()
                outside = ~cond
                cond_n = max(1, int(cond.sum()))

                def _dice(a, b):
                    inter = int((a & b).sum())
                    denom = int(a.sum()) + int(b.sum())
                    return 2.0 * inter / denom if denom else 0.0

                no_out = int((modality_only_seg & outside).sum())
                mask_out = int((mask_seg & outside).sum())
                retained = int((modality_only_seg & mask_seg & outside).sum()) / max(1, no_out)
                local_pair_metrics.append(
                    {
                        "step": step,
                        "save_id": save_id,
                        "target_modality": tgt,
                        "seg_step": int(st),
                        "modality_only_mask_dice": _dice(modality_only_seg, cond),
                        "mask_guided_mask_dice": _dice(mask_seg, cond),
                        "modality_only_outside_per_mask": no_out / cond_n,
                        "mask_guided_outside_per_mask": mask_out / cond_n,
                        "outside_suppression": (no_out - mask_out) / max(1, no_out),
                        "modality_only_outside_retained": retained,
                    }
                )
        local_prompts = list(records.values())
        for r in records.values():
            log.info(
                f"[infer] inspection saved val_idx={r['idx']} m{r['tgt_modality']} modes={list(r['prompts'])} seg@{seg_steps_insp}"
            )
    except Exception as e:
        log.info(f"[val] rank{rank} gen error: {type(e).__name__}: {e}")
        eval_errors.append(
            dict(stage="initialization_or_loop", error=f"{type(e).__name__}: {e}")
        )
    finally:
        for stats in mode_stats.values():
            stats["not_run"] = stats["requested"] - sum(
                (stats[k] for k in ("ok", "skipped", "failed", "ineligible"))
            )
        val_ds._force_task = None
        val_ds._inpaint_void_tumor = False
        val_ds._sr_native_input = False
        val_ds._force_missing_target_mod = None
        val_ds._force_seg_ref_count = None
        val_ds._force_seg_acc = None
        val_ds._force_seg_mods = None
        vae = None
        eval_model = None
        if eval_qwen is not None:
            del eval_qwen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    reports = [{"modes": mode_stats, "errors": eval_errors}]
    if distributed:
        reports = [None] * world
        dist.all_gather_object(reports, {"modes": mode_stats, "errors": eval_errors})
    mode_stats = {
        m: {k: sum((r["modes"][m][k] for r in reports)) for k in mode_stats[m]}
        for m in modes_to_eval
    }
    eval_errors = [
        dict(rank=i, **e) for i, report in enumerate(reports) for e in report["errors"]
    ]
    eval_complete = not eval_errors and all(
        (
            st["requested"] > 0 and st["ok"] == st["requested"]
            for st in mode_stats.values()
        )
    )

    def _write_validation_status():
        nonlocal eval_complete
        if is_main_process():
            try:
                (out_dir / "validation_status.json").write_text(
                    json.dumps(
                        {
                            "step": step,
                            "complete": eval_complete,
                            "modes": mode_stats,
                            "errors": eval_errors,
                        },
                        indent=2,
                    )
                )
            except OSError as exc:
                eval_complete = False
                log.error(f"[val] cannot persist validation status: {exc}")

    def _record_output_error(stage, exc):
        nonlocal eval_complete, fid
        eval_complete = False
        fid = None
        eval_errors.append(
            dict(rank=rank, stage=stage, error=f"{type(exc).__name__}: {exc}")
        )
        _write_validation_status()

    _write_validation_status()
    if is_main_process():
        log.info(f"[val] global counts complete={eval_complete}: {mode_stats}")
    modes = tuple(modes_to_eval)
    if distributed:
        gathered: list = [None] * world
        dist.all_gather_object(gathered, local_fid_by_mode)
        gen_by_mode = {
            m: [
                v.astype(np.float32)
                for part in gathered
                for v in (part or {}).get(m, [])
            ]
            if is_main_process()
            else []
            for m in modes
        }
        gathered_reals: list = [None] * world
        dist.all_gather_object(gathered_reals, local_reals_by_mode)
        reals_by_mode = {
            m: [
                v.astype(np.float32)
                for part in gathered_reals
                for v in (part or {}).get(m, [])
            ]
            if is_main_process()
            else []
            for m in modes
        }
        gathered_prompts: list = [None] * world
        dist.all_gather_object(gathered_prompts, local_prompts)
        all_prompts = [p for part in gathered_prompts for p in part]
        gathered_dice: list = [None] * world
        dist.all_gather_object(gathered_dice, local_dice)
        gathered_pairs: list = [None] * world
        dist.all_gather_object(gathered_pairs, local_pair_metrics)
        all_pair_metrics = [p for part in gathered_pairs for p in part or []]
        gathered_gen_seg: list = [None] * world
        dist.all_gather_object(gathered_gen_seg, local_generative_seg_metrics)
        all_generative_seg_metrics = [
            p for part in gathered_gen_seg for p in part or []
        ]
        dice_by_step: dict[int, list] = {}
        if is_main_process():
            for part in gathered_dice:
                for st, vals in (part or {}).items():
                    dice_by_step.setdefault(int(st), []).extend(vals)
    else:
        gen_by_mode = {
            m: [v.astype(np.float32) for v in local_fid_by_mode[m]] for m in modes
        }
        reals_by_mode = {
            m: [v.astype(np.float32) for v in local_reals_by_mode[m]] for m in modes
        }
        all_prompts = list(local_prompts)
        dice_by_step = {int(st): list(v) for st, v in local_dice.items()}
        all_pair_metrics = list(local_pair_metrics)
        all_generative_seg_metrics = list(local_generative_seg_metrics)
    if is_main_process() and all_prompts:
        mapping = {
            f"{p['save_id']}_m{p['tgt_modality']}_{p['mode']}": p
            for p in sorted(
                all_prompts, key=lambda p: p.get("file_prefix", p["save_id"])
            )
        }
        try:
            (out_dir / "prompts.json").write_text(
                json.dumps(mapping, indent=2, ensure_ascii=False)
            )
            log.info(
                f"[infer] wrote {out_dir / 'prompts.json'} ({len(mapping)} prompts)"
            )
        except Exception as _pe:
            log.info(f"[infer] prompts.json write failed: {_pe}")
            _record_output_error("manifest", _pe)
    if is_main_process() and dice_by_step:
        adh_csv = exp_dir / "mask_adherence.csv"
        new_adh = not adh_csv.exists()
        use_wb = bool(cfg.get("use_wandb", False)) and wandb is not None
        try:
            with open(adh_csv, "a") as cf:
                if new_adh:
                    cf.write("step,seg_step,dice_mean,n_samples\n")
                for st in sorted(dice_by_step):
                    vals = dice_by_step[st]
                    dm = float(np.mean(vals)) if vals else 0.0
                    cf.write(f"{step},{st},{dm:.6f},{len(vals)}\n")
                    log.info(
                        f"[adherence] step={step} seg_step={st} maskDice={dm:.4f} (n={len(vals)})"
                    )
                    if use_wb:
                        wandb.log({f"mask_adherence_dice_seg{st}": dm, "step": step})
        except OSError as exc:
            _record_output_error("metric_output", exc)
    if is_main_process() and all_pair_metrics:
        pair_csv = exp_dir / "mask_modality_only_comparison.csv"
        fields = [
            "step",
            "save_id",
            "target_modality",
            "seg_step",
            "modality_only_mask_dice",
            "mask_guided_mask_dice",
            "modality_only_outside_per_mask",
            "mask_guided_outside_per_mask",
            "outside_suppression",
            "modality_only_outside_retained",
        ]
        try:
            new_pair_csv = not pair_csv.exists()
            with open(pair_csv, "a") as pf:
                if new_pair_csv:
                    pf.write(",".join(fields) + "\n")
                for row in all_pair_metrics:
                    pf.write(",".join((str(row[k]) for k in fields)) + "\n")
            means = {
                k: float(np.mean([float(r[k]) for r in all_pair_metrics]))
                for k in fields[4:]
            }
            log.info(f"[mask-vs-modality-only] step={step} n={len(all_pair_metrics)} {means}")
            if bool(cfg.get("use_wandb", False)) and wandb is not None:
                wandb.log(
                    {f"mask_vs_modality_only/{k}": v for k, v in means.items()}
                    | {"step": step}
                )
        except OSError as exc:
            _record_output_error("metric_output", exc)
    if is_main_process() and all_generative_seg_metrics:
        seg_csv = exp_dir / "generative_seg_dice.csv"
        fields = [
            "step",
            "mode",
            "idx",
            "foreground",
            "class_1",
            "class_2",
            "class_3",
            "macro",
        ]
        try:
            new_seg_csv = not seg_csv.exists()
            with open(seg_csv, "a") as sf:
                if new_seg_csv:
                    sf.write(",".join(fields) + "\n")
                for row in all_generative_seg_metrics:
                    sf.write(",".join((str(row.get(key, "")) for key in fields)) + "\n")
            for mode in sorted({row["mode"] for row in all_generative_seg_metrics}):
                rows = [
                    row for row in all_generative_seg_metrics if row["mode"] == mode
                ]
                means = {
                    key: float(np.mean([float(row[key]) for row in rows]))
                    for key in ("foreground", "macro")
                }
                log.info(
                    f"[generative-seg] step={step} mode={mode} n={len(rows)} {means}"
                )
                if bool(cfg.get("use_wandb", False)) and wandb is not None:
                    wandb.log(
                        {f"generative_seg/{mode}_{k}": v for k, v in means.items()}
                        | {"step": step}
                    )
        except OSError as exc:
            _record_output_error("metric_output", exc)
    if is_main_process() and fid_enable and any((gen_by_mode.get(m) for m in modes)):
        try:
            from utils.slice_fid import SliceFID

            sfid = SliceFID(
                device,
                slice_axis=int(infer.get("fid_slice_axis", 2)),
                slice_frac=tuple(infer.get("fid_slice_frac", (0.2, 0.85))),
                weights_path=infer.get("fid_weights"),
            )
            fid_csv = exp_dir / "fid.csv"
            new_csv = not fid_csv.exists()
            use_wb = bool(cfg.get("use_wandb", False)) and wandb is not None
            with open(fid_csv, "a") as cf:
                if new_csv:
                    cf.write("step,task,fid,n_gen,n_real\n")
                for mode in modes:
                    vols = gen_by_mode.get(mode, [])
                    if not vols:
                        continue
                    rvols = reals_by_mode.get(mode, [])
                    if not rvols:
                        log.info(
                            f"[fid] step={step} task={mode} skipped (no reals collected)"
                        )
                        continue
                    m_mu, m_sigma, _ = sfid.stats_from_volumes(rvols)
                    n_real = len(rvols)
                    fm = sfid.fid_against_real(vols, m_mu, m_sigma)
                    if not math.isfinite(fm):
                        raise ValueError(f"non-finite FID for mode={mode}: {fm}")
                    if mode == primary_fid_mode and eval_complete:
                        fid = fm
                    cf.write(f"{step},{mode},{fm:.6f},{len(vols)},{n_real}\n")
                    log.info(
                        f"[fid] step={step} task={mode} slice_FID={fm:.4f} (n_gen={len(vols)}, n_real={n_real})"
                    )
                    if use_wb:
                        wandb.log({f"slice_fid_{mode}": fm, "step": step})
        except Exception as e:
            log.info(f"[fid] skipped (error: {type(e).__name__}: {e})")
            _record_output_error("fid", e)
    barrier()
    return fid if eval_complete else None


def main():
    global DEBUG_LOGGING
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument(
        "--curriculum-preset",
        choices=("foundation", "full"),
        default=None,
        help="Manual unified-training preset; defaults to curriculum.default_preset.",
    )
    parser.add_argument("--resume", choices=["", "latest"], default=None)
    parser.add_argument("--resume-exp", dest="resume_exp", type=str, default=None)
    parser.add_argument("--resume-step", dest="resume_step", type=str, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Weight-only initialization (step 0, fresh optimizer); supports legacy 5-row modality embeddings.",
    )
    parser.add_argument(
        "--flash-attention",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override mmdit.attn_impl: enabled=flash_varlen, disabled=sdpa.",
    )
    args = parser.parse_args()
    cfg = apply_curriculum_preset(
        load_train_config(args.config), preset=args.curriculum_preset
    )
    if args.resume is not None:
        cfg["resume"] = args.resume
    if args.resume_exp is not None:
        cfg["resume_exp"] = args.resume_exp
    if args.resume_step is not None:
        cfg["resume_step"] = args.resume_step
    if args.flash_attention is not None:
        cfg.setdefault("mmdit", {})["attn_impl"] = (
            "flash_varlen" if args.flash_attention else "sdpa"
        )
    DEBUG_LOGGING = bool(cfg.get("debug_logging", False))
    if DEBUG_LOGGING:
        log.info(
            f"[startup] pid={os.getpid()} config={args.config} RANK={os.environ.get('RANK', '<unset>')} LOCAL_RANK={os.environ.get('LOCAL_RANK', '<unset>')} WORLD_SIZE={os.environ.get('WORLD_SIZE', '<unset>')} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
        log.info(f"[startup] pid={os.getpid()} config loaded")
    if DEBUG_LOGGING:
        log.info(f"[startup] pid={os.getpid()} init_distributed begin")
    rank, world_size, local_rank, device = init_distributed()
    log_rank(
        f"init_distributed done local_rank={local_rank} world_size={world_size} device={device} cuda_count={torch.cuda.device_count()}",
        rank,
        device,
    )
    set_seed(int(cfg.get("seed", 42)) + rank)
    try:
        torch.backends.cuda.enable_cudnn_sdp(True)
    except Exception as e:
        log_rank(f"enable_cudnn_sdp unavailable: {e}", rank, device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    use_wandb = bool(cfg.get("use_wandb", False))
    if use_wandb and wandb is None:
        raise RuntimeError("use_wandb=true requires the wandb package.")
    if is_main_process():
        print("=" * 60, flush=True)
        print("MMDiT text-to-MRI training", flush=True)
        print("=" * 60, flush=True)
        for k in sorted(cfg.keys()):
            if not k.startswith("__"):
                print(f"  {k:<28s}: {cfg[k]}", flush=True)
        print("=" * 60, flush=True)
        if use_wandb:
            wb_tags = cfg.get("wandb_tags") or None
            if isinstance(wb_tags, str):
                wb_tags = [t.strip() for t in wb_tags.split(",") if t.strip()]
            wandb.init(
                project=cfg.get("project_name", "UniGlioma"),
                name=cfg.get("run_name", "uniglioma"),
                group=cfg.get("wandb_group") or None,
                job_type=cfg.get("wandb_job_type") or None,
                tags=wb_tags,
                config=cfg,
                mode=str(cfg.get("wandb_mode", "online")),
            )
            wandb.define_metric("step")
            wandb.define_metric("loss", step_metric="step")
            wandb.define_metric("lr", step_metric="step")
            wandb.define_metric("slice_fid", step_metric="step")
    else:
        os.environ["WANDB_SILENT"] = "true"
    t_startup = time.perf_counter()
    log_rank("building MMDiT config", rank, device)
    mmdit_cfg = build_mmdit_cfg(cfg)
    t0 = time.perf_counter()
    log_rank("building MMDiT begin", rank, device)
    raw_model = MMDiT3D(mmdit_cfg).to(device)
    init_ema_state = None
    init_path = args.init_checkpoint or cfg.get("init_checkpoint")
    if init_path and str(cfg.get("resume", "")).strip() in ("auto", "latest"):
        raise ValueError(
            "--init-checkpoint and resume=auto/latest are mutually exclusive"
        )
    if init_path:
        payload = torch.load(str(init_path), map_location="cpu", weights_only=False)
        source_state = payload.get("model", payload)
        source_state = _normalize_eval_state_dict(source_state)
        source_state, migrated = adapt_unified_modality_embedding(
            source_state, raw_model.state_dict()
        )
        raw_model.load_state_dict(source_state, strict=True)
        ema_source = payload.get("ema")
        ema_path = Path(str(init_path)).with_name(Path(str(init_path)).stem + "_ema.pt")
        if "ema" not in payload and ema_path.is_file():
            ema_payload = torch.load(ema_path, map_location="cpu", weights_only=False)
            if ema_payload.get("step") != payload.get("step"):
                raise RuntimeError(
                    "initialization companion EMA step does not match model"
                )
            ema_source = ema_payload.get("model", ema_payload)
        if ema_source is not None:
            init_ema_state, ema_migrated = adapt_unified_modality_embedding(
                _normalize_eval_state_dict(ema_source), raw_model.state_dict()
            )
            migrated = migrated or ema_migrated
        if is_main_process():
            log.info(
                f"[init] loaded weight-only checkpoint {init_path}; step=0 optimizer=fresh modality_embedding_migrated={migrated} ema={('loaded' if init_ema_state else 'seeded_from_model')}"
            )
    log_rank(f"building MMDiT done {time.perf_counter() - t0:.2f}s", rank, device)
    use_fsdp = str(cfg.get("parallel", "ddp")).lower() == "fsdp" or bool(
        cfg.get("use_fsdp", False)
    )
    freeze_backbone = bool(cfg.get("freeze_backbone", False))
    stage2_extra_trainable = tuple(cfg.get("stage2_extra_trainable", ()))
    if freeze_backbone:
        n_tr = apply_freeze(
            raw_model, freeze_backbone=True, extra_trainable=stage2_extra_trainable
        )
        if is_main_process():
            log.info(
                f"[train][freeze] backbone frozen; trainable params (tensors)={n_tr} extra_trainable={stage2_extra_trainable}"
            )
    if is_main_process():
        n_total = sum((p.numel() for p in raw_model.parameters()))
        n_train = sum((p.numel() for p in raw_model.parameters() if p.requires_grad))
        n_double = sum((p.numel() for p in raw_model.double_blocks.parameters()))
        n_single = sum((p.numel() for p in raw_model.single_blocks.parameters()))
        eff_batch = int(cfg["batch_size"]) * world_size
        log.info(
            "=" * 60
            + f"\nMMDiT params: total {n_total / 1000000.0:.1f}M ({n_total / 1000000000.0:.3f}B) | trainable {n_train / 1000000.0:.1f}M\n  double_blocks ({mmdit_cfg.depth_double}x) {n_double / 1000000.0:.1f}M | single_blocks ({mmdit_cfg.depth_single}x) {n_single / 1000000.0:.1f}M | other {(n_total - n_double - n_single) / 1000000.0:.1f}M\n  width {mmdit_cfg.width} heads {mmdit_cfg.heads} head_dim {mmdit_cfg.width // mmdit_cfg.heads} mlp {mmdit_cfg.mlp_variant} qk_norm {mmdit_cfg.use_qk_norm} attn {mmdit_cfg.attn_impl} grad_ckpt {mmdit_cfg.grad_checkpoint} arch_v{mmdit_cfg.arch_version} parallel {('fsdp' if use_fsdp else 'ddp')}\n  batch_size {cfg['batch_size']}/gpu x {world_size} gpu = {eff_batch} effective\n"
            + "=" * 60
        )
    compile_dit = bool(cfg.get("compile_mmdit", True))
    compile_mode = cfg.get("compile_mode", None) or None
    compile_kwargs = {"mode": compile_mode} if compile_mode else {}
    if compile_dit and compile_mode and is_main_process():
        log.info(
            f"torch.compile(mode={compile_mode}) — first steps slow while it autotunes"
        )
    t0 = time.perf_counter()
    if use_fsdp:
        fsdp_cfg = fsdp_config_from_dict(cfg.get("fsdp", {}))
        log_rank(f"FSDP wrap begin sharding={fsdp_cfg.sharding}", rank, device)
        train_model = wrap_fsdp_mmdit(raw_model, fsdp_cfg, device=device)
        log_rank(f"FSDP wrap done {time.perf_counter() - t0:.2f}s", rank, device)
    else:
        if compile_dit:
            tc = time.perf_counter()
            log_rank("torch.compile(MMDiT) begin", rank, device)
            train_model = torch.compile(raw_model, **compile_kwargs)
            _silence_compile_logs()
            log_rank(
                f"torch.compile(MMDiT) done {time.perf_counter() - tc:.2f}s",
                rank,
                device,
            )
        else:
            train_model = raw_model
        log_rank("DDP wrap begin", rank, device)
        ddp_static = bool(cfg.get("ddp_static_graph", False))
        train_model = DDP(
            train_model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False
            if ddp_static
            else bool(cfg.get("ddp_find_unused_parameters", True)),
            static_graph=ddp_static,
        )
        log_rank(
            f"DDP: static_graph={ddp_static} find_unused={not ddp_static and bool(cfg.get('ddp_find_unused_parameters', True))}",
            rank,
            device,
        )
        log_rank(f"DDP wrap done {time.perf_counter() - t0:.2f}s", rank, device)
    log_rank("building optimizer", rank, device)
    opt_target = train_model if use_fsdp else raw_model
    param_groups = build_param_groups(
        opt_target,
        base_wd=float(cfg.get("weight_decay", 0.01)),
        embedder_lr_scale=float(cfg.get("embedder_lr_scale", 1.0)),
        backbone_lr_scale=float(cfg.get("backbone_lr_scale", 1.0)),
    )
    optim = torch.optim.AdamW(
        param_groups,
        lr=float(cfg["lr"]),
        betas=tuple(cfg.get("betas", (0.9, 0.95))),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        fused=torch.cuda.is_available(),
    )
    use_ema = bool(cfg.get("use_ema", True))
    use_cpu_ema = bool(cfg.get("ema_cpu", cfg.get("cpu_ema", False)))
    ema_device = "cpu" if use_cpu_ema else None
    ema_dtype = torch.float32 if use_cpu_ema else None
    ema_track_all = bool(cfg.get("ema_track_all", freeze_backbone))
    if use_ema and use_fsdp:
        with FSDP.summon_full_params(train_model, writeback=False):
            ema = EMA(
                raw_model,
                decay=float(cfg.get("ema_decay", 0.999)),
                device=ema_device,
                dtype=ema_dtype,
                track_all=ema_track_all,
            )
    elif use_ema:
        ema = EMA(
            raw_model,
            decay=float(cfg.get("ema_decay", 0.999)),
            device=ema_device,
            dtype=ema_dtype,
            track_all=ema_track_all,
        )
    else:
        ema = None
    if ema is not None and init_ema_state is not None:
        ema.load_state_dict(init_ema_state)
    ema_start_step = int(cfg.get("ema_start_step", 0))
    exp_dir = resolve_exp_dir(cfg, args.config)
    dump_resolved_config(exp_dir, cfg, mmdit_cfg)
    step = 0
    _rmode = str(cfg.get("resume", "")).strip()
    _ckdir = exp_dir / "ckpts"
    _has_ckpt = has_dit_checkpoint(_ckdir)
    if _rmode == "latest" or (_rmode == "auto" and _has_ckpt):
        log_rank(f"resume requested from {exp_dir}", rank, device)
        step = maybe_resume(
            raw_model,
            optim,
            exp_dir,
            cfg,
            ema=ema,
            train_model=train_model,
            use_fsdp=use_fsdp,
        )
        if is_main_process():
            log.info(f"[resume] starting at step {step}")
    if use_fsdp and compile_dit:
        tc = time.perf_counter()
        log_rank("torch.compile(MMDiT) [over FSDP, post-resume] begin", rank, device)
        train_model = torch.compile(train_model, **compile_kwargs)
        _silence_compile_logs()
        log_rank(
            f"torch.compile(MMDiT) done {time.perf_counter() - tc:.2f}s", rank, device
        )
    seg_start_step = int((cfg["mmdit"].get("seg_head", {}) or {}).get("start_step", 0))
    active_tasks = {
        str(task)
        for task, weight in (cfg.get("task_weights") or {}).items()
        if float(weight) > 0
    }
    labels_required_by_task = bool(active_tasks & LABEL_REQUIRED_TASKS)
    aux_seg_supervision = bool(cfg.get("aux_seg_supervision", True))
    seg_head_config_enabled = bool(
        (cfg["mmdit"].get("seg_head", {}) or {}).get("enabled", False)
    )
    labels_on = labels_required_by_task or (
        aux_seg_supervision and seg_head_config_enabled and (step >= seg_start_step)
    )
    t0 = time.perf_counter()
    log_rank("building train DataLoader begin", rank, device)
    train_loader = build_train_loader(
        cfg, world_size=world_size, rank=rank, load_labels=labels_on
    )
    log_rank(
        f"building train DataLoader done {time.perf_counter() - t0:.2f}s dataset={type(train_loader.dataset).__name__} len={len(train_loader)} samples={len(train_loader.dataset)} workers={cfg.get('num_workers', 4)} prefetch={cfg.get('prefetch_factor', 2)}",
        rank,
        device,
    )
    train_iter = iter(train_loader)

    def next_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    latent_sample = bool(cfg.get("latent_sample", False))
    _tw = cfg.get("task_weights") or {}
    _online_ref_tasks = {
        str(task)
        for task, weight in _tw.items()
        if float(weight) > 0 and task in ONLINE_VAE_TASKS
    }
    needs_online_vae = bool(_online_ref_tasks)
    online_vae = None
    if needs_online_vae:
        log_rank(
            f"building resident VAE for online ref encode (tasks={sorted(_online_ref_tasks)})",
            rank,
            device,
        )
        online_vae = build_vae(cfg).to(device).eval()
    if is_main_process():
        log.info(
            f"[train] cache-only training: latent_sample={latent_sample} (text encoder never run; VAE {('RESIDENT for online ref encode' if online_vae else 'not')} in loop)"
        )
    cfg_train_states = cfg.get("cfg_train_states") or None
    exposure_tasks = list((cfg.get("task_weights") or {}).keys())
    exposure_task_to_id = {str(name): i for i, name in enumerate(exposure_tasks)}
    exposure_counts = torch.zeros(
        len(exposure_tasks), len(_CFG_STATE_ORDER), dtype=torch.long, device=device
    )
    inpaint_bin_names = tuple(
        (
            str(x)
            for x in cfg.get("inpaint_size_bin_names", ("small", "medium", "large"))
        )
    )
    inpaint_sampling_counts = torch.zeros(
        5 + len(inpaint_bin_names), dtype=torch.float64, device=device
    )
    conditioned_seg_counts = torch.zeros(2, dtype=torch.long, device=device)
    p_text = float(cfg.get("p_uncond_text", 0.1))
    p_mod = float(cfg.get("p_uncond_modality", 0.1))
    p_ref = float(cfg.get("p_uncond_ref", 0.1))
    p_label_cond = float(cfg.get("p_label_cond", 0.5))
    p_label_cond_schedule = cfg.get("p_label_cond_schedule") or None
    if p_label_cond_schedule and is_main_process():
        log.info(
            f"[train] p_label_cond schedule: base={p_label_cond} {p_label_cond_schedule}"
        )
    if cfg_train_states and is_main_process():
        log.info(f"[train][cfg] task-aware states enabled: {cfg_train_states}")
    if "inpaint" in _online_ref_tasks and is_main_process():
        bin_edges = tuple(
            (float(x) for x in cfg.get("inpaint_size_bin_edges", (0.01, 0.025)))
        )
        base_weights = tuple(
            (float(x) for x in cfg.get("inpaint_size_bin_weights", (1, 1, 1)))
        )
        large_boost = float(cfg.get("inpaint_large_oversample", 2.0))
        effective_weights = list(base_weights)
        effective_weights[-1] *= large_boost
        log.info(
            f"[train][inpaint-sampling] bins={inpaint_bin_names} edges(hole/brain)={bin_edges} effective_weights={tuple(effective_weights)} case_tries={int(cfg.get('inpaint_case_max_tries', 64))} place_tries={int(cfg.get('inpaint_place_max_tries', 48))} contralateral_large={bool(cfg.get('inpaint_contralateral_large', True))}"
        )
    spatial_enabled = bool(mmdit_cfg.use_ref_slab)
    if spatial_enabled and is_main_process():
        log.info(
            "[train][mask-ref] tumor-mask (type1) / inpaint-region (type2) ride VAE-encoded ref slots (ref_type_emb)."
        )
    _seg = cfg["mmdit"].get("seg_head", {}) or {}
    seg_enabled = (
        bool(_seg.get("enabled", False))
        and aux_seg_supervision
        and mmdit_cfg.seg_head_enabled
    )
    seg_lambda = float(_seg.get("lambda", 0.2))
    conditioned_mask_seg_prob = float(
        _seg.get("conditioned_mask_supervision_prob", 0.0)
    )
    conditioned_mask_seg_lambda = float(
        _seg.get("conditioned_mask_supervision_lambda", seg_lambda)
    )
    if not 0.0 <= conditioned_mask_seg_prob <= 1.0:
        raise ValueError(
            f"seg_head.conditioned_mask_supervision_prob must be in [0,1]; got {conditioned_mask_seg_prob}"
        )
    seg_t_weight = str(_seg.get("t_weight", "linear")).lower()
    seg_num_classes = int(_seg.get("num_classes", 4))
    flow_region_weights = {
        str(k): float(v) for k, v in (cfg.get("flow_region_weights") or {}).items()
    }
    mask_guide_region_weight = float(
        flow_region_weights.get("mask_guide", _seg.get("region_weight", 0.0))
    )
    seg_class_weights = None
    if _seg.get("class_weights") is not None:
        seg_class_weights = torch.tensor(
            [float(w) for w in _seg["class_weights"]],
            device=device,
            dtype=torch.float32,
        )
        if seg_class_weights.numel() != seg_num_classes:
            raise ValueError(
                f"seg_head.class_weights has {seg_class_weights.numel()} entries != num_classes={seg_num_classes}"
            )
    if seg_enabled and is_main_process():
        log.info(
            f"[train][seg] enabled tap_layer={mmdit_cfg.seg_tap_layer} classes={seg_num_classes} lambda={seg_lambda} t_weight={seg_t_weight} class_weights={(None if seg_class_weights is None else seg_class_weights.tolist())} full_shape={mmdit_cfg.seg_full_shape} region_weight={mask_guide_region_weight} supervision=all_labelled_targets conditioned_mask_prob={conditioned_mask_seg_prob} conditioned_mask_lambda={conditioned_mask_seg_lambda}"
        )
    grad_clip = float(cfg.get("grad_clip", 1.0))
    total_steps = int(cfg["total_steps"])
    base_lr = float(cfg["lr"])
    warmup_steps = int(cfg.get("warmup_steps", 0))
    lr_schedule = str(cfg.get("lr_schedule", "cosine"))
    lr_min = float(cfg.get("lr_min", base_lr * 0.05))
    lr_decay_steps = int(cfg.get("lr_decay_steps", total_steps))
    log_every = int(cfg.get("log_every", 50))
    ckpt_every = int(cfg.get("ckpt_every", 1000))
    ckpt_latest_every = int(cfg.get("ckpt_latest_every", 0))
    infer_every = int(cfg.get("infer_every", 0))
    keep_last_ckpts = int(cfg.get("keep_last_checkpoints", 0))
    startup_trace_steps = int(cfg.get("startup_trace_steps", 2)) if DEBUG_LOGGING else 0
    epoch_size = max(len(train_loader), 1)
    if is_main_process():
        log.info(
            f"[train] batch_size={cfg['batch_size']} world_size={world_size} effective_batch={int(cfg['batch_size']) * world_size} dataset={type(train_loader.dataset).__name__} samples={len(train_loader.dataset)} steps_per_epoch={len(train_loader)}"
        )
    log_rank(f"startup complete {time.perf_counter() - t_startup:.2f}s", rank, device)
    eff_batch = int(cfg["batch_size"]) * world_size
    win_t0 = time.perf_counter()
    win_step0 = step
    iters_since_start = 0
    early_speed_t0 = None
    early_speed_logged = False
    primary_fid_mode = str((cfg.get("infer", {}) or {}).get("best_fid_mode", "modality_only"))
    best_fid = float("inf")
    _best_ck = next((exp_dir / "ckpts" / name for name in
                     ("dit_ckpt_step_best.pt", "ckpt_step_best.pt")
                     if (exp_dir / "ckpts" / name).is_file()),
                    exp_dir / "ckpts" / "dit_ckpt_step_best.pt")
    if _best_ck.exists():
        try:
            best_fid = float(
                torch.load(_best_ck, map_location="cpu", weights_only=False).get(
                    "fid", float("inf")
                )
            )
            log_rank(f"prior best slice_FID={best_fid:.3f}", rank, device)
        except Exception:
            pass
    compile_status_logged = False
    train_model.train()
    while step < total_steps:
        if not labels_on and seg_enabled and (step >= seg_start_step):
            labels_on = True
            log_rank(
                f"[seg] step {step}: reached seg_head.start_step={seg_start_step} → enabling label loading (mask ref + seg target) + seg head; rebuilding loader",
                rank,
                device,
            )
            train_loader = build_train_loader(
                cfg, world_size=world_size, rank=rank, load_labels=True
            )
            train_iter = iter(train_loader)
        if step % epoch_size == 0 and hasattr(train_loader.sampler, "set_epoch"):
            log_rank(f"set_epoch {step // epoch_size}", rank, device)
            train_loader.sampler.set_epoch(step // epoch_size)
        trace = step < startup_trace_steps
        step_t0 = time.perf_counter()
        if trace:
            log_rank(f"step {step} begin", rank, device)
            log_rank(f"step {step}: next_batch begin", rank, device)
        batch = next_batch()
        if trace:
            log_rank(
                f"step {step}: next_batch done {time.perf_counter() - step_t0:.2f}s latent_shape={tuple(batch['latent_mean'].shape)} n_text={batch['text_emb'].shape[0]}",
                rank,
                device,
            )
        t0 = time.perf_counter()
        z = sample_target_latents(batch, device, latent_sample)
        target_needs = batch.get("target_needs_encode")
        if target_needs is not None and bool(target_needs.any()):
            if online_vae is None or "ref_image" not in batch:
                raise RuntimeError(
                    "synthetic SR targets require the online VAE and ref_image"
                )
            rows_cpu = torch.nonzero(target_needs.bool(), as_tuple=False).flatten()
            slots_cpu = batch["target_encode_slot"][rows_cpu].long()
            if bool((slots_cpu < 0).any()):
                raise RuntimeError(
                    "target_needs_encode row has no valid target_encode_slot"
                )
            target_imgs = batch["ref_image"][rows_cpu, slots_cpu].unsqueeze(1)
            encoded_targets = encode_ref_images(
                online_vae, target_imgs, mmdit_cfg, device
            )
            z = z.clone()
            z[rows_cpu.to(device)] = encoded_targets
        mod_id = batch["modality_id"].to(device, non_blocking=True).long()
        text_emb = batch["text_emb"].to(device, dtype=torch.bfloat16, non_blocking=True)
        text_mask = batch["text_mask"].to(device, non_blocking=True).long()
        B = z.shape[0]
        tasks = [str(name) for name in batch.get("task", ["modality_only"] * B)]
        grid = tuple(mmdit_cfg.latent_shape)
        if trace:
            log_rank(
                f"step {step}: cache load done {time.perf_counter() - t0:.2f}s z={tuple(z.shape)}",
                rank,
                device,
            )
        has_text = batch.get("has_text")
        has_label = batch.get("has_label")
        has_text = (
            has_text.to(device)
            if has_text is not None
            else torch.ones(B, dtype=torch.bool, device=device)
        )
        has_label = (
            has_label.to(device)
            if has_label is not None
            else torch.zeros(B, dtype=torch.bool, device=device)
        )
        inpaint_bins = batch.get("inpaint_size_bin")
        if inpaint_bins is not None:
            inpaint_bins = inpaint_bins.to(device, non_blocking=True).long()
            inpaint_rows = inpaint_bins >= 0
            if bool(inpaint_rows.any()):
                hole_voxels = (
                    batch["inpaint_hole_voxels"].to(device, non_blocking=True).double()
                )
                hole_fraction = (
                    batch["inpaint_hole_fraction"]
                    .to(device, non_blocking=True)
                    .double()
                )
                placement_attempts = (
                    batch["inpaint_placement_attempts"]
                    .to(device, non_blocking=True)
                    .double()
                )
                placement_failures = (
                    batch["inpaint_placement_failures"]
                    .to(device, non_blocking=True)
                    .double()
                )
                inpaint_sampling_counts[0] += inpaint_rows.sum()
                inpaint_sampling_counts[1] += hole_voxels[inpaint_rows].sum()
                inpaint_sampling_counts[2] += hole_fraction[inpaint_rows].sum()
                inpaint_sampling_counts[3] += placement_attempts[inpaint_rows].sum()
                inpaint_sampling_counts[4] += placement_failures[inpaint_rows].sum()
                inpaint_sampling_counts[5:] += torch.bincount(
                    inpaint_bins[inpaint_rows], minlength=len(inpaint_bin_names)
                ).double()
        t = sample_train_timesteps(B, device, cfg)
        tb = t.view(B, 1, 1, 1, 1)
        eps = torch.randn_like(z)
        x_t = (1.0 - tb) * z + tb * eps
        v_target = eps - z
        cfg_keep_refs = None
        if cfg_train_states:
            cfg_keep_text, cfg_keep_refs, _cfg_state_ids = sample_cfg_train_states(
                tasks, cfg_train_states, device
            )
            text_drop = ~has_text | ~cfg_keep_text
            with torch.no_grad():
                for task_name in dict.fromkeys(tasks):
                    task_i = exposure_task_to_id.get(task_name)
                    if task_i is None:
                        continue
                    sample_idx = torch.tensor(
                        [i for i, name in enumerate(tasks) if name == task_name],
                        device=device,
                    )
                    exposure_counts[task_i] += torch.bincount(
                        _cfg_state_ids[sample_idx], minlength=len(_CFG_STATE_ORDER)
                    )
        else:
            text_drop = ~has_text | (torch.rand(B, device=device) < p_text)
            mod_drop = torch.rand(B, device=device) < p_mod
            mod_id = torch.where(
                mod_drop, torch.full_like(mod_id, mmdit_cfg.num_modalities), mod_id
            )
        seg_supervise = all_labelled_seg_supervision(has_label, tasks)
        conditioned_mask_seg_supervise = torch.zeros(B, dtype=torch.bool, device=device)
        conditioned_mask_seg_step = False
        region_w = None
        R = int(mmdit_cfg.max_refs)
        ref_latent = ref_modality_id = ref_dt = ref_valid = ref_type_id = None
        if spatial_enabled:
            ref_latent = torch.zeros(B, R, mmdit_cfg.in_channels, *grid, device=device)
            ref_modality_id = torch.zeros(B, R, dtype=torch.long, device=device)
            ref_dt = torch.zeros(B, R, device=device)
            ref_valid = torch.zeros(B, R, dtype=torch.bool, device=device)
            ref_type_id = torch.zeros(B, R, dtype=torch.long, device=device)
        if spatial_enabled and R > 0 and ("ref_latent" in batch):
            ref_latent = batch["ref_latent"].to(device, non_blocking=True).float()
            ref_modality_id = (
                batch["ref_modality_id"].to(device, non_blocking=True).long()
            )
            ref_dt = batch["ref_dt"].to(device, non_blocking=True).float()
            ref_valid = batch["ref_valid"].to(device, non_blocking=True).bool()
            rti = batch.get("ref_type_id")
            ref_type_id = (
                rti.to(device, non_blocking=True).long()
                if rti is not None
                else torch.zeros(B, R, dtype=torch.long, device=device)
            )
            needs = batch.get("ref_needs_encode")
            if (
                online_vae is not None
                and needs is not None
                and bool(needs.any())
                and ("ref_image" in batch)
            ):
                needs = needs.to(device).bool()
                ref_img = batch["ref_image"].to(device, non_blocking=True)
                for s in range(R):
                    if not bool(needs[:, s].any()):
                        continue
                    enc = encode_ref_images(
                        online_vae, ref_img[:, s].unsqueeze(1), mmdit_cfg, device
                    )
                    ref_latent[:, s] = torch.where(
                        needs[:, s].view(-1, 1, 1, 1, 1), enc, ref_latent[:, s]
                    )
        if spatial_enabled:
            if cfg_train_states:
                ref_valid = ref_valid & cfg_keep_refs.view(B, 1)
            else:
                p_label_cond_now = p_label_cond_at(
                    step, p_label_cond, p_label_cond_schedule
                )
                is_mask = (ref_type_id == 1) & ref_valid
                has_mask_ref = is_mask.any(dim=1)
                keep_mask = torch.rand(B, device=device) < p_label_cond_now
                withhold = has_mask_ref & ~keep_mask
                ref_valid = ref_valid & ~(is_mask & withhold.view(B, 1))
                ref_drop = torch.rand(B, device=device) < p_ref
                ref_valid = ref_valid & ~ref_drop.view(B, 1)
            visible_ref = ref_valid.any(dim=1)
            visible_mask = ((ref_type_id == 1) & ref_valid).any(dim=1)
            if (
                seg_enabled
                and step >= seg_start_step
                and (conditioned_mask_seg_prob > 0.0)
            ):
                conditioned_eligible = has_label & visible_mask
                conditioned_mask_seg_step = (
                    random.Random(
                        int(cfg.get("seed", 42)) + int(step) * 1000003 + 97
                    ).random()
                    < conditioned_mask_seg_prob
                )
                if conditioned_mask_seg_step:
                    conditioned_mask_seg_supervise = conditioned_eligible
                conditioned_seg_counts[0] += conditioned_mask_seg_supervise.sum()
                conditioned_seg_counts[1] += conditioned_eligible.sum()
            lf = batch.get("label_full")
            if lf is not None and lf.numel() > 0:
                fg = F.adaptive_max_pool3d(
                    (lf.to(device, non_blocking=True) > 0).float().unsqueeze(1), grid
                )
                mask_rows = (
                    torch.tensor(
                        [name == "mask_guide" for name in tasks], device=device
                    )
                    & visible_mask
                )
                seg_rows = torch.tensor(
                    [name == "seg" for name in tasks], device=device
                )
                for rows, weight in (
                    (mask_rows, mask_guide_region_weight),
                    (seg_rows, float(flow_region_weights.get("seg", 0.0))),
                ):
                    if weight > 0.0 and bool(rows.any()):
                        term = weight * fg * rows.view(B, 1, 1, 1, 1).to(fg.dtype)
                        region_w = (region_w if region_w is not None else 1.0) + term
            rm = batch.get("region_mask")
            if rm is not None:
                hole = rm.to(device, non_blocking=True).float()
                pseudo_rows = batch.get("pseudo_healthy")
                pseudo_rows = (
                    torch.zeros(B, dtype=torch.bool, device=device)
                    if pseudo_rows is None
                    else pseudo_rows.to(device=device, dtype=torch.bool)
                )
                for task_name in ("inpaint", "whole_brain"):
                    weight = float(flow_region_weights.get(task_name, 0.0))
                    if weight <= 0.0:
                        continue
                    rows = (
                        torch.tensor(
                            [name == task_name for name in tasks], device=device
                        )
                        & visible_ref
                        & ~pseudo_rows
                    )
                    if bool(rows.any()):
                        term = weight * hole * rows.view(B, 1, 1, 1, 1).to(hole.dtype)
                        region_w = (region_w if region_w is not None else 1.0) + term
                if bool(pseudo_rows.any()):
                    fw = 1.0 - hole
                    fw = torch.where(
                        pseudo_rows.view(B, 1, 1, 1, 1), fw, torch.ones_like(fw)
                    )
                    region_w = fw if region_w is None else region_w * fw
        seg_active = seg_enabled and step >= seg_start_step
        fwd_kwargs = dict(
            text_drop_mask=text_drop,
            ref_latent=ref_latent,
            ref_modality_id=ref_modality_id,
            ref_dt=ref_dt,
            ref_valid=ref_valid,
            ref_type_id=ref_type_id,
        )
        if trace:
            log_rank("step {step}: forward begin".format(step=step), rank, device)
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if seg_active:
                if conditioned_mask_seg_step:
                    v_pred, seg_logits, frozen_seg_logits = train_model(
                        x_t,
                        t,
                        text_emb,
                        text_mask,
                        mod_id,
                        return_seg=True,
                        return_frozen_seg=True,
                        **fwd_kwargs,
                    )
                else:
                    v_pred, seg_logits = train_model(
                        x_t,
                        t,
                        text_emb,
                        text_mask,
                        mod_id,
                        return_seg=True,
                        **fwd_kwargs,
                    )
                    frozen_seg_logits = None
            else:
                v_pred = train_model(x_t, t, text_emb, text_mask, mod_id, **fwd_kwargs)
            if region_w is not None:
                sq = (v_pred.float() - v_target.float()) ** 2
                w = region_w.float()
                flow_loss = (w * sq).sum() / (w.sum() * sq.shape[1]).clamp_min(1.0)
            else:
                flow_loss = F.mse_loss(v_pred.float(), v_target.float())
            if seg_active:
                seg_label = batch["label_full"].to(device, non_blocking=True).long()
                seg_loss, seg_parts = seg_aux_loss(
                    seg_logits,
                    seg_label,
                    t,
                    num_classes=seg_num_classes,
                    class_weights=seg_class_weights,
                    t_weight=seg_t_weight,
                    mask=seg_supervise,
                )
                if frozen_seg_logits is not None:
                    conditioned_seg_loss, conditioned_seg_parts = seg_aux_loss(
                        frozen_seg_logits,
                        seg_label,
                        t,
                        num_classes=seg_num_classes,
                        class_weights=seg_class_weights,
                        t_weight=seg_t_weight,
                        mask=conditioned_mask_seg_supervise,
                    )
                else:
                    conditioned_seg_loss = flow_loss.new_zeros(())
                    conditioned_seg_parts = {
                        "seg_ce": flow_loss.new_zeros(()),
                        "seg_dice_loss": flow_loss.new_zeros(()),
                        "fg_dice": flow_loss.new_zeros(()),
                        "seg_n": flow_loss.new_zeros(()),
                    }
                loss = (
                    flow_loss
                    + seg_lambda * seg_loss
                    + conditioned_mask_seg_lambda * conditioned_seg_loss
                )
            else:
                loss = flow_loss
        if trace:
            log_rank(
                f"step {step}: forward done {time.perf_counter() - t0:.2f}s loss={loss.item():.6f}",
                rank,
                device,
            )
        if trace:
            log_rank(f"step {step}: backward begin", rank, device)
        t0 = time.perf_counter()
        loss.backward()
        if trace:
            log_rank(
                f"step {step}: backward done {time.perf_counter() - t0:.2f}s",
                rank,
                device,
            )
        if trace:
            log_rank(f"step {step}: optimizer begin", rank, device)
        t0 = time.perf_counter()
        lr_now = lr_at(
            step,
            base_lr,
            warmup_steps,
            total_steps,
            lr_min,
            lr_schedule,
            decay_steps=lr_decay_steps,
        )
        for g in optim.param_groups:
            g["lr"] = lr_now * g.get("lr_scale", 1.0)
        if use_fsdp:
            train_model.clip_grad_norm_(grad_clip)
        else:
            nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
        optim.step()
        optim.zero_grad(set_to_none=True)
        if ema is not None:
            sync_or_update_ema(
                ema,
                raw_model,
                train_model,
                use_fsdp=use_fsdp,
                update=step >= ema_start_step,
            )
        if trace:
            log_rank(
                f"step {step}: optimizer done {time.perf_counter() - t0:.2f}s",
                rank,
                device,
            )
        iters_since_start += 1
        if iters_since_start == 1:
            early_speed_t0 = time.perf_counter()
            if torch.cuda.is_available():
                dev = torch.cuda.current_device()
                _tot = torch.cuda.get_device_properties(dev).total_memory
                _peak = torch.cuda.max_memory_reserved(dev)
                _frac = _peak / max(1, _tot)
                if _frac > 0.97:
                    raise RuntimeError(
                        f"batch too large for this device: peak reserved {_peak / 1000000000.0:.1f} G = {_frac:.1%} of {_tot / 1000000000.0:.1f} G (>97% → allocator-wall deadlock). Reduce batch_size / num_refs and relaunch (resume:auto)."
                    )
        elif not early_speed_logged and iters_since_start == 50:
            early_speed_logged = True
            if is_main_process():
                _dt = time.perf_counter() - early_speed_t0
                _n = iters_since_start - 1
                _ips = _n / _dt if _dt > 0 else 0.0
                _mem = (
                    torch.cuda.max_memory_reserved() / 1000000000.0
                    if torch.cuda.is_available()
                    else 0.0
                )
                log.info(
                    f"[speed@50] {_ips:.3f} it/s | {_ips * eff_batch:.1f} samp/s (avg over {_n} post-compile steps) | peak mem {_mem:.1f}G | parallel={cfg.get('parallel')} batch={cfg['batch_size']} ws={world_size}"
                )
        exposure_report = None
        if cfg_train_states and step % log_every == 0:
            exposure_report = exposure_counts.clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(exposure_report, op=dist.ReduceOp.SUM)
            exposure_counts.zero_()
        inpaint_sampling_report = None
        if step % log_every == 0:
            inpaint_sampling_report = inpaint_sampling_counts.clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(inpaint_sampling_report, op=dist.ReduceOp.SUM)
            inpaint_sampling_counts.zero_()
        conditioned_seg_report = None
        if conditioned_mask_seg_prob > 0.0 and step % log_every == 0:
            conditioned_seg_report = conditioned_seg_counts.clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(conditioned_seg_report, op=dist.ReduceOp.SUM)
            conditioned_seg_counts.zero_()
        if is_main_process() and step % log_every == 0:
            lr_now = optim.param_groups[0]["lr"]
            now = time.perf_counter()
            n = max(step - win_step0, 1)
            dt = now - win_t0
            ips = n / dt if dt > 0 else 0.0
            sps = ips * eff_batch
            if infer_every > 0:
                eta_steps = (step // infer_every + 1) * infer_every - step
            else:
                eta_steps = total_steps - step
            eta_s = eta_steps / ips if ips > 0 else 0.0
            eta_h, rem = divmod(int(eta_s), 3600)
            eta_m, eta_sec = divmod(rem, 60)
            mem_alloc = mem_resv = mem_peak = 0.0
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / 1000000000.0
                mem_resv = torch.cuda.memory_reserved() / 1000000000.0
                mem_peak = torch.cuda.max_memory_reserved() / 1000000000.0
            log_dict = {
                "step": step,
                "loss": float(loss.item()),
                "lr": lr_now,
                "it_s": ips,
                "samples_s": sps,
                "mem_alloc_gb": mem_alloc,
                "mem_resv_gb": mem_resv,
                "mem_peak_gb": mem_peak,
            }
            seg_str = ""
            log_dict["flow_loss"] = float(flow_loss.item())
            if seg_active:
                log_dict["seg_loss"] = float(seg_loss.item())
                seg_ce = float(seg_parts["seg_ce"].item())
                seg_dice_loss = float(seg_parts["seg_dice_loss"].item())
                fg_dice = float(seg_parts["fg_dice"].item())
                log_dict["seg_ce"] = seg_ce
                log_dict["seg_dice_loss"] = seg_dice_loss
                log_dict["fg_dice"] = fg_dice
                log_dict["seg_n"] = float(seg_parts["seg_n"].item())
                conditioned_seg_n = float(conditioned_seg_parts["seg_n"].item())
                conditioned_fg_dice = float(conditioned_seg_parts["fg_dice"].item())
                log_dict["conditioned_seg_loss"] = float(conditioned_seg_loss.item())
                log_dict["conditioned_seg_n"] = conditioned_seg_n
                log_dict["conditioned_fg_dice"] = conditioned_fg_dice
                selected_window = (
                    int(conditioned_seg_report[0])
                    if conditioned_seg_report is not None
                    else 0
                )
                eligible_window = (
                    int(conditioned_seg_report[1])
                    if conditioned_seg_report is not None
                    else 0
                )
                conditioned_rate = selected_window / max(1, eligible_window)
                log_dict["conditioned_seg_selected_window"] = selected_window
                log_dict["conditioned_seg_eligible_window"] = eligible_window
                log_dict["conditioned_seg_rate_window"] = conditioned_rate
                seg_str = f" | flow {flow_loss.item():.6f} | seg {seg_loss.item():.4f} | ce {seg_ce:.4f} | dice {seg_dice_loss:.4f} | fgDice {fg_dice:.4f} | cSeg {conditioned_seg_loss.item():.4f} cFgDice {conditioned_fg_dice:.4f} cN {conditioned_seg_n:.0f} cWin {selected_window}/{eligible_window}={conditioned_rate:.3f}"
            if spatial_enabled:
                with torch.no_grad():
                    log_dict["ref_type_emb_norm"] = float(
                        raw_model.ref_type_emb.weight.norm().item()
                    )
            inpaint_str = ""
            if inpaint_sampling_report is not None:
                ip_n = int(inpaint_sampling_report[0].item())
                ip_place_attempts = int(inpaint_sampling_report[3].item())
                ip_place_failures = int(inpaint_sampling_report[4].item())
                ip_hole_voxels = float(inpaint_sampling_report[1].item()) / max(1, ip_n)
                ip_hole_fraction = float(inpaint_sampling_report[2].item()) / max(
                    1, ip_n
                )
                ip_empty_rate = ip_place_failures / max(1, ip_place_attempts)
                log_dict["inpaint/samples"] = ip_n
                log_dict["inpaint/hole_voxels"] = ip_hole_voxels
                log_dict["inpaint/hole_fraction"] = ip_hole_fraction
                log_dict["inpaint/empty_rate"] = ip_empty_rate
                log_dict["inpaint/placement_attempts"] = ip_place_attempts
                log_dict["inpaint/placement_failures"] = ip_place_failures
                bin_pieces = []
                for bin_i, bin_name in enumerate(inpaint_bin_names):
                    bin_count = int(inpaint_sampling_report[5 + bin_i].item())
                    bin_freq = bin_count / max(1, ip_n)
                    log_dict[f"inpaint/bin_{bin_name}_count"] = bin_count
                    log_dict[f"inpaint/bin_{bin_name}_freq"] = bin_freq
                    bin_pieces.append(f"{bin_name}:{bin_count}({bin_freq:.2f})")
                if ip_n > 0:
                    inpaint_str = (
                        f" | inpaint n={ip_n} hole={ip_hole_voxels:.0f}vox/{100.0 * ip_hole_fraction:.2f}% empty={ip_empty_rate:.3f} "
                        + " ".join(bin_pieces)
                    )
                sampling_csv = exp_dir / "inpaint_sampling.csv"
                sampling_cols = [
                    "step",
                    "samples",
                    "hole_voxels",
                    "hole_fraction",
                    "empty_rate",
                    "placement_attempts",
                    "placement_failures",
                ]
                sampling_row = {
                    "step": step,
                    "samples": ip_n,
                    "hole_voxels": ip_hole_voxels,
                    "hole_fraction": ip_hole_fraction,
                    "empty_rate": ip_empty_rate,
                    "placement_attempts": ip_place_attempts,
                    "placement_failures": ip_place_failures,
                }
                for bin_i, bin_name in enumerate(inpaint_bin_names):
                    count_key = f"bin_{bin_name}_count"
                    freq_key = f"bin_{bin_name}_freq"
                    sampling_cols.extend((count_key, freq_key))
                    bin_count = int(inpaint_sampling_report[5 + bin_i].item())
                    sampling_row[count_key] = bin_count
                    sampling_row[freq_key] = bin_count / max(1, ip_n)
                try:
                    sampling_new = not sampling_csv.exists()
                    with open(sampling_csv, "a") as sf:
                        if sampling_new:
                            sf.write(",".join(sampling_cols) + "\n")
                        sf.write(
                            ",".join((str(sampling_row[c]) for c in sampling_cols))
                            + "\n"
                        )
                except OSError:
                    pass
            exposure_str = ""
            if exposure_report is not None:
                total_seen = max(1, int(exposure_report.sum().item()))
                pieces = []
                exposure_csv = exp_dir / "conditioning_exposure.csv"
                exposure_new = not exposure_csv.exists()
                try:
                    with open(exposure_csv, "a") as ef:
                        if exposure_new:
                            ef.write("step,task,B,T,R,F,total\n")
                        for task_i, task_name in enumerate(exposure_tasks):
                            row = exposure_report[task_i].tolist()
                            row_total = int(sum(row))
                            ef.write(
                                f"{step},{task_name},{row[0]},{row[1]},{row[2]},{row[3]},{row_total}\n"
                            )
                            log_dict[f"exposure/task_{task_name}"] = (
                                row_total / total_seen
                            )
                            for state_i, state_name in enumerate(_CFG_STATE_ORDER):
                                log_dict[f"exposure/{task_name}_{state_name}"] = row[
                                    state_i
                                ] / max(1, row_total)
                            pieces.append(f"{task_name}:{row_total}")
                    exposure_str = " | tasks " + " ".join(pieces)
                except OSError:
                    pass
            if use_wandb:
                wandb.log(log_dict, step=step)
            csv_path = exp_dir / "metrics.csv"
            cols = [
                "step",
                "loss",
                "flow_loss",
                "seg_loss",
                "fg_dice",
                "conditioned_seg_loss",
                "conditioned_fg_dice",
                "conditioned_seg_n",
                "lr",
                "conditioned_seg_selected_window",
                "conditioned_seg_eligible_window",
                "conditioned_seg_rate_window",
                "it_s",
                "samples_s",
                "mem_alloc_gb",
                "mem_resv_gb",
                "mem_peak_gb",
            ]
            try:
                new = not csv_path.exists()
                with open(csv_path, "a") as cf:
                    if new:
                        cf.write(",".join(cols) + "\n")
                    cf.write(",".join((f"{log_dict.get(c, '')}" for c in cols)) + "\n")
            except OSError:
                pass
            log.info(
                f"step {step:6d} | loss {loss.item():.6f}{seg_str} | lr {lr_now:.3e} | {ips:.3f} it/s | {sps:.1f} samp/s | mem {mem_resv:.1f}/{mem_peak:.1f}G | nextInfer~{eta_h:d}h{eta_m:02d}m{eta_sec:02d}s (step {((step // infer_every + 1) * infer_every if infer_every > 0 else total_steps)}){exposure_str}{inpaint_str}"
            )
            win_t0 = now
            win_step0 = step
        if is_main_process() and (not compile_status_logged):
            import torch._dynamo as _dyn

            _is_opt = lambda m: (
                type(m).__name__ == "OptimizedModule" or hasattr(m, "_orig_mod")
            )
            wrapped = _is_opt(train_model) or _is_opt(
                getattr(train_model, "module", train_model)
            )
            stats = dict(_dyn.utils.counters.get("stats", {}))
            n_breaks = sum(_dyn.utils.counters.get("graph_break", {}).values())
            log.info(
                f"[compile] requested={compile_dit} wrapped={wrapped} unique_graphs={stats.get('unique_graphs')} graph_breaks={n_breaks} (flash_varlen is dynamo-disabled by design, so some breaks are expected)"
            )
            compile_status_logged = True
        if trace:
            log_rank(
                f"step {step} done total {time.perf_counter() - step_t0:.2f}s",
                rank,
                device,
            )
        step += 1
        if step > 0:
            do_milestone = step % ckpt_every == 0
            do_latest = ckpt_latest_every > 0 and step % ckpt_latest_every == 0
            if do_milestone or do_latest:
                save_ckpt(
                    raw_model,
                    optim,
                    step,
                    exp_dir,
                    ema=ema,
                    train_model=train_model,
                    use_fsdp=use_fsdp,
                    keep_last=keep_last_ckpts,
                    milestone=do_milestone,
                )
        if infer_every > 0 and step > 0 and (step % infer_every == 0):
            fid = run_validation_eval(
                cfg,
                raw_model,
                train_model,
                device,
                exp_dir,
                step,
                mmdit_cfg,
                ema=ema,
                use_fsdp=use_fsdp,
            )
            if is_main_process() and fid is not None and use_wandb:
                wandb.log({"slice_fid": fid, "step": step})
            is_best_t = torch.tensor(
                [1 if fid is not None and fid < best_fid else 0],
                device=device,
                dtype=torch.int32,
            )
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(is_best_t, src=0)
            if int(is_best_t.item()) == 1:
                if fid is not None:
                    best_fid = fid
                save_best_ckpt(
                    raw_model,
                    exp_dir,
                    fid if fid is not None else best_fid,
                    ema=ema,
                    train_model=train_model,
                    use_fsdp=use_fsdp,
                )
                if is_main_process():
                    log.info(
                        f"[best] new best {primary_fid_mode} slice_FID={best_fid:.3f} @ step {step} → dit_ckpt_step_best.pt"
                    )
            train_model.train()
    save_ckpt(
        raw_model,
        optim,
        step,
        exp_dir,
        ema=ema,
        train_model=train_model,
        use_fsdp=use_fsdp,
        keep_last=keep_last_ckpts,
    )
    if is_main_process():
        log.info("Training complete.")
        if use_wandb:
            wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
