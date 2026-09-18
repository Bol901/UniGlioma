from __future__ import annotations
import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Set, Type
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)

try:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    _HAS_AC = True
except Exception:
    _HAS_AC = False
from .mmdit import MMDiT3D, MMDiTCfg, DoubleStreamBlock, SingleStreamBlock


@dataclass
class FSDPConfig:
    wrap_policy: str = "transformer"
    min_params: int = 1000000
    sharding: str = "SHARD_GRAD_OP"
    cpu_offload: bool = False
    compute_dtype: str = "bf16"
    param_dtype: Optional[str] = None
    reduce_dtype: Optional[str] = None
    use_orig_params: bool = True
    backward_prefetch: str = "BACKWARD_PRE"
    sync_module_states: bool = True
    limit_all_gathers: bool = True
    activation_ckpt: bool = True
    ckpt_targets: Optional[Set[Type[nn.Module]]] = None


def _dtype_from_str(s: Optional[str]) -> Optional[torch.dtype]:
    if s is None:
        return None
    s = s.lower()
    if s == "bf16":
        return torch.bfloat16
    if s == "fp16":
        return torch.float16
    if s == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def _sharding_from_str(s: str) -> ShardingStrategy:
    s = s.upper()
    if s in {"FULL_SHARD", "FSDP"}:
        return ShardingStrategy.FULL_SHARD
    if s in {"SHARD_GRAD_OP", "HYBRID_SHARD"}:
        return ShardingStrategy.SHARD_GRAD_OP
    if s in {"NO_SHARD", "DP"}:
        return ShardingStrategy.NO_SHARD
    raise ValueError(f"Unsupported sharding: {s}")


def _backward_prefetch_from_str(s: str) -> BackwardPrefetch:
    s = s.upper()
    if s == "BACKWARD_PRE":
        return BackwardPrefetch.BACKWARD_PRE
    if s == "BACKWARD_POST":
        return BackwardPrefetch.BACKWARD_POST
    raise ValueError(f"Unsupported backward_prefetch: {s}")


def _build_mixed_precision(cfg: FSDPConfig) -> Optional[MixedPrecision]:
    comp = _dtype_from_str(cfg.compute_dtype)
    par = _dtype_from_str(cfg.param_dtype or cfg.compute_dtype)
    red = _dtype_from_str(cfg.reduce_dtype or cfg.compute_dtype)
    if comp == torch.float32 and par == torch.float32 and (red == torch.float32):
        return None
    return MixedPrecision(param_dtype=par, reduce_dtype=red, buffer_dtype=None)


def _build_auto_wrap_policy(cfg: FSDPConfig):
    if cfg.wrap_policy in ("none", "None", None):
        return None
    if cfg.wrap_policy == "transformer":
        layer_set = {DoubleStreamBlock, SingleStreamBlock}
        try:
            return functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls=layer_set
            )
        except TypeError:
            return functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls_set=layer_set
            )
    if cfg.wrap_policy == "size_based":
        return functools.partial(
            size_based_auto_wrap_policy, min_num_params=int(cfg.min_params)
        )
    raise ValueError(f"Unsupported wrap_policy: {cfg.wrap_policy}")


def _maybe_apply_activation_ckpt(model: nn.Module, cfg: FSDPConfig) -> None:
    if not cfg.activation_ckpt or not _HAS_AC:
        return
    target_set = cfg.ckpt_targets or {DoubleStreamBlock, SingleStreamBlock}

    def check_fn(m: nn.Module) -> bool:
        return any((isinstance(m, t) for t in target_set))

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
            m, checkpoint_impl=CheckpointImpl.NO_REENTRANT, preserve_rng_state=False
        ),
        check_fn=check_fn,
    )


def fsdp_config_from_dict(d: dict) -> FSDPConfig:
    return FSDPConfig(
        wrap_policy=d.get("wrap_policy", "transformer"),
        min_params=int(d.get("min_params", 1000000)),
        sharding=d.get("sharding", "SHARD_GRAD_OP"),
        cpu_offload=bool(d.get("cpu_offload", False)),
        compute_dtype=d.get("compute_dtype", "bf16"),
        param_dtype=d.get("param_dtype"),
        reduce_dtype=d.get("reduce_dtype"),
        use_orig_params=bool(d.get("use_orig_params", True)),
        backward_prefetch=d.get("backward_prefetch", "BACKWARD_PRE"),
        sync_module_states=bool(d.get("sync_module_states", True)),
        limit_all_gathers=bool(d.get("limit_all_gathers", True)),
        activation_ckpt=bool(d.get("activation_ckpt", True)),
    )


def build_fsdp_mmdit(
    mmdit_cfg: MMDiTCfg,
    fsdp_cfg: FSDPConfig = FSDPConfig(),
    device: Optional[torch.device] = None,
) -> FSDP:
    if device is None:
        device = torch.device(
            "cuda", torch.cuda.current_device() if torch.cuda.is_available() else 0
        )
    model = MMDiT3D(mmdit_cfg)
    _maybe_apply_activation_ckpt(model, fsdp_cfg)
    auto_wrap_policy = _build_auto_wrap_policy(fsdp_cfg)
    mp_policy = _build_mixed_precision(fsdp_cfg)
    return FSDP(
        module=model,
        sharding_strategy=_sharding_from_str(fsdp_cfg.sharding),
        cpu_offload=CPUOffload(offload_params=fsdp_cfg.cpu_offload),
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        use_orig_params=fsdp_cfg.use_orig_params,
        backward_prefetch=_backward_prefetch_from_str(fsdp_cfg.backward_prefetch),
        sync_module_states=fsdp_cfg.sync_module_states,
        limit_all_gathers=fsdp_cfg.limit_all_gathers,
        device_id=device if device.type == "cuda" else None,
    )


def wrap_fsdp_mmdit(
    model: MMDiT3D,
    fsdp_cfg: FSDPConfig = FSDPConfig(),
    device: Optional[torch.device] = None,
) -> FSDP:
    if device is None:
        device = torch.device(
            "cuda", torch.cuda.current_device() if torch.cuda.is_available() else 0
        )
    _maybe_apply_activation_ckpt(model, fsdp_cfg)
    auto_wrap_policy = _build_auto_wrap_policy(fsdp_cfg)
    mp_policy = _build_mixed_precision(fsdp_cfg)
    return FSDP(
        module=model,
        sharding_strategy=_sharding_from_str(fsdp_cfg.sharding),
        cpu_offload=CPUOffload(offload_params=fsdp_cfg.cpu_offload),
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        use_orig_params=fsdp_cfg.use_orig_params,
        backward_prefetch=_backward_prefetch_from_str(fsdp_cfg.backward_prefetch),
        sync_module_states=fsdp_cfg.sync_module_states,
        limit_all_gathers=fsdp_cfg.limit_all_gathers,
        device_id=device if device.type == "cuda" else None,
    )


@contextmanager
def mmdit_autocast(fsdp_cfg: FSDPConfig):
    dtype = _dtype_from_str(fsdp_cfg.compute_dtype)
    if dtype == torch.float32:
        yield
    else:
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
