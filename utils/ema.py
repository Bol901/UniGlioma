from __future__ import annotations
from contextlib import contextmanager
import torch


def _norm_name(n: str) -> str:
    if n.startswith("module."):
        n = n[len("module.") :]
    n = n.replace("._fsdp_wrapped_module", "").replace("_fsdp_wrapped_module.", "")
    n = n.replace("._checkpoint_wrapped_module", "").replace(
        "_checkpoint_wrapped_module.", ""
    )
    return n


class EMA:
    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        track_all: bool = False,
    ):
        self.decay = float(decay)
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype
        self.track_all = bool(track_all)
        self.shadow: dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if self.track_all or p.requires_grad:
                self.shadow[_norm_name(n)] = self._copy_param(p)

    def _copy_param(self, p: torch.Tensor) -> torch.Tensor:
        dst = p.detach()
        if self.device is not None or self.dtype is not None:
            dst = dst.to(
                device=self.device if self.device is not None else dst.device,
                dtype=self.dtype if self.dtype is not None else dst.dtype,
            )
        return dst.clone()

    @torch.no_grad()
    def sync(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.track_all or p.requires_grad:
                self.shadow[_norm_name(n)] = self._copy_param(p)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            nn = _norm_name(n)
            s = self.shadow.get(nn)
            if s is None:
                self.shadow[nn] = self._copy_param(p)
            else:
                p_detached = p.detach().to(device=s.device, dtype=s.dtype)
                s.mul_(d).add_(p_detached, alpha=1.0 - d)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    @torch.no_grad()
    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        loaded_from_wrapped: set[str] = set()
        for n, t in sd.items():
            nn = _norm_name(n)
            was_wrapped = nn != n
            if nn in loaded_from_wrapped and (not was_wrapped):
                continue
            existing = self.shadow.get(nn)
            target_device = (
                self.device
                if self.device is not None
                else existing.device
                if existing is not None
                else t.device
            )
            target_dtype = (
                self.dtype
                if self.dtype is not None
                else existing.dtype
                if existing is not None
                else t.dtype
            )
            self.shadow[nn] = t.detach().to(target_device, dtype=target_dtype).clone()
            if was_wrapped:
                loaded_from_wrapped.add(nn)

    @contextmanager
    @torch.no_grad()
    def apply(self, model: torch.nn.Module):
        backup = {}
        try:
            for n, p in model.named_parameters():
                nn = _norm_name(n)
                if nn in self.shadow:
                    backup[n] = p.detach().cpu().clone()
                    p.data.copy_(self.shadow[nn].to(device=p.device, dtype=p.dtype))
            yield
        finally:
            for n, p in model.named_parameters():
                if n in backup:
                    p.data.copy_(backup[n].to(device=p.device, dtype=p.dtype))
