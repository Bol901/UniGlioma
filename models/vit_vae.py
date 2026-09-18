"""Volumetric variational autoencoder architecture; no pretrained weights included."""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class DiagonalGaussian:
    def __init__(self, mu: torch.Tensor, logvar: torch.Tensor):
        self.mu = mu
        self.logvar = logvar

    @torch.no_grad()
    def mode(self):
        return self.mu

    def sample(self):
        std = (0.5 * self.logvar).exp()
        eps = torch.randn_like(std)
        return self.mu + eps * std

    def kl(self):
        return -0.5 * (1 + self.logvar - self.mu.pow(2) - self.logvar.exp())


@dataclass
class EncodeOutput:
    latent_dist: DiagonalGaussian
    grid_shape: Tuple[int, int, int]


def exists(x):
    return x is not None


def prod(x):
    out = 1
    for v in x:
        out *= v
    return out


class RoPE3D(nn.Module):
    def __init__(self, head_dim: int, base_theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        rope_dim = head_dim // 6 * 6
        self.rope_dim = rope_dim
        self.base_theta = base_theta
        if rope_dim > 0:
            twoL = rope_dim // 3
            twoL = twoL // 2 * 2
            self.twoL = twoL
            self.rope_dim = twoL * 3
            d_pair = twoL // 2
            inv_freq = 1.0 / base_theta ** (torch.arange(d_pair).bfloat16() / d_pair)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        else:
            self.twoL = 0
            self.register_buffer("inv_freq", torch.tensor([]), persistent=False)

    def _build_angles(self, npos: int, device):
        d_pair = self.inv_freq.shape[0]
        t = torch.arange(npos, device=device).bfloat16().unsqueeze(-1)
        return t * self.inv_freq.unsqueeze(0)

    @staticmethod
    def _apply_rotary_pairs(x_axis_chunk, cos, sin):
        B, Hn, N, twoL = x_axis_chunk.shape
        d_pair = twoL // 2
        x_pair = x_axis_chunk.view(B, Hn, N, d_pair, 2)
        x_even = x_pair[..., 0]
        x_odd = x_pair[..., 1]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1).reshape(B, Hn, N, twoL)
        return x_rot

    def forward(self, q, k, D: int, H: int, W: int):
        if self.rope_dim == 0:
            return (q, k)
        B, Hn, N, Hd = q.shape
        device = q.device
        twoL = self.twoL
        rope_dim = self.rope_dim
        q_rot, q_rest = (q[..., :rope_dim], q[..., rope_dim:])
        k_rot, k_rest = (k[..., :rope_dim], k[..., rope_dim:])
        qx, qy, qz = torch.split(q_rot, [twoL, twoL, twoL], dim=-1)
        kx, ky, kz = torch.split(k_rot, [twoL, twoL, twoL], dim=-1)
        idx = torch.arange(N, device=device)
        zz = idx // (H * W)
        yy = idx // W % H
        xx = idx % W
        d_pair = twoL // 2
        angles_x = self._build_angles(W, device)
        cos_x = angles_x.cos()[xx]
        sin_x = angles_x.sin()[xx]
        angles_y = self._build_angles(H, device)
        cos_y = angles_y.cos()[yy]
        sin_y = angles_y.sin()[yy]
        angles_z = self._build_angles(D, device)
        cos_z = angles_z.cos()[zz]
        sin_z = angles_z.sin()[zz]
        qx = self._apply_rotary_pairs(qx, cos_x, sin_x)
        qy = self._apply_rotary_pairs(qy, cos_y, sin_y)
        qz = self._apply_rotary_pairs(qz, cos_z, sin_z)
        kx = self._apply_rotary_pairs(kx, cos_x, sin_x)
        ky = self._apply_rotary_pairs(ky, cos_y, sin_y)
        kz = self._apply_rotary_pairs(kz, cos_z, sin_z)
        q_out = torch.cat([qx, qy, qz, q_rest], dim=-1)
        k_out = torch.cat([kx, ky, kz, k_rest], dim=-1)
        return (q_out, k_out)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        inner = int(dim * mult)
        self.proj = nn.Linear(dim, inner * 2, bias=False)
        self.out = nn.Linear(inner, dim, bias=False)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return self.out(F.silu(gate) * x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        rope: Optional[RoPE3D] = None,
        layer_scale_init: float = 0.0001,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** (-0.5)
        inner = heads * dim_head
        self.qkv = nn.Linear(dim, inner * 3, bias=qkv_bias)
        self.proj = nn.Linear(inner, dim, bias=False)
        self.attn_dropout = attn_dropout
        self.resid_dropout = resid_dropout
        self.rope = rope
        self.gamma = nn.Parameter(torch.ones(dim) * layer_scale_init)

    def forward(self, x, D: int, H: int, W: int):
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        Hn, Dh = (self.heads, self.dim_head)
        q = q.view(B, N, Hn, Dh).transpose(1, 2)
        k = k.view(B, N, Hn, Dh).transpose(1, 2)
        v = v.view(B, N, Hn, Dh).transpose(1, 2)
        if exists(self.rope):
            q, k = self.rope(q, k, D, H, W)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        out = attn_out.transpose(1, 2).contiguous().view(B, N, Hn * Dh)
        out = self.proj(out)
        return x + out * self.gamma


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_mult: float,
        rope: Optional[RoPE3D],
        layer_scale_init: float = 0.0001,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(
            dim,
            heads,
            dim_head,
            rope,
            layer_scale_init=layer_scale_init,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
        )
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(SwiGLU(dim, mult=mlp_mult))
        self.gamma2 = nn.Parameter(torch.ones(dim) * layer_scale_init)

    def forward(self, x, D, H, W):
        x = x.to(x.dtype)
        x = self.attn(self.norm1(x), D, H, W)
        x = x + self.mlp(self.norm2(x)) * self.gamma2
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels: int, dim: int, patch_size: Tuple[int, int, int]):
        super().__init__()
        pz, py, px = patch_size
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, dim, kernel_size=(pz, py, px), stride=(pz, py, px)
        )

    def forward(self, x):
        _, _, Din, Hin, Win = x.shape
        pz, py, px = self.patch_size
        if Din % pz != 0 or Hin % py != 0 or Win % px != 0:
            raise ValueError(
                f"Input spatial shape {(Din, Hin, Win)} must be divisible by patch_size {self.patch_size}."
            )
        x = self.proj(x)
        B, C, D, H, W = x.shape
        x = x.view(B, C, D * H * W).transpose(1, 2).contiguous()
        return (x, (D, H, W))


class PatchUnembed3D(nn.Module):
    def __init__(
        self,
        out_channels: int,
        dim: int,
        patch_size: Tuple[int, int, int],
        out_activation: Optional[str] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.out_activation = out_activation
        pz, py, px = patch_size
        self.patch_proj = nn.Linear(dim, out_channels * pz * py * px, bias=False)

    def forward(self, tokens, grid_shape, target_shape):
        B, N, C = tokens.shape
        Dg, Hg, Wg = grid_shape
        pz, py, px = self.patch_size
        D, H, W = target_shape
        patches = self.patch_proj(tokens)
        patches = patches.view(B, N, self.out_channels, pz, py, px)
        patches = patches.view(B, Dg, Hg, Wg, self.out_channels, pz, py, px)
        patches = patches.permute(0, 4, 1, 5, 2, 6, 3, 7)
        out = patches.reshape(B, self.out_channels, Dg * pz, Hg * py, Wg * px)
        if exists(self.out_activation):
            if self.out_activation == "sigmoid":
                out = out.sigmoid()
            elif self.out_activation == "tanh":
                out = out.tanh()
        return out


def tag_fsdp_wrap_blocks(model: nn.Module):
    for m in model.modules():
        if m.__class__.__name__ == "TransformerBlock":
            setattr(m, "_fsdp_wrap", True)
        else:
            setattr(m, "_fsdp_wrap", False)
    return model


def make_block_policy(min_params: int = 0):

    def _should_wrap(mod, recurse, nonwrapped_numel):
        flag = getattr(mod, "_fsdp_wrap", False)
        if not flag:
            return False
        if min_params > 0:
            n = sum((p.numel() for p in mod.parameters(recurse=False)))
            return n >= min_params
        return True

    def policy(module, recurse, nonwrapped_numel):
        return _should_wrap(module, recurse, nonwrapped_numel)

    return policy


class ViTEncoder3D(nn.Module):
    def __init__(
        self,
        in_channels,
        dim,
        depth,
        heads,
        dim_head,
        mlp_mult,
        patch_size,
        rope: Optional[RoPE3D],
    ):
        super().__init__()
        self.patch = PatchEmbed3D(in_channels, dim, patch_size)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim, heads, dim_head, mlp_mult, rope)
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(dim)

    def forward(self, x):
        tokens, (Dz, Dy, Dx) = self.patch(x)
        for blk in self.blocks:
            tokens = blk(tokens, Dz, Dy, Dx)
        tokens = self.norm(tokens)
        return (tokens, (Dz, Dy, Dx))


class ViTDecoder3D(nn.Module):
    def __init__(
        self,
        out_channels,
        dim,
        depth,
        heads,
        dim_head,
        mlp_mult,
        patch_size,
        rope: Optional[RoPE3D],
        out_activation=None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim, heads, dim_head, mlp_mult, rope)
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(dim)
        self.unpatch = PatchUnembed3D(
            out_channels, dim, patch_size, out_activation=out_activation
        )

    def forward(self, tokens, grid_shape, target_shape):
        Dz, Dy, Dx = grid_shape
        for blk in self.blocks:
            tokens = blk(tokens, Dz, Dy, Dx)
        tokens = self.norm(tokens)
        x = self.unpatch(tokens, grid_shape, target_shape)
        return x


@dataclass
class VAE3DConfig:
    in_channels: int = 1
    out_channels: int = 1
    patch_size: Tuple[int, int, int] = (4, 8, 8)
    dim: int = 768
    depth_enc: int = 8
    depth_dec: int = 8
    heads: int = 12
    dim_head: int = 64
    mlp_mult: float = 4.0
    latent_dim: int = 16
    rope_base_theta: float = 10000.0
    out_activation: Optional[str] = "sigmoid"


class ViTVAE3D(nn.Module):
    def __init__(self, cfg: VAE3DConfig, use_compile: bool = True):
        super().__init__()
        self.cfg = cfg
        rope = RoPE3D(head_dim=cfg.dim_head, base_theta=cfg.rope_base_theta)
        self.encoder = ViTEncoder3D(
            in_channels=cfg.in_channels,
            dim=cfg.dim,
            depth=cfg.depth_enc,
            heads=cfg.heads,
            dim_head=cfg.dim_head,
            mlp_mult=cfg.mlp_mult,
            patch_size=cfg.patch_size,
            rope=rope,
        )
        self.to_mu = nn.Linear(cfg.dim, cfg.latent_dim, bias=False)
        self.to_logvar = nn.Linear(cfg.dim, cfg.latent_dim, bias=False)
        self.from_z = nn.Linear(cfg.latent_dim, cfg.dim, bias=False)
        self.decoder = ViTDecoder3D(
            out_channels=cfg.out_channels,
            dim=cfg.dim,
            depth=cfg.depth_dec,
            heads=cfg.heads,
            dim_head=cfg.dim_head,
            mlp_mult=cfg.mlp_mult,
            patch_size=cfg.patch_size,
            rope=rope,
            out_activation=cfg.out_activation,
        )
        self._last_grid_shape: Optional[Tuple[int, int, int]] = None
        self._last_target_shape: Optional[Tuple[int, int, int]] = None

        def _encode_tensors_fn(x: torch.Tensor):
            tokens, grid = self.encoder(x)
            mu = self.to_mu(tokens)
            logvar = self.to_logvar(tokens)
            Dz, Dy, Dx = grid
            grid_tensor = torch.tensor([Dz, Dy, Dx], device=x.device, dtype=torch.int32)
            target_tensor = torch.tensor(
                [x.shape[2], x.shape[3], x.shape[4]], device=x.device, dtype=torch.int32
            )
            return (mu, logvar, grid_tensor, target_tensor)

        def _decode_tensors_fn(
            z: torch.Tensor, Dz: int, Dy: int, Dx: int, D: int, H: int, W: int
        ):
            tokens = self.from_z(z)
            x_hat = self.decoder(tokens, (Dz, Dy, Dx), (D, H, W))
            return x_hat

        if use_compile:
            self._encode_tensors = torch.compile(
                _encode_tensors_fn,
                dynamic=True,
                fullgraph=False,
                mode="max-autotune-no-cudagraphs",
            )
            self._decode_tensors = torch.compile(
                _decode_tensors_fn,
                dynamic=True,
                fullgraph=False,
                mode="max-autotune-no-cudagraphs",
            )
        else:
            self._encode_tensors = _encode_tensors_fn
            self._decode_tensors = _decode_tensors_fn

    def encode(self, x) -> EncodeOutput:
        mu, logvar, grid_tensor, target_tensor = self._encode_tensors(x)
        Dz, Dy, Dx = [int(v.item()) for v in grid_tensor]
        D, H, W = [int(v.item()) for v in target_tensor]
        self._last_grid_shape = (Dz, Dy, Dx)
        self._last_target_shape = (D, H, W)
        return EncodeOutput(
            latent_dist=DiagonalGaussian(mu, logvar), grid_shape=(Dz, Dy, Dx)
        )

    def decode(
        self,
        z: torch.Tensor,
        grid: Optional[Tuple[int, int, int]] = None,
        target_shape: Optional[Tuple[int, int, int]] = None,
    ):
        pz, py, px = self.cfg.patch_size
        if grid is None and target_shape is not None:
            D, H, W = (int(target_shape[0]), int(target_shape[1]), int(target_shape[2]))
            if D % pz != 0 or H % py != 0 or W % px != 0:
                raise ValueError(
                    f"target_shape {(D, H, W)} must be divisible by patch_size {self.cfg.patch_size}."
                )
            grid = (D // pz, H // py, W // px)
        if grid is None:
            if self._last_grid_shape is None:
                raise ValueError(
                    "decode requires grid when called before encode/forward."
                )
            grid = self._last_grid_shape
        expected_target_shape = (
            int(grid[0]) * pz,
            int(grid[1]) * py,
            int(grid[2]) * px,
        )
        if target_shape is None:
            target_shape = expected_target_shape
        else:
            target_shape = (
                int(target_shape[0]),
                int(target_shape[1]),
                int(target_shape[2]),
            )
            if target_shape != expected_target_shape:
                raise ValueError(
                    f"Inconsistent decode args: grid {grid} with patch_size {self.cfg.patch_size} implies target_shape {expected_target_shape}, but got {target_shape}."
                )
        Dz, Dy, Dx = (int(grid[0]), int(grid[1]), int(grid[2]))
        D, H, W = (int(target_shape[0]), int(target_shape[1]), int(target_shape[2]))
        return self._decode_tensors(z, Dz, Dy, Dx, D, H, W)

    def forward(self, x):
        mu, logvar, grid_tensor, target_tensor = self._encode_tensors(x)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        Dz, Dy, Dx = (
            int(grid_tensor[0].item()),
            int(grid_tensor[1].item()),
            int(grid_tensor[2].item()),
        )
        D, H, W = (
            int(target_tensor[0].item()),
            int(target_tensor[1].item()),
            int(target_tensor[2].item()),
        )
        self._last_grid_shape = (Dz, Dy, Dx)
        self._last_target_shape = (D, H, W)
        x_hat = self._decode_tensors(z, Dz, Dy, Dx, D, H, W)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl = kl.mean(dim=(1, 2))
        return (x_hat, kl)
