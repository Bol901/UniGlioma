"""3D diffusion transformer with spatially aligned reference-axis fusion."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

_FLASH_BACKEND = None
_FLASH_IMPORT_ERRORS = []
try:
    from flash_attn import flash_attn_varlen_func

    _FLASH_BACKEND = "fa2"
except Exception as exc:
    _FLASH_IMPORT_ERRORS.append(("fa2", exc))
    try:
        from flash_attn_interface import flash_attn_varlen_func

        _FLASH_BACKEND = "fa3"
    except Exception as exc:
        _FLASH_IMPORT_ERRORS.append(("fa3", exc))
_HAS_FLASH = _FLASH_BACKEND is not None


@dataclass
class MMDiTCfg:
    in_channels: int = 16
    latent_shape: tuple = (24, 24, 18)
    patch_size: tuple = (2, 2, 2)
    max_refs: int = 3
    ref_fusion_heads: int = 0
    delta_time_dim: int = 256
    use_ref_slab: bool = True
    width: int = 1152
    text_hidden: int = 768
    text_max_len: int = 512
    heads: int = 16
    mlp_ratio: float = 4.0
    depth_double: int = 12
    depth_single: int = 8
    num_modalities: int = 4
    use_uncond_modality_slot: bool = True
    timestep_embed_dim: int = 256
    pool_text_for_vec: bool = True
    rope_mode: str = "axial3d"
    rope_theta_base: float = 10000.0
    dropout: float = 0.0
    use_qk_norm: bool = False
    mlp_variant: str = "gelu"
    attn_impl: str = "sdpa"
    grad_checkpoint: bool = False
    seg_head_enabled: bool = False
    seg_tap_layer: int = -1
    seg_channels: tuple = (256, 128, 64, 32, 16)
    seg_num_classes: int = 4
    seg_full_shape: tuple = (192, 192, 144)
    n_ref_types: int = 3
    arch_version: int = 1


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = (x[..., ::2], x[..., 1::2])
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def build_rope_axial3d(Dp, Hp, Wp, Dh, device, theta=10000.0, d_offset: int = 0):
    assert Dh % 6 == 0, "axial3d requires head_dim % 6 == 0"
    per = Dh // 3
    half = per // 2
    d = torch.arange(Dp, device=device) + int(d_offset)
    h = torch.arange(Hp, device=device)
    w = torch.arange(Wp, device=device)
    gd, gh, gw = torch.meshgrid(d, h, w, indexing="ij")
    pos_d = gd.reshape(-1).float()
    pos_h = gh.reshape(-1).float()
    pos_w = gw.reshape(-1).float()
    inv = lambda: 1.0 / theta ** (torch.arange(0, half, device=device).float() / half)
    idf, ihf, iwf = (inv(), inv(), inv())
    fd = torch.einsum("n,d->nd", pos_d, idf)
    fh = torch.einsum("n,d->nd", pos_h, ihf)
    fw = torch.einsum("n,d->nd", pos_w, iwf)
    cd, sd = (
        torch.cos(fd).repeat_interleave(2, -1),
        torch.sin(fd).repeat_interleave(2, -1),
    )
    ch, sh = (
        torch.cos(fh).repeat_interleave(2, -1),
        torch.sin(fh).repeat_interleave(2, -1),
    )
    cw, sw = (
        torch.cos(fw).repeat_interleave(2, -1),
        torch.sin(fw).repeat_interleave(2, -1),
    )
    cos = torch.cat([cd, ch, cw], -1)
    sin = torch.cat([sd, sh, sw], -1)
    return (cos, sin)


def build_rope_with_refs(Dp, Hp, Wp, n_refs: int, Dh, device, theta=10000.0):
    if n_refs <= 0:
        return build_rope_axial3d(Dp, Hp, Wp, Dh, device, theta)
    D_OFFSET = Dp + 4
    cos_list, sin_list = ([], [])
    for k in range(n_refs + 1):
        c, s = build_rope_axial3d(Dp, Hp, Wp, Dh, device, theta, d_offset=k * D_OFFSET)
        cos_list.append(c)
        sin_list.append(s)
    return (torch.cat(cos_list, dim=0), torch.cat(sin_list, dim=0))


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch: int, dim: int, patch, latent_shape, *, bias: bool = True):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, dim, kernel_size=patch, stride=patch, bias=bias)
        D, H, W = latent_shape
        pd, ph, pw = patch
        self.Dp, self.Hp, self.Wp = (D // pd, H // ph, W // pw)
        self.N = self.Dp * self.Hp * self.Wp
        self.patch = tuple(patch)
        self.in_ch = in_ch

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        Dp, Hp, Wp = (x.shape[2], x.shape[3], x.shape[4])
        tokens = x.flatten(2).transpose(1, 2)
        return (tokens, (Dp, Hp, Wp))


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool = True):
        super().__init__()
        self.double = double
        out = 6 if double else 3
        self.lin = nn.Linear(dim, dim * out)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, vec: torch.Tensor):
        out = self.lin(F.silu(vec))
        if self.double:
            shift1, scale1, gate1, shift2, scale2, gate2 = out.chunk(6, dim=-1)
            return ((shift1, scale1, gate1), (shift2, scale2, gate2))
        shift, scale, gate = out.chunk(3, dim=-1)
        return (shift, scale, gate)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _qkv_split(qkv: torch.Tensor, heads: int):
    B, N, _ = qkv.shape
    qkv = qkv.view(B, N, 3, heads, -1).permute(2, 0, 3, 1, 4)
    return (qkv[0], qkv[1], qkv[2])


@torch._dynamo.disable
def _flash_varlen_attention(q, k, v, seq_valid, dropout_p):
    B, H, N, Dh = q.shape
    qb, kb, vb = (
        t.permute(0, 2, 1, 3).to(torch.bfloat16).contiguous() for t in (q, k, v)
    )
    valid = seq_valid.reshape(-1)
    idx = torch.nonzero(valid, as_tuple=False).flatten()
    flat_q, flat_k, flat_v = (t.reshape(B * N, H, Dh) for t in (qb, kb, vb))
    q_un = flat_q.index_select(0, idx)
    k_un = flat_k.index_select(0, idx)
    v_un = flat_v.index_select(0, idx)
    lengths = seq_valid.sum(dim=1, dtype=torch.int32)
    cu = F.pad(lengths.cumsum(dim=0, dtype=torch.int32), (1, 0))
    max_s = int(lengths.max().item())
    kwargs = dict(max_seqlen_q=max_s, max_seqlen_k=max_s, causal=False)
    if _FLASH_BACKEND == "fa2":
        kwargs["dropout_p"] = dropout_p
    out = flash_attn_varlen_func(q_un, k_un, v_un, cu, cu, **kwargs)
    out = out.new_zeros((B * N, H, Dh)).index_copy(0, idx, out)
    out = out.view(B, N, H, Dh)
    return out.permute(0, 2, 1, 3)


def _attention(q, k, v, seq_valid, dropout_p, use_flash):
    if (
        use_flash
        and _HAS_FLASH
        and (seq_valid is not None)
        and (not (_FLASH_BACKEND == "fa3" and dropout_p != 0.0))
    ):
        return _flash_varlen_attention(q, k, v, seq_valid, dropout_p)
    attn_mask = None
    if seq_valid is not None:
        B, N = seq_valid.shape
        attn_mask = seq_valid.view(B, 1, 1, N)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
    )


def joint_attention(
    q_img,
    k_img,
    v_img,
    q_cond,
    k_cond,
    v_cond,
    cond_attn_mask: Optional[torch.Tensor],
    dropout_p: float = 0.0,
    use_flash: bool = False,
    img_attn_mask: Optional[torch.Tensor] = None,
):
    B, H, N_img, Dh = q_img.shape
    q = torch.cat([q_img, q_cond], dim=2)
    k = torch.cat([k_img, k_cond], dim=2)
    v = torch.cat([v_img, v_cond], dim=2)
    seq_valid = None
    if cond_attn_mask is not None or img_attn_mask is not None:
        if img_attn_mask is None:
            img_valid = torch.ones(B, N_img, device=q.device, dtype=torch.bool)
        else:
            img_valid = img_attn_mask.bool()
        if cond_attn_mask is None:
            cond_valid = torch.ones(
                B, q_cond.shape[2], device=q.device, dtype=torch.bool
            )
        else:
            cond_valid = cond_attn_mask.bool()
        seq_valid = torch.cat([img_valid, cond_valid], dim=1)
    out = _attention(q, k, v, seq_valid, dropout_p, use_flash)
    return (out[:, :, :N_img], out[:, :, N_img:])


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add_(self.eps).rsqrt_()
        return (x32 * rms).to(dtype) * self.weight


def _qk_norm_layer(head_dim: int, enabled: bool) -> nn.Module:
    return RMSNorm(head_dim) if enabled else nn.Identity()


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w_gate_up = nn.Linear(dim, 2 * hidden)
        self.w_down = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


def _build_mlp(dim: int, mlp_ratio: float, variant: str) -> nn.Module:
    hidden = int(dim * mlp_ratio)
    if variant == "gelu":
        return nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim)
        )
    if variant == "swiglu":
        return SwiGLU(dim, hidden)
    raise ValueError(f"unknown mlp_variant {variant!r}; expected 'gelu' or 'swiglu'")


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
        use_qk_norm: bool = False,
        mlp_variant: str = "gelu",
        use_flash: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.use_flash = use_flash
        self.img_mod = Modulation(dim, double=True)
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-06)
        self.img_qkv = nn.Linear(dim, dim * 3)
        self.img_q_norm = _qk_norm_layer(self.head_dim, use_qk_norm)
        self.img_k_norm = _qk_norm_layer(self.head_dim, use_qk_norm)
        self.img_proj = nn.Linear(dim, dim)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-06)
        self.img_mlp = _build_mlp(dim, mlp_ratio, mlp_variant)
        self.cond_mod = Modulation(dim, double=True)
        self.cond_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-06)
        self.cond_qkv = nn.Linear(dim, dim * 3)
        self.cond_q_norm = _qk_norm_layer(self.head_dim, use_qk_norm)
        self.cond_k_norm = _qk_norm_layer(self.head_dim, use_qk_norm)
        self.cond_proj = nn.Linear(dim, dim)
        self.cond_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-06)
        self.cond_mlp = _build_mlp(dim, mlp_ratio, mlp_variant)

    def forward(
        self,
        img: torch.Tensor,
        cond: torch.Tensor,
        vec: torch.Tensor,
        rope_cos_sin,
        cond_attn_mask: Optional[torch.Tensor],
        img_attn_mask: Optional[torch.Tensor] = None,
    ):
        (i_s1, i_sc1, i_g1), (i_s2, i_sc2, i_g2) = self.img_mod(vec)
        (c_s1, c_sc1, c_g1), (c_s2, c_sc2, c_g2) = self.cond_mod(vec)
        img_m = modulate(self.img_norm1(img), i_s1, i_sc1)
        cond_m = modulate(self.cond_norm1(cond), c_s1, c_sc1)
        q_img, k_img, v_img = _qkv_split(self.img_qkv(img_m), self.heads)
        q_cond, k_cond, v_cond = _qkv_split(self.cond_qkv(cond_m), self.heads)
        q_img, k_img = (self.img_q_norm(q_img), self.img_k_norm(k_img))
        q_cond, k_cond = (self.cond_q_norm(q_cond), self.cond_k_norm(k_cond))
        cos, sin = rope_cos_sin
        q_img = apply_rope(q_img, cos, sin)
        k_img = apply_rope(k_img, cos, sin)
        out_img, out_cond = joint_attention(
            q_img,
            k_img,
            v_img,
            q_cond,
            k_cond,
            v_cond,
            cond_attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            use_flash=self.use_flash,
            img_attn_mask=img_attn_mask,
        )
        B, H, N_img, Dh = q_img.shape
        N_cond = q_cond.shape[2]
        out_img = out_img.transpose(1, 2).reshape(B, N_img, H * Dh)
        out_cond = out_cond.transpose(1, 2).reshape(B, N_cond, H * Dh)
        img = img + i_g1.unsqueeze(1) * self.img_proj(out_img)
        cond = cond + c_g1.unsqueeze(1) * self.cond_proj(out_cond)
        img = img + i_g2.unsqueeze(1) * self.img_mlp(
            modulate(self.img_norm2(img), i_s2, i_sc2)
        )
        cond = cond + c_g2.unsqueeze(1) * self.cond_mlp(
            modulate(self.cond_norm2(cond), c_s2, c_sc2)
        )
        return (img, cond)


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
        use_qk_norm: bool = False,
        mlp_variant: str = "gelu",
        use_flash: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.use_flash = use_flash
        self.dim = dim
        self.mlp_variant = mlp_variant
        hidden = int(dim * mlp_ratio)
        self.hidden = hidden
        self.mod = Modulation(dim, double=False)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-06)
        self.q_norm = _qk_norm_layer(self.head_dim, use_qk_norm)
        self.k_norm = _qk_norm_layer(self.head_dim, use_qk_norm)
        if mlp_variant == "gelu":
            self.linear1 = nn.Linear(dim, dim * 3 + hidden)
            self.linear2 = nn.Linear(dim + hidden, dim)
        elif mlp_variant == "swiglu":
            self.linear1 = nn.Linear(dim, dim * 3 + 2 * hidden)
            self.linear2 = nn.Linear(dim + hidden, dim)
        else:
            raise ValueError(f"unknown mlp_variant {mlp_variant!r}")

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        rope_cos_sin,
        cond_attn_mask: Optional[torch.Tensor],
        n_img: int,
        img_attn_mask: Optional[torch.Tensor] = None,
    ):
        shift, scale, gate = self.mod(vec)
        x_m = modulate(self.norm(x), shift, scale)
        fused = self.linear1(x_m)
        if self.mlp_variant == "gelu":
            qkv, mlp_in = fused.split([self.dim * 3, self.hidden], dim=-1)
            mlp_out = F.gelu(mlp_in, approximate="tanh")
        else:
            qkv, mlp_g, mlp_u = fused.split(
                [self.dim * 3, self.hidden, self.hidden], dim=-1
            )
            mlp_out = F.silu(mlp_g) * mlp_u
        q, k, v = _qkv_split(qkv, self.heads)
        q, k = (self.q_norm(q), self.k_norm(k))
        cos, sin = rope_cos_sin
        q_img = apply_rope(q[:, :, :n_img], cos, sin)
        k_img = apply_rope(k[:, :, :n_img], cos, sin)
        q = torch.cat([q_img, q[:, :, n_img:]], dim=2)
        k = torch.cat([k_img, k[:, :, n_img:]], dim=2)
        seq_valid = None
        if cond_attn_mask is not None or img_attn_mask is not None:
            B = x.shape[0]
            if img_attn_mask is None:
                img_valid = torch.ones(B, n_img, device=x.device, dtype=torch.bool)
            else:
                img_valid = img_attn_mask.bool()
            if cond_attn_mask is None:
                n_cond = x.shape[1] - n_img
                cond_valid = torch.ones(B, n_cond, device=x.device, dtype=torch.bool)
            else:
                cond_valid = cond_attn_mask.bool()
            seq_valid = torch.cat([img_valid, cond_valid], dim=1)
        attn = _attention(
            q,
            k,
            v,
            seq_valid,
            dropout_p=self.dropout if self.training else 0.0,
            use_flash=self.use_flash,
        )
        B, H, N, Dh = attn.shape
        attn = attn.transpose(1, 2).reshape(B, N, H * Dh)
        out = self.linear2(torch.cat([attn, mlp_out], dim=-1))
        return x + gate.unsqueeze(1) * out


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_channels: int, patch_size):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-06)
        self.lin = nn.Linear(
            dim, out_channels * patch_size[0] * patch_size[1] * patch_size[2]
        )
        self.ada = nn.Linear(dim, dim * 2)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        self.patch_size = patch_size
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada(F.silu(vec)).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.lin(x)


class SegHead(nn.Module):
    def __init__(self, width, base_grid, full_shape, channels, num_classes):
        super().__init__()
        self.base_grid = tuple((int(x) for x in base_grid))
        self.c0 = int(channels[0])
        factors = [int(f) // int(b) for f, b in zip(full_shape, self.base_grid)]
        n_stages = [math.log2(f) for f in factors]
        for f, ns in zip(factors, n_stages):
            assert ns == int(ns) and f >= 1, (
                f"SegHead needs power-of-2 upsample per axis; got factors={factors}"
            )
        n = int(n_stages[0])
        assert all((int(s) == n for s in n_stages)), (
            f"SegHead needs uniform upsample across axes; got {factors}"
        )
        assert len(channels) == n + 1, (
            f"seg_channels must have {n + 1} entries (c0 + {n} deconv widths) for factor {factors}; got {len(channels)}"
        )
        self.in_proj = nn.Conv3d(width, self.c0, kernel_size=1)
        ups = []
        for i in range(n):
            cin, cout = (channels[i], channels[i + 1])
            ups.append(nn.ConvTranspose3d(cin, cout, kernel_size=2, stride=2))
            ups.append(nn.GroupNorm(min(8, cout), cout))
            ups.append(nn.SiLU())
        self.ups = nn.Sequential(*ups)
        self.head = nn.Conv3d(channels[-1], num_classes, kernel_size=1)

    def forward(
        self, tokens: torch.Tensor, grid, *, freeze_params: bool = False
    ) -> torch.Tensor:
        B = tokens.shape[0]
        Dp, Hp, Wp = grid
        width = tokens.shape[-1]
        x = tokens.view(B, Dp, Hp, Wp, width).permute(0, 4, 1, 2, 3).contiguous()
        if not freeze_params:
            x = self.in_proj(x)
            x = self.ups(x.float() if not x.is_floating_point() else x)
            return self.head(x)
        x = F.conv3d(
            x,
            self.in_proj.weight.detach(),
            self.in_proj.bias.detach() if self.in_proj.bias is not None else None,
            stride=self.in_proj.stride,
            padding=self.in_proj.padding,
            dilation=self.in_proj.dilation,
            groups=self.in_proj.groups,
        )
        x = x.float() if not x.is_floating_point() else x
        for layer in self.ups:
            if isinstance(layer, nn.ConvTranspose3d):
                x = F.conv_transpose3d(
                    x,
                    layer.weight.detach(),
                    layer.bias.detach() if layer.bias is not None else None,
                    stride=layer.stride,
                    padding=layer.padding,
                    output_padding=layer.output_padding,
                    groups=layer.groups,
                    dilation=layer.dilation,
                )
            elif isinstance(layer, nn.GroupNorm):
                x = F.group_norm(
                    x,
                    layer.num_groups,
                    layer.weight.detach() if layer.weight is not None else None,
                    layer.bias.detach() if layer.bias is not None else None,
                    layer.eps,
                )
            else:
                x = layer(x)
        return F.conv3d(
            x,
            self.head.weight.detach(),
            self.head.bias.detach() if self.head.bias is not None else None,
            stride=self.head.stride,
            padding=self.head.padding,
            dilation=self.head.dilation,
            groups=self.head.groups,
        )


class DeltaTimeEmb(nn.Module):
    def __init__(self, feat_dim: int, width: int):
        super().__init__()
        assert feat_dim % 2 == 0, "delta_time_dim must be even"
        half = feat_dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), half))
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, width), nn.SiLU(), nn.Linear(width, width)
        )

    def _phi(self, dt: torch.Tensor) -> torch.Tensor:
        ang = dt[..., None].float() * self.freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        shape = dt.shape
        flat = dt.reshape(-1)
        emb = self.mlp(self._phi(flat)) - self.mlp(self._phi(torch.zeros_like(flat)))
        return emb.view(*shape, -1)


class CrossRefFusion(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.hd = width // heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.o = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-06)
        self.scale = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, R: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        B, N, K, W = R.shape
        m = valid[..., None].to(R.dtype)
        denom = m.sum(dim=2).clamp_min(1.0)
        base = (R * m).sum(dim=2) / denom
        if K == 1:
            return base
        multi = (valid.sum(dim=2, keepdim=True) > 1).to(R.dtype)
        BN = B * N
        q = self.q(self.norm(base)).view(BN, 1, self.heads, self.hd).transpose(1, 2)
        k = self.k(R).reshape(BN, K, self.heads, self.hd).transpose(1, 2)
        v = self.v(R).reshape(BN, K, self.heads, self.hd).transpose(1, 2)
        vmask = valid.reshape(BN, K)
        escape = ~vmask.any(dim=1, keepdim=True)
        idx0 = torch.arange(K, device=R.device)[None, :] == 0
        vmask = vmask | escape & idx0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=vmask.view(BN, 1, 1, K))
        out = self.o(out.transpose(1, 2).reshape(B, N, W))
        return base + self.scale.to(base.dtype) * multi * out


class MMDiT3D(nn.Module):
    def __init__(self, cfg: MMDiTCfg):
        super().__init__()
        self.cfg = cfg
        W = cfg.width
        self.max_refs = int(cfg.max_refs)
        self.use_ref_slab = bool(cfg.use_ref_slab)
        self.patch = PatchEmbed3D(cfg.in_channels, W, cfg.patch_size, cfg.latent_shape)
        self.base_grid = (self.patch.Dp, self.patch.Hp, self.patch.Wp)
        self.Dp, self.Hp, self.Wp = self.base_grid
        self.N_img = self.Dp * self.Hp * self.Wp
        fusion_heads = cfg.ref_fusion_heads or cfg.heads
        self.delta_time_emb = DeltaTimeEmb(cfg.delta_time_dim, W)
        self.cross_ref_fusion = CrossRefFusion(W, fusion_heads)
        self.ref_type_emb = nn.Embedding(int(getattr(cfg, "n_ref_types", 3)), W)
        nn.init.zeros_(self.ref_type_emb.weight)
        head_dim = cfg.width // cfg.heads
        _cos, _sin = build_rope_with_refs(
            self.Dp,
            self.Hp,
            self.Wp,
            1,
            head_dim,
            torch.device("cpu"),
            cfg.rope_theta_base,
        )
        self.register_buffer("rope_cos", _cos, persistent=False)
        self.register_buffer("rope_sin", _sin, persistent=False)
        _cos_t, _sin_t = build_rope_with_refs(
            self.Dp,
            self.Hp,
            self.Wp,
            0,
            head_dim,
            torch.device("cpu"),
            cfg.rope_theta_base,
        )
        self.register_buffer("rope_cos_target", _cos_t, persistent=False)
        self.register_buffer("rope_sin_target", _sin_t, persistent=False)
        _half = cfg.timestep_embed_dim // 2
        _freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), _half))
        self.register_buffer("t_freqs", _freqs, persistent=False)
        self.register_buffer(
            "text_pos_ids", torch.arange(cfg.text_max_len), persistent=False
        )
        self.t_mlp = nn.Sequential(
            nn.Linear(cfg.timestep_embed_dim, W), nn.SiLU(), nn.Linear(W, W)
        )
        n_mod = cfg.num_modalities + (1 if cfg.use_uncond_modality_slot else 0)
        self.modality_emb = nn.Embedding(n_mod, W)
        self.modality_vec_proj = nn.Sequential(nn.SiLU(), nn.Linear(W, W))
        self.text_proj = nn.Linear(cfg.text_hidden, W)
        self.text_pos = nn.Embedding(cfg.text_max_len, W)
        nn.init.normal_(self.text_pos.weight, std=0.02)
        self.text_pool_proj = nn.Sequential(nn.SiLU(), nn.Linear(W, W))
        self.null_text = nn.Parameter(torch.zeros(1, 1, W))
        nn.init.normal_(self.null_text, std=0.02)
        self.null_ref = nn.Parameter(torch.zeros(1, 1, W))
        nn.init.normal_(self.null_ref, std=0.02)
        use_flash = cfg.attn_impl == "flash_varlen" and _HAS_FLASH
        if cfg.attn_impl == "flash_varlen" and (not _HAS_FLASH):
            print(
                "[mmdit] attn_impl='flash_varlen' but FA2/FA3 not importable; falling back to SDPA.",
                flush=True,
            )
        elif use_flash:
            print(f"[mmdit] using FlashAttention backend: {_FLASH_BACKEND}", flush=True)
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    W,
                    cfg.heads,
                    cfg.mlp_ratio,
                    cfg.dropout,
                    use_qk_norm=cfg.use_qk_norm,
                    mlp_variant=cfg.mlp_variant,
                    use_flash=use_flash,
                )
                for _ in range(cfg.depth_double)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    W,
                    cfg.heads,
                    cfg.mlp_ratio,
                    cfg.dropout,
                    use_qk_norm=cfg.use_qk_norm,
                    mlp_variant=cfg.mlp_variant,
                    use_flash=use_flash,
                )
                for _ in range(cfg.depth_single)
            ]
        )
        self.final = FinalLayer(W, cfg.in_channels, cfg.patch_size)
        self.seg_head_enabled = bool(cfg.seg_head_enabled)
        if self.seg_head_enabled:
            n_blocks = cfg.depth_double + cfg.depth_single
            tap = cfg.seg_tap_layer
            if tap is None or int(tap) < 0:
                tap = cfg.depth_double + cfg.depth_single // 3
            self.seg_tap_layer = int(tap)
            if not 1 <= self.seg_tap_layer <= n_blocks:
                raise ValueError(
                    f"seg_tap_layer={self.seg_tap_layer} out of range [1,{n_blocks}]"
                )
            self.seg_head = SegHead(
                W,
                self.base_grid,
                tuple(cfg.seg_full_shape),
                tuple(cfg.seg_channels),
                cfg.seg_num_classes,
            )
        else:
            self.seg_tap_layer = -1
        self.grad_checkpoint = bool(cfg.grad_checkpoint)

    @property
    def width(self) -> int:
        return self.cfg.width

    def get_null_text(self) -> torch.Tensor:
        return self.null_text

    def _embed(self, x: torch.Tensor):
        return self.patch(x)

    def _build_fused_ref(
        self,
        target_tokens,
        ref_latent,
        ref_modality_id,
        ref_dt,
        ref_valid,
        ref_type_id=None,
    ):
        B, N, W = target_tokens.shape
        device = target_tokens.device
        R = self.max_refs
        if ref_latent is not None and R > 0:
            ref_valid = (
                ref_latent.new_ones(B, R) if ref_valid is None else ref_valid
            ).bool()
            rl = ref_latent.reshape(B * R, *ref_latent.shape[2:])
            rtok, _ = self._embed(rl)
            rtok = rtok.view(B, R, N, W)
            if ref_modality_id is not None:
                rtok = rtok + self.modality_emb(ref_modality_id.long()).unsqueeze(2)
            if ref_dt is not None:
                rtok = rtok + self.delta_time_emb(ref_dt.float()).unsqueeze(2)
            if ref_type_id is not None:
                rtok = rtok + self.ref_type_emb(ref_type_id.long()).unsqueeze(2)
            rtok = rtok * ref_valid.view(B, R, 1, 1).to(rtok.dtype)
            R_stack = rtok.permute(0, 2, 1, 3).contiguous()
            valid_pos = ref_valid.view(B, 1, R).expand(B, N, R)
            fused = self.cross_ref_fusion(R_stack, valid_pos)
            any_ref = ref_valid.any(dim=1)
        else:
            fused = target_tokens.new_zeros(B, N, W)
            any_ref = torch.zeros(B, dtype=torch.bool, device=device)
        slab_valid = any_ref
        null0 = self.null_ref.to(fused.dtype).expand(B, 1, W)
        head = torch.where((~slab_valid).view(B, 1, 1), null0, fused[:, :1])
        fused = torch.cat([head, fused[:, 1:]], dim=1)
        return (fused, slab_valid)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        modality_id: torch.Tensor,
        text_drop_mask: Optional[torch.Tensor] = None,
        *,
        ref_latent: Optional[torch.Tensor] = None,
        ref_modality_id: Optional[torch.Tensor] = None,
        ref_dt: Optional[torch.Tensor] = None,
        ref_valid: Optional[torch.Tensor] = None,
        ref_type_id: Optional[torch.Tensor] = None,
        return_seg: bool = False,
        return_frozen_seg: bool = False,
    ):
        B = x.shape[0]
        device = x.device
        if x.shape[1] != self.cfg.in_channels:
            raise ValueError(
                f"target x has {x.shape[1]} channels; expected {self.cfg.in_channels}"
            )
        target_tokens, (Dp, Hp, Wp) = self._embed(x)
        n_target = target_tokens.shape[1]
        if self.use_ref_slab:
            fused_ref, slab_valid = self._build_fused_ref(
                target_tokens,
                ref_latent,
                ref_modality_id,
                ref_dt,
                ref_valid,
                ref_type_id,
            )
            img_stream = torch.cat([target_tokens, fused_ref], dim=1)
            target_valid = torch.ones(B, n_target, device=device, dtype=torch.bool)
            first_only = torch.zeros(B, n_target, device=device, dtype=torch.bool)
            first_only[:, 0] = True
            all_valid = torch.ones(B, n_target, device=device, dtype=torch.bool)
            ref_slab_mask = torch.where(slab_valid.view(B, 1), all_valid, first_only)
            img_attn_mask = torch.cat([target_valid, ref_slab_mask], dim=1)
            n_rope_refs = 1
        else:
            img_stream = target_tokens
            img_attn_mask = None
            n_rope_refs = 0
        n_img = img_stream.shape[1]
        if (Dp, Hp, Wp) == (self.Dp, self.Hp, self.Wp):
            rope_cos_sin = (
                (self.rope_cos, self.rope_sin)
                if n_rope_refs
                else (self.rope_cos_target, self.rope_sin_target)
            )
        else:
            head_dim = self.cfg.width // self.cfg.heads
            cos, sin = build_rope_with_refs(
                Dp, Hp, Wp, n_rope_refs, head_dim, device, self.cfg.rope_theta_base
            )
            rope_cos_sin = (cos, sin)
        mod_tok = self.modality_emb(modality_id).unsqueeze(1)
        text_tok = self.text_proj(text_emb)
        L = text_tok.shape[1]
        if L <= self.text_pos_ids.shape[0]:
            pos = self.text_pos_ids[:L]
        else:
            pos = torch.arange(L, device=device).clamp(max=self.cfg.text_max_len - 1)
        text_tok = text_tok + self.text_pos(pos).unsqueeze(0)
        if text_drop_mask is not None:
            null = self.null_text.expand(B, 1, -1)
            replaced = torch.zeros_like(text_tok)
            replaced[:, :1] = null
            replaced_mask = torch.zeros_like(text_mask)
            replaced_mask[:, 0] = 1
            drop = text_drop_mask.view(B, 1, 1).to(text_tok.dtype)
            text_tok = drop * replaced + (1.0 - drop) * text_tok
            drop_mask = text_drop_mask.view(B, 1).to(text_mask.dtype)
            text_mask = drop_mask * replaced_mask + (1 - drop_mask) * text_mask
        cond = torch.cat([mod_tok, text_tok], dim=1)
        cond_attn_mask = torch.cat(
            [torch.ones(B, 1, device=device, dtype=text_mask.dtype), text_mask], dim=1
        )
        ang = t[:, None].float() * self.t_freqs[None, :]
        t_emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if self.cfg.timestep_embed_dim % 2:
            t_emb = F.pad(t_emb, (0, 1))
        vec = self.t_mlp(t_emb)
        vec = vec + self.modality_vec_proj(self.modality_emb(modality_id))
        if self.cfg.pool_text_for_vec:
            m = text_mask.unsqueeze(-1).to(text_tok.dtype)
            denom = m.sum(dim=1).clamp_min(1.0)
            pooled = (text_tok * m).sum(dim=1) / denom
            vec = vec + self.text_pool_proj(pooled)
        ckpt = self.grad_checkpoint and self.training
        want_seg = return_seg and self.seg_head_enabled
        seg_tap = None
        blk_idx = 0
        for blk in self.double_blocks:
            if ckpt:
                img_stream, cond = torch.utils.checkpoint.checkpoint(
                    blk,
                    img_stream,
                    cond,
                    vec,
                    rope_cos_sin,
                    cond_attn_mask,
                    img_attn_mask,
                    use_reentrant=False,
                )
            else:
                img_stream, cond = blk(
                    img_stream,
                    cond,
                    vec,
                    rope_cos_sin,
                    cond_attn_mask,
                    img_attn_mask=img_attn_mask,
                )
            blk_idx += 1
            if want_seg and blk_idx == self.seg_tap_layer:
                seg_tap = img_stream[:, :n_target]
        x_combined = torch.cat([img_stream, cond], dim=1)
        for blk in self.single_blocks:
            if ckpt:
                x_combined = torch.utils.checkpoint.checkpoint(
                    blk,
                    x_combined,
                    vec,
                    rope_cos_sin,
                    cond_attn_mask,
                    n_img,
                    img_attn_mask,
                    use_reentrant=False,
                )
            else:
                x_combined = blk(
                    x_combined,
                    vec,
                    rope_cos_sin,
                    cond_attn_mask,
                    n_img,
                    img_attn_mask=img_attn_mask,
                )
            blk_idx += 1
            if want_seg and blk_idx == self.seg_tap_layer:
                seg_tap = x_combined[:, :n_target]
        img_out = x_combined[:, :n_target]
        out = self.final(img_out, vec)
        C = self.cfg.in_channels
        pd, ph, pw = self.cfg.patch_size
        out = out.view(B, Dp, Hp, Wp, C, pd, ph, pw)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        out = out.view(B, C, Dp * pd, Hp * ph, Wp * pw)
        if want_seg:
            if seg_tap is None:
                raise RuntimeError(
                    f"seg tap not captured (seg_tap_layer={self.seg_tap_layer})"
                )
            seg_logits = self.seg_head(seg_tap, (Dp, Hp, Wp))
            if return_frozen_seg:
                frozen_seg_logits = self.seg_head(
                    seg_tap, (Dp, Hp, Wp), freeze_params=True
                )
                return (out, seg_logits, frozen_seg_logits)
            return (out, seg_logits)
        return out


def get_mmdit_block_classes() -> set[type]:
    return {DoubleStreamBlock, SingleStreamBlock}


def _smoke():
    torch.manual_seed(0)
    cfg = MMDiTCfg(
        in_channels=16,
        latent_shape=(8, 8, 8),
        patch_size=(2, 2, 2),
        width=360,
        text_hidden=128,
        text_max_len=64,
        heads=6,
        mlp_ratio=2.0,
        depth_double=2,
        depth_single=2,
        num_modalities=4,
        timestep_embed_dim=128,
        delta_time_dim=64,
        max_refs=3,
    )
    model = MMDiT3D(cfg)
    B = 2
    ld = (8, 8, 8)
    x = torch.randn(B, 16, *ld)
    t = torch.rand(B)
    text_emb = torch.randn(B, 32, cfg.text_hidden)
    text_mask = torch.ones(B, 32, dtype=torch.long)
    text_mask[1, 20:] = 0
    mod_id = torch.tensor([1, 4])
    drop = torch.tensor([False, True])
    rl = torch.randn(B, 3, 16, *ld)
    rmod = torch.randint(0, 4, (B, 3))
    rdt = torch.zeros(B, 3)
    rvalid = torch.zeros(B, 3, dtype=torch.bool)
    rvalid[:, 0] = True
    rtype = torch.zeros(B, 3, dtype=torch.long)
    kw = dict(
        ref_latent=rl,
        ref_modality_id=rmod,
        ref_dt=rdt,
        ref_valid=rvalid,
        ref_type_id=rtype,
    )
    out = model(x, t, text_emb, text_mask, mod_id, drop, **kw)
    assert out.shape == x.shape, f"out {out.shape} != x {x.shape}"
    loss = out.pow(2).mean()
    loss.backward()
    with torch.no_grad():
        out0 = model(x, t, text_emb, text_mask, mod_id, drop, **kw)
        for n, p in model.named_parameters():
            if n.startswith(("cross_ref_fusion.", "delta_time_emb.")):
                p.add_(torch.randn_like(p) * 5.0)
        out1 = model(x, t, text_emb, text_mask, mod_id, drop, **kw)
    assert torch.allclose(out0, out1, atol=1e-05), "single-ref degeneracy broken!"
    n_param = sum((p.numel() for p in model.parameters()))
    print(
        f"[ref-fusion] patch_ch={model.patch.in_ch} max_refs={model.max_refs} out={tuple(out.shape)} degeneracy=OK loss={loss.item():.4f} params={n_param / 1000000.0:.1f}M"
    )


def _smoke_seg():
    torch.manual_seed(0)
    full = (16, 16, 16)
    base_cfg = dict(
        in_channels=16,
        latent_shape=(8, 8, 8),
        patch_size=(2, 2, 2),
        width=360,
        text_hidden=128,
        text_max_len=64,
        heads=6,
        mlp_ratio=2.0,
        depth_double=2,
        depth_single=2,
        num_modalities=4,
        timestep_embed_dim=128,
        seg_head_enabled=True,
        seg_tap_layer=2,
        seg_channels=(32, 16, 8),
        seg_num_classes=4,
        seg_full_shape=full,
    )
    B = 2
    x = torch.randn(B, 16, 8, 8, 8)
    t = torch.rand(B)
    text_emb = torch.randn(B, 32, 128)
    text_mask = torch.ones(B, 32, dtype=torch.long)
    mod_id = torch.tensor([1, 4])
    drop = torch.tensor([False, True])
    cfg = MMDiTCfg(**base_cfg)
    model = MMDiT3D(cfg)
    out, seg = model(x, t, text_emb, text_mask, mod_id, drop, return_seg=True)
    assert out.shape == x.shape
    assert seg.shape == (B, 4, *full)
    (out.pow(2).mean() + seg.pow(2).mean()).backward()
    out2 = model(x, t, text_emb, text_mask, mod_id)
    assert isinstance(out2, torch.Tensor)
    print(f"[seg] out={tuple(out.shape)} seg={tuple(seg.shape)} n_img={model.N_img}")


if __name__ == "__main__":
    _smoke()
    _smoke_seg()
