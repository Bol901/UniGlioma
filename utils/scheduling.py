import math
import torch


def make_ts(steps, device, kind="power", t_min=0.001, rho=3.0):
    u = torch.linspace(0, 1, steps + 1, device=device)
    if kind == "uniform":
        ts = 1.0 - u
    elif kind == "power":
        ts = (1.0 - u) ** rho
        ts = (1.0 - t_min) * ts + t_min
    elif kind == "karras":
        a, b = (t_min ** (1.0 / rho), 1.0 ** (1.0 / rho))
        ts = (a + (b - a) * (1.0 - u)) ** rho
    elif kind == "cosine":
        s = 0.008

        def f(x):
            return torch.cos((x + s) / (1 + s) * math.pi / 2) ** 2

        ts = f(u) * (1 - t_min) + t_min
    elif kind == "exp":
        ts = (1 - t_min) * t_min**u + t_min
    else:
        raise ValueError("unknown schedule kind")
    ts[-1] = t_min
    ts[0] = 1.0
    return ts


def grid_to_tokens(x):
    B, C, D, H, W = x.shape
    return x.view(B, C, D * H * W).transpose(1, 2).contiguous()


def tokens_to_grid(tokens, shape_info):
    B, N, C = tokens.shape
    D, H, W = shape_info
    return tokens.transpose(1, 2).contiguous().view(B, C, D, H, W)
