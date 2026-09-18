import os
from datetime import timedelta
import torch
import torch.distributed as dist


def init_distributed(backend: str = "nccl"):
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
        return (rank, world_size, local_rank, device)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    timeout_min = int(os.environ.get("DIST_TIMEOUT_MIN", "60"))
    init_kwargs = dict(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=timeout_min),
    )
    if torch.cuda.is_available():
        init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
    try:
        dist.init_process_group(**init_kwargs)
    except (TypeError, ValueError):
        init_kwargs.pop("device_id", None)
        dist.init_process_group(**init_kwargs)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return (rank, world_size, local_rank, device)


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
