"""Single-process (RTX 3060, 1xH100) and torchrun/DDP (8xH100) setup."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")

    @property
    def is_dist(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed(timeout_min: int = 30) -> DistInfo:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            device = torch.device("cuda", local)
        else:
            device = torch.device("cpu")
        # NCCL on Linux; gloo fallback (Windows / CPU tests). Async error handling turns hung
        # collectives into errors after the timeout, so the supervisor can restart the job.
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        backend = "nccl" if device.type == "cuda" and sys.platform != "win32" else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_min))
        return DistInfo(rank, local, world, device)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    return DistInfo(0, 0, 1, device)


def barrier(d: DistInfo) -> None:
    if d.is_dist:
        dist.barrier()


def all_reduce_(t: torch.Tensor, d: DistInfo, op: str = "sum") -> torch.Tensor:
    if d.is_dist:
        dist.all_reduce(t, op={"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX, "min": dist.ReduceOp.MIN}[op])
    return t


def cleanup(d: DistInfo) -> None:
    if d.is_dist and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
