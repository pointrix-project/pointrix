import os
import torch
from torch import nn

from dataclasses import dataclass
from typing import Optional, Union
from omegaconf import DictConfig

from ..utils.config import parse_structured

def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")

class BaseObject:
    @dataclass
    class Config:
        pass

    cfg: Config  # add this to every subclass of BaseObject to enable static type checking

    def __init__(
        self, 
        cfg: Optional[Union[dict, DictConfig]] = None, *args, **kwargs
    ) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.device = get_device()
        self.setup(*args, **kwargs)

    def setup(self, *args, **kwargs) -> None:
        pass
    
class BaseModule(BaseObject, nn.Module):
    pass