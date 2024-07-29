import os
import torch
import numpy as np
from copy import deepcopy
from PIL import Image
from torch import Tensor
from pathlib import Path
from random import randint
from jaxtyping import Float
from abc import abstractmethod
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Tuple, Any, Dict, Union, List, Optional

from ..utils.registry import Registry
from ..utils.config import parse_structured
from .utils.dataprior import CamerasPrior, CameraPrior

DATA_SET_REGISTRY = Registry("DATA_SET", modules=["pointrix.dataset"])
DATA_SET_REGISTRY.__doc__ = ""

@DATA_SET_REGISTRY.register()
class BaseDataset(Dataset):
    """
    Basic dataset used in Datapipeline.
    """

    @dataclass
    class Config:
        data_path: str = "data"
        data_set: str = "BaseImageDataset"
        meta_dirs_dict: Dict[str, str] = field(default_factory=lambda: dict({"image": "images"}))
        cached_metadata: bool = True
        white_bg: bool = False
        enable_camera_training: bool = False
        scale: float = 1.0
        trainable_camera: bool = False
        device: str = "cuda"

    def __init__(self, cfg, split) -> None:
        self.cfg = parse_structured(self.Config, cfg)
        self.data_root = Path(self.cfg.data_path)
        self.split = split
        self.scale = self.cfg.scale
        self.meta_dirs_dict = self.cfg.meta_dirs_dict
        self.enable_camera_training = self.cfg.enable_camera_training
        self.cached_metadata = self.cfg.cached_metadata
        self.device = self.cfg.device

        self.camera_list, self.meta, self.pointcloud = self._load_data_list(split)

        self.cameras = CamerasPrior(self.camera_list)
        self.radius = self.cameras.radius.detach().cpu().numpy() * 1.1
        self.bg = [1., 1., 1.] if self.cfg.white_bg else [0., 0., 0.]
        self.meta = self._transform_metadata(self.meta, split)
        
        self.frame_idx_list = np.arange(len(self.camera_list))

    def _load_data_list(self, split):
        """
        The foundational function for formating the data

        Parameters
        ----------
        split: The split of the data.
        """
        camera = self._load_camera(split=split)
        metadata = self._load_metadata(split=split)
        pointcloud = self._load_pointcloud()
        return camera, metadata, pointcloud

    @abstractmethod
    def _load_camera(self, split) -> List[CameraPrior]:
        """
        The function for loading the camera typically requires user customization.

        Parameters
        ----------
        split: The split of the data.
        """
        raise NotImplementedError

    def _load_pointcloud(self) -> dict:
        """
        The function for loading the Pointcloud for initialization of gaussian model.
        """
        return None

    def _load_metadata(self, split) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _transform_metadata(self, meta, split):
        raise NotImplementedError

    # TODO: full init
    def __len__(self):
        return len(self.camera_list)

    def __getitem__(self, idx):
        camera = self.camera_list[idx]
        meta = self.meta[idx]
        frame_idx = self.frame_idx_list[idx]
        for key, value in meta.items():
            meta[key] = value.to(self.device)
        return {
            **meta,
            "camera": camera,
            "frame_idx": frame_idx,
            "camera_idx": int(camera.idx),
            "height": int(camera.image_height),
            "width": int(camera.image_width),
        }

