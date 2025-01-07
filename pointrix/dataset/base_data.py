import os
import torch
import numpy as np
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
from .utils.dataprior import CamerasPrior, CameraPrior, PointsPrior

DATA_SET_REGISTRY = Registry("DATA_SET", modules=["pointrix.dataset"])
DATA_SET_REGISTRY.__doc__ = ""

@DATA_SET_REGISTRY.register()
class BaseDataset(Dataset):
    """
    Basic dataset used in Datapipeline.
    """
    @dataclass
    class Config:
        """
        Parameters
        ----------
        data_path: str
            The path to the data
        data_set: str
            The dataset used in the pipeline, indexed in DATA_SET_REGISTRY
        observed_data_dirs_dict: Dict[str, str]
            The observed data directories, e.g., {"image": "images"}, which means the variable image is stored in "images" directory
        cached_observed_data: bool
            Whether the observed data is cached
        white_bg: bool
            Whether the background is white
        enable_camera_training: bool
            Whether the camera is trainable
        scale: float
            The image scale of the dataset
        device: str
            The device used in the pipeline
        """
        data_path: str = "data"
        data_set: str = "BaseImageDataset"
        observed_data_dirs_dict: Dict[str, str] = field(default_factory=lambda: dict({"image": "images"}))
        cached_observed_data: bool = True
        white_bg: bool = False
        enable_camera_training: bool = False
        scale: float = 1.0
        device: str = "cuda"

    def __init__(self, cfg:Config, split:str) -> None:
        self.cfg = parse_structured(self.Config, cfg)
        self.data_root = Path(self.cfg.data_path)
        self.split = split
        self.scale = self.cfg.scale
        self.observed_data_dirs_dict = self.cfg.observed_data_dirs_dict
        self.enable_camera_training = self.cfg.enable_camera_training
        self.cached_observed_data = self.cfg.cached_observed_data
        self.device = self.cfg.device

        self.camera_list, self.observed_data, self.pointcloud = self.load_data_list(split)

        self.cameras = CamerasPrior(self.camera_list)
        self.radius = self.cameras.radius.detach().cpu().numpy() * 1.1
        self.background_color = [1., 1., 1.] if self.cfg.white_bg else [0., 0., 0.]
        self.observed_data = self.transform_observed_data(self.observed_data, split)
        
        self.frame_idx_list = np.arange(len(self.camera_list))

    def load_data_list(self, split: str) -> Tuple[List[CameraPrior], Dict[str, Any], PointsPrior]:
        """
        The foundational function for formating the data

        Parameters
        ----------
        split: The split of the data.
        
        Returns
        -------
        camera: List[CameraPrior]
            The list of cameras prior
        observed_data: Dict[str, Any]
            The observed data
        pointcloud: PointsPrior
            The pointcloud for the gaussian model.
        """
        camera = self.load_camera_prior(split=split)
        observed_data = self.load_observed_data(split=split)
        pointcloud = self.load_pointcloud_prior()
        return camera, observed_data, pointcloud

    @abstractmethod
    def load_camera_prior(self, split:str) -> List[CameraPrior]:
        """
        The function for loading the camera typically requires user customization.

        Parameters
        ----------
        split: The split of the data.
        """
        raise NotImplementedError

    def load_pointcloud_prior(self) -> PointsPrior:
        """
        The function for loading the Pointcloud for initialization of gaussian model.
        """
        return None

    def load_observed_data(self, split:str) -> Dict[str, Any]:
        """
        The function for loading the observed_data, such as image, depth, normal, etc.
        
        Parameters
        ----------
        split: The split of the data
        
        Returns
        -------
        observed_data: Dict[str, Any]
            The observed data for the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def transform_observed_data(self, observed_data:Dict, split:str):
        """
        The function for transforming the observed_data.
        
        Parameters
        ----------
        observed_data: Dict[str, Any]
            The observed_data for the dataset.
        
        Returns
        -------
        observed_data: Dict[str, Any]
            The transformed observed_data.
        """
        raise NotImplementedError

    # TODO: full init
    def __len__(self):
        return len(self.camera_list)

    def __getitem__(self, idx):
        camera = self.camera_list[idx]
        observed_data = self.observed_data[idx]
        frame_idx = self.frame_idx_list[idx]
        for key, value in observed_data.items():
            observed_data[key] = value.to(self.device)
        return {
            **observed_data,
            "camera": camera,
            "frame_idx": frame_idx,
            "camera_idx": int(camera.idx),
            "height": int(camera.image_height),
            "width": int(camera.image_width)
        }

