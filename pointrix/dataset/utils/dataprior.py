import math
import json
import logging
import functools
import math
import torch
import numpy as np
from torch import nn
from torch import Tensor
from jaxtyping import Float
from typing import Union, List
from numpy.typing import NDArray
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from plyfile import PlyData, PlyElement

from ...utils.pose import ConcatRT, ViewScaling, GetCamcenter

@dataclass()
class CameraPrior:
    """
    Camera prior info used in data Pipeline

    Parameters
    ----------
    idx: int
        The index of the camera.
    width: int
        The width of the image.
    height: int
        The height of the image.
    R: Float[Tensor, "3 3"]
        The rotation matrix of the camera.
    T: Float[Tensor, "3 1"]
        The translation vector of the camera.
    fx: float
        The focal length of the camera in x direction.
    fy: float
        The focal length of the camera in y direction.
    cx: float
        The center of the image in x direction.
    cy: float
        The center of the image in y direction.
    rgb_file_name: str
        The path of the image.
    scene_scale: float
        The scale of the scene.

    Notes
    -----
    fx, fy, cx, cy and fovX, fovY are mutually exclusive. 
    If fx, fy, cx, cy are provided, fovX, fovY will be calculated from them. 
    If fovX, fovY are provided, fx, fy, cx, cy will be calculated from them.

    Examples
    --------
    >>> idx = 1
    >>> width = 800
    >>> height = 600
    >>> R = np.eye(3)
    >>> T = np.zeros(3)
    >>> focal_length_x = 800
    >>> focal_length_y = 800
    >>> camera = Camera_Prior(idx=idx, R=R, T=T, width=width, height=height, rgb_file_name='1_rgb.png',
                        fx=focal_length_x, fy=focal_length_y, cx=width/2, cy=height/2, scene_scale=1.0)
    """
    idx: int
    image_width: int
    image_height: int
    R: Union[Float[Tensor, "3 3"], NDArray]
    T: Union[Float[Tensor, "3 1"], NDArray]
    fx: Union[float, None] = None
    fy: Union[float, None] = None
    cx: Union[float, None] = None
    cy: Union[float, None] = None
    rgb_file_name: str = None
    rgb_file_path: str = None
    scene_scale: float = 1.0
    device: str = 'cuda'
    
    def __post_init__(self):
        if not isinstance(self.R, Tensor):
            self.R = torch.tensor(self.R)
        if not isinstance(self.T, Tensor):
            self.T = torch.tensor(self.T)
        self.extrinsic_matrix = ViewScaling(ConcatRT(self.R, self.T), scale=self.scene_scale).to(self.device)
        self.intrinsic_params = torch.tensor([self.fx, self.fy, self.cx, self.cy], dtype=torch.float32).to(self.device)
        self.camera_center = GetCamcenter(self.extrinsic_matrix).to(self.device)
        
@dataclass()

class PointsPrior:
    """
    Point cloud initialization used in data Pipeline
    """
    
    positions: Union[Float[Tensor, "N 3"], NDArray, None]=None
    colors: Union[Float[Tensor, "N 3"], NDArray, None]=None
    normals: Union[Float[Tensor, "N 3"], NDArray, None]=None
    
    def save_ply(self, path):
        # Define the structured array dtype for vertices
        attribute = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue']
        type = ['f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'u1', 'u1', 'u1']
        dtype = [(a, t) for a, t in zip(attribute, type)]
        # Combine xyz, normals, and rgb into one structured array
        ply = np.zeros(self.positions.shape[0], dtype=dtype)
        for i, (pos, r) in enumerate(zip(self.positions, self.colors)):
            ply[i] = (*pos, *self.normals[i], *r)

        # Create and write the PlyData object
        ply_data = PlyData([PlyElement.describe(ply, 'vertex')])
        ply_data.write(path)
        
    def read_ply(self, path):
        plyData = PlyData.read(path)
        coordinates_attributes = ['x', 'y', 'z']
        color_attributes = ['red', 'green', 'blue']
        normal_attributes = ['nx', 'ny', 'nz']
        assert all([attr in plyData['vertex'] for attr in coordinates_attributes]), "Missing coordinate attributes"
        assert all([attr in plyData['vertex'] for attr in color_attributes]), "Missing color attributes"
        assert all([attr in plyData['vertex'] for attr in normal_attributes]), "Missing normal attributes"
        
        self.positions = np.stack([plyData['vertex'][a] for a in coordinates_attributes], axis=1)
        self.colors = np.stack([plyData['vertex'][a] for a in color_attributes], axis=1) / 255.
        self.normals = np.stack([plyData['vertex'][a] for a in normal_attributes], axis=1)

class CamerasPrior:
    """
    Cameras class used in Pointrix, which are used to generate camera paths.

    Parameters
    ----------
    camera_list: List[CameraPrior]
        The list of the CameraPrior.

    Examples
    --------
    >>> width = 800
    >>> height = 600
    >>> R = np.eye(3)
    >>> T = np.zeros(3)
    >>> focal_length_x = 800
    >>> focal_length_y = 800
    >>> camera1 = CameraPrior(idx=1, R=R, T=T, width=width, height=height, rgb_file_name='1_rgb.png',
                        fx=focal_length_x, fy=focal_length_y, cx=width/2, cy=height/2, scene_scale=1.0)
    >>> camera2 = CameraPrior(idx=2, R=R, T=T, width=width, height=height, rgb_file_name='1_rgb.png', 
                        fx=focal_length_x, fy=focal_length_y, cx=width/2, cy=height/2, scene_scale=1.0)
    >>> cameras = CamerasPrior([camera1, camera2])
    """

    def __init__(self, camera_list: List[CameraPrior]):
        self.camera_type = camera_list[0].__class__
        self.cameras = camera_list
        self.num_cameras = len(camera_list)
        self.Rs = torch.stack([cam.R for cam in camera_list], dim=0)
        self.Ts = torch.stack([cam.T for cam in camera_list], dim=0)
        self.camera_centers = torch.stack(
            [cam.camera_center for cam in camera_list], dim=0)  # (N, 3)

        self.radius = self.get_radius()

    def __len__(self):
        return self.num_cameras

    def __getitem__(self, index):
        return self.cameras[index]

    def get_radius(self):
        """
        Get the path radius of the cameras.

        Returns
        -------
        camera_radius: float
            The radius of the cameras.
        """
        cams_center = torch.mean(self.camera_centers, dim=0, keepdims=True)
        dist = torch.linalg.norm(self.camera_centers - cams_center, dim=1)
        camera_radius = torch.max(dist) 
        return camera_radius