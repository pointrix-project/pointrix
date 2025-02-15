import torch
from torch import nn
from dataclasses import dataclass

from pointrix.model.point_cloud import PointCloud, POINTSCLOUD_REGISTRY
from pointrix.model.utils.gaussian_utils import (
    sigmoid_inv
)

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .utils.point_utils import k_nearest_sklearn

sigmoid_inv = lambda x: torch.log(x/(1-x))

def tdgaussian_point_init(position, max_sh_degree, opc_init_scale=0.1):
    num_points = len(position)    
    distances= k_nearest_sklearn(position.data, 3)
    distances = torch.from_numpy(distances)
    avg_dist = distances.mean(dim=-1, keepdim=True)

    scales = torch.log(avg_dist).repeat(1, 2)
    # Efficiently create a batch of identity quaternions
    rots = torch.eye(4)[:1].repeat(num_points, 1)  
    opacities = sigmoid_inv(opc_init_scale * torch.ones((num_points, 1), dtype=torch.float32))
    features_rest = torch.zeros(
        (num_points, (max_sh_degree+1) ** 2 - 1, 3),
        dtype=torch.float32
    )

    return scales, rots, opacities, features_rest

@POINTSCLOUD_REGISTRY.register()
class TDGaussianPointCloud(PointCloud):
    """
    A class for 2d Gaussian point cloud.

    Parameters
    ----------
    PointCloud : PointCloud
        The point cloud for initialisation.
    """
    @dataclass
    class Config(PointCloud.Config):
        max_sh_degree: int = 3
        lambda_dssim: float = 0.2

    cfg: Config

    def setup(self, point_cloud=None):
        super().setup(point_cloud)
        # Activation funcitons
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = sigmoid_inv
        self.rotation_activation = torch.nn.functional.normalize

        scales, rots, opacities, features_rest = tdgaussian_point_init(
            position=self.position,
            max_sh_degree=self.cfg.max_sh_degree,
        )

        fused_color = self.features.unsqueeze(1)
        self.features = (
            nn.Parameter(
                fused_color.contiguous().requires_grad_(True)
            )
        ) 
        self.register_atribute("features_rest", features_rest)
        self.register_atribute("scaling", scales)
        self.register_atribute("rotation", rots)
        self.register_atribute("opacity", opacities)
        
    def re_init(self, num_points):
        super().re_init(num_points)
        fused_color = self.features.unsqueeze(1)
        self.features = (
            nn.Parameter(
                fused_color.contiguous().requires_grad_(True)
            )
        )
        scales, rots, opacities, features_rest = tdgaussian_point_init(
            position=self.position,
            max_sh_degree=self.cfg.max_sh_degree,
        )
        self.register_atribute("features_rest", features_rest)
        self.register_atribute("scaling", scales)
        self.register_atribute("rotation", rots)
        self.register_atribute("opacity", opacities)

    @property
    def get_opacity(self):
        return self.opacity_activation(self.opacity)

    @property
    def get_scaling(self):
        return self.scaling_activation(self.scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self.rotation)

    @property
    def get_shs(self):
        return torch.cat([
            self.features, self.features_rest,
        ], dim=1)

    @property
    def get_position(self):
        return self.position
