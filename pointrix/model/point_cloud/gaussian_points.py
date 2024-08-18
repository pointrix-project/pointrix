import torch
from torch import nn
from dataclasses import dataclass

from .points import PointCloud, POINTSCLOUD_REGISTRY
from .utils.point_utils import (
    sigmoid_inv,
    gaussian_point_init
)

@POINTSCLOUD_REGISTRY.register()
class GaussianPointCloud(PointCloud):
    """
    A class for Gaussian point cloud.

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

        scales, rots, opacities, features_rest = gaussian_point_init(
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
        scales, rots, opacities, features_rest = gaussian_point_init(
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
