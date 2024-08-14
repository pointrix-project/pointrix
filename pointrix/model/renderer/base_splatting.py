import torch
import math
from dataclasses import dataclass

from ...utils.base import BaseObject
from ...utils.pose import Fov2ProjectMat
from ...logger import Logger
from .msplat import RENDERER_REGISTRY
from .msplat import MsplatRender

from diff_gaussian_rasterization import GaussianRasterizationSettings as GSettings
from diff_gaussian_rasterization import GaussianRasterizer as GSRenderer

@RENDERER_REGISTRY.register()
class GaussianSplattingRender(MsplatRender):
    """
    A class for rendering point clouds using Gaussian splatting.
    """
    def setup(self, white_bg, device, **kwargs):
        self.sh_degree = 0
        self.device = device
        bg_color = [1, 1, 1] if white_bg else [0, 0, 0]
        self.bg_color = torch.tensor(bg_color, dtype=torch.float32, device=self.device)

    def render_iter(self,
                    height,
                    width,
                    extrinsic_matrix,
                    intrinsic_params,
                    camera_center,
                    position,
                    opacity,
                    scaling,
                    rotation,
                    shs,
                    **kwargs) -> dict:
        """
        Render the point cloud for one iteration

        Parameters
        ----------
        height : int
            The height of the image.
        width : int
            The width of the image.
        extrinsic_matrix : torch.Tensor
            The extrinsic matrix.
        intrinsic_params : list
            The intrinsic parameters.
        camera_center : torch.Tensor
            The camera center.
        position : torch.Tensor
            The position of the point cloud.
        opacity : torch.Tensor
            The opacity of the point cloud.
        scaling : torch.Tensor
            The scaling of the point cloud.
        rotation : torch.Tensor
            The rotation of the point cloud.
        shs : torch.Tensor
            The spherical harmonics.
        
        Returns
        -------
        dict
            The rendered point cloud.
        """
        fx, fy = intrinsic_params[0], intrinsic_params[1]

        fovx = 2*math.atan(width/(2*fx))
        fovy = 2*math.atan(height/(2*fy))
        
        projection_matrix = Fov2ProjectMat(fovx, fovy).to(extrinsic_matrix.device)
        ndc = torch.zeros_like(position,
                                dtype=position.dtype,
                                requires_grad=True,
                                device=self.device)
        try:
            ndc.retain_grad()
        except:
            Logger.warn("NDC does not have grad")

        renderer = GSRenderer(GSettings(int(height), int(width), math.tan(fovx/2.),
                                        math.tan(fovy/2.), self.bg_color, 1.0,
                                        extrinsic_matrix.transpose(0, 1),
                                        (projection_matrix @ extrinsic_matrix).transpose(0, 1),
                                        self.sh_degree,
                                        camera_center,
                                        prefiltered=False,
                                        debug=False,
                                        ))
        rendered_features_split = {}
        rendered_image, radii = renderer(position.contiguous(),
                                        ndc.contiguous(),
                                        opacity.contiguous(),
                                        shs.contiguous(),
                                        None,
                                        scaling.contiguous(),
                                        rotation.contiguous())
        rendered_features_split['rgb'] = rendered_image

        return {"rendered_features_split": rendered_features_split,
                "uv_points": ndc,
                "visibility": radii > 0,
                "radii": radii}
