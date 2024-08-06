import torch
import math
from dataclasses import dataclass

from ...utils.base import BaseObject
from ...utils.pose import Fov2ProjectMat
from ...logger import Logger
from .msplat import RENDERER_REGISTRY, MsplatRender

from gsplat.rendering import rasterization

@RENDERER_REGISTRY.register()
class GsplatRender(MsplatRender):
    """
    A class for rendering point clouds using Gaussian splatting.

    Config
    ------
    update_sh_iter : int, optional
        The iteration to update the spherical harmonics degree, by default 1000.
    max_sh_degree : int, optional
        The maximum spherical harmonics degree, by default 3.
    render_depth : bool, optional
        Whether to render the depth or not, by default False.
    """
    @dataclass
    class Config:
        update_sh_iter: int = 1000
        max_sh_degree: int = 3
        render_depth: bool = False
    
    cfg: Config
    
    """
    A class for rendering point clouds using Gaussian splatting.

    Parameters
    ----------
    white_bg : bool
        Whether the background is white or not.
    device : str
        The device to use.
    """
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
        intrinsic_params : torch.Tensor
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
        """
    
        shs = shs.unsqueeze(0)

        fx, fy = intrinsic_params[0], intrinsic_params[1]
        cx, cy = intrinsic_params[2], intrinsic_params[3]

        render_mode="RGB+ED" if self.cfg.render_depth else "RGB"

        Ks = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=self.device).unsqueeze(0)
        Es = extrinsic_matrix.unsqueeze(0)
        renders, alphas, info = rasterization(
            means=position,
            quats=rotation,
            scales=scaling,
            opacities=opacity.squeeze(-1),
            colors=shs,
            viewmats=Es,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode="classic",
            sh_degree=self.sh_degree,
            render_mode=render_mode
        )
        rendered_features_split = {}
        assert (
            "means2d" in info
        ), "The 2D means of the Gaussians is required but missing."
        if info["means2d"].requires_grad:
            info["means2d"].retain_grad()

        if renders.shape[-1] == 4:
            colors, depths = renders[..., 0:3], renders[..., 3:4]
        else:
            colors, depths = renders, None

        rendered_features_split['rgb'] = colors[0].permute(2, 0, 1)
        if self.cfg.render_depth:
            rendered_features_split['depth'] = depths[0].permute(2, 0, 1)
        radii = info["radii"][0]

        return {"rendered_features_split": rendered_features_split,
                "uv_points": info["means2d"],
                "visibility": radii > 0,
                "radii": radii} 
