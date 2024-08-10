import torch
import msplat

from typing import List
from dataclasses import dataclass
from ...utils.base import BaseObject
from ...utils.registry import Registry
from .utils.renderer_utils import RenderFeatures
RENDERER_REGISTRY = Registry("RENDERER", modules=["pointrix.model.renderer"])


@RENDERER_REGISTRY.register()
class MsplatRender(BaseObject):
    """
    A class for rendering point clouds using msplat.

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

    def setup(self, white_bg, device, **kwargs):
        self.sh_degree = 0
        self.device = device
        super().setup(white_bg, device, **kwargs)
        self.bg_color = 1. if white_bg else 0.

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
        direction = (position.cuda() -
                     camera_center.repeat(position.shape[0], 1).cuda())
        direction = direction / direction.norm(dim=1, keepdim=True)
        
        # set sh mark for sh warm-up
        sh_coeff = shs.permute(0, 2, 1)
        sh_mask = torch.zeros_like(sh_coeff)
        sh_mask[..., :(self.sh_degree + 1)**2] = 1.0
        
        # NOTE: compute_sh is not sh2rgb
        rgb = msplat.compute_sh(sh_coeff * sh_mask, direction)
        rgb = (rgb + 0.5).clamp(min=0.0)
        
        extrinsic_matrix = extrinsic_matrix[:3, :]

        (uv, depth) = msplat.project_point(
            position,
            intrinsic_params,
            extrinsic_matrix,
            width, height, nearest=0.2)

        visible = depth != 0

        # compute cov3d
        cov3d = msplat.compute_cov3d(scaling, rotation, visible)

        # ewa project
        (conic, radius, tiles_touched) = msplat.ewa_project(
            position,
            cov3d,
            intrinsic_params,
            extrinsic_matrix,
            uv,
            width,
            height,
            visible
        )

        # sort
        (gaussian_ids_sorted, tile_range) = msplat.sort_gaussian(
            uv, depth, width, height, radius, tiles_touched
        )

        Render_Features = RenderFeatures(
            rgb=rgb, depth=depth) if self.cfg.render_depth else RenderFeatures(rgb=rgb)
        render_features = Render_Features.combine()

        ndc = torch.zeros_like(uv, requires_grad=True)
        try:
            ndc.retain_grad()
        except:
            raise ValueError("ndc does not have grad")

        # alpha blending
        rendered_features = msplat.alpha_blending(
            uv, conic, opacity, render_features,
            gaussian_ids_sorted, tile_range, self.bg_color, width, height, ndc
        )
        rendered_features_split = Render_Features.split(rendered_features)

        return {"rendered_features_split": rendered_features_split,
                "uv_points": ndc,
                "visibility": radius > 0,
                "radii": radius
                }

    def render_batch(self, render_dict: dict, batch: List[dict]) -> dict:
        """
        Render the batch of point clouds.

        Parameters
        ----------
        render_dict : dict
            The render dictionary.
        batch : List[dict]
            The batch data.

        Returns
        -------
        dict
            The rendered image, the viewspace points, 
            the visibility filter, the radii, the xyz, 
            the color, the rotation, the scales, and the xy.
        """
        rendered_features = {}
        uv_points = []
        visibilitys = []
        radii = []
    
        batched_render_keys = ["extrinsic_matrix", "intrinsic_params", "camera_center"]

        for i, b_i in enumerate(batch):
            for key in render_dict.keys():
                if key not in batched_render_keys:
                    b_i[key] = render_dict[key]
                else:
                    b_i[key] = render_dict[key][i, ...]
            render_results = self.render_iter(**b_i)
            for feature_name in render_results["rendered_features_split"].keys():
                if feature_name not in rendered_features:
                    rendered_features[feature_name] = []
                rendered_features[feature_name].append(
                    render_results["rendered_features_split"][feature_name])

            uv_points.append(render_results["uv_points"])
            visibilitys.append(
                render_results["visibility"].unsqueeze(0))
            radii.append(render_results["radii"].unsqueeze(0))

        for feature_name in rendered_features.keys():
            rendered_features[feature_name] = torch.stack(
                rendered_features[feature_name], dim=0)

        return {**rendered_features,
                "uv_points": uv_points,
                "visibility": torch.cat(visibilitys).any(dim=0),
                "radii": torch.cat(radii, 0).max(dim=0).values
                }

    def update_sh_degree(self, step):
        """
        Update the spherical harmonics degree in render

        Parameters
        ----------
        step : int
            The current training step.
        """
        if step % self.cfg.update_sh_iter == 0:
            if self.sh_degree < self.cfg.max_sh_degree:
                self.sh_degree += 1

    def load_state_dict(self, state_dict):
        """
        Load the state dictionary of render.

        Parameters
        ----------
        state_dict : dict
            The state dictionary
        """
        self.sh_degree = state_dict["sh_degree"]

    def state_dict(self):
        return {"sh_degree": self.sh_degree}