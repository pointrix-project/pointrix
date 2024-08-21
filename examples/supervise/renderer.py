import torch
import msplat

from pointrix.model.renderer.utils.renderer_utils import RenderFeatures
from pointrix.model.renderer.msplat import RENDERER_REGISTRY, MsplatRender


@RENDERER_REGISTRY.register()
class MsplatNormalRender(MsplatRender):
    """
    A class for rendering point clouds using DPTR.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    white_bg : bool
        Whether the background is white or not.
    device : str
        The device to use.
    update_sh_iter : int, optional
        The iteration to update the spherical harmonics degree, by default 1000.
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
                    normals=None,
                    **kwargs) -> dict:

        direction = (position -
                     camera_center.repeat(position.shape[0], 1))
        direction = direction / direction.norm(dim=1, keepdim=True)
        rgb = msplat.compute_sh(shs.permute(0, 2, 1), direction)
        extrinsic_matrix = extrinsic_matrix[:3, :]

        (uv, depth) = msplat.project_point(
            position,
            intrinsic_params,
            extrinsic_matrix,
            width, height)

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

        Render_Features = RenderFeatures(rgb=rgb, normal=normals) if normals is not None else RenderFeatures(rgb=rgb)
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

        if normals is not None:
            normals = rendered_features_split["normal"]
            
            # convert normals from [-1,1] to [0,1]
            normals_im = normals / normals.norm(dim=0, keepdim=True)
            normals_im = (normals_im + 1) / 2
            
            rendered_features_split["normal"] = normals_im

        return {"rendered_features_split": rendered_features_split,
                "uv_points": ndc,
                "visibility": radius > 0,
                "radii": radius
                }
