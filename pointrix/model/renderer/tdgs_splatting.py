import torch
import math

from dataclasses import dataclass

from pointrix.utils.pose import Fov2ProjectMat
from pointrix.logger.writer import Logger
from .msplat import RENDERER_REGISTRY, MsplatRender
from pointrix.utils.spatial import depth_to_normal
from diff_surfel_rasterization import GaussianRasterizationSettings as TDGSettings
from diff_surfel_rasterization import GaussianRasterizer as TDGSRenderer

@RENDERER_REGISTRY.register()
class TDGSRender(MsplatRender):
    """
    A class for rendering 2d gaussian point clouds.
    """
    @dataclass
    class Config(MsplatRender.Config):
        depth_ratio: float = 0.0

    cfg: Config
    
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
                    normals=None,
                    **kwargs) -> dict:
        intrinsic_params = intrinsic_params.squeeze()
        rendered_features_split = {}
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
            
        renderer = TDGSRenderer(TDGSettings(int(height), int(width), math.tan(fovx/2.),
                                        math.tan(fovy/2.), self.bg_color, 1.0,
                                        extrinsic_matrix.transpose(0, 1),
                                        (projection_matrix @ extrinsic_matrix).transpose(0, 1),
                                        self.sh_degree,
                                        camera_center,
                                        prefiltered=False,
                                        debug=False
                                        ))
        
        rendered_image, radii, allmap = renderer(means3D=position.contiguous(),
                                        means2D=ndc.contiguous(),
                                        opacities=opacity.contiguous(),
                                        shs=shs.contiguous(),
                                        colors_precomp=None,
                                        scales=scaling.contiguous(),
                                        rotations=rotation.contiguous(),
                                        cov3D_precomp=None)
        
        render_alpha = allmap[1:2]

        normal = allmap[2:5]
        
        normal = (normal.permute(1, 2, 0) @ torch.diag(torch.tensor([-1., -1., -1.], device="cuda"))).permute(2, 0, 1)

        # normals_im = normal / normal.norm(dim=0, keepdim=True)
        normals_im = (normal + 1.) / 2.
        
        # get median depth map
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        # get expected depth map
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        
        # get depth distortion map
        render_dist = allmap[6:7]

        surf_depth = render_depth_expected * (1-self.cfg.depth_ratio) + (self.cfg.depth_ratio) * render_depth_median
        
        # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
        depth_normal = depth_to_normal(extrinsic_matrix, surf_depth, int(height), 
                                                int(width), intrinsic_params[0], intrinsic_params[1])
        depth_normal = depth_normal @ extrinsic_matrix[:3,:3].T @ torch.diag(torch.tensor([-1., -1., -1.], device="cuda"))
        depth_normal = depth_normal.permute(2,0,1)
        # remember to multiply with accum_alpha since render_normal is unnormalized.
        depth_normal = depth_normal * (render_alpha).detach()
        
        rendered_features_split['rgb'] = rendered_image
        rendered_features_split['alpha'] = render_alpha
        rendered_features_split['normal'] = normal
        rendered_features_split['normal_im'] = normals_im
        rendered_features_split['render_dist'] = render_dist
        rendered_features_split['depth'] = surf_depth
        rendered_features_split['depth_normal'] = depth_normal
        
        
        return {"rendered_features_split": rendered_features_split,
                "uv_points": ndc,
                "visibility": radii > 0,
                "radii": radii}
        
            
        

