import torch
import math
from pointrix.utils.pose import Fov2ProjectMat

def depths_to_points(w2c, depthmap, H, W, fx, fy):
    """
    The function for converting depthmap to points.
    
    Parameters
    ----------
    w2c: torch.Tensor
        The world to camera matrix.
    depthmap: torch.Tensor
        The depthmap.
    H: int  
        Height of the image.
    W: int
        Width of the image.
    fx: float
        The focal length in x direction.
    fy: float
        The focal length in y direction.
    
    Returns
    -------
    points: torch.Tensor
        The points in the world coordinate.
    """
    c2w = w2c.inverse()
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2.],
        [0, H / 2, 0, (H) / 2.],
        [0, 0, 0, 1]]).float().cuda().T
    
    fovx = 2*math.atan(W/(2*fx))
    fovy = 2*math.atan(H/(2*fy))
    
    projection_matrix = Fov2ProjectMat(fovx, fovy).to(depthmap.device).transpose(0,1)
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(w2c, depth, H, W, fx, fy):
    """
    The function for converting depthmap to normal map.
    
    Parameters
    ----------
    w2c: torch.Tensor
        The world to camera matrix.
    depthmap: torch.Tensor
        The depthmap.
    H: int  
        Height of the image.
    W: int
        Width of the image.
    fx: float
        The focal length in x direction.
    fy: float
        The focal length in y direction.
    
    Returns
    -------
    normal_map: torch.Tensor
        The normal map.
    """
    points = depths_to_points(w2c, depth, H, W, fx, fy).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output