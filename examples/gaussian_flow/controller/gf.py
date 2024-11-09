import torch
from pointrix.utils.pose import quat_to_rotmat
from pointrix.controller.base import BaseDensificationController, CONTROLER_REGISTRY
from pointrix.controller.gs import DensificationController

@CONTROLER_REGISTRY.register()
class GFDensificationController(DensificationController):
    """
    The controller class for densifying the point cloud.
    """

    def new_pos_scale(self, mask):
        scaling = self.point_cloud.get_scaling
        position = self.point_cloud.position[:, :3]
        pos_traj = self.point_cloud.position[:, 3:]
        rotation = self.point_cloud.rotation[:, :4]
        split_num = self.split_num
        
        stds = scaling[mask].repeat(split_num, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        # TODO: make new rots depend on timestamp
        rots = quat_to_rotmat(
            rotation[mask]
        ).repeat(split_num, 1, 1)
        new_pos_base = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        ) + (
            position[mask].repeat(split_num, 1)
        )
        new_pos_traj = (
            pos_traj[mask].repeat(split_num, 1)
        )
        new_pos = torch.cat([new_pos_base, new_pos_traj], dim=-1)
        new_scaling = self.point_cloud.scaling_inverse_activation(
            scaling[mask].repeat(split_num, 1) / (0.8*split_num)
        )
        return new_pos, new_scaling