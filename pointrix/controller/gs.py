import torch
from torch import Tensor
from torch.optim import Optimizer
from dataclasses import dataclass
from typing import Tuple, List

from pointrix.utils.config import C

from .base import BaseDensificationController, CONTROLER_REGISTRY

from ..model.utils.gaussian_utils import sigmoid_inv
from ..utils.pose import quat_to_rotmat

@CONTROLER_REGISTRY.register()
class DensificationController(BaseDensificationController):
    """
    The controller class for densifying the point cloud.
    """
    @dataclass
    class Config(BaseDensificationController.Config):
        """
        Parameters
        ----------
        split_num : int
            The number of points to split.
        control_module : str
            The control module.
        percent_dense : float
            The percentage of dense points.
        opacity_reset_interval : int
            The interval to reset the opacity.
        densify_grad_threshold : float
            The controller gradient threshold.
        min_opacity : float
            The minimum opacity.
        normalize_grad : bool
            Whether to normalize the gradient.
        """
        # Densification
        split_num: int = 2
        control_module: str = "point_cloud"
        percent_dense: float = 0.01
        opacity_reset_interval: int = 3000
        densify_grad_threshold: float = 0.0002
        min_opacity: float = 0.005
        normalize_grad: bool = True

    cfg: Config

    def setup(self, optimizer, model, **kwargs) -> None:

        super().setup(optimizer, model, **kwargs)

        # TODO: Not sure if this is needed
        if len(optimizer.param_groups) > 1:
            self.base_param_settings = {
                'params': torch.tensor([0.0], dtype=torch.float)
            }
            self.base_param_settings.update(**self.optimizer.defaults)
        else:
            self.base_param_settings = None  # type: ignore

        self.cameras_extent = kwargs.get("cameras_extent", 1.0)

        self.width = model.training_camera_model.image_width
        self.height = model.training_camera_model.image_height
        # Densification setup
        num_points = len(self.point_cloud)
        self.max_radii = torch.zeros(num_points).to(self.device)
        self.percent_dense = self.cfg.percent_dense
        self.opacity_deferred = False

    def reset_opacity(self, reset_scele=0.01) -> None:
        """
        Reset the opacity of the point cloud. 
        
        Parameters
        ----------
        reset_scele : float
            The reset scale.
        """
        opc = self.point_cloud.get_opacity
        opacities_new = sigmoid_inv(
            torch.min(opc, torch.ones_like(opc)*reset_scele)
        )
        self.point_cloud.replace(
            {"opacity": opacities_new},
            self.optimizer
        )

    def generate_clone_mask(self, grads: Tensor) -> Tensor:
        """
        Generate the mask for cloning.
        Parameters
        ----------
        grads : torch.Tensor
            The gradients.
        Returns
        -------
        torch.Tensor
            The mask for cloning.
        """
        scaling = self.point_cloud.get_scaling
        cameras_extent = self.cameras_extent
        max_grad = self.densify_grad_threshold

        mask = torch.where(torch.norm(
            grads, dim=-1) >= max_grad, True, False)
        mask = torch.logical_and(
            mask,
            torch.max(
                scaling,
                dim=1
            ).values <= self.percent_dense*cameras_extent
        )
        return mask

    def generate_split_mask(self, grads: Tensor) -> Tensor:
        """
        Generate the mask for splitting.
        
        Parameters
        ----------
        grads : torch.Tensor
            The gradients.
        """
        scaling = self.point_cloud.get_scaling
        cameras_extent = self.cameras_extent
        max_grad = self.densify_grad_threshold

        num_points = len(self.point_cloud)
        padded_grad = torch.zeros((num_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        mask = torch.where(padded_grad >= max_grad, True, False)

        mask = torch.logical_and(
            mask,
            torch.max(
                scaling,
                dim=1
            ).values > self.percent_dense*cameras_extent
        )
        return mask

    def new_pos_scale(self, mask: Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate new position and scaling for splitting.
        
        Parameters
        ----------
        mask : torch.Tensor
            The mask for splitting.
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The new position and scaling.
        """
        scaling = self.point_cloud.get_scaling
        position = self.point_cloud.position
        rotation = self.point_cloud.rotation
        split_num = self.split_num

        stds = scaling[mask].repeat(split_num, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        if stds.shape != means.shape:
            stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        samples = torch.normal(mean=means, std=stds)
        rots = quat_to_rotmat(
            rotation[mask]
        ).repeat(split_num, 1, 1)
        new_pos = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        ) + (
            position[mask].repeat(split_num, 1)
        )
        new_scaling = self.point_cloud.scaling_inverse_activation(
            scaling[mask].repeat(split_num, 1) / (0.8*split_num)
        )
        return new_pos, new_scaling

    def densify_clone(self, grads: Tensor) -> None:
        """
        Densify the point cloud by cloning.
        
        Parameters
        ----------
        grads : torch.Tensor
            The gradients.
        """
        mask = self.generate_clone_mask(grads)
        atributes = self.point_cloud.select_atributes(mask)
        self.point_cloud.extand_points(atributes, self.optimizer)
        self.reset_controller_state()

    def densify_split(self, grads: Tensor) -> None:
        """
        Densify the point cloud by splitting.
        
        Parameters
        ----------
        grads : torch.Tensor
            The gradients.
        """
        mask = self.generate_split_mask(grads)
        new_pos, new_scaling = self.new_pos_scale(mask)
        atributes = self.point_cloud.select_atributes(mask)

        # Replace position and scaling from selected atributes
        atributes["position"] = new_pos
        atributes["scaling"] = new_scaling

        # Update rest of atributes
        for key, value in atributes.items():
            # Skip position and scaling, since they are already updated
            if key == "position" or key == "scaling":
                continue
            # Create a tuple of n_dim ones
            sizes = [1 for _ in range(len(value.shape))]
            sizes[0] = self.split_num
            sizes = tuple(sizes)

            # Repeat selected atributes in the fist dimension
            atributes[key] = value.repeat(*sizes)

        self.point_cloud.extand_points(atributes, self.optimizer)
        self.reset_controller_state()

        # TODO: need to remove unused operation
        prune_filter = torch.cat((mask, torch.zeros(self.split_num * mask.sum(),
                                                    device=self.device, dtype=bool)))
        valid_points_mask = ~prune_filter
        self.point_cloud.remove_points(valid_points_mask, self.optimizer)
        self.prune_postprocess(valid_points_mask)

    def prune_postprocess(self, valid_points_mask):
        """
        Postprocess after pruning.
        
        Parameters
        ----------
        valid_points_mask : torch.Tensor
            The mask for valid points.
        """
        self.grad_accum = self.grad_accum[valid_points_mask]
        self.acc_steps = self.acc_steps[valid_points_mask]
        self.max_radii = self.max_radii[valid_points_mask]

    def reset_controller_state(self) -> None:
        """
        Reset the controller state.
        """
        num_points = len(self.point_cloud)
        self.grad_accum = torch.zeros(
            (num_points, 1), device=self.device)
        self.acc_steps = torch.zeros((num_points, 1), device=self.device)
        self.max_radii = torch.zeros((num_points), device=self.device)

    @torch.no_grad()
    def accumulate_viewspace_grad(self, uv_points: Tensor) -> Tensor:
        """
        Accumulate viewspace gradients for batch.
        
        Parameters
        ----------
        uv_points : torch.Tensor
            The view space points.
        Returns
        -------
        torch.Tensor
            The viewspace gradients.
        """
        # Accumulate viewspace gradients for batch
        viewspace_grad = torch.zeros_like(
            uv_points[0].squeeze(0)
        )
        for vp in uv_points:
            viewspace_grad += vp.grad.squeeze(0).clone()
        
        if self.cfg.normalize_grad:
            viewspace_grad[..., 0] *= self.width / 2.0 
            viewspace_grad[..., 1] *= self.height / 2.0

        return viewspace_grad

    def prune(self, **kwargs) -> None:
        """
        Prune the point cloud.
        
        Parameters
        ----------
        step : int
            The current step.
        """
        # TODO: fix me
        size_threshold = 20 if self.step > self.opacity_reset_interval else None
        cameras_extent = self.cameras_extent

        prune_filter = (
            self.point_cloud.get_opacity < self.min_opacity
        ).squeeze()
        if size_threshold:
            big_points_vs = self.max_radii > size_threshold
            big_points_ws = self.point_cloud.get_scaling.max(
                dim=1).values > 0.1 * cameras_extent
            prune_filter = torch.logical_or(prune_filter, big_points_vs)
            prune_filter = torch.logical_or(prune_filter, big_points_ws)

        valid_points_mask = ~prune_filter
        self.point_cloud.remove_points(valid_points_mask, self.optimizer)
        self.prune_postprocess(valid_points_mask)

    def precess_grad(self, **kwargs) -> None:
        return self.accumulate_viewspace_grad(
            kwargs.get("uv_points", None)
        )

    def preprocess(self, **kwargs) -> None:
        selected_points = kwargs.get("visibility", None)
        point_radii = kwargs.get("radii", None)
        point_grad = self.precess_grad(**kwargs)

        # Keep track of max radii in image-space for pruning
        self.max_radii[selected_points] = torch.max(
            self.max_radii[selected_points], 
            point_radii[selected_points]
        )
        self.grad_accum[selected_points] += torch.norm(
            point_grad[selected_points, :2],
            dim=-1,
            keepdim=True
        )
        self.acc_steps[selected_points] += 1

    def duplicate(self, grads: Tensor, **kwargs) -> None:
        self.densify_clone(grads)
        self.densify_split(grads)

    def postprocess(self, **kwargs) -> None:
        if self.step % self.opacity_reset_interval == 0:
            self.reset_opacity()