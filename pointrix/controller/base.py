import torch
from torch import Tensor
from torch.optim import Optimizer
from dataclasses import dataclass
from typing import Tuple, List

from pointrix.utils.config import C

from ..utils.base import BaseObject
from ..utils.registry import Registry

CONTROLER_REGISTRY = Registry("CONTROLER", modules=["pointrix.controller"])
CONTROLER_REGISTRY.__doc__ = ""

@CONTROLER_REGISTRY.register()
class BaseDensificationController(BaseObject):
    @dataclass
    class Config:
        """
        Parameters
        ----------
        prue_interval: int
            The interval to prune the point cloud
        min_opacity: float
            The minimum opacity of the point cloud
        densify_start_iter: int
            The iteration to start densifying the point cloud
        densify_stop_iter: int
            The iteration to stop densifying the point cloud
        densify_grad_threshold: float
            The threshold to densify the point cloud
        optimizer_name: str
            The name of the optimizer which is used to update the point cloud
        """
        prune_interval: int = 100
        min_opacity: float = 0.005
        duplicate_interval: int = 100
        densify_start_iter: int = 500
        densify_stop_iter: int = 15000
        densify_grad_threshold: float = 0.0002
        optimizer_name: str = "optimizer_1"
        max_points: int = 5000000

    cfg: Config

    def get_step_value(self, value):
        """
        Get the step value for the given value.
        
        Parameters
        ----------
        value : float
            The value to convert to step value.
        """
        if isinstance(value, float):
            return C(value, 0, self.step)
        else:
            return int(C(value, 0, self.step)) 

    def setup(self, optimizers, model, **kwargs) -> None:
        self.step = 0

        self.point_cloud = model.point_cloud

        # Densification setup
        num_points = len(self.point_cloud)
        self.grad_accum = torch.zeros((num_points, 1)).to(self.device)
        self.acc_steps = torch.zeros((num_points, 1)).to(self.device)
        self.max_radii = torch.zeros((num_points), device=self.device)
        self.optimizer = optimizers.optimizer_dict[self.cfg.optimizer_name].optimizer

        self.update_states()

    def reset(self) -> None:
        self.step = 0

    def update_states(self) -> None:
        """
        Update the hyperparameters of the optimizer.
        """
        self.step += 1
        self.split_num = self.cfg.split_num
        self.min_opacity = self.cfg.min_opacity
        self.prune_interval = self.cfg.prune_interval
        self.densify_grad_threshold = self.cfg.densify_grad_threshold
        self.duplicate_interval = self.cfg.duplicate_interval
        self.opacity_reset_interval = self.cfg.opacity_reset_interval
    
    def densify(self, **kwargs):
        if self.step % self.duplicate_interval == 0:
            avg_grad = self.grad_accum / self.acc_steps 
            avg_grad[avg_grad.isnan()] = 0.0
            self.duplicate(avg_grad, **kwargs)

        if self.step % self.prune_interval == 0:
            self.prune(**kwargs)
        torch.cuda.empty_cache()
        self.postprocess(**kwargs)

    def f_step(self, **kwargs):
        self.process_grad(**kwargs)
        if kwargs['num_points'] < self.cfg.max_points:
            if self.step < self.cfg.densify_stop_iter:
                self.preprocess(**kwargs)
                if self.step > self.cfg.densify_start_iter:
                    self.densify(**kwargs)

        self.update_states()

    def prune(self, **kwargs):
        pass

    def duplicate(self, avg_grad, **kwargs):
        pass

    def process_grad(self, **kwargs):
        # update self.grad_accum
        pass

    def preprocess(self, **kwargs):
        pass

    def postprocess(self, **kwargs):
        pass