import os
from typing import Optional, List
from dataclasses import dataclass, field

import torch
from pathlib import Path

from ..model import parse_model
from ..logger import parse_writer, Logger
from ..hook import parse_hooks
from ..dataset import parse_data_set
from ..utils.config import parse_structured
from ..optimizer import parse_optimizer, parse_scheduler
from ..exporter import parse_exporter
from ..densification.gs import DensificationController
from .default_datapipeline import BaseDataPipeline
from .base_trainer import BaseTrainer


class DefaultTrainer(BaseTrainer):
    """
    The default trainer class for training and testing the model.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    exp_dir : str
        The experiment directory.
    device : str, optional
        The device to use, by default "cuda".
    """

    def train_loop(self) -> None:
        """
        The training loop for the model.
        """
        loop_range = range(self.start_steps, self.cfg.max_steps+1)
        self.global_step = self.start_steps
        self.call_hook("before_train")
        for iteration in loop_range:
            self.call_hook("before_train_iter")
            # structure of batch {"frame_index": frame_index, "image": image, "depth": depth}
            batch = self.datapipeline.next_train(self.global_step)
            # update learning rate
            self.schedulers.step(self.global_step, self.optimizer)
            # model forward step
            self.train_step(batch)
            # update optimizer and densify point cloud
            with torch.no_grad():
                self.controller.f_step(**self.optimizer_dict)
                self.optimizer.update_model(**self.optimizer_dict)
            self.call_hook("after_train_iter")
            self.global_step += 1
            
            if self.cfg.val_interval > 0:
                if (iteration+1) % self.cfg.val_interval == 0 or iteration+1 == self.cfg.max_steps:
                    self.call_hook("before_val")
                    self.validation()
                    self.call_hook("after_val")
        self.call_hook("after_train")
        
    def train_step(self, batch: List[dict]) -> None:
        """
        The training step for the model.

        Parameters
        ----------
        batch : dict
            The batch data.
        """
        # structure of render_dict: {}
        #  render_dict = {
        #     "extrinsic_matrix": extrinsic_matrix,
        #     "intrinsic_params": intrinsic_params,
        #     "camera_center": camera_center,
        #     "position": point_cloud.position,
        #     "opacity": self.point_cloud.get_opacity,
        #     "scaling": self.point_cloud.get_scaling,
        #     "rotation": self.point_cloud.get_rotation,
        #     "shs": self.point_cloud.get_shs,
        # }
        render_results = self.model(batch, iteration=self.global_step)
        # structure of render_results: {}
        # example of render_results = {
        #     "rgb": rgb,
        #     "depth": depth,
        #     "normal": normal, ....
        # }

        self.loss_dict = self.model.get_loss_dict(render_results, batch)
        self.loss_dict['loss'].backward()
        # structure of optimizer_dict: {}
        # example of optimizer_dict = {
        #   "loss": loss,
        #   "uv_points": uv_points,
        #   "visibility": visibility,
        #   "radii": radii,
        #   "white_bg": white_bg
        self.optimizer_dict = self.model.get_optimizer_dict(self.loss_dict,
                                                            render_results,
                                                            self.white_bg)
        

    @torch.no_grad()
    def validation(self):
        self.val_dataset_size = len(self.datapipeline.validation_dataset)
        for i in range(0, self.val_dataset_size):
            self.call_hook("before_val_iter")
            batch = self.datapipeline.next_val(i)
            render_results = self.model(batch, training=False)
            self.metric_dict = self.model.get_metric_dict(render_results, batch)
            self.call_hook("after_val_iter")

    @torch.no_grad()
    def test(self, model_path=None) -> None:
        """
        The testing method for the model.
        """
        if model_path is None:
            model_path = os.path.join(self.exp_dir,
                                "chkpnt" + str(self.global_step) + ".pth")
        model_path = Path(model_path)
        self.load_model(model_path)
        self.model.to(self.device)
        # test_view_render(self.model,
        #                  self.datapipeline, output_path=self.cfg.output_path)
        # novel_view_render(self.model,
        #                   self.datapipeline, output_path=self.cfg.output_path)
        self.exporter(model_path.parent)
