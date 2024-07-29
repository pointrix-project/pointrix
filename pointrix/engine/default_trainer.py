import os
from typing import Optional, List
from dataclasses import dataclass, field

import torch
from pathlib import Path

from ..model import parse_model
from ..logger import parse_writer, Logger
from ..hook import parse_hooks
from ..renderer import parse_renderer
from ..dataset import parse_data_set
from ..utils.config import parse_structured
from ..optimizer import parse_optimizer, parse_scheduler
from ..exporter.novel_view import test_view_render, novel_view_render
# from ..exporter import parse_exporter
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
            batch = self.datapipeline.next_train(self.global_step)
            self.renderer.update_sh_degree(iteration)
            self.schedulers.step(self.global_step, self.optimizer)
            self.train_step(batch)
            with torch.no_grad():
                self.controler.f_step(**self.optimizer_dict)
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
        render_dict = self.model(batch)
        render_results = self.renderer.render_batch(render_dict, batch)
        self.loss_dict = self.model.get_loss_dict(render_results, batch)
        self.loss_dict['loss'].backward()
        self.optimizer_dict = self.model.get_optimizer_dict(self.loss_dict,
                                                            render_results,
                                                            self.white_bg)
        

    @torch.no_grad()
    def validation(self):
        self.val_dataset_size = len(self.datapipeline.validation_dataset)
        for i in range(0, self.val_dataset_size):
            self.call_hook("before_val_iter")
            batch = self.datapipeline.next_val(i)
            render_dict = self.model(batch, training=False)
            render_results = self.renderer.render_batch(render_dict, batch)
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
        test_view_render(self.model, self.renderer,
                         self.datapipeline, output_path=self.cfg.output_path)
        # novel_view_render(self.model, self.renderer,
        #                   self.datapipeline, output_path=self.cfg.output_path)
        # self.exporter(self.model, self.datapipeline, self.renderer, model_path.parent)
