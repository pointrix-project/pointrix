import os
from typing import Optional, List
from dataclasses import dataclass, field

import torch
from pathlib import Path
from abc import abstractmethod

import threading
import time

from ..model import parse_model
from ..logger import parse_writer, Logger
from ..hook import parse_hooks
from ..dataset import parse_data_set
from ..utils.config import parse_structured
from ..optimizer import parse_optimizer, parse_scheduler
from ..exporter import parse_exporter
from ..controller import parse_controller
from ..webgui import parse_gui
from ..controller.gs import DensificationController
from .default_datapipeline import BaseDataPipeline


class BaseTrainer:
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
    @dataclass
    class Config:
        # Modules
        model: dict = field(default_factory=dict)
        optimizer: dict = field(default_factory=dict)
        scheduler: Optional[dict] = field(default_factory=dict)
        writer: dict = field(default_factory=dict)
        hooks: dict = field(default_factory=dict)
        exporter: dict = field(default_factory=dict)
        controller: dict = field(default_factory=dict)
        gui: dict = field(default_factory=dict)
        # Dataset
        dataset_name: str = "NeRFDataset"
        datapipeline: dict = field(default_factory=dict)

        # Device
        device: str = "cuda"

        # Test config
        training: bool = True
        test_model_path: str = ""

        # Training config
        batch_size: int = 1
        num_workers: int = 0
        max_steps: int = 30000
        val_interval: int = 2000
        spatial_lr_scale: bool = True

        # Progress bar
        bar_upd_interval: int = 10
        # Output path
        output_path: str = "output"
        
        enable_gui: bool = True

    cfg: Config

    def __init__(self, cfg: Config, exp_dir: Path, name: str) -> None:
        super().__init__()
        self.exp_dir = exp_dir
        self.start_steps = 1
        self.global_step = 0
        # build config
        self.cfg = parse_structured(self.Config, cfg)
        self.device = self.cfg.device
        # build hooks
        self.hooks = parse_hooks(self.cfg.hooks)
        self.call_hook("before_run")
        # build datapipeline
        self.datapipeline = BaseDataPipeline(
            self.cfg.datapipeline, device=self.device)

        # build point cloud model
        self.white_bg = self.datapipeline.white_bg
        self.model = parse_model(
            self.cfg.model, self.datapipeline, device=self.device)

        # build logger and hooks
        self.writer = parse_writer(
            self.cfg.writer, exp_dir, experiment_name=name, logcfg=self.cfg)

        # build exporter
        
        self.exporter = parse_exporter(
                self.cfg.exporter, self.model, self.datapipeline, device=self.device)
        
        if self.cfg.enable_gui:
            self.gui = parse_gui(self.cfg.gui, self.model, device=self.device)
        
        if self.cfg.training:
            if self.cfg.enable_gui:
                self.lock = threading.Lock()
                def gui_thread():
                    while True:
                        if self.gui.need_update:
                            self.gui.update()
                        else:
                            time.sleep(3)
                viewer_thread = threading.Thread(target=gui_thread)
                viewer_thread.start()
            cameras_extent = self.datapipeline.training_dataset.radius
            self.schedulers = parse_scheduler(self.cfg.scheduler,
                                              cameras_extent if self.cfg.spatial_lr_scale else 1.
                                              )
            self.optimizer = parse_optimizer(self.cfg.optimizer,
                                             self.model, datapipeline=self.datapipeline,
                                             cameras_extent=cameras_extent)

            self.controller = parse_controller(self.cfg.controller, self.optimizer, self.model, cameras_extent=cameras_extent)
            
    @abstractmethod
    def train_loop(self) -> None:
        """
        The training loop for the model.
        """
        raise NotImplementedError

    def call_hook(self, fn_name: str, **kwargs) -> None:
        """
        Call the hook method.

        Parameters
        ----------
        fn_name : str
            The hook method name.
        kwargs : dict
            The keyword arguments.
        """
        for hook in self.hooks:
            if hasattr(hook, fn_name):
                try:
                    getattr(hook, fn_name)(self, **kwargs)
                except TypeError as e:
                    raise TypeError(f'{e} in {hook}') from None

    def load_model(self, path: Path = None) -> None:
        data_list = torch.load(path)
        for k, v in data_list.items():
            print(f"Loaded {k} from checkpoint")
            # get arrtibute from model
            try:
                arrt = getattr(self, k)
                if hasattr(arrt, 'load_state_dict'):
                    arrt.load_state_dict(v)
                else:
                    setattr(self, k, v)
            except:
                if not self.cfg.training and k == 'optimizer':
                    Logger.log("optimizer is not needed in test mode")
                else:
                    Logger.print_exception()

    def save_model(self, path: Path = None) -> None:
        if path is None:
            path = os.path.join(self.exp_dir,
                                "chkpnt" + str(self.global_step) + ".pth")
        data_list = {
            "global_step": self.global_step,
            "optimizer": self.optimizer.state_dict(),
            "model": self.model.get_state_dict(),
        }
        torch.save(data_list, path)

    @torch.no_grad()
    def validation(self):
        raise NotImplementedError

    @torch.no_grad()
    def test(self, model_path=None) -> None:
        """
        The testing method for the model.
        """
        raise NotImplementedError
