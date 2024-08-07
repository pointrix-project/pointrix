import torch
from ..dataset import BaseDataset

from typing import Any, Dict, List
from dataclasses import dataclass, field
from ..utils.config import parse_structured
from ..dataset import parse_data_set

class BaseDataPipeline:
    """
    Basic Pipline used in Pointrix

    data_path: str
        The path to the data
    data_set: str
        The dataset used in the pipeline
    shuffle: bool
        Whether shuffle the data
    batch_size: int
        The batch size used in data loader
    num_workers: int
        The number of workers used in data loader
    white_bg: bool
        Whether the background is white
    scale: float
        The image scale of the dataset
    use_dataloader: bool
        Whether use dataloader
    trainable_camera: bool
        Whether the camera is trainable
    extra_cfg: Dict[str, Any]
        The extra configuration for the pipeline
    
    Notes
    -----
    BaseDataPipeline is called by build_data_pipline
    """
    @dataclass
    class Config:
        data_path: str = "data"
        data_set: str = "BaseImageDataset"
        shuffle: bool = True
        batch_size: int = 1
        num_workers: int = 1
        use_dataloader: bool = True
        dataset: dict = field(default_factory=dict)
    cfg: Config

    def __init__(self, cfg: Config, device) -> None:
        self.cfg = parse_structured(self.Config, cfg)
        self.device = device
        Dataset = parse_data_set(self.cfg, self.device)
        # load camera to device
        self.device = device
        self.white_bg = self.cfg.dataset.white_bg
        self.use_dataloader = self.cfg.use_dataloader

        self.training_dataset = Dataset(self.cfg.dataset, split="train")
        self.validation_dataset = Dataset(self.cfg.dataset, split="val")

        self.point_cloud = self.training_dataset.pointcloud
        self.training_cameras = self.training_dataset.cameras
        self.validation_cameras = self.validation_dataset.cameras

        self.training_loader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=list,
            pin_memory=False
        )
        self.validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=list,
            pin_memory=False
        )
        self.iter_train_image_dataloader = iter(self.training_loader)
        self.iter_val_image_dataloader = iter(self.validation_loader)

    def next_loader_train_iter(self):
        """
        Get the next batch of training data
        """
        try:
            return next(self.iter_train_image_dataloader)
        except StopIteration:
            self.iter_train_image_dataloader = iter(self.training_loader)
            return next(self.iter_train_image_dataloader)

    def next_loader_eval_iter(self):
        """
        Get the next batch of evaluation data
        """
        try:
            return next(self.iter_val_image_dataloader)
        except StopIteration:
            self.iter_val_image_dataloader = iter(self.validation_loader)
            return next(self.iter_val_image_dataloader)

    def next_train(self, step: int = -1) -> Any:
        """
        Generate batch data for trainer

        Parameters
        ----------
        cfg: step
            the training step in trainer.
        """
        return self.next_loader_train_iter()
    def next_val(self, step: int = -1) -> Any:
        """
        Generate batch data for validation

        Parameters
        ----------
        cfg: step
            the validation step in validate progress.
        """
        return self.next_loader_eval_iter()

    @property
    def training_dataset_size(self) -> int:
        """
        Return training dataset size
        """
        return len(self.training_dataset)

    @property
    def validation_dataset_size(self) -> int:
        """
        Return validation dataset size
        """
        return len(self.validation_dataset)

    def get_param_groups(self) -> Any:
        """
        Return trainable parameters.
        """
        camera_params = {}

        for i, camera in enumerate(self.train_format_data.Camera_list):
            camera_params['camera_{}'.format(camera.idx)] = camera.param_groups

        return camera_params