import os
import random
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional

import torch
import imageio
from ..utils.system import mkdir_p
from ..utils.visualize import visualize_depth, visualize_rgb
from ..model.loss import psnr, ssim, LPIPS, l1_loss

from ..utils.base import BaseModule
from ..utils.registry import Registry
from ..engine.default_datapipeline import BaseDataPipeline
from ..model.base_model import BaseModel

from ..logger import ProgressLogger


EXPORTER_REGISTRY = Registry("EXPORTER", modules=["pointrix.exporter"])
EXPORTER_REGISTRY.__doc__ = ""


class ExporterList:
    """
    A wrapper for multiple exporters.
    """
    def __init__(self, exporter_dict: dict) -> None:
        """
        Parameters
        ----------
        exporter_dict : dict
            The dictionary of the exporters.
        """
        from .mesh_exporter import TSDFFusion
        from .video_exporter import VideoExporter
        for key, value in exporter_dict.items():
            assert isinstance(value, (MetricExporter, TSDFFusion, VideoExporter)), (
                '`ExporWrapperDict` only accept (BaseOptimizer, TSDFFusion, VideoExporter) instance, '
                f'but got {key}: {type(value)}')
        self.exporter_dict = exporter_dict
    
    def export(self, output_path, **kwargs) -> None:
        """
        Exporters for image, mesh or video.

        Parameters
        ----------
        kwargs : dict
            The keyword arguments.
        """
        for name, exporter in self.exporter_dict.items():
            exporter.forward(output_path, **kwargs)
    
    def __len__(self) -> int:
        """
        Get the number of the exporters.

        Returns
        -------
        int
            The number of the exporters.
        """
        return len(self.exporter_dict)
    
    def __contains__(self, key: str) -> bool:
        """
        Check if the key is in the exporter dictionary.

        Parameters
        ----------
        key : str
            The key to check.
        
        Returns
        -------
        bool
            Whether the key is in the exporter dictionary.
        """
        return key in self.exporter_dict


@EXPORTER_REGISTRY.register()
class MetricExporter(BaseModule):
    """
    Base class for all exporters.

    Parameters
    ----------
    cfg : Optional[Union[dict, DictConfig]]
        The configuration dictionary.
    model : BaseModel
        The model which is used to render
    datapipeline : BaseDataPipeline
        The data pipeline which is used to initialize the point cloud.
    device : str, optional
        The device to use, by default "cuda".
    """
    @dataclass
    class Config:
        voxel_size: float = 0.02
        """tsdf voxel size"""
        sdf_truc: float = 0.08
        """TSDF truncation"""
        total_points: int = 8_000_000
        """Total target surface samples"""
        target_triangles: Optional[int] = None
        """Target number of triangles to simplify mesh to."""
    cfg: Config

    def setup(self, model: BaseModel, datapipeline: BaseDataPipeline, device="cuda"):
        self.device  = device
        self.datapipeline = datapipeline
        self.model = model
        
    @torch.no_grad()
    def forward(self, output_path):
        """
        Render the test view and save the images to the output path.

        Parameters
        ----------
        model : BaseModel
            The point cloud model.
        datapipeline : DataPipeline
            The data pipeline object.
        output_path : str
            The output path to save the images.
        """
        l1 = 0.0
        psnr_metric = 0.0
        ssim_metric  = 0.0
        lpips_metric  = 0.0
        lpips_func = LPIPS()
        val_dataset = self.datapipeline.validation_dataset
        val_dataset_size = len(val_dataset)
        progress_logger = ProgressLogger(description='Extracting metrics', suffix='iters/s')
        progress_logger.add_task(f'Metric', f'Extracting metrics', val_dataset_size)
        mkdir_p(os.path.join(output_path, 'test_view'))

        with progress_logger.progress as progress:
            for i in range(0, val_dataset_size):
                batch = self.datapipeline.next_val(i)
                render_results = self.model(batch, training=False)
                image_name = os.path.basename(batch[0]['camera'].rgb_file_name)
                gt = torch.clamp(batch[0]['image'].to("cuda").float(), 0.0, 1.0)
                image = torch.clamp(
                    render_results['rgb'], 0.0, 1.0).squeeze()
                visualize_feature = ['rgb']

                for feat_name in visualize_feature:
                    feat = render_results[feat_name]
                    visual_feat = eval(f"visualize_{feat_name}")(feat.squeeze())
                    if not os.path.exists(os.path.join(output_path, f'test_view_{feat_name}')):
                        os.makedirs(os.path.join(
                            output_path, f'test_view_{feat_name}'))
                    imageio.imwrite(os.path.join(
                        output_path, f'test_view_{feat_name}', image_name), visual_feat)

                l1 += l1_loss(image, gt, return_mean=True).double()
                psnr_metric += psnr(image, gt).mean().double()
                ssim_metric += ssim(image, gt).mean().double()
                lpips_metric += lpips_func(image, gt).mean().double()
                progress_logger.update(f'Metric', step=1)
        l1 /= val_dataset_size
        psnr_metric /= val_dataset_size
        ssim_metric /= val_dataset_size
        lpips_metric /= val_dataset_size
        print(
            f"Test results: L1 {l1:.5f} PSNR {psnr_metric:.5f} SSIM {ssim_metric:.5f} LPIPS (VGG) {lpips_metric:.5f}")