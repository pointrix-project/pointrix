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


EXPORTER_REGISTRY = Registry("EXPORTER", modules=["pointrix.exporter"])
EXPORTER_REGISTRY.__doc__ = ""


@EXPORTER_REGISTRY.register()
class BaseExporter(BaseModule):
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
        l1_test = 0.0
        psnr_test = 0.0
        ssim_test = 0.0
        lpips_test = 0.0
        lpips_func = LPIPS()
        val_dataset = self.datapipeline.validation_dataset
        val_dataset_size = len(val_dataset)
        progress_bar = tqdm(
            range(0, val_dataset_size),
            desc="Validation progress",
            leave=False,
        )

        mkdir_p(os.path.join(output_path, 'test_view'))

        for i in range(0, val_dataset_size):
            batch = self.datapipeline.next_val(i)
            render_results = self.model(batch, training=False)
            image_name = os.path.basename(batch[0]['camera'].rgb_file_name)
            gt_image = torch.clamp(batch[0]['image'].to("cuda").float(), 0.0, 1.0)
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

            l1_test += l1_loss(image, gt_image).mean().double()
            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            lpips_test += lpips_func(image, gt_image).mean().double()
            progress_bar.update(1)
        progress_bar.close()
        l1_test /= val_dataset_size
        psnr_test /= val_dataset_size
        ssim_test /= val_dataset_size
        lpips_test /= val_dataset_size
        print(
            f"Test results: L1 {l1_test:.5f} PSNR {psnr_test:.5f} SSIM {ssim_test:.5f} LPIPS (VGG) {lpips_test:.5f}")