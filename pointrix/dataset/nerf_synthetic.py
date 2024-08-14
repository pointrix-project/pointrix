import os
import math
import json
import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, List
from pathlib import Path

from .utils.dataprior import CameraPrior
from ..dataset.base_data import DATA_SET_REGISTRY, BaseDataset
from ..logger.writer import Logger, ProgressLogger
from .utils.dataset import ExtractBlenderCamInfo, load_from_json

@DATA_SET_REGISTRY.register()
class NerfSyntheticDataset(BaseDataset):
    """
    The dataset class for the synthetic NeRF dataset.
    """
    def load_camera_prior(self, split: str) -> List[CameraPrior]:
        """
        The function for loading the camera typically requires user customization.

        Parameters
        ----------
        split: The split of the data.
        """
        if split == 'train':
            datainfo = load_from_json(self.data_root / Path("transforms_train.json"))
        elif split == 'val':
            datainfo = load_from_json(self.data_root / Path("transforms_test.json"))
        cameras = []
        sorted_datainfo= sorted(datainfo["frames"], key=lambda x: x["file_path"])
        for idx, info in enumerate(sorted_datainfo):
            rgb_file_name = os.path.join(self.data_root, info["file_path"] + '.png')
            R, T = ExtractBlenderCamInfo(info["transform_matrix"])
            image = np.array(Image.open(rgb_file_name))
            fovx = datainfo["camera_angle_x"]
            fx = image.shape[1] / (2 * np.tan(fovx / 2)) * self.scale
            fy = fx
            cx = image.shape[1] * self.scale / 2.
            cy = image.shape[0] * self.scale / 2.
            camera = CameraPrior(idx=idx, R=R, T=T, image_width=image.shape[1]* self.scale, image_height=image.shape[0]* self.scale,
                                 rgb_file_name=rgb_file_name, fx=fx, fy=fy, cx=cx, cy=cy, device='cuda')
            cameras.append(camera)
        return cameras
    
    def transform_observed_data(self, observed_data, split):
        """
        The function for transforming the observed_data.

        Parameters:
        -----------
        meta: List[Dict[str, Any]]
            The observed_data for the dataset.
        
        Returns:
        --------
        meta: List[Dict[str, Any]]
            The transformed observed_data.
        """
        cached_progress = ProgressLogger(description='transforming cached observed data', suffix='iters/s')
        cached_progress.add_task(f'Transforming', f'Transforming {split} cached observed data', len(observed_data))
        with cached_progress.progress as progress:
            for i in range(len(observed_data)):
                # Transform Image
                image = observed_data[i]['image']
                w, h = image.size
                image = image.resize((int(w * self.scale), int(h * self.scale)))
                image = np.array(image) / 255.
                if image.shape[2] == 4:
                    image = image[:, :, :3] * image[:, :, 3:4] + self.background_color * (1 - image[:, :, 3:4])
                observed_data[i]['image'] = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().clamp(0.0, 1.0)
                cached_progress.update(f'Transforming', step=1)
        return observed_data
    
    def load_observed_data(self, split):
        observed_data = []
        if split == 'train':
            datainfo = load_from_json(self.data_root / Path("transforms_train.json"))
        elif split == 'val':
            datainfo = load_from_json(self.data_root / Path("transforms_test.json"))
        sorted_datainfo = sorted(datainfo["frames"], key=lambda x: x["file_path"])
        cached_progress = ProgressLogger(description='Loading cached observed data', suffix='iters/s')
        cached_progress.add_task(f'cache_image', f'Loading {split} cached image', len(datainfo["frames"]))
        with cached_progress.progress as progress:
            for idx, info in enumerate(sorted_datainfo):
                if len(observed_data) <= idx:
                    observed_data.append({})
                image_path = self.data_root / Path(info["file_path"] + '.png')
                image = Image.open(image_path)
                observed_data[idx].update({"image": image})
                cached_progress.update(f'cache_image', step=1)
        return observed_data
        
            
            
            