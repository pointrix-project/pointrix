import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any, Dict, List

from pointrix.dataset.colmap_data import ColmapDataset
from pointrix.dataset.base_data import DATA_SET_REGISTRY, BaseDataset
from pointrix.logger.writer import Logger, ProgressLogger


@DATA_SET_REGISTRY.register()
class ColmapDepthNormalDataset(ColmapDataset):
    def _transform_observed_data(self, observed_data, split):

        cached_progress = ProgressLogger(description='transforming cached observed_data', suffix='iters/s')
        cached_progress.add_task(f'Transforming', f'Transforming {split} cached observed_data', len(observed_data))
        with cached_progress.progress as progress:
            for i in range(len(observed_data)):
                # Transform Image
                image = observed_data[i]['image']
                w, h = image.size
                image = image.resize((int(w * self.scale), int(h * self.scale)))
                image = np.array(image) / 255.
                if image.shape[2] == 4:
                    image = image[:, :, :3] * image[:, :, 3:4] + self.bg * (1 - image[:, :, 3:4])
                observed_data[i]['image'] = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().clamp(0.0, 1.0)
                cached_progress.update(f'Transforming', step=1)

                # Transform Normal
                observed_data[i]['normal'] = (torch.from_numpy(np.array(observed_data[i]['normal'])) / 255.0).float().permute(2, 0, 1)
        return observed_data
