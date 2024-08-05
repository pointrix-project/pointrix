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
    def _transform_metadata(self, meta, split):
        """
        The function for transforming the metadata.

        Parameters:
        -----------
        meta: List[Dict[str, Any]]
            The metadata for the dataset.
        
        Returns:
        --------
        meta: List[Dict[str, Any]]
            The transformed metadata.
        """
        cached_progress = ProgressLogger(description='transforming cached meta', suffix='iters/s')
        cached_progress.add_task(f'Transforming', f'Transforming {split} cached meta', len(meta))
        cached_progress.start()
        for i in range(len(meta)):
            # Transform Image
            image = meta[i]['image']
            w, h = image.size
            image = image.resize((int(w * self.scale), int(h * self.scale)))
            image = np.array(image) / 255.
            if image.shape[2] == 4:
                image = image[:, :, :3] * image[:, :, 3:4] + self.bg * (1 - image[:, :, 3:4])
            meta[i]['image'] = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().clamp(0.0, 1.0)
            cached_progress.update(f'Transforming', step=1)
            
            # Transform Depth
            meta[i]['depth'] = torch.from_numpy(np.array(meta[i]['depth'])).float().permute(2, 0, 1)
            # Transform Normal
            meta[i]['normal'] = (torch.from_numpy(np.array(meta[i]['normal'])) / 255.0).float().permute(2, 0, 1)
        cached_progress.stop()
        return meta
