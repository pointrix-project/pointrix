import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any, Dict, List

from .utils.dataprior import CameraPrior, PointsPrior
from ..dataset.base_data import DATA_SET_REGISTRY, BaseDataset
from ..logger.writer import Logger, ProgressLogger

from .utils.colmap  import (
    read_colmap_extrinsics,
    read_colmap_intrinsics,
    ExtractColmapCamInfo,
    read_3D_points_binary
)

@DATA_SET_REGISTRY.register()
class ColmapDataset(BaseDataset):
    """
    The dataset class for the Colmap based dataset.
    """
    def _load_camera(self, split: str) -> List[CameraPrior]:
        """
        The function for loading the camera information.
        
        Parameters:
        -----------
        split: str
            The split of the dataset.

        """
        extrinsics = read_colmap_extrinsics(self.data_root / Path("sparse/0") / Path("images.bin"))
        intrinsics = read_colmap_intrinsics(self.data_root / Path("sparse/0") / Path("cameras.bin"))
        # TODO: more methods for splitting the data
        splithold = 8
        cameras = []
        for idx, key in enumerate(extrinsics):
            colmapextr = extrinsics[key]
            colmapintr = intrinsics[colmapextr.camera_id]
            R, T, fx, fy, cx, cy, width, height = ExtractColmapCamInfo(colmapextr, colmapintr, self.scale)

            camera = CameraPrior(idx=idx, R=R, T=T, image_width=width, image_height=height, rgb_file_name=os.path.basename(colmapextr.name),
                            fx=fx, fy=fy, cx=cx, cy=cy, device='cuda')
            cameras.append(camera)
        sorted_camera = sorted(cameras.copy(), key=lambda x: x.rgb_file_name)
        index = list(range(len(sorted_camera)))
        self.train_index = [i for i in index if i % splithold != 0]
        self.val_index = [i for i in index if i not in self.train_index]
        cameras_results = [sorted_camera[i] for i in self.train_index] if split == 'train' else [sorted_camera[i] for i in self.val_index] 
        return cameras_results
    
    def _load_pointcloud(self) -> dict:
        """
        The function for loading the Pointcloud for initialization of gaussian model.

        Returns:
        --------
        point_cloud : dict
            The point cloud for the gaussian model.
        """
        points3d_ply_path = self.data_root / Path("sparse/0/points3D.ply")
        points3d_bin_path = self.data_root / Path("sparse/0/points3D.bin")
        if not points3d_ply_path.exists() and points3d_bin_path.exists():
            print("convert binary to ply for the first time...")
            positions, colors = read_3D_points_binary(points3d_bin_path)
            normals = np.zeros_like(positions)
            point_cloud = PointsPrior(positions=positions, colors=colors, normals=normals)
            point_cloud.save_ply(points3d_ply_path)
        else:
            point_cloud = PointsPrior()
            point_cloud.read_ply(points3d_ply_path)
        return point_cloud
    
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
        cached_progress.stop()
        return meta
    
    def _load_metadata(self, split):
        """
        The function for loading the metadata.

        Parameters:
        -----------
        split: str
            The split of the dataset.
        
        Returns:
        --------
        meta_data: List[Dict[str, Any]]
            The metadata for the dataset.
        """
        meta_data = []
        for k, v in self.meta_dirs_dict.items():
            meta_path = self.data_root / Path(v)
            if not os.path.exists(meta_path):
                Logger.error(f"Meta path {meta_path} does not exist.")
            meta_file_names = sorted(os.listdir(meta_path))
            meta_file_names_split = [meta_file_names[i] for i in self.train_index] if split == "train" else [meta_file_names[i] for i in self.val_index]
            cached_progress = ProgressLogger(description='Loading cached meta', suffix='iters/s')
            cached_progress.add_task(f'cache_{k}', f'Loading {split} cached {k}', len(meta_file_names_split))
            cached_progress.start()
            for idx, file in enumerate(meta_file_names_split):
                if len(meta_data) <= idx:
                    meta_data.append({})
                if file.endswith('.npy'):
                    meta_data[idx].update({k: np.load(meta_path / Path(file))})
                elif file.endswith('png') or file.endswith('jpg') or file.endswith('JPG'):
                    meta_data[idx].update({k: Image.open(meta_path / Path(file))})
                cached_progress.update(f'cache_{k}', step=1)
            cached_progress.stop()
        return meta_data

