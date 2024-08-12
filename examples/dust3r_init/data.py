import os
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from pathlib import Path
from typing import Any, Dict, List

from pointrix.dataset.utils.dataprior import CameraPrior, PointsPrior
from pointrix.dataset.base_data import DATA_SET_REGISTRY, BaseDataset
from pointrix.logger.writer import Logger, ProgressLogger

from utils  import read_dust3r_ply_binary

from pointrix.dataset.colmap_data import ColmapDataset

def load_cameras(data_root):
    with np.load(data_root / Path('camera.npz')) as data:
        focals = data['focals']
        cam2world = data['cam2world']
    return focals, cam2world

DUST_3R_Scale = 5.

@DATA_SET_REGISTRY.register()
class Dust3RDataset(ColmapDataset):
    """
    The dataset class for the Colmap based dataset.
    """
    def _load_camera_prior(self, split: str) -> List[CameraPrior]:
        """
        The function for loading the camera information.
        
        Parameters:
        -----------
        split: str
            The split of the dataset.

        """
        focals, cams2world = load_cameras(self.data_root)

        file_names = os.listdir(os.path.join(self.data_root, "images_train"))
        file_names.sort()

        with Image.open(os.path.join(self.data_root, "images_train", file_names[0])) as img:
            width, height = img.size

        width *= self.scale
        height *= self.scale
        dust3r_origin_ratio = max(width, height) / 512 
        mean_focal_new = np.mean(focals[:, 0]) * dust3r_origin_ratio 

        cameras = []

        for i, pose_c2w in enumerate(cams2world):
            R = pose_c2w[0:3, 0:3]
            T = pose_c2w[0:3, 3].reshape(-1)

            T = - R.T @ T
            R = R.T
            fx = fy = mean_focal_new
            T = T * DUST_3R_Scale
            camera = CameraPrior(idx=i, R=R, T=T, image_width=width, image_height=height, rgb_file_name=file_names[i],
                            fx=fx, fy=fy, cx=width/2, cy=height/2, device='cuda')
            cameras.append(camera)
        cameras_results = cameras
        return cameras_results
    
    
    def _load_pointcloud_prior(self) -> dict:
        """
        The function for loading the Pointcloud for initialization of gaussian model.

        Returns:
        --------
        point_cloud : dict
            The point cloud for the gaussian model.
        """
        positions, colors = read_dust3r_ply_binary(self.data_root)
        positions = positions * DUST_3R_Scale
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        voxel_size = 0.005

        # downsampling
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)

        downsampled_positions = np.asarray(downsampled_pcd.points)
        downsampled_colors = np.asarray(downsampled_pcd.colors)
        
        normals = np.zeros_like(positions)
        
        point_cloud = PointsPrior(positions=positions, colors=colors, normals=normals)

        return point_cloud
    
    def _load_observed_data(self, split):
        """
        The function for loading the observed_data.

        Parameters:
        -----------
        split: str
            The split of the dataset.
        
        Returns:
        --------
        observed_data: List[Dict[str, Any]]
            The observed_datafor the dataset.
        """
        observed_data = []
        for k, v in self.observed_data_dirs_dict.items():
            observed_data_path = self.data_root / Path(v)
            if not os.path.exists(observed_data_path):
                Logger.error(f"observed_data path {observed_data_path} does not exist.")
            observed_data_file_names = sorted(os.listdir(observed_data_path))
            cached_progress = ProgressLogger(description='Loading cached observed_data', suffix='iters/s')
            cached_progress.add_task(f'cache_{k}', f'Loading {split} cached {k}', len(observed_data_file_names))
            with cached_progress.progress as progress:
                for idx, file in enumerate(observed_data_file_names):
                    if len(observed_data) <= idx:
                        observed_data.append({})
                    if file.endswith('.npy'):
                        observed_data[idx].update({k: np.load(observed_data_path / Path(file))})
                    elif file.endswith('png') or file.endswith('jpg') or file.endswith('JPG'):
                        observed_data[idx].update({k: Image.open(observed_data_path / Path(file))})
                    else:
                        print(f"File format {file} is not supported.")
                    cached_progress.update(f'cache_{k}', step=1)
        return observed_data