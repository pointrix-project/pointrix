import os
import numpy as np
from pathlib import Path
from PIL import Image

from pointrix.dataset.utils.dataprior import CameraPrior, PointsPrior
from pointrix.dataset.base_data import BaseDataset, DATA_SET_REGISTRY
from pointrix.dataset.colmap_data import ColmapDataset
from pointrix.dataset.utils.dataset import load_from_json
from pointrix.logger.writer import Logger, ProgressLogger

class TimeCameraPrior(CameraPrior):
    def __init__(self, **kwargs):
        self.time = kwargs.pop('time')
        super().__init__(**kwargs)

def camera_nerfies_from_JSON(path, scale)->dict:
    """
    Loads a JSON camera into memory.
    
    Parameters
    ----------
    path : str
        Path to the JSON file.
    scale : float
        Scale factor to apply to the camera parameters.
    
    """
    camera_json = load_from_json(path)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
        camera_json['tangential_distortion'] = camera_json['tangential']

    return dict(
        orientation=np.array(camera_json['orientation']),
        position=np.array(camera_json['position']),
        focal_length=camera_json['focal_length'] * scale,
        principal_point=np.array(camera_json['principal_point']) * scale,
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.array(camera_json['radial_distortion']),
        tangential_distortion=np.array(camera_json['tangential_distortion']),
        image_size=np.array((int(round(camera_json['image_size'][0] * scale)),
                             int(round(camera_json['image_size'][1] * scale)))),
    )

@DATA_SET_REGISTRY.register()
class NerfiesDataset(ColmapDataset):
    
    def load_camera_prior(self, split: str):
        scene_json = load_from_json(self.data_root / Path('scene.json'))
        meta_json = load_from_json(self.data_root / Path('metadata.json'))
        dataset_json = load_from_json(self.data_root / Path('dataset.json'))
        
        self.coord_scale = scene_json['scale']
        self.scene_center = scene_json['center']
        
        if "val_ids" in dataset_json:
            train_img = dataset_json['train_ids']
            val_img = dataset_json['val_ids']
            all_img = train_img + val_img
            ratio = 0.25
        else:
            all_id = dataset_json['ids']
            train_img = all_id[::4]
            val_img = all_id[2::4]
            all_img = train_img + val_img
            ratio = 0.5
        
        all_time = [meta_json[i]['appearance_id'] for i in all_img]
        max_time = max(all_time)
        all_time = [meta_json[i]['appearance_id'] / max_time for i in all_img]
        
        train_time = [meta_json[i]['appearance_id'] / max_time for i in train_img]
        val_time = [meta_json[i]['appearance_id'] / max_time for i in val_img]

        # all poses
        train_cam_params = []
        val_cam_params = []
        for im in train_img:
            camera = camera_nerfies_from_JSON(Path(f'{self.data_root}/camera/{im}.json'), ratio)
            camera['position'] = camera['position'] - self.scene_center
            camera['position'] = camera['position'] * self.coord_scale
            train_cam_params.append(camera)
        
        for im in val_img:
            camera = camera_nerfies_from_JSON(Path(f'{self.data_root}/camera/{im}.json'), ratio)
            camera['position'] = camera['position'] - self.scene_center
            camera['position'] = camera['position'] * self.coord_scale
            val_cam_params.append(camera)

        train_img = [f'{self.data_root}/rgb/{int(1 / ratio)}x/{i}.png' for i in train_img]
        val_img =  [f'{self.data_root}/rgb/{int(1 / ratio)}x/{i}.png' for i in val_img]

        self.cameras = []
        images = train_img if split == 'train' else val_img
        cam_params = train_cam_params if split == 'train' else val_cam_params
        times = train_time if split == 'train' else val_time

        for idx in range(len(images)):
            image_path = images[idx]
            image = np.array(Image.open(image_path))
            image = Image.fromarray((image).astype(np.uint8))
            image_name = Path(image_path).stem

            orientation = cam_params[idx]['orientation'].T
            position = -cam_params[idx]['position'] @ orientation
            focal = cam_params[idx]['focal_length']
            fid = times[idx]
            T = position
            R = orientation.T

            camera = TimeCameraPrior(idx=idx, R=R, T=T, image_width=image.size[0], image_height=image.size[1], rgb_file_name=image_name,
                            rgb_file_path=image_path, fx=focal, fy=focal, cx=image.size[0]/2, cy=image.size[1]/2, time=fid)
            self.cameras.append(camera)
        return self.cameras
    
    def load_pointcloud_prior(self):
        xyz = np.load(self.data_root / Path("points.npy"))
        xyz = (xyz - self.scene_center) * self.coord_scale
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = PointsPrior(positions=xyz, colors=shs, normals=np.zeros_like(xyz))
        return pcd
    
    def load_observed_data(self, split):
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
                Logger.print(f"observed_data path {observed_data_path} does not exist.")
    
            cached_progress = ProgressLogger(description='Loading cached observed_data', suffix='iters/s')
            cached_progress.add_task(f'cache_{k}', f'Loading {split} cached {k}', len(self.cameras))
            with cached_progress.progress as progress:
                for i, camera in enumerate(self.cameras):
                    if len(observed_data) <= camera.idx:
                        observed_data.append({})
                    if k == 'image':
                        image = np.array(Image.open(camera.rgb_file_path))
                        image = Image.fromarray((image).astype(np.uint8))
                        observed_data[i].update({k: image})
                    cached_progress.update(f'cache_{k}', step=1)
        return observed_data