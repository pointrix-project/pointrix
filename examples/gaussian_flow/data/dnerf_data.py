import os
import json
import math
import numpy as np
from pathlib import Path
from PIL import Image

from pointrix.dataset.utils.dataprior import PointsPrior
from pointrix.dataset.utils.dataset import ExtractBlenderCamInfo
from pointrix.dataset.base_data import BaseDataset, DATA_SET_REGISTRY
from pointrix.dataset.nerf_synthetic import NerfSyntheticDataset
from pointrix.dataset.utils.dataset import load_from_json
from pointrix.logger.writer import Logger, ProgressLogger

from .data import TimeCameraPrior

@DATA_SET_REGISTRY.register()
class DNeRFDataset(NerfSyntheticDataset):
    
    def load_camera_prior(self, split: str):
        with open(os.path.join(self.data_root, "transforms_train.json")) as json_file:
            train_json = json.load(json_file)
        with open(os.path.join(self.data_root, "transforms_test.json")) as json_file:
            val_json = json.load(json_file)
            
            
        time_line = [frame["time"] for frame in train_json["frames"]] + [frame["time"] for frame in val_json["frames"]]
        max_timestamp = max(time_line)
        
       
        
        if split == 'video':
            cameras = generateCamerasFromTransforms(
                self.data_root, 
                "transforms_train.json", "png",
                maxtime=max_timestamp
            )
            return cameras

        if split == 'train':
            json_file = train_json
        elif split == 'val':
            json_file = val_json
        fovx = json_file["camera_angle_x"]
        frames = sorted(json_file["frames"], key=lambda x: x["file_path"])
        cameras = []
        
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(
                self.data_root, frame["file_path"] + '.png')
            
            timestamp = frame["time"] / max_timestamp
            
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            # R is stored transposed due to 'glm' in CUDA code
            R = w2c[:3, :3]
            T = w2c[:3, 3]
            
            image_path = os.path.join(self.data_root, cam_name)
            image_name = Path(cam_name).stem

            image = Image.open(image_path)
            image_np = np.array(image)
            
            fx = image_np.shape[1] / (2 * math.tan(fovx / 2))* self.scale
            fy = fx
            cx = image_np.shape[1] * self.scale / 2.
            cy = image_np.shape[0] * self.scale / 2.

            camera = TimeCameraPrior(
                idx=idx, 
                R=R, 
                T=T, 
                image_width=image.size[0] * self.scale, 
                image_height=image.size[1] * self.scale, 
                rgb_file_name=image_name,
                rgb_file_path=image_path, 
                fx=fx, 
                fy=fy, 
                cx=cx, 
                cy=cy, 
                time=timestamp
            )
            cameras.append(camera)
        return cameras