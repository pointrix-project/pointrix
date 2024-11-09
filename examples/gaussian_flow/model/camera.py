import math
import torch
import numpy as np
from torch import nn
from torch import Tensor
from jaxtyping import Float
from typing import Union, List
from dataclasses import dataclass
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz

from pointrix.model.camera.camera_model import CameraModel, CAMERA_REGISTRY

@CAMERA_REGISTRY.register()
class TimeCameraModel(CameraModel):
    """
    Camera class used in Pointrix
    """
    def setup(self, camerasprior, device="cuda")->None:
        """
        Setup the camera class
        
        Parameters
        ----------
        camerasprior: CamerasPrior
            The camera priors
        """
        self.qrots = []
        self.tvecs = []
        self.intrs = []
        self.times = []
        self.enable_training = self.cfg.enable_training
        for cameraprior in camerasprior:
            extrinsic_matrix = cameraprior.extrinsic_matrix
            intrinsic_params = cameraprior.intrinsic_params
            self.width = cameraprior.image_width
            self.height = cameraprior.image_height
            time = cameraprior.time

            if not self.enable_training:
                self.qrots.append(torch.tensor(quat_xyzw_to_wxyz(
                    rotmat_to_unitquat(extrinsic_matrix[:3, :3]))).to(device))
                self.tvecs.append(torch.tensor(
                    extrinsic_matrix[:3, 3]).to(device))
                self.intrs.append(torch.tensor(intrinsic_params).to(device))
            else:
                self.qrots.append(nn.Parameter(torch.tensor(quat_xyzw_to_wxyz(
                    rotmat_to_unitquat(extrinsic_matrix[:3, :3])))).to(device))
                self.tvecs.append(nn.Parameter(
                    torch.tensor(extrinsic_matrix[:3, 3])).to(device))
                self.intrs.append(nn.Parameter(
                    torch.tensor(intrinsic_params)).to(device))
                
            self.times.append(torch.tensor(time).to(device))
    
    def get_time(self, idx_list):
        return torch.stack([self.times[idx] for idx in idx_list], dim=0)