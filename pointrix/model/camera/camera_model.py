import math
import torch
import numpy as np
from torch import nn
from torch import Tensor
from jaxtyping import Float
from typing import Union, List
from numpy.typing import NDArray
from dataclasses import dataclass
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz

from ...dataset.utils.dataprior import CameraPrior, CamerasPrior
from ...utils.base import BaseObject
from ...utils.pose import ViewScaling, unitquat_to_rotmat



class CameraModel(BaseObject):
    """
    Camera class used in Pointrix
    """

    @dataclass
    class Config:
        """
        Parameters
        ----------
        enable_training: bool
            Whether the camera is trainable
        scene_scale: float
            The scale of the scene
        """
        enable_training: bool = False
        scene_scale: float = 1.0

    def setup(self, camerasprior:CamerasPrior, device="cuda")->None:
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
        self.enable_training = self.cfg.enable_training
        for cameraprior in camerasprior:
            extrinsic_matrix = cameraprior.extrinsic_matrix
            intrinsic_params = cameraprior.intrinsic_params
            self.width = cameraprior.image_width
            self.height = cameraprior.image_height

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

    def intrinsic_params(self, idx_list) -> Float[Tensor, "C 4"]:
        """
        Get the intrinsics matrix of the cameras.
        
        Parameters
        ----------
        idx_list: int
            The index list of the camera.

        Returns
        -------
        intrinsic_params: Float[Tensor, "4"]

        Notes
        -----
        property of the camera class

        """
        return torch.stack([self.intrs[idx] for idx in idx_list], dim=0)

    def rotation_matrices(self, idx_list) -> Float[Tensor, "C 3 3"]:
        """
        Get the rotation matrix of the cameras.
        
        Parameters
        ----------
        idx_list: int
            The index list  of the camera.
        
        Returns
        -------
        rotation_matrix: Float[Tensor, "C 3 3"]
        """
        rotation_matrix_list = [unitquat_to_rotmat(torch.nn.functional.normalize(self.qrots[idx], dim=-1)) for idx in idx_list]
        return torch.stack(rotation_matrix_list, dim=0)

    def __len__(self):
        return self.qrot.shape[0]

    def extrinsic_matrices(self, idx_list) -> Float[Tensor, "C 4 4"]:
        """
        Get the extrinsic matrix from the cameras.
        
        Parameters
        ----------
        idx: int
            The index of the camera.

        Returns
        -------
        _extrinsic_matrix: Float[Tensor, "C 4 4"]

        Notes
        -----
        property of the camera class

        """
        R = self.rotation_matrices(idx_list)
        t = self.translation_vectors(idx_list).unsqueeze(-1)

        Rt = torch.concat([R, t], dim=-1)
        tmp = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=Rt.device).view(1, 1, 4).repeat(Rt.shape[0], 1, 1)
        Rt_hom = torch.concat([Rt, tmp], dim=-2)

        return Rt_hom

    def translation_vectors(self, idx_list) -> Float[Tensor, "C 3"]:
        """
        Get the translation vector from the cameras.
        
        Parameters
        ----------
        idx_list: int
            The index list of the camera.
        
        Returns
        -------
        _translation_vector: Float[Tensor, "C 3"]
        """

        return torch.stack([self.tvecs[idx] for idx in idx_list], dim=0)

    def camera_centers(self, idx_list) -> Float[Tensor, "C 3"]:
        """
        Get the camera center from the cameras.
        
        Parameters
        ----------
        idx_list: int
            The index list of the camera.

        Returns
        -------
        _camera_center: Float[Tensor, "C 3"]

        Notes
        -----
        property of the camera class

        """
        # [C, 3, 3]
        R = self.rotation_matrices(idx_list)
        inv_R = R.permute(*range(R.ndimension()-2), -1, -2)
        return torch.matmul(-inv_R, self.translation_vectors(idx_list).unsqueeze(-1)).squeeze(-1)

    @property
    def image_height(self) -> int:
        """
        Get the image height from the cameras.

        Returns
        -------
        height: int
            The image height.

        Notes
        -----
        property of the camera class

        """
        return self.height

    @property
    def image_width(self) -> int:
        """
        Get the image width from the cameras.

        Returns
        -------
        width: int
            The image width.

        Notes
        -----
        property of the camera class

        """
        return self.width
