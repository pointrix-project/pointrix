import math
import json
import logging
import functools
import math
import torch
import numpy as np
from torch import nn
from torch import Tensor
from jaxtyping import Float
from typing import Union, List
from numpy.typing import NDArray
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from plyfile import PlyData, PlyElement

from ...utils.pose import ConcatRT, ViewScaling, GetCamcenter

@dataclass()
class CameraPrior:
    """
    Camera prior info used in data Pipeline

    Parameters
    ----------
    idx: int
        The index of the camera.
    width: int
        The width of the image.
    height: int
        The height of the image.
    R: Float[Tensor, "3 3"]
        The rotation matrix of the camera.
    T: Float[Tensor, "3 1"]
        The translation vector of the camera.
    fx: float
        The focal length of the camera in x direction.
    fy: float
        The focal length of the camera in y direction.
    cx: float
        The center of the image in x direction.
    cy: float
        The center of the image in y direction.
    rgb_file_name: str
        The path of the image.
    scene_scale: float
        The scale of the scene.

    Notes
    -----
    fx, fy, cx, cy and fovX, fovY are mutually exclusive. 
    If fx, fy, cx, cy are provided, fovX, fovY will be calculated from them. 
    If fovX, fovY are provided, fx, fy, cx, cy will be calculated from them.

    Examples
    --------
    >>> idx = 1
    >>> width = 800
    >>> height = 600
    >>> R = np.eye(3)
    >>> T = np.zeros(3)
    >>> focal_length_x = 800
    >>> focal_length_y = 800
    >>> camera = Camera_Prior(idx=idx, R=R, T=T, width=width, height=height, rgb_file_name='1_rgb.png',
                        fx=focal_length_x, fy=focal_length_y, cx=width/2, cy=height/2, scene_scale=1.0)
    """
    idx: int
    image_width: int
    image_height: int
    R: Union[Float[Tensor, "3 3"], NDArray]
    T: Union[Float[Tensor, "3 1"], NDArray]
    fx: Union[float, None] = None
    fy: Union[float, None] = None
    cx: Union[float, None] = None
    cy: Union[float, None] = None
    rgb_file_name: str = None
    rgb_file_path: str = None
    scene_scale: float = 1.0
    device: str = 'cuda'
    
    def __post_init__(self):
        if not isinstance(self.R, Tensor):
            self.R = torch.tensor(self.R)
        if not isinstance(self.T, Tensor):
            self.T = torch.tensor(self.T)
        self.extrinsic_matrix = ViewScaling(ConcatRT(self.R, self.T), scale=self.scene_scale).to(self.device)
        self.intrinsic_params = torch.tensor([self.fx, self.fy, self.cx, self.cy], dtype=torch.float32).to(self.device)
        self.camera_center = GetCamcenter(self.extrinsic_matrix).to(self.device)
        
@dataclass()
class PointsPrior:
    """
    Point cloud initialization used in data Pipeline
    """
    
    positions: Union[Float[Tensor, "N 3"], NDArray, None]=None
    colors: Union[Float[Tensor, "N 3"], NDArray, None]=None
    normals: Union[Float[Tensor, "N 3"], NDArray, None]=None
    
    def save_ply(self, path):
        # Define the structured array dtype for vertices
        attribute = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue']
        type = ['f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'u1', 'u1', 'u1']
        dtype = [(a, t) for a, t in zip(attribute, type)]
        # Combine xyz, normals, and rgb into one structured array
        ply = np.zeros(self.positions.shape[0], dtype=dtype)
        for i, (pos, r) in enumerate(zip(self.positions, self.colors)):
            ply[i] = (*pos, *self.normals[i], *r)

        # Create and write the PlyData object
        ply_data = PlyData([PlyElement.describe(ply, 'vertex')])
        ply_data.write(path)
        
    def read_ply(self, path):
        plyData = PlyData.read(path)
        coordinates_attributes = ['x', 'y', 'z']
        color_attributes = ['red', 'green', 'blue']
        normal_attributes = ['nx', 'ny', 'nz']
        assert all([attr in plyData['vertex'] for attr in coordinates_attributes]), "Missing coordinate attributes"
        assert all([attr in plyData['vertex'] for attr in color_attributes]), "Missing color attributes"
        assert all([attr in plyData['vertex'] for attr in normal_attributes]), "Missing normal attributes"
        
        self.positions = np.stack([plyData['vertex'][a] for a in coordinates_attributes], axis=1)
        self.colors = np.stack([plyData['vertex'][a] for a in color_attributes], axis=1) / 255.
        self.normals = np.stack([plyData['vertex'][a] for a in normal_attributes], axis=1)

class CamerasPrior:
    """
    Cameras class used in Pointrix, which are used to generate camera paths.

    Parameters
    ----------
    camera_list: List[CameraPrior]
        The list of the CameraPrior.

    Examples
    --------
    >>> width = 800
    >>> height = 600
    >>> R = np.eye(3)
    >>> T = np.zeros(3)
    >>> focal_length_x = 800
    >>> focal_length_y = 800
    >>> camera1 = CameraPrior(idx=1, R=R, T=T, width=width, height=height, rgb_file_name='1_rgb.png',
                        fx=focal_length_x, fy=focal_length_y, cx=width/2, cy=height/2, scene_scale=1.0)
    >>> camera2 = CameraPrior(idx=2, R=R, T=T, width=width, height=height, rgb_file_name='1_rgb.png', 
                        fx=focal_length_x, fy=focal_length_y, cx=width/2, cy=height/2, scene_scale=1.0)
    >>> cameras = CamerasPrior([camera1, camera2])
    """

    def __init__(self, camera_list: List[CameraPrior]):
        self.camera_type = camera_list[0].__class__
        self.cameras = camera_list
        self.num_cameras = len(camera_list)
        self.Rs = torch.stack([cam.R for cam in camera_list], dim=0)
        self.Ts = torch.stack([cam.T for cam in camera_list], dim=0)
        self.camera_centers = torch.stack(
            [cam.camera_center for cam in camera_list], dim=0)  # (N, 3)

        self.radius = self.get_radius()

    def __len__(self):
        return self.num_cameras

    def __getitem__(self, index):
        return self.cameras[index]

    def get_radius(self):
        """
        Get the path radius of the cameras.

        Returns
        -------
        camera_radius: float
            The radius of the cameras.
        """
        cams_center = torch.mean(self.camera_centers, dim=0, keepdims=True)
        dist = torch.linalg.norm(self.camera_centers - cams_center, dim=1)
        camera_radius = torch.max(dist) 
        return camera_radius

    def generate_camera_path(self, num_frames: int, mode: str = "Dolly"):
        """
        Generate the camera path.

        Parameters
        ----------
        num_frames: int
            The number of frames of the camera path.

        Returns
        -------
        camera_path: Float[Tensor, "num_frames 4 4"]
            The camera path.
        """
        SE3_poses = torch.zeros(self.num_cameras, 3, 4)
        for i in range(self.num_cameras):
            SE3_poses[i, :3, :3] = self.cameras[i].R
            SE3_poses[i, :3, 3] = self.cameras[i].T

        mean_pose = torch.mean(SE3_poses[:, :, 3], 0)

        # Select the best idx for rendering
        render_idx = 0
        best_dist = 1000000000
        for iidx in range(SE3_poses.shape[0]):
            cur_dist = torch.mean((SE3_poses[iidx, :, 3] - mean_pose) ** 2)
            if cur_dist < best_dist:
                best_dist = cur_dist
                render_idx = iidx

        c2w = SE3_poses.cpu().detach().numpy()[render_idx]

        fx = self.cameras[render_idx].fx
        fy = self.cameras[render_idx].fy
        width = self.cameras[render_idx].image_width
        height = self.cameras[render_idx].image_height

        if mode == "Dolly":
            return self.dolly(c2w, [fx, fy], width, height, sc=1., length=SE3_poses.shape[0], num_frames=num_frames)
        elif mode == "Zoom":
            return self.zoom(c2w, [fx, fy], width, height, sc=1., length=SE3_poses.shape[0], num_frames=num_frames)
        elif mode == "Spiral":
            return self.spiral(c2w, [fx, fy], width, height, sc=1., length=SE3_poses.shape[0], num_frames=num_frames)
        elif mode == "Circle":
            return self.circle([fx, fy], width, height, sc=1., length=SE3_poses.shape[0], num_frames=num_frames)

    def pose_to_cam(self, poses, focals, width, height):
        """
        Generate the camera path from poses.

        Parameters
        ----------
        poses: Float[Tensor, "num_frames 3 4"]
            The poses of the camera path.
        focals: Float[Tensor, "num_frames"]
            The focal lengths of the camera path.
        width: int
            The width of the image.
        height: int 
            The height of the image.
        """
        camera_list = []
        for idx in range(focals.shape[0]):
            pose = poses[idx]
            focal = focals[idx]
            R = pose[:3, :3]
            T = pose[:3, 3]
            cam = self.camera_type(idx=idx, R=R, T=T, image_width=width, image_height=height, rgb_file_name='',
                                   fx=focal, fy=focal, cx=width/2., cy=height/2., scene_scale=1.0)
            camera_list.append(cam)
        return camera_list

    def dolly(self, c2w, focal, width, height, sc, length, num_frames):
        """
        Generate the camera path with dolly zoom.

        Parameters
        ----------
        c2w: Float[Tensor, "3 4"]
            The camera to world transform.
        focal: list[float]
            The focal length of the camera.
        sc: float
            The scale of the scene.
        length: int
            The length of the camera path.
        num_frames: int
            The number of frames of the camera path.

        Returns
        -------
        camera_path: Float[Tensor, "num_frames 4 4"]
            The camera path.
        """
        # TODO: how to define the max_disp
        max_disp = 2.0

        max_trans = max_disp / focal[0] * sc
        dolly_poses = []
        dolly_focals = []
        # Dolly zoom
        for i in range(num_frames):
            x_trans = 0.0
            y_trans = 0.0
            z_trans = max_trans * 2.5 * i / float(30 // 2)
            i_pose = np.concatenate(
                [
                    np.concatenate(
                        [np.eye(3), np.array([x_trans, y_trans, z_trans])[
                            :, np.newaxis]],
                        axis=1,
                    ),
                    np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
                ],
                axis=0,
            )
            i_pose = np.linalg.inv(i_pose)
            ref_pose = np.concatenate(
                [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[
                    np.newaxis, :]], axis=0
            )
            render_pose = np.dot(ref_pose, i_pose)
            dolly_poses.append(render_pose[:3, :])
            new_focal = focal[0] - focal[0] * 0.1 * z_trans / max_trans / 2.5
            dolly_focals.append(new_focal)
        dolly_poses = np.stack(dolly_poses, 0)[:, :3]
        dolly_focals = np.stack(dolly_focals, 0)

        return self.pose_to_cam(dolly_poses, dolly_focals, width, height)

    def zoom(self, c2w, focal, width, height, sc, length, num_frames):
        """
        Generate the camera path with zoom.

        Parameters
        ----------
        c2w: Float[Tensor, "3 4"]
            The camera to world transform.
        focal: list[float]
            The focal length of the camera.
        width: int
            The width of the image.
        height: int
            The height of the image.
        sc: float
            The scale of the scene.
        length: int
            The length of the camera path.
        num_frames: int
            The number of frames of the camera path.

        Returns
        -------
        camera_path: Float[Tensor, "num_frames 4 4"]
            The camera path.
        """
        # TODO: how to define the max_disp
        max_disp = 20.0

        max_trans = max_disp / focal[0] * sc
        zoom_poses = []
        zoom_focals = []
        # Zoom in
        # Zoom in
        for i in range(num_frames):
            x_trans = 0.0
            y_trans = 0.0
            # z_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_novelviews)) * args.z_trans_multiplier
            z_trans = max_trans * 2.5 * i / float(30 // 2)
            i_pose = np.concatenate(
                [
                    np.concatenate(
                        [np.eye(3), np.array([x_trans, y_trans, z_trans])[
                            :, np.newaxis]],
                        axis=1,
                    ),
                    np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
                ],
                axis=0,
            )

            # torch.tensor(np.linalg.inv(i_pose)).float()
            i_pose = np.linalg.inv(i_pose)

            ref_pose = np.concatenate(
                [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[
                    np.newaxis, :]], axis=0
            )

            render_pose = np.dot(ref_pose, i_pose)
            zoom_poses.append(render_pose[:3, :])
            zoom_focals.append(focal[0])

        zoom_poses = np.stack(zoom_poses, 0)[:, :3]
        zoom_focals = np.stack(zoom_focals, 0)
        return self.pose_to_cam(zoom_poses, zoom_focals, width, height)

    def spiral(self, c2w, focal, width, height, sc, length, num_frames):
        """
        Generate the camera path with spiral.

        Parameters
        ----------
        c2w: Float[Tensor, "3 4"]
            The camera to world transform.
        focal: list[float]
            The focal length of the camera.
        width: int
            The width of the image.
        height: int
            The height of the image.
        sc: float
            The scale of the scene.
        length: int
            The length of the camera path.
        num_frames: int
            The number of frames of the camera path.
        """
        # TODO: how to define the max_disp
        max_disp = 10.0

        max_trans = max_disp / focal[0] * sc

        spiral_poses = []
        spiral_focals = []
        # Rendering teaser. Add translation.
        for i in range(num_frames):
            x_trans = max_trans * 1.5 * \
                np.sin(2.0 * np.pi * float(i) / float(60)) * 2.0
            y_trans = (
                max_trans
                * 1.5
                * (np.cos(2.0 * np.pi * float(i) / float(60)) - 1.0)
                * 2.0
                / 3.0
            )
            z_trans = 0.0

            i_pose = np.concatenate(
                [
                    np.concatenate(
                        [np.eye(3), np.array([x_trans, y_trans, z_trans])[
                            :, np.newaxis]],
                        axis=1,
                    ),
                    np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
                ],
                axis=0,
            )

            i_pose = np.linalg.inv(i_pose)

            ref_pose = np.concatenate(
                [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[
                    np.newaxis, :]], axis=0
            )

            render_pose = np.dot(ref_pose, i_pose)
            # output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
            spiral_poses.append(render_pose[:3, :])
            spiral_focals.append(focal[0])
        spiral_poses = np.stack(spiral_poses, 0)[:, :3]
        spiral_focals = np.stack(spiral_focals, 0)
        return self.pose_to_cam(spiral_poses, spiral_focals, width, height)