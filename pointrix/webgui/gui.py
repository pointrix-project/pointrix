import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from torch import Tensor
from pathlib import Path
from random import randint
from jaxtyping import Float
from abc import abstractmethod
from collections import deque
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Tuple, Any, Dict, Union, List, Optional

from ..model.base_model import BaseModel

import viser
import viser.transforms as tf

from ..utils.registry import Registry
from ..utils.base import BaseModule, BaseObject

GUI_REGISTRY = Registry("GUI", modules=["pointrix.webgui"])
GUI_REGISTRY.__doc__ = ""

def get_w2c(camera):
    wxyz = camera.wxyz
    position = camera.position
    R = tf.SO3(wxyz=wxyz)
    R = R @ tf.SO3.from_x_radians(np.pi)
    R = torch.tensor(R.as_matrix())
    pos = torch.tensor(position, dtype=torch.float64)
    c2w = torch.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = pos
    c2w[:3, 1:3] *= -1
    w2c = torch.linalg.inv(c2w)
    return w2c

@GUI_REGISTRY.register()
class BaseGUI(BaseObject):
    @dataclass
    class Config:
        viewer_port: int = 8888
    cfg: Config
    
    def setup(self, model, device="cuda"):
        self.device = device
        self.model = model
        
        self.position = model.point_cloud.position
        self.opacity = model.point_cloud.get_opacity
        self.scaling = model.point_cloud.get_scaling
        self.rotation = model.point_cloud.get_rotation
        self.shs = model.point_cloud.get_shs
        
        self.render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.cfg.viewer_port)
        self.reset_view_button = self.server.add_gui_button("Reset View")
        
        self.server.on_client_disconnect(self._handle_client_disconnect)
        
        self.need_update = False
        self.pause_training = False
        
        self.train_viewer_update_period_slider = self.server.add_gui_slider(
            "Train Viewer Update Period",
            min=1,
            max=100,
            step=1,
            initial_value=10,
            disabled=self.pause_training,
        )
        
        self.pause_training_button = self.server.add_gui_button("Pause Training")
        self.sh_order = self.server.add_gui_slider(
            "SH Order", min=1, max=4, step=1, initial_value=1
        )
        self.resolution_slider = self.server.add_gui_slider(
            "Resolution", min=384, max=4096, step=2, initial_value=1024
        )
        self.time_slider = self.server.add_gui_slider(
            "Time", min=0, max=1, step=0.01, initial_value=0
        )
        self.near_plane_slider = self.server.add_gui_slider(
            "Near", min=0.1, max=30, step=0.5, initial_value=0.1
        )
        self.far_plane_slider = self.server.add_gui_slider(
            "Far", min=30.0, max=1000.0, step=10.0, initial_value=1000.0
        )

        self.show_train_camera = self.server.add_gui_checkbox(
            "Show Train Camera", initial_value=False
        )

        self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)
        
        @self.show_train_camera.on_update
        def _(_):
            self.need_update = True

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.near_plane_slider.on_update
        def _(_):
            self.need_update = True

        @self.far_plane_slider.on_update
        def _(_):
            self.need_update = True

        @self.pause_training_button.on_click
        def _(_):
            self.pause_training = not self.pause_training
            self.train_viewer_update_period_slider.disabled = not self.pause_training
            self.pause_training_button.name = (
                "Resume Training" if self.pause_training else "Pause Training"
            )

        @self.reset_view_button.on_click
        def _(_):
            self.need_update = True
            for client in self.server.get_clients().values():
                client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                    [0.0, -1.0, 0.0]
                )
        
        self.c2ws = []
        self.camera_infos = []
        
        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

        self.debug_idx = 0
        
    def _handle_client_disconnect(self, client):
        self.need_update = False
        
    def update_point_cloud(self, model):
        self.position = model.point_cloud.position
        self.opacity = model.point_cloud.get_opacity
        self.scaling = model.point_cloud.get_scaling
        self.rotation = model.point_cloud.get_rotation
        self.shs = model.point_cloud.get_shs
        
    @torch.no_grad()
    def update(self):
        if self.need_update:
            start = time.time()
            interval = None
            for client in self.server.get_clients().values():
                camera = client.camera
                w2c = torch.Tensor(get_w2c(camera)).to(self.device).float()

                W = self.resolution_slider.value
                H = int(self.resolution_slider.value/camera.aspect)
                focal_x = W/2/np.tan(camera.fov/2)
                focal_y = H/2/np.tan(camera.fov/2)

                start_cuda = torch.cuda.Event(enable_timing=True)
                end_cuda = torch.cuda.Event(enable_timing=True)
                start_cuda.record()

                intrinsic_params = torch.tensor([focal_x, focal_y, W/2, H/2]).to(self.device).float()
                camera_center = w2c.inverse()[:3, 3]
                # Render Image
                render_dict = {
                        "extrinsic_matrix": w2c.unsqueeze(0),
                        "intrinsic_params": intrinsic_params,
                        "camera_center": camera_center.unsqueeze(0),
                        "position": self.position,
                        "opacity": self.opacity,
                        "scaling": self.scaling,
                        "rotation": self.rotation,
                        "shs": self.shs,
                        "height": H,
                        "width": W
                    }
                
                render_results = self.model.renderer.render_batch(render_dict)
                
                end_cuda.record()
                torch.cuda.synchronize()
                interval = start_cuda.elapsed_time(end_cuda)/1000.

                out = render_results["rgb"].squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
                client.set_background_image(out, format="jpeg")
            
            if interval is not None:
                self.render_times.append(interval)
                self.fps.value = f"{1.0 / np.mean(self.render_times):.3g}"
                # print(f"Update time: {end - start:.3g}")
        
        