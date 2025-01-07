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
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from ..utils.registry import Registry
from ..utils.base import BaseModule, BaseObject
from ..utils.visualize import visualize_depth

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
        
        self.rgb_button = self.server.add_gui_button("rgb")
        self.depth_button = self.server.add_gui_button("depth")
        
        self.server.on_client_disconnect(self._handle_client_disconnect)
        
        self.display_mode = "rgb"
        
        self.need_update = False
        self.pause_training = False
        
        fx = model.training_camera_model.intrs[0].cpu().numpy()[0]
        fy = model.training_camera_model.intrs[1].cpu().numpy()[0]

        self.train_viewer_update_period_slider = self.server.add_gui_slider(
            "Gaussian Point Cloud Update Period",
            min=1,
            max=100,
            step=1,
            initial_value=10,
            disabled=self.pause_training,
        )
        @self.rgb_button.on_click
        def _(event: viser.GuiEvent):
            if event.client is None:
                return
            self.display_mode = "rgb"
        
        @self.depth_button.on_click
        def _(event: viser.GuiEvent):
            if event.client is None:
                return
            if not model.cfg.renderer.render_depth:
                print("Depth rendering is disabled in the renderer, please enable it")
            self.display_mode = "depth"
        
        self.resolution_slider = self.server.add_gui_slider(
            "Resolution", min=512, max=4096, step=2, initial_value=1024
        )
        
        self.fx_slider = self.server.add_gui_slider(
            "fx", min=0, max=2 * fx, step=0.1, initial_value=fx
        )
        
        self.fy_slider = self.server.add_gui_slider(
            "fy", min=0, max=2 * fy, step=0.1, initial_value=fy
        )
        self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)
        self.num_points = self.server.add_gui_text("Num Points", initial_value=str(len(model.point_cloud.position)), disabled=True)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True
        
        @self.reset_view_button.on_click
        def _(_):
            self.need_update = True
            for client in self.server.get_clients().values():
                client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                    [0.0, -1.0, 0.0]
                )

        self.debug_idx = 0
        
        buttons = (
            TitlebarButton(
                text="Pointrix",
                icon="GitHub",
                href="https://github.com/pointrix-project/pointrix",
            ),
        )
        image = TitlebarImage(
            image_url_light="https://viser.studio/latest/_static/logo.svg",
            image_alt="Logo",
            href="https://github.com/nerfstudio-project/viser"
        )
        titlebar_theme = TitlebarConfig(buttons=buttons, image=image)
        brand_color = self.server.add_gui_rgb("Brand color", (10, 10, 10), visible=False)
        self.server.configure_theme(
            titlebar_content=titlebar_theme,
            show_logo=True,
            brand_color=brand_color.value,
        )
        
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
                focal_x = self.fx_slider.value
                focal_y = self.fy_slider.value

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

                if self.display_mode == 'rgb':
                    out = render_results["rgb"].squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
                elif self.display_mode == 'depth':
                    out = render_results["depth"].squeeze()
                    out = visualize_depth(out, tensorboard=False)
                client.set_background_image(out, format="jpeg")
            
            if interval is not None:
                self.render_times.append(interval)
                self.fps.value = f"{1.0 / np.mean(self.render_times):.3g}"
                self.num_points.value = str(len(self.position))
        
        