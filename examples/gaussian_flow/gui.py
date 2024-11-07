from pointrix.webgui.gui import BaseGUI, GUI_REGISTRY, get_w2c

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

import viser
import viser.transforms as tf
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage
from pointrix.utils.visualize import visualize_depth

@GUI_REGISTRY.register()
class GaussianFlowGUI(BaseGUI):
    def setup(self, model, device="cuda"):
        super().setup(model, device)
        
        
        self.time_slider = self.server.add_gui_slider(
            "Time", min=0, max=1, step=0.01, initial_value=0
        )
    
    
    @torch.no_grad()
    def update(self):
        if self.need_update:
            start = time.time()
            interval = None

            for client in self.server.get_clients().values():
                try:
                    camera = client.camera
                    w2c = torch.Tensor(get_w2c(camera)).to(self.device).float()

                    W = self.resolution_slider.value
                    H = int(self.resolution_slider.value/camera.aspect)
                    focal_x = self.fx_slider.value
                    focal_y = self.fy_slider.value

                    start_cuda = torch.cuda.Event(enable_timing=True)
                    end_cuda = torch.cuda.Event(enable_timing=True)
                    start_cuda.record()

                    intrinsic_params = torch.tensor([[focal_x, focal_y, W/2, H/2]]).to(self.device).float()
                    camera_center = w2c.inverse()[:3, 3]
                    
                    t = self.time_slider.value
        
                    self.model.point_cloud.set_timestep(
                        t=t,
                        training=False,
                        training_step=None
                    )
                    # Render Image
                    render_dict = {
                            "extrinsic_matrix": w2c.unsqueeze(0),
                            "intrinsic_params": intrinsic_params,
                            "camera_center": camera_center.unsqueeze(0),
                            "position": self.model.point_cloud.get_position_flow,
                            "opacity": self.model.point_cloud.get_opacity,
                            "scaling": self.model.point_cloud.get_scaling,
                            "rotation": self.model.point_cloud.get_rotation_flow,
                            "shs": self.model.point_cloud.get_shs_flow,
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
                except Exception as e:
                    print(e)
            
            if interval is not None:
                self.render_times.append(interval)
                self.fps.value = f"{1.0 / np.mean(self.render_times):.3g}"
                self.num_points.value = str(len(self.position))