import os
import random
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field

import torch
import imageio
from ..utils.visualize import visualize_depth, visualize_rgb
from ..exporter.base_exporter import EXPORTER_REGISTRY, BaseExporter

@EXPORTER_REGISTRY.register()
class VideoExporter(BaseExporter):
    """
    The exporter class for the video export.
    """
    def forward(self, output_path, novel_view_list=["Dolly", "Zoom", "Spiral"]):
        """
        Render the novel view and save the images to the output path.

        Parameters
        ----------
        output_path : str
            The output path to save the images.
        novel_view_list : list, optional
            The list of novel views to render, by default ["Dolly", "Zoom", "Spiral"]
        """
        cameras = self.datapipeline.training_cameras
        print("Rendering Novel view ...............")
        for novel_view in novel_view_list:
            feat_frame = {}
            novel_view_camera_list = cameras.generate_camera_path(50, novel_view)
            for i, camera in enumerate(novel_view_camera_list):
                atributes_dict = self.model(training=False, render=False)
                render_dict = {
                    "camera": camera,
                    "height": int(camera.image_height),
                    "width": int(camera.image_width),
                    "extrinsic_matrix": camera.extrinsic_matrix.to(self.device),
                    "intrinsic_params": camera.intrinsic_params.to(self.device),
                    "camera_center": camera.camera_center.to(self.device),
                }
                atributes_dict.update(render_dict)
                render_results = self.model.renderer.render_iter(**atributes_dict)
                for feat_name, feat in render_results['rendered_features_split'].items():
                    visual_feat = eval(f"visualize_{feat_name}")(feat.squeeze())
                    if feat_name not in feat_frame:
                        feat_frame[feat_name] = []
                    feat_frame[feat_name].append(visual_feat)
                    if not os.path.exists(os.path.join(output_path, f'{novel_view}_{feat_name}')):
                        os.makedirs(os.path.join(
                            output_path, f'{novel_view}_{feat_name}'))
                    imageio.imwrite(os.path.join(
                        output_path, f'{novel_view}_{feat_name}', "{:0>3}.png".format(i)), visual_feat)

            for feat_name, feat in feat_frame.items():
                imageio.mimwrite(os.path.join(
                    output_path, f'{novel_view}_{feat_name}', f'{novel_view}_{feat_name}.mp4'), feat, fps=30, quality=8)
