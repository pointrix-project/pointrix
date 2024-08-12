import os
import vdbfusion
import random
import numpy as np
from pathlib import Path
import open3d as o3d
from torch import Tensor
from typing import Optional, Tuple, List

import torch
import imageio

from ..exporter.base_exporter import EXPORTER_REGISTRY, MetricExporter
from ..logger import ProgressLogger


@EXPORTER_REGISTRY.register()
class TSDFFusion(MetricExporter):
    """
    The exporter class for the mesh export using tsdffusion.
    modified from https://github.com/maturk/dn-splatter/blob/main/dn_splatter/export_mesh.py
    """

    def forward(self, output_dir: str):
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        frame_count = len(
            self.datapipeline.iter_train_image_dataloader)  # type: ignore
        samples_per_frame = (self.cfg.total_points +
                             frame_count) // frame_count
        tsdf_volume = vdbfusion.VDBVolume(
            voxel_size=self.cfg.voxel_size, sdf_trunc=self.cfg.sdf_truc, space_carving=True
        )
        point_list = []
        color_list = []
        self.model.point_cloud.cuda()
        with torch.no_grad():
            progress_logger = ProgressLogger(
                description='Extracting mesh using TSDF', suffix='iters/s')
            progress_logger.add_task(f'Mesh', f'Extracting mesh using TSDF', len(
                self.datapipeline.iter_train_image_dataloader))
            with progress_logger.progress as progress:
                for i, batch in enumerate(self.datapipeline.iter_train_image_dataloader):
                    # Assume batch size == 1
                    data = batch[0]
                    camera_info = data["camera"]
                    render_output = self.model(batch)
                    try:
                        depth_map = render_output["depth"].squeeze()
                    except:
                        raise ValueError(
                            'No depth in render_output, please set config trainer.model.renderer.render_depth as True')

                    camera_to_world = torch.linalg.inv(
                        camera_info.extrinsic_matrix)[:3, :4]
                    height, width = int(camera_info.image_height), int(
                        camera_info.image_width)

                    sampled_indices = random.sample(
                        range(height * width), samples_per_frame)

                    points, colors = self.get_colored_points_from_depth(
                        depths=depth_map,
                        rgbs=render_output["rgb"].squeeze().permute(1, 2, 0),
                        fx=camera_info.fx,
                        fy=camera_info.fy,
                        cx=camera_info.cx,
                        cy=camera_info.cy,
                        img_size=(width, height),
                        c2w=camera_to_world,
                        mask=sampled_indices,
                    )

                    point_list.append(points)
                    color_list.append(colors)
                    tsdf_volume.integrate(
                        points.double().cpu().numpy(),
                        extrinsic=camera_to_world[:3,
                                                  3].double().cpu().numpy(),
                    )

                    progress_logger.update(f'Mesh', step=1)
                vertices, faces = tsdf_volume.extract_triangle_mesh(
                    min_weight=5)

                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh.compute_vertex_normals()
                colors = torch.cat(color_list, dim=0)
                colors = colors.cpu().numpy()
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

                # Simplify mesh
                if self.cfg.target_triangles is not None:
                    mesh = mesh.simplify_quadric_decimation(
                        self.cfg.target_triangles)

                o3d.io.write_triangle_mesh(
                    str(output_dir / "TSDFfusion_baseline_mesh.ply"),
                    mesh,
                )
                print(
                    f"Finished computing mesh: {str(output_dir / 'TSDFfusion.ply')}"
                )

    def get_colored_points_from_depth(self, depths, rgbs, c2w, fx, fy, cx, cy, img_size,
                                      mask: Optional[Tensor] = None):
        """Return colored pointclouds from depth and rgb frame and c2w. Optional masking.

        Returns:
            Tuple of (points, colors)
        """
        points, _ = self.get_means3d_backproj(depths=depths.float(),
                                            fx=fx,
                                            fy=fy,
                                            cx=cx,
                                            cy=cy,
                                            img_size=img_size,
                                            c2w=c2w.float(),
                                            device=depths.device)
        points = points.squeeze(0)
        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.tensor(mask, device=depths.device)
            colors = rgbs.view(-1, 3)[mask]
            points = points[mask]
        else:
            colors = rgbs.view(-1, 3)
            points = points
        return (points, colors)

    def get_means3d_backproj(self, 
                             depths: Tensor, 
                             fx: float, 
                             fy: float, 
                             cx: int,
                             cy: int,
                             img_size: tuple,
                             c2w: Tensor,
                             device: torch.device,
                             mask: Optional[Tensor] = None,
                             ) -> Tuple[Tensor, List]:
        """Backprojection using camera intrinsics and extrinsics

        image_coords -> (x,y,depth) -> (X, Y, depth)

        Returns:
            Tuple of (means: Tensor, image_coords: Tensor)
        """

        if depths.dim() == 3:
            depths = depths.view(-1, 1)
        elif depths.shape[-1] != 1:
            depths = depths.unsqueeze(-1).contiguous()
            depths = depths.view(-1, 1)
        if depths.dtype != torch.float:
            depths = depths.float()
            c2w = c2w.float()
        if c2w.device != device:
            c2w = c2w.to(device)

        image_coords = self.get_camera_coords(img_size)
        image_coords = image_coords.to(device)  # note image_coords is (H,W)

        # TODO: account for skew / radial distortion
        means3d = torch.empty(
            size=(img_size[0], img_size[1], 3), dtype=torch.float32, device=device
        ).view(-1, 3)
        means3d[:, 0] = (image_coords[:, 0] - cx) * depths[:, 0] / fx  # x
        means3d[:, 1] = (image_coords[:, 1] - cy) * depths[:, 0] / fy  # y
        means3d[:, 2] = depths[:, 0]  # z

        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.tensor(mask, device=depths.device)
            means3d = means3d[mask]
            image_coords = image_coords[mask]

        if c2w is None:
            c2w = torch.eye((means3d.shape[0], 4, 4), device=device)

        # to world coords
        means3d = means3d @ torch.linalg.inv(
            c2w[..., :3, :3]) + c2w[..., :3, 3]
        return means3d, image_coords

    def get_camera_coords(self, 
                          img_size: tuple, 
                          pixel_offset: float = 0.5) -> Tensor:
        """Generates camera pixel coordinates [W,H]

        Returns:
            stacked coords [H*W,2] where [:,0] corresponds to W and [:,1] corresponds to H
        """

        # img size is (w,h)
        image_coords = torch.meshgrid(
            torch.arange(img_size[0]),
            torch.arange(img_size[1]),
            indexing="xy",  # W = u by H = v
        )
        image_coords = (
            torch.stack(image_coords, dim=-1) + pixel_offset
        )  # stored as (x, y) coordinates
        image_coords = image_coords.view(-1, 2)
        image_coords = image_coords.float()

        return image_coords
