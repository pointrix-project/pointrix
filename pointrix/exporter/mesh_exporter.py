import os
import random
import numpy as np
from pathlib import Path
import open3d as o3d
from torch import Tensor
from typing import Optional, Tuple, List

import torch
import imageio
from ..exporter.base_exporter import EXPORTER_REGISTRY, BaseExporter

@EXPORTER_REGISTRY.register()
class TSDFFusion(BaseExporter):
    """
    The exporter class for the mesh export using tsdffusion.
    """
    def forward(self, output_dir):
        import vdbfusion
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        num_frames = len(self.datapipeline.iter_train_image_dataloader)  # type: ignore
        samples_per_frame = (self.cfg.total_points + num_frames) // (num_frames)
        TSDFvolume = vdbfusion.VDBVolume(
            voxel_size=self.cfg.voxel_size, sdf_trunc=self.cfg.sdf_truc, space_carving=True
        )
        points = []
        colors = []
        self.model.point_cloud.cuda()
        with torch.no_grad():
            for i, batch_data in enumerate(self.datapipeline.iter_train_image_dataloader):
                print(i)
                ## assume batch size == 1
                data = batch_data[0]
                camera = data["camera"]
                render_results = self.model(batch_data)
                # TODO
                try:
                    depth = render_results["depth"].squeeze()
                except:
                    raise ValueError('no depth in render_results,please set config --render_depth as True')
                c2w = torch.eye(4, dtype=torch.float, device=depth.device)
                c2w[:3, :4] = torch.linalg.inv(camera.extrinsic_matrix)[:3, :4]

                # c2w = c2w @ torch.diag(
                #     torch.tensor([1, -1, -1, 1], device=c2w.device, dtype=torch.float)
                # )
                c2w = c2w[:3, :4]
                H, W = int(camera.image_height), int(camera.image_width)

                indices = random.sample(range(H * W), samples_per_frame)

                xyzs, rgbs = self.get_colored_points_from_depth(
                    depths=depth,
                    rgbs=render_results["rgb"].squeeze().permute(1, 2, 0),
                    fx=camera.fx,
                    fy=camera.fy,
                    cx=camera.cx,  # type: ignore
                    cy=camera.cy,  # type: ignore
                    img_size=(W, H),
                    c2w=c2w,
                    mask=indices,
                )

                points.append(xyzs)
                colors.append(rgbs)
                TSDFvolume.integrate(
                    xyzs.double().cpu().numpy(),
                    extrinsic=c2w[:3, 3].double().cpu().numpy(),
                )
            vertices, faces = TSDFvolume.extract_triangle_mesh(min_weight=5)

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            colors = torch.cat(colors, dim=0)
            colors = colors.cpu().numpy()
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

            # simplify mesh
            if self.cfg.target_triangles is not None:
                mesh = mesh.simplify_quadric_decimation(self.cfg.target_triangles)

            o3d.io.write_triangle_mesh(
                str(output_dir / "TSDFfusion_baseline_mesh.ply"),
                mesh,
            )
            print(
                f"Finished computing mesh: {str(output_dir / 'TSDFfusion.ply')}"
            )

            mesh_clean = self.post_process_mesh(mesh, cluster_to_keep=1)
            o3d.io.write_triangle_mesh(
                str(output_dir / "TSDFfusion_baseline_mesh_clean.ply"),
                mesh_clean,
            )


    def get_colored_points_from_depth(
        self,
        depths: Tensor,
        rgbs: Tensor,
        c2w: Tensor,
        fx: float,
        fy: float,
        cx: int,
        cy: int,
        img_size: tuple,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return colored pointclouds from depth and rgb frame and c2w. Optional masking.

        Returns:
            Tuple of (points, colors)
        """
        points, _ = self.get_means3d_backproj(
            depths=depths.float(),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            img_size=img_size,
            c2w=c2w.float(),
            device=depths.device,
        )
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
    
    def post_process_mesh(self, mesh, cluster_to_keep=1000):
        """
        Post-process a mesh to filter out floaters and disconnected parts
        """
        import copy
        print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
        mesh_0 = copy.deepcopy(mesh)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
        n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
        mesh_0.remove_unreferenced_vertices()
        mesh_0.remove_degenerate_triangles()
        print("num vertices raw {}".format(len(mesh.vertices)))
        print("num vertices post {}".format(len(mesh_0.vertices)))
        return mesh_0
    
    def get_means3d_backproj(
        self,
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
        means3d = means3d @ torch.linalg.inv(c2w[..., :3, :3]) + c2w[..., :3, 3]
        return means3d, image_coords
    
    def get_camera_coords(self, img_size: tuple, pixel_offset: float = 0.5) -> Tensor:
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
