import os
import json
import vdbfusion
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import open3d as o3d
from torch import Tensor
from typing import Optional, Tuple, List
from typing import Literal, Optional, Tuple
import torch
import imageio

from ..exporter.base_exporter import EXPORTER_REGISTRY, MetricExporter
from ..logger import ProgressLogger, Logger
from ..utils.mesh import mesh_metric

def pick_indices_at_random(valid_mask, samples_per_frame):
    indices = torch.nonzero(torch.ravel(valid_mask))
    if samples_per_frame < len(indices):
        which = torch.randperm(len(indices))[:samples_per_frame]
        indices = indices[which]
    return torch.ravel(indices)

def post_process_mesh(mesh, cluster_to_keep=1000):
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

@EXPORTER_REGISTRY.register()
class DepthAndNormalMapsPoisson(MetricExporter):
    """
    The exporter class for the mesh export using Poisson reconstruction.
    modified from https://github.com/maturk/dn-splatter/blob/main/dn_splatter/export_mesh.py
    """
    
    @dataclass
    class Config(MetricExporter.Config):
        total_points: int = 2_000_000
        use_masks: bool = True
        filter_edges_from_depth_maps: bool = False
        down_sample_voxel: Optional[float] = None
        outlier_removal: bool = False
        std_ratio: float = 2.0
        edge_threshold: float = 0.004
        edge_dilation_iterations: int = 10
        poisson_depth: int = 9
        gt_mesh_path: Optional[str] = None
        cut: bool = False
        
    def forward(self, output_dir: str):
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        train_data_loader = iter(self.datapipeline.training_loader)
        frame_count = len(train_data_loader)
        samples_per_frame = (self.cfg.total_points + frame_count) // frame_count
        Logger.print("samples per frame: ", samples_per_frame)
        
        points_list = []
        normals_list = []
        colors_list = []
        
        self.model.point_cloud.cuda()
        
        with torch.no_grad():
            progress_logger = ProgressLogger(
                description='Extracting mesh using DepthAndNormalMapsPoisson', suffix='iters/s')
            progress_logger.add_task(f'Mesh', f'Extracting mesh using DepthAndNormalMapsPoisson', len(
                train_data_loader))
            
            with progress_logger.progress as progress:
                for i, batch in enumerate(train_data_loader):
                    
                    data = batch[0]
                    camera_info = self.model(batch, render=False)
                    render_output = self.model(batch)
                    try:
                        depth_map = render_output["depth"].squeeze()
                    except:
                        raise ValueError(
                            'No depth in render_output, please set config trainer.model.renderer.render_depth as True')

                    camera_to_world = torch.linalg.inv(
                        camera_info["extrinsic_matrix"].squeeze(0))[:3, :4]
                    
                    height, width = int(data['height']), int(data['width'])
                    
                    valid_depth = depth_map
                    valid_mask = valid_depth
                    
                    indices = pick_indices_at_random(valid_mask, samples_per_frame)
                    
                    if len(indices) == 0:
                        continue
                    intrinsic_params = camera_info["intrinsic_params"].squeeze(0).cpu().numpy()
                    
                    fx, fy, cx, cy = intrinsic_params[0], intrinsic_params[1], intrinsic_params[2], intrinsic_params[3]
                    
                    points, colors = self.get_colored_points_from_depth(
                        depths=depth_map,
                        rgbs=render_output["rgb"].squeeze().permute(1, 2, 0),
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                        img_size=(width, height),
                        c2w=camera_to_world,
                        mask=indices,
                    )
                    
                    normal_map = render_output["normal"].squeeze(0)
                    _, h, w = normal_map.shape
                    normal_map = normal_map.permute(1, 2, 0)
                    normal_map = normal_map.view(-1, 3)
                    
                    normal_map = normal_map @ torch.diag(
                        torch.tensor(
                            [-1, -1, -1], device=normal_map.device, dtype=torch.float
                        )
                    )
                    
                    normal_map = normal_map.view(h, w, 3)
                    
                    rot = camera_to_world[:3, :3]
                    normal_map = normal_map.permute(2, 0, 1).reshape(3, -1)
                    normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)
                    normal_map = rot @ normal_map
                    normal_map = normal_map.permute(1, 0).reshape(h, w, 3)

                    normal_map = normal_map.view(-1, 3)[indices]
                
                    points_list.append(points)
                    colors_list.append(colors)
                    normals_list.append(normal_map)
                    
                    progress_logger.update(f'Mesh', step=1)
            
            points = torch.cat(points_list, dim=0)
            colors = torch.cat(colors_list, dim=0)
            normals = torch.cat(normals_list, dim=0)
            
            points = points.cpu().numpy()
            normals = normals.cpu().numpy()
            colors = colors.cpu().numpy()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(
                str(output_dir / "DepthAndNormalMapsPoisson_pcd.ply"), pcd
            )
            Logger.print("Computing Mesh... this may take a while.")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=self.cfg.poisson_depth
            )
            vertices_to_remove = densities < np.quantile(densities, 0.05)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            Logger.print("[bold green]:white_check_mark: Computing Mesh")

            Logger.print(
                f"Saving Mesh to {str(output_dir / 'DepthAndNormalMapsPoisson_poisson_mesh.ply')}"
            )
            o3d.io.write_triangle_mesh(
                str(output_dir / "DepthAndNormalMapsPoisson_poisson_mesh.ply"),
                mesh,
            )
            
            mesh_clean = post_process_mesh(mesh, cluster_to_keep=1)
            o3d.io.write_triangle_mesh(
                str(output_dir / "DepthAndNormalMapsPoisson_poisson_mesh_clean.ply"),
                mesh_clean,
            )
            
            self.mesh_results = mesh_metric(self.cfg.gt_mesh_path, output_dir / "DepthAndNormalMapsPoisson_poisson_mesh.ply", cut=self.cfg.cut)
            json.dump(self.mesh_results, open(output_dir / Path("metrics_point.json"), "w"))
            print(self.mesh_results)
            
                    

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
        


@EXPORTER_REGISTRY.register()
class TSDFFusion(MetricExporter):
    """
    The exporter class for the mesh export using tsdffusion.
    modified from https://github.com/maturk/dn-splatter/blob/main/dn_splatter/export_mesh.py
    """
    @dataclass
    class Config(MetricExporter.Config):
        gt_mesh_path: str = "/NASdata/clz/data/mushroom/vr_room"
    cfg: Config

    
    def forward(self, output_dir: str):
        output_dir = Path(output_dir) / Path("mesh")
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
                    camera_info = self.model(batch, render=False)
                    render_output = self.model(batch)
                    try:
                        depth_map = render_output["depth"].squeeze()
                    except:
                        raise ValueError(
                            'No depth in render_output, please set config trainer.model.renderer.render_depth as True')

                    camera_to_world = torch.linalg.inv(
                        camera_info["extrinsic_matrix"].squeeze(0))[:3, :4]
                    height, width = int(data['height']), int(data['width'])
                    intrinsic_params = camera_info["intrinsic_params"].squeeze(0).cpu().numpy()
                    fx, fy, cx, cy = intrinsic_params[0], intrinsic_params[1], intrinsic_params[2], intrinsic_params[3]
                    sampled_indices = random.sample(
                        range(height * width), samples_per_frame)

                    points, colors = self.get_colored_points_from_depth(
                        depths=depth_map,
                        rgbs=render_output["rgb"].squeeze().permute(1, 2, 0),
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
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
                    str(output_dir / Path("TSDFfusion_baseline_mesh.ply")),
                    mesh,
                )
                print(
                    f"Finished computing mesh: {str(output_dir / Path('TSDFfusion_baseline_mesh.ply'))}"
                )
                
        self.mesh_results = mesh_metric(self.cfg.gt_mesh_path, output_dir)
        print(self.mesh_results)
      
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
