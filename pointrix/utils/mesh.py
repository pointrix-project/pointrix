import glob
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import open3d as o3d
import pyrender
import torch
import trimesh
import tyro
from matplotlib import patches, pyplot as plt
from scipy.spatial import cKDTree


def get_threshold_percentage(dist, thresholds):
    """Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    """
    in_threshold = [(dist <= t).astype(np.float32).mean() for t in thresholds]
    return in_threshold


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = cKDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array([np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_colored_pcd(pcd, metric):
    cmap = plt.cm.get_cmap("jet")
    color = cmap(metric / 0.10)[..., :3]
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d


def cull_by_bounds(points, scene_bounds):
    eps = 0.02
    inside_mask = np.all(points >= (scene_bounds[0] - eps), axis=1) & np.all(
        points <= (scene_bounds[1] + eps), axis=1
    )
    return inside_mask


def compute_metrics(mesh_pred, mesh_target):
    # mesh_pred.export("coffee.ply")

    area_pred = int(mesh_pred.area * 1e4)
    area_tgt = int(mesh_target.area * 1e4)
    print("pred: {}, target: {}".format(area_pred, area_tgt))

    # iou, v_gt, v_pred = compute_iou(mesh_pred, mesh_target)

    pointcloud_pred, idx = mesh_pred.sample(area_pred, return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normals_pred = mesh_pred.face_normals[idx]

    pointcloud_tgt, idx = mesh_target.sample(area_tgt, return_index=True)
    pointcloud_tgt = pointcloud_tgt.astype(np.float32)
    normals_tgt = mesh_target.face_normals[idx]

    thresholds = np.array([0.05])

    # for every point in gt compute the min distance to points in pred
    completeness, completeness_normals = distance_p2p(
        pointcloud_tgt, normals_tgt, pointcloud_pred, normals_pred
    )
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness**2

    # color gt_point_cloud using completion
    com_mesh = get_colored_pcd(pointcloud_tgt, completeness)

    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()

    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(
        pointcloud_pred, normals_pred, pointcloud_tgt, normals_tgt
    )
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy**2

    # color pred_point_cloud using completion
    acc_mesh = get_colored_pcd(pointcloud_pred, accuracy)

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()

    # Chamfer distance
    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals
    chamferL1 = 0.5 * (completeness + accuracy)

    # F-Score
    F = [
        2 * precision[i] * recall[i] / (precision[i] + recall[i])
        for i in range(len(precision))
    ]
    rst = {
        "Acc": accuracy,
        "Comp": completeness,
        "C-L1": chamferL1,
        "NC": normals_correctness,
        "F-score": F[0],
    }

    return rst


def load_poses(posedir):
    poses = []
    names = []
    pose_list = sorted(
        glob.glob(os.path.join(posedir, "*.txt")),
        key=lambda x: int(os.path.basename(x)[:-4]),
    )

    for item in pose_list:
        c2w = np.loadtxt(item).astype(np.float64).reshape(4, 4)
        # c2w = np.matmul(TRANSFORM_WORLD, c2w)
        poses.append(c2w)
        names.append(item.split("/")[-1].split(".txt")[0])

    return poses


def load_from_json(filename):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def axis_angle_to_matrix(data):
    batch_dims = data.shape[:-1]

    theta = torch.norm(data, dim=-1, keepdim=True)
    omega = data / theta

    omega1 = omega[..., 0:1]
    omega2 = omega[..., 1:2]
    omega3 = omega[..., 2:3]
    zeros = torch.zeros_like(omega1)

    K = torch.cat(
        [
            torch.cat([zeros, -omega3, omega2], dim=-1)[..., None, :],
            torch.cat([omega3, zeros, -omega1], dim=-1)[..., None, :],
            torch.cat([-omega2, omega1, zeros], dim=-1)[..., None, :],
        ],
        dim=-2,
    )
    I = torch.eye(3, device=data.device).expand(*batch_dims, 3, 3)

    return (
        I
        + torch.sin(theta).unsqueeze(-1) * K
        + (1.0 - torch.cos(theta).unsqueeze(-1)) * (K @ K)
    )


def pose6d_to_matrix(batch_poses):
    c2w = torch.eye(4).unsqueeze(0).repeat(batch_poses.shape[0], 1, 1)
    c2w[:, :3, :3] = axis_angle_to_matrix(batch_poses[:, :, 0])
    c2w[:, :3, 3] = batch_poses[:, :, 1]
    return c2w


def render_depth_maps(mesh, poses, H, W, K, far=10.0):
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.IntrinsicsCamera(
        fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.01, zfar=far
    )
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    renderer = pyrender.OffscreenRenderer(W, H)
    render_flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY

    depth_maps = []
    for i in range(poses.shape[0]):
        scene.set_pose(camera_node, poses[i])
        depth = renderer.render(scene, render_flags)
        depth_maps.append(depth)

    return depth_maps


def render_depth_maps_doublesided(mesh, poses, H, W, K, far=10.0):
    depth_maps_1 = render_depth_maps(mesh, poses, H, W, K, far=far)
    mesh.faces[:, [1, 2]] = mesh.faces[:, [2, 1]]
    depth_maps_2 = render_depth_maps(mesh, poses, H, W, K, far=far)
    mesh.faces[:, [1, 2]] = mesh.faces[
        :, [2, 1]
    ]  # it's a pass by reference, so I restore the original order

    depth_maps = []
    for i in range(len(depth_maps_1)):
        depth_map = np.where(depth_maps_1[i] > 0, depth_maps_1[i], depth_maps_2[i])
        depth_map = np.where(
            (depth_maps_2[i] > 0) & (depth_maps_2[i] < depth_map),
            depth_maps_2[i],
            depth_map,
        )
        depth_maps.append(depth_map)

    return depth_maps


def cull_from_one_pose(
    points,
    pose,
    H,
    W,
    K,
    rendered_depth=None,
    depth_gt=None,
    remove_missing_depth=True,
    remove_occlusion=True,
):
    c2w = deepcopy(pose)
    # to OpenCV
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1
    w2c = np.linalg.inv(c2w)
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]

    # pts under camera frame
    camera_space = rotation @ points.transpose() + translation[:, None]  # [3, N]
    uvz = (K @ camera_space).transpose()  # [N, 3]
    pz = uvz[:, 2] + 1e-8
    px = uvz[:, 0] / pz
    py = uvz[:, 1] / pz

    # step 1: inside frustum
    in_frustum = (0 <= px) & (px <= W - 1) & (0 <= py) & (py <= H - 1) & (pz > 0)
    u = np.clip(px, 0, W - 1).astype(np.int32)
    v = np.clip(py, 0, H - 1).astype(np.int32)
    eps = 0.02
    obs_mask = in_frustum
    # step 2: not occluded
    if remove_occlusion:
        obs_mask = in_frustum & (
            pz < (rendered_depth[v, u] + eps)
        )  # & (depth_gt[v, u] > 0.)

    # step 3: valid depth in gt
    if remove_missing_depth:
        invalid_mask = in_frustum & (depth_gt[v, u] <= 0.0)
    else:
        invalid_mask = np.zeros_like(obs_mask)

    return obs_mask.astype(np.int32), invalid_mask.astype(np.int32)


def get_grid_culling_pattern(
    points,
    poses,
    H,
    W,
    K,
    rendered_depth_list=None,
    depth_gt_list=None,
    remove_missing_depth=True,
    remove_occlusion=True,
    verbose=False,
):
    obs_mask = np.zeros(points.shape[0])
    invalid_mask = np.zeros(points.shape[0])
    for i in range(poses.shape[0]):
        if verbose:
            print("Processing pose " + str(i + 1) + " out of " + str(poses.shape[0]))
        rendered_depth = (
            rendered_depth_list[i] if rendered_depth_list is not None else None
        )
        depth_gt = depth_gt_list[i] if depth_gt_list is not None else None
        obs, invalid = cull_from_one_pose(
            points,
            poses[i],
            H,
            W,
            K,
            rendered_depth=rendered_depth,
            depth_gt=depth_gt,
            remove_missing_depth=remove_missing_depth,
            remove_occlusion=remove_occlusion,
        )
        obs_mask = obs_mask + obs
        invalid_mask = invalid_mask + invalid

    return obs_mask, invalid_mask


def cull_mesh(mesh_pred, ref_poses, depth_gt_list, H, W, K):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    # cull with subdivide
    vertices = mesh_pred.vertices
    triangles = mesh_pred.faces
    vertices, triangles = trimesh.remesh.subdivide_to_size(
        vertices, triangles, max_edge=0.015, max_iter=10
    )

    # cull with unseen faces
    rendered_depth_maps = render_depth_maps_doublesided(
        mesh_pred, ref_poses, H, W, K, far=10.0
    )

    # we don't need subdivided mesh to render depth
    mesh_pred = trimesh.Trimesh(vertices, triangles, process=False)
    mesh_pred.remove_unreferenced_vertices()

    # # Cull faces
    points = vertices[:, :3]
    obs_mask, invalid_mask = get_grid_culling_pattern(
        points,
        ref_poses,
        H,
        W,
        K,
        rendered_depth_list=rendered_depth_maps,
        depth_gt_list=depth_gt_list,
        remove_missing_depth=True,
        remove_occlusion=True,
        verbose=True,
    )
    obs1 = obs_mask[triangles[:, 0]]
    obs2 = obs_mask[triangles[:, 1]]
    obs3 = obs_mask[triangles[:, 2]]
    th1 = 3
    obs_mask = (obs1 > th1) | (obs2 > th1) | (obs3 > th1)
    inv1 = invalid_mask[triangles[:, 0]]
    inv2 = invalid_mask[triangles[:, 1]]
    inv3 = invalid_mask[triangles[:, 2]]
    invalid_mask = (inv1 > 0.7 * obs1) & (inv2 > 0.7 * obs2) & (inv3 > 0.7 * obs3)
    valid_mask = obs_mask & (~invalid_mask)
    triangles_in_frustum = triangles[valid_mask, :]

    mesh_pred = trimesh.Trimesh(vertices, triangles_in_frustum, process=False)
    mesh_pred.remove_unreferenced_vertices()

    return mesh_pred


def cull_mesh_iphone(mesh_pred):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    # cull with subdivide
    vertices = mesh_pred.vertices
    triangles = mesh_pred.faces
    vertices, triangles = trimesh.remesh.subdivide_to_size(
        vertices, triangles, max_edge=0.015, max_iter=10
    )

    # we don't need subdivided mesh to render depth
    mesh_pred = trimesh.Trimesh(vertices, triangles, process=False)
    mesh_pred.remove_unreferenced_vertices()

    return mesh_pred


def cut_projected_mesh(projection, predicted_mesh, type, kernel_size, dilate=True):

    max_val = projection.max(axis=0)
    min_val = projection.min(axis=0)
    projection = ((projection - min_val) / (max_val - min_val) * 499).astype(np.int32)

    image = np.zeros((500, 500), dtype=np.uint8)

    for x, y in projection:
        image[y, x] = 255

    if kernel_size != None:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if dilate:
            rescale_image = cv2.dilate(image, kernel, iterations=1)
        elif dilate == False:
            rescale_image = cv2.erode(image, kernel, iterations=1)

        contours, _ = cv2.findContours(
            rescale_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
    else:
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    result = np.ones((500, 500, 3), dtype=np.uint8) * 255

    for x, y in projection:
        result[y, x] = [238, 215, 189]

    # cv2.drawContours(result, contours, -1, (0, 0, 255), 1)

    # Convert contour points back to their original scale
    contour_points = [
        np.array(c).squeeze() * (max_val - min_val) / 499 + min_val for c in contours
    ]

    cloud_points = np.asarray(predicted_mesh.vertices)
    inside = np.zeros(len(cloud_points), dtype=bool)
    if type == "xy":
        project_points = cloud_points[:, :2]
    elif type == "xz":
        project_points = cloud_points[:, [0, 2]]
    elif type == "yz":
        project_points = cloud_points[:, 1:]

    inside = np.array(
        [
            any(
                patches.Path(contour).contains_point(point)
                for contour in contour_points
                if len(contour.shape) >= 2
            )
            for point in project_points
        ]
    )

    filtered_cloud = cloud_points[inside]

    # Visualize

    # plt.scatter(filtered_cloud[:, 0], filtered_cloud[:, 2], s=1)
    # plt.show()
    # exit()

    old_to_new_indices = {old: new for new, old in enumerate(np.where(inside)[0])}

    triangles = np.asarray(predicted_mesh.triangles)
    for i in range(triangles.shape[0]):
        for j in range(3):
            if triangles[i, j] in old_to_new_indices:
                triangles[i, j] = old_to_new_indices[triangles[i, j]]
            else:
                triangles[i, j] = -1

    valid_triangles = (triangles != -1).all(axis=1)
    filtered_triangles = triangles[valid_triangles]

    filtered_mesh = o3d.geometry.TriangleMesh()
    filtered_mesh.vertices = o3d.utility.Vector3dVector(filtered_cloud)
    filtered_mesh.triangles = o3d.utility.Vector3iVector(filtered_triangles)

    return filtered_mesh


def cut_mesh(gt_mesh, pred_mesh, kernel_size, dilate=True):
    vertices = np.asarray(gt_mesh.vertices)
    # Extract vertex data and project it onto XY plane
    print("cutting xy plane")
    vertices_2d = vertices[:, :2]  # Keep only X and Y coordinates
    filtered_mesh = cut_projected_mesh(
        vertices_2d, pred_mesh, "xy", kernel_size, dilate=dilate
    )

    # Keep only X and Z coordinates

    print("cutting xz plane")
    vertices_2d = vertices[:, [0, 2]]
    filtered_mesh = cut_projected_mesh(
        vertices_2d, filtered_mesh, "xz", kernel_size, dilate=dilate
    )

    # Keep only Y and Z coordinates
    print("cutting yz plane")
    vertices_2d = vertices[:, 1:]
    filtered_mesh = cut_projected_mesh(
        vertices_2d, filtered_mesh, "yz", kernel_size, dilate=dilate
    )

    return filtered_mesh


def open3d_mesh_from_trimesh(tri_mesh):
    vertices = np.asarray(tri_mesh.vertices)
    faces = np.asarray(tri_mesh.faces)

    # Create open3d TriangleMesh object
    o3d_mesh = o3d.geometry.TriangleMesh()

    # Assign vertices and faces to open3d mesh
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    return o3d_mesh


def trimesh_from_open3d_mesh(open3d_mesh):
    vertices = np.asarray(open3d_mesh.vertices)
    faces = np.asarray(open3d_mesh.triangles)

    # Create open3d TriangleMesh object
    tri_mesh = trimesh.Trimesh()
    tri_mesh.vertices = vertices
    tri_mesh.faces = faces
    return tri_mesh


def mesh_metric(
    gt_mesh_path: Path,  # path to gt mesh folder
    pred_mesh_path: Path,  # path to the pred mesh folder\
    cut: bool = False
):
    if cut: 
        gt_mesh_path = Path(gt_mesh_path)
        gt_mesh = o3d.io.read_triangle_mesh(str(gt_mesh_path / Path("gt_mesh.ply")))
        gt_mesh = gt_mesh.remove_unreferenced_vertices()
        pred_mesh_path = str(pred_mesh_path) if '.ply' in str(pred_mesh_path) else str(pred_mesh_path / Path("TSDFfusion_baseline_mesh.ply"))
        pred_mesh = trimesh.load(
            pred_mesh_path,
            force="mesh",
            process=False,
        )
        nerf_scale = 1.0
        scale_mat = np.eye(4).astype(np.float32)
        scale_mat[:3] *= nerf_scale
        align_T = np.linalg.inv(scale_mat)
        pred_mesh.apply_transform(align_T)
        # transfer mesh to align gt mesh
        initial_transformation = np.array(
            json.load(open(gt_mesh_path / Path("icp_iphone.json")))["gt_transformation"]
        ).reshape(4, 4)
        pred_mesh = pred_mesh.apply_transform(initial_transformation)

        pred_mesh = open3d_mesh_from_trimesh(pred_mesh)
        pred_mesh = cut_mesh(gt_mesh, pred_mesh, kernel_size=15, dilate=True)

        # evaluate culled mesh
        
        # pred_mesh_path = "/".join(pred_mesh_path.split("/")[:-1])
        # o3d.io.write_triangle_mesh(
        #     str(pred_mesh_path / Path("poisson_mesh_cull.ply")), pred_mesh
        # )
        print("finished save and cut the mesh")

        gt_mesh = trimesh_from_open3d_mesh(gt_mesh)
        pred_mesh = trimesh_from_open3d_mesh(pred_mesh)

        rst = compute_metrics(pred_mesh, gt_mesh)
        return rst
    else:
        gt_mesh_ply = trimesh.load(gt_mesh_path, process=False)
        pd_mesh_ply = trimesh.load(pred_mesh_path, process=False)

        rst = compute_metrics(pd_mesh_ply, gt_mesh_ply)
        print(rst)
        return rst


if __name__ == "__main__":
    tyro.cli(mesh_metric)