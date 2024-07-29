import struct
import numpy as np
from dataclasses import dataclass
from pointrix.utils.pose import unitquat_to_rotmat


@dataclass
class ColmapCameraModel:
    model_id: int
    model_name: str
    num_params: int


@dataclass
class ColmapExtrinsics:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point_ids: np.ndarray


@dataclass
class ColmapIntrinsics:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


model_name = ["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "OPENCV_FISHEYE",
              "FULL_OPENCV", "FOV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "THIN_PRISM_FISHEYE"]
num_params = [3, 4, 4, 5, 8, 8, 12, 5, 4, 5, 12]

CAMERA_MODELS = [ColmapCameraModel(
    i, model_name[i], num_params[i]) for i in range(11)]
CAMERA_MODEL_IDS = {model.model_id: model for model in CAMERA_MODELS}


def read_colmap_extrinsics(colmap_file_path):
    """
    modified from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    extrinsics = {}
    with open(colmap_file_path, "rb") as fid:
        num_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_images):
            binary_image = struct.unpack("<idddddddi", fid.read(64))
            image_id = binary_image[0]
            qvec = np.array(binary_image[1:5])
            tvec = np.array(binary_image[5:8])
            camera_id = binary_image[8]
            image_name = ""
            current_char = struct.unpack("<c", fid.read(1))[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = struct.unpack("<c", fid.read(1))[0]
            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            x_y_id = struct.unpack(
                "<" + "ddq" * num_points2D, fid.read(24 * num_points2D))
            xys = np.column_stack([tuple(map(float, x_y_id[0::3])),
                                   tuple(map(float, x_y_id[1::3]))])
            point_ids = np.array(tuple(map(int, x_y_id[2::3])))
            extrinsics[image_id] = ColmapExtrinsics(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point_ids=point_ids)
    return extrinsics


def read_colmap_intrinsics(colmap_file_path):
    """
    modified from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    intrinsics = {}
    with open(colmap_file_path, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_properties = struct.unpack("<iiQQ", fid.read(24))
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = struct.unpack(
                "<" + "d" * num_params, fid.read(8 * num_params))
            intrinsics[camera_id] = ColmapIntrinsics(id=camera_id,
                                                     model=model_name,
                                                     width=width,
                                                     height=height,
                                                     params=np.array(params))
        assert len(
            intrinsics) == num_cameras, "The number of cameras does not match the number of intrinsics"
    return intrinsics

def read_3D_points_binary(point_3d_file_path):
    """
    modified from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    with open(point_3d_file_path, "rb") as file:
        num_points = struct.unpack("<Q", file.read(8))[0]

        coordinates = np.empty((num_points, 3))
        colors = np.empty((num_points, 3))

        for point_id in range(num_points):
            binary_point_line_properties = struct.unpack(
                "<QdddBBBd", file.read(43))
            coordinate = np.array(binary_point_line_properties[1:4])
            color = np.array(binary_point_line_properties[4:7])
            track_length = struct.unpack("<Q", file.read(8))[0]
            track_elements = struct.unpack(
                "<" + "ii" * track_length, file.read(8 * track_length))
            coordinates[point_id] = coordinate
            colors[point_id] = color
    return coordinates, colors

def ExtractColmapCamInfo(colmapextr, colmapintr, scale):
    height = colmapintr.height * scale
    width = colmapintr.width * scale

    R = unitquat_to_rotmat(colmapextr.qvec)
    T = np.array(colmapextr.tvec)

    fx = colmapintr.params[0] * scale
    fy = fx if colmapintr.model == "SIMPLE_PINHOLE" else colmapintr.params[1] * scale

    cx = width / 2
    cy = height / 2
    return R, T, fx, fy, cx, cy, width, height