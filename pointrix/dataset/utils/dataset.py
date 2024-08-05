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

# opengl to opencv transformation matrix
OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

def ExtractBlenderCamInfo(transform_matrix):
    """
    Extracts the rotation and translation from a Blender camera matrix.
    Args:
        transform_matrix: 4x4 matrix representing the camera pose.
    Returns:
        R: 3x3 rotation matrix.
        T: 3x1 translation vector.
    """
    transform_matrix = np.array(transform_matrix)
    transform_matrix[:3, 1:3] *= -1
    transform_matrix = np.linalg.inv(transform_matrix)

    return transform_matrix[:3, :3], transform_matrix[:3, 3]

def load_from_json(filename: Path):
    """
    load from a json file

    Parameters
    ----------
    path: Path
        the path to the json file
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)
    