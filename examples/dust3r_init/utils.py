
import os
import math
import json
import struct
import logging
import functools
import math
import torch
import numpy as np
from jaxtyping import Float
from typing import Union, List
from numpy.typing import NDArray
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from plyfile import PlyData, PlyElement

def read_dust3r_ply_binary(data_root):
    """
    Read the binary PLY file from dust3r dataset.
    
    Parameters
    ----------
    data_root: str
        The root directory of the dataset.
    Returns
    -------
    points_all: np.ndarray
        The point cloud data.
    """
    points_all = []
    colors_all = []

    entries = os.listdir(os.path.join(data_root, 'points'))
    entries.sort()

    for entry in entries:
        filepath = os.path.join(data_root, 'points', entry)
        with open(filepath, 'rb') as f:
            header = b""
            while True:
                line = f.readline()
                header += line
                if line.strip() == b"end_header":
                    break


            if b"binary_little_endian" in header:
                endian = '<'
            elif b"binary_big_endian" in header:
                endian = '>'
            else:
                raise ValueError("Unsupported format or incorrect header information")

            vertex_format = f'{endian}ffffff'   # x, y, z, r, g, b
            vertex_size = struct.calcsize(vertex_format)

            vertices = []
            while True:
                binary_block = f.read(vertex_size)
                if not binary_block:
                    break

                unpacked_data = struct.unpack(vertex_format, binary_block)
                vertex = list(unpacked_data[:3])  # x, y, z
                color = list(unpacked_data[3:])   # r, g, b
                vertices.append(vertex + color)

            vertices = np.array(vertices)
            points = vertices[:, 0:3]
            colors = vertices[:, 3:6]

            points_all.append(points)
            colors_all.append(colors)

    points_all = np.concatenate(points_all, axis=0)
    colors_all = np.concatenate(colors_all, axis=0)

    return points_all, colors_all