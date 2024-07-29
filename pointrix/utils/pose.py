import torch
import numpy as np
from typing import Tuple
from torch import Tensor
from jaxtyping import Float

def Fov2ProjectMat(fovx: float, fovy: float, near: float = 0.01, far: float = 100) -> Float[Tensor, "4 4"]:
    """
    Get the projection matrix.

    Parameters
    ----------
    fovX: float
        The field of view of the camera in x direction.
    fovY: float
        The field of view of the camera in y direction.
    near: float
        The near plane of the camera.
    far: float
        The far plane of the camera.

    Returns
    -------
    P: Float[Tensor, "4 4"]
        The projection matrix.
    Notes
    -----
    only used in the camera class

    """
    tanfovx = np.tan((fovx / 2))
    tanfovy = np.tan((fovy / 2))
    P = torch.Tensor([[2.0 * near / (tanfovx * near + tanfovx * near), 0, (tanfovx * near -tanfovx * near) / (tanfovx * near + tanfovx * near), 0],
                    [0, 2.0 * near / (tanfovy * near + tanfovy * near), (tanfovy * near -tanfovy * near) / (tanfovy * near + tanfovy * near), 0],
                    [0, 0, far / (far - near), -(far * near) / (far - near)],
                    [0, 0, 1.0, 0]])

    return P

def unitquat_to_rotmat(quat):
    """
    Converts unit quaternion into rotation matrix representation.

    Args:
        quat (...x4 tensor, XYZW convention): batch of unit quaternions.
            No normalization is applied before computation.
    Returns:
        batch of rotation matrices (...x3x3 tensor).
    """
    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/spatial/transform/rotation.py#L912
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]
    w = quat[..., 0]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    if isinstance(quat, torch.Tensor):
        matrix = torch.empty(quat.shape[:-1] + (3, 3), dtype=quat.dtype, device=quat.device)
    else:
        matrix = np.empty(quat.shape[:-1] + (3, 3), dtype=quat.dtype)
    matrix[..., 0, 0] = x2 - y2 - z2 + w2
    matrix[..., 1, 0] = 2 * (xy + zw)
    matrix[..., 2, 0] = 2 * (xz - yw)

    matrix[..., 0, 1] = 2 * (xy - zw)
    matrix[..., 1, 1] = - x2 + y2 - z2 + w2
    matrix[..., 2, 1] = 2 * (yz + xw)

    matrix[..., 0, 2] = 2 * (xz + yw)
    matrix[..., 1, 2] = 2 * (yz - xw)
    matrix[..., 2, 2] = - x2 - y2 + z2 + w2
    return matrix

def quat_to_unitquat(r):
    """
    Converts quaternion into unit quaternion representation.

    Args:
        quat (...x4 tensor, XYZW convention): batch of quaternions.
            No normalization is applied before computation.
    Returns:
        batch of unit quaternions (...x4 tensor).
    """
    norm = torch.sqrt(r[:, 0]*r[:, 0] + r[:, 1]*r[:, 1] +
                      r[:, 2]*r[:, 2] + r[:, 3]*r[:, 3])

    q = r / norm[:, None]
    return q

def ConcatRT(R: Float[Tensor, "3 3"],
            t: Float[Tensor, "3 1"]) -> Float[Tensor, "4 4"]:
    """
    Concatenate a rotation matrix `R` and a translation vector `t`
    """
    extrinsic_matrix = torch.zeros((4, 4))
    extrinsic_matrix[:3, :3] = R
    extrinsic_matrix[:3, 3] = t
    extrinsic_matrix[3, 3] = 1.0
    return extrinsic_matrix.float()


def ViewScaling(extrinsic_matrix: Float[Tensor, "4 4"],
         scale: float = 1.0,
         t: Float[Tensor, "3"] = torch.tensor([0., 0., 0.])):
    """
    Scale and translate the camera view matrix `Rt`
    """
    t = t.to(extrinsic_matrix.device)
    extrinsic_matrix_inv = torch.linalg.inv(extrinsic_matrix)
    extrinsic_matrix_inv[:3, 3] = (extrinsic_matrix_inv[:3, 3] + t) * scale
    extrinsic_matrix = torch.linalg.inv(extrinsic_matrix_inv)
    return extrinsic_matrix.float()

def GetCamcenter(Rt: Float[Tensor, "4 4"]) -> Float[Tensor, "3 1"]:
    """
    Get the camera center from the view matrix `Rt`
    """
    return Rt.transpose(0, 1).inverse()[3, :3]