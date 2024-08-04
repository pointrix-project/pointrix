import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pointrix.utils.pose import unitquat_to_rotmat, quat_to_unitquat

sigmoid_inv = lambda x: torch.log(x/(1-x))

def k_nearest_sklearn(x: torch.Tensor, k: int):
    # Convert tensor to numpy array
    x_np = x.cpu().numpy()
    distances, _ = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np).kneighbors(x_np)
    return distances[:, 1:].astype(np.float32)

def gaussian_point_init(position, max_sh_degree, opc_init_scale=0.1):
    num_points = len(position)    
    distances= k_nearest_sklearn(position.data, 3)
    distances = torch.from_numpy(distances)
    avg_dist = distances.mean(dim=-1, keepdim=True)

    scales = torch.log(avg_dist).repeat(1, 3)
    # Efficiently create a batch of identity quaternions
    rots = torch.eye(4)[:1].repeat(num_points, 1)  
    opacities = sigmoid_inv(opc_init_scale * torch.ones((num_points, 1), dtype=torch.float32))
    features_rest = torch.zeros(
        (num_points, (max_sh_degree+1) ** 2 - 1, 3),
        dtype=torch.float32
    )

    return scales, rots, opacities, features_rest