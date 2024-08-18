
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from ....utils.pose import unitquat_to_rotmat, quat_to_unitquat

from ....logger.writer import Logger

def unwarp_name(name, prefix="point_cloud."):
    return name.replace(prefix, "")

def get_random_points(num_points, radius):
    pos = np.random.random((num_points, 3)) * 2 * radius - radius
    pos = torch.from_numpy(pos).float()
    return pos

def get_random_feauture(num_points, feat_dim):
    feart = np.random.random((num_points, feat_dim)) / 255.0
    feart = torch.from_numpy(feart).float()
    return feart

def points_init(init_cfg, point_cloud):
    init_type = init_cfg.init_type
    
    if init_type == 'random' and point_cloud is None:
        num_points = init_cfg.num_points
        Logger.log("Number of points at initialisation : ", num_points)
        pos = get_random_points(num_points, init_cfg.radius)
        features = get_random_feauture(num_points, init_cfg.feat_dim)
        
    else:
        Logger.log("Number of points at initialisation : ", point_cloud.positions.shape[0])
        pos = np.asarray(point_cloud.positions)
        pos = torch.from_numpy(pos).float()
        features = (torch.tensor(np.asarray(point_cloud.colors)).float() - 0.5) / 0.28209479177387814
        
        if "random" in init_type:
            num_points = init_cfg.num_points
            print("Extend the initialiased point with random : ", num_points)
            max_dis = torch.abs(pos).max().item()
            pos_ext = get_random_points(num_points, max_dis * init_cfg.radius)
            features_ext = get_random_feauture(num_points, features.shape[1])
            
            pos = torch.cat((pos, pos_ext), dim=0)
            features = torch.cat((features, features_ext), dim=0)
            
    return pos, features

ADAM_STATES = ["exp_avg", "exp_avg_sq"]

def reduce_opt_by_mask(optim, mask, param):
    state = optim.state.get(param, None)
    
    if state is not None:
        for key in ADAM_STATES:
            state[key] = state[key][mask]
        del optim.state[param]
        
    param = torch.nn.Parameter(
        param[mask].contiguous().requires_grad_(True)
    )
    
    if state is not None:
        optim.state[param] = state
        
    return param

def extend_opt_by_tensor(optim, new_tensor, param):
    state = optim.state.get(param, None)
    
    if state is not None:
        for key in ADAM_STATES:
            state[key] = torch.cat([
                state[key], torch.zeros_like(new_tensor)
            ], dim=0)
            
        del optim.state[param]
        
    param = torch.nn.Parameter(
        torch.cat([
            param, new_tensor,
        ], dim=0).contiguous().requires_grad_(True)
    )
    
    if state is not None:
        optim.state[param] = state
        
    return param

def replace_opt_tensor(optim, new_tensor, param):
    state = optim.state.get(param, None)
    
    if state is not None:
        for key in ADAM_STATES:
            state[key] = torch.zeros_like(new_tensor)
        
        del optim.state[param]

    param = torch.nn.Parameter(
        new_tensor.contiguous().requires_grad_(True)
    )
        
    if state is not None:
        optim.state[param] = state
        
    return param

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