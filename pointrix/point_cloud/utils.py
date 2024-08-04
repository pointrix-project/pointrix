
import torch
import numpy as np
from ..logger.writer import Logger

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