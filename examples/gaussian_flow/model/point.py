import polyfourier
import numpy as np

import roma
import torch
import torch.nn as nn

from dataclasses import dataclass
from pointrix.model.point_cloud.gaussian_points import GaussianPointCloud
from pointrix.model.point_cloud import POINTSCLOUD_REGISTRY

from .utils import set_traj_base_dim, get_knn

@POINTSCLOUD_REGISTRY.register()
class GaussianFlowPointCloud(GaussianPointCloud):
    @dataclass
    class Config(GaussianPointCloud.Config):
        pos_traj_type: str = 'poly_fourier'
        pos_traj_dim: int = 3
        rot_traj_type: str = 'poly_fourier'
        rot_traj_dim: int = 3
        
        feat_traj_type: str = 'poly_fourier'
        feat_traj_dim: int = 3
        
        rescale_t: bool = True
        rescale_value: float = 1.0
        
        offset_t: bool = True
        offset_value: float = 0.0
        
        normliaze_rot: bool = False
        normalize_timestamp: bool = False
        
        random_noise: bool = False
        max_steps: int = 30000
        
    cfg: Config

    def setup(self, point_cloud=None):
        super().setup(point_cloud)
        
        self.rot_traj_base_dim, rot_extend_dim = set_traj_base_dim(
            self.cfg.rot_traj_type, self.cfg.rot_traj_dim, 4
        )
            
        # rots = torch.zeros((len(self), 4+rot_extend_dim))
        # rots[:, 0] = 1
        # self.rotation = nn.Parameter(
        #     rots.contiguous().requires_grad_(True)
        # )
        self.register_atribute(
            "rot_params", 
            torch.zeros((len(self), self.cfg.rot_traj_dim, 4, self.rot_traj_base_dim)),
            # torch.zeros((len(self), rot_extend_dim))
        )
            
        self.rot_fit_model = polyfourier.get_fit_model(
            type_name=self.cfg.rot_traj_type
        )
        
        # init position trajectory
        self.pos_traj_base_dim, pos_extend_dim = set_traj_base_dim(
            self.cfg.pos_traj_type, self.cfg.pos_traj_dim, 3
        )
            
        # self.position = nn.Parameter(
        #     torch.cat([
        #         self.position,
        #         torch.zeros(
        #             (len(self), pos_extend_dim),
        #             dtype=torch.float32
        #         )
        #     ], dim=1).contiguous().requires_grad_(True)
        # )
        self.register_atribute(
            "pos_params", 
            torch.zeros((len(self), self.cfg.pos_traj_dim, 3, self.pos_traj_base_dim)),
            # torch.zeros((len(self), pos_extend_dim))
        )
        self.pos_fit_model = polyfourier.get_fit_model(
            type_name=self.cfg.pos_traj_type
        )
        
        self.feat_traj_base_dim, feat_extend_dim = set_traj_base_dim(
            self.cfg.feat_traj_type, self.cfg.feat_traj_dim, 3
        )
        
        self.register_atribute(
            "feat_params", 
            torch.zeros((len(self), self.cfg.feat_traj_dim, 3, self.feat_traj_base_dim)),
            # torch.zeros((len(self), feat_extend_dim))
        )
        self.feat_fit_model = polyfourier.get_fit_model(
            type_name=self.cfg.feat_traj_type
        )
        
        self.register_atribute("time_center", torch.zeros((len(self), 1)))
        
    @torch.no_grad()
    def gen_knn(self):
        self.set_timestep(0.)
        theta_w = 100_000
        self.knn_distances_0, self.knn_indices_0 = get_knn(
            self.get_position_flow, k=20
        )
        self.knn_weights_0 = torch.exp(-theta_w*torch.pow(self.knn_distances_0, 2))

    def knn_loss(self, t):
        timestamp = self.make_time_features(t)
        if t == self.max_timestamp:
            t1 = timestamp-self.offset_width
            t2 = timestamp
        else:
            t1 = timestamp
            t2 = timestamp+self.offset_width
            
        self.fwd_flow(t1)
        t1_pos = self.get_position_flow
        t1_rot = self.get_rotation_flow
        self.fwd_flow(t2)
        t2_pos = self.get_position_flow
        t2_rot = self.get_rotation_flow
        
        t1_dist = t1_pos[self.knn_indices_0] - t1_pos.unsqueeze(1)
        t2_dist = t2_pos[self.knn_indices_0] - t2_pos.unsqueeze(1)
        
        R1 = roma.unitquat_to_rotmat(t1_rot)
        R2 = roma.unitquat_to_rotmat(t2_rot)
        R = R1 @ R2.inverse()
        
        dist = (t1_dist - (R @ t2_dist)) ** 2
        loss = (self.knn_weights_0.unsqueeze(-1) * dist).mean()
        return loss
        
    def make_time_features(self, t, training=False, training_step=0):
        # if isinstance(t, torch.Tensor):
        #     t = t.item()
            
        if self.cfg.normalize_timestamp:
            self.timestamp = t / self.max_timestamp
            self.offset_width = (1/self.max_frames)*0.1
        else:
            self.timestamp = t
            self.offset_width = 0.01
            
        if self.cfg.rescale_t:
            self.timestamp *= self.cfg.rescale_value
            self.offset_width *= self.cfg.rescale_value
            
        if self.cfg.offset_t:
            self.timestamp += self.cfg.offset_value
            
        if self.cfg.random_noise and training:
            noise_weight = self.offset_width * (1 - (training_step/self.cfg.max_steps))
            self.timestamp += noise_weight*np.random.randn()

        return self.timestamp - self.time_center.unsqueeze(0)
    
    def fwd_flow(self, timestamp_batch):
        pos_base = self.position[:, :3]
        rot_base = self.rotation[:, :4]
        
        self.position_flow_list = []
        self.rotation_flow_list = []
        self.feat_flow_list = []
        
        for i in range(timestamp_batch.size(0)):
            timestamp = timestamp_batch[i]
            pos_traj = self.pos_fit_model(
                self.pos_params, 
                # pos_traj_params,
                timestamp, 
                self.cfg.pos_traj_dim,
            )
            self.position_flow_list.append(pos_base + pos_traj)
            rot_traj = self.rot_fit_model(
                # rot_traj_params, 
                self.rot_params,
                timestamp, 
                self.cfg.rot_traj_dim,
            )
            self.rotation_flow_list.append(rot_base + rot_traj)
            
            
            feat_traj = self.feat_fit_model(
                self.feat_params, 
                timestamp, 
                self.cfg.feat_traj_dim,
            )
            self.feat_flow_list.append(self.features + feat_traj.unsqueeze(1))
        self.position_flow = torch.stack(self.position_flow_list, dim=0)
        self.rotation_flow = torch.stack(self.rotation_flow_list, dim=0)
        self.feat_flow = torch.stack(self.feat_flow_list, dim=0)
        
    def set_timestep(self, t, training=False, training_step=0):
        self.t = t
        timestamp = self.make_time_features(t, training, training_step)
        self.fwd_flow(timestamp)

    @property
    def get_rotation_flow(self):
        return self.rotation_activation(self.rotation_flow)

    @property
    def get_position_flow(self):
        return self.position_flow
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self.rotation[:, :4])
    
    @property
    def get_position(self):
        return self.position[:, :3]
    
    @property
    def get_shs_flow(self):
        return torch.cat([
            self.feat_flow, self.features_rest.unsqueeze(0),
        ], dim=2)
