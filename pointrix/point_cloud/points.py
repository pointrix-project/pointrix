

import os
import torch
import numpy as np
from pathlib import Path
from jaxtyping import Float
from torch import nn, Tensor
from torch.optim import Optimizer
from typing import Union
from plyfile import PlyData, PlyElement
from dataclasses import dataclass, field

from ..utils.system import mkdir_p
from ..utils.base import BaseModule
from ..utils.registry import Registry

from .utils import (
    unwarp_name,
    points_init,
    get_random_points,
    get_random_feauture,
    reduce_opt_by_mask,
    extend_opt_by_tensor,
    replace_opt_tensor,
)

POINTSCLOUD_REGISTRY = Registry("POINTSCLOUD", modules=["pointrix.model"])
POINTSCLOUD_REGISTRY.__doc__ = ""

@POINTSCLOUD_REGISTRY.register()
class PointCloud(BaseModule):
    @dataclass
    class Config:
        point_cloud_type: str = ""
        initializer: dict = field(default_factory=dict)
        trainable: bool = True
        unwarp_prefix: str = "point_cloud"
    
    cfg: Config
    
    def setup(self, point_cloud:Union[dict, None]=None) -> None:
        """
        The function for setting up the point cloud.
        
        Parameters
        ----------
        point_cloud: PointCloud
            The point cloud for initialisation.
        """
        self.atributes = []
        position, features = points_init(self.cfg.initializer, point_cloud)
        self.register_buffer('position', position)
        self.register_buffer('features', features)
        self.atributes.append({
            'name': 'position',
            'trainable': self.cfg.trainable,
        })
        self.atributes.append({
            'name': 'features',
            'trainable': self.cfg.trainable,
        })
        
        if self.cfg.trainable:
            self.position = nn.Parameter(
                position.contiguous().requires_grad_(True)
            )
            self.features = nn.Parameter(
                features.contiguous().requires_grad_(True)
            )

        self.prefix_name = self.cfg.unwarp_prefix + "."
    
    def re_init(self, num_points) -> None:
        """
        re-initialize the point cloud.
        """
        for atribute in self.atributes:
            name = atribute['name']
            delattr(self, name)
        position = get_random_points(num_points, 1.)
        features = get_random_feauture(num_points, self.cfg.initializer.feat_dim)
        self.position = nn.Parameter(
            position.contiguous().requires_grad_(True)
        )
        self.features = nn.Parameter(
            features.contiguous().requires_grad_(True)
        )
    
    def set_prefix_name(self, name:str) -> None:
        """
        set the prefix name to distinguish different point cloud.

        Parameters
        ----------
        name: str
            The prefix name.
        """
        self.prefix_name = name + "."

    def register_atribute(self, name:str, value:Float[Tensor, "3 1"], trainable=True) -> None:
        """
        register trainable atribute of the point cloud.
        
        Parameters
        ----------
        name: str
            The name of the atribute.
        value: Tensor
            The value of the atribute.
        trainable: bool
            Whether the atribute is trainable.

        Examples
        --------
        >>> point_cloud = PointsCloud(cfg)
        >>> point_cloud.register_atribute('position', position)
        >>> point_cloud.register_atribute('rgb', rgb)
        """
        self.register_buffer(name, value)
        if self.cfg.trainable and trainable:
            setattr(
                self, 
                name, 
                nn.Parameter(
                    value.contiguous().requires_grad_(True)
                )
            )
        self.atributes.append({
            'name': name,
            'trainable': trainable,
        })
            
    def __len__(self):
        return len(self.position)
    
    def unwarp(self, name) -> str:
        """
        remove the prefix name of the atribute.

        Parameters
        ----------
        name: str
            The name of the atribute.
        
        Returns
        -------
        name: str
            The name of the atribute without prefix name.
        """
        return unwarp_name(name, self.prefix_name)
    
    def set_all_atributes_trainable(self) -> None:
        """
        set all atributes of the point cloud trainable.
        """
        for atribute in self.atributes:
            name = atribute['name']
            value = getattr(self, name)
            setattr(
                self, 
                name, 
                nn.Parameter(
                    value.contiguous().requires_grad_(True)
                )
            )
    
    def get_all_atributes(self) -> list:
        """
        return all atribute of the point cloud.
        
        Returns
        -------
        atributes: list
            The list of all atributes of the point cloud.
        """
        return self.atributes
    
    def select_atributes(self, mask:Tensor) -> dict:
        """
        select atribute of the point cloud by input mask.
        
        Parameters
        ----------
        mask: Tensor
            The mask for selecting the atributes.
        
        Returns
        -------
        selected_atributes: dict
            The dict of selected atributes.
        """
        selected_atributes = {}
        for atribute in self.atributes:
            name = atribute['name']
            value = getattr(self, name)
            selected_atributes[name] = value[mask]
        return selected_atributes
    
    def replace(self, new_atributes:dict, optimizer:Union[Optimizer, None]=None) -> None:
        """
        replace atribute of the point cloud with new atribute.
        
        Parameters
        ----------
        new_atributes: dict
            The dict of new atributes.
        optimizer: Optimizer
            The optimizer for the point cloud.
        """
        if optimizer is not None:
            replace_tensor = self.replace_optimizer(
                new_atributes, 
                optimizer
            )
            for key, value in replace_tensor.items():
                setattr(self, key, value)
        else:
            for key, value in new_atributes.items():
                name = key
                value = getattr(self, name)
                replace_atribute = nn.Parameter(
                    value.contiguous().requires_grad_(True)
                )
                setattr(self, key, replace_atribute)
    
    def extand_points(self, new_atributes:dict, optimizer:Union[Optimizer, None]=None) -> None:
        """
        extand atribute of the point cloud with new atribute.
        
        Parameters
        ----------
        new_atributes: dict
            The dict of new atributes.
        optimizer: Optimizer
            The optimizer for the point cloud.
        """
        if optimizer is not None:
            extended_tensor = self.extend_optimizer(
                new_atributes, 
                optimizer
            )
            for key, value in extended_tensor.items():
                setattr(self, key, value)
        else:
            for atribute in self.atributes:
                name = atribute['name']
                value = getattr(self, name)
                extend_atribute = nn.Parameter(
                    torch.cat((
                        value, 
                        new_atributes['name']
                    ), dim=0).contiguous().requires_grad_(True)
                )
                setattr(self, key, extend_atribute)
    
    def remove_points(self, mask:Tensor, optimizer:Union[Optimizer, None]=None) -> None:
        """
        remove points of the point cloud with mask.
        
        Parameters
        ----------
        mask: Tensor
            The mask for removing the points.
        """
        if optimizer is not None:
            prune_tensor = self.prune_optimizer(
                mask, 
                optimizer
            )
            for key, value in prune_tensor.items():
                setattr(self, key, value)
        else:
            for atribute in self.atributes:
                name = atribute['name']
                prune_value = nn.Parameter(
                    getattr(
                        self, name
                    )[mask].contiguous().requires_grad_(True)
                )
                setattr(self, key, prune_value)
    
    def prune_optimizer(self, mask:Tensor, optimizer:Union[Optimizer, None])->None:
        """
        prune the point cloud in optimizer with mask.
        
        Parameters
        ----------
        mask: Tensor
            The mask for removing the points.
        optimizer: Optimizer
            The optimizer for the point cloud.
        """
        new_tensors = {}
        for group in optimizer.param_groups:
            if self.prefix_name in group["name"]:
                unwarp_ground = self.unwarp(group["name"])
                new_tensors[unwarp_ground] = reduce_opt_by_mask(
                    optimizer,
                    mask,
                    group['params'][0],
                )
                group['params'][0] = new_tensors[unwarp_ground]
        return new_tensors
    
    def extend_optimizer(self, new_atributes:dict, optimizer:Optimizer)->dict:
        """
        extend the point cloud in optimizer with new atribute.
        
        Parameters
        ----------
        new_atributes: dict
            The dict of new atributes.
        optimizer: Optimizer
            The optimizer for the point cloud.

        Return
        ------
        new_tensors: dict
        """
        new_tensors = {}
        for group in optimizer.param_groups:
            if self.prefix_name in group["name"]:
                unwarp_ground = self.unwarp(group["name"])
                extension_tensor = new_atributes[unwarp_ground]
                new_tensors[unwarp_ground] = extend_opt_by_tensor(
                    optimizer,
                    extension_tensor,
                    group['params'][0],
                )
                group['params'][0] = new_tensors[unwarp_ground]
        return new_tensors
    
    def replace_optimizer(self, new_atributes:dict, optimizer:Optimizer)->dict:
        """
        replace the point cloud in optimizer with new atribute.
        
        Parameters
        ----------
        new_atributes: dict
            The dict of new atributes.
        optimizer: Optimizer
            The optimizer for the point cloud.
        """
        new_tensors = {}
        for group in optimizer.param_groups:
            for key, replace_tensor in new_atributes.items():
                if group["name"] == self.prefix_name + key:
                    unwarp_ground = unwarp_name(group["name"])
                    new_tensors[unwarp_ground] = replace_opt_tensor(
                        optimizer,
                        replace_tensor,
                        group['params'][0],
                    )
                    group['params'][0] = new_tensors[unwarp_ground]

        return new_tensors
    
    def list_of_attributes(self) -> list:
        '''
        return the list of all attributes of the point cloud for ply saving.
        '''
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for atribute in self.atributes:
            name = atribute['name']
            if name != 'position':
                for i in range(np.prod(getattr(self, name).shape[1:])):
                    l.append('{}_{}'.format(name, i))
        return l
    
    def save_ply(self, path:Path) -> None:
        '''
        save the point cloud to ply file.

        Parameters
        ----------
        path: Path
            The path of the ply file.
        '''
        mkdir_p(os.path.dirname(path))
        num_points = self.position.shape[0]
        normals = np.zeros_like(self.position.detach().cpu().numpy())
        ply_atribute_list = [self.position.detach().cpu().numpy(), normals]

        for atribute in self.atributes:
            name = atribute['name']
            if name != 'position':
                ply_atribute_list.append(getattr(self, name).reshape(num_points, -1).detach().cpu().numpy())

        ply_atributes = np.concatenate(ply_atribute_list, axis=1)
        dtype_full = [(attribute, 'f4') for attribute in self.list_of_attributes()]

        elements = np.empty(num_points, dtype=dtype_full)
        elements[:] = list(map(tuple, ply_atributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path:Path):
        """
        load the point cloud from ply file.

        Parameters
        ----------
        path: Path
            The path of the ply file.
        
        """
        plydata = PlyData.read(path)
        if self.cfg.trainable:
            self.position = nn.Parameter(torch.from_numpy(np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)).float())
        else:
            self.position = np.stack((np.asarray(plydata.elements[0]["x"]),
                            np.asarray(plydata.elements[0]["y"]),
                            np.asarray(plydata.elements[0]["z"])), axis=1)
        
        for attribute in self.atributes:
            name = attribute['name']
            shapes = getattr(self, name).shape[1:]
            if name != 'position':
                value = np.stack([np.asarray(plydata.elements[0][name + '_{}'.format(i)]) for i in range(np.prod(shapes))], axis=1)
                if self.cfg.trainable:
                    setattr(self, name, nn.Parameter(torch.from_numpy(value.reshape(-1, *shapes)).float()))
                else:
                    setattr(self, name, value.reshape(-1, *shapes))