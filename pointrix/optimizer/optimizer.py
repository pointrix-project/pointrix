from typing import Dict, List
import torch
from torch.optim import Optimizer
from dataclasses import dataclass
from ..utils.base import BaseObject
from ..utils.registry import Registry

class OptimizerList:
    """
    A wrapper for multiple optimizers.
    """
    def __init__(self, optimizer_dict: dict) -> None:
        """
        Parameters
        ----------
        optimizer_dict : dict
            The dictionary of the optimizers.
        """
        for key, value in optimizer_dict.items():
            assert isinstance(value, BaseOptimizer), (
                '`OptimWrapperDict` only accept BaseOptimizer instance, '
                f'but got {key}: {type(value)}')
        self.optimizer_dict = optimizer_dict
    
    def update_model(self, **kwargs) -> None:
        """
        update the model with the loss.

        Parameters
        ----------
        loss : torch.Tensor
            The loss tensor.
        kwargs : dict
            The keyword arguments.
        """
        for name, optimizer in self.optimizer_dict.items():
            optimizer.update_model(**kwargs)
    
    def state_dict(self) -> dict:
        """
        A wrapper of ``Optimizer.state_dict``.

        Returns
        -------
        dict
            The state dictionary of the optimizer.
        """
        state_dict = dict()
        for name, optimizer in self.optimizer_dict.items():
            state_dict[name] = optimizer.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        A wrapper of ``Optimizer.load_state_dict``.

        Parameters
        ----------
        state_dict : dict
            The state dictionary of the optimizer.
        """
        for name, _state_dict in state_dict.items():
            assert name in self.optimizer_dict, (
                f'Mismatched `state_dict`! cannot found {name} in '
                'OptimWrapperDict')
            self.optimizer_dict[name].load_state_dict(_state_dict)

    def __len__(self) -> int:
        """
        Get the number of the optimizers.

        Returns
        -------
        int
            The number of the optimizers.
        """
        return len(self.optimizer_dict)
    
    def __contains__(self, key: str) -> bool:
        """
        Check if the key is in the optimizer dictionary.

        Parameters
        ----------
        key : str
            The key to check.
        
        Returns
        -------
        bool
            Whether the key is in the optimizer dictionary.
        """
        return key in self.optimizer_dict
    
    @property
    def param_groups(self):
        """
        Get the parameter groups of the optimizers.

        Returns
        -------
        list
            The parameter groups of the optimizers.
        """
        param_groups = []
        for key, value in self.optimizer_dict.items():
            param_groups.extend(value.param_groups)
        return param_groups

OPTIMIZER_REGISTRY = Registry("OPTIMIZER", modules=["pointrix.optimizer"])
OPTIMIZER_REGISTRY.__doc__ = ""


@OPTIMIZER_REGISTRY.register()
class BaseOptimizer(BaseObject):
    '''
    Base class for all optimizers.
    '''
    @dataclass
    class Config:
        backward: bool = False
    cfg: Config
    
    def setup(self, optimizer:Optimizer, **kwargs):
        self.optimizer = optimizer
        self.step = 1

    def update_model(self, **kwargs) -> None:
        """
        update the model with the loss.
        you need backward first, then call this function to update the model.

        Parameters
        ----------
        loss : torch.Tensor
            The loss tensor.
        """
        with torch.no_grad():
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self) -> dict:
        """
        A wrapper of ``Optimizer.state_dict``.
        """
        state_dict = self.optimizer.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """
        A wrapper of ``Optimizer.load_state_dict``.
        """
        # load state_dict of optimizer
        self.optimizer.load_state_dict(state_dict)

    def get_lr(self)-> Dict[str, List[float]]:
        """
        Get learning rate of the optimizer.

        Returns
        -------
        Dict[str, List[float]]
            The learning rate of the optimizer.
        """
        res = {}

        res['lr'] = [group['lr'] for group in self.optimizer.param_groups]
        return res

    def get_momentum(self) -> Dict[str, List[float]]:
        """
        Get momentum of the optimizer.

        Returns
        -------
        Dict[str, List[float]]
            The momentum of the optimizer.
        """
        momentum = []
        for group in self.optimizer.param_groups:
            # Get momentum of SGD.
            if 'momentum' in group.keys():
                momentum.append(group['momentum'])
            # Get momentum of Adam.
            elif 'betas' in group.keys():
                momentum.append(group['betas'][0])
            else:
                momentum.append(0)
        return dict(momentum=momentum)
    
    @property
    def param_groups(self) -> List[dict]:
        """
        Get the parameter groups of the optimizer.

        Returns
        -------
        List[dict]
            The parameter groups of the optimizer.
        """
        return self.optimizer.param_groups
