import numpy as np
import torch

from ..utils.registry import Registry
from .optimizer import OptimizerList
SCHEDULER_REGISTRY = Registry("Scheduler", modules=["pointrix.optimizer"])
SCHEDULER_REGISTRY.__doc__ = ""

@SCHEDULER_REGISTRY.register()
class ExponLRScheduler:
    """
    A learning rate scheduler using exponential decay.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    lr_scale : float, optional
        The learning rate scale, by default 1.0
    """

    def __init__(self, config:dict, lr_scale=1.0) -> None:
        scheduler = self.get_exponential_lr
        self.config = config
        params = [
            {
                "name": name,
                "init": values["init"] * lr_scale,
                "final": values["final"] * lr_scale,
                "max_steps": values["max_steps"]
            }
            for name, values in config.params.items()
        ]
        self.scheduler_funcs = {}
        for param in params:
            self.scheduler_funcs[param["name"]] = (
                scheduler(
                    init_lr=param["init"],
                    final_lr=param["final"],
                    max_steps=param["max_steps"],
                )
            )

    def get_exponential_lr(self, init_lr: float, final_lr: float, max_steps: int = 1000000) -> callable:
        """
        Generates a function to compute the exponential learning rate based on the current step.

        Parameters
        ----------
        init_lr : float
            The initial learning rate.
        final_lr : float
            The final learning rate.
        max_steps : int, optional
            The maximum number of steps (default is 1000000).

        Returns
        -------
        callable
            A function that takes the current step as input and returns the learning rate for that step.
        """
        def lr_for_step(step: int) -> float:
            progress = np.clip(step / max_steps, 0, 1)
            return np.exp(np.log(init_lr) * (1 - progress) + np.log(final_lr) * progress)

        return lr_for_step
    
    def step(self, global_step: int, optimizer_list: OptimizerList) -> None:
        """
        Update the learning rate for the optimizer.

        Parameters
        ----------
        global_step : int
            The global step in training.
        optimizer_list : OptimizerList
            The list of all the optimizers which need to be updated.
        """
        for param_group in optimizer_list.param_groups:
            name = param_group['name']
            if name in self.scheduler_funcs.keys():
                lr = self.scheduler_funcs[name](global_step)
                param_group['lr'] = lr
        
