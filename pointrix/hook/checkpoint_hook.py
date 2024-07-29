import os
import torch
from .base_hook import HOOK_REGISTRY, Hook


@HOOK_REGISTRY.register()
class CheckPointHook(Hook):
    """
    A hook to save the checkpoint during the training loop.
    """
    def after_train_iter(self, trainner) -> None:
        """
        some operations after the training iteration ends.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """
        if trainner.global_step % 5000 == 0:
            trainner.model.point_cloud.save_ply(os.path.join(
                trainner.exp_dir, "{:0>5}.ply".format(trainner.global_step)))
            trainner.save_model()

    def after_train(self, trainner) -> None:
        """
        some operations after the training loop ends.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """
        data_list = {
            "global_step": trainner.global_step,
            "optimizer": trainner.optimizer.state_dict(),
            "point_cloud": trainner.model.point_cloud.state_dict(),
        }

        path = os.path.join(
            trainner.exp_dir, 
            "chkpnt" + "{:0>5}.pth".format(trainner.global_step))
        torch.save(data_list, path)