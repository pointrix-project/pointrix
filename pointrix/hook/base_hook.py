from ..utils.registry import Registry

HOOK_REGISTRY = Registry("HOOK", modules=["pointrix.hook"])
HOOK_REGISTRY.__doc__ = ""

class Hook:
    """
    A hook is a base class that can be used to modify the behavior of the trainer.
    """
    priority = 'NORMAL'
    locations = ('before_run', 'after_load_checkpoint',
                 'before_train', 'before_train_iter', 'after_train_iter',
                 'before_val', 'before_val_iter', 'after_val_iter',
                 'after_val', 'before_save_checkpoint', 'after_train',
                 'after_run')
    def __init__(self):
        pass

    def before_run(self, trainner) -> None:
        """
        some operations before the training loop starts.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """

    def after_load_checkpoint(self, trainner, checkpoint) -> None:
        """
        some operations after the checkpoint is loaded, used for resume training.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        checkpoint : Dict
            The checkpoint loaded.
        """

    def before_train(self, trainner) -> None:
        """
        some operations before the training loop starts.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """

    def before_train_iter(self, trainner) -> None:
        """
        some operations before the training iteration starts.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """

    def after_train_iter(self, trainner) -> None:
        """
        some operations after the training iteration ends.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """

    def before_val(self, trainner) -> None:
        """
        some operations before the validation loop starts.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """

    def before_val_iter(self, trainner) -> None:
        """
        some operations before the validation iteration starts.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """

    def after_val_iter(self, trainner) -> None:
        """
        some operations after the validation iteration ends.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """

    def after_val(self, trainner) -> None:
        """
        some operations after the validation loop ends.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """

    def after_train(self, trainner) -> None:
        """
        some operations after the training loop ends.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """

    def after_run(self, trainner) -> None:
        """
        some operations after the training run ends.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """
