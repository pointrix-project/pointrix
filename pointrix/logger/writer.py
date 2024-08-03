from pointrix.utils.registry import Registry
import torch
import os
import shutil
from pathlib import Path
from torch import Tensor
from jaxtyping import Float
from abc import abstractmethod
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, Optional, Union

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)
from rich.text import Text

Logger = Console(width=120)

from ..utils.base import BaseObject


LOGGER_REGISTRY = Registry("LOGGER", modules=["pointrix.hook"])
LOGGER_REGISTRY.__doc__ = ""


class ItersPerSecColumn(ProgressColumn):
    """Renders the iterations per second for a progress bar."""

    def __init__(self, suffix="it/s") -> None:
        super().__init__()
        self.suffix = suffix

    def render(self, task: Task) -> Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        return Text(f"{speed:.2f} {self.suffix}", style="progress.data.speed")


class LogColumn(ProgressColumn):
    """Renders the log for a progress bar."""

    def __init__(self) -> None:
        super().__init__()

    def render(self, task: Task) -> Text:
        """Show data transfer speed."""
        text = ""
        if task.fields:
            for k, v in task.fields.items():
                text += f"{k}: {v} "
        else:
            text = ''
        return Text(text)


def logproject(file_dir: Path, save_dir: Path, suffix: str):
    """
    Filter files with suffix and save them to save_dir.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filelist = []
    for dir_path, dir_names, file_names in os.walk(file_dir):
        for file in file_names:
            file_type = file.split('.')[-1]
            if (file_type in suffix):
                file_fullname = os.path.join(dir_path, file)
                filelist.append(file_fullname)
    for file in filelist:
        shutil.copy(file, save_dir)


class ProgressLogger:
    """
    A class to log the progress of the training.

    Parameters
    ----------
    description : str
        The description of the progress.
    total_iter : int
        The total number of iterations.
    suffix : Optional[str], optional
        The suffix of the progress, by default None

    Examples
    --------
    >>> progress_logger = ProgressLogger("Training", 1000)
    >>> progress_logger.create_task("Training", 1000)
    >>> with progress_logger.progress:
    >>>     for i in range(1000):
    >>>         progress_logger.update()
    """

    def __init__(self, description: str, suffix: Optional[str] = None):
        progress_list = [TextColumn("[progress.description]{task.description}"), BarColumn(
        ), TaskProgressColumn(show_speed=True)]
        progress_list += [ItersPerSecColumn(suffix=suffix)] if suffix else []
        progress_list += [TextColumn(
            "[progress.completed]{task.completed:>6d}/{task.total:>6d}")]
        progress_list += [TimeElapsedColumn()]
        progress_list += [TimeRemainingColumn(
            elapsed_when_finished=True, compact=True)]
        progress_list += [LogColumn()]
        self.progress = Progress(*progress_list)

        self.tasks = {}

    def add_task(self, name: str, description: str, total_iter: int, log_dict: Optional[Dict[str, Any]] = {}):
        """
        Add a task to the progress.

        Parameters
        ----------
        description : str
            The description of the task.
        total_iter : int
            The total number of iterations.
        """
        return self.tasks.update({name: self.progress.add_task(description=description, total=total_iter, **log_dict)})

    def update(self, name: str, step: int = 1, log: Optional[Dict[str, Any]] = {}):
        """
        Update the progress.

        Parameters
        ----------
        step : int, optional
            The step to advance, by default 1
        log : Optional[Dict[str, Any]], optional
            The log dictionary, by default None
        """

        self.progress.update(self.tasks[name], **log)
        self.progress.advance(self.tasks[name], advance=step)

    def start(self):
        """
        Start the progress.

        Examples
        --------
        >>> progress_logger.start()
        """
        self.progress.start()

    def stop(self):
        """
        Stop the progress.

        Examples
        --------
        >>> progress_logger.stop()
        """
        self.progress.stop()

    def start_task(self, name: str):
        """
        Start the task.

        Examples
        --------
        >>> progress_logger.start_task()
        """
        self.progress.start_task(self.tasks[name])

    def stop_task(self, name: str):
        """
        Stop the task.

        Examples
        --------
        >>> progress_logger.stop_task()
        """
        self.progress.stop_task(self.tasks[name])

    def reset(self, name: str, visible: Optional[bool] = True):
        """
        Reset the progress.

        Examples
        --------
        >>> progress_logger.reset()
        """
        self.progress.reset(self.tasks[name], visible=visible)


class Writer:
    """
    Base class for writers.

    Parameters
    ----------
    log_dir : Path
        The directory to save the logs.
    """

    def __init__(self, log_dir: Path):
        self.logfolder = log_dir

    @abstractmethod
    def write_scalar(self, name: str, scalar: Union[float, torch.Tensor], step: int):
        """
        Write a scalar value to the writer.

        Parameters
        ----------
        name : str
            The name of the scalar.
        scalar : Union[float, torch.Tensor]
            The scalar value.
        step : int
            The step of the scalar.
        """
        assert NotImplementedError

    @abstractmethod
    def write_image(self, name: str, image: Float[Tensor, "H W C"], step: int, caption: Union[str, None] = None):
        """
        Write an image to the writer.

        Parameters
        ----------
        name : str
            The name of the image.
        image : Float[Tensor, "H W C"]
            The image.
        step : int
            The step of the image.
        caption : Union[str, None], optional
            The caption of the image, by default None
        """
        assert NotImplementedError

    @abstractmethod
    def write_config(self, name: str, config_dict: Dict[str, Any], step: int):
        """
        Write a config to the writer.

        Parameters
        ----------
        name : str
            The name of the config.
        config_dict : Dict[str, Any]
            The config.
        step : int
            The step of the config.
        """
        assert NotImplementedError


@LOGGER_REGISTRY.register()
class TensorboardWriter(Writer):
    """
    Tensorboard writer.

    Parameters
    ----------
    log_dir : Path
        The directory to save the logs.
    """

    def __init__(self, log_dir, **kwargs):
        self.writer = SummaryWriter(log_dir)

    def write_scalar(self, name: str, scalar: Union[float, torch.Tensor], step: int):
        """
        Write a scalar value to the writer.

        Parameters
        ----------
        name : str
            The name of the scalar.
        scalar : Union[float, torch.Tensor]
            The scalar value.
        step : int
            The step of the scalar.
        """
        self.writer.add_scalar(name, scalar, global_step=step)

    def write_image(self, name: str, image: Float[Tensor, "H W C"], step: int, caption: Union[str, None] = None):
        """
        Write an image to the writer.

        Parameters
        ----------
        name : str
            The name of the image.
        image : Float[Tensor, "H W C"]
            The image.
        step : int
            The step of the image.
        caption : Union[str, None], optional
            The caption of the image, by default None
        """
        self.writer.add_image(name, image, step)

    def write_config(self, name: str, config_dict: Dict[str, Any], step: int):
        """
        Write a config to the writer.

        Parameters
        ----------
        name : str
            The name of the config.
        config_dict : Dict[str, Any]
            The config.
        step : int
            The step of the config.
        """
        self.writer.add_text("config", str(config_dict))
        
    def finish(self):
        self.writer.close()


@LOGGER_REGISTRY.register()
class WandbWriter(Writer):
    def __init__(self, log_dir, experiment_name: str, project_name: str = "pointrix-project", logcfg: Dict[str, Any] = {}):
        """
        Wandb writer.

        Parameters
        ----------
        log_dir : Path
            The directory to save the logs.
        experiment_name : str
            The name of the experiment.
        project_name : str, 
            The name of the project, by default "pointrix-project"
        """
        import wandb
        wandb.init(project=project_name,
                   name=experiment_name,
                   dir=log_dir,
                   reinit=True,
                   config=dict(logcfg))
        
        arti_code = wandb.Artifact(experiment_name, type='code')
        arti_code.add_dir(os.path.join(log_dir, "project_file"))
        wandb.log_artifact(arti_code)

    def write_scalar(self, name: str, scalar: Union[float, torch.Tensor], step: int):
        """
        Write a scalar value to the writer.

        Parameters
        ----------
        name : str
            The name of the scalar.
        scalar : Union[float, torch.Tensor]
            The scalar value.
        step : int
            The step of the scalar.
        """
        import wandb
        wandb.log({name: scalar}, step=step)

    def write_image(self, name: str, image: Float[Tensor, "H W C"], step: int, caption=None):
        """
        Write an image to the writer.

        Parameters
        ----------
        name : str
            The name of the image.
        image : Float[Tensor, "H W C"]
            The image.
        step : int
            The step of the image.
        caption : Union[str, None], optional
            The caption of the image, by default None
        """
        import wandb
        wandb.log(
            {name: [wandb.Image(image, caption=name if caption == None else caption)]}, step=step)

    def write_config(self, name: str, config_dict: Dict[str, Any], step: int):
        """Function that writes out the config to wandb

        Args:
            config: config dictionary to write out
        """
        import wandb

        wandb.config.update(config_dict, allow_val_change=True, step=step)
    
    def finish(self):
        import wandb
        wandb.finish()
