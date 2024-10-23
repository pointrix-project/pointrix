from .base import BaseDensificationController, CONTROLER_REGISTRY
from .gs import DensificationController

__all__ = ["DensificationController", "BaseDensificationController"]

def parse_controller(cfg, optimizer, model, **kwargs):
    """
    Parse the model.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    datapipeline : BaseDataPipeline
        The data pipeline.
    device : str
        The device to use.
    """
    name = cfg.pop("name")
    return CONTROLER_REGISTRY.get(name)(cfg,  optimizer, model, **kwargs)