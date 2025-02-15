from ..engine.default_datapipeline import BaseDataPipeline
from .base_model import BaseModel, MODEL_REGISTRY
from .point_cloud.gaussian_points import GaussianPointCloud
from .point_cloud.tdgaussian_points import TDGaussianPointCloud

__all__ = ["GaussianPointCloud", "BaseModel", "TDGaussianPointCloud"]


def parse_model(cfg, datapipeline:BaseDataPipeline, device="cuda"):
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
    return MODEL_REGISTRY.get(name)(cfg, datapipeline, device)