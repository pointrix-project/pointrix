from ..engine.default_datapipeline import BaseDataPipeline
from ..model.base_model import BaseModel
from .mesh_exporter import TSDFFusion
from .video_exporter import VideoExporter 
from .base_exporter import EXPORTER_REGISTRY, BaseExporter



def parse_exporter(cfg, model: BaseModel, datapipeline: BaseDataPipeline, device="cuda"):
    """
    Parse the exporter.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    model: BaseModel
        The model
    datapipeline : BaseDataPipeline
        The data pipeline.
    device : str
        The device to use.
    """
    name = cfg.pop("name")
    return EXPORTER_REGISTRY.get(name)(cfg, model, datapipeline, device)