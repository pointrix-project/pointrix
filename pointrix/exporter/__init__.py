from ..engine.default_datapipeline import BaseDataPipeline
from ..model.base_model import BaseModel
from .mesh_exporter import TSDFFusion
from .video_exporter import VideoExporter 
from .base_exporter import EXPORTER_REGISTRY, BaseExporter, ExporterList


def parse_exporter(configs, model: BaseModel, datapipeline: BaseDataPipeline, device="cuda"):
    """
    Parse the exporter.

    Parameters
    ----------
    configs : dict
        The configuration dictionary.
    model: BaseModel
        The model
    datapipeline : BaseDataPipeline
        The data pipeline.
    device : str
        The device to use.
    """
    exporter_dict = {}
    for name, config in configs.items():
        exporter_type = config.type
        exporter = EXPORTER_REGISTRY.get(exporter_type)

        if "extra_cfg" in config.keys():
            extra_args = getattr(config, "extra_cfg", BaseExporter.Config)
        else:
            extra_args = {}
        exporter_dict[name] = exporter(extra_args, model, datapipeline, device)
    
    return ExporterList(exporter_dict)