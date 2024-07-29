from .colmap_data import ColmapDataset
# from .nerf_data import NerfReFormat
from .base_data import DATA_SET_REGISTRY, BaseDataset


def parse_data_set(cfg: dict, device: str):
    """
    Parse the data set.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    """
    if len(cfg) == 0:
        return None
    data_set = cfg.data_set
    dataset = DATA_SET_REGISTRY.get(data_set)

    return dataset
