from .msplat import RENDERER_REGISTRY, MsplatRender
from .gsplat import GsplatRender
from .dptr import DPTRRender

def parse_renderer(cfg, **kwargs):
    """
    Parse the renderer.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    """
    name = cfg.pop("name")
    if name == 'GaussianSplattingRender':
        from .base_splatting import GaussianSplattingRender
    return RENDERER_REGISTRY.get(name)(cfg, **kwargs)