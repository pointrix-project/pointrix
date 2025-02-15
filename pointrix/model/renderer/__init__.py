from .msplat import RENDERER_REGISTRY, MsplatRender


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
    elif name == 'GsplatRender':
        from .gsplat import GsplatRender
    elif name == 'TDGSRender':
        from .tdgs_splatting import TDGSRender
    return RENDERER_REGISTRY.get(name)(cfg, **kwargs)