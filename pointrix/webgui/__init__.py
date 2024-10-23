from .gui import BaseGUI, GUI_REGISTRY

__all__ = ["BaseGUI"]


def parse_gui(cfg, model, device="cuda"):
    """
    Parse the gui.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    device : str
        The device to use.
    """
    name = cfg.pop("name")
    return GUI_REGISTRY.get(name)(cfg, model, device)