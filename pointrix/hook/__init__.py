from .base_hook import HOOK_REGISTRY
from .log_hook import LogHook
from .checkpoint_hook import CheckPointHook
from pointrix_generalize.hook import visualize_hook
from pointrix_generalize.hook import metric_hook


def parse_hooks(cfg: dict):
    """
    Parse the hooks.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    """
    if len(cfg) == 0:
        return None
    hooks = []
    for hook in cfg:
        hook_name = cfg[hook]['name']
        hook = HOOK_REGISTRY.get(hook_name)
        assert hook is not None, "Hook is not registered: {}".format(
            hook_name
        )
        hooks.append(hook())
    return hooks
