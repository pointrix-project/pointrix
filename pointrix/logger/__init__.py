from .writer import TensorboardWriter, WandbWriter, LOGGER_REGISTRY, ProgressLogger, Logger


def parse_writer(cfg, log_dir, **kwargs):
    """
    Parse the writer.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    log_dir : str
        The log directory.
    """
    writer_type = cfg.writer_type
    writer = LOGGER_REGISTRY.get(writer_type)
    return writer(log_dir, **kwargs)
