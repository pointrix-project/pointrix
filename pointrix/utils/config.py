import os
from dataclasses import dataclass, field
from datetime import datetime

from omegaconf import OmegaConf

# Basic types
from typing import Any, Optional, Union

from omegaconf import DictConfig


def C(value: Any, epoch: int, global_step: int) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError(
                "Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) /
                    (end_step - start_step)), 0.0
            )
        elif isinstance(end_step, float):
            current_step = epoch
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) /
                    (end_step - start_step)), 0.0
            )
    return value


def C_max(value: Any) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError(
                "Scalar specification only supports list, got", type(value))
        if len(value) >= 6:
            max_value = value[2]
            for i in range(4, len(value), 2):
                max_value = max(max_value, value[i])
            value = [value[0], value[1], max_value, value[3]]
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        value = max(start_value, end_value)
    return value


@dataclass
class ExperimentConfig:
    name: str = "default"
    timestamp: str = ""
    exp_dir: str = ""
    use_timestamp: bool = True
    trainer: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.use_timestamp:
            self.timestamp = datetime.now().strftime("@%Y%m%d-%H%M%S")
        self.exp_dir = os.path.join(
            self.trainer.output_path, self.name) + self.timestamp
        os.makedirs(self.exp_dir, exist_ok=True)


def load_config(*yamls: str, cli_args: list = [], from_string=False, **kwargs) -> Any:
    if from_string:
        yaml_confs = [OmegaConf.create(s) for s in yamls]
    else:
        yaml_confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    scfg = parse_structured(ExperimentConfig, cfg)
    return scfg


def config_to_primitive(config, resolve: bool = True) -> Any:
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path: str, config) -> None:
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)


def parse_structured(
    fields: Any,
    cfg: Optional[Union[dict, DictConfig]] = None
) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg
