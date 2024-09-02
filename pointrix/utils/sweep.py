import copy

def convert2wandb(sweep_config, config):
    config_type = type(config)
    def collect_paths(d, d_sweep, path=[]):
        paths = []
        if isinstance(d, config_type):
            for key, value in d.items():
                if key in d_sweep.keys():
                    new_path = path + [key]
                    if isinstance(value, config_type):
                        paths.extend(collect_paths(value, d_sweep[key], new_path))
                    else:
                        paths.append(new_path)
        return paths
    wandb_config = copy.deepcopy(sweep_config)
    wandb_config["parameters"] = {}
    param_paths = collect_paths(config, sweep_config["parameters"])
    
    for path in param_paths:
        sw_config = copy.deepcopy(sweep_config["parameters"])
        for p in path:
            sw_config = sw_config[p]
        path_name = "_".join(path)
        wandb_config["parameters"].update({path_name: sw_config})
    
    return wandb_config, param_paths