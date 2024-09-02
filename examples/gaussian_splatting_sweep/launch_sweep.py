import os
import wandb
import copy
import warnings
import argparse
from functools import partial
from pointrix.utils.config import load_config, load_yaml
from pointrix.engine.default_trainer import DefaultTrainer
from pointrix.logger.writer import logproject, Logger
from pointrix.utils.sweep import convert2wandb


def main(cfg, param_paths) -> None:
    run = wandb.init()
    warnings.filterwarnings("ignore")
    project_path = os.path.dirname(os.path.abspath(__file__))

    logproject(project_path, os.path.join(cfg.exp_dir, 'project_file'), ['py', 'yaml'])

    trainer_cfg = cfg.trainer
    # change the config values to the wandb search values
    for path in param_paths:
        cfg_search = trainer_cfg
        for p in path[:-1]:
            cfg_search = cfg_search[p]
        
        path_name = "_".join(path)
        cfg_search[path[-1]] = wandb.config.__getattr__(path_name)
    try:
        gaussian_trainer = DefaultTrainer(
                            cfg.trainer,
                            cfg.exp_dir,
                            cfg.name
                            )
        if cfg.trainer.training:
            gaussian_trainer.train_loop()
            model_path = os.path.join(
                cfg.exp_dir,
                "chkpnt" + str(gaussian_trainer.global_step) + ".pth"
            )
            gaussian_trainer.save_model(path=model_path)
            gaussian_trainer.test()
        else:
            gaussian_trainer.test(cfg.trainer.test_model_path)
        wandb.log({"psnr": gaussian_trainer.exporter.exporter_dict["exporter_1"].psnr_metric, "ssim": gaussian_trainer.exporter.exporter_dict["exporter_1"].ssim_metric,
                   "lpips": gaussian_trainer.exporter.exporter_dict["exporter_1"].lpips_metric, "l1_loss": gaussian_trainer.exporter.exporter_dict["exporter_1"].l1})
        Logger.print("\nTraining complete.")
    except:
        Logger.print_exception(show_locals=False)
        for hook in gaussian_trainer.hooks:
            hook.exception()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--config_sweep", required=True, help="path to config file")
    parser.add_argument("--smc_file", type=str, default=None)
    args, extras = parser.parse_known_args()
    
    sweep_config = load_yaml(args.config_sweep)
    cfg = load_config(args.config, cli_args=extras)
    
    sweep_config, param_paths = convert2wandb(sweep_config, cfg.trainer)
    sweep_id = wandb.sweep(sweep=sweep_config, project="pointrix")

    print("Sweep ID:", sweep_id)
    main_func = partial(main, cfg=cfg, param_paths=param_paths)
    
    wandb.agent(sweep_id, function=main_func, count=30, project="pointrix")
