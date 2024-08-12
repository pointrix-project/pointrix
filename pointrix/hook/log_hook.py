import os
from rich.panel import Panel
from rich.table import Table

from .base_hook import HOOK_REGISTRY, Hook
from ..utils.visualize import visualize_depth
from ..logger.writer import Logger, ProgressLogger


@HOOK_REGISTRY.register()
class LogHook(Hook):
    """
    A hook to log the training and validation losses.
    """

    def __init__(self):
        self.ema_loss_for_log = 0.
        self.bar_info = {}

        self.losses_test = {"L1_loss": 0., "psnr": 0., "ssims": 0., "lpips": 0.}

    def before_run(self, trainner) -> None:
        """
        some print operations before the training loop starts.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """
        Pointrix_logo = r"""
                                     _____        _         _          _       
                                    |  __ \      (_)       | |        (_)      
                                    | |__) |___   _  _ __  | |_  _ __  _ __  __
                                    |  ___// _ \ | || '_ \ | __|| '__|| |\ \/ /
                                    | |   | (_) || || | | || |_ | |   | | >  < 
                                    |_|    \___/ |_||_| |_| \__||_|   |_|/_/\_\

                            A light-weight differentiable point-based rendering framework.
        """                        
                                                                                                                                                                                                  
        try:
            Logger.print(Panel(Pointrix_logo, title="Welcome to Pointrix", subtitle="Thank you"))
            Logger.log("The experiment name is {}".format(trainner.exp_dir))
        except AttributeError:
            Logger.print(
                "ERROR!!..Please provide the exp_name in config file..")
    
    def before_train(self, trainner) -> None:
        """
        some operations before the training loop starts.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """
        self.progress_bar = ProgressLogger(description='training', suffix='iter/s')
        self.progress_bar.add_task("train", "Training Progress", trainner.cfg.max_steps, log_dict={})
        self.progress_bar.add_task("validation", "Validation Progress", len(trainner.datapipeline.validation_dataset), log_dict={})
        self.progress_bar.reset("validation", visible=False)
        self.progress_bar.start()

    def after_train_iter(self, trainner) -> None:
        """
        some operations after the training iteration ends.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """
        for param_group in trainner.optimizer.param_groups:
            name = param_group['name']
            if name == "point_cloud." + "position":
                pos_lr = param_group['lr']
                break

        log_dict = {
            "num_pt": len(trainner.model.point_cloud),
            "pos_lr": pos_lr
        }
        log_dict.update(trainner.loss_dict)

        for key, value in log_dict.items():
            if key == 'loss':
                self.ema_loss_for_log = 0.4 * value.item() + 0.6 * self.ema_loss_for_log
                self.bar_info.update(
                    {key: f"{self.ema_loss_for_log:.{7}f}"})

            if trainner.writer and key != "optimizer_params":
                trainner.writer.write_scalar(key, value, trainner.global_step)

        if trainner.global_step % trainner.cfg.bar_upd_interval == 0:
            self.bar_info.update({
                "num_pts": f"{len(trainner.model.point_cloud)}",
            })
            self.progress_bar.update("train", step=trainner.cfg.bar_upd_interval, log=self.bar_info)

    def before_val(self, trainner) -> None:
        """
        some operations before the validation loop starts.

        Parameters
        ----------
        trainner : Trainer
            The trainer object.
        """
        self.progress_bar.reset("validation", visible=True)
        # if trainner.model.training_camera_model.enable_training:
        #     # matplotlib poses visualization
        #     fig = visualize_camera(trainner.model.training_camera_model)
        #     trainner.writer.write_image("camera_poses", fig, trainner.global_step)


    def after_val_iter(self, trainner) -> None:
        self.progress_bar.update("validation", step=1)
        for key, value in trainner.metric_dict.items():
            if key in self.losses_test:
                self.losses_test[key] += value

        image_name = os.path.basename(trainner.metric_dict['rgb_file_name'])
        iteration = trainner.global_step
        if 'depth' in trainner.metric_dict:
            visual_depth = visualize_depth(trainner.metric_dict['depth'].squeeze(), tensorboard=True)
            trainner.writer.write_image(
            "test" + f"_view_{image_name}/depth",
            visual_depth, step=iteration)
        trainner.writer.write_image(
            "test" + f"_view_{image_name}/render",
            trainner.metric_dict['images'].squeeze(),
            step=iteration)

        trainner.writer.write_image(
            "test" + f"_view_{image_name}/ground_truth",
            trainner.metric_dict['gt_images'].squeeze(),
            step=iteration)

    def after_val(self, trainner) -> None:
        log_info = f"[ITER {trainner.global_step}] Evaluating test:"
        table = Table(title=log_info)

        row_test = []
        for key in self.losses_test:
            self.losses_test[key] /= trainner.val_dataset_size
            trainner.writer.write_scalar(
                "test" + '/loss_viewpoint - ' + key,
                self.losses_test[key],
                trainner.global_step
            )
            log_info += f" {key} {self.losses_test[key]:.5f}"
            table.add_column(key, justify="right", style="cyan")
            row_test += [f"{self.losses_test[key]:.5f}"]
        
        table.add_row(*row_test)
        Logger.print('\n', table, '\n')
        for key in self.losses_test:
            self.losses_test[key] = 0.
        self.progress_bar.reset("validation", visible=False)
    
    def after_train(self, trainner) -> None:
        self.progress_bar.stop()
        
    def exception(self) -> None:
        self.progress_bar.stop()
