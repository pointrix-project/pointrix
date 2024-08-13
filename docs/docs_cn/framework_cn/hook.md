# 钩子函数

钩子模块可以作为一个工具类，在某些固定的时间点对训练器进行一些操作。CheckPointHook 和 LogHook 是两个继承自基类的示例钩子，它们分别用于在训练循环中保存检查点和记录训练和验证损失。

LogHook 依赖于一个名为 Logger 的 Console 对象，在终端打印一些日志信息，以及一个 ProgressLogger 对象来可视化训练进度。在指定的时间点，它将调用训练器的 writer 来记录日志信息。

对于与训练器绑定的 writer，它应该继承自基类 `Writer`，并完成 `write_scalar()`、`write_image()` 和 `write_config()` 这三个抽象函数。您可以创建更多类型的 Writer。例如，TensorboardWriter 通过重写这三个接口函数封装了 `torch.utils.tensorboard.SummaryWriter`。您还可以在 `.yaml` 配置文件中方便地指定要使用的 writer 类型。

## 使用场合

在训练器中，钩子函数将在特定位置被调用：

```{code-block} python
:lineno-start: 1 
:emphasize-lines: "26, 35, 40, 48, 50, 56, 59, 61, 62"
:caption: |
:    We *highlight* the hook part.

class DefaultTrainer:
    """
    The default trainer class for training and testing the model.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    exp_dir : str
        The experiment directory.
    device : str, optional
        The device to use, by default "cuda".
    """
    def __init__(self, cfg: Config, exp_dir: Path, device: str = "cuda") -> None:
        super().__init__()
        self.exp_dir = exp_dir
        self.device = device

        self.start_steps = 1
        self.global_step = 0

        # build config
        self.cfg = parse_structured(self.Config, cfg)
        # build hooks
        self.hooks = parse_hooks(self.cfg.hooks)
        self.call_hook("before_run")
        # build datapipeline
        
        # some code are ignored

    @torch.no_grad()
    def validation(self):
        self.val_dataset_size = len(self.datapipeline.validation_dataset)
        for i in range(0, self.val_dataset_size):
            self.call_hook("before_val_iter")
            batch = self.datapipeline.next_val(i)
            render_dict = self.model(batch)
            render_results = self.renderer.render_batch(render_dict, batch)
            self.metric_dict = self.model.get_metric_dict(render_results, batch)
            self.call_hook("after_val_iter")

    def train_loop(self) -> None:
        """
        The training loop for the model.
        """
        loop_range = range(self.start_steps, self.cfg.max_steps+1)
        self.global_step = self.start_steps
        self.call_hook("before_train")
        for iteration in loop_range:
            self.call_hook("before_train_iter")
            batch = self.datapipeline.next_train(self.global_step)
            self.renderer.update_sh_degree(iteration)
            self.schedulers.step(self.global_step, self.optimizer)
            self.train_step(batch)
            self.optimizer.update_model(**self.optimizer_dict)
            self.call_hook("after_train_iter")
            self.global_step += 1
            if iteration % self.cfg.val_interval == 0 or iteration == self.cfg.max_steps:
                self.call_hook("before_val")
                self.validation()
                self.call_hook("after_val")
        self.call_hook("after_train")

    def call_hook(self, fn_name: str, **kwargs) -> None:
        """
        Call the hook method.

        Parameters
        ----------
        fn_name : str
            The hook method name.
        kwargs : dict
            The keyword arguments.
        """
        for hook in self.hooks:
            # support adding additional custom hook methods
            if hasattr(hook, fn_name):
                try:
                    getattr(hook, fn_name)(self, **kwargs)
                except TypeError as e:
                    raise TypeError(f'{e} in {hook}') from None
```

## 更多类型的钩子

您可以通过定义钩子函数来修改训练器的进度，例如，如果您想在训练迭代后记录一些内容：

```{note}
在钩子函数中可以完全访问训练器。
我们默认提供了日志钩子和检查点钩子。
```
```
@HOOK_REGISTRY.register()
class LogHook(Hook):
    """
    A hook to log the training and validation losses.
    """

    def __init__(self):
        self.ema_loss_for_log = 0.
        self.bar_info = {}
        
        self.losses_test = {"L1_loss": 0., "psnr": 0., "ssims": 0., "lpips": 0.}

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

            if trainner.logger and key != "optimizer_params":
                trainner.logger.write_scalar(key, value, trainner.global_step)
```

您可以参考教程部分或方法部分，了解更多钩子函数的示例。