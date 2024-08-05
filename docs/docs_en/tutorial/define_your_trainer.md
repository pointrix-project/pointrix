# Define your own trainer

You do not need to change trainer in the common cases. 

```{note}
You can define your own process in trainer by adding hook function for the common cases. 
```

However, in some case, for example,
if you want to add an input for your model in the training procee, you can add input:

```{code-block} python
:lineno-start: 1 
:emphasize-lines: "11"
:caption: |
:    We *highlight* the modified part.
class Trainer(DefaultTrainer):
    def train_step(self, batch: List[dict]) -> None:
        """
        The training step for the model.

        Parameters
        ----------
        batch : dict
            The batch data.
        """
        render_dict = self.model(batch, step=self.global_step, training=True)
        render_results = self.renderer.render_batch(render_dict, batch)
        self.loss_dict = self.model.get_loss_dict(render_results, batch)
        self.loss_dict['loss'].backward()
        self.optimizer_dict = self.model.get_optimizer_dict(self.loss_dict,
                                                            render_results,
                                                            self.white_bg)
```

In the example, we only want to add input:  `step` and `training` in self.model for training process, so we only need change `train_step` function.

```{note}
You can read more example on how to change your trainer in `projects`, which contains implementation of different tasks. 
```