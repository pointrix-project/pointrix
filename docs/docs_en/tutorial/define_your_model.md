# Define your own model

In Pointrix, the model is defined in `model/base_model.py`, which contains a Gaussian Point Cloud model in default.

```python

@MODEL_REGISTRY.register()
class BaseModel(torch.nn.Module):
    """
    Base class for all models.

    Parameters
    ----------
    cfg : Optional[Union[dict, DictConfig]]
        The configuration dictionary.
    datapipline : BaseDataPipeline
        The data pipeline which is used to initialize the point cloud.
    device : str, optional
        The device to use, by default "cuda".
    """
    @dataclass
    class Config:
        name: str = "BaseModel"
        point_cloud: dict = field(default_factory=dict)
        lambda_dssim: float = 0.2

    cfg: Config

    def __init__(self, cfg: Optional[Union[dict, DictConfig]], datapipline, device="cuda"):
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.point_cloud = parse_point_cloud(self.cfg.point_cloud,
                                             datapipline).to(device)
        self.point_cloud.set_prefix_name("point_cloud")
        self.device = device

```

The default forward function will return a dict which will be required for rendering function:

```python
render_dict = {
            "position": self.point_cloud.position,
            "opacity": self.point_cloud.get_opacity,
            "scaling": self.point_cloud.get_scaling,
            "rotation": self.point_cloud.get_rotation,
            "shs": self.point_cloud.get_shs,
        }
```

And the default loss function contains L1_loss and ssim_loss, the default metric also be defined in base_model.

```{note}
you can refer to model/base_model.py for more detail. If you are interested in point cloud in base_model, you can look for detail in model/gaussian_points.
```

if you want to define your own model, first you need inherit the base model, then add your modification, do not forget to register your model.

```python
@MODEL_REGISTRY.register()
class DeformGaussian(BaseModel):
    def __init__(self, cfg, datapipline, device="cuda"):
        super().__init__(cfg, datapipline, device)

        # Add your modification here
    
    def forward(self, batch):

        # Add your modification here
```

and change the model name in configuiation:

```{code-block} yaml
:lineno-start: 1 
:emphasize-lines: "10"
:caption: |
:    We *highlight* the modified part.
name: "garden"

trainer:
  output_path: "garden_fix"
  max_steps: 30000
  val_interval: 5000

  model:
    # name: BaseModel
    name: DeformGaussian
    lambda_dssim: 0.2
    point_cloud:
      point_cloud_type: "GaussianPointCloud"  
      max_sh_degree: 3
      trainable: true
      unwarp_prefix: "point_cloud"
      initializer:
        init_type: 'colmap'

    ...
```
or command:

```bash
python launch.py trainer.model.name=DeformGaussian
```

```{note}
You can read more example on how to change your model in `projects`, which contains implementation of different tasks.
```
