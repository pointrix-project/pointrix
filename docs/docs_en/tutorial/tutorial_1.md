# Dynamic 3DGS

There is a simple example that how to extend our pointrix 
framework to dynamic gaussaian (**CVPR 2024: Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction**), 
The deformation of 3D Gaussian Points are generated
by MLP which take time step as input.

## Add your dataset
There are only static scene reader in pointrix. 
To add our dynamic dataset, we need to inhert `BaseReFormatData` class and
rewrite `load_camera` and `load_pointcloud`.

First, we need to import base data format from pointrix so that 
we can inherit, registry and modify them.

```python
from pointrix.dataset.base_data import DATA_FORMAT_REGISTRY, BaseReFormatData, SimplePointCloud
from pointrix.camera.camera import Camera
```

Then, we need to implement the function to load the camera in 
dataset. 

```{note} Pointrix support common dataset reading so you do not 
need to implement by yourself in the most cases. There is just an
example to illustrate how to add your own dataset.
```

```{code-block} python
:lineno-start: 1 
:emphasize-lines: "17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29"
:caption: |
:    We *highlight* the modified part.

# we add time attribute in original camera class.
class TimeCamera(Camera):
    def __init__(self, **kwargs):
        self.fid = kwargs.get('fid', 0.0)
        kwargs.pop('fid', None)
        super().__init__(**kwargs)

@DATA_FORMAT_REGISTRY.register()
class NerfiesReFormat(BaseReFormatData):
    def __init__(self,
                 data_root: Path,
                 split: str = 'train',
                 cached_image: bool = True,
                 scale: float = 1.0):
        super().__init__(data_root, split, cached_image, scale)
    
    def load_camera(self, split: str):
        ## load your camera here
        ## for full implemetation, please refer to projects/deformable_gaussian/dataformat.py
        return cameras_results
    
    def load_pointcloud(self):
        xyz = np.load(os.path.join(self.data_root, "points.npy"))
        xyz = (xyz - self.scene_center) * self.coord_scale
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = SimplePointCloud(positions=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))
        return pcd
```
We first add a new camera class, which inherit from original camera and 
add a time attribute to inference deformation in model. then
we inherit BaseReFormatData and define new function: `load_camera`
and `load_pointcloud`.


## Add your model

Then, we need to import base model from pointrix so that 
we can inherit, registry and modify them.

```python
from pointrix.model.base_model import BaseModel, MODEL_REGISTRY
```

We implemet our model based BaseModel which 
contains full gaussian point implemetation.

```{note} Your can refer to pointrix/model/base_model.py for more detail if
    you care about the full gaussian point implemetation.
```

```{code-block} python
:lineno-start: 1 
:emphasize-lines: "5,6,7,8,11,12,13,14,15,16,17,18,19,20,21,22,25,27,28"
:caption: |
:    We *highlight* the modified part.
@MODEL_REGISTRY.register()
class DeformGaussian(BaseModel):
    def __init__(self, cfg, datapipeline, device="cuda"):
        super().__init__(cfg, datapipeline, device)
        self.deform = DeformNetwork(is_blender=False).to(self.device)
        self.time_interval = 1. / datapipeline.training_dataset_size
        self.smooth_term = get_linear_noise_func(
            lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

    def forward(self, batch, step=-1, training=False):
        N = len(self.point_cloud.position)
        camera_fid = torch.Tensor([batch[0]['camera'].fid]).float().to(self.device)
        position = self.point_cloud.get_position
        time_input = camera_fid.unsqueeze(0).expand(position.shape[0], -1)
        ast_noise = torch.randn(1, 1, device='cuda').expand(N, -1) * self.time_interval * self.smooth_term(step)
        if step < 3000 and training:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            if training:
                d_xyz, d_rotation, d_scaling = self.deform(position.detach(), time_input+ast_noise)
            else:
                d_xyz, d_rotation, d_scaling = self.deform(position.detach(), time_input)

        render_dict = {
            "position": self.point_cloud.get_position + d_xyz,
            "opacity": self.point_cloud.get_opacity,
            "scaling": self.point_cloud.get_scaling + d_scaling,
            "rotation": self.point_cloud.get_rotation + d_rotation,
            "shs": self.point_cloud.get_shs,
        }
        
        return render_dict
```

As modified above, we first read time attributes from modified camera (we will discuss later),
then we get `d_xyz`, `d_rotation`, `d_scaling` from `self.deform` which is a deform MLP network.
Finally we return the deformed gaussian points at the end of `forward` function.
Do not worry, the trainer will process other parts of training automatically.


## Modify Trainer
In deformable gaussian, we need to pass the value of global step to model,
so we can inherit it and add our own operation:

```{code-block} python
:lineno-start: 1 
:emphasize-lines: " 13"
:caption: |
:    We *highlight* the modified part.

from pointrix.engine.default_trainer import DefaultTrainer
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
```


## Run pointrix
We can add our model and dataset implemented above in 
pointrix framework.

```{note} we need to modify the model and dataset name 
to our names of new model and datasets implemented above.
Pointrix framework can find your model and datasets by registry.
```

```{code-block} python
:lineno-start: 1  # this is a comment
: # this is also a comment
:emphasize-lines: "1, 2, 8, 9, 10, 11, 12, 13, 14"
:caption: |
:    We *highlight* the modified part.

from model import DeformGaussian
from dataformat import NerfiesReFormat

def main(args, extras) -> None:
    
    cfg = load_config(args.config, cli_args=extras)

    cfg.trainer.model.name = "DeformGaussian"
    cfg.trainer.dataset.data_type = "NerfiesReFormat"
    # you need to modify this path to your dataset.
    cfg.trainer.dataset.data_path = "/home/clz/data/dnerf/cat"
    cfg['trainer']['optimizer']['optimizer_1']['params']['deform'] = {}
    cfg['trainer']['optimizer']['optimizer_1']['params']['deform']['lr'] = 0.00016 * 5.0
    cfg.trainer.val_interval = 5000

    gaussian_trainer = Trainer(
        cfg.trainer,
        cfg.exp_dir,
    )
    gaussian_trainer.train_loop()    
    model_path = os.path.join(
        cfg.exp_dir, 
        "chkpnt" + str(gaussian_trainer.global_step) + ".pth"
    )
    gaussian_trainer.save_model(model_path)
    print("\nTraining complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--smc_file", type=str, default = None)
    args, extras = parser.parse_known_args()
    
    main(args, extras)
```

Finally, you can run the command to run your code.

```bash
python launch.py --config default.yaml
```
