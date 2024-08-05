# Define your own dataset

## Change your camera

We implement the base camera in Pointrix which have the following parameters:

```python
@dataclass()
class Camera:
    idx: int
    width: int
    height: int
    R: Union[Float[Tensor, "3 3"], NDArray]
    T: Union[Float[Tensor, "3 1"], NDArray]
    fx: Union[float, None] = None
    fy: Union[float, None] = None
    cx: Union[float, None] = None
    cy: Union[float, None] = None
    fovX: Union[float, None] = None
    fovY: Union[float, None] = None
    rgb_file_name: str = None
    rgb_file_path: str = None
    scene_scale: float = 1.0
    projection_matrix: Float[Tensor, "4 4"] = field(init=False)
    intrinsic_params: Float[Tensor, "4"] = field(init=False)
    full_proj_transform: Float[Tensor, "4 4"] = field(init=False)
    camera_center: Float[Tensor, "3"] = field(init=False)

    ...
```

For example, if you want add parameters in base camera, you need add code:

```python

from pointrix.camera.camera import Camera
class TimeCamera(Camera):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fid = kwargs.get('fid', 0.0)

```

And add your camera model in data reformat.

## Change your dataset

To add our dataset, we need to inhert BaseReFormatData class and
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
:emphasize-lines: "10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22"
:caption: |
:    We *highlight* the modified part.
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

dataset:
    data_path: ""
    data_type: "NerfiesReFormat"
    cached_image: True
    shuffle: True
    batch_size: 1
    num_workers: 0
    scale: 0.25
    white_bg: False

    ...
```

or command:

```bash
python launch.py trainer.dataset.data_type=NerfiesReFormat
```

```{note}
You can read more example on how to change your dataset in `projects`, which contains implementation of different tasks.
```
