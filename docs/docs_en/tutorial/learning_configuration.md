# Learning the configuration

The common configuration in Pointrix contains:

```yaml
name: "demo_camera_opt"

trainer:
  output_path: "garden_dptr"
  max_steps: 30000
  val_interval: 5000
  training: True
  device: 'cuda'

  model:
    name: BaseModel
    lambda_dssim: 0.2
    point_cloud:
      point_cloud_type: "GaussianPointCloud"  
      max_sh_degree: 3
      trainable: true
      unwarp_prefix: "point_cloud"
      initializer:
        init_type: 'colmap'
        feat_dim: 3

  optimizer:
    optimizer_1:
      type: GaussianSplattingOptimizer
      name: Adam
      camera_params:
        trainable: ${trainer.dataset.trainable_camera}
        lr: 1e-3
      args:
        eps: 1e-15
      extra_cfg:
        control_module: "point_cloud" # the variable name that need to be densification
        percent_dense: 0.01
        split_num: 2
        densify_start_iter: 500
        densify_stop_iter: 15000
        prune_interval: 100
        duplicate_interval: 100
        opacity_reset_interval: 3000
        densify_grad_threshold: 0.0002
        min_opacity: 0.005
      params:
        point_cloud.position:
          lr: 0.00016
        point_cloud.features:
          lr: 0.0025
        point_cloud.features_rest:
          lr: 0.000125 # features/20
        point_cloud.scaling:
          lr: 0.005
        point_cloud.rotation:
          lr: 0.001
        point_cloud.opacity:
          lr: 0.05

  scheduler:
    name: "ExponLRScheduler"
    max_steps: ${trainer.max_steps}
    params:
      point_cloud.position:
        init:  0.00016
        final: 0.0000016
  dataset:
    data_path: ""
    data_type: "ColmapReFormat"
    cached_image: True
    shuffle: True
    batch_size: 1
    num_workers: 0
    scale: 1.0
    white_bg: False
    trainable_camera: True

  renderer:
    name: "DPTRRender"
    max_sh_degree: ${trainer.model.point_cloud.max_sh_degree}
  writer:
    writer_type: "TensorboardWriter"
  
  hooks:
    LogHook:
      name: LogHook
    CheckPointHook:
      name: CheckPointHook
```
## trainer
- output_path: The path for log and checkpoints saving.
- max_steps: The max number of training step in training.
- val_interval: The validation interval.
- training: Whether training, if not training, the optimizer and scheduler will not be loaded.
- device: the global device in Pointrix.

### model
- name: The name for model, which will be found by registry.
- lambda_dssim: the loss weight for ssim loss
- point_cloud
    - point_cloud_type: the type of point cloud, 'GaussianPointCloud' is used for gaussian splatting based method.
    - max_sh_degree: the max sh degree in pointrix.
    - trainable: whether the point cloud model is trainable.
    - unwarp_prefix: which is used to distinguish different point cloud group in optimizer.
    - initializer
        - init_type: the initialization method for point cloud, contains 'colmap' and 'random'
        - feat_dim: the feat_dim for point cloud to render rgb.

### optimizer
- optimizer_x: the x_th optimizer, you can add arbitary num of optimizer, the pointrix will process them automatically.
    - type: the type of optimizer, which will be found by registry.
    - name
    - camera_params
        - trainable: whether train the camera.
        - lr: the learning rate of camera parameters.
    - extra_cfg
        - control_module: the variable name that need to be densification
        - percent_dense: percentage of scene extent (0--1) a point must exceed to be forcibly densified.
        - split_num: the split num of pointcloud when do split operation.
        - densify_start_iter: iteration where densification starts.
        - densify_stop_iter: iteration where densification stops
        - prune_interval: how frequently to prune
        - duplicate_interval: how frequently to duplicate
        - opacity_reset_interval: how frequently to reset opacity
        - densify_grad_threshold: ;imit that decides if points should be densified based on 2D position gradient
        - min_opacity: the min opacity which will be used in prune operation.

    - params
        - xxx.lr: The learning rate of xxx params

### scheduler
- name: The name for scheduler, which will be found by registry.
- max_steps: the max_steps in scheduler
- params: the parameters that scheduler processes.

### dataset
- data_path: the path of data
- data_type: The data type for dataset, which will be found by registry.
- cached_image: whether cache image, which will increase the training speed.
- shuffle: whether shuffle the dataset.
- batch_size: the batch size for training.
- num_workers: num_workers in dataloader
- scale: the resize scale of input image.
- white_bg: white backgroud or black background.
- trainable_camera: whether make the camera trainable.

### renderer
- name: DPTR or original gaussian kernel, which will be found by registry.
- max_sh_degree: max sh degree.

### writer
- writer_type: Tensorboard or Wandb

### hooks
- LogHook
- CheckPointHook







