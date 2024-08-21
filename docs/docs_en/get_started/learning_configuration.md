# Configuration Files in Pointrix

Common configurations in Pointrix include:


```yaml
name: "garden"

trainer:
  output_path: "/home/linzhuo/clz/log/garden"
  max_steps: 30000
  val_interval: 5000
  training: True

  model:
    name: BaseModel
    lambda_ssim: 0.2
    point_cloud:
      point_cloud_type: "GaussianPointCloud"  
      max_sh_degree: 3
      trainable: true
      unwarp_prefix: "point_cloud"
      initializer:
        init_type: 'colmap'
        feat_dim: 3
    camera_model:
      enable_training: False
    renderer:
      name: "MsplatRender"
      render_depth: False
      max_sh_degree: ${trainer.model.point_cloud.max_sh_degree}
  
  controller:
    normalize_grad: False
    control_module: "point_cloud"
    split_num: 2
    prune_interval: 100
    min_opacity: 0.005
    percent_dense: 0.01
    densify_grad_threshold: 0.0002
    duplicate_interval: 100
    densify_start_iter: 500
    densify_stop_iter: 15000
    opacity_reset_interval: 3000
    optimizer_name: "optimizer_1"

  optimizer:
    optimizer_1:
      type: BaseOptimizer
      name: Adam
      args:
        eps: 1e-15
      extra_cfg:
        backward: False
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
      # camera_params:
      #   lr: 1e-3

  scheduler:
    name: "ExponLRScheduler"
    params:
      point_cloud.position:
        init:  0.00016
        final: 0.0000016
        max_steps: ${trainer.max_steps}
  
  datapipeline:
    data_set: "ColmapDataset"
    shuffle: True
    batch_size: 1
    num_workers: 0
    dataset:
      data_path: "/home/linzhuo/gj/data/garden"
      cached_observed_data: True
      scale: 0.25
      white_bg: False

  writer:
    writer_type: "TensorboardWriter"
  
  hooks:
    LogHook:
      name: LogHook
    CheckPointHook:
      name: CheckPointHook
  
  exporter:
    exporter_1:
      type: MetricExporter
    exporter_2:
      type: TSDFFusion
      extra_cfg:
        voxel_size: 0.02
        sdf_truc: 0.08
        total_points: 8_000_000
    exporter_3:
      type: VideoExporter
```

## Configuration Files in Pointrix

In Pointrix, common configurations include:

We can see that the Pointrix trainer consists of **model, controller, optimizer, scheduler, datapipeline, writer, hooks, exporter**. In daily tasks, we only need to adjust parameters in the configuration to accomplish different tasks.

## trainer
- output_path: Path to save logs and checkpoints.
- max_steps: Maximum number of steps during training.
- val_interval: Interval for validation.
- training: Whether to train; if set to false, optimizer and scheduler won't be loaded.
- device: Global device setting in Pointrix.

### model
- name: Name of the model, looked up in the registry.
- lambda_dssim: Weight of SSIM loss.
- point_cloud
  - point_cloud_type: Type of point cloud, 'GaussianPointCloud' for Gaussian Splatting method.
  - max_sh_degree: Maximum SH order in Pointrix.
  - trainable: Whether the point cloud model is trainable.
  - unwarp_prefix: Prefix used to distinguish different point cloud groups in the optimizer; only relevant if multiple point clouds are required.
  - initializer
      - init_type: Initialization method for point cloud, including 'colmap' and 'random'.
      - feat_dim: Feature dimension for rendering RGB point clouds.
- camera_model
  - enable_training: Whether to enable camera optimization.

- renderer
  - name: msplat or original Gaussian kernel, indexed by registry.
  - max_sh_degree: Maximum SH order.
  - render_depth: Whether render depth

### controller
- control_module: Name of variables to be densified.
- percent_dense: Percentage of scene range (0-1); controller is enforced if exceeded.
- split_num: Number of splits when performing segmentation on point clouds.
- densify_start_iter: Iteration to start controller.
- densify_stop_iter: Iteration to stop controller.
- prune_interval: Pruning frequency.
- duplicate_interval: Duplication frequency.
- opacity_reset_interval: Frequency to reset opacity.
- densify_grad_threshold: Threshold based on 2D position gradient to determine if points should be densified.
- min_opacity: Minimum opacity used in pruning operations.

### optimizer
- optimizer_x: X-th optimizer; you can add any number of optimizers, which Pointrix will handle automatically.
    - type: Type of optimizer, indexed by registry.
    - name: Name of the optimizer.
    - params：Names of parameters that need optimization and their corresponding learning rates. Pointrix will automatically parse them. **If you have added any learnable parameters on top of the Basemodel, please include them in this configuration.**
    - camera_params: Camera parameters
        - lr: Learning rate for camera parameters, applicable if camera_model.enable_training == True.

### scheduler
- name: Name of scheduler, indexed by registry.
- max_steps: Maximum steps in the scheduler.
- params: Parameters handled by the scheduler.

### dataset
- data_path: Path to the dataset.
- data_set: Type of dataset, indexed by registry.
- shuffle: Whether to shuffle data.
- batch_size: Batch size.
- num_workers: num_workers in the dataloader.
- dataset
  - cached_metadata: Whether to load data with cached metadata.
  - scale: Image scale size.
  - white_bg: Whether to use a white background.

### writer
- writer_type: Tensorboard or Wandb.

### hook
- LogHook
- CheckPointHook

### exporter
- exporter_x: x-th exporter in pointrix
  - type: Type of exporter, indexed by registry.