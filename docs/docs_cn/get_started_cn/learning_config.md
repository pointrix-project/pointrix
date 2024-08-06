# Pointrix 中的配置文件

Pointrix中常见的配置包括：

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
  
  controller:
    normalize_grad: False
    control_module: str = "point_cloud"
    split_num: int = 2
    prune_interval: int = 100
    min_opacity: float = 0.005
    percent_dense: float = 0.01
    min_opacity: float = 0.005
    densify_grad_threshold: float = 0.0002
    duplicate_interval: int = 100
    densify_start_iter: int = 500
    densify_stop_iter: int = 15000
    opacity_reset_interval: int = 3000
    optimizer_name: str = "optimizer_1"

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
      camera_params:
        lr: 1e-3

  scheduler:
    name: "ExponLRScheduler"
    params:
      point_cloud.position:
        init:  0.00016
        final: 0.0000016
        max_steps: ${trainer.max_steps}
  
  datapipeline:
    data_path: "/home/linzhuo/gj/data/garden"
    data_set: "ColmapDataset"
    shuffle: True
    batch_size: 1
    num_workers: 0
    dataset:
      cached_metadata: ${trainer.training}
      scale: 0.25
      white_bg: False

  renderer:
    name: "GaussianSplattingRender"
    max_sh_degree: ${trainer.model.point_cloud.max_sh_degree}
  writer:
    writer_type: "WandbWriter"
  
  hooks:
    LogHook:
      name: LogHook
    CheckPointHook:
      name: CheckPointHook
```

我们可以看到，Pointrix 的trainer 由 **model, controller, optimizer, scheduler, datapipeline, writer, hooks, exporter** 组成。
在日常需求中，我们只需要调整配置中的参数即可完成不同的任务。

## trainer
- output_path: 保存日志和检查点的路径。
- max_steps: 训练中的最大步数。
- val_interval: 验证间隔。
- training: 是否训练，如果不进行训练，则不会加载优化器和调度器。
- device: Pointrix中的全局设备。

### model
- name: 模型的名称，将由注册表查找。
- lambda_dssim: SSIM损失的权重。
- point_cloud
  - point_cloud_type: 点云的类型，'GaussianPointCloud'用于基于Gaussian Splatting的方法。
  - max_sh_degree: Pointrix中的最大SH阶数。
  - trainable: 点云模型是否可训练。
  - unwarp_prefix: 用于区分优化器中不同点云组的前缀,除非有多个点云的需求，否则不需要关心。
  - initializer
      - init_type: 点云的初始化方法，包括'colmap'和'random'。
      - feat_dim: 用于渲染RGB的点云的特征维度。
- camera_model
  - enable_training: 是否开启相机优化

### controller
- control_module: 需要稠密化的变量名称。
- percent_dense: 场景范围的百分比（0-1），必须超过该百分比才能进行强制稠密化。
- split_num: 执行分割操作时点云的分割数量。
- densify_start_iter: 开始稠密化的迭代次数。
- densify_stop_iter: 停止稠密化的迭代次数。
- prune_interval: 剪枝频率。
- duplicate_interval: 复制频率。
- opacity_reset_interval: 重置不透明度的频率。
- densify_grad_threshold: 基于 2D 位置梯度决定是否应稠密化点的阈值。
- min_opacity: 剪枝操作中将使用的最小不透明度。

### optimizer
- optimizer_x: 第 x 个优化器，您可以添加任意数量的优化器，Pointrix 将自动处理它们。
    - type: 优化器的类型，由注册器索引。
    - name: 优化器的名称。
    - camera_params: 相机参数
        - lr: 相机参数的学习率，需要camera_model.enable_training==True

### scheduler
- name: 调度器的名称，由注册器索引
- max_steps: 调度器中的最大步数。
- params: 调度器处理的参数。

### dataset

- data_path: 数据集的路径
- data_set: 数据集的类型，由注册器索引。
- shuffle: 是否随机打乱数据
- batch_size: 批处理大小
- num_workers: dataloader 中的num_workers
- dataset
  - cached_metadata: 是否引入缓存加载数据
  - scale: 图片尺度大小
  - white_bg: 是否白色背景

### renderer
- name: MSplat, GSplat 或原始高斯核，将由注册表找到。
- max_sh_degree: 最大的 sh 阶数。

### writer
- writer_type: Tensorboard 或 Wandb

### hook
- LogHook
- CheckPointHook
