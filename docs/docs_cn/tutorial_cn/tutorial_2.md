# 为点云渲染添加监督

我们以表面法线为例，来说明如何为点云渲染模型添加表面法线先验的监督。所有的代码在`example/supervise` 文件夹下。
本教程用到的数据下载链接如下：

https://pan.baidu.com/share/init?surl=MEb0rXkbJMlmT8cu7TirTA&pwd=qg8c.

我们使用DSINE 模型来为 Tanks and Temple 数据集的truck场景生成Normal。
## 数据部分的修改
由于Tanks and Temple 数据集为Colmap格式，因此我们选择继承Pointrix 中的Colmap Dataset进行修改。
为了读取DSINE 模型的Normal 先验输出，我们首先需要修改配置：

```bash
trainer.datapipeline.dataset.observed_data_dirs_dict={"image": "images", "normal":"normals"},
```

其中 ``normal`` 为存入Normal 的文件夹名称，``normal``为这个数据的变量名。

Pointrix 会根据当前数据路径和文件夹名称
依据后缀自动读取数据，相关的读取代码在Pointrix中展示如下：


```{code-block} python
:lineno-start: 1 
:emphasize-lines: "16,17,20,25,28,30"
:caption: |
:    Colmap 依据后缀自动读取数据的相关部分代码.

def load_observed_data(self, split):
    """
    The function for loading the observed_data.

    Parameters:
    -----------
    split: str
        The split of the dataset.
    
    Returns:
    --------
    observed_data: List[Dict[str, Any]]
        The observed_datafor the dataset.
    """
    observed_data = []
    for k, v in self.observed_data_dirs_dict.items():
        observed_data_path = self.data_root / Path(v)
        if not os.path.exists(observed_data_path):
            Logger.error(f"observed_data path {observed_data_path} does not exist.")
        observed_data_file_names = sorted(os.listdir(observed_data_path))
        observed_data_file_names_split = [observed_data_file_names[i] for i in self.train_index] if split == "train" else [observed_data_file_names[i] for i in self.val_index]
        cached_progress = ProgressLogger(description='Loading cached observed_data', suffix='iters/s')
        cached_progress.add_task(f'cache_{k}', f'Loading {split} cached {k}', len(observed_data_file_names_split))
        with cached_progress.progress as progress:
            for idx, file in enumerate(observed_data_file_names_split):
                if len(observed_data) <= idx:
                    observed_data.append({})
                if file.endswith('.npy'):
                    observed_data[idx].update({k: np.load(observed_data_path / Path(file))})
                elif file.endswith('png') or file.endswith('jpg') or file.endswith('JPG'):
                    observed_data[idx].update({k: Image.open(observed_data_path / Path(file))})
                else:
                    print(f"File format {file} is not supported.")
                cached_progress.update(f'cache_{k}', step=1)
    return observed_data
```

在使用Pointrix的自动数据读取功能后，我们需要对读取后的Normal数据进行处理。我们需要重载Colmap Dataset并修改其中的``_transform_observed_data``
函数来实现对读取观测数据 (表面法向) 的处理：具体代码在``examples/gaussian_splatting_supervise/dataset.py``.

```{code-block} yaml
:lineno-start: 1 
:emphasize-lines: "3, 12"
:caption: |
:    Modify configuration to read Normal data automatically.
trainer:
    datapipeline:
        data_set: "ColmapDepthNormalDataset"
        shuffle: True
        batch_size: 1
        num_workers: 0
        dataset:
            data_path: "/home/linzhuo/gj/data/garden"
            cached_observed_data: ${trainer.training}
            scale: 0.25
            white_bg: False
            observed_data_dirs_dict: {"image": "images", "normal": "normals"}
```

```{code-block} python
:lineno-start: 1 
:emphasize-lines: "20,21,22"
:caption: |
:    We highlight the modificated part.

# Registry
@DATA_SET_REGISTRY.register()
class ColmapDepthNormalDataset(ColmapDataset):
    def _transform_observed_data(self, observed_data, split):
        cached_progress = ProgressLogger(description='transforming cached observed_data', suffix='iters/s')
        cached_progress.add_task(f'Transforming', f'Transforming {split} cached observed_data', len(observed_data))
        with cached_progress.progress as progress:
            for i in range(len(observed_data)):
                # Transform Image
                image = observed_data[i]['image']
                w, h = image.size
                image = image.resize((int(w * self.scale), int(h * self.scale)))
                image = np.array(image) / 255.
                if image.shape[2] == 4:
                    image = image[:, :, :3] * image[:, :, 3:4] + self.bg * (1 - image[:, :, 3:4])
                observed_data[i]['image'] = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().clamp(0.0, 1.0)
                cached_progress.update(f'Transforming', step=1)

                # Transform Normal
                observed_data[i]['normal'] = \
                    (torch.from_numpy(np.array(observed_data[i]['normal'])) \
                    / 255.0).float().permute(2, 0, 1)
        return observed_data
```

我们将处理后的Normal 数据存入observed_data 中后，Pointrix中的Datapipeline 会自动帮我们在训练过程中生产对应的数据，数据部分修改完成。

## 模型部分的修改
首先，我们需要从Pointrix中导入基本模型，以便我们可以继承、注册和修改它们。

```python
from pointrix.model.base_model import BaseModel, MODEL_REGISTRY
```

其中基本模型包含一个高斯点云模型，和一个相机模型。由于我们需要得到点云模型的表面法线，因此我们需要对高斯点云
模型进行对应的修改,从而使得其在`forward`函数前向输出表面法向，同时我们需要在`get_loss_dict`函数中获得对应的normal损失，使得normal监督
加入反向传播，并且
在`get_metric_dict`函数中得到渲染后的表面法向图片，为可视化预测表面法向做准备：


```{code-block} python
:lineno-start: 1 
:emphasize-lines: "1,15,16,17,28,76,77,81, 36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,106,107,108,110,111,112"
:caption: |
:    我们高亮相较于BaseModel修改的代码。

@MODEL_REGISTRY.register()
class NormalModel(BaseModel):
    def forward(self, batch=None, training=True, render=True, iteration=None) -> dict:

        if iteration is not None:
            self.renderer.update_sh_degree(iteration)
        frame_idx_list = [batch[i]["frame_idx"] for i in range(len(batch))]
        extrinsic_matrix = self.training_camera_model.extrinsic_matrices(frame_idx_list) \
            if training else self.validation_camera_model.extrinsic_matrices(frame_idx_list)
        intrinsic_params = self.training_camera_model.intrinsic_params(frame_idx_list) \
            if training else self.validation_camera_model.intrinsic_params(frame_idx_list)
        camera_center = self.training_camera_model.camera_centers(frame_idx_list) \
            if training else self.validation_camera_model.camera_centers(frame_idx_list)

        point_normal = self.get_normals
        projected_normal = self.process_normals(
            point_normal, camera_center, extrinsic_matrix)

        render_dict = {
            "extrinsic_matrix": extrinsic_matrix,
            "intrinsic_params": intrinsic_params,
            "camera_center": camera_center,
            "position": self.point_cloud.position,
            "opacity": self.point_cloud.get_opacity,
            "scaling": self.point_cloud.get_scaling,
            "rotation": self.point_cloud.get_rotation,
            "shs": self.point_cloud.get_shs,
            "normals": projected_normal
        }
        if render:
            render_results = self.renderer.render_batch(render_dict, batch)
            return render_results
        return render_dict
    # 通过高斯点云的最短轴得到表面法向
    @property
    def get_normals(self):
        scaling = self.point_cloud.scaling.clone()
        normal_arg_min = torch.argmin(scaling, dim=-1)
        normal_each = F.one_hot(normal_arg_min, num_classes=3)
        normal_each = normal_each.float()

        rotatation_matrix = unitquat_to_rotmat(self.point_cloud.get_rotation)
        normal_each = torch.bmm(
            rotatation_matrix, normal_each.unsqueeze(-1)).squeeze(-1)

        normal_each = F.normalize(normal_each, dim=-1)
        return normal_each

    # 将高斯点云的表面法向投影到相机坐标系
    def process_normals(self, normals, camera_center, E):
        xyz = self.point_cloud.position
        direction = (camera_center.repeat(
            xyz.shape[0], 1).cuda().detach() - xyz.cuda().detach())
        direction = direction / direction.norm(dim=1, keepdim=True)
        dot_for_judge = torch.sum(direction*normals, dim=-1)
        normals[dot_for_judge < 0] = -normals[dot_for_judge < 0]
        w2c = E[:3, :3].cuda().float()
        normals_image = normals @ w2c.T
        return normals_image

    def get_loss_dict(self, render_results, batch) -> dict:
        loss = 0.0
        gt_images = torch.stack(
            [batch[i]["image"] for i in range(len(batch))],
            dim=0
        )
        normal_images = torch.stack(
            [batch[i]["normal"] for i in range(len(batch))],
            dim=0
        )
        L1_loss = l1_loss(render_results['rgb'], gt_images)
        ssim_loss = 1.0 - ssim(render_results['rgb'], gt_images)
        loss += (1.0 - self.cfg.lambda_ssim) * L1_loss
        loss += self.cfg.lambda_ssim * ssim_loss
        # normal 监督的损失
        normal_loss = 0.1 * l1_loss(render_results['normal'], normal_images)
        loss += normal_loss
        loss_dict = {"loss": loss,
                     "L1_loss": L1_loss,
                     "ssim_loss": ssim_loss,
                     "normal_loss": normal_loss}
        return loss_dict

    @torch.no_grad()
    def get_metric_dict(self, render_results, batch) -> dict:
        gt_images = torch.clamp(torch.stack(
            [batch[i]["image"].to(self.device) for i in range(len(batch))],
            dim=0), 0.0, 1.0)
        rgb = torch.clamp(render_results['rgb'], 0.0, 1.0)
        L1_loss = l1_loss(rgb, gt_images).mean().double()
        psnr_test = psnr(rgb.squeeze(), gt_images.squeeze()).mean().double()
        ssims_test = ssim(rgb, gt_images, size_average=True).mean().item()
        lpips_vgg_test = self.lpips_func(rgb, gt_images).mean().item()
        metric_dict = {"L1_loss": L1_loss,
                       "psnr": psnr_test,
                       "ssims": ssims_test,
                       "lpips": lpips_vgg_test,
                       "gt_images": gt_images,
                       "images": rgb,
                       "rgb_file_name": batch[0]["camera"].rgb_file_name}

        if 'depth' in render_results:
            depth = render_results['depth']
            metric_dict['depth'] = depth

        if 'normal' in render_results:
            normal = render_results['normal']
            metric_dict['normal'] = normal

        if 'normal' in batch[0]:
            normal = batch[0]['normal']
            metric_dict['normal_gt'] = normal

        return metric_dict
```


## 渲染部分的修改
得益于Msplat 的多目标渲染，我们仅需要修改`render_iter`，即将点云模型输出的Normal特征加入渲染器即可, 同样，新修改后的渲染器需要
使用注册器注册，以便于我们通过配置来索引它。其中14行的Normals即为模型forward的输出参数，Pointrix 
将自动对接：

```{code-block} python
:lineno-start: 1 
:emphasize-lines: "37,53,55,56,57,59"
:caption: |
:    我们高亮相较于MsplatRender修改的代码。

@RENDERER_REGISTRY.register()
class MsplatNormalRender(MsplatRender):
    def render_iter(self, height, width, extrinsic_matrix, intrinsic_params, camera_center, position, opacity,
                    scaling, rotation, shs, normals, **kwargs) -> dict:
        direction = (position -
                     camera_center.repeat(position.shape[0], 1))
        direction = direction / direction.norm(dim=1, keepdim=True)
        rgb = msplat.compute_sh(shs.permute(0, 2, 1), direction)
        extrinsic_matrix = extrinsic_matrix[:3, :]

        (uv, depth) = msplat.project_point(
            position,
            intrinsic_params,
            extrinsic_matrix,
            width, height)

        visible = depth != 0

        # compute cov3d
        cov3d = msplat.compute_cov3d(scaling, rotation, visible)

        # ewa project
        (conic, radius, tiles_touched) = msplat.ewa_project(position, cov3d,
                                                            intrinsic_params,
                                                            extrinsic_matrix,
                                                            uv,
                                                            width,
                                                            height,
                                                            visible
        )

        # sort
        (gaussian_ids_sorted, tile_range) = msplat.sort_gaussian(
            uv, depth, width, height, radius, tiles_touched
        )

        Render_Features = RenderFeatures(rgb=rgb, depth=depth, normal=normals)
        render_features = Render_Features.combine()

        ndc = torch.zeros_like(uv, requires_grad=True)
        try:
            ndc.retain_grad()
        except:
            raise ValueError("ndc does not have grad")

        # alpha blending
        rendered_features = msplat.alpha_blending(
            uv, conic, opacity, render_features,
            gaussian_ids_sorted, tile_range, self.bg_color, width, height, ndc
        )
        rendered_features_split = Render_Features.split(rendered_features)

        normals = rendered_features_split["normal"]
        
        # convert normals from [-1,1] to [0,1]
        normals_im = normals / normals.norm(dim=0, keepdim=True)
        normals_im = (normals_im + 1) / 2
        
        rendered_features_split["normal"] = normals_im

        return {"rendered_features_split": rendered_features_split,
                "uv_points": ndc,
                "visibility": radius > 0,
                "radii": radius
                }


```


## 利用Hook 函数添加相关的日志
最后，我们希望每次验证过程中，可视化模型预测的表面法向图片，因此我们需要修改对应的钩子函数，来达到每次验证后可视化表面法向的效果：


```{code-block} python
:lineno-start: 1 
:emphasize-lines: "26,27,28,29,30,31,32,33"
:caption: |
:    我们高亮相较于LogHook修改的代码。

@HOOK_REGISTRY.register()
class NormalLogHook(LogHook):
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
        
        trainner.writer.write_image(
            "test" + f"_view_{image_name}/normal",
            trainner.metric_dict['normal'].squeeze(),
            step=iteration)
        trainner.writer.write_image(
            "test" + f"_view_{image_name}/normal_gt",
            trainner.metric_dict['normal_gt'].squeeze(),
            step=iteration)

```

最后，我们需要修改我们的配置，从而将修改后的模型，渲染器，数据集，钩子函数添加到Pointrix 训练流中：


```{warning}
**如果您在Basemodel 基础上新加入了可学习的参数（例如卷积网络或者MLP），请在optimizer配置中添加对应的可学习的参数，这样新参数才会优化。**
```



```{code-block} yaml
:lineno-start: 1 
:emphasize-lines: "10, 23, 61, 70, 77"
:caption: |
:    We highlight the modified part.

name: "garden"

trainer:
  output_path: "/home/linzhuo/clz/log/garden"
  max_steps: 30000
  val_interval: 5000
  training: True

  model:
    name: NormalModel
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
      name: "MsplatNormalRender"
      max_sh_degree: ${trainer.model.point_cloud.max_sh_degree}

  controller:
    normalize_grad: False

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
    data_set: "ColmapDepthNormalDataset"
    shuffle: True
    batch_size: 1
    num_workers: 0
    dataset:
      data_path: "/home/linzhuo/gj/data/garden"
      cached_observed_data: ${trainer.training}
      scale: 0.25
      white_bg: False
      observed_data_dirs_dict: {"image": "images", "normal": "normals"}

  writer:
    writer_type: "TensorboardWriter"
  
  hooks:
    LogHook:
      name: NormalLogHook
    CheckPointHook:
      name: CheckPointHook
  
  exporter:
    exporter_a:
      type: MetricExporter
    exporter_b:
      type: TSDFFusion
      extra_cfg:
        voxel_size: 0.02
        sdf_truc: 0.08
        total_points: 8_000_000
    exporter_c:
      type: VideoExporter
```

经过上述修改（所有代码的高亮部分），我们即完成了对高斯点云表面法向的监督。所有的代码在`example/supervise` 文件夹下。

我们通过下面的命令运行代码：

```bash
python launch.py --config colmap.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.datapipeline.dataset.scale=0.5 trainer.output_path=your_log_path
```

实验结果如下：

![](../../images/compare.png)