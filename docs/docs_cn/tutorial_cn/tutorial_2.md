# 为点云渲染添加监督

我们以表面法线为例，来说明如何为点云渲染模型添加表面法线先验的监督。本教程用到的数据下载链接如下：
我们使用DSINE 模型来为 Tanks and Temple 数据集的truck场景生成Normal。
## 数据部分的修改
由于Tanks and Temple 数据集为Colmap格式，因此我们选择继承Pointrix 中的Colmap Dataset进行修改。
为了读取DSINE 模型的Normal 先验输出，我们首先需要修改配置：

```bash
trainer.datapipeline.dataset.meta_dirs_dict={"image": "images", "normal":"normals"},
```

其中 ``normal`` 为存入Normal 的文件夹名称，``normal``为这个数据的变量名。

Pointrix 会根据当前数据路径和文件夹名称
依据后缀自动读取数据，相关的读取代码在Pointrix中展示如下：


```{code-block} python
:lineno-start: 1 
:emphasize-lines: "16,17,20,25,28,30"
:caption: |
:    Colmap 依据后缀自动读取数据的相关部分代码.

def _load_metadata(self, split):
    """
    The function for loading the metadata.

    Parameters:
    -----------
    split: str
        The split of the dataset.
    
    Returns:
    --------
    meta_data: List[Dict[str, Any]]
        The metadata for the dataset.
    """
    meta_data = []
    for k, v in self.meta_dirs_dict.items():
        meta_path = self.data_root / Path(v)
        if not os.path.exists(meta_path):
            Logger.error(f"Meta path {meta_path} does not exist.")
        meta_file_names = sorted(os.listdir(meta_path))
        meta_file_names_split = [meta_file_names[i] for i in self.train_index] if split == "train" else [meta_file_names[i] for i in self.val_index]
        cached_progress = ProgressLogger(description='Loading cached meta', suffix='iters/s')
        cached_progress.add_task(f'cache_{k}', f'Loading {split} cached {k}', len(meta_file_names_split))
        cached_progress.start()
        for idx, file in enumerate(meta_file_names_split):
            if len(meta_data) <= idx:
                meta_data.append({})
            if file.endswith('.npy'):
                meta_data[idx].update({k: np.load(meta_path / Path(file))})
            elif file.endswith('png') or file.endswith('jpg') or file.endswith('JPG'):
                meta_data[idx].update({k: Image.open(meta_path / Path(file))})
            cached_progress.update(f'cache_{k}', step=1)
        cached_progress.stop()
    return meta_data
```

在使用Pointrix的自动数据读取功能后，我们需要对读取后的Normal数据进行处理。我们需要重载Colmap Dataset并修改其中的``_transform_metadata``
函数来实现对读取观测数据 (表面法向) 的处理：具体代码在``examples/gaussian_splatting_supervise/dataset.py``.

```{code-block} python
:lineno-start: 1 
:emphasize-lines: "32,33"
:caption: |
:    我们高亮相较于原版Colmap Dataset中修改的代码。

# 将修改后的数据集使用注册器进行注册
@DATA_SET_REGISTRY.register()
class ColmapDepthNormalDataset(ColmapDataset):
    def _transform_metadata(self, meta, split):
        """
        The function for transforming the metadata.

        Parameters:
        -----------
        meta: List[Dict[str, Any]]
            The metadata for the dataset.
        
        Returns:
        --------
        meta: List[Dict[str, Any]]
            The transformed metadata.
        """
        cached_progress = ProgressLogger(description='transforming cached meta', suffix='iters/s')
        cached_progress.add_task(f'Transforming', f'Transforming {split} cached meta', len(meta))
        cached_progress.start()
        for i in range(len(meta)):
            # Transform Image
            image = meta[i]['image']
            w, h = image.size
            image = image.resize((int(w * self.scale), int(h * self.scale)))
            image = np.array(image) / 255.
            if image.shape[2] == 4:
                image = image[:, :, :3] * image[:, :, 3:4] + self.bg * (1 - image[:, :, 3:4])
            meta[i]['image'] = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().clamp(0.0, 1.0)
            cached_progress.update(f'Transforming', step=1)
            
            # Transform Normal
            meta[i]['normal'] = (torch.from_numpy(np.array(meta[i]['normal'])) / 255.0).float().permute(2, 0, 1)
        cached_progress.stop()
        return meta
```

我们将处理后的Normal 数据存入meta 中后，Pointrix中的Datapipeline 会自动帮我们在训练过程中生产对应的数据，数据部分修改完成。

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
:emphasize-lines: "13,14,15,16, 27,31,32,33,34,35,36,37,38,39,40,41,42,43,44, 46,47,48,49,50,51,52,53,54,55,56,57,58,74,75,76,105,106,107,108,109,110,111"
:caption: |
:    我们高亮相较于BaseModel修改的代码。

@MODEL_REGISTRY.register()
class NormalModel(BaseModel):
    def forward(self, batch=None, training=True) -> dict:

        frame_idx_list = [batch[i]["frame_idx"] for i in range(len(batch))]
        extrinsic_matrix = self.training_camera_model.extrinsic_matrices(frame_idx_list) \
            if training else self.validation_camera_model.extrinsic_matrices(frame_idx_list)
        intrinsic_params = self.training_camera_model.intrinsic_params(frame_idx_list) \
            if training else self.validation_camera_model.intrinsic_params(frame_idx_list)
        camera_center = self.training_camera_model.camera_centers(frame_idx_list) \
            if training else self.validation_camera_model.camera_centers(frame_idx_list)

        # 获得表面法向
        point_normal = self.get_normals
        projected_normal = self.process_normals(
            point_normal, camera_center[0, ...], extrinsic_matrix[0, ...])

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
        normals_image = normals @ w2c.T @ torch.diag(
                torch.tensor([-1, -1, 1], device=normals.device, dtype=torch.float)
            )
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
:emphasize-lines: "14, 51, 67, 69, 70, 71, 73"
:caption: |
:    我们高亮相较于MsplatRender修改的代码。

@RENDERER_REGISTRY.register()
class MsplatNormalRender(MsplatRender):
    """
    A class for rendering point clouds using DPTR.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    white_bg : bool
        Whether the background is white or not.
    device : str
        The device to use.
    update_sh_iter : int, optional
        The iteration to update the spherical harmonics degree, by default 1000.
    """

    def render_iter(self,
                    height,
                    width,
                    extrinsic_matrix,
                    intrinsic_params,
                    camera_center,
                    position,
                    opacity,
                    scaling,
                    rotation,
                    shs,
                    normals,
                    **kwargs) -> dict:

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
        (conic, radius, tiles_touched) = msplat.ewa_project(
            position,
            cov3d,
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

```{code-block} yaml
:lineno-start: 1 
:emphasize-lines: "10, 72, 82, 89"
:caption: |
:    我们高亮相较于默认配置修改的配置。

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
    data_set: "ColmapDepthNormalDataset"
    shuffle: True
    batch_size: 1
    num_workers: 0
    dataset:
      cached_metadata: ${trainer.training}
      scale: 0.25
      white_bg: False

  renderer:
    name: "DPTRNormalRender"
    max_sh_degree: ${trainer.model.point_cloud.max_sh_degree}
  writer:
    writer_type: "WandbWriter"
  
  hooks:
    LogHook:
      name: NormalLogHook
    CheckPointHook:
      name: CheckPointHook
```

经过上述修改（所有代码的高亮部分），我们即完成了对高斯点云表面法向的监督。所有的代码在`example/supervise` 文件夹下。

