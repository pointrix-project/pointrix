# Adding Supervision for 3DGS
We use surface normals as an example to illustrate how to add supervision for surface normal priors to the model for point cloud rendering. 
All code can be found in the `example/supervise` directory.
The data download link for this tutorial is provided below: 

https://pan.baidu.com/s/1NlhxylY7q3SmVf9j29we3Q?pwd=f95m.


We employ the DSINE model to generate normals for the truck scene in the Tanks and Temple dataset.

## Modification of Data Section
Since the Tanks and Temple dataset is in Colmap format, we opt to modify the Colmap Dataset inherited from Pointrix.
To read the normal prior outputs of the DSINE model, we first need to modify the configuration:

```bash
trainer.datapipeline.dataset.observed_data_dirs_dict={"image": "images", "normal":"normals"},
```

Where `normal` is the folder name where Normal is stored, and `normal` is the variable name for this data.

Pointrix will automatically read the data based on the current data path and folder name according to the suffix. The relevant reading code in Pointrix is shown below:


```{code-block} python
:lineno-start: 1 
:emphasize-lines: "16,17,20,25,28,30"
:caption: |
:    The relevant code section in Colmap for automatically reading data based on the suffix.

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

After utilizing Pointrix's automatic data reading feature, we need to process the read Normal data. We must override the Colmap Dataset and modify the `_transform_observed_data` function to handle the observed data (surface normals). The specific code is located in `examples/gaussian_splatting_supervise/dataset.py`.

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

After storing the processed Normal data into `observed_data`, the Datapipeline in Pointrix automatically generates corresponding data during the training process once the data section modifications are complete.

## Model Section Modifications
Firstly, we need to import basic models from Pointrix so that we can inherit, register, and modify them.

```python
from pointrix.model.base_model import BaseModel, MODEL_REGISTRY
```

The basic models include a Gaussian point cloud model and a camera model. Since we aim to obtain surface normals from the point cloud model, we need to modify the Gaussian point cloud model accordingly. This modification ensures that it outputs surface normals in the `forward` function. Additionally, we need to obtain the corresponding normal loss in the `get_loss_dict` function to include normal supervision in the backward pass. Furthermore, in the `get_metric_dict` function, we obtain rendered surface normal images to prepare for visualizing predicted surface normals.

```{code-block} python
:lineno-start: 1 
:emphasize-lines: "13,14,15,16, 27,31,32,33,34,35,36,37,38,39,40,41,42,43,44, 46,47,48,49,50,51,52,53,54,55,56,57,58,74,75,76,105,106,107,108,109,110,111"
:caption: |
:   We highlight the modified part.
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
        projected_normal = self.process_normals(point_normal, camera_center, extrinsic_matrix)

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

    # get the normal from min scale
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

    # project the surface normal to the camera coordinates
    def process_normals(self, normals, camera_center, E):
        camera_center = camera_center.squeeze(0)
        E = E.squeeze(0)
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
        # normal loss
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


## Rendering Section Modifications
Thanks to Msplat's multi-target rendering capabilities, we only need to modify `render_iter` by incorporating the Normal features outputted by the point cloud model into the renderer. Similarly, the newly modified renderer needs to be registered using a registry so that we can reference it through configuration. The Normals on line 14 correspond to the output parameters of the model's `forward` function, which Pointrix will automatically interface with.

```{code-block} python
:lineno-start: 1 
:emphasize-lines: "14, 51, 67, 69, 70, 71, 73"
:caption: |
:    We highlight the modified part.

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


## Adding Relevant Logging Using Hook Functions

Finally, we aim to visualize predicted surface normal images during each validation process. Therefore, we need to modify the corresponding hook function to achieve the visualization of surface normals after each validation.


```{code-block} python
:lineno-start: 1 
:emphasize-lines: "26,27,28,29,30,31,32,33"
:caption: |
:    We highlight the modified part.

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
Lastly, we need to modify our configuration to integrate the updated model, renderer, dataset, and hook functions into the Pointrix training pipeline.

```{warning}
**If you have added learnable parameters (such as convolutional networks or MLPs) on top of the Basemodel, please include the corresponding learnable parameters in the optimizer configuration so that the new parameters can be optimized.**
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

After the above modifications (highlighted portions in all code), we have successfully implemented supervision for Gaussian point cloud surface normals. All code can be found in the `example/supervise` directory.

We run the following command：

```bash
python launch.py --config colmap.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.datapipeline.dataset.scale=0.5 trainer.output_path=your_log_path
```

The results show below：

![](../../images/compare.png)

