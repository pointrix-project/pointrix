# Model

As depicted in the diagram below, the components related to model training consist of **BaseModel**, **Controller**, and **Optimizer**.

- **BaseModel**: Provides a convenient and flexible way to organize the training process, manage parameters, define computations, and leverage PyTorch's automatic differentiation capabilities. It is associated with a Point Cloud Model, Camera Model, and Renderer, which are the primary targets for optimization.
  
- **Optimizer**: Responsible for automatically updating the model parameters, supporting multiple optimizers.
  
- **Controller**: Responsible for automatically updating the structure of the point cloud model.

The interaction between the model and optimizer during training in Pointrix is illustrated below:

![](../../images/model.png)

The data passed to the model includes **Point Cloud Priors**, **Observations**, and **Camera Priors**, all provided by the data pipeline on the left.

## Model

The model consists of three main components:

- **Point Cloud Model**: An optimizable point cloud model where users can extend its features by registering optimizable attributes. For example:
  ```python
  point_cloud = PointsCloud(cfg)
  point_cloud.register_attribute('position', position)
  point_cloud.register_attribute('rgb', rgb)
  ```
  With the above code, users register 'position' and 'rgb' attributes for each point in the point cloud. Users can extend the point cloud's features by defining custom attributes. For instance, Gaussian point clouds may include attributes such as 'sh', 'scale', 'rotation', 'opacity', and 'position'.

- **Camera Model**: Mainly includes three optimizable attributes: `qrots`, `tvecs`, and `intrs`, representing rotation, translation, and camera intrinsic parameters. Support for extending custom attributes will be added in the future.

- **Renderer**: Includes support for various rendering methods such as the original Gaussian kernel, Gsplat, and Msplat. Typically, as shown in the figure, the renderer reads the camera and point cloud data for the corresponding viewpoints from the model and outputs the rendered results.

## Optimizer and Controller

The optimizer is responsible for updating the model parameters, while the controller is responsible for updating the model structure (usually the point cloud model).

## Related Configuration

### Model Configuration:
```yaml
trainer:
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
            render_depth: True
            max_sh_degree: ${trainer.model.point_cloud.max_sh_degree}
```

- `name`: The name of the model, which will be looked up in the registry.
- `lambda_ssim`: The weight for the SSIM loss.
- `point_cloud`
  - `point_cloud_type`: The type of point cloud; 'GaussianPointCloud' is used for Gaussian Splatting-based methods.
  - `max_sh_degree`: The maximum SH degree in Pointrix.
  - `trainable`: Whether the point cloud model is trainable.
  - `unwarp_prefix`: A prefix used to differentiate between different point cloud groups in the optimizer; generally, this is not a concern unless multiple point clouds are needed.
  - `initializer`
      - `init_type`: The initialization method for the point cloud, including 'colmap' and 'random'.
      - `feat_dim`: The feature dimension of the point cloud used for rendering RGB.
- `camera_model`
  - `enable_training`: Whether to enable camera optimization.
- `renderer`
    - `name`: MSplat, GSplat, or the original Gaussian kernel, which will be indexed by the registry.
    - `max_sh_degree`: The maximum SH degree for the renderer.
    - `render_depth`: Whether to render depth.