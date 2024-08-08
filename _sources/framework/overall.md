# Overview

Pointrix is a powerful and easily extensible framework built around point cloud rendering, with the following core features:

- The core of point cloud rendering supports various rendering functionalities and advanced operations, with all inputs returning gradients (point cloud properties, camera parameters).
- The point cloud rendering core is modularized to maximize user interface openness, allowing extensive customization.
- All components are modular to simplify researchers' steps in secondary development as much as possible.
- Supports current mainstream Gaussian point cloud work.

The overall framework of Pointrix is illustrated in the diagram below:
![](../../images/framework_new.png)

## Data Components
- **Dataset Reader**: Parses various types of data provided by developers into a unified data format.
- **Data Pipeline**: Manages the flow of unified data format to the trainer.

## Model Components
- **Point Cloud Model**: Computation graph component based on Gaussian point clouds.
- **Msplat Renderer**: Core component of point cloud rendering. Supports various rendering techniques. All inputs return gradients (camera intrinsic and extrinsic parameters), and interfaces are modularized to the extent possible.
- **Camera Model**：Computation graph component based on Camera model。

## Logging, Visualization, Configuration, Registration, and Callback Components
- **Logger and GUI**: Interact with all components, supporting logging and visualization of intermediate results generated throughout the process.
- **Hooks, Configuration, and Registrar**: Interact with all components, supporting developers in customizing workflows and configurations for all components.
