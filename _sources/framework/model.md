# Model

As depicted in the diagram below, the components related to model training consist of **BaseModel**, **Controller**, and **Optimizer**.

- **BaseModel**: Provides a convenient and flexible way to organize the training process, manage parameters, define computations, and leverage PyTorch's automatic differentiation capabilities. It is associated with a Point Cloud Model, Camera Model, and Renderer, which are the primary targets for optimization.
  
- **Optimizer**: Responsible for automatically updating the model parameters, supporting multiple optimizers.
  
- **Controller**: Responsible for automatically updating the structure of the point cloud model.

The interaction between the model and optimizer during training in Pointrix is illustrated below:

![](../../images/model.png)

The data passed to the model includes **Point Cloud Priors**, **Observations**, and **Camera Priors**, all provided by the data pipeline on the left.