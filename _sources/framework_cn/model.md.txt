# 模型

如下图所示，模型训练相关的由**BaseModel**、**Controller**和**Optimizer**构成。
其中**BaseModel**中包含了默认的**Point Cloud Model** 与 **Camera Model**，以及 **Renderer**.

- **BaseModel**：提供了一种方便灵活的方式来组织训练过程、管理参数、定义计算，并利用PyTorch的自动微分能力。它与一个PointCloud模型与相机模型关联，这是优化的主要目标。
- **Optimizer**：负责自动更新模型的参数,支持多优化器。
- **Controller**：负责自动更新点云模型的结构。

以下是在Pointrix 训练中模型与优化器的交互过程：

![](../../images/model.png)

其中左侧的 **Point Cloud Prior**，**Observation** 与 **Camera Prior**均为数据流水线传递来的数据。
