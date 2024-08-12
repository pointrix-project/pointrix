# 相机优化

## 配置修改
Pointrix 基于 Msplat 支持相机优化功能：
用户可以修改配置
```bash
trainer.camera_model.enable_training=True
```

开启相机优化，以及配置对应的学习率：

```bash
trainer.optimizer.camera_params.lr=1e-3
```
同时，为了支持相机的梯度反传，我们需要将渲染器改为Msplat 或者 Gsplat: 

```bash
trainer.model.renderer.name=MsplatRender
```

综上所述，开启相机优化的命令为：

```bash
python launch.py --config ./configs/colmap.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.datapipeline.dataset.scale=1.0 trainer.output_path=your_log_path trainer.model.renderer.name=MsplatRender trainer.model.camera_model.enable_training=True trainer.optimizer.optimizer_1.camera_params.lr=1e-3
```


## 实验结果

我们随机给真值相机位姿增加扰动作为相机先验，然后可视化优化的相机位姿态与真值的距离，可视化结果如下：

![](../../images/pose.gif)

同时我们对比了使用Dust3R 作为相机和点云的初始化，开启相机优化与否的对比结果：

![图片1](../../images/camera.gif)
![图片2](../../images/nocamera.gif)