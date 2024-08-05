# 相机优化

Pointrix 基于 Msplat 支持相机优化功能：
用户可以修改配置
```bash
trainer.camera_model.enable_training=True
```

开启相机优化，以及配置对应的学习率：

```bash
trainer.optimizer.camera_params.lr=1e-3
```
