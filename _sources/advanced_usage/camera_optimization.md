# Camera Optimization

Pointrix supports camera optimization based on Msplat:
Users can modify configurations
```bash
trainer.camera_model.enable_training=True
```

Enable camera optimization and configure the corresponding learning rate:

```bash
trainer.optimizer.camera_params.lr=1e-3
```