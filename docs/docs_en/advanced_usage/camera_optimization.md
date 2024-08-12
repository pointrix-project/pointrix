# Camera Optimization

## Configuration Changes
Pointrix, based on Msplat, supports camera optimization:
Users can modify the configuration with:
```bash
trainer.camera_model.enable_training=True
```

Enable camera optimization and set the learning rate:
```bash
trainer.optimizer.camera_params.lr=1e-3
```
Additionally, to support gradient backpropagation for the camera, we need to change the renderer to Msplat or Gsplat:
```bash
trainer.model.renderer.name=MsplatRender
```

In summary, the command to enable camera optimization is:
```bash
python launch.py --config ./configs/colmap.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.datapipeline.dataset.scale=1.0 trainer.output_path=your_log_path trainer.model.renderer.name=MsplatRender trainer.model.camera_model.enable_training=True trainer.optimizer.optimizer_1.camera_params.lr=1e-3
```

## Experimental Results

We added random perturbations to the ground truth camera poses as camera priors, then visualized the distance between the optimized camera poses and the ground truth. The visualization results are as follows:

![](../../images/pose.gif)

We also compared using Dust3R for camera and point cloud initialization, with and without camera optimization:

![Image1](../../images/camera.gif)
![Image2](../../images/nocamera.gif)