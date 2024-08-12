# Dust3r Initialization for Camera Model and Point Cloud (Beta)

## Usage Instructions:
1. Switch to the Beta branch.
2. Download [Dust3r](https://github.com/naver/dust3r) to `examples/dust3r_init` and follow the installation instructions.
3. Move `convert_dust3r.py` to the `examples/dust3r_init/dust3r` folder.
4. Navigate to `examples/dust3r_init/dust3r`, and then use Dust3r to extract point cloud priors and camera priors:
```bash
python convert_dust3r.py --model_path your_dust3r_weights --filelist your_image_path
```
5. Run the program.

```{note}
Dust3r can only extract point cloud and camera priors for dozens of images at a time.
```

## Experimental Results:

Results using twelve images in the Garden scene:
![Image1](../../images/camera.gif)

When camera optimization is disabled, rendering accuracy is reduced, as shown below:
![Image2](../../images/nocamera.gif)