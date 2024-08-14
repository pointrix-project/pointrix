<div align="center">
  <p align="center">
      <picture>
      <source srcset="https://github.com/user-attachments/assets/14d54372-01e6-4e16-aa20-91ec9fc5c257" media="(prefers-color-scheme: dark)">
      <source srcset="https://github.com/user-attachments/assets/a83ee3b1-5452-4614-84f0-662d8d0d9a7f" media="(prefers-color-scheme: light)">
      <img alt="Pointrix" src="https://github.com/user-attachments/assets/a83ee3b1-5452-4614-84f0-662d8d0d9a7f" width="80%">
      </picture>

  </p>
  <p align="center">
    A differentiable point-based rendering library.
    <br />
    <a href="https://pointrix-project.github.io/pointrix/">
    <strong>Document🏠</strong></a>  | 
    <a href="https://pointrix-project.github.io/pointrix/index_cn.html">
    <strong>中文文档🏠</strong></a> | 
    <a href="https://pointrix-project.github.io/pointrix/">
    <strong>Paper(Comming soon)📄</strong></a> | 
    <a href="https://github.com/pointrix-project/msplat">
    <strong>Msplat Backend🌐</strong></a>
    <br />
    <br />
    <!-- <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a> -->
  </p>
</div>

Pointrix is a differentiable point-based rendering framework which has following properties:

- **Highly Extensible**:
  - Python API
  - Modular design for both researchers and beginners
  - Implementing your own method without touching CUDA
- **Powerful Backend**:
  - CUDA Backend
  - Forward Anything: rendering image, depth, normal, optical flow, etc.
  - Backward Anything: optimizing even intrinsics and extrinsics.
- **Rich Features**:
  - Support camera parameters optimization.
  - Support Dynmamic scene reconstruction task and Generation task (WIP).
  - Support mesh extraction and different type of initialization.

<!-- ## Comparation with original 3D gaussian code

### nerf_synthetic dataset (PSNR)

| Method                  | lego        | chair        | ficus        | drums        | hotdog        | ship        | materials        | mic        | average        |
| -----------             | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Pointrix | 35.84       | 36.12       | 35.02       | 26.18       | 37.81       | 30.98       | 29.95       | 35.34       |  33.40       |
| [original](https://github.com/graphdeco-inria/gaussian-splatting)        | 35.88        | 35.92        | 35.00        | 26.21        | 37.81        | 30.95        | 30.02        | 35.35        |   33.39       |

we obtain the result of 3D gaussian code by running following command in their repository.
```bash
 python train.py -s nerf_synthetic_root --eval -w
``` -->

## Quickstart

### Installation


Clone pointrix:

```bash
git clone https://github.com/pointrix-project/pointrix.git  --recursive
cd pointrix
```

Create a new conda environment with pytorch:

```bash
conda create -n pointrix python=3.9
conda activate pointrix
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install Pointrix and MSplat:

```bash
cd msplat
pip install .

cd ..
pip install -r requirements.txt
pip install -e .
```

(Optional) You can also install gsplat or diff-gaussian-rasterization:

```bash
pip install gsplat

git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
cd diff-gaussian-rasterization
python setup.py install
pip install .
```


###  Train Your First 3D Gaussian

#### Tanks and Temples Dataset Demo (Colmap format dataset)
Download the demo truck scene [data](https://pan.baidu.com/s/1MEb0rXkbJMlmT8cu7TirTA?pwd=qg8c) and run:
```bash
cd examples/gaussian_splatting
# For Tanks and Temples data which have high-res images and need to downsample.
python launch.py --config ./configs/colmap.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.datapipeline.dataset.scale=0.5 trainer.output_path=your_log_path

# you can also use GaussianSplatting renderer or GSplat renderer
python launch.py --config ./configs/colmap.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.datapipeline.dataset.scale=0.5 trainer.output_path=your_log_path trainer.model.renderer.name=GaussianSplattingRender

python launch.py --config ./configs/colmap.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.datapipeline.dataset.scale=0.5 trainer.output_path=your_log_path trainer.controller.normalize_grad=True trainer.model.renderer.name=GsplatRender
```
The scale should be set as 0.25 for mipnerf 360 datasets.

For other colmap dataset which do not need to downsample:

```bash
python launch.py --config ./configs/colmap.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.datapipeline.dataset.scale=1.0 trainer.output_path=your_log_path
```
if you want test your model:

```bash
cd examples/gaussian_splatting
# For Tanks and Temples data which have high-res images and need to downsample.
python launch.py --config ./configs/colmap.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.datapipeline.dataset.scale=0.25 trainer.output_path=your_log_path trainer.training=False trainer.test_model_path=your_model_path
```

#### NeRF-Lego (NeRF-Synthetic format dataset)
Download the lego data:

```bash
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
```

Run the following (with adjusted data path):

```bash
cd examples/gaussian_splatting
python launch.py --config ./configs/nerf.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.output_path=your_log_path
```

if you want to test the model:

```bash
python launch.py --config ./configs/nerf.yaml trainer.training=False trainer.datapipeline.dataset.data_path=your_data_path trainer.test_model_path=your_model_path
```

## Advanced Approaches

#### Camera optimization

To enable camera optimization, you should set trainer.model.camera_model.enable_training=True and trainer.optimizer.optimizer_1.camera_params.lr=1e-3:
The renderer must be setted as MsplatRender.

```bash
python launch.py --config ./configs/colmap.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.datapipeline.dataset.scale=1.0 trainer.output_path=your_log_path trainer.model.renderer.name=MsplatRender trainer.model.camera_model.enable_training=True trainer.optimizer.optimizer_1.camera_params.lr=1e-3
```

![pose](https://github.com/user-attachments/assets/42f20422-45be-463a-8b4b-744ede05de84)

#### Post-Processing Results Extraction (Metric, Mesh, Video)

Pointrix uses exporters to obtain desired post-processing results, such as mesh and video. The relevant configuration is as follows:

```yaml
trainer:
    exporter:
        exporter_a:
            type: MetricExporter
        exporter_b:
            type: TSDFFusion
            extra_cfg:
                voxel_size: 0.02
                sdf_trunc: 0.08
                total_points: 8_000_000 
        exporter_c:
            type: VideoExporter
```
Users can specify multiple exporters to obtain various post-processing results. For example, with the above configuration, users can get Metric and Mesh extraction results as well as Video post-processing results. 
Mesh is obtained using the TSDF fusion method by default.
The renderer must be set as MsplatRender or GsplatRender. You need to set trainer.model.renderer.render_depth as True to enable TSDFFusion.

```bash
cd pointrix/projects/gaussian_splatting
python launch.py --config ./configs/nerf.yaml trainer.training=False trainer.datapipeline.dataset.data_path=your_data_path trainer.test_model_path=your_model_path trainer.model.renderer.render_depth=True
```

#### Dust3r initialization (Beta)
1. Switch to the Beta branch.

2. Download Dust3r to examples/dust3r_init and follow the installation instructions.

3. Move convert_dust3r.py to the examples/dust3r_init/dust3r folder.

4. Navigate to examples/dust3r_init/dust3r, and then use Dust3r to extract point cloud priors and camera priors:

```bash
python convert_dust3r.py --model_path your_dust3r_weights --filelist your_image_path
```
5. Run the program

```bash
python launch.py --config config.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.output_path=your_log_path
```

## Release Plans
- [x] Nerf_synthetic dataset (this week).
- [x] Dust3r initialization (this week).
- [x] Mesh exstraction (this week).
- [ ] Introduction video (this week)
- [ ] reformat the document (this week)
- [ ] Dynamic Gaussian Project(next week).

Welcome to discuss with us and submit PR on new ideas and methods.

## Acknowledgment
Thanks to the developers and contributors of the following open-source repositories, whose invaluable work has greatly inspire our project:

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting): 3D Gaussian Splatting for Real-Time Radiance Field Rendering.
- [Threestudio](https://github.com/threestudio-project): A unified framework for 3D content creation
- [OmegaConf](https://github.com/omry/omegaconf): Flexible Python configuration system.
- [SSIM](https://github.com/Po-Hsun-Su/pytorch-ssim): pytorch SSIM loss implemetation.
- [GSplat](https://github.com/nerfstudio-project/gsplat): An open-source library for CUDA accelerated rasterization of gaussians with python bindings. 
- [detectron2](https://github.com/facebookresearch/detectron2): Detectron2 is Facebook AI Research's next generation library that provides state-of-the-art detection and segmentation algorithms. 
- [DN-Splatter](https://github.com/maturk/dn-splatter): Depth and Normal Priors for Gaussian Splatting and Meshing
- [GOF](https://github.com/autonomousvision/gaussian-opacity-fields): Efficient and Compact Surface Reconstruction in Unbounded Scenes


This is project is licensed under Apache License. However, if you use MSplat or the original 3DGS kernel in your work, please follow their license.

## Contributors
<a href="https://github.com/pointrix-project/pointrix/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pointrix-project/pointrix" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

