# Run Your First Model

## 1. Colmap Dataset

We'll use the Mip-Nerf 360 dataset as an example:
- Download Mip-Nerf 360 dataset from http://storage.googleapis.com/gresearch/refraw360/360_v2.zip to your local folder.

- Run the following command to train your model (... replace data paths in the configuration file ...):

```bash
cd examples/gaussian_splatting
# For Mip-NeRF 360 data which have high-res images and need to downsample.
python launch.py --config ./configs/colmap.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.datapipeline.dataset.scale=0.25 trainer.output_path=your_log_path
```

If successful, you will see the interface below:

```{image} ../../images/run.png
:alt: fishy
:class: bg-primary
:width: 800px
:align: center
```

```{note}
The above box records experiment log paths from top to bottom, tasks running in each progress bar from left to right, runtime and remaining time, and numerical values recorded during training (loss, number of point clouds), etc.
These displays can also be customized.
```

If you want to switch to another rendering kernel, modify `trainer.renderer.name` in the command above.

To use other rendering kernels such as Gsplat or GaussianSplattingRender, run the following commands:

```bash
# you can also use GaussianSplattingRender renderer or GSplat renderer
python launch.py --config ./configs/colmap.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.datapipeline.dataset.scale=0.25 trainer.output_path=your_log_path trainer.model.renderer.name=GaussianSplattingRender

python launch.py --config ./configs/colmap.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.datapipeline.dataset.scale=0.25 trainer.output_path=your_log_path trainer.conrtroler.normalize_grad=True trainer.model.renderer.name=GsplatRender
```


if you want to test the model:

```bash
python launch.py --config ./configs/colmap.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.datapipeline.dataset.scale=0.25 trainer.output_path=your_log_path trainer.training=False trainer.test_model_path=your_model_path
```

## 2. NeRF-Lego (NeRF-Synthetic format dataset)
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