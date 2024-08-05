Here's the translation of your text into English:

---

# Run Your First Model

## 1. Colmap Dataset

We'll use the Mip-Nerf 360 dataset as an example:
- Download Mip-Nerf 360 dataset from http://storage.googleapis.com/gresearch/refraw360/360_v2.zip to your local folder.

- Run the following command to train your model (... replace data paths in the configuration file ...):

```bash
cd Pointrix
pip install -e .
cd projects/gaussian_splatting
# the default scale of Garden is 0.25
python launch.py --config ./configs/colmap_dptr.yaml \
                trainer.datapipeline.data_path=your_data_path \
                trainer.datapipeline.dataset.scale=0.25 \
                trainer.renderer.name=GaussianSplattingRender
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

To use other rendering kernels such as Gsplat or diff-gaussian-rasterization, run the following commands:

```bash
python launch.py --config ./configs/colmap_dptr.yaml \
                trainer.datapipeline.data_path=your_data_path \
                trainer.datapipeline.dataset.scale=0.25 \
                trainer.renderer.name=GsplatRender

python launch.py --config ./configs/colmap_dptr.yaml \
                trainer.datapipeline.data_path=your_data_path \
                trainer.datapipeline.dataset.scale=0.25 \
                trainer.renderer.name=GaussianSplattingRender
```