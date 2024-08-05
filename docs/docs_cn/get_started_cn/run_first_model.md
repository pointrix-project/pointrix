# 运行你的第一个模型

## 1. Colmap 数据集
我们以Mip-Nerf 360 数据集为例子
- 下载Mip-Nerf 360:http://storage.googleapis.com/gresearch/refraw360/360_v2.zip 数据到你的文件夹下:

- 运行以下命令来训练你的模型 (...数据路径在配置文件下...):

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

如果你运行成功，你会看到下方界面：

```{image} ../../images/run.png
:alt: fishy
:class: bg-primary
:width: 800px
:align: center
```
```{note}
上述方框从上到下，从左到右分别记录了实验log路径，每个进度条运行的任务，运行时间与剩余时间，训练过程中记录的数值 (loss, 点云的数目)等。
这些显示的内容同样是可以自定义的。
```

如果你想换用其他的渲染内核，可以通过修改上述命令中的trainer.renderer.name 来实现。


如果你想换用其他的渲染内核，例如Gsplat, 或者diff-gaussian-rasterization,可以运行下面的命令：

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