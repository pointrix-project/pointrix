# 运行你的第一个模型

## 1. Colmap 数据集
我们以Mip-Nerf 360 数据集为例子
- 下载Mip-Nerf 360:http://storage.googleapis.com/gresearch/refraw360/360_v2.zip 数据到你的文件夹下:

- 运行以下命令来训练你的模型 (...数据路径在配置文件下...):

```bash
cd examples/gaussian_splatting
# For Mip-NeRF 360 data which have high-res images and need to downsample.
python launch.py --config ./configs/colmap.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.datapipeline.dataset.scale=0.25 trainer.output_path=your_log_path
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


如果你想换用其他的渲染内核，例如Gsplat, 或者Msplat可以运行下面的命令：

```bash
# you can also use GaussianSplatting renderer or GSplat renderer
python launch.py --config ./configs/colmap.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.datapipeline.dataset.scale=0.25 trainer.output_path=your_log_path trainer.model.renderer.name=GaussianSplattingRender

python launch.py --config ./configs/colmap.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.datapipeline.dataset.scale=0.25 trainer.output_path=your_log_path trainer.conrtroler.normalize_grad=True trainer.model.renderer.name=GsplatRender
```


如果你想直接测试训练好的模型：

```bash
python launch.py --config ./configs/colmap.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.datapipeline.dataset.scale=0.25 trainer.output_path=your_log_path trainer.training=False trainer.test_model_path=your_model_path
```

## 2. NeRF-Lego (NeRF-Synthetic format dataset)
下载lego 数据:

```bash
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
```

运行下方代码(with adjusted data path):

```bash
cd examples/gaussian_splatting
python launch.py --config ./configs/nerf.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.output_path=your_log_path
```

如果你想直接测试训练好的模型：

```bash
python launch.py --config ./configs/nerf.yaml trainer.training=False trainer.datapipeline.dataset.data_path=your_data_path trainer.test_model_path=your_model_path
```