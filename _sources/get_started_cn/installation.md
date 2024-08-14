# 安装
通过以下几个步骤，开始你的Pointrix 安装

安装示例视频:
<iframe src="//player.bilibili.com/player.html?isOutside=true&aid=112955291141260&bvid=BV13uYyeHE8c&cid=500001648459184&p=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>

## 1. 相关库安装

首先，你需要下载Pointrix:

```bash
git clone https://github.com/pointrix-project/pointrix.git  --recursive
cd pointrix
```

创建一个带Pytorch的Conda 环境。

```bash
conda create -n pointrix python=3.9
conda activate pointrix
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

安装 Pointrix 和 Msplat:

```
pip install -r requirements.txt
pip install -e .

cd ..
cd msplat
pip install .
```

（可选）你也可以安装gsplat 或者 diff-gaussian-rasterization

```
pip install gsplat

git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
python setup.py install
pip install .
```

完成上述步骤后，Pointrix 安装完成。