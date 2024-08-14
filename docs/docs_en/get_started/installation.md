# Installation

Start your Pointrix installation by following these steps:

Installation video shows below:
<iframe src="//player.bilibili.com/player.html?isOutside=true&aid=112955291141260&bvid=BV13uYyeHE8c&cid=500001648459184&p=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>

## 1. Installing Required Libraries

First, clone Pointrix from GitHub:

```bash
git clone https://github.com/pointrix-project/pointrix.git --recursive
cd pointrix
```

Create a Conda environment with Python 3.9 and activate it:

```bash
conda create -n pointrix python=3.9
conda activate pointrix
```

Install PyTorch with CUDA support (version 2.1.1 for PyTorch, 0.16.1 for torchvision, and CUDA version 12.1):

```bash
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install Pointrix and Msplat:

```bash
cd pointrix
pip install -r requirements.txt
pip install -e .

cd ..
cd msplat
pip install .
```

(Optional) You can also install gsplat or diff-gaussian-rasterization:

```bash
pip install gsplat

git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
cd diff-gaussian-rasterization
python setup.py install
pip install .
```

Once you've completed these steps, Pointrix is installed and ready to use.