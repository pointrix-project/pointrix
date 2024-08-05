# Installation

Get started with our package with these steps:

## 1. Install package

Clone pointrix:

```bash
git clone https://github.com/pointrix-project/pointrix.git  --recursive
```

Create a new conda environment with pytorch:

```bash
conda create -n pointrix python=3.9
conda activate pointrix
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
Install Pointrix and DPTR:

```bash
cd ..
pip install -r requirements.txt
pip install -e .

cd dptr
pip install .
```