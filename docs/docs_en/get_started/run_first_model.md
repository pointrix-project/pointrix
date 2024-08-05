# Run your first model

## 1. Lego
1. Download the lego data and put it in your folder:

```bash
wget http://cseweb.ucsd.edu/\~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
```

2. Run the following command to train the model (...data path in the config file...):

```bash
cd Pointrix
pip install -e .
cd projects/gaussian_splatting
python launch.py --config ./configs/nerf_dptr.yaml trainer.dataset.data_path=your_data_path

# you can also run this if you have installed gaussian original kernel
python launch.py --config ./configs/nerf.yaml trainer.dataset.data_path=your_data_path
```

If you run Pointrix successfully, you will see the following:

```{image} ../../images/run.png
:alt: fishy
:class: bg-primary
:width: 800px
:align: center
```
```{note}
From top to bottom, from left to right, the above box records the experiment log path, the task run by each progress bar, the running time and remaining time, and the value (loss, points) recorded during the training process.
```

## 2. Mip-nerf 360 or other colmap dataset
1. Download the data and put it in your folder:

http://storage.googleapis.com/gresearch/refraw360/360_v2.zip

2. Run the following command to train the model (...data path in the config file...):

```bash
cd Pointrix
pip install -e .
cd projects/gaussian_splatting
python launch.py --config ./configs/colmap_dptr.yaml trainer.dataset.data_path=your_data_path

# you can also run this if you have install gaussian original kernel
python launch.py --config ./configs/colmap.yaml trainer.dataset.data_path=your_data_path
```


## 3. Dynamic Gaussian
1. Download the iphone dataset and put it in your folder:
https://drive.google.com/drive/folders/1cBw3CUKu2sWQfc_1LbFZGbpdQyTFzDEX

2. Run the following command to train the model  (...data path in the config file...):

```bash
cd Pointrix
pip install -e .
cd projects/deformable_gaussian
python launch.py --config deform.yaml trainer.dataset.data_path=your_data_path
```