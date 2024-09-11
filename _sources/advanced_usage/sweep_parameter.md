# Hyperparameter Search

Pointrix supports searching over all parameters specified in the configuration files. Example code can be found in `examples/gaussian_splatting_sweep`. 

Here, we'll use the `colmap_sweep.yaml` configuration file as an example:

```yaml
method: "random"
name: "colmap_sweep"
metric: {goal: "maximize", name: "psnr"}

parameters:
  controller:
    densify_grad_threshold: 
      min: 0.0002
      max: 0.0004
    densify_stop_iter:
      values: [10000, 15000]
  optimizer:
    optimizer_1:
      params:
        point_cloud.position:
          lr: 
            min: 0.00016
            max: 0.00032
```

This configuration file specifies the parameters to be searched within `colmap.yaml`. For example, `densify_grad_threshold` ranges from 0.0002 to 0.0004, and `densify_stop_iter` can take values of 10000 or 15000, among others.

The `method` can be **random**, **grid**, or **bayes**, corresponding to random search, grid search, and Bayesian optimization, respectively. Users can also configure the search range for any other parameters in `colmap.yaml`.

The `metric` specifies the goal to optimize. In this example, we aim to maximize the `psnr` parameter. Related parameters can be read in `launch_sweep.py`.

Once `colmap_sweep.yaml` and `colmap.yaml` are configured, run the following command:

```bash
cd examples/gaussian_splatting_sweep
python launch_sweep.py --config configs/colmap.yaml --config_sweep configs/colmap_sweep.yaml trainer.datapipeline.dataset.data_path=your_data_path trainer.output_path=your_log_path
```

During the run, you can visit the provided URL to view the analysis results of the hyperparameters and metrics.

![xxx](https://github.com/user-attachments/assets/e25ea893-b3ba-4f1d-ae2d-78834588d42c)