# 搜索超参数

Pointrix 支持配置文件中的所有参数的搜索。示例代码在`examples/gaussian_splatting_sweep`中。
我们以其中的配置文件`colmap_sweep.yaml`为例进行说明:

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

这个配置文件规定了`colmap.yaml`中需要搜索的参数，例如densify_grad_threshold的范围为0.0002到0.0004之间，densify_stop_iter
的取值为10000/15000等。

method可以为**random**, **grid** 或者 **bayes**，即分别为随机搜索，网格搜索以及贝叶斯搜索。用户同样也可以配置colmap.yaml中其他所有参数的搜索范围。

metric为我们希望优化的目标，在这个示例中，我们想要最大化psnr这个参数。相关的参数可以在 `launch_sweep.py`中进行读取。

配置好`colmap_sweep.yaml`以及`colmap.yaml` 后，我们运行：
```bash
cd examples/gaussian_splatting_sweep
python launch_sweep.py --config configs/colmap.yaml --config_sweep configs/colmap_sweep.yaml trainer.datapipeline.dataset.data_path=your_data_path  trainer.output_path=your_log_path
```
在运行期间，我们可以进入运行中提示的网站，得到下面超参数与指标的分析结果：

![2024-09-02 18-33-13屏幕截图](https://github.com/user-attachments/assets/e25ea893-b3ba-4f1d-ae2d-78834588d42c)