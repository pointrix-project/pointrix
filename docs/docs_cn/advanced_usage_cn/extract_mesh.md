# 后处理结果提取 (Metric, Mesh, Video)
## 配置修改
Pointrix 使用exporter 来得到用户需要的后处理结果，例如mesh，视频等，相关的配置如下：

```yaml
trainer:
    exporter:
        exporter_a:
            type: MetricExporter
        exporter_b:
        type: TSDFFusion
            extra_cfg:
                voxel_size: 0.02
                sdf_truc: 0.08
                total_points: 8_000_000 
        exporter_c:
            type: VideoExporter
```

其中用户可指定多个exporter 来得到多个后处理结果。例如上述配置，用户可以得到Metric, Mesh提取结果以及Video的后处理结果。
Mesh 默认使用TSDF fusion方式获得。