# Post-Processing Results Extraction (Metric, Mesh, Video)

## Configuration Changes
Pointrix uses exporters to obtain desired post-processing results, such as mesh and video. The relevant configuration is as follows:

```yaml
trainer:
    exporter:
        exporter_a:
            type: MetricExporter
        exporter_b:
            type: TSDFFusion
            extra_cfg:
                voxel_size: 0.02
                sdf_trunc: 0.08
                total_points: 8_000_000 
        exporter_c:
            type: VideoExporter
```

Users can specify multiple exporters to obtain various post-processing results. For example, with the above configuration, users can get Metric and Mesh extraction results as well as Video post-processing results. Mesh is obtained using the TSDF fusion method by default.