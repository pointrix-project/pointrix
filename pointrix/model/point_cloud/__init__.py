from .points import PointCloud

from .points import POINTSCLOUD_REGISTRY

def parse_point_cloud(cfg, datapipeline):
    
    if len(cfg) == 0:
        return None
    point_cloud_type = cfg.point_cloud_type
    point_cloud = POINTSCLOUD_REGISTRY.get(point_cloud_type)
    assert point_cloud is not None, "Point Cloud is not registered: {}".format(
        point_cloud_type
    )
    return point_cloud(cfg, datapipeline.point_cloud)