from .camera_model import CameraModel, CAMERA_REGISTRY

def parse_camera_model(cfg, datapipeline, device="cuda", training=True):
    
    if len(cfg) == 0:
        return None
    name = cfg["name"]
    if training:
        return CAMERA_REGISTRY.get(name)(cfg, datapipeline.training_cameras)
    return CAMERA_REGISTRY.get(name)(cfg, datapipeline.validation_cameras)