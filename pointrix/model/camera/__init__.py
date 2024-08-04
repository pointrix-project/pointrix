from .camera_model import CameraModel

def parse_camera_model(cfg, datapipeline, device="cuda", training=True):
    
    if len(cfg) == 0:
        return None
    if training:
        return CameraModel(cfg, datapipeline.training_cameras)
    return CameraModel(cfg, datapipeline.validation_cameras)