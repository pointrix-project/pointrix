import torch
from dataclasses import dataclass
from pointrix.model.loss import l1_loss
from pointrix.model.point_cloud import parse_point_cloud
from pointrix.model.base_model import BaseModel, MODEL_REGISTRY

@MODEL_REGISTRY.register()
class GaussianFlow(BaseModel):
    @dataclass
    class Config(BaseModel.Config):
        lambda_param_l1: float = 0.0
        lambda_knn: float = 0.0
    cfg: Config
    
    def setup(self, datapipeline, device="cuda"):
        super().setup(datapipeline, device)
        self.global_step = 0
    
    def get_gaussian(self):
        atributes_dict = {
            "opacity": self.point_cloud.get_opacity,
            "scaling": self.point_cloud.get_scaling,
        }
        return atributes_dict
    
    def get_flow(self):
        atributes_dict = {
            "position": self.point_cloud.get_position_flow,
            "rotation": self.point_cloud.get_rotation_flow,
            "shs": self.point_cloud.get_shs_flow,
        }
        return atributes_dict
    
    def forward(self, batch=None, training=True, render=True, iteration=None) -> dict:
        N = len(self.point_cloud.position)
        camera_model = self.training_camera_model if training else self.validation_camera_model

        if batch is None:
            return {
                "position": self.point_cloud.position,
                "opacity": self.point_cloud.get_opacity,
                "scaling": self.point_cloud.get_scaling,
                "rotation": self.point_cloud.get_rotation,
                "shs": self.point_cloud.get_shs}
        
        frame_idx_list = [batch[i]["frame_idx"] for i in range(len(batch))]
        time_input = camera_model.get_time(frame_idx_list)
        extrinsic_matrix = camera_model.extrinsic_matrices(frame_idx_list)
        intrinsic_params = camera_model.intrinsic_params(frame_idx_list) 
        camera_center = camera_model.camera_centers(frame_idx_list) 
        
        self.point_cloud.set_timestep(
            t=time_input,
            training=True,
            training_step=iteration
        )
        
        render_dict = {
            "extrinsic_matrix": extrinsic_matrix,
            "intrinsic_params": intrinsic_params,
            "camera_center": camera_center,
            "position": self.point_cloud.get_position_flow,    # [B, N, 3]
            "opacity": self.point_cloud.get_opacity,           # [N, 1]
            "scaling": self.point_cloud.get_scaling,
            "rotation": self.point_cloud.get_rotation_flow,    # [B, N, 3]
            "shs": self.point_cloud.get_shs_flow,              # [B, N, D]
            "height": batch[0]['height'],
            "width": batch[0]['width']
        }
        
        if render:
            render_results = self.renderer.render_batch(render_dict)
            return render_results
        return render_dict
    
    def _params_l1_regulizer(self):
        pos = self.point_cloud.pos_params
        rot = self.point_cloud.rot_params
        pos_abs = torch.abs(pos)
        # pos_norm = pos_abs / pos_abs.max(dim=1, keepdim=True)[0]
        
        rot_abs = torch.abs(rot)
        # rot_norm = rot_abs / rot_abs.max(dim=1, keepdim=True)[0]
        
        loss_l1 = pos_abs.mean() + rot_abs.mean()
        # loss_norm = pos_norm.mean() + rot_norm.mean()
        
        return loss_l1
    
    def get_loss_dict(self, render_results, batch) -> dict:
        gt_images = torch.stack(
            [batch[i]["image"].to(self.device) for i in range(len(batch))],
            dim=0
        )
        L1_loss = l1_loss(render_results['rgb'], gt_images)
        loss = L1_loss
        loss_dict = {
            "loss": loss,
            "L1_loss": L1_loss
        }
        
        if self.cfg.lambda_param_l1 > 0:
            param_l1 = self.params_l1_regulizer()
            loss_dict.update({
                "pl1_loss": param_l1,
            })
            loss += self.cfg.lambda_param_l1 * param_l1
            
        if self.cfg.lambda_knn > 0:
            if self.global_step == self.after_densifi_step:
                self.point_cloud.gen_knn()
                
            if self.global_step > self.after_densifi_step:
                knn_loss = self.point_cloud.knn_loss()
                loss_dict.update({
                    "knn_loss": knn_loss,
                })
                loss += self.cfg.lambda_knn * knn_loss
        
        return loss_dict


