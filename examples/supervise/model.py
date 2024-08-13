import torch
import torch.nn.functional as F
from pointrix.model.loss import l1_loss, ssim, psnr
from pointrix.utils.pose import unitquat_to_rotmat
from pointrix.model.base_model import BaseModel, MODEL_REGISTRY


@MODEL_REGISTRY.register()
class NormalModel(BaseModel):
    def forward(self, batch=None, training=True, render=True, iteration=None) -> dict:
        if iteration is not None:
            self.renderer.update_sh_degree(iteration)
        if batch is None:
            return {
                    "position": self.point_cloud.position,
                    "opacity": self.point_cloud.get_opacity,
                    "scaling": self.point_cloud.get_scaling,
                    "rotation": self.point_cloud.get_rotation,
                    "shs": self.point_cloud.get_shs,
                }
        frame_idx_list = [batch[i]["frame_idx"] for i in range(len(batch))]
        extrinsic_matrix = self.training_camera_model.extrinsic_matrices(frame_idx_list) \
            if training else self.validation_camera_model.extrinsic_matrices(frame_idx_list)
        intrinsic_params = self.training_camera_model.intrinsic_params(frame_idx_list) \
            if training else self.validation_camera_model.intrinsic_params(frame_idx_list)
        camera_center = self.training_camera_model.camera_centers(frame_idx_list) \
            if training else self.validation_camera_model.camera_centers(frame_idx_list)

        point_normal = self.get_normals
        projected_normal = self.process_normals(
            point_normal, camera_center, extrinsic_matrix)
        
        render_dict = {
            "extrinsic_matrix": extrinsic_matrix,
            "intrinsic_params": intrinsic_params,
            "camera_center": camera_center,
            "position": self.point_cloud.position,
            "opacity": self.point_cloud.get_opacity,
            "scaling": self.point_cloud.get_scaling,
            "rotation": self.point_cloud.get_rotation,
            "shs": self.point_cloud.get_shs,
            "normals": projected_normal
        }
        if render:
            render_results = self.renderer.render_batch(render_dict, batch)
            return render_results
        return render_dict

    @property
    def get_normals(self):
        scaling = self.point_cloud.scaling.clone()
        normal_arg_min = torch.argmin(scaling, dim=-1)
        normal_each = F.one_hot(normal_arg_min, num_classes=3)
        normal_each = normal_each.float()

        rotatation_matrix = unitquat_to_rotmat(self.point_cloud.get_rotation)
        normal_each = torch.bmm(
            rotatation_matrix, normal_each.unsqueeze(-1)).squeeze(-1)

        normal_each = F.normalize(normal_each, dim=-1)
        return normal_each

    def process_normals(self, normals, camera_center, E):
        camera_center = camera_center.squeeze(0)
        E = E.squeeze(0)
        xyz = self.point_cloud.position
        direction = (camera_center.repeat(
            xyz.shape[0], 1).cuda().detach() - xyz.cuda().detach())
        direction = direction / direction.norm(dim=1, keepdim=True)
        dot_for_judge = torch.sum(direction*normals, dim=-1)
        normals[dot_for_judge < 0] = -normals[dot_for_judge < 0]
        w2c = E[:3, :3].cuda().float()
        normals_image = normals @ w2c.T
        return normals_image

    def get_loss_dict(self, render_results, batch) -> dict:
        loss = 0.0
        gt_images = torch.stack(
            [batch[i]["image"] for i in range(len(batch))],
            dim=0
        )
        normal_images = torch.stack(
            [batch[i]["normal"] for i in range(len(batch))],
            dim=0
        )
        L1_loss = l1_loss(render_results['rgb'], gt_images)
        ssim_loss = 1.0 - ssim(render_results['rgb'], gt_images)
        loss += (1.0 - self.cfg.lambda_ssim) * L1_loss
        loss += self.cfg.lambda_ssim * ssim_loss
        # normal_loss = 0.1 * l1_loss(render_results['normal'], normal_images)
        # loss += normal_loss
        loss_dict = {"loss": loss,
                     "L1_loss": L1_loss,
                     "ssim_loss": ssim_loss}
        return loss_dict

    @torch.no_grad()
    def get_metric_dict(self, render_results, batch) -> dict:
        gt_images = torch.clamp(torch.stack(
            [batch[i]["image"].to(self.device) for i in range(len(batch))],
            dim=0), 0.0, 1.0)
        rgb = torch.clamp(render_results['rgb'], 0.0, 1.0)
        L1_loss = l1_loss(rgb, gt_images).mean().double()
        psnr_test = psnr(rgb.squeeze(), gt_images.squeeze()).mean().double()
        ssims_test = ssim(rgb, gt_images, size_average=True).mean().item()
        lpips_vgg_test = self.lpips_func(rgb, gt_images).mean().item()
        metric_dict = {"L1_loss": L1_loss,
                       "psnr": psnr_test,
                       "ssims": ssims_test,
                       "lpips": lpips_vgg_test,
                       "gt_images": gt_images,
                       "images": rgb,
                       "rgb_file_name": batch[0]["camera"].rgb_file_name}

        if 'depth' in render_results:
            depth = render_results['depth']
            metric_dict['depth'] = depth

        if 'normal' in render_results:
            normal = render_results['normal']
            metric_dict['normal'] = normal

        if 'normal' in batch[0]:
            normal = batch[0]['normal']
            metric_dict['normal_gt'] = normal

        return metric_dict
