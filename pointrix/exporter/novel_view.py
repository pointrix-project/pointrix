import os
import random
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field

import torch
import imageio
from ..utils.system import mkdir_p
from ..utils.visualize import visualize_depth, visualize_rgb
from ..model.loss import psnr, ssim, LPIPS, l1_loss


@torch.no_grad()
def test_view_render(model, datapipeline, output_path, device='cuda'):
    """
    Render the test view and save the images to the output path.

    Parameters
    ----------
    model : BaseModel
        The point cloud model.
    datapipeline : DataPipeline
        The data pipeline object.
    output_path : str
        The output path to save the images.
    """
    l1_test = 0.0
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    lpips_func = LPIPS()
    val_dataset = datapipeline.validation_dataset
    val_dataset_size = len(val_dataset)
    progress_bar = tqdm(
        range(0, val_dataset_size),
        desc="Validation progress",
        leave=False,
    )

    mkdir_p(os.path.join(output_path, 'test_view'))

    for i in range(0, val_dataset_size):
        batch = datapipeline.next_val(i)
        render_results = model(batch, training=False)
        image_name = os.path.basename(batch[0]['camera'].rgb_file_name)
        gt_image = torch.clamp(batch[0]['image'].to("cuda").float(), 0.0, 1.0)
        image = torch.clamp(
            render_results['rgb'], 0.0, 1.0).squeeze()

        visualize_feature = ['rgb']
        
        for feat_name in visualize_feature:
            feat = render_results[feat_name]
            visual_feat = eval(f"visualize_{feat_name}")(feat.squeeze())
            if not os.path.exists(os.path.join(output_path, f'test_view_{feat_name}')):
                os.makedirs(os.path.join(
                    output_path, f'test_view_{feat_name}'))
            imageio.imwrite(os.path.join(
                output_path, f'test_view_{feat_name}', image_name), visual_feat)

        l1_test += l1_loss(image, gt_image).mean().double()
        psnr_test += psnr(image, gt_image).mean().double()
        ssim_test += ssim(image, gt_image).mean().double()
        lpips_test += lpips_func(image, gt_image).mean().double()
        progress_bar.update(1)
    progress_bar.close()
    l1_test /= val_dataset_size
    psnr_test /= val_dataset_size
    ssim_test /= val_dataset_size
    lpips_test /= val_dataset_size
    print(f"Test results: L1 {l1_test:.5f} PSNR {psnr_test:.5f} SSIM {ssim_test:.5f} LPIPS (VGG) {lpips_test:.5f}")


def novel_view_render(model, renderer, datapipeline, output_path, novel_view_list=["Dolly", "Zoom", "Spiral"], device='cuda'):
    """
    Render the novel view and save the images to the output path.

    Parameters
    ----------
    model : BaseModel
        The point cloud model.
    renderer : Renderer
        The renderer object.
    datapipeline : DataPipeline
        The data pipeline object.
    output_path : str
        The output path to save the images.
    novel_view_list : list, optional
        The list of novel views to render, by default ["Dolly", "Zoom", "Spiral"]
    """
    cameras = datapipeline.training_cameras
    print("Rendering Novel view ...............")
    for novel_view in novel_view_list:
        feat_frame = {}
        novel_view_camera_list = cameras.generate_camera_path(50, novel_view)
        for i, camera in enumerate(novel_view_camera_list):

            atributes_dict = model(camera)
            render_dict = {
                "camera": camera,
                "height": int(camera.image_height),
                "width": int(camera.image_width),
                "extrinsic_matrix": camera.extrinsic_matrix.to(device),
                "intrinsic_params": camera.intrinsic_params.to(device),
                "camera_center": camera.camera_center.to(device),
            }
            render_dict.update(atributes_dict)
            render_results = renderer.render_iter(**render_dict)
            for feat_name, feat in render_results['rendered_features_split'].items():
                visual_feat = eval(f"visualize_{feat_name}")(feat.squeeze())
                if feat_name not in feat_frame:
                    feat_frame[feat_name] = []
                feat_frame[feat_name].append(visual_feat)
                if not os.path.exists(os.path.join(output_path, f'{novel_view}_{feat_name}')):
                    os.makedirs(os.path.join(
                        output_path, f'{novel_view}_{feat_name}'))
                imageio.imwrite(os.path.join(
                    output_path, f'{novel_view}_{feat_name}', "{:0>3}.png".format(i)), visual_feat)
    
        for feat_name, feat in feat_frame.items():
            imageio.mimwrite(os.path.join(output_path, f'{novel_view}_{feat_name}', f'{novel_view}_{feat_name}.mp4'), feat, fps=30, quality=8)
            
    

