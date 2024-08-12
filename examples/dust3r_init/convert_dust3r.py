import copy
import os
import numpy as np
import argparse
import torch
import struct
from pathlib import Path

from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import load_model
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy

from pointrix.logger.writer import Logger, ErrorLogger, ProgressLogger

batch_size = 1

def get_reconstructed_scene(model, device, silent, image_size, filelist, schedule, niter,
                            scenegraph_type, winsize, refid):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    return scene


def write_ply_binary(points_list, colors_list, folder, filename='output.ply', endian='little'):
    """
    Write a list of points and colors to a binary PLY file.
    
    Parameters
    ----------
    points_list : list of np.ndarray
        List of point coordinates.
    colors_list : list of np.ndarray
        List of point colors.
    folder : str
        Folder to save the PLY files.
    filename : str
        Name of the PLY file.
    endian : str
        Endianness of the binary data. Can be 'little' or 'big'.
    """
    endian_char = '<' if endian == 'little' else '>'
    header = f"ply\nformat binary_{endian}_endian 1.0\n"

    try:
        os.makedirs(os.path.join(folder, "points"), exist_ok=True)
        Logger.print(f"Folder 'points' created or already exists.")
    except Exception as e:
        ErrorLogger.print(f"Failed to create folder: {e}")
        return

    for i, (points, colors) in enumerate(zip(points_list, colors_list)):
        assert points.shape == colors.shape, "Points and colors must have the same shape."

        vertex_count = len(points)
        full_header = (header +
                       f"element vertex {vertex_count}\n"
                       "property float x\n"
                       "property float y\n"
                       "property float z\n"
                       "property float red\n"
                       "property float green\n"
                       "property float blue\n"
                       "end_header\n").encode('ascii')

        filepath = os.path.join(folder, 'points', f'{i}_{filename}')
        with open(filepath, 'wb') as f:
            f.write(full_header)
            for point, color in zip(points, colors):
                packed_data = struct.pack(f'{endian_char}ffffff',
                                          point[0], point[1], point[2],
                                          color[0], color[1], color[2])
                f.write(packed_data)

        print(f"File {filepath} written successfully.")\
            

def main(args):
    Logger.print("loading model...")

    model = load_model(args.model_path, args.device, verbose=args.verbose)

    scene = get_reconstructed_scene(
        model, device=args.device, silent=args.silent, image_size=args.image_size, 
        filelist=os.path.join(args.filelist, "images"), schedule=args.schedule, niter=args.niter,
        scenegraph_type=args.scenegraph_type, winsize=args.winsize, refid=args.refid
    )

    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(args.min_conf_thr)))
    msk = to_numpy(scene.get_masks())

    pts3ds = to_numpy(scene.get_pts3d())
    positions = [pts3d[m].reshape(-1, 3) for m, pts3d in zip(msk, pts3ds)]

    rgbimgs = scene.imgs

    colors = [rgbimg[m].reshape(-1, 3) for m, rgbimg in zip(msk, rgbimgs)]

    write_ply_binary(positions, colors, filename='output.ply', folder=args.filelist)

    focals = to_numpy(scene.get_focals().cpu())
    cams2world = to_numpy(scene.get_im_poses().cpu())

    np.savez_compressed(Path(args.filelist) / Path('camera.npz'), focals=focals, cam2world=cams2world)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert 3D model outputs to PLY file.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--silent', action='store_true', help='Disable verbose output during scene reconstruction')
    parser.add_argument('--image_size', type=int, default=512, help='Image size for the model input')
    parser.add_argument('--filelist', type=str, required=True, help='Path to the list of files for reconstruction')
    parser.add_argument('--schedule', type=str, default='linear', help='Schedule type for the reconstruction')
    parser.add_argument('--niter', type=int, default=300, help='Number of iterations for reconstruction')
    parser.add_argument('--scenegraph_type', type=str, default='complete', help='Type of scenegraph to use')
    parser.add_argument('--winsize', type=int, default=1, help='Window size for the process')
    parser.add_argument('--refid', type=int, default=0, help='Reference ID for the process')
    parser.add_argument('--min_conf_thr', type=float, default=0, help='minimum confidence threshold for PointMap')

    args = parser.parse_args()
    main(args)