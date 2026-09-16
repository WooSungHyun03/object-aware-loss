#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

if TYPE_CHECKING:
    from scene.cameras import Camera

WARNED = False
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _find_mask(mask_dir, image_path):
    if not os.path.isdir(mask_dir):
        return None

    image_name = os.path.basename(image_path)
    image_stem = Path(image_name).stem
    matches = sorted(
        os.path.join(mask_dir, name)
        for name in os.listdir(mask_dir)
        if os.path.isfile(os.path.join(mask_dir, name))
        and Path(name).stem == image_stem
        and Path(name).suffix.lower() in MASK_EXTENSIONS
    )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple masks match image '{image_path}': {', '.join(matches)}"
        )
    return matches[0] if matches else None


def _load_dataset_mask(args, cam_info, resolution):
    mask_dir = os.path.join(args.source_path, "mask")
    mask_path = _find_mask(mask_dir, cam_info.image_path)
    if mask_path is None:
        return None

    nearest = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
    try:
        with Image.open(mask_path) as mask_image:
            mask_array = np.asarray(
                mask_image.convert("L").resize(resolution, resample=nearest),
                dtype=np.uint8,
            )
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Cannot read object mask '{mask_path}': {error}") from error

    threshold = 0 if mask_array.max() <= 1 else 127
    binary_mask = (mask_array > threshold).astype(np.float32, copy=False)
    return torch.from_numpy(binary_mask).unsqueeze(0)

def loadCam(args, id, cam_info, resolution_scale):
    from scene.cameras import Camera

    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    loaded_mask = _load_dataset_mask(args, cam_info, resolution)

    if len(cam_info.image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        if loaded_mask is None:
            alpha = cam_info.image.getchannel("A").resize(resolution, Image.Resampling.NEAREST)
            loaded_mask = torch.from_numpy((np.asarray(alpha) > 127).astype(np.float32)).unsqueeze(0)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        gt_image = resized_image_rgb

    if loaded_mask is None and getattr(args, "require_masks", False):
        raise FileNotFoundError(
            f"No object mask was found for '{cam_info.image_path}'. Expected a "
            f"matching file in '{os.path.join(args.source_path, 'mask')}' or an "
            "alpha channel in the input image."
        )

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
