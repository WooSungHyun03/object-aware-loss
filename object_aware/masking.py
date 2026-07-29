"""Mask discovery, validation, loading, and optional COLMAP undistortion."""

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PINHOLE_CAMERA_MODELS = {"PINHOLE", "SIMPLE_PINHOLE"}


def list_image_files(directory: str) -> List[str]:
    """Return supported image files in deterministic order."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, name))
        and Path(name).suffix.lower() in IMAGE_SUFFIXES
    )


def resolve_mask_path(mask_directory: str, image_path: str) -> Optional[str]:
    """Resolve one flat-directory mask with the same filename or stem."""
    if not os.path.isdir(mask_directory):
        return None

    image_name = os.path.basename(image_path)
    image_stem = Path(image_name).stem
    matches = sorted(
        os.path.join(mask_directory, name)
        for name in os.listdir(mask_directory)
        if os.path.isfile(os.path.join(mask_directory, name))
        and Path(name).stem == image_stem
        and Path(name).suffix.lower() in IMAGE_SUFFIXES
    )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple masks match image '{image_path}': {', '.join(matches)}"
        )
    return matches[0] if matches else None


def _image_size(path: str) -> Tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.load()
            return image.size
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Cannot read image or mask '{path}': {error}") from error


def validate_mask_associations(
    images_directory: str,
    mask_directory: str,
    require_all: bool = True,
) -> List[Tuple[str, str, Tuple[int, int], Tuple[int, int]]]:
    """Validate filename association and return image/mask size mismatches."""
    image_paths = list_image_files(images_directory)
    if not image_paths:
        raise FileNotFoundError(
            f"No supported RGB images were found in '{images_directory}'."
        )
    if not os.path.isdir(mask_directory):
        if require_all:
            raise FileNotFoundError(
                f"Object mask directory not found: '{mask_directory}'. "
                "Expected one mask per RGB image."
            )
        return []

    mismatches = []
    missing_images = []
    for image_path in image_paths:
        mask_path = resolve_mask_path(mask_directory, image_path)
        if mask_path is None:
            missing_images.append(image_path)
            continue
        image_size = _image_size(image_path)
        mask_size = _image_size(mask_path)
        if image_size != mask_size:
            mismatches.append((image_path, mask_path, image_size, mask_size))

    if require_all and missing_images:
        first_missing = missing_images[0]
        raise FileNotFoundError(
            f"No mask matching RGB image '{first_missing}' was found in "
            f"'{mask_directory}' ({len(missing_images)} missing mask(s) total)."
        )
    return mismatches


def load_mask_for_camera(
    mask_directory: str,
    image_path: str,
    resolution: Sequence[int],
) -> Optional[torch.Tensor]:
    """Load a mask as ``float32 [1, H, W]`` with foreground=1.

    Masks are converted to one luminance channel, normalized from 8-bit values
    to ``[0, 1]``, and resized with nearest-neighbor sampling when required.
    The returned tensor remains on CPU; camera/device policy is handled by the
    caller.
    """
    mask_path = resolve_mask_path(mask_directory, image_path)
    if mask_path is None:
        return None

    nearest = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
    try:
        with Image.open(mask_path) as mask_image:
            mask_image = mask_image.convert("L").resize(tuple(resolution), resample=nearest)
            mask_array = np.array(mask_image, dtype=np.uint8, copy=True)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Cannot load mask '{mask_path}' for image '{image_path}': {error}") from error

    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).to(dtype=torch.float32)
    mask_tensor.div_(255.0)
    return mask_tensor


def _read_colmap_camera_models(model_directory: str) -> List[str]:
    from scene.colmap_loader import read_intrinsics_binary

    cameras_binary = os.path.join(model_directory, "cameras.bin")
    if os.path.isfile(cameras_binary):
        try:
            return [camera.model for camera in read_intrinsics_binary(cameras_binary).values()]
        except (OSError, RuntimeError, ValueError):
            return []

    cameras_text = os.path.join(model_directory, "cameras.txt")
    if not os.path.isfile(cameras_text):
        return []

    models = []
    with open(cameras_text, "r", encoding="utf-8") as camera_file:
        for line in camera_file:
            line = line.strip()
            if line and not line.startswith("#"):
                fields = line.split()
                if len(fields) >= 2:
                    models.append(fields[1])
    return models


def _find_colmap_undistorter_model(source_path: str) -> Optional[str]:
    candidates = [
        os.path.join(source_path, "distorted", "sparse", "0"),
        os.path.join(source_path, "distorted", "sparse"),
        os.path.join(source_path, "sparse", "0"),
        os.path.join(source_path, "sparse"),
    ]
    first_valid_model = None
    distorted_fallback = None
    for model_directory in candidates:
        if not os.path.isdir(model_directory):
            continue
        if not any(
            os.path.isfile(os.path.join(model_directory, filename))
            for filename in ("cameras.bin", "cameras.txt")
        ):
            continue
        if first_valid_model is None:
            first_valid_model = model_directory
        models = _read_colmap_camera_models(model_directory)
        if any(model not in PINHOLE_CAMERA_MODELS for model in models):
            return model_directory
        if "distorted" in Path(model_directory).parts and distorted_fallback is None:
            distorted_fallback = model_directory
    return distorted_fallback or first_valid_model


def _write_image_list(image_paths: Sequence[str], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as image_list_file:
        for image_path in image_paths:
            image_list_file.write(os.path.basename(image_path) + "\n")


def _run_colmap_mask_undistorter(
    dataset,
    images_directory: str,
    mask_directory: str,
) -> Optional[str]:
    colmap_binary = shutil.which("colmap")
    if colmap_binary is None:
        print("[WARN] COLMAP was not found; masks will use nearest-neighbor resize.")
        return None

    input_model = _find_colmap_undistorter_model(dataset.source_path)
    if input_model is None:
        print("[WARN] No COLMAP model was found; masks will use nearest-neighbor resize.")
        return None

    output_root = os.path.join(dataset.model_path, "mask_undistorter")
    output_mask_directory = os.path.join(output_root, "images")
    image_paths = list_image_files(images_directory)

    if os.path.isdir(output_mask_directory):
        try:
            mismatches = validate_mask_associations(
                images_directory, output_mask_directory, require_all=True
            )
        except (FileNotFoundError, ValueError):
            shutil.rmtree(output_root)
        else:
            if mismatches:
                print(
                    "[WARN] Reusing COLMAP-undistorted masks with nearest-neighbor "
                    "resize for remaining size differences."
                )
            else:
                print(f"[INFO] Reusing COLMAP-undistorted masks: {output_mask_directory}")
            return output_mask_directory
    elif os.path.exists(output_root):
        raise FileExistsError(
            f"Mask-undistortion output exists but is not a directory: '{output_root}'"
        )

    image_list_path = os.path.join(dataset.model_path, "mask_undistorter_image_list.txt")
    _write_image_list(image_paths, image_list_path)
    command = [
        colmap_binary,
        "image_undistorter",
        "--image_path",
        mask_directory,
        "--input_path",
        input_model,
        "--output_path",
        output_root,
        "--output_type",
        "COLMAP",
        "--image_list_path",
        image_list_path,
    ]
    print(f"[INFO] Undistorting object masks with COLMAP into '{output_mask_directory}'.")
    start_time = time.time()
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed_seconds = time.time() - start_time
    with open(
        os.path.join(dataset.model_path, "mask_undistorter_time.txt"),
        "w",
        encoding="utf-8",
    ) as timing_file:
        timing_file.write(f"{elapsed_seconds:.6f}\n")

    if result.returncode != 0:
        print("[WARN] COLMAP mask undistortion failed; using nearest-neighbor resize.")
        output_tail = "\n".join(result.stdout.splitlines()[-20:])
        if output_tail:
            print(output_tail)
        return None
    if not os.path.isdir(output_mask_directory):
        print("[WARN] COLMAP did not create an images directory; using original masks.")
        return None

    try:
        validate_mask_associations(
            images_directory, output_mask_directory, require_all=True
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"[WARN] Incomplete COLMAP mask output ({error}); using original masks.")
        return None
    return output_mask_directory


def prepare_dataset_masks(dataset, require_masks: bool = True) -> None:
    """Validate a scene's masks and configure the path used by camera loading.

    The expected directory is ``<source>/mask``. If source image and mask sizes
    differ, the existing workflow first attempts COLMAP's image undistorter and
    otherwise uses nearest-neighbor resizing during loading.
    """
    image_directory_name = dataset.images if dataset.images is not None else "images"
    images_directory = os.path.join(dataset.source_path, image_directory_name)
    mask_directory = os.path.join(dataset.source_path, "mask")
    dataset.mask_path = mask_directory
    dataset.mask_resize_fallback = False

    mismatches = validate_mask_associations(
        images_directory,
        mask_directory,
        require_all=require_masks,
    )
    if not mismatches:
        return

    image_path, mask_path, image_size, mask_size = mismatches[0]
    print(
        "[INFO] Image/mask size mismatch detected "
        f"('{mask_path}' {mask_size} vs '{image_path}' {image_size})."
    )
    undistorted_directory = _run_colmap_mask_undistorter(
        dataset, images_directory, mask_directory
    )
    if undistorted_directory is not None:
        dataset.mask_path = undistorted_directory
        remaining_mismatches = validate_mask_associations(
            images_directory, undistorted_directory, require_all=require_masks
        )
        dataset.mask_resize_fallback = bool(remaining_mismatches)
    else:
        dataset.mask_resize_fallback = True
        print("[INFO] Loading original masks with nearest-neighbor resizing.")


def camera_mask_on_device(
    viewpoint_camera,
    device: torch.device,
    dtype: torch.dtype,
    channels: int,
    eps: float = 1e-6,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return a camera mask and cached masked-loss denominator on ``device``."""
    mask = viewpoint_camera.gt_alpha_mask
    if mask is None:
        return None, None

    cache_on_gpu = getattr(viewpoint_camera, "cache_masks_on_gpu", False)
    if cache_on_gpu:
        cached_mask = getattr(viewpoint_camera, "_object_aware_gpu_mask", None)
        if cached_mask is None or cached_mask.device != device or cached_mask.dtype != dtype:
            cached_mask = mask.to(device=device, dtype=dtype, non_blocking=True)
            viewpoint_camera._object_aware_gpu_mask = cached_mask
        mask = cached_mask
    else:
        if hasattr(viewpoint_camera, "_object_aware_gpu_mask"):
            delattr(viewpoint_camera, "_object_aware_gpu_mask")
        mask = mask.to(device=device, dtype=dtype, non_blocking=True)

    cached_denominator = getattr(viewpoint_camera, "_object_aware_mask_denominator", None)
    cached_channels = getattr(
        viewpoint_camera, "_object_aware_mask_denominator_channels", None
    )
    if (
        cached_denominator is None
        or cached_denominator.device != device
        or cached_denominator.dtype != dtype
        or cached_channels != channels
    ):
        cached_denominator = mask.sum().mul(channels).add(eps)
        viewpoint_camera._object_aware_mask_denominator = cached_denominator
        viewpoint_camera._object_aware_mask_denominator_channels = channels
    return mask, cached_denominator
