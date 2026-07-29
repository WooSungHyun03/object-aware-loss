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

"""Compute image-space PSNR, SSIM, and LPIPS for rendered test views."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_images(
    renders_directory: Path,
    ground_truth_directory: Path,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    """Load matching rendered and ground-truth RGB images."""
    import torchvision.transforms.functional as tf

    if not renders_directory.is_dir():
        raise FileNotFoundError(f"Render directory not found: '{renders_directory}'")
    if not ground_truth_directory.is_dir():
        raise FileNotFoundError(
            f"Ground-truth directory not found: '{ground_truth_directory}'"
        )

    image_names = sorted(
        path.name
        for path in renders_directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_names:
        raise FileNotFoundError(f"No rendered images found in '{renders_directory}'")

    renders = []
    ground_truths = []
    for image_name in image_names:
        render_path = renders_directory / image_name
        ground_truth_path = ground_truth_directory / image_name
        if not ground_truth_path.is_file():
            raise FileNotFoundError(
                f"Ground-truth image matching '{render_path}' is missing: "
                f"'{ground_truth_path}'"
            )
        with Image.open(render_path) as render_image:
            render_tensor = tf.to_tensor(render_image.convert("RGB")).unsqueeze(0)
        with Image.open(ground_truth_path) as ground_truth_image:
            ground_truth_tensor = tf.to_tensor(
                ground_truth_image.convert("RGB")
            ).unsqueeze(0)
        if render_tensor.shape != ground_truth_tensor.shape:
            raise ValueError(
                f"Image shape mismatch for '{image_name}': render "
                f"{tuple(render_tensor.shape)} vs ground truth "
                f"{tuple(ground_truth_tensor.shape)}"
            )
        renders.append(render_tensor.to(device))
        ground_truths.append(ground_truth_tensor.to(device))
    return renders, ground_truths, image_names


def evaluate(
    model_paths: Sequence[str],
    device: torch.device,
    fail_fast: bool = False,
) -> None:
    """Evaluate every rendered method directory below each model path."""
    import torch
    from tqdm import tqdm

    from lpipsPyTorch import LPIPS
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    lpips_metric = LPIPS(net_type="vgg", version="0.1").to(device).eval()

    for model_path_string in model_paths:
        model_path = Path(model_path_string)
        try:
            print(f"Scene: {model_path}")
            test_directory = model_path / "test"
            if not test_directory.is_dir():
                raise FileNotFoundError(
                    f"Rendered test directory not found: '{test_directory}'. "
                    "Run render.py without --skip_test first."
                )

            scene_results: Dict[str, Dict[str, float]] = {}
            per_view_results: Dict[str, Dict[str, Dict[str, float]]] = {}
            method_directories = sorted(path for path in test_directory.iterdir() if path.is_dir())
            if not method_directories:
                raise FileNotFoundError(f"No rendered methods found in '{test_directory}'")

            for method_directory in method_directories:
                method_name = method_directory.name
                print(f"Method: {method_name}")
                renders, ground_truths, image_names = read_images(
                    method_directory / "renders",
                    method_directory / "gt",
                    device,
                )

                ssims = []
                psnrs = []
                lpips_values = []
                with torch.no_grad():
                    for render_image, ground_truth_image in tqdm(
                        zip(renders, ground_truths),
                        total=len(renders),
                        desc="Metric evaluation",
                    ):
                        ssims.append(ssim(render_image, ground_truth_image).reshape(()))
                        psnrs.append(
                            psnr(render_image, ground_truth_image).mean().reshape(())
                        )
                        lpips_values.append(
                            lpips_metric(render_image, ground_truth_image)
                            .mean()
                            .reshape(())
                        )

                metric_tensors = {
                    "SSIM": torch.stack(ssims),
                    "PSNR": torch.stack(psnrs),
                    "LPIPS": torch.stack(lpips_values),
                }
                scene_results[method_name] = {
                    metric_name: values.mean().item()
                    for metric_name, values in metric_tensors.items()
                }
                per_view_results[method_name] = {
                    metric_name: {
                        image_name: value
                        for image_name, value in zip(image_names, values.detach().cpu().tolist())
                    }
                    for metric_name, values in metric_tensors.items()
                }

                print(f"  SSIM : {scene_results[method_name]['SSIM']:>12.7f}")
                print(f"  PSNR : {scene_results[method_name]['PSNR']:>12.7f}")
                print(f"  LPIPS: {scene_results[method_name]['LPIPS']:>12.7f}\n")

            with open(model_path / "results.json", "w", encoding="utf-8") as result_file:
                json.dump(scene_results, result_file, indent=2)
            with open(model_path / "per_view.json", "w", encoding="utf-8") as per_view_file:
                json.dump(per_view_results, per_view_file, indent=2)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            if fail_fast:
                raise
            print(f"Unable to compute metrics for '{model_path}': {error}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compute PSNR, SSIM, and LPIPS for render.py test outputs."
    )
    parser.add_argument(
        "--model_paths",
        "-m",
        required=True,
        nargs="+",
        type=str,
        help="One or more trained model output directories.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch evaluation device (default: cuda:0).",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop on the first invalid model directory or evaluation error.",
    )
    cli_args = parser.parse_args()

    import torch

    evaluation_device = torch.device(cli_args.device)
    if evaluation_device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA evaluation was requested, but torch.cuda.is_available() is false")
    if evaluation_device.type == "cuda":
        torch.cuda.set_device(evaluation_device)
    evaluate(cli_args.model_paths, evaluation_device, fail_fast=cli_args.fail_fast)
