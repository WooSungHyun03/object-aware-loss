#!/usr/bin/env python3
"""Generate loader-ready object masks with the official SAM 2 package.

The first image in lexicographic order is prompted with foreground/background
points or one foreground box. SAM 2 then propagates that object through the
remaining images. The Apache-2.0-licensed SAM 2 implementation is included as
a Git submodule; model checkpoints remain external downloads.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from contextlib import nullcontext
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
OBJECT_ID = 1
MASK_LOGIT_THRESHOLD = 0.0
MIN_REGION_AREA = 64
SAM2_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "submodules" / "sam2"

# Use the implementation pinned by this repository's gitlink. SAM 2's runtime
# dependencies are installed by environment.yml; installing its pyproject would
# create an isolated build environment and download a second PyTorch copy.
if (SAM2_SOURCE_ROOT / "sam2" / "__init__.py").is_file():
    sys.path.insert(0, str(SAM2_SOURCE_ROOT))


class MaskGenerationError(RuntimeError):
    """An actionable SAM 2 preprocessing failure."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prompt the first scene image and propagate a binary object mask "
            "with the official SAM 2 video predictor."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Scene root containing the RGB image directory.",
    )
    parser.add_argument(
        "--images",
        default="images",
        help="RGB directory relative to the scene root (default: images).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Downloaded official SAM 2 checkpoint.",
    )
    parser.add_argument(
        "--model-cfg",
        default=DEFAULT_MODEL_CONFIG,
        help="SAM 2 package configuration identifier.",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--point",
        action="append",
        nargs=3,
        type=float,
        metavar=("X", "Y", "LABEL"),
        help=(
            "Prompt point on the first image; LABEL is 1 for foreground or 0 "
            "for background. Repeat the option to provide multiple points."
        ),
    )
    prompt_group.add_argument(
        "--box",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Foreground box on the first image in pixel coordinates.",
    )
    prompt_group.add_argument(
        "--ui",
        action="store_true",
        help="Collect foreground/background clicks on the first image in an interactive UI.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device used for SAM 2 inference (default: cuda).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace masks with exactly matching filenames.",
    )
    return parser


def _load_image_metadata(image_paths: Sequence[Path]):
    try:
        from PIL import Image
    except ImportError as exc:
        raise MaskGenerationError(
            "Pillow is unavailable; create the environment from environment.yml."
        ) from exc

    expected_size = None
    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                size = image.size
        except Exception as exc:
            raise MaskGenerationError(f"Cannot read image '{image_path}': {exc}") from exc
        if expected_size is None:
            expected_size = size
        elif size != expected_size:
            raise MaskGenerationError(
                "SAM 2 video propagation requires equal image dimensions; "
                f"'{image_path.name}' is {size}, expected {expected_size}."
            )
    return Image, expected_size


def _collect_images(image_dir: Path) -> List[Path]:
    if not image_dir.is_dir():
        raise MaskGenerationError(f"Image directory does not exist: '{image_dir}'.")

    unsupported = sorted(
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS
    )
    if unsupported:
        preview = ", ".join(unsupported[:5])
        raise MaskGenerationError(
            f"Unsupported files in '{image_dir}': {preview}. "
            "Use PNG, JPEG, BMP, or TIFF images only."
        )

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise MaskGenerationError(f"No supported images found in '{image_dir}'.")

    stems = [path.stem for path in image_paths]
    duplicate_stems = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicate_stems:
        raise MaskGenerationError(
            "Image stems must be unique for unambiguous mask loading; duplicates: "
            + ", ".join(duplicate_stems)
        )
    return image_paths


def _validate_output_targets(
    output_dir: Path,
    image_paths: Sequence[Path],
    overwrite: bool,
) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise MaskGenerationError(f"Mask output path is not a directory: '{output_dir}'.")
    if not output_dir.exists():
        return

    existing = [
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    for image_path in image_paths:
        matches = [path for path in existing if path.stem == image_path.stem]
        ambiguous = [path for path in matches if path.name != image_path.name]
        if ambiguous:
            raise MaskGenerationError(
                f"Refusing ambiguous output for '{image_path.name}'; remove or rename "
                + ", ".join(path.name for path in ambiguous)
                + "."
            )
        target = output_dir / image_path.name
        if target.exists() and not overwrite:
            raise MaskGenerationError(
                f"Mask already exists: '{target}'. Pass --overwrite to replace it."
            )


def _collect_ui_prompt(image_path: Path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise MaskGenerationError(
            "The interactive UI requires Matplotlib; create the environment from environment.yml."
        ) from exc

    noninteractive_backends = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    if plt.get_backend().lower() in noninteractive_backends:
        raise MaskGenerationError(
            "The Matplotlib backend is non-interactive. Run the UI in a desktop "
            "session or with X11 forwarding enabled."
        )

    with Image.open(image_path) as image:
        rgb_image = np.asarray(image.convert("RGB"))
    points = []
    labels = []
    point_artists = []

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.imshow(rgb_image)
    axis.set_title(
        "SAM 2 prompt UI\n"
        "Left click: foreground | Right click: background | u: undo | Enter: confirm"
    )
    axis.axis("off")

    def redraw_points():
        while point_artists:
            point_artists.pop().remove()
        for (x, y), label in zip(points, labels):
            color = "lime" if label == 1 else "red"
            point_artists.append(
                axis.scatter(
                    x,
                    y,
                    c=color,
                    s=70,
                    edgecolors="white",
                    linewidths=1.2,
                )
            )
        figure.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != axis or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            label = 1
        elif event.button == 3:
            label = 0
        else:
            return
        points.append([float(event.xdata), float(event.ydata)])
        labels.append(label)
        redraw_points()

    def on_key(event):
        if event.key == "u" and points:
            points.pop()
            labels.pop()
            redraw_points()
        elif event.key == "enter":
            plt.close(figure)

    click_id = figure.canvas.mpl_connect("button_press_event", on_click)
    key_id = figure.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    figure.canvas.mpl_disconnect(click_id)
    figure.canvas.mpl_disconnect(key_id)

    if not points:
        raise MaskGenerationError("No prompt points were provided in the UI.")
    return points, labels, None


def _validate_prompt(args, width: int, height: int, first_image_path: Path):
    if args.ui:
        return _collect_ui_prompt(first_image_path)
    if args.point:
        points = []
        labels = []
        for x, y, label in args.point:
            if not (0 <= x < width and 0 <= y < height):
                raise MaskGenerationError(
                    f"Prompt point ({x}, {y}) is outside the first image ({width} x {height})."
                )
            if label not in (0.0, 1.0):
                raise MaskGenerationError("Point LABEL must be 1 (foreground) or 0 (background).")
            points.append([x, y])
            labels.append(int(label))
        return points, labels, None

    x1, y1, x2, y2 = args.box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise MaskGenerationError(
            f"Prompt box must lie inside the first image ({width} x {height}) "
            "with X1 < X2 and Y1 < Y2."
        )
    return None, None, [x1, y1, x2, y2]


def _validate_model_config(model_cfg: str) -> None:
    try:
        config_resource = resources.files("sam2").joinpath(model_cfg)
    except (ModuleNotFoundError, AttributeError) as exc:
        raise MaskGenerationError(
            "SAM 2 is unavailable. Initialize submodules recursively and create "
            "the environment from environment.yml."
        ) from exc
    if not config_resource.is_file():
        raise MaskGenerationError(
            f"SAM 2 model configuration was not found in the installed package: '{model_cfg}'."
        )


def _prepare_video_frames(Image, image_paths: Sequence[Path], frame_dir: Path) -> None:
    frame_dir.mkdir()
    digits = max(6, len(str(len(image_paths) - 1)))
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            image.convert("RGB").save(
                frame_dir / f"{index:0{digits}d}.jpg",
                format="JPEG",
                quality=100,
                subsampling=0,
            )


def _clean_mask(mask, remove_small_regions):
    mask, _ = remove_small_regions(mask, MIN_REGION_AREA, mode="holes")
    mask, _ = remove_small_regions(mask, MIN_REGION_AREA, mode="islands")
    return mask


def _save_mask(Image, mask, target: Path) -> None:
    import numpy as np

    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    suffix = target.suffix.lower()
    save_options = {}
    if suffix in {".jpg", ".jpeg"}:
        save_options = {"format": "JPEG", "quality": 100, "subsampling": 0}
    elif suffix == ".png":
        save_options = {"format": "PNG", "compress_level": 9}
    mask_image.save(target, **save_options)


def _verify_staged_masks(
    Image,
    staged_dir: Path,
    image_paths: Sequence[Path],
    expected_size: Tuple[int, int],
) -> None:
    import numpy as np

    staged_paths = [staged_dir / image_path.name for image_path in image_paths]
    if len(staged_paths) != len(image_paths) or any(not path.is_file() for path in staged_paths):
        raise MaskGenerationError("Generated image and mask counts do not match.")

    first_has_foreground = False
    for index, mask_path in enumerate(staged_paths):
        try:
            with Image.open(mask_path) as mask_image:
                if mask_image.size != expected_size:
                    raise MaskGenerationError(
                        f"Generated mask '{mask_path.name}' has size {mask_image.size}, "
                        f"expected {expected_size}."
                    )
                binary = np.asarray(mask_image.convert("L")) >= 128
        except MaskGenerationError:
            raise
        except Exception as exc:
            raise MaskGenerationError(
                f"Cannot verify generated mask '{mask_path}': {exc}"
            ) from exc
        if index == 0:
            first_has_foreground = bool(binary.any())
    if not first_has_foreground:
        raise MaskGenerationError(
            "The prompt produced an empty mask on the first image; adjust the prompt."
        )


def generate_masks(args) -> Path:
    dataset_dir = args.dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise MaskGenerationError(f"Dataset directory does not exist: '{dataset_dir}'.")
    if not (SAM2_SOURCE_ROOT / "sam2" / "__init__.py").is_file():
        raise MaskGenerationError(
            "The SAM 2 submodule is missing. Clone with --recursive or run "
            "'git submodule update --init --recursive'."
        )
    if Path(args.images).is_absolute() or ".." in Path(args.images).parts:
        raise MaskGenerationError("--images must name a directory inside --dataset-dir.")
    image_dir = dataset_dir / args.images
    output_dir = dataset_dir / "mask"
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise MaskGenerationError(f"SAM 2 checkpoint does not exist: '{checkpoint}'.")

    image_paths = _collect_images(image_dir)
    Image, expected_size = _load_image_metadata(image_paths)
    prompt_points, prompt_labels, prompt_box = _validate_prompt(
        args,
        expected_size[0],
        expected_size[1],
        image_paths[0],
    )
    _validate_output_targets(output_dir, image_paths, args.overwrite)

    try:
        import numpy as np
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.utils.amg import remove_small_regions
    except ImportError as exc:
        raise MaskGenerationError(
            "SAM 2 dependencies are incomplete. Initialize submodules recursively "
            "and create the environment from environment.yml."
        ) from exc

    _validate_model_config(args.model_cfg)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise MaskGenerationError("CUDA was requested, but PyTorch reports no available CUDA device.")

    with tempfile.TemporaryDirectory(prefix=".sam2-mask-", dir=dataset_dir) as temp_name:
        temp_root = Path(temp_name)
        frame_dir = temp_root / "frames"
        staged_dir = temp_root / "mask"
        staged_dir.mkdir()
        _prepare_video_frames(Image, image_paths, frame_dir)

        try:
            predictor = build_sam2_video_predictor(
                args.model_cfg,
                str(checkpoint),
                device=str(device),
            )
        except Exception as exc:
            raise MaskGenerationError(
                f"Failed to initialize SAM 2 with config '{args.model_cfg}' and "
                f"checkpoint '{checkpoint}': {exc}"
            ) from exc

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        predictions: Dict[int, object] = {}
        try:
            with torch.inference_mode(), autocast_context:
                inference_state = predictor.init_state(
                    video_path=str(frame_dir),
                    async_loading_frames=False,
                )
                if prompt_points is not None:
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=OBJECT_ID,
                        points=np.asarray(prompt_points, dtype=np.float32),
                        labels=np.asarray(prompt_labels, dtype=np.int32),
                    )
                else:
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=OBJECT_ID,
                        box=np.asarray(prompt_box, dtype=np.float32),
                    )

                for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(
                    inference_state
                ):
                    object_ids = list(object_ids)
                    if OBJECT_ID not in object_ids:
                        raise MaskGenerationError(
                            f"SAM 2 omitted the prompted object on frame {frame_idx}."
                        )
                    object_index = object_ids.index(OBJECT_ID)
                    mask = (
                        mask_logits[object_index] > MASK_LOGIT_THRESHOLD
                    ).detach().cpu().numpy().squeeze()
                    if mask.ndim != 2:
                        raise MaskGenerationError(
                            f"SAM 2 returned an invalid mask shape on frame {frame_idx}: {mask.shape}."
                        )
                    predictions[int(frame_idx)] = _clean_mask(mask.astype(bool), remove_small_regions)
        except MaskGenerationError:
            raise
        except Exception as exc:
            raise MaskGenerationError(f"SAM 2 mask propagation failed: {exc}") from exc

        expected_indices = set(range(len(image_paths)))
        if set(predictions) != expected_indices:
            missing = sorted(expected_indices - set(predictions))
            raise MaskGenerationError(
                "SAM 2 did not return one mask per image; missing frame indices: "
                + ", ".join(map(str, missing))
            )

        for index, image_path in enumerate(image_paths):
            _save_mask(Image, predictions[index], staged_dir / image_path.name)
        _verify_staged_masks(Image, staged_dir, image_paths, expected_size)

        output_dir.mkdir(exist_ok=True)
        for image_path in image_paths:
            os.replace(staged_dir / image_path.name, output_dir / image_path.name)

    print(f"Wrote {len(image_paths)} masks to '{output_dir}'.")
    return output_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        generate_masks(args)
    except MaskGenerationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
