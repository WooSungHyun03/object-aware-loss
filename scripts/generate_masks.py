import os
import tempfile
from argparse import ArgumentParser
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image


def collect_points(image):
    import matplotlib.pyplot as plt

    points, labels = [], []
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title('Left: foreground | Right: background | u: undo | Enter: confirm')

    def on_click(event):
        if event.inaxes == ax and event.button in (1, 3):
            points.append([event.xdata, event.ydata])
            labels.append(int(event.button == 1))
            ax.scatter(*points[-1], c='lime' if labels[-1] else 'red')
            fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'u' and points:
            points.pop()
            labels.pop()
            ax.collections[-1].remove()
            fig.canvas.draw_idle()
        elif event.key == 'enter':
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    if not points:
        raise ValueError('No prompt points selected. Use --point or --box without a display.')
    return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def generate_masks(args):
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.utils.amg import remove_small_regions

    image_dir = args.dataset_dir / args.images
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in extensions and p.is_file())
    if not image_paths:
        raise ValueError(f'No images found in {image_dir}')
    if len({p.stem for p in image_paths}) != len(image_paths):
        raise ValueError('Image stems must be unique')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    output_dir = args.dataset_dir / 'mask'
    if output_dir.exists():
        for path in image_paths:
            matches = [p for p in output_dir.iterdir() if p.stem == path.stem and p.suffix.lower() in extensions]
            if any(p.name != path.stem + '.png' for p in matches):
                raise ValueError(f'Conflicting mask extension for {path.name} in {output_dir}')
            if matches and not args.overwrite:
                raise FileExistsError(f'Mask exists for {path.name}; use --overwrite to replace it')

    with Image.open(image_paths[0]) as image:
        first_image = image.convert('RGB')
    width, height = first_image.size
    points, labels, box = None, None, None
    if args.ui:
        points, labels = collect_points(first_image)
    elif args.point:
        prompts = np.asarray(args.point, dtype=np.float32)
        points, labels = prompts[:, :2], prompts[:, 2]
        if not np.isin(labels, [0, 1]).all():
            raise ValueError('Point labels must be 0 or 1')
        if not ((points >= 0).all() and (points[:, 0] < width).all() and (points[:, 1] < height).all()):
            raise ValueError('Prompt points must lie inside the first image')
        labels = labels.astype(np.int32)
    else:
        x1, y1, x2, y2 = args.box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('Prompt box must lie inside the first image with x1 < x2 and y1 < y2')
        box = np.asarray(args.box, dtype=np.float32)

    # SAM2 expects numerically named JPEG frames.
    with tempfile.TemporaryDirectory(prefix='.sam2-', dir=args.dataset_dir) as temp:
        frame_dir = Path(temp) / 'frames'
        mask_dir = Path(temp) / 'mask'
        frame_dir.mkdir()
        mask_dir.mkdir()
        for index, path in enumerate(image_paths):
            with Image.open(path) as image:
                if image.size != (width, height):
                    raise ValueError(f'All images must have the same dimensions: {path}')
                image.convert('RGB').save(frame_dir / f'{index:06d}.jpg', quality=100, subsampling=0)

        predictor = build_sam2_video_predictor(args.model_cfg, str(args.checkpoint), device=args.device)
        autocast = torch.autocast('cuda', dtype=torch.bfloat16) if torch.device(args.device).type == 'cuda' else nullcontext()
        with torch.inference_mode(), autocast:
            state = predictor.init_state(video_path=str(frame_dir))
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, points=points, labels=labels, box=box)
            for index, object_ids, logits in predictor.propagate_in_video(state):
                mask = (logits[list(object_ids).index(1), 0] > 0).cpu().numpy()
                mask, _ = remove_small_regions(mask, 64, mode='holes')
                mask, _ = remove_small_regions(mask, 64, mode='islands')
                if index == 0 and not mask.any():
                    raise ValueError('The first mask is empty; adjust the prompt')
                Image.fromarray(mask.astype(np.uint8) * 255).save(mask_dir / (image_paths[index].stem + '.png'))

        if len(list(mask_dir.iterdir())) != len(image_paths):
            raise RuntimeError('SAM2 did not return a mask for every image')
        output_dir.mkdir(exist_ok=True)
        for path in mask_dir.iterdir():
            os.replace(path, output_dir / path.name)
    print(f'Wrote {len(image_paths)} masks to {output_dir}')


if __name__ == '__main__':
    parser = ArgumentParser(description='Prompt an object on the first view and propagate masks with SAM2.')
    parser.add_argument('--dataset-dir', required=True, type=Path)
    parser.add_argument('--images', default='images')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--model-cfg', default='configs/sam2.1/sam2.1_hiera_l.yaml')
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument('--point', action='append', nargs=3, type=float, metavar=('X', 'Y', 'LABEL'))
    prompt.add_argument('--box', nargs=4, type=float, metavar=('X1', 'Y1', 'X2', 'Y2'))
    prompt.add_argument('--ui', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    generate_masks(args)
