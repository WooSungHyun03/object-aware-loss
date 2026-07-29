# Object-Aware Loss for Mask-Guided Mesh Reconstruction Based on 2D Gaussian Splatting

**Sunghyun Woo · [remaining author list to be confirmed]**

[Paper: available after publication] · [Project page: coming soon]

![Object-aware mask-guided 2DGS pipeline](assets/pipeline.png)

*Method pipeline. The unchanged source is available as
[PDF](assets/fig1.pdf).*

This repository contains the research implementation of an object-aware loss
for target-object mesh reconstruction based on
[2D Gaussian Splatting (2DGS)](https://github.com/hbb1/2d-gaussian-splatting).
It combines foreground-mask RGB supervision, alpha polarization, a learnable
per-Gaussian ObjectMark, rendered ObjectMark polarization, and score-based
Gaussian pruning while retaining the inherited 2DGS geometric regularizers and
mesh extraction workflow.

> **License.** This is a derivative of 2DGS and 3D Gaussian Splatting. The root
> [Gaussian-Splatting License](LICENSE.md) limits use to non-commercial research
> and evaluation and requires its notices to be retained. See
> [NOTICE.md](NOTICE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Overview

Standard 2DGS optimizes a scene-level surfel representation. This method adds
object supervision without replacing that representation:

- **Mask-guided reconstruction** computes L1 RGB error over foreground pixels.
- **Alpha polarization** aligns accumulated Gaussian alpha with the foreground
  mask.
- **ObjectMark estimation** learns one object-relevance logit for each Gaussian.
- **ObjectMark polarization** renders those values through Gaussian compositing
  and aligns the response with the same mask.
- **ObjectMark pruning** repeatedly removes Gaussians whose probability is at or
  below the configured threshold.

The repository was previously presented internally as **FocusGS**. That name is
retained only in the historical GitHub URL and repository directory; it is not
the current paper title. The serialized field `objectmark_score` and legacy
`mask_label` loader alias remain unchanged for checkpoint and PLY compatibility.

## Method pipeline

Inputs are posed multi-view RGB images and matching per-view object masks. The
2DGS renderer produces RGB, alpha, normals, depth, and distortion maps. During
optimization, foreground-masked RGB loss supervises appearance, alpha
polarization separates object and background, and a dedicated compositing pass
produces the rendered ObjectMark response. The per-Gaussian scores propagate
through densification and are used to prune background Gaussians. The remaining
2D surfels are converted to a bounded TSDF mesh or an unbounded contracted-space
mesh using the inherited 2DGS extraction code.

Paper-specific losses, masks, schedules, and pruning orchestration are isolated
in `object_aware/`; generic rasterization remains in `gaussian_renderer/`, and
Gaussian parameters plus optimizer-state updates remain in
`scene/gaussian_model.py`.

## Installation

The code targets CUDA-capable Linux. The retained experimental environment and
local release validation use Python 3.8.18, PyTorch 2.0.0, torchvision 0.15.0,
PyTorch's CUDA 11.8 runtime, and Open3D 0.18.0. The exact paper compiler,
driver, GPU, and system CUDA compiler toolkit were not recorded, so this
release does not claim an exact paper hardware/toolchain combination. Building
the extensions requires a system CUDA toolkit compatible with the PyTorch
CUDA 11.8 build.

```bash
git clone --recursive https://github.com/WooSungHyun03/FocusGS.git
cd FocusGS

conda env create -f environment.yml
conda activate object-aware-2dgs
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

`environment.yml` installs both extensions. To rebuild them explicitly after a
CUDA or compiler change:

```bash
pip install --no-build-isolation --force-reinstall submodules/diff-surfel-rasterization
pip install --no-build-isolation --force-reinstall submodules/simple-knn
```

Verify the Python packages and compiled modules:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import diff_surfel_rasterization, simple_knn; print('CUDA extensions import successfully')"
```

`requirements.txt` is for an existing compatible PyTorch/CUDA environment; it
does not install PyTorch or the two local CUDA extensions.

## Dataset preparation

For COLMAP scenes, use this layout:

```text
data/<scene>/
├── images/
│   ├── 000000.png
│   └── ...
├── mask/
│   ├── 000000.png
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

The mask directory is `mask` (singular). A mask must share the RGB image's full
filename or stem. PNG, JPEG, BMP, and TIFF masks are accepted, converted to one
channel, and normalized to `[0, 1]`; white is foreground and black is
background. Missing, ambiguous, or unreadable masks stop object-aware training
with the affected path. Nearest-neighbor resizing follows the RGB resolution;
when original mask/image dimensions differ, the loader first attempts the
existing COLMAP undistortion path if `colmap` and a suitable camera model are
available.

COLMAP cameras must already use `PINHOLE` or `SIMPLE_PINHOLE`. With
`--resolution -1`, inherited 2DGS behavior automatically limits image width to
1,600 pixels; explicit factors `1`, `2`, `4`, and `8` apply the same resize to
RGB and masks.

SAM 2 is not included or invoked. It is optional external preprocessing for
generating masks; this code consumes pre-generated masks only. Dataset-specific
notes and license reminders are in [docs/DATASETS.md](docs/DATASETS.md).

## Training

Minimal object-aware training:

```bash
python train.py \
    -s /path/to/data/scene \
    -m output/scene
```

The default run performs 30,000 iterations, tests and saves at 7,000 and 30,000,
activates ObjectMark on `[0, 30000)`, applies ObjectMark guidance from iteration
0, and prunes at 2,500, 3,000, ..., 29,500. Outputs are written below the `-m`
directory.

For the DTU command reconstructed from the inherited driver and recorded
object-aware defaults:

```bash
DATA_ROOT=/path/to/DTU \
OUTPUT_ROOT=output/dtu \
SCENE=scan24 \
bash scripts/train_dtu.sh
```

Important method arguments:

| Argument | Default | Meaning |
| --- | ---: | --- |
| `--lambda_pa` | `0.1` | Alpha polarization weight |
| `--lambda_po` | `1.0` | Rendered ObjectMark polarization weight |
| `--objectmark_score_lr` | `0.01` | ObjectMark-logit learning rate |
| `--objectmark_guidance_from_iter` | `0` | First ObjectMark-loss iteration |
| `--objectmark_start_iter` | `0` | Inclusive ObjectMark activation iteration |
| `--objectmark_end_iter` | `30000` | Exclusive ObjectMark activation iteration |
| `--objectmark_pruning_threshold` | `0.5` | Prune scores `<=` this value |
| `--objectmark_pruning_iterations` | generated | Explicit pruning iterations; omission uses `2500:500:29500` for a 30k run |
| `--cache_masks_on_gpu` | off | Cache all masks on GPU instead of transferring per view |
| `--disable_mask_l1` | off | Use full-image L1 for ablation/baseline operation |
| `--disable_polarization_alpha_loss` | off | Disable alpha polarization |
| `--disable_objectmark_filtering` | off | Remove ObjectMark parameter, render, loss, and pruning |

`--lambda_polarization`, `--lambda_objectmark_foreground`, and
`--lambda_objectmark_background` remain as legacy command aliases. New commands
should use `--lambda_pa` and `--lambda_po`. SSIM remains available for evaluation
but is deliberately not part of the proposed masked RGB training loss.

## Rendering and mesh extraction

Render train and test views without meshing:

```bash
python render.py -m output/scene --iteration 30000 --skip_mesh
```

Extract a bounded TSDF mesh without exporting view images:

```bash
python render.py \
    -m output/scene \
    --iteration 30000 \
    --skip_train --skip_test \
    --depth_ratio 1.0 \
    --voxel_size 0.004 \
    --sdf_trunc 0.016 \
    --depth_trunc 3.0 \
    --num_cluster 1
```

If `--voxel_size`, `--sdf_trunc`, or `--depth_trunc` remain negative, the
bounded extractor estimates them from the camera radius and `--mesh_res`.
`--depth_ratio` blends expected depth (`0`) and median depth (`1`). For an
unbounded scene:

```bash
python render.py \
    -m output/scene \
    --iteration 30000 \
    --skip_train --skip_test \
    --unbounded --mesh_res 1024
```

Meshes are written to `output/scene/train/ours_30000/` as `fuse.ply` and
`fuse_post.ply`, or `fuse_unbounded.ply` and `fuse_unbounded_post.ply`.

## Evaluation

For an `--eval` training run, render test views and compute image metrics:

```bash
python render.py -m output/scene --iteration 30000 --skip_train --skip_mesh
python metrics.py -m output/scene
```

`metrics.py` computes PSNR, SSIM, and LPIPS and writes `results.json` and
`per_view.json`. Geometric DTU evaluation requires the separately downloaded
official DTU data:

```bash
DATA_ROOT=/path/to/DTU \
OFFICIAL_DTU_ROOT=/path/to/Official_DTU_Dataset \
OUTPUT_ROOT=output/dtu \
SCENE=scan24 \
bash scripts/evaluate_dtu.sh
```

The bundled DTU evaluator reports its two directed distances and their Chamfer
average. The repository does not directly compute mIoU or mACC. Training time,
allocated VRAM, and Gaussian-count diagnostics are recorded in `time.txt`,
`train_vram.txt`, and `gs.txt`; no paper-table aggregation or verified numeric
results are claimed here. Tanks and Temples evaluation also requires its
official external evaluator data. See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the boundary between
implemented and external metrics.

## Reproduction scripts

- `scripts/train_dtu.sh` trains one of the 15 upstream DTU evaluation scenes with
  explicit object-aware settings.
- `scripts/render_dtu.sh` renders and extracts its bounded mesh.
- `scripts/evaluate_dtu.sh` runs the bundled DTU culling/evaluation path against
  official external data.

All accept `DATA_ROOT`, `OUTPUT_ROOT`, `SCENE`, and `PYTHON_BIN` environment
variables and contain no machine-specific executable paths. The inherited
Python drivers for Mip-NeRF 360, NeRF Synthetic, DTU, and Tanks and Temples are
retained for 2DGS compatibility, but their object-aware paper settings are not
asserted.

## Repository structure

```text
.
├── train.py                  # training entry point
├── render.py                 # view rendering and mesh extraction
├── metrics.py                # PSNR, SSIM, and LPIPS evaluation
├── arguments/                # inherited 2DGS parameter groups
├── gaussian_renderer/        # 2DGS renderer and ObjectMark pass
├── object_aware/             # masks, losses, schedules, and pruning
├── scene/                    # cameras, datasets, Gaussian model/checkpoints
├── utils/                    # inherited geometry, image, and mesh utilities
├── scripts/                  # reproduction and external-evaluator drivers
├── docs/                     # dataset and reproducibility records
├── submodules/               # CUDA rasterizer and Simple-KNN
└── assets/                   # pipeline PNG and original PDF
```

## Troubleshooting

- **CUDA extension build failure:** confirm that `nvcc`, the compiler, PyTorch,
  and the CUDA runtime target compatible versions, then rebuild both submodules
  with `--no-build-isolation`.
- **Submodule import failure:** run `git submodule update --init --recursive` and
  reinstall `submodules/diff-surfel-rasterization` and `submodules/simple-knn`.
- **Missing or mismatched mask:** use one foreground-white mask per RGB image in
  `mask/`; the error reports the first affected path. Size differences are
  nearest-neighbor resized, while filename mismatches are not guessed.
- **Out of memory:** increase `--resolution`, leave `--cache_masks_on_gpu` off,
  or lower `--mesh_res` for unbounded extraction.
- **No or broken bounded mesh:** inspect rendered depth, choose a valid
  `--depth_trunc` for the scene scale, and reduce `--voxel_size` only when memory
  permits. Unbounded extraction is slower but avoids a fixed TSDF bound.
- **No test metrics:** train with `--eval`, render without `--skip_test`, and
  confirm `test/ours_<iteration>/renders` and `gt` both exist.

## Acknowledgements

This code derives from
[2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) by Huang
et al., itself based on
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
by Kerbl et al. It uses the differentiable surfel rasterizer and Simple-KNN
submodules, Open3D TSDF integration, MultiNeRF-derived rendering utilities,
LPIPS evaluation, and adapted DTU and Tanks and Temples evaluators. SAM 2 may be
used externally for preprocessing but is neither included nor executed.
Dataset authors and licenses remain applicable to DTU, BlendedMVS, Mip-NeRF
360, Tanks and Temples, and NeRF Synthetic data. Detailed provenance and known
license-metadata gaps are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

The complete author list, publication URL, year, volume, issue, pages, and DOI
have not yet been verified. Until maintainers supply them, use this conservative
entry and replace the visible author placeholder before publication:

```bibtex
@article{woo_object_aware_2dgs,
  title   = {Object-Aware Loss for Mask-Guided Mesh Reconstruction Based on 2D Gaussian Splatting},
  author  = {Woo, Sunghyun}, % Remaining authors pending verification
  journal = {Mathematics},
  note    = {Publication metadata pending}
}
```

Please also cite the inherited 2DGS work:

```bibtex
@inproceedings{Huang2DGS2024,
  title     = {2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
  author    = {Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
  booktitle = {SIGGRAPH 2024 Conference Papers},
  year      = {2024},
  doi       = {10.1145/3641519.3657428}
}
```

## License

The repository is not MIT- or Apache-licensed as a whole. Inherited 3DGS/2DGS
code and this derivative remain subject to the
[Gaussian-Splatting License](LICENSE.md), including its non-commercial research
and evaluation restriction. Separately identified utilities and dependencies
retain their own notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
