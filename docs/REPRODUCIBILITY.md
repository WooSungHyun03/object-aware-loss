# Reproducibility record

## Recorded software environment

`environment.yml` is the canonical environment specification inherited from
the experimental repository and cleaned for release. Its core recorded pins
are Python 3.8.18, PyTorch 2.0.0, torchvision 0.15.0, the PyTorch CUDA 11.8
runtime, Pillow 10.2.0, and Open3D 0.18.0. `requirements.txt` contains only
Python dependencies for users who already have a compatible PyTorch/CUDA
installation.

The repository contains no trustworthy record of the system CUDA compiler,
compiler version, paper GPU model, driver, operating-system release, or other
paper hardware. Those details must be filled in by the maintainers before
claiming exact hardware reproduction. A CUDA-capable Linux environment and a
CUDA compiler compatible with the PyTorch 11.8 build are required by the two
compiled submodules and the training code.

Release smoke tests were run on an NVIDIA GeForce RTX 4080 with driver
580.173.02, using the pre-existing compiled extensions. This identifies the
validation host only; it is not asserted to be the hardware used for the paper.

## Data and masks

See `docs/DATASETS.md`. For object-aware training, every RGB training image must
have a readable foreground-white mask in `<scene>/mask`. The loader validates
associations before model initialization.

## DTU principal experiment

The settings below are reconstructed from the repository's DTU driver and the
recorded object-aware defaults. Set paths without editing the scripts:

```bash
DATA_ROOT=/data/DTU \
OUTPUT_ROOT=output/dtu \
SCENE=scan24 \
bash scripts/train_dtu.sh

DATA_ROOT=/data/DTU \
OUTPUT_ROOT=output/dtu \
SCENE=scan24 \
bash scripts/render_dtu.sh

DATA_ROOT=/data/DTU \
OFFICIAL_DTU_ROOT=/data/Official_DTU_Dataset \
OUTPUT_ROOT=output/dtu \
SCENE=scan24 \
bash scripts/evaluate_dtu.sh
```

The training script makes the relevant settings explicit: 30,000 iterations,
resolution factor 2, `depth_ratio=1`, `lambda_dist=1000`, `lambda_pa=0.1`,
`lambda_po=1.0`, ObjectMark active on `[0, 30000)`, pruning threshold `0.5`,
and repeated pruning at 2,500, 3,000, ..., 29,500. A score exactly equal to the
threshold is pruned.

## Randomness and determinism

`utils.general_utils.safe_state()` seeds Python, NumPy, and PyTorch with `0`
before training and fixes the active device to `cuda:0` (or the first device
made visible through `CUDA_VISIBLE_DEVICES`). It does not enable PyTorch
deterministic algorithms or seed every multi-GPU generator. CUDA rasterization,
atomic accumulation, floating-point reduction order, densification, and pruning
can therefore produce small run-to-run differences.

## Checkpoints and outputs

- `-m <output>` stores `cfg_args`, `cameras.json`, and the initial point cloud.
- `point_cloud/iteration_<N>/point_cloud.ply` is the saved Gaussian model.
- `chkpnt<N>.pth` is created only for requested `--checkpoint_iterations` and
  stores `(model_state, iteration)` including optimizer state.
- Checkpoints and PLYs containing `objectmark_score` remain supported. The
  legacy PLY field and optimizer-group name `mask_label` are accepted on load.
- `train/ours_<N>/renders`, `gt`, and `vis` are created by `render.py`.
- Bounded mesh extraction writes `fuse.ply` and `fuse_post.ply`; unbounded
  extraction writes `fuse_unbounded.ply` and `fuse_unbounded_post.ply`.
- `time.txt`, `train_vram.txt`, `gs.txt`, and `ObjectMark.txt` are diagnostic
  outputs. Training time excludes optional mask undistortion.

## Evaluation scope

`metrics.py` computes PSNR, SSIM, and LPIPS for test renders and writes
`results.json` plus `per_view.json`. These are evaluation metrics; SSIM is not
part of the proposed masked RGB training loss.

The bundled `scripts/eval_dtu/` code computes DTU geometric accuracy after mask
culling but requires separately downloaded official DTU geometry and masks.
The Tanks and Temples scripts likewise require official external evaluation
data. The repository does not implement paper-table aggregation for mIoU,
mACC, training time, VRAM, or Gaussian count; raw timing/count logs are emitted,
and any aggregation protocol must be supplied or verified separately.

## Unverified steps

- Full 15-scene retraining against the paper's original data snapshot.
- Exact paper CUDA compiler/toolchain and hardware reproduction.
- Any BlendedMVS paper scene list or preprocessing recipe.
- Numerical agreement with paper tables; no verified result table is stored in
  the release tree.
- Pretrained checkpoint download and project page; neither is available in the
  repository.
