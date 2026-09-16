# Object-Aware Loss for Mask-Guided Mesh Reconstruction Based on 2D Gaussian Splatting

**Sunghyun Woo, Jeongil Seo, HeeKyung Lee**
*Mathematics*, 2026, 14(17), 3221

[Paper](https://www.mdpi.com/2227-7390/14/17/3221) | [DOI](https://doi.org/10.3390/math14173221)

![Overview](assets/pipeline.png)

This repository implements our object-aware loss on top of
[2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting).
Masked L1 supervision, alpha polarization, and learnable ObjectMark scores
focus reconstruction on a target object. Gaussians with low ObjectMark scores
are pruned during training; geometry regularization and TSDF fusion follow 2DGS.

## Installation

Requires an NVIDIA GPU, a CUDA-compatible C++ compiler (e.g. GCC 11), and a
complete CUDA Toolkit 11.8 installation for the CUDA
extensions. The environment uses Python 3.10, PyTorch 2.5.1, torchvision 0.20.1,
and Open3D 0.18.0. Set `CUDA_HOME` to your CUDA 11.8 installation if multiple
toolkits are installed.

```bash
git clone --recursive https://github.com/WooSungHyun03/object-aware-loss.git
cd object-aware-loss
conda env create --file environment.yml
conda activate object-aware-2dgs
```

For an existing clone, run `git submodule update --init --recursive` first.
The rasterizer and Simple-KNN are compiled during environment creation.
SAM2 is optional and is installed separately below.

## Data preparation

Use undistorted images and COLMAP camera poses, following
[2DGS data preparation](https://github.com/hbb1/2d-gaussian-splatting#quick-examples).
For custom captures, `python convert.py -s <scene>` runs the inherited COLMAP
conversion from `<scene>/input/`; COLMAP and ImageMagick must be installed separately.
Generate masks after undistortion, so masks and training images share coordinates.

```text
<scene>/
├── images/
│   ├── 000000.jpg
│   └── ...
├── mask/
│   ├── 000000.png
│   └── ...
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

Provide one binary target-object mask per view: white is foreground, black is
background. Masks in `mask/` must have the same filename stem as the RGB image.
Alternatively, use the alpha channel of RGBA images, as in the preprocessed
DTU data. External masks take precedence. Both forms are binarized and resized
with nearest-neighbor interpolation; missing or ambiguous masks stop training.

The paper uses 15 DTU scans, eight BlendedMVS scenes, and the bonsai and kitchen
scenes from Mip-NeRF 360. DTU and BlendedMVS use supplied ground-truth masks;
Mip-NeRF 360 uses SAM2 masks. Download the
[preprocessed DTU data](https://huggingface.co/datasets/dylanebert/2DGS),
[BlendedMVS](https://github.com/YoYo000/BlendedMVS), and
[Mip-NeRF 360](https://jonbarron.info/mipnerf360/) from their respective sources.
BlendedMVS must be prepared in the COLMAP layout above; its native MVS camera
format is not read by this repository.

### SAM2 masks (optional)

Install the pinned [SAM2](https://github.com/facebookresearch/sam2) submodule
only if masks are unavailable:

```bash
pip install --no-build-isolation -e submodules/sam2
mkdir -p checkpoints
curl -L https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -o checkpoints/sam2.1_hiera_large.pt
python scripts/generate_masks.py --dataset-dir <scene> \
  --checkpoint checkpoints/sam2.1_hiera_large.pt --ui
```

The first image in lexicographic order is shown for prompting. Left-click adds
foreground points, right-click adds background points, `u` undoes a point, and
Enter starts propagation. Without a display, replace `--ui` with
`--point X Y 1` (repeat for more points) or `--box X1 Y1 X2 Y2`.
Use `--images images_2` for downsampled Mip-NeRF 360 images.

The script propagates one object across equal-sized views and writes lossless
PNG masks to `mask/`. Existing masks require `--overwrite`. Inspect the masks
before training, especially across large viewpoint changes. The helper retains
the existing 64-pixel hole/island cleanup. The paper does not specify the SAM2
checkpoint, prompts, or this cleanup; the provided SAM2.1 configuration is a
usable preprocessing option, not a record of the original experimental masks.

## Training

For a bounded object scene:

```bash
python train.py -s <scene> -m output/<scene> --lambda_dist 1000 --depth_ratio 1
```

Training uses 30,000 iterations, `lambda_pa=0.1`, `lambda_po=1.0`, and
`objectmark_lr=0.01`. ObjectMark scores start at 0.5 and are inherited by cloned
and split Gaussians. Every 500 iterations from iteration 2500 through the final
iteration, scores at or below 0.5 are pruned before saving. These settings are
exposed in `arguments/__init__.py` with the corresponding `--` options.
D-SSIM is not used in the proposed loss; `--lambda_dssim` is inherited but unused.

Depth distortion starts after iteration 3000 and normal consistency after
iteration 7000, following upstream. Set `--lambda_dist 1000` for bounded scenes
or `--lambda_dist 100` for unbounded scenes as described in the paper; the
upstream CLI default is 0. Use `--depth_ratio 1` for median depth or `0` for
expected depth. Specify the same depth ratio when rendering.

The original implementation normalizes Masked L1 by the number of foreground
RGB samples. Equation (12) does not specify this reduction. This convention is
retained rather than changing the relative loss scale. ObjectMark is rendered
in a separate pass as opacity (Equation (16)), with geometry detached so that
this loss updates only ObjectMark scores, consistent with Section 5.6.

For DTU, add `-r 2`. For Mip-NeRF 360, use `-i images_2 -r 1` to obtain the
paper's factor-two resolution. Add `--eval` **during training** to hold out every
eighth COLMAP view for the inherited novel-view evaluation.

## Rendering and evaluation

Extract a bounded TSDF mesh:

```bash
python render.py -m output/<scene> --iteration 30000 --depth_ratio 1 \
  --skip_train --skip_test
```

The scene path is loaded from `cfg_args`; override it with `-s <scene>` if moved.
Meshes are written to `train/ours_30000/fuse.ply` and `fuse_post.ply`. The latter
retains the largest connected components (`--num_cluster`). Bounded fusion uses
available object masks to suppress background depth, as in upstream 2DGS.
Adjust `--voxel_size`, `--depth_trunc`, and `--sdf_trunc` for your scene scale.
The inherited unbounded extractor is available with `--unbounded --mesh_res 1024`.

For models trained with `--eval`, render held-out Gaussian views and compute
full-image PSNR, SSIM, and LPIPS:

```bash
python render.py -m output/<scene> --depth_ratio 1 --skip_train --skip_mesh
python metrics.py -m output/<scene>
```

These are Gaussian-rendering metrics against the original RGB images. The
paper instead evaluates **mesh renderings**, with the mask applied only to the
ground truth for PSNR. Its nvdiffrast mesh-rendering mIoU, mACC, and PSNR pipeline
is not included, so the commands above do not reproduce those tables.

Dataset scripts run the same training and meshing commands:

```bash
python scripts/dtu_eval.py --dtu <prepared_dtu_root> --dtu-official <official_dtu_root>
python scripts/blendedmvs_eval.py --blendedmvs <prepared_root> --scenes <scene_1> <scene_2>
python scripts/m360_eval.py --mipnerf360 <prepared_mipnerf360_root>
```

DTU uses the 15 paper scans by default; select a subset with `--scenes scan24`.
Its Chamfer evaluator additionally needs per-scan `cameras.npz`, the original
1600×1200 PNG images and masks, and the official `ObsMask/` and `Points/stl/`
directories. `--skip-evaluation` runs training and meshing alone. The evaluator
culls and transforms the mesh to DTU world coordinates once before measuring
distance in millimeters.

BlendedMVS and Mip-NeRF 360 scripts train and extract meshes only. The exact
eight BlendedMVS scene IDs, prepared cameras, SAM2 prompts/masks, and original
run configurations are not supplied. Full benchmark reproduction still requires
those assets and the mesh-rendering evaluator. The inherited NeRF-Synthetic and
Tanks and Temples scripts are not paper benchmarks; TnT evaluation retains its
separate Open3D 0.10 requirement.

## Acknowledgements

This code builds on [2DGS](https://github.com/hbb1/2d-gaussian-splatting) and
[3DGS](https://github.com/graphdeco-inria/gaussian-splatting). We thank the authors
of these projects and the inherited Open3D, MultiNeRF, LPIPS, DTU evaluation,
and Tanks and Temples utilities. SAM2 is used for optional mask preprocessing.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and licenses.

This work was supported by the Ministry of Education under the Glocal University
30 Project at Dong-A University (2026-STR-S1-06), and the ETRI grant funded by the
Korean government (2710109867, Development of User-Customized Freeform Future Display).

## Citation

```bibtex
@article{woo2026objectaware,
  title   = {Object-Aware Loss for Mask-Guided Mesh Reconstruction Based on 2D Gaussian Splatting},
  author  = {Woo, Sunghyun and Seo, Jeongil and Lee, HeeKyung},
  journal = {Mathematics},
  year    = {2026},
  volume  = {14},
  number  = {17},
  pages   = {3221},
  doi     = {10.3390/math14173221}
}
```

Please also cite 2DGS:

```bibtex
@inproceedings{Huang2DGS2024,
  title     = {2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
  author    = {Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
  booktitle = {ACM SIGGRAPH 2024 Conference Papers},
  year      = {2024},
  doi       = {10.1145/3641519.3657428}
}
```

## License

The inherited [Gaussian-Splatting License](LICENSE.md) applies to this derivative.
See [NOTICE.md](NOTICE.md) and the component notices for the applicable terms.
