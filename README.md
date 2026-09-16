# Object-Aware Loss for Mask-Guided Mesh Reconstruction Based on 2D Gaussian Splatting

[Paper](https://www.mdpi.com/2227-7390/14/17/3221) | [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)

![Overview](assets/pipeline.png)

This repository adds an object-aware loss to [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) for mask-guided, object-level mesh reconstruction. Given a per-view binary object mask, training (1) restricts the photometric loss to the object region, (2) polarizes the rendered alpha and a learnable per-Gaussian ObjectMark score toward the mask, and (3) periodically prunes Gaussians whose ObjectMark score stays low. 2DGS's geometry regularization and TSDF meshing are unchanged.

## Installation

```bash
git clone --recursive https://github.com/WooSungHyun03/object-aware-loss.git
cd object-aware-loss
conda env create --file environment.yml
conda activate object-aware-2dgs
```

## Dataset

Prepare a COLMAP scene as in [2DGS](https://github.com/hbb1/2d-gaussian-splatting?tab=readme-ov-file#quick-examples), and additionally provide one binary object mask per view:

```
<scene>/
├── images/
├── mask/          # white = object, black = background; same filename as the image
└── sparse/0/
```

A mask can instead be supplied through the image's alpha channel, as in the preprocessed DTU/BlendedMVS data. Masks can come from any segmentation tool (e.g. [SAM2](https://github.com/facebookresearch/sam2)); the paper uses the dataset-provided masks for DTU/BlendedMVS and SAM2 for Mip-NeRF 360.

## Training

```bash
python train.py -s <scene> -m <output>
```

New arguments on top of 2DGS:
```bash
--lambda_pa  # weight of the alpha polarization loss
--lambda_po  # weight of the ObjectMark polarization loss
```
The masked photometric loss replaces 2DGS's L1 + D-SSIM term with L1 only, so `--lambda_dssim` has no effect.

## Rendering / mesh extraction

Unchanged from 2DGS:
```bash
python render.py -s <scene> -m <output>
```
See the [2DGS README](https://github.com/hbb1/2d-gaussian-splatting#testing) for the meshing arguments (`--depth_ratio`, `--voxel_size`, `--depth_trunc`, `--unbounded`, ...). Since training loads an object mask, bounded mesh extraction suppresses background depth using 2DGS's existing mask-based fusion.

## Evaluation

Dataset drivers for the paper's benchmarks, mirroring 2DGS's own `dtu_eval.py`/`m360_eval.py`:
```bash
python scripts/dtu_eval.py --dtu <dtu_root> --dtu-official <dtu_official_root>
python scripts/blendedmvs_eval.py --blendedmvs <blendedmvs_root> --scenes <scene1> <scene2> ...
python scripts/m360_eval.py --mipnerf360 <mipnerf360_root>
```

## Acknowledgements

This project builds on [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting), which builds on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). We thank the authors for their code.

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

This project is released under the inherited [Gaussian-Splatting License](LICENSE.md).
