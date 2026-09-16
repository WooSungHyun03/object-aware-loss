# Object-Aware Loss for Mask-Guided Mesh Reconstruction Based on 2D Gaussian Splatting

Sunghyun Woo, Jeongil Seo, HeeKyung Lee — *Mathematics* (MDPI), 2026

[Paper](https://www.mdpi.com/2227-7390/14/17/3221)

![Overview](assets/pipeline.png)

We add an object-aware loss to 2D Gaussian Splatting for mask-guided, object-level mesh reconstruction: photometric supervision is restricted to the object mask, and a learnable per-Gaussian ObjectMark score is polarized toward the mask and used to prune background Gaussians during training.

## Installation

```bash
git clone --recursive https://github.com/WooSungHyun03/object-aware-loss.git
cd object-aware-loss
conda env create --file environment.yml
conda activate object-aware-2dgs
```

## Dataset

Prepare a COLMAP scene as in [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting?tab=readme-ov-file#quick-examples), and additionally provide one binary object mask per view:

```
<scene>/
├── images/
├── mask/          # white = object, black = background; same filename as the image
└── sparse/0/
```

A mask may instead be supplied through the image's alpha channel, as in the preprocessed DTU/BlendedMVS data.

## Training

```bash
python train.py -s <scene> -m <output>
```

## Rendering / mesh extraction

```bash
python render.py -s <scene> -m <output>
```
See the [2DGS README](https://github.com/hbb1/2d-gaussian-splatting#testing) for meshing options.

## Evaluation

`scripts/dtu_eval.py` and `scripts/m360_eval.py` are unchanged from 2DGS; see the [2DGS README](https://github.com/hbb1/2d-gaussian-splatting#full-evaluation) for usage.

## Acknowledgements

This project builds on [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting), which builds on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). This derivative is released under the inherited [Gaussian-Splatting License](LICENSE.md).

## Citation

```bibtex
@article{woo2026objectaware,
  title   = {Object-Aware Loss for Mask-Guided Mesh Reconstruction Based on 2D Gaussian Splatting},
  author  = {Woo, Sunghyun and Seo, Jeongil and Lee, HeeKyung},
  journal = {Mathematics},
  year    = {2026},
  volume  = {14},
  number  = {17},
  pages   = {3221}
}
```
