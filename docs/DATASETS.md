# Dataset preparation

The object-aware method consumes a 2DGS-compatible scene plus one foreground
mask per RGB image. No dataset files or masks are redistributed in this
repository.

## Common COLMAP layout

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

`cameras.txt`, `images.txt`, and `points3D.txt` are accepted in place of the
binary COLMAP files. Camera models must already be undistorted
`PINHOLE` or `SIMPLE_PINHOLE`; the inherited loader rejects other models.

Mask requirements:

- The directory name is exactly `mask` (singular).
- Each RGB image must have exactly one mask with the same full filename or the
  same stem and one of `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, or `.tiff`.
- Masks are converted to one luminance channel and normalized to `[0, 1]`.
- White (`1`) is foreground/object and black (`0`) is background.
- If dimensions differ, training attempts COLMAP mask undistortion when a
  suitable model and the `colmap` executable are available. Remaining
  differences use nearest-neighbor resizing. Missing or unreadable masks stop
  object-aware training with a path-specific error.
- RGB images follow inherited 2DGS downsampling: `--resolution 1/2/4/8` applies
  that factor; `--resolution -1` limits images wider than 1,600 pixels unless
  an explicit resolution is supplied. Masks use the identical final size.

SAM 2 is not bundled and this repository does not run mask inference. It may be
used externally to prepare masks, but any method producing the required binary
foreground convention is acceptable.

## DTU

The inherited 2DGS repository documents a preprocessed DTU/COLMAP distribution,
and the release scripts retain its 15-scene evaluation set:

`scan24`, `scan37`, `scan40`, `scan55`, `scan63`, `scan65`, `scan69`, `scan83`,
`scan97`, `scan105`, `scan106`, `scan110`, `scan114`, `scan118`, and `scan122`.

- Prepared 2DGS DTU data: use the download source linked by the upstream
  [2DGS repository](https://github.com/hbb1/2d-gaussian-splatting).
- Official geometry/evaluation data: obtain it from the official
  [DTU Robot Image Data Sets](https://roboimagedata.compute.dtu.dk/).
- Add or regenerate the `mask/` directory according to the convention above;
  confirm the dataset's own terms before use or redistribution.

Run a single scene with `scripts/train_dtu.sh`, then
`scripts/render_dtu.sh` and `scripts/evaluate_dtu.sh`.

## Other inherited dataset loaders

The code retains the upstream COLMAP loader and Blender/NeRF Synthetic loader,
as well as evaluation drivers for Mip-NeRF 360 and Tanks and Temples. Their
object-aware paper settings and scene selections are not recorded in this
repository, so they are not presented as paper-reproduction targets here.
Prepare the same `mask/` association for any such scene.

Use the official dataset sources and comply with their licenses:

- [Mip-NeRF 360](https://jonbarron.info/mipnerf360/)
- [Tanks and Temples](https://www.tanksandtemples.org/)
- [NeRF Synthetic](https://www.matthewtancik.com/nerf)
- BlendedMVS: use the official project distribution; scene selection for this
  paper still needs maintainer confirmation.
