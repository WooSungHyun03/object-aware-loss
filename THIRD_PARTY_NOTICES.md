# Third-party notices

This inventory records the provenance visible in the repository. It describes
included notices; it does not offer a legal interpretation of those licenses.

| Component | Project / source | Role | Form in this repository | License or notice |
| --- | --- | --- | --- | --- |
| Gaussian Splatting | `graphdeco-inria/gaussian-splatting` | Base Gaussian model, scene, camera, training, and utility conventions | Modified derivative | Gaussian-Splatting License; root `LICENSE.md` and retained file headers |
| 2D Gaussian Splatting | `hbb1/2d-gaussian-splatting` | Surfel renderer integration, 2DGS regularizers, meshing, rendering, and evaluation layout | Modified derivative | Gaussian-Splatting License; inherited ShanghaiTech/Inria headers |
| Differentiable Surfel Rasterization | `hbb1/diff-surfel-rasterization` | CUDA rasterization extension | Git submodule | Gaussian-Splatting License in the submodule |
| Simple-KNN | `graphdeco-inria/simple-knn` | CUDA nearest-neighbor initialization | Git submodule | Gaussian-Splatting License in the submodule |
| MultiNeRF utilities | `google-research/multinerf` | Camera-path and rendering utilities in `utils/render_utils.py` | Copied/adapted utility | Apache License 2.0 notice retained in the file |
| PlenOctree spherical harmonics | The PlenOctree Authors | SH utilities in `utils/sh_utils.py` | Copied/adapted utility | BSD-style notice retained in the file |
| sdfstudio marching cubes | `autonomousvision/sdfstudio` | Contracted marching-cubes helper in `utils/mcube_utils.py` | Adapted utility inherited through 2DGS | Source URL retained in the file; no separate copied license file is present |
| Open3D | `isl-org/Open3D` | TSDF integration and mesh processing | Python dependency | Apache License 2.0 (dependency not vendored) |
| Trimesh | `mikedh/trimesh` | Mesh manipulation | Python dependency | MIT (dependency not vendored) |
| LPIPS | `richzhang/PerceptualSimilarity`; wrapper inherited through 2DGS | LPIPS image metric and downloaded calibration weights | Copied wrapper in `lpipsPyTorch/` | The copied wrapper contains no standalone license file; provenance is recorded here and should be confirmed by maintainers before release |
| DTU evaluation | `jzhangbs/DTUeval-python` | Chamfer-style DTU evaluator in `scripts/eval_dtu/` | Copied/adapted evaluator | Source attribution is present; no standalone upstream license file is included here |
| Tanks and Temples toolbox | Tanks and Temples Python evaluation toolbox | TnT culling and F-score evaluation in `scripts/eval_tnt/` | Copied/adapted evaluator | MIT headers retained in the applicable files; dataset terms are separate |

SAM 2 is not included or invoked by this codebase. If used to create input
masks, it is an external preprocessing tool and remains subject to its own
license. Dataset downloads and licenses are likewise not redistributed here;
see `docs/DATASETS.md`.

Two copied evaluation areas (`lpipsPyTorch/` and `scripts/eval_dtu/`) do not
carry standalone license files in the inherited tree. Their source attribution
has been preserved and the missing license metadata is disclosed rather than
silently assigned a license.
