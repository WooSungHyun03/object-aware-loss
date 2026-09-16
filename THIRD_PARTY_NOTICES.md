# Third-party notices

This inventory records the provenance visible in the repository. It describes
included notices; it does not offer a legal interpretation of those licenses.

| Component | Project / source | Role | Form in this repository | License or notice |
| --- | --- | --- | --- | --- |
| Gaussian Splatting | `graphdeco-inria/gaussian-splatting` | Base Gaussian model, scene, camera, training, and utility conventions | Modified derivative | Gaussian-Splatting License; root `LICENSE.md` and retained file headers |
| 2D Gaussian Splatting | `hbb1/2d-gaussian-splatting` | Surfel renderer integration, 2DGS regularizers, meshing, rendering, and evaluation layout | Modified derivative | Gaussian-Splatting License; inherited ShanghaiTech/Inria headers |
| Differentiable Surfel Rasterization | `hbb1/diff-surfel-rasterization` | CUDA rasterization extension | Git submodule | Gaussian-Splatting License in the submodule |
| Simple-KNN | `https://gitlab.inria.fr/bkerbl/simple-knn` | CUDA nearest-neighbor initialization | Git submodule | Gaussian-Splatting License in the submodule |
| MultiNeRF utilities | `google-research/multinerf` | Camera-path and rendering utilities in `utils/render_utils.py` | Copied/adapted utility | Apache License 2.0 notice retained in the file; `THIRD_PARTY_LICENSES/Apache-2.0.txt` |
| PlenOctree spherical harmonics | The PlenOctree Authors | SH utilities in `utils/sh_utils.py` | Copied/adapted utility | BSD-style notice retained in the file |
| sdfstudio marching cubes | `autonomousvision/sdfstudio` | Contracted marching-cubes helper in `utils/mcube_utils.py` | Adapted utility inherited through 2DGS | Apache License 2.0; source URL retained in the file; `THIRD_PARTY_LICENSES/Apache-2.0.txt` |
| Open3D | `isl-org/Open3D` | TSDF integration and mesh processing | Python dependency | Apache License 2.0 (dependency not vendored) |
| Trimesh | `mikedh/trimesh` | Mesh manipulation | Python dependency | MIT (dependency not vendored) |
| LPIPS wrapper | `S-aiueo32/lpips-pytorch`, inherited through 2DGS | LPIPS network wrapper | Copied code in `lpipsPyTorch/` | BSD 2-Clause, Copyright (c) 2020 Sou Uchida; `THIRD_PARTY_LICENSES/lpips-pytorch-BSD-2-Clause.txt` |
| LPIPS weights | `richzhang/PerceptualSimilarity` | Downloaded LPIPS calibration weights | Runtime download | BSD 2-Clause; `THIRD_PARTY_LICENSES/LPIPS-BSD-2-Clause.txt` |
| DTU evaluation | `jzhangbs/DTUeval-python` | Chamfer-style DTU evaluator in `scripts/eval_dtu/` | Copied/adapted evaluator | MIT; `THIRD_PARTY_LICENSES/DTUeval-MIT.txt` |
| Tanks and Temples toolbox | Tanks and Temples Python evaluation toolbox | TnT culling and F-score evaluation in `scripts/eval_tnt/` | Copied/adapted evaluator | MIT headers retained in the applicable files; dataset terms are separate |
| Segment Anything 2 | `facebookresearch/sam2` | Prompted object-mask propagation used by `scripts/generate_masks.py` | Git submodule at `submodules/sam2`; checkpoints are not vendored | Apache License 2.0 in the submodule; `THIRD_PARTY_LICENSES/Apache-2.0.txt` |

The SAM 2 wrapper calls the public API of the official repository pinned by the
`submodules/sam2` gitlink. Model checkpoints are not redistributed. Checkpoints
and dataset downloads remain subject to their respective terms.

License texts for the copied and adapted components are collected in
`THIRD_PARTY_LICENSES/`; source-file notices remain authoritative.

Pinned submodule revisions (unchanged from the incoming repository):

- `diff-surfel-rasterization`: `e0ed0207b3e0669960cfad70852200a4a5847f61`
- `simple-knn`: `86710c2d4b46680c02301765dd79e465819c8f19`
- `sam2`: `2b90b9f5ceec907a1c18123530e92e794ad901a4`

The rasterizer also includes GLM at `5c46b9c07008ae65cb81ab79cd677ecc1934b903`;
its MIT/Happy Bunny license is retained in the nested submodule.
