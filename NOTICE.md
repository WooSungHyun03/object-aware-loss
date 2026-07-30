# Derivative-work notice

**Object-Aware Loss for Mask-Guided Mesh Reconstruction Based on 2D Gaussian
Splatting** is a research-code derivative of
[2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting), which in
turn derives from
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

The inherited code and this derivative are distributed under the
Gaussian-Splatting License in [LICENSE.md](LICENSE.md). The license limits use
to non-commercial research and evaluation and requires preservation of the
license and attribution notices. This summary is not a substitute for the
license text.

The paper-specific implementation is concentrated in `train.py`,
`utils/loss_utils.py`, `utils/camera_utils.py`, `scene/gaussian_model.py`, and
`gaussian_renderer/`. The first-party SAM 2 preprocessing wrapper is located at
`scripts/mask_generater.py`; the official SAM 2 implementation is included as a
Git submodule, while its model checkpoints remain external downloads. The
internal `objectmark_score` name is retained for checkpoint and PLY
compatibility.

All upstream source-file headers have been retained. Additional component and
license provenance is recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
