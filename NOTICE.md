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

The paper-specific implementation is concentrated in `object_aware/`, the
ObjectMark integration in `scene/gaussian_model.py` and `gaussian_renderer/`,
the mask-aware camera path in `utils/camera_utils.py`, and the corresponding
training orchestration in `train.py`. The internal `objectmark_score` name is
retained for checkpoint and PLY compatibility.

All upstream source-file headers have been retained. Additional component and
license provenance is recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
