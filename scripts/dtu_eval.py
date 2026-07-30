"""Train, mesh, and evaluate the 15 DTU scans used in the paper."""

import shlex
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path


DTU_SCENES = [
    "scan24", "scan37", "scan40", "scan55", "scan63", "scan65",
    "scan69", "scan83", "scan97", "scan105", "scan106", "scan110",
    "scan114", "scan118", "scan122",
]
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def run(command):
    print(shlex.join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dtu", required=True, type=Path, help="Prepared DTU root.")
    parser.add_argument("--output", type=Path, default=Path("output/dtu"))
    parser.add_argument("--scenes", nargs="+", choices=DTU_SCENES, default=DTU_SCENES)
    parser.add_argument("--dtu-official", type=Path, help="Official DTU evaluation root.")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-rendering", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    if not args.skip_evaluation and args.dtu_official is None:
        parser.error("--dtu-official is required unless --skip-evaluation is set")

    for scene in args.scenes:
        source = args.dtu / scene
        model = args.output / scene
        if not args.skip_training:
            run([
                sys.executable, "train.py", "-s", source, "-m", model,
                "--quiet", "--test_iterations", "-1", "--resolution", "2",
                "--depth_ratio", "1.0", "--lambda_dist", "1000",
            ])
        if not args.skip_rendering:
            run([
                sys.executable, "render.py", "-s", source, "-m", model,
                "--iteration", "30000", "--quiet", "--skip_train", "--skip_test",
                "--depth_ratio", "1.0", "--num_cluster", "1",
                "--voxel_size", "0.004", "--sdf_trunc", "0.016",
                "--depth_trunc", "3.0",
            ])
        if not args.skip_evaluation:
            run([
                sys.executable, "scripts/eval_dtu/evaluate_single_scene.py",
                "--input_mesh", model / "train/ours_30000/fuse_post.ply",
                "--scan_id", scene[4:],
                "--output_dir", model / "dtu_evaluation",
                "--mask_dir", args.dtu,
                "--DTU", args.dtu_official,
            ])


if __name__ == "__main__":
    main()
