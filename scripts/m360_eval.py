"""Train and extract meshes for the Mip-NeRF 360 scenes used in the paper."""

import shlex
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path


MIP360_SCENES = ["bonsai", "kitchen"]
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def run(command):
    print(shlex.join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--mipnerf360", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/mipnerf360"))
    parser.add_argument("--scenes", nargs="+", choices=MIP360_SCENES, default=MIP360_SCENES)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-rendering", action="store_true")
    args = parser.parse_args()
    args.mipnerf360 = args.mipnerf360.resolve()
    args.output = args.output.resolve()

    for scene in args.scenes:
        source = args.mipnerf360 / scene
        model = args.output / scene
        if not args.skip_training:
            run([
                sys.executable, "train.py", "-s", source, "-m", model,
                "--images", "images_2", "--resolution", "1", "--eval", "--quiet",
                "--test_iterations", "-1", "--lambda_dist", "100",
            ])
        if not args.skip_rendering:
            run([
                sys.executable, "render.py", "-s", source, "-m", model,
                "--iteration", "30000", "--quiet", "--skip_train", "--skip_test",
            ])


if __name__ == "__main__":
    main()
