"""Train and extract meshes for prepared BlendedMVS scenes."""

import shlex
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def run(command):
    print(shlex.join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--blendedmvs", required=True, type=Path)
    parser.add_argument("--scenes", required=True, nargs="+", help="Prepared scene directory names.")
    parser.add_argument("--output", type=Path, default=Path("output/blendedmvs"))
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-rendering", action="store_true")
    args = parser.parse_args()
    if len(args.scenes) != 8 or len(set(args.scenes)) != 8:
        parser.error("--scenes must contain eight distinct BlendedMVS scene names")

    for scene in args.scenes:
        source = args.blendedmvs / scene
        model = args.output / scene
        if not args.skip_training:
            run([
                sys.executable, "train.py", "-s", source, "-m", model,
                "--quiet", "--test_iterations", "-1", "--resolution", "800",
            ])
        if not args.skip_rendering:
            run([
                sys.executable, "render.py", "-s", source, "-m", model,
                "--iteration", "30000", "--quiet", "--skip_train", "--skip_test",
            ])


if __name__ == "__main__":
    main()
