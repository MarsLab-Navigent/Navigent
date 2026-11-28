#!/usr/bin/env python3
"""Build the fixed TRAIN task set and execute one navigation run."""

import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SCENES = ROOT / "data" / "scene_datasets"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "output")
    args = parser.parse_args(argv)

    output = args.output if args.output.is_absolute() else HERE / args.output
    output.mkdir(parents=True, exist_ok=True)
    task_file = output / "train_tasks.json"

    build_command = [
        sys.executable,
        str(HERE / "make_train_tasks.py"),
        "--dataset",
        str(args.dataset.resolve()),
        "--output",
        str(task_file.resolve()),
    ]
    build = subprocess.run(build_command, cwd=ROOT)
    if build.returncode:
        return build.returncode

    run_command = [
        sys.executable,
        "-m",
        "eval",
        "--registration",
        str(task_file.resolve()),
        "--data-path",
        str(args.dataset.resolve()),
        "--scenes-dir",
        str(DEFAULT_SCENES.resolve()),
        "--episode",
        "all",
        "--gpu",
        "0",
        "--output-root",
        str((output / "baseline").resolve()),
    ]
    return subprocess.run(run_command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
