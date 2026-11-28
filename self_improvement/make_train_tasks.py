#!/usr/bin/env python3
"""Extract a fixed TRAIN task set."""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from task_builder import build_train_tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = build_train_tasks(args.dataset, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Selected {len(value['tasks'])} TRAIN tasks: {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
