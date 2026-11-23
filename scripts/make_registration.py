#!/usr/bin/env python3
"""Register a full dataset or an explicitly selected subset."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.evaluation import dataset_tasks, load_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "val_seen", "val_unseen", "test"),
        default="val_unseen",
    )
    parser.add_argument("--ids-file", type=Path, help="JSON array of episode IDs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = dataset_tasks(args.dataset)
    if args.ids_file:
        selected = json.loads(args.ids_file.read_text())
        if not isinstance(selected, list) or not selected:
            parser.error("ids-file must contain a nonempty JSON array")
        ids = [str(value) for value in selected]
        if len(ids) != len(set(ids)) or not set(ids) <= {row["id"] for row in tasks}:
            parser.error("ids-file contains duplicate or absent episode IDs")
        tasks = [row for row in tasks if row["id"] in set(ids)]
    value = {"split": args.split, "tasks": tasks}
    # Validate before replacing any existing registration.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory) / "registration.json"
        temporary.write_text(json.dumps(value))
        load_tasks(temporary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and json.loads(args.output.read_text()) != value:
        parser.error(
            "Output already contains a different registration; choose another path"
        )
    args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    print(f"Registered {len(tasks)} episodes: {args.output}")


if __name__ == "__main__":
    main()
