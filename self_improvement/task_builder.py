"""Build and validate a fixed TRAIN task file."""

import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.evaluation import dataset_tasks, load_tasks

TASK_COUNT = 50
SEED = 20261128


def select_train_tasks(tasks, count=TASK_COUNT, seed=SEED):
    if count < 1:
        raise ValueError("task count must be positive")
    if count > len(tasks):
        raise ValueError(f"Requested {count} tasks, but dataset has {len(tasks)}")

    by_scene = {}
    for task in tasks:
        by_scene.setdefault(task["scene"], []).append(task)
    rng = random.Random(seed)
    scenes = sorted(by_scene)
    rng.shuffle(scenes)
    for rows in by_scene.values():
        rng.shuffle(rows)

    selected = []
    while len(selected) < count:
        for scene in scenes:
            if by_scene[scene] and len(selected) < count:
                selected.append(by_scene[scene].pop())
        if len(selected) < count and not any(by_scene.values()):
            raise ValueError("Unable to select the requested TRAIN tasks")
    return selected


def build_train_tasks(dataset, output, count=TASK_COUNT, seed=SEED):
    tasks = select_train_tasks(dataset_tasks(dataset), count, seed)
    value = {"split": "train", "tasks": tasks}
    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / "train_tasks.json"
        candidate.write_text(json.dumps(value), encoding="utf-8")
        load_tasks(candidate)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and json.loads(output.read_text(encoding="utf-8")) != value:
        raise ValueError("Output already contains a different TRAIN task file")
    output.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return value
