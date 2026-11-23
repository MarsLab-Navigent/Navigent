"""Dataset selection, run identity, and complete-cohort accounting."""

import gzip
import hashlib
import json
import math
import re
from contextlib import contextmanager
from pathlib import Path

SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _episode_rows(value):
    """Return Habitat episodes from the supported JSON container shapes."""
    if not isinstance(value, dict) or not isinstance(value.get("episodes"), list):
        raise ValueError("Dataset must contain an episodes list")
    return value["episodes"]


def _dataset_kind(episode):
    instruction = episode.get("instruction") or {}
    return "rxr" if instruction.get("language") or episode.get("info", {}).get("role") else "r2r"


def sha(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tasks(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Registration must contain a nonempty tasks list")
    result = []
    seen = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Each task must be an object")
        identifier = str(task.get("id", ""))
        if not SAFE_ID.fullmatch(identifier) or identifier in seen:
            raise ValueError("Invalid or duplicate episode ID: " + identifier)
        if any(
            not isinstance(task.get(key), str) or not task[key].strip()
            for key in ("scene", "instruction")
        ):
            raise ValueError("Each task needs scene and instruction strings")
        seen.add(identifier)
        result.append({**task, "id": identifier})
    split = value.get("split", "val_unseen")
    if split not in ("train", "val_seen", "val_unseen", "test"):
        raise ValueError("Unsupported split")
    return split, result


def dataset_tasks(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    tasks = []
    for episode in _episode_rows(value):
        instruction = episode.get("instruction") or {}
        task = {
            "id": str(episode["episode_id"]),
            "scene": episode["scene_id"],
            "instruction": instruction["instruction_text"],
            "dataset": _dataset_kind(episode),
        }
        if episode.get("reference_path"):
            task["reference_path"] = episode["reference_path"]
        if episode.get("trajectory_id") is not None:
            task["trajectory_id"] = str(episode["trajectory_id"])
        if instruction.get("language"):
            task["language"] = instruction["language"]
        tasks.append(task)
    return tasks


def validate_dataset(tasks, path):
    rows = dataset_tasks(path)
    lookup = {row["id"]: row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Dataset contains duplicate episode IDs")
    for task in tasks:
        expected = lookup.get(task["id"])
        if expected is None or any(
            task[key] != expected[key] for key in ("scene", "instruction")
        ):
            raise ValueError("Registered task differs from dataset: " + task["id"])
        if task.get("dataset") and task["dataset"] != expected["dataset"]:
            raise ValueError("Registered dataset kind differs: " + task["id"])
        if expected.get("reference_path") and task.get("reference_path") != expected[
            "reference_path"
        ]:
            raise ValueError("Registered reference path differs: " + task["id"])


def trajectory_scores(predicted, reference, success, success_distance=3.0):
    """Compute the standard continuous VLN trajectory scores.

    Positions are simulator XYZ coordinates. The evaluator owns the reference
    path; it is never included in actor or reviewer requests.
    """
    import numpy as np

    pred = np.asarray(predicted, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if pred.ndim != 2 or pred.shape[1] != 3 or len(pred) == 0:
        raise ValueError("Predicted trajectory must be a nonempty N x 3 array")
    if ref.ndim != 2 or ref.shape[1] != 3 or len(ref) == 0:
        raise ValueError("Reference trajectory must be a nonempty N x 3 array")
    if not np.isfinite(pred).all() or not np.isfinite(ref).all():
        raise ValueError("Trajectory positions must be finite")
    if success_distance <= 0:
        raise ValueError("success_distance must be positive")
    def compress(points):
        kept = [points[0]]
        for point in points[1:]:
            if not np.allclose(point, kept[-1], atol=1e-6, rtol=0.0):
                kept.append(point)
        return np.asarray(kept, dtype=float)

    pred = compress(pred)
    ref = compress(ref)
    point_distances = np.linalg.norm(pred[:, None, :] - ref[None, :, :], axis=2)
    table = np.full((len(pred) + 1, len(ref) + 1), np.inf, dtype=float)
    table[0, 0] = 0.0
    for i in range(1, len(pred) + 1):
        for j in range(1, len(ref) + 1):
            table[i, j] = point_distances[i - 1, j - 1] + min(
                table[i - 1, j], table[i, j - 1], table[i - 1, j - 1]
            )
    dtw = float(table[-1, -1])
    ndtw = float(np.exp(-dtw / (len(ref) * float(success_distance))))
    return {
        "dtw": dtw,
        "ndtw": ndtw,
        "sdtw": ndtw if bool(success) else 0.0,
        "reference_path_length": len(ref),
        "predicted_path_length": len(pred),
    }


@contextmanager
def episode_lock(directory):
    import fcntl

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Episode is already running: " + directory.name) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def summarize(tasks, results):
    """Keep missing episodes in the denominator without inventing final scores."""
    if not tasks:
        raise ValueError("Cannot summarize an empty selection")
    rows = []
    for task in tasks:
        result = results.get(task["id"])
        if result is None:
            rows.append({"id": task["id"], "status": "missing"})
            continue
        if str(result.get("episode_id")) != task["id"]:
            raise ValueError("Result episode ID mismatch")
        if result.get("status") not in ("stopped", "truncated"):
            raise ValueError("Result is not terminal: " + task["id"])
        for key in ("success", "spl"):
            score = result.get(key)
            if (
                isinstance(score, bool)
                or not isinstance(score, (float, int))
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError("Invalid " + key + " for " + task["id"])
        if result["status"] != "stopped" and (result["success"] or result["spl"]):
            raise ValueError("Truncated episode cannot have success or SPL")
        if result["success"] not in (0, 1):
            raise ValueError("Success must be binary")
        if result["spl"] > result["success"]:
            raise ValueError("SPL exceeds success")
        metrics = result.get("metrics") or {}
        extras = {
            "navigation_error_m": metrics.get("distance_to_goal"),
            "oracle_success_rate": metrics.get("oracle_success"),
            "sdk_usd": result.get("sdk_usd"),
        }
        trajectory = metrics.get("trajectory") or {}
        for key in ("dtw", "ndtw", "sdtw"):
            if trajectory.get(key) is not None:
                extras[key] = trajectory[key]
        for key, value in extras.items():
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Invalid " + key + " for " + task["id"])
        for key in ("ndtw", "sdtw"):
            if extras.get(key) is not None and extras[key] > 1:
                raise ValueError("Invalid " + key + " for " + task["id"])
        if extras["oracle_success_rate"] is not None and extras[
            "oracle_success_rate"
        ] not in (0, 1):
            raise ValueError("Oracle success must be binary")
        rows.append(
            {
                **extras,
                "id": task["id"],
                "status": result["status"],
                "success": result["success"],
                "spl": result["spl"],
            }
        )
    completed = sum(row["status"] != "missing" for row in rows)
    complete = completed == len(tasks)
    aggregates = {}
    for key in (
        "navigation_error_m",
        "oracle_success_rate",
        "sdk_usd",
        "dtw",
        "ndtw",
        "sdtw",
    ):
        values = [row.get(key) for row in rows]
        known = complete and all(value is not None for value in values)
        aggregates[key] = (
            (sum(values) if key == "sdk_usd" else sum(values) / len(values))
            if known
            else None
        )
    return {
        **aggregates,
        "expected": len(tasks),
        "completed": completed,
        "missing": len(tasks) - completed,
        "complete": complete,
        "successes": sum(row.get("success", 0) for row in rows),
        "sr": sum(row["success"] for row in rows) / len(rows) if complete else None,
        "spl": sum(row["spl"] for row in rows) / len(rows) if complete else None,
        "rows": rows,
    }
