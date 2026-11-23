#!/usr/bin/env python3
"""Evaluate registered navigation episodes in Habitat."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from eval.evaluation import (
    episode_lock,
    load_tasks,
    sha,
    summarize,
    validate_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
Env = get_config = habitat = None


def require_habitat():
    global Env, get_config, habitat
    if Env is not None:
        return
    source = os.environ.get("HABITAT_LAB_PATH")
    if source:
        for folder in ("habitat-lab", "habitat-baselines"):
            sys.path.insert(0, str(Path(source).expanduser() / folder))
    try:
        import habitat as installed
        from habitat import Env as installed_env
        from habitat_baselines.config.default import get_config as installed_config
        from eval import measures  # noqa: F401
    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            "Install compatible Habitat-Sim, Habitat-Lab and Habitat-Baselines"
        ) from exc
    habitat, Env, get_config = installed, installed_env, installed_config


def jsonable(value):
    from eval.storage import jsonable as convert

    return convert(value)


def parser():
    result = argparse.ArgumentParser(prog="python -m eval", description=__doc__)
    result.add_argument("--registration", type=Path, required=True)
    result.add_argument(
        "--episode", default="all", help="Episode ID, comma-separated IDs, or all"
    )
    result.add_argument(
        "--data-path", required=True, help="Habitat episode JSON(.gz); accepts {split}"
    )
    result.add_argument("--scenes-dir", type=Path, required=True)
    result.add_argument("--provider", choices=("codex",), default="codex")
    result.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    result.add_argument("--gpu", type=int, default=0)
    result.add_argument("--workers", type=int, default=1)
    result.add_argument(
        '--route-review-policy',
        choices=('periodic', 'active', 'stage_stall'),
        default='periodic',
        help='Route review trigger policy; periodic reviews every 10 completed decisions',
    )
    result.add_argument(
        "--decisions",
        type=int,
        default=0,
        help="Total decision limit; 0 uses the primitive limit",
    )
    result.add_argument(
        "--chunk-end", type=int, help="Pause before this cumulative decision"
    )
    result.add_argument(
        "--action-horizon", choices=("free", "3_to_2", "4_to_3"), default="4_to_3"
    )
    result.add_argument("--preflight", action="store_true")
    result.add_argument(
        "--summarize",
        action="store_true",
        help="Only validate and summarize saved results",
    )
    return result


def configuration(args, split, dataset_kind="rxr"):
    return dict(
        mode="navigation_" + args.action_horizon,
        split=split,
        dataset_kind=dataset_kind,
        route_review_policy=args.route_review_policy,
        fork_decision=0,
        fork_frame=0,
        no_semantic_map=True,
        pose_without_map=True,
        map_enabled=False,
        provider=args.provider,
        execution_feedback_candidate=False,
        depth_preflight_candidate=True,
        adaptive_batching=False,
        horizon_mode=args.action_horizon,
        data_path=str(Path(args.data_path.format(split=split)).expanduser().resolve()),
        scenes_dir=str(args.scenes_dir.expanduser().resolve()),
        episode_budget={"decisions": args.decisions or 2**31 - 1, "primitives": 400},
    )


def output_dir(args, task):
    return (
        args.output_root.resolve()
        / f'{args.provider}_{args.action_horizon}_{task["id"]}_d{args.decisions}'
    )


def identity(config, task):
    return {
        "config": config,
        "task": task,
        "version": {"id": "navigation", "source": None, "source_sha256": "empty"},
    }


def read_result(args, task, config):
    folder = output_dir(args, task)
    record = folder / "registration.json"
    result = folder / "run/result.json"
    if record.exists() and json.loads(record.read_text()) != identity(config, task):
        raise ValueError("Saved run configuration differs: " + task["id"])
    if not result.exists():
        return None
    if not record.exists():
        raise ValueError("Result lacks a run registration: " + task["id"])
    value = json.loads(result.read_text())
    summarize([task], {task["id"]: value})
    return value


def evaluate_one(args, config, task):
    from eval.storage import write
    from eval.storage import CampaignHold, immutable
    from eval.navigation import RGBBackend

    directory = output_dir(args, task)
    with episode_lock(directory):
        if read_result(args, task, config) is not None:
            return 0
        registration = identity(config, task)
        immutable(directory / "registration.json", registration)
        run = directory / "run"
        run.mkdir(exist_ok=True)
        backend = RGBBackend(directory, config, args.gpu)
        backend.chunk_end = args.chunk_end or args.decisions or 400
        try:
            write(
                directory / "state.json", {"status": "running", "episode": task["id"]}
            )
            value = backend.evaluate(registration["version"], task, run)
            summarize([task], {task["id"]: value})
            immutable(run / "result.json", value)
            write(
                directory / "state.json", {"status": "complete", "episode": task["id"]}
            )
            return 0
        except CampaignHold as exc:
            write(
                directory / "state.json",
                {"status": "resumable_hold", "episode": task["id"], "reason": str(exc)},
            )
            return 2
        except Exception as exc:
            write(
                directory / "state.json",
                {
                    "status": "error",
                    "episode": task["id"],
                    "reason": type(exc).__name__ + ": " + str(exc),
                },
            )
            raise
        finally:
            backend.close()


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.decisions < 0 or args.workers < 1 or args.gpu < 0:
        cli.error("decisions and gpu must be nonnegative; workers must be positive")
    if args.chunk_end is not None and (
        args.chunk_end < 1 or (args.decisions and args.chunk_end > args.decisions)
    ):
        cli.error("chunk-end must be positive and within the decision limit")
    split, tasks = load_tasks(args.registration)
    dataset_kind = "rxr" if any(task.get("dataset") == "rxr" for task in tasks) else "r2r"
    if args.episode != "all":
        requested = args.episode.split(",")
        lookup = {task["id"]: task for task in tasks}
        if len(set(requested)) != len(requested) or not set(requested) <= lookup.keys():
            cli.error("episode selection contains duplicates or unregistered IDs")
        tasks = [lookup[key] for key in requested]
    config = configuration(args, split, dataset_kind)
    data = Path(config["data_path"])
    if not data.is_file() or not Path(config["scenes_dir"]).is_dir():
        cli.error("data-path must be a file and scenes-dir must be a directory")
    validate_dataset(tasks, data)
    config["dataset_sha256"] = sha(data)
    config["runtime_hashes"] = {
        str(path.relative_to(ROOT)): sha(path)
        for path in sorted((ROOT / "eval").rglob("*.py"))
    }
    config["runtime_hashes"].update(
        {
            "eval/cli.py": sha(ROOT / "eval/cli.py"),
            "config/habitat.yaml": sha(ROOT / "config/habitat.yaml"),
        }
    )
    config["runtime_hashes"].update(
        {
            str(path.relative_to(ROOT)): sha(path)
            for path in sorted((ROOT / "config").glob("*"))
            if path.is_file()
        }
    )
    if args.preflight:
        from eval.sensors import sensor_rig

        print(
            json.dumps(
                {
                    "episodes": len(tasks),
                    "split": split,
                    "primitive_limit": 400,
                    "forward_step_m": 0.25,
                    "turn_step_deg": 15,
                    "rgb_hfov": 90,
                    "history_panoramas": 8,
                    "sensor_rig": sensor_rig(),
                    "review_enabled": True,
                    "node_return_enabled": True,
                    "forced_review_backtrack": False,
                    "action_horizon": args.action_horizon,
                    "route_review_policy": args.route_review_policy,
                    "map_input": False,
                    "own_pose_input": True,
                    "dataset_sha256": config["dataset_sha256"],
                    "dataset_kind": dataset_kind,
                    "reference_metrics": ["dtw", "ndtw", "sdtw"],
                },
                indent=2,
            )
        )
        return 0
    for task in tasks:
        read_result(args, task, config)
    if not args.summarize:
        if len(tasks) == 1:
            evaluate_one(args, config, tasks[0])
        else:

            def launch(task):
                if read_result(args, task, config) is not None:
                    return 0
                command = [
                    sys.executable,
                    "-m",
                    "eval",
                    "--registration",
                    str(args.registration.resolve()),
                    "--episode",
                    task["id"],
                    "--data-path",
                    config["data_path"],
                    "--scenes-dir",
                    config["scenes_dir"],
                    "--provider",
                    args.provider,
                    "--output-root",
                    str(args.output_root.resolve()),
                    "--gpu",
                    str(args.gpu),
                    "--decisions",
                    str(args.decisions),
                    "--action-horizon",
                    args.action_horizon,
                    '--route-review-policy',
                    args.route_review_policy,
                ]
                if args.chunk_end:
                    command += ["--chunk-end", str(args.chunk_end)]
                folder = output_dir(args, task)
                folder.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                existing_path = env.get("PYTHONPATH")
                env["PYTHONPATH"] = (
                    str(ROOT)
                    if not existing_path
                    else str(ROOT) + os.pathsep + existing_path
                )
                with (folder / "worker.log").open("a") as log:
                    return subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, env=env
                    ).returncode

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(launch, tasks))
    results = {task["id"]: read_result(args, task, config) for task in tasks}
    summary = summarize(tasks, results)
    from eval.storage import write

    selection = ",".join(task["id"] for task in tasks).encode()
    import hashlib

    label = hashlib.sha256(selection).hexdigest()[:12]
    summary_path = args.output_root / f"summary_{label}.json"
    write(summary_path, summary)
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "rows"}, indent=2
        )
    )
    print("Summary:", summary_path)
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
