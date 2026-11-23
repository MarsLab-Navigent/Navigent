import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from eval.memory import uniform_history_indices


def history_images(payload, limit=8):
    catalog = payload.get("memory_catalog") or {}
    archive = catalog.get("archive") or {}
    root = Path(catalog.get("root", "/nonexistent")).resolve()
    ordered = sorted(
        ((int(frame), item) for frame, item in archive.items()),
        key=lambda row: (row[1]["decision"], row[0]),
    )
    selected = []
    for index in uniform_history_indices(len(ordered), limit):
        frame, item = ordered[index]
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Historical image outside the registered archive")
        selected.append((frame, item, base64.b64encode(path.read_bytes()).decode()))
    return selected


def without_semantic_map(context):
    context = json.loads(json.dumps(context))
    observed = context.get("observed_semantic_map")
    if isinstance(observed, dict):
        observed.pop("room_stage_hypotheses", None)
        observed.pop("unnamed_node_notice", None)
        observed.pop("unnamed_places", None)
        for node in observed.get("nodes", []):
            for key in ("name", "caption", "annotation_anchor", "annotations"):
                node.pop(key, None)
    for key in ("keyframe_index", "uniform_history_index"):
        for entry in context.get(key, []):
            if isinstance(entry, dict):
                entry.pop("description_hypothesis", None)
                entry.pop("description", None)
    return context


def recent_execution_evidence(payload):
    root = Path((payload.get("memory_catalog") or {}).get("root", "/nonexistent"))
    path = root / "trajectory.partial.json"
    if not path.is_file():
        return None
    rows = json.loads(path.read_text())[-6:]
    attempts = []
    for row in rows:
        output = row.get("output") or {}
        actual = (row.get("execution") or {}).get("actual") or []
        forward = [item for item in actual if item.get("action") == 1]
        if not forward:
            continue
        attempts.append(
            {
                "decision": row["decision"],
                "maneuver": output.get("maneuver"),
                "actual_forward_m": round(
                    sum(
                        float(item.get("measured_translation_m") or 0)
                        for item in forward
                    ),
                    3,
                ),
                "height_change_m": round(
                    sum(float(item.get("gt_height_delta_m") or 0) for item in forward),
                    3,
                ),
            }
        )
    return attempts


def action_pacing(context):
    return (
        "Choose each forward distance freely from 25/50/75/100 cm and each "
        "turn from 15 to 90 degrees in 15-degree increments. Combine "
        "continuous same-heading motion into the fewest valid commands. "
        "Measured progress is checked after each execution primitive."
    )


def navigation_prompt(base_prompt, no_semantic_map):
    if not no_semantic_map:
        return base_prompt
    return re.sub(
        r"ROOM STAGE MEMORY:.*?(?=STABLE TASK-ENDPOINT ANCHOR:)",
        "",
        base_prompt,
        flags=re.DOTALL,
    )


def pose_only_prompt(base_prompt):
    prompt = re.sub(
        r"MEMORY CONTRACT:.*?(?=ROOM STAGE MEMORY:|STABLE TASK-ENDPOINT ANCHOR:|HORIZON OVERRIDE:|\Z)", "", base_prompt, flags=re.DOTALL
    )
    prompt = re.sub(
        r"MAP SENSOR OVERRIDE:.*?(?=GT LOCALIZATION OVERRIDE:)",
        "",
        prompt,
        flags=re.DOTALL,
    )
    prompt = prompt.replace(
        "Own pose and RGB-D observed mapping are available;",
        "Measured own pose is available;",
    )
    return prompt + (
        "\nPOSE AND VISITED NODES: The current input gives your measured "
        "own_pose relative to start and a list of visited node IDs with "
        "recorded connections and relative return bearings. There is no "
        "map image or map geometry in your input. Use the current four-view "
        "RGB and historical photos to judge route identity. You may request "
        "navigate_to_node for an existing visited node ID; the executor "
        "follows only recorded movement. Set room_stage_hypothesis to "
        'relation=none, node_id=null, room="", instruction_stage=null, '
        "evidence_frame=null; do not annotate nodes.\n"
        "POSE HISTORY is a short chronological list of visited own poses, "
        "not a floor plan or a route to the hidden goal. Compare its "
        "heading and height with the instruction and the actual images. "
        "DEPTH PREFLIGHT samples a narrow forward corridor for roughly "
        "two 25 cm steps. possible_obstacle asks for a new visual check "
        "or shorter approach; unknown is not clear. The horizontal "
        "depth view cannot certify low steps, glass, or room identity. "
        "A warning around two steps ahead is advisory; only an "
        "imminent obstacle may interrupt a forward primitive. "
        "you still choose every subsequent movement and STOP.\n"
    )


def cli_command(payload, image_paths, output_path, model="gpt-6-astra"):
    codex = os.environ.get("NAV_CODEX_BIN") or shutil.which("codex")
    if not codex:
        raise FileNotFoundError("Codex CLI not found; set NAV_CODEX_BIN")
    session = payload.get("session_id")
    if session:
        command = [
            codex,
            "exec",
            "resume",
            "--model",
            model,
            "--skip-git-repo-check",
            "--ignore-rules",
            "--json",
            "--output-last-message",
            str(output_path),
        ]
        for path in image_paths:
            command.extend(["--image", str(path)])
        command.extend([session, "-"])
    else:
        command = [
            codex,
            "exec",
            "--model",
            model,
            "--sandbox",
            "read-only",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--json",
            "--output-last-message",
            str(output_path),
        ]
        for path in image_paths:
            command.extend(["--image", str(path)])
        command.append("-")
    return command


def navigate_cli(payload, model="gpt-6-astra", timeout=600):
    try:
        from eval.prompts import PROMPT, NAV_SCHEMA

        selected = history_images(payload)
        no_semantic_map = payload.get("no_semantic_map", False)
        context = (
            without_semantic_map(payload["context"])
            if no_semantic_map
            else payload["context"]
        )
        base_prompt = navigation_prompt(PROMPT, no_semantic_map)
        if payload.get("pose_without_map"):
            base_prompt = pose_only_prompt(base_prompt)
        base_prompt += (
            "\nLANDMARK IDENTITY CHECK: Before committing to an opening or landmark "
            "that advances an instruction clause, compare the selected candidate "
            "with the strongest visible alternative in the current or attached "
            "historical panoramas. In review_response.current_test, name both by "
            "visible appearance and frame/view, or state that no alternative is "
            "visible. State the distinguishing instruction relation: approach "
            "direction, order, landmark association, or turn at the junction. "
            "A matching object category or an enlarging target supports recognition "
            "or approach, but does not distinguish route identity. A small steering "
            "correction does not by itself establish an instructed junction turn. "
            "If both candidates remain plausible, keep identity unresolved and "
            "choose a movement whose next view can distinguish them, using the "
            "existing action horizon. Do not invent an alternative or reject a "
            "candidate solely because another looks more typical. Once evidence "
            "distinguishes the candidate, proceed without redundant comparison. "
            "Record this check in existing fields; do not add output fields.\n"
        )
        memory_prompt = (
            "Up to eight historical four-view panoramas are attached automatically, "
            "uniformly sampled from the full chronological decision history. "
            "Historical images retain their original heading. No memory tools are available. "
        )
        prompt = (
            base_prompt
            + "\nOnly use attached images and public navigation context. "
            + memory_prompt
            + "Set history_panorama_ids=[]. Return one JSON object, with no Markdown. "
            "review_response.current_test must be a nonempty sentence even when "
            "review_response.resolution is not_applicable; state the current visual check. "
            "If context.action_contract_error exists, correct that rejected output. "
            "It must satisfy this JSON schema; the runtime rejects invalid output:\n"
            + json.dumps(NAV_SCHEMA, separators=(",", ":"))
            + "\n\n"
            + json.dumps(context, separators=(",", ":"))
        )
        if no_semantic_map and not payload.get("pose_without_map"):
            prompt += (
                "\nThe supplied map contains geometry and visited-node connectivity only. "
                "Do not infer room identity from node IDs. Return room_stage_hypothesis "
                'with relation=none, node_id=null, room="", instruction_stage=null, '
                "evidence_frame=null. Do not annotate nodes: map_operation must be "
                "none or navigate_to_node. Visited-node return remains available."
            )
        if payload.get("execution_feedback_candidate"):
            attempts = recent_execution_evidence(payload)
            if attempts:
                prompt += (
                    "\nEXECUTION EVIDENCE (measured, not a route verdict): "
                    + json.dumps(attempts, separators=(",", ":"))
                )
                prompt += (
                    "\nIf repeated stair attempts near the same place made no net height "
                    "gain, change the approach position before another climb. "
                    "A brief rise followed by equal descent is not stair progress. "
                    "Near a suspected final doorway, use one short observation-producing "
                    "move and reassess the doorway relationship. Do not claim arrival "
                    "from a blocked action. You choose the action and STOP."
                )
        if payload.get("merge_forward_commands"):
            prompt += (
                "\nACTION FORMAT: Choose freely from one to three movement commands. "
                "Each forward command is 25, 50, 75, or 100 cm; each turn is "
                "15 to 90 degrees in 15 degree increments. Consecutive "
                "same-heading forward commands whose total is at most 100 cm "
                "represent one continuous movement: express them as one "
                "forward command with that total distance. If the total "
                "exceeds 100 cm, use the fewest valid forward commands. "
                "Choose the distance yourself from the available observation; "
                "there is no minimum distance or scene-specific distance rule. "
                "STOP is separate and has actions=[]."
            )
        pacing = action_pacing(context) if payload.get("adaptive_batching") else None
        if pacing:
            prompt += "\n" + pacing
        images = [
            (frame["label"], frame["data"]) for frame in payload.get("frames", [])
        ]
        for frame_id, item, data in selected:
            images.append(
                (
                    f"HISTORICAL frame {frame_id}: "
                    + (
                        ""
                        if no_semantic_map
                        else item.get("description", "unverified place hypothesis")
                    ),
                    data,
                )
            )
        with tempfile.TemporaryDirectory(prefix="vln_codex_") as directory:
            directory = Path(directory)
            output = directory / "output.json"
            paths = []
            for index, (label, data) in enumerate(images):
                path = directory / f"image_{index:02d}.jpg"
                path.write_bytes(base64.b64decode(data, validate=True))
                paths.append(path)
                prompt += f"\nAttached image {index}: {label}"
            command = cli_command(payload, paths, output, model)
            env = os.environ.copy()
            env.setdefault("CODEX_HOME", str(Path.home() / ".codex"))
            process = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=timeout,
                cwd=directory,
                env=env,
            )
            events = []
            for line in process.stdout.splitlines():
                try:
                    event = json.loads(line)
                    if isinstance(event, dict):
                        events.append(event)
                except ValueError:
                    continue
            raw = output.read_text() if output.exists() else ''
            diagnostics = dict(stdout=process.stdout, stderr=process.stderr, raw_output=raw)
            if process.returncode != 0 or not output.exists():
                detail = next(
                    (
                        event.get("message", "")
                        for event in reversed(events)
                        if event.get("type") == "error"
                    ),
                    "",
                )
                return {
                    "error": f"Codex CLI exit {process.returncode}: "
                    f"{detail or process.stderr[-500:]}",
                    "usage_unknown": True,
                    **diagnostics,
                }
            answer = json.loads(raw)
            if not isinstance(answer, dict):
                raise ValueError("Codex CLI did not return a navigation object")
            thread = next(
                (
                    event.get("thread_id")
                    for event in events
                    if event.get("type") == "thread.started"
                ),
                None,
            )
            starts = [e.get('thread_id') for e in events if e.get('type') == 'thread.started']
            session_id = thread
            if not isinstance(thread, str) or not thread.strip() or starts != [thread]:
                raise ValueError('Expected one thread.started with a resumable thread ID')
            if payload.get('session_id') and payload['session_id'] != thread:
                raise ValueError('Resume returned a different thread ID')
            usage = (
                next(
                    (
                        event.get("usage")
                        for event in reversed(events)
                        if event.get("type") == "turn.completed"
                    ),
                    None,
                )
                or {}
            )
            frames = [frame_id for frame_id, _, _ in selected]
            return {
                "output": answer,
                "session_id": session_id,
                "model": model,
                "usage": usage,
                "cost_usd": None,
                "retrieved_frame_ids": frames,
                "action_pacing_prompt": pacing,
                "history_strategy": "uniform",
                "history_limit": 8,
                "memory_calls": (
                    [{"tool": "automatic_history", "frames": frames}] if frames else []
                ),
            }
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        label = 'TIMEOUT' if isinstance(exc, subprocess.TimeoutExpired) else 'CODEX_ERROR'
        print(f'[{label}] detail={exc}', file=sys.stderr, flush=True)
        return {"error": f"{type(exc).__name__}: {exc}", "usage_unknown": True,
                'retryable_format': isinstance(exc, json.JSONDecodeError) and 'diagnostics' in locals(),
                **locals().get('diagnostics', {})}


import copy
import json
import sys

from eval.contracts import HORIZONS, crop_horizon
import eval.prompts as contract


def run(payload):
    mode = payload["horizon_mode"]
    if mode == "free":
        return navigate_cli(dict(payload, merge_forward_commands=True))
    predicted, executed = HORIZONS[mode]
    original_prompt = contract.PROMPT
    original_schema = contract.NAV_SCHEMA
    try:
        contract.NAV_SCHEMA = copy.deepcopy(original_schema)
        contract.NAV_SCHEMA["properties"]["actions"]["maxItems"] = predicted
        contract.PROMPT = original_prompt.replace(
            "Output skill=move or stop; actions contains 1 to 3 ordered commands for move, empty for STOP.",
            "Output skill=move or stop; MOVE predicts an ordered command horizon, STOP has actions=[].",
        ).replace(
            "Maximum THREE commands per decision, not three primitives.",
            "The horizon override below sets the predicted and executed command counts.",
        ) + (
            f"\nHORIZON OVERRIDE: MOVE predicts exactly {predicted} ordered high-level "
            f"commands; only the first {executed} will execute, then the robot "
            "reobserves. Put at least one forward command in that executed prefix. "
            "STOP is separate: skill=stop and actions=[]. On stairs use supported "
            "forward commands from the same 25/50/75/100 cm set as elsewhere. The "
            "executed prefix "
            "may request at most 150 cm total forward travel. The execution engine "
            "checks measured progress after every primitive and may interrupt early. "
            "This override supersedes the older three-command "
            "limits in the preceding prompt. EXCEPTION: navigate_to_node uses actions=[]; "
            "the recorded-path executor supplies its primitives, not the predicted horizon."
        )
        attempts = []
        current = payload
        for _ in range(2):
            response = navigate_cli(current)
            if response.get("error"):
                return dict(response, horizon_attempts=attempts)
            raw = response["output"]
            try:
                cropped, audit = crop_horizon(raw, mode)
            except ValueError as exc:
                attempts.append({"error": str(exc), "proposed_output": raw})
                current = dict(
                    payload,
                    session_id=response["session_id"],
                    context=dict(
                        payload["context"],
                        action_contract_error={
                            "error": str(exc),
                            "rejected_output": raw,
                        },
                    ),
                )
                continue
            response["output"] = cropped
            response["horizon"] = audit
            response["horizon_attempts"] = attempts
            return response
        return {
            "error": "Invalid predicted horizon after one repair",
            "horizon_attempts": attempts,
            "usage_unknown": True,
        }
    finally:
        contract.PROMPT = original_prompt
        contract.NAV_SCHEMA = original_schema


if __name__ == "__main__":
    print(json.dumps(run(json.load(sys.stdin))))
