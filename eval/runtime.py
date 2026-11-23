import base64
import hashlib
import json
from pathlib import Path
from eval.storage import digest

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LiveBackend:
    def __init__(self, root, config, gpu):
        self.root = Path(root)
        self.config = config
        self.gpu = gpu
        self.env = None
        self.episodes = {}

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def run_identity(self, version, task):
        return digest(
            {
                "task": task,
                "source": version["source_sha256"],
                "code": self.config["runtime_hashes"],
                "budget": self.config["episode_budget"],
                "mode": self.config["mode"],
            }
        )

    def runtime_unchanged(self):
        return all(
            sha(ROOT / name) == expected
            for name, expected in self.config["runtime_hashes"].items()
        )

    def role_request(self, role, body, directory, images=None):
        content = [{"type": "text", "text": json.dumps(body)}]
        for item in images or []:
            content.extend(
                [
                    {"type": "text", "text": item["label"]},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.b64encode(
                                Path(item["source"]).read_bytes()
                            ).decode(),
                        },
                    },
                ]
            )
        return self.request(
            "eval/model_reviewer.py",
            {"role": role, "content": content},
            directory,
            600,
        )["output"]


import base64, copy, json, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from eval.storage import immutable, read, CampaignHold
from eval.sensors import create_rgb_env, views, integrate, EvidenceLedger
from eval.geometry import RGBPose
from eval.memory import context as compact_context
from eval.contracts import validate_action, primitives
from eval.critics import NavigationCodeCritic
from eval.memory import RouteReviewMemory, merge_images, active_review_reason, stage_stall_review_reason
from eval.storage import digest
from eval.storage import write
from eval.geometry import OwnPoseMap
from eval.geometry import forward_clearance
from eval.visited_nodes import VisitedNodes
from eval.critics import SemanticEvidenceAudit
from eval.critics import ExecutionContractCritic
from eval.geometry import TargetAnchorCritic
from eval.entrances import EntranceGraph
from eval.contracts import instruction_constraints
from eval.contracts import validate_compound_plan
from eval.entrances import route_progress_feedback
from eval.critics import PoseStallCritic
from eval.critics import GoalDistanceCheckpoint
from eval.contracts import (
    pending_route_review,
    review_remains_open,
    validate_review_resolution,
)
from eval.geometry import return_arrival
from eval.memory import RoomStageLedger
from eval.memory import node_frame_index
from eval.contracts import validate_stop_clauses
from eval.memory import uniform_history_indices
from eval.memory import navigation_context
from eval.evaluation import trajectory_scores


class RGBBackend(LiveBackend):
    def create_semantic_auditor(self, instruction):
        return SemanticEvidenceAudit(instruction)

    def run_identity(self, version, task):
        return digest(
            {
                "base": super().run_identity(version, task),
                "route_review_policy": self.config.get('route_review_policy', 'periodic'),
                "localization": "GT own pose and executed cumulative map",
                "runtime": "observed_geometry_navigation",
            }
        )

    def ensure_env(self):
        if self.env is None:
            self.env = create_rgb_env(
                self.config.get("split", "train"),
                self.gpu,
                map_depth=True,
                data_path=self.config["data_path"],
                scenes_dir=self.config["scenes_dir"],
                max_steps=self.config["episode_budget"]["primitives"] + 1,
                dataset_kind=self.config.get("dataset_kind", "rxr"),
            )
            self.episodes = {str(e.episode_id): e for e in self.env.episodes}

    def request(self, module, payload, directory, timeout=600):
        if payload.get("role") in ("online", "route_check"):
            raise ValueError("Only scheduled periodic_route_review is enabled")
        if payload.get("role") == "periodic_route_review":
            module = "eval/model_reviewer.py"

        return super().request(module, payload, directory, timeout)

    def role_request(self, role, body, directory, images=None):
        if role in ("online", "route_check"):
            raise ValueError("Use scheduled periodic_route_review")
        body = json.loads(json.dumps(body, sort_keys=True))
        if role in ("actor", "verifier"):
            body = dict(
                body,
                public_observation={
                    "keys": [
                        "instruction",
                        "frame_id",
                        "decision_id",
                        "cause",
                        "history",
                        "events",
                        "last_execution",
                        "sensor_map",
                        "pose_evidence",
                        "online_route_verification",
                        "recovery_event",
                        "code_critic",
                    ],
                    "events": "Actor hypotheses with frames, not independent confirmations.",
                    "code_critic": "Pure code rule proposals; no visual model judgment. Unknown goal/route semantics.",
                    "online_route_verification": "Compatibility wrapper with unknown route/completion; no model assessment.",
                    "last_execution": "Actual primitive rows; unknown metric translation. Do not count motion intervals as stair steps.",
                    "authority": "Every navigation action/STOP chosen by navigation model, no veto. Uniform8 plus explicit key panoramas and periodic fallible route review.",
                },
            )
        return super().role_request(role, body, directory, images)

    def evaluate(self, version, task, directory):
        from eval.cli import jsonable

        self.ensure_env()
        directory = Path(directory)
        immutable(
            directory / "manifest.json",
            {
                "identity": self.run_identity(version, task),
                "task": task,
                "version": version,
                "runtime_hashes": self.config["runtime_hashes"],
                "budget": self.config["episode_budget"],
                "critic": "Local Python rules every primitive; periodic visual route review",
                "route_review_policy": self.config.get('route_review_policy', 'active'),
                "localization": "GT own pose and executed cumulative map",
                "input": "current labeled four-view panorama; historical four-view panoramas through model-chosen tools",
            },
        )
        episode = self.episodes[task["id"]]
        if episode.instruction.instruction_text != task["instruction"]:
            raise ValueError("Instruction differs from dataset")
        if (
            Path(episode.scene_id).resolve()
            != (Path(self.config["scenes_dir"]) / task["scene"]).resolve()
        ):
            raise ValueError("Scene differs from dataset")
        self.env.episodes = [episode]
        obs = self.env.reset()
        if not any(k.endswith("map_depth") for k in obs):
            raise ValueError("Map depth sensors missing")
        gtmap = OwnPoseMap(self.env.sim)
        map_state = VisitedNodes(self.env.sim)
        motioncritic = ExecutionContractCritic()
        targetcritic = TargetAnchorCritic()
        task_goal_critic = TargetAnchorCritic()
        stallcritic = PoseStallCritic()
        goal_checkpoint = GoalDistanceCheckpoint(
            interval=10, regression_tolerance_m=0.5
        )
        entrances = EntranceGraph()
        future_entrances = entrances
        ledger = EvidenceLedger(task['instruction'])
        pose = RGBPose()
        critic = NavigationCodeCritic(task["instruction"], ledger.clauses)
        semantic = self.create_semantic_auditor(task["instruction"])
        memory = RouteReviewMemory()
        review_policy = self.config.get('route_review_policy', 'periodic')
        last_review_attempt = None
        review_calls = 0
        room_stage = RoomStageLedger()
        history = []
        trace = []
        steps = 0
        last = {}
        branch_anchor = None
        branch_target = ""
        return_decisions = 0
        key_events = []
        recall_event = False
        recall_frame = None
        last_node_count = 0
        candidate_conflicts = []
        stall_alert = None
        goal_checkpoint_stop = None
        last_review_resolution = None
        review_inspection_counts = {}
        review_inspections = {}
        pending_node_return = None
        return_iterator = None
        backtracking_attempts = []
        active_return = None
        excluded_frames = set()
        rejected_branch = None
        review_failure = None
        status = "truncated"
        tokens = 0
        usd = 0.0
        cost_missing = False
        accounted = set()
        started = time.monotonic()
        previous = None
        session_id = None

        def account(dp):
            nonlocal tokens, usd, cost_missing
            for f in Path(dp).glob("response_*.json"):
                if str(f) in accounted:
                    continue
                accounted.add(str(f))
                try:
                    d = read(f)
                    if not isinstance(d, dict):
                        raise ValueError('Invalid usage record')
                except ValueError:
                    cost_missing = True
                    continue
                u = d.get("usage") or {}
                tokens += sum(
                    u.get(k, 0) or 0
                    for k in (
                        "input_tokens",
                        "output_tokens",
                        "cache_read_input_tokens",
                        "cache_creation_input_tokens",
                    )
                )
                if d.get("cost_usd") is None:
                    cost_missing = True
                else:
                    usd += float(d["cost_usd"])

        def save_image(path, rgb, label, frame):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                error = float(
                    np.mean(
                        np.abs(
                            rgb.astype(float)
                            - np.asarray(Image.open(path).convert("RGB")).astype(float)
                        )
                    )
                )
                if error > 12:
                    raise CampaignHold("Replay image mismatch")
            else:
                Image.fromarray(rgb).save(path, "JPEG", quality=85)
            return {
                "source": str(path),
                "label": label,
                "frame": frame,
                "sha256": sha(path),
            }

        def compatibility(decision):
            return {
                "verdict": {
                    "motion": "unknown",
                    "route": "unknown",
                    "completion": "unknown",
                    "evidence": "Local code proposals only; no model semantic judge",
                    "next_check": "Use attached code_critic and real RGB.",
                },
                "checked_decision": decision,
                "age_in_decisions": 0,
                "reason": "local_python_rules",
            }

        for decision in range(self.config["episode_budget"]["decisions"]):
            if (
                steps >= self.config["episode_budget"]["primitives"]
                or self.env.episode_over
            ):
                break
            if decision >= getattr(
                self, "chunk_end", self.config["episode_budget"]["decisions"]
            ):
                raise CampaignHold(
                    f"Decision chunk complete at {decision}; resume this run for the next chunk"
                )
            if (
                steps >= self.config["episode_budget"]["primitives"]
                or self.env.episode_over
            ):
                break
            if decision == self.config.get("fork_decision", 0):
                entrances = future_entrances
            frame = steps
            pano = views(obs)[0][1]
            if history:
                old_heading = history[-1]["heading"]
                turn = (pose.commanded_yaw - old_heading + 180) % 360 - 180
                if abs(turn) >= 45 and not any(
                    e["frame"] == history[-1]["id"] and e["kind"] == "direction_change"
                    for e in key_events
                ):
                    key_events.append(
                        {
                            "frame": history[-1]["id"],
                            "kind": "direction_change",
                            "description": f"{round(turn)} degree heading change",
                        }
                    )
            current = save_image(
                directory
                / "images"
                / f"{decision:03d}_{frame:04d}_PANORAMA_FRONT_RIGHT_REAR_LEFT.jpg",
                pano,
                f"CURRENT frame {frame} PANORAMA_FRONT_RIGHT_REAR_LEFT",
                frame,
            )
            map_state.record_pose()
            frame_path_index = len(map_state.path) - 1
            if len(map_state.nodes) > last_node_count and history:
                key_events.append(
                    {
                        "frame": history[-1]["id"],
                        "kind": "visited_node",
                        "description": history[-1]["place"],
                    }
                )
            last_node_count = len(map_state.nodes)
            map_info = {"enabled": False}
            critic.before_decision(frame, history, obs, branch_anchor)
            code = critic.packet()
            wrapper = compatibility(decision)
            if previous:
                immutable(
                    directory / "transitions" / f'{previous["decision"]:03d}.json',
                    {
                        "decision": previous["decision"],
                        "before": previous["images"],
                        "input_file": previous["input_file"],
                        "instruction": task["instruction"],
                        "output": previous["output"],
                        "execution": last,
                        "after": [current],
                        "expected_change": previous["output"]["note"],
                        "anomaly_reasons": [p["rule_id"] for p in code["proposals"]],
                        "observed_change": "Code audit + raw executed actions retained; no pixel-change threshold alarm",
                    },
                )
            events = ledger.snapshot()
            key_images = []
            active = [r for r in history if r['id'] not in excluded_frames]
            def unresolved_review():
                addressed = (
                    last_review_resolution is not None and memory.latest is not None
                    and last_review_resolution['checked_before_decision'] == memory.latest['checked_before_decision']
                    and not last_review_resolution['requires_followup']
                )
                return pending_route_review(memory.latest, addressed=addressed)

            pending = unresolved_review()
            review_trigger = active_review_reason(
                decision,
                history,
                last,
                key_events,
                candidate_conflicts,
                stall_alert,
                pending_node_return,
                pending,
            )
            if review_policy == 'periodic':
                review_trigger = (
                    'periodic_route_review'
                    if pending is None and pending_node_return is None
                    and memory.due(decision)
                    else None
                )
            elif review_policy == 'stage_stall':
                review_trigger = (
                    stage_stall_review_reason(active, last_review_attempt)
                    if pending is None and pending_node_return is None else None
                )
            if review_trigger and active:
                last_review_attempt = decision
                window_size = 16 if review_policy == 'stage_stall' else 10
                window = [record for record in active if record['decision'] >= decision - window_size]
                recent = (
                    [window[index] for index in uniform_history_indices(len(window), 8)]
                    if review_policy == 'stage_stall' else window
                )
                sheet = Image.new("RGB", (1536, 954), (23, 28, 34))
                draw = ImageDraw.Draw(sheet)
                for index, record in enumerate(recent):
                    tile = (
                        Image.open(record["image"]["source"])
                        .convert("RGB")
                        .resize((384, 288), Image.Resampling.LANCZOS)
                    )
                    x = (index % 4) * 384
                    y = (index // 4) * 318
                    sheet.paste(tile, (x, y + 25))
                    draw.text(
                        (x + 8, y + 5),
                        f"decision {record['decision']} | frame {record['id']}",
                        fill="white",
                    )
                sheet_path = directory / "images" / f"review_{decision:03d}_history.jpg"
                sheet.save(sheet_path, "JPEG", quality=78)
                review_images = [
                    *[dict(r['image'], label=f"REVIEW frame {r['id']} four-view panorama") for r in recent],
                    current,
                ]
                supplied = {r["id"] for r in recent} | {frame}
                review_node_frames = node_frame_index(map_state.nodes, active, gtmap.rows, map_state.path[0])
                body = {
                    "instruction": task["instruction"],
                    "decision_id": decision,
                    "current_frame_id": frame,
                    "review_window": {
                        "start_decision": window[0]['decision'],
                        "end_decision": decision - 1,
                        "start_frame": recent[0]["id"] if recent else frame,
                        "end_frame": recent[-1]["id"] if recent else frame,
                        'sampled_history_decisions': [record['decision'] for record in recent],
                    },
                    "previous_review_end": (memory.latest or {}).get(
                        "checked_before_decision"
                    ),
                    "previous_review_summary": (memory.latest or {}).get(
                        "task_summary", ""
                    ),
                    "available_frame_ids": sorted(supplied),
                    "actual_recent_transitions": [
                        {
                            "frame": r["id"],
                            "stage_hypothesis": r["stage"],
                            "place_hypothesis": r["place"],
                            "expected_effect": r["output"].get("expected_effect"),
                            "execution": {
                                "blocked": r['last_execution'].get('blocked', False),
                                "forward_attempts": sum(
                                    a["action"] == 1
                                    for a in r["last_execution"].get("actual", [])
                                ),
                                "measured_translation_m": round(
                                    sum(
                                        a.get("measured_translation_m", 0)
                                        for a in r["last_execution"].get("actual", [])
                                    ),
                                    2,
                                ),
                                "critic_interrupted": r["last_execution"].get(
                                    "critic_interrupted", False
                                ),
                            },
                        }
                        for r in window
                    ],
                    "previous_task_summary": (memory.latest or {}).get(
                        "task_summary", ""
                    ),
                    "gt_own_pose_map": gtmap.context(branch_anchor),
                    "pose_history": gtmap.history_summary([r['id'] for r in active]),
                    "visited_nodes": map_state.context(),
                    "node_poses": [
                        {'node_id': n['id'], 'xyz_from_start_m':
                         (map_state.return_target(n['id']) - map_state.path[0]).round(3).tolist(),
                         'anchor_frame': review_node_frames.get(str(n['id']))}
                        for n in map_state.nodes
                    ],
                    "return_anchor_frames": sorted(set(review_node_frames.values())),
                    "clause_states": ledger.snapshot(),
                    "provenance": "Own pose is measured; landmark and stage descriptions are fallible. No hidden task state.",
                    "trigger_type": review_trigger,
                }
                rp = directory / "route_reviews" / f"{decision:03d}"
                try:
                    verdict = self.role_request("periodic_route_review", body, rp, review_images)
                    account(rp)
                    memory.accept(verdict, decision, supplied, active)
                    memory.latest['trigger_type'] = review_trigger
                    review_calls += 1
                    review_failure = None
                    immutable(rp / "evidence.json", {"body": body, "images": review_images, "verdict": memory.latest})
                except (CampaignHold, ValueError, KeyError, TypeError) as exc:
                    review_failure = {"decision": decision, "error": f"{type(exc).__name__}: {exc}"}
                    write(rp / "review_unavailable.json", {"body": body, "error": review_failure})
            route_memory = memory.context(decision, history, branch_anchor)
            route_memory["task_summary"] = (memory.latest or {}).get("task_summary", "")
            pending_review = unresolved_review()
            if pending_node_return and pending_review is None:
                pending_review = {'checked_before_decision': decision, 'decision': 'backtrack'}
            if (
                pending_review is not None
                and last_review_resolution
                and last_review_resolution.get("requires_followup")
            ):
                pending_review["followup_after_inspection"] = True
                pending_review["inspection_result_to_check"] = (
                    last_review_resolution.get("current_test")
                )
            if pending_review is not None:
                pending_review["inspection_count"] = review_inspection_counts.get(
                    pending_review["checked_before_decision"], 0
                )
                pending_review["last_inspection"] = review_inspections.get(
                    pending_review["checked_before_decision"]
                )
            if pending_review is not None and pending_node_return is not None:
                pending_review["node_return_in_progress"] = pending_node_return
            if semantic is not None:
                route_memory["semantic_evidence_audit"] = semantic.context()
            if memory.latest:
                wrapper = {
                    "verdict": memory.latest,
                    "checked_decision": memory.latest["checked_before_decision"],
                    "age_in_decisions": decision
                    - memory.latest["checked_before_decision"],
                    "reason": memory.latest['trigger_type'],
                }
            ob = {
                "instruction": task["instruction"],
                "frame_id": frame,
                "decision_id": decision,
                "cause": "navigate",
                "history": [
                    {
                        k: r[k]
                        for k in ("id", "stage", "heading", "place", "note", "events")
                    }
                    for r in history
                ],
                "events": events,
                "last_execution": last,
                "pose_evidence": pose.context(),
                "sensor_map": {"available": False},
                "online_route_verification": wrapper,
                "recovery_event": code["recovery_event"],
                "code_critic": code,
                "route_memory": route_memory,
                "gt_own_pose_map": gtmap.context(branch_anchor),
                "observed_semantic_map": map_state.context(),
                "active_branch": {
                    "return_anchor_frame": branch_anchor,
                    "target": branch_target,
                    "return_decisions": return_decisions,
                },
                "pending_route_review": pending_review,
                "last_review_resolution": last_review_resolution,
            }
            route_feedback = route_progress_feedback(
                previous, last, entrances.context(self.env.sim)
            )
            if decision >= self.config.get("fork_decision", 0):
                constraints = instruction_constraints(task["instruction"])
                ob["compound_route"] = constraints["linked_landmark_turn"]
                ob["instruction_constraints"] = constraints
                ob["predicted_semantic_graph"] = entrances.context(self.env.sim)
                ob["previous_compound_assessment"] = (
                    (previous or {}).get("output", {}).get("compound_route_assessment")
                )
                ob["route_progress_feedback"] = route_feedback

            front_path = (
                directory / "images" / f"{decision:03d}_{frame:04d}_CURRENT_FRONT.jpg"
            )
            Image.fromarray(np.asarray(obs["rgb"])[..., :3]).convert("RGB").save(
                front_path, "JPEG", quality=82
            )
            side = Image.new("RGB", (1152, 310), (24, 28, 33))
            side_draw = ImageDraw.Draw(side)
            for index, (label, key) in enumerate(
                (("RIGHT", "right_rgb"), ("REAR", "rear_rgb"), ("LEFT", "left_rgb"))
            ):
                tile = (
                    Image.fromarray(np.asarray(obs[key])[..., :3])
                    .convert("RGB")
                    .resize((384, 288), Image.Resampling.LANCZOS)
                )
                side.paste(tile, (index * 384, 22))
                side_draw.text((index * 384 + 8, 4), label, fill="white")
            side_path = (
                directory / "images" / f"{decision:03d}_{frame:04d}_CURRENT_270.jpg"
            )
            side.save(side_path, "JPEG", quality=78)
            images = [
                {
                    "source": current["source"],
                    "label": f"CURRENT frame {frame} four simultaneous 90-degree views: FRONT, RIGHT, REAR, LEFT; tile edges are not continuous",
                    "frame": frame,
                }
            ]
            by_id = {r["id"]: r for r in active}
            key_ids = {
                e["frame"]
                for e in key_events
                if e["frame"] in by_id
                and e["kind"]
                in (
                    "visited_node",
                    "direction_change",
                    "target_first_seen",
                    "entrance_candidate",
                )
            }
            key_ids.update(
                r["id"]
                for index, r in enumerate(history)
                if index and r['id'] in by_id and r["stage"] != history[index - 1]["stage"]
            )
            key_ids.update(
                n["anchor"]["frame_id"]
                for n in map_state.nodes
                if n.get("anchor") and n["anchor"].get("frame_id") in by_id
            )
            if recall_frame in by_id:
                key_ids.add(recall_frame)
            uniform_ids = [
                active[i]["id"] for i in uniform_history_indices(len(active))
            ]
            archive = {
                str(r["id"]): {
                    "path": r["image"]["source"],
                    "decision": i,
                    "description": r["place"],
                }
                for i, r in enumerate(history)
                if r['id'] in by_id
            }
            catalog = {
                "root": str(directory.resolve()),
                "archive": archive,
                "keyframe_ids": sorted(key_ids),
                "uniform_frame_ids": uniform_ids,
                "node_frames": node_frame_index(
                    map_state.nodes, active, gtmap.rows, map_state.path[0]
                ),
            }
            key_index = [
                {
                    "frame": fid,
                    "decision": archive[str(fid)]["decision"],
                    "description_hypothesis": archive[str(fid)]["description"],
                    "reasons": [e["kind"] for e in key_events if e["frame"] == fid],
                }
                for fid in sorted(key_ids)
            ]
            allowed = {frame}
            all_frames = {frame, *by_id}
            recall_event = False
            recall_frame = None
            ctx = {
                "instruction": task["instruction"],
                "cause": "navigate",
                "evidence_ledger": events,
                "last_execution": last,
                "pose_evidence": pose.context(),
                "code_critic": code,
                "actions_remaining": self.config["episode_budget"]["primitives"]
                - steps,
                "task_goal_anchor_measurement": task_goal_critic.observe(self.env.sim),
                "goal_distance_checkpoints": goal_checkpoint.samples,
                "available_frame_ids": sorted(allowed),
                "max_travel_cm": 100,
                "keyframe_index": key_index,
                "uniform_history_index": [
                    {"frame": fid, "decision": archive[str(fid)]["decision"]}
                    for fid in uniform_ids
                ],
                "memory_tools": [
                    "retrieve_keyframes",
                    "retrieve_uniform_history",
                    "observe_node",
                ],
                "selected_history_panorama_ids": [],
                "key_events": key_events[-2:],
                "target_anchor_measurement": targetcritic.observe(self.env.sim),
                "predicted_semantic_graph": entrances.context(self.env.sim),
                "candidate_conflicts": candidate_conflicts,
                "pose_stall_alert": stall_alert,
                "route_progress_feedback": route_feedback,
                "return_anchor_candidates": [
                    {"frame_id": r["id"], "description_hypothesis": r["place"]}
                    for r in history[-8:]
                ],
                "active_branch": {
                    "return_anchor_frame": branch_anchor,
                    "target": branch_target,
                    "return_decisions": return_decisions,
                },
                "online_verification": wrapper,
                "recovery_event": code["recovery_event"],
                "route_memory": route_memory,
                "pending_route_review": pending_review,
                "last_review_resolution": last_review_resolution,
                "pending_node_return": pending_node_return,
            }
            ctx = compact_context(ctx)

            ctx.pop("pose_evidence", None)
            for field in ("metric_translation_m", "metric_scale"):
                ctx["last_execution"].pop(field, None)
            gtctx = gtmap.context(branch_anchor)
            gtctx["cumulative_map"]["points"] = gtctx["cumulative_map"]["points"][-4:]
            ctx["gt_own_pose_map"] = gtctx
            ctx["pose_history"] = gtmap.history_summary([r["id"] for r in active])
            mapctx = map_state.context()
            mapctx["nodes"] = [
                {
                    k: v
                    for k, v in n.items()
                    if k
                    in (
                        "id",
                        "name",
                        "caption",
                        "visits",
                        "links",
                        "straight_m",
                        "return_walked_m",
                        "turn_deg_left",
                        "annotation_anchor",
                        "xy_m",
                        "xy_from_start_m",
                        "path_index",
                    )
                }
                for n in mapctx["nodes"]
            ]
            mapctx["room_stage_hypotheses"] = room_stage.context()
            mapctx["unnamed_node_notice"] = [
                f"Node {n['id']} is visited but unnamed; inspect with observe_node and annotate if it matters to the instruction."
                for n in mapctx["nodes"]
                if not n["name"]
            ]
            mapctx["coordinate_rule"] = (
                "Map x/z axes are world-fixed. Use turn_deg_left for robot-relative node bearing: positive left, negative right; return_walked_m follows recorded path, not free-space distance."
            )
            ctx["observed_semantic_map"] = mapctx
            poses_by_step = {row['step']: row for row in gtmap.rows}
            for node in mapctx['nodes']:
                source = catalog['node_frames'].get(str(node['id']))
                position = map_state.return_target(node['id']) - map_state.path[0]
                node['pose_from_start_m'] = position.round(3).tolist()
                node['return_anchor_frame'] = source
                node['semantic_hypothesis'] = {'status': 'unknown'}
                if source in by_id and np.linalg.norm(position - poses_by_step[source]['xyz_m']) <= 0.5:
                    node['semantic_hypothesis'] = {'status': 'candidate', 'frame': source,
                                                   'place': by_id[source]['place']}
                node_path_index = (node.get('annotation_anchor') or node)['path_index']
                node['branch_status'] = ('direction_wrong_history' if rejected_branch and
                                         node_path_index > rejected_branch['anchor_path_index'] and source is None
                                         else 'active_recorded_path')
            ctx["observed_map_status"] = {
                k: v for k, v in map_info.items() if k not in ("image", "nodes")
            }
            dp = directory / "decisions" / f"{decision:03d}"
            model_ctx = navigation_context(
                task["instruction"],
                decision,
                frame,
                last,
                stall_alert,
                mapctx,
                key_index,
                pending_node_return,
                ctx["pose_history"] if self.config.get("pose_without_map") else None,
                forward_clearance(obs) if self.config.get("pose_without_map") else None,
                memory.latest,
            )
            model_ctx.update(pending_route_review=pending_review, rejected_branch=rejected_branch,
                             clause_states=ledger.snapshot(), review_unavailable=review_failure)
            if session_id is None:
                model_ctx['instruction'] = task['instruction']
            model_ctx["actions_remaining"] = max(
                0, self.config["episode_budget"]["primitives"] - steps
            )
            if decision == 0:
                model_ctx["memory_tools"] = []
                model_ctx["uniform_history_index"] = ctx["uniform_history_index"]
            if candidate_conflicts:
                model_ctx["candidate_conflicts"] = candidate_conflicts[:2]
            immutable(
                dp / "evidence.json",
                {
                    "context": model_ctx,
                    "images": images,
                    "step": frame,
                    "decision": decision,
                },
            )
            payload = {
                "role": "navigate",
                "context": model_ctx,
                "memory_catalog": catalog,
                "session_id": session_id,
                "frames": [
                    {
                        "label": im["label"],
                        "data": base64.b64encode(
                            Path(im["source"]).read_bytes()
                        ).decode(),
                    }
                    for im in images
                ],
            }
            response = self.request("eval/model.py", payload, dp, 360)
            account(dp)
            out = copy.deepcopy(response["output"])
            if not response.get("session_id"):
                raise ValueError(
                    "Model response did not include a resumable session ID"
                )
            session_id = response["session_id"]
            retrieved = set(response.get("retrieved_frame_ids", []))
            if not retrieved <= set(by_id):
                raise ValueError("Memory tool returned an unknown frame")
            allowed.update(retrieved)
            ctx["available_frame_ids"] = sorted(allowed)
            write(
                dp / "memory_tool_calls.json",
                {
                    "strategy": response.get("history_strategy"),
                    "limit": response.get("history_limit"),
                    "calls": response.get("memory_calls", []),
                    "retrieved_frame_ids": sorted(retrieved),
                },
            )
            stall_alert = None
            for correction in range(2):
                try:
                    picks = out.get("history_panorama_ids")
                    if picks != []:
                        raise ValueError(
                            "History is attached automatically; history_panorama_ids must be []"
                        )
                    target = out.get("target", {})
                    point = target.get("point_norm")
                    if point is not None and (
                        len(point) != 2
                        or any(
                            type(x) not in (int, float) or not 0 <= x <= 1
                            for x in point
                        )
                    ):
                        raise ValueError(
                            "target point_norm must be two numbers in [0,1]"
                        )

                    out["entrance_candidates"] = [
                        candidate
                        for candidate in out["entrance_candidates"]
                        if candidate.get("evidence_frame") == frame
                    ]
                    for candidate in out["entrance_candidates"]:
                        box = candidate["bbox_norm"]
                        if candidate["evidence_frame"] != frame:
                            raise ValueError(
                                "Entrance candidates must cite current visual frame"
                            )
                        if box is not None and (
                            len(box) != 4
                            or any(
                                type(x) not in (int, float) or not 0 <= x <= 1
                                for x in box
                            )
                            or box[0] >= box[2]
                            or box[1] >= box[3]
                        ):
                            raise ValueError("Invalid entrance bbox_norm")
                    operation = out.get("map_operation")
                    if (
                        decision == 0
                        and out.get("maneuver") == "travel"
                        and operation != "navigate_to_node"
                        and out.get("return_anchor_frame") == frame
                    ):
                        out["return_anchor_frame"] = None
                    if operation == "navigate_to_node":
                        selected = (
                            map_state.nodes[out.get("map_node_id")]
                            if type(out.get("map_node_id")) is int
                            and 0 <= out["map_node_id"] < len(map_state.nodes)
                            else None
                        )
                        if selected is None:
                            raise ValueError("unknown visited map node")
                        if out.get("maneuver") != "return_to_anchor":
                            raise ValueError(
                                "node return requires return_to_anchor maneuver"
                            )
                        anchor_frame = (
                            active_return['anchor_frame']
                            if pending_node_return and selected['id'] == active_return['node_id']
                            else catalog['node_frames'].get(str(selected['id']))
                        )
                        if anchor_frame is None:
                            raise ValueError(
                                "Selected map node has no retained panorama anchor"
                            )
                        if out.get('return_anchor_frame') not in (None, anchor_frame):
                            raise ValueError(
                                f"Node {selected['id']} binds to return_anchor_frame {anchor_frame}; "
                                "use that frame or null for executor binding"
                            )
                        return_path_index = map_state.return_path_index(
                            selected['id'], by_id[anchor_frame]['path_index']
                        )
                        out["return_anchor_frame"] = anchor_frame
                    elif operation == "annotate_node":
                        map_state.return_target(out.get("map_node_id"))
                        if not out.get("map_annotation", "").strip():
                            raise ValueError("empty node annotation")
                    elif operation != "none" or out.get("map_node_id") is not None:
                        raise ValueError("invalid map operation")
                    room_claim = out["room_stage_hypothesis"]
                    if room_claim["relation"] != "none":
                        if room_claim["node_id"] not in {
                            node["id"] for node in map_state.nodes
                        }:
                            raise ValueError(
                                "Room-stage hypothesis requires an already visited node"
                            )
                        if room_claim["evidence_frame"] not in allowed:
                            raise ValueError(
                                "Room-stage hypothesis cites a historical panorama not retrieved this decision"
                            )
                    validate_review_resolution(
                        ctx.get("pending_route_review"),
                        out["review_response"],
                        out,
                        allowed,
                        frame,
                    )
                    validate_stop_clauses(out)
                    if out["skill"] == "stop" or (
                        history and out["stage"] > history[-1]["stage"]
                    ):
                        cited = {
                            fid
                            for row in out["route_evidence"]["landmarks"]
                            + out["route_evidence"]["prerequisites"]
                            if row["verdict"] == "confirmed"
                            for fid in row["frames"]
                        }
                        cited.update(
                            out["route_evidence"]["route_frames"]
                            if out["route_evidence"]["route_verdict"] != "unknown"
                            else []
                        )
                        if not cited <= allowed:
                            raise ValueError(
                                "STOP or stage advance cited a historical image not retrieved this decision"
                            )
                    validate_action(
                        out, task["instruction"], allowed, {r["id"] for r in history}
                    )
                    ledger.validate_events(out['events'])
                    if decision >= self.config.get("fork_decision", 0):
                        validate_compound_plan(
                            out,
                            dict(
                                ctx,
                                compound_route=instruction_constraints(
                                    task["instruction"]
                                )["linked_landmark_turn"],
                            ),
                        )
                    break
                except (ValueError, KeyError, TypeError) as exc:
                    if correction:
                        failed = dp / 'action_contract_repair' / 'response_0.json'
                        if failed.exists():
                            failed.rename(failed.with_name(f'response_0_error_{time.time_ns()}.json'))
                        raise CampaignHold(f'Invalid action after repair: {exc}') from exc
                    repair = copy.deepcopy(payload)
                    repair["context"]["action_contract_error"] = {
                        "error": str(exc),
                        "rejected_output": out,
                    }
                    repair["session_id"] = session_id
                    response = self.request(
                        "eval/model.py", repair, dp / "action_contract_repair", 360
                    )
                    account(dp / "action_contract_repair")
                    out = copy.deepcopy(response["output"])
                    if not response.get("session_id"):
                        raise ValueError(
                            "Model repair did not include a resumable session ID"
                        )
                    session_id = response["session_id"]
                    retrieved = set(response.get("retrieved_frame_ids", []))
                    allowed.update(retrieved)
                    ctx["available_frame_ids"] = sorted(allowed)
                    write(
                        dp / "action_contract_repair" / "memory_tool_calls.json",
                        {
                            "calls": response.get("memory_calls", []),
                            "retrieved_frame_ids": sorted(retrieved),
                        },
                    )
            task_goal = out["task_goal_anchor"]
            if (
                task_goal_critic.anchor is None
                and task_goal["status"] in ("seen", "approaching", "reached")
                and task_goal["point_norm"] is not None
            ):
                endpoint_hypothesis = copy.deepcopy(task_goal)
                if endpoint_hypothesis["status"] == "reached":
                    endpoint_hypothesis["status"] = "seen"
                endpoint = task_goal_critic.bind(
                    endpoint_hypothesis, obs, self.env.sim, frame
                )
                if endpoint.get("status") == "tracked":
                    goal_checkpoint.set_initial(
                        endpoint["distance_xz_m"], completed_decisions=decision
                    )
            if (
                out["target"]["status"] in ("seen", "approaching")
                and out["target"]["point_norm"] is not None
            ):
                bound = targetcritic.bind(out["target"], obs, self.env.sim, frame)
                if bound["status"] == "tracked" and not any(
                    e["kind"] == "target_first_seen" for e in key_events
                ):
                    key_events.append(
                        {
                            "frame": frame,
                            "kind": "target_first_seen",
                            "description": out["target"]["description"],
                        }
                    )
            here = np.asarray(self.env.sim.get_agent_state().position, dtype=float)
            node = min(
                map_state.nodes,
                key=lambda n: np.linalg.norm(
                    (np.asarray(n["position"]) - here)[[0, 2]]
                ),
            )["id"]
            room_claim = out["room_stage_hypothesis"]
            if room_claim["relation"] != "none":
                room_stage.update(
                    room_claim["node_id"],
                    room_claim["room"],
                    room_claim["relation"],
                    room_claim["instruction_stage"],
                    room_claim["evidence_frame"],
                    allowed,
                )
            entrances.observe_candidates(
                out["entrance_candidates"], obs, self.env.sim, frame, node
            )
            if entrances is not future_entrances:
                future_entrances.observe_candidates(
                    out["entrance_candidates"], obs, self.env.sim, frame, node
                )
            for item in entrances.candidates:
                same_node = sum(
                    e["kind"] == "entrance_candidate" and e.get("visited_node") == node
                    for e in key_events
                )
                if (
                    item["first_seen_frame"] == frame
                    and item["status"] != "archived"
                    and item["instruction_relevance"] != "unlikely"
                    and same_node < 2
                ):
                    key_events.append(
                        {
                            "frame": frame,
                            "kind": "entrance_candidate",
                            "visited_node": node,
                            "description": item["description_hypothesis"],
                        }
                    )

            semantic_audit = None
            if semantic is not None:
                semantic_audit = semantic.inspect(out, frame, allowed, history)
                immutable(
                    directory / "semantic_audits" / f"{decision:03d}.json",
                    semantic_audit,
                )
            if out["map_operation"] == "annotate_node":
                map_state.annotate_node(
                    out["map_node_id"],
                    out["map_annotation"],
                    "frames " + str(out["evidence_frames"]),
                    clause=out.get("stage"),
                    frame=frame,
                    image=current["source"],
                )
            pa = critic.proposal(out, frame, history, allowed)
            immutable(directory / "code_critic" / f"{decision:03d}_proposal.json", pa)
            out["events"] = [
                dict(e, frames=sorted(set(e["frames"]) & all_frames))
                for e in out["events"]
            ]
            if out['map_operation'] == 'navigate_to_node':
                if return_iterator is None:
                    return_iterator = map_state.actions_to(
                        out['map_node_id'], anchor_path_index=return_path_index
                    )
                    active_return = {'attempt_id': len(backtracking_attempts) + 1,
                                     'node_id': out['map_node_id'], 'start_decision': decision,
                                     'target_position': map_state.return_target(out['map_node_id']).tolist(),
                                     'anchor_frame': out['return_anchor_frame'],
                                     'anchor_path_index': return_path_index,
                                     'anchor_position': map_state.path[return_path_index].tolist(),
                                     'status': 'in_progress', 'primitives': 0,
                                     'route_recovery': 'not_evaluated'}
                    backtracking_attempts.append(active_return)
                sequence = return_iterator
            else:
                sequence = primitives(out)
            actual = []
            interrupted = False
            before_obs = obs
            heading = pose.commanded_yaw
            node_pose = gtmap.rows[-1].copy()
            goal_checkpoint_sample = None
            candidate_conflicts = []
            if out.get("return_anchor_frame") is not None:
                if out["return_anchor_frame"] != branch_anchor:
                    return_decisions = 0
                branch_anchor = out["return_anchor_frame"]
                branch_target = out.get("branch_target", "")
            if out["maneuver"] == "return_to_anchor":
                return_decisions += 1
            critic.save_observation(frame, obs, out)
            for action in sequence:
                if (
                    steps >= self.config["episode_budget"]["primitives"]
                    or self.env.episode_over
                ):
                    break
                if action == 1 and self.config.get("depth_preflight_candidate"):
                    depth_risk = forward_clearance(obs)
                    if (
                        depth_risk["status"] == "possible_obstacle"
                        and depth_risk["clearance_m"] <= 0.30
                    ):
                        stall_alert = {
                            "frame": steps,
                            "reason": "depth predicts a near obstacle before forward",
                            "depth_preflight": depth_risk,
                            "completion": "unknown; inspect or adjust approach",
                        }
                        interrupted = True
                        break
                before = obs
                prior_target = targetcritic.observe(self.env.sim)
                oldpos = np.array(self.env.sim.get_agent_state().position)
                oldheading = gtmap.rows[-1]["heading_xz_deg"]
                obs = self.env.step(action)
                steps += 1
                gtmap.record(steps)
                delta = np.array(self.env.sim.get_agent_state().position) - oldpos
                map_state.record_pose()
                row = integrate(before, obs, action, steps)
                row["relative_pose"] = pose.update(before, obs, action)
                actual.append(row)
                row["measured_translation_m"] = float(np.linalg.norm(delta))
                row["metric_odometry_source"] = "simulator GT own position"
                row["gt_height_delta_m"] = float(delta[1])
                stallcheck = stallcritic.check(action, delta[[0, 2]], oldheading, steps)
                row["pose_stall_critic"] = stallcheck
                immutable(
                    directory / "pose_stall_critic" / f"{steps:04d}.json", stallcheck
                )
                motioncheck = motioncritic.evaluate(
                    {"maneuver": out["maneuver"], "travel_heading_deg": oldheading},
                    {
                        "step": steps,
                        "action": action,
                        "source": "simulator GT own pose",
                        "height_delta_m": float(delta[1]),
                        "translation_xz_m": delta[[0, 2]].tolist(),
                    },
                )
                immutable(
                    directory / "gt_motion_critic" / f"{steps:04d}.json", motioncheck
                )
                write(
                    directory / "gt_own_pose_map.json",
                    {"points": gtmap.rows, "context": gtmap.context(branch_anchor)},
                )
                checked = critic.after_primitive(
                    row, obs, history, branch_anchor, out["events"]
                )
                immutable(
                    directory
                    / "code_critic"
                    / f"{decision:03d}_primitive_{steps:04d}.json",
                    checked,
                )
                targetcheck = targetcritic.after_primitive(
                    self.env.sim,
                    action,
                    prior_target,
                    out["expected_effect"]["expected_distance_change"],
                )
                entrance_alerts = entrances.after_primitive(
                    self.env.sim, action, row["measured_translation_m"]
                )
                if entrances is not future_entrances:
                    future_entrances.after_primitive(
                        self.env.sim, action, row["measured_translation_m"]
                    )
                row["target_critic"] = targetcheck
                row["entrance_alerts"] = entrance_alerts
                immutable(
                    directory / "target_critic" / f"{steps:04d}.json", targetcheck
                )
                if action == 0:
                    status = "stopped"
                    break
                if stallcheck["interrupt"]:
                    stall_alert = {
                        "frame": steps,
                        "forward_m": stallcheck["forward_m"],
                        "reason": stallcheck["reason"],
                        "completion": "unknown; blockage does not imply arrival",
                    }
                    interrupted = True
                    break
                if entrance_alerts and out['map_operation'] != 'navigate_to_node':
                    candidate_conflicts = entrance_alerts
                    recall_event = True
                    recall_frame = entrance_alerts[0]["first_seen_frame"]
                    interrupted = True
                    break
                if targetcheck["interrupt"] and out['map_operation'] != 'navigate_to_node':
                    recall_event = True
                    interrupted = True
                    break
                if motioncheck["interrupt"] or any(
                    p["interrupt"] for p in checked["proposals"]
                ):
                    interrupted = True
                    break
            if status != "stopped" and not self.env.episode_over:
                goal_checkpoint_sample = goal_checkpoint.check(
                    decision + 1, task_goal_critic.observe(self.env.sim)
                )

            last = {
                "actual": actual,
                "requested_primitives": (
                    len(actual)
                    if out["map_operation"] == "navigate_to_node"
                    else len(sequence)
                ),
                "executed_primitives": len(actual),
                "critic_interrupted": interrupted,
                "blocked": bool(stall_alert),
                "blocked_status": stall_alert
                or "no measured stall in this action chunk",
                "goal_distance_checkpoint": goal_checkpoint_sample,
                "endpoint_regression_advice": (
                    goal_checkpoint_sample
                    if goal_checkpoint_sample and goal_checkpoint_sample["stop"]
                    else None
                ),
                "forced_stop": None,
            }
            node_return_check = None
            if out["map_operation"] == "navigate_to_node":
                target_position = active_return['target_position']
                current_position = self.env.sim.get_agent_state().position
                node_return_check = return_arrival(target_position, current_position)
                anchor_check = return_arrival(active_return['anchor_position'], current_position)
                node_return_check['anchor_distance_3d_m'] = anchor_check['distance_3d_m']
                node_return_check['arrived'] = node_return_check['arrived'] and anchor_check['arrived']
                active_return.update(last_decision=decision, arrival=node_return_check)
                active_return['primitives'] += len(actual)
                active_return['status'] = 'arrived' if node_return_check['arrived'] else 'interrupted' if interrupted else 'not_arrived'
                write(directory / 'backtracking_attempts.json', backtracking_attempts)
                if not node_return_check['arrived'] and not interrupted and steps < self.config['episode_budget']['primitives'] and not self.env.episode_over:
                    active_return['status'] = 'path_exhausted_before_arrival'
                    write(directory / 'backtracking_attempts.json', backtracking_attempts)
                    raise CampaignHold('Recorded return path ended before measured arrival; no successful return recorded')
                if node_return_check["arrived"]:
                    return_iterator = None
                    anchor = out['return_anchor_frame']
                    excluded_frames.update(r['id'] for r in history if r['id'] > anchor)
                    if frame > anchor:
                        excluded_frames.add(frame)
                    departure = next((r for r in active if r['id'] == anchor), {})
                    poses = {r['step']: r for r in gtmap.rows}
                    anchor_pose = poses.get(anchor)
                    travel = next((r['step'] for r in gtmap.rows if anchor < r['step'] <= frame and
                                   anchor_pose is not None and
                                   np.linalg.norm(np.subtract(r['xyz_m'], anchor_pose['xyz_m'])) > 0.2), None)
                    rejected_branch = {'anchor_frame': anchor, 'node_id': out['map_node_id'],
                                       'anchor_path_index': active_return['anchor_path_index'],
                                       'departure_delta_xyz_m': None if travel is None or anchor_pose is None else np.subtract(
                                           poses[travel]['xyz_m'], anchor_pose['xyz_m']).round(3).tolist(),
                                       'departure_commands': departure.get('output', {}).get('actions'),
                                       'departure_heading': departure.get('heading'),
                                       'direction_status': 'direction_wrong',
                                       'route_status': 'recovery_required',
                                       'verdict': 'executed direction contradicted; branch rejected'}
                    # Keep the original Codex thread. The rejected branch is
                    # excluded from future visual evidence, but the episode
                    # continues in the same session.
                    memory.latest = None
                    pending_node_return = None
                else:
                    pending_node_return = {
                        "node_id": out["map_node_id"],
                        "return_anchor_frame": active_return['anchor_frame'],
                        "distance_xz_m": node_return_check["distance_xz_m"],
                        "reason": "Recorded-path return was interrupted or incomplete; route review remains unresolved.",
                    }
                last["node_return_check"] = node_return_check
            if pending_review is not None:
                review_id = pending_review["checked_before_decision"]
                if out["review_response"]["resolution"] == "inspect_suspected_branch":
                    review_inspection_counts[review_id] = (
                        review_inspection_counts.get(review_id, 0) + 1
                    )
                    view = gtmap.rows[-1]
                    translation = float(np.linalg.norm(np.subtract(view['xyz_m'], node_pose['xyz_m'])))
                    yaw = abs((view['heading_xz_deg'] - node_pose['heading_xz_deg'] + 180) % 360 - 180)
                    review_inspections[review_id] = {
                        "frame": frame,
                        "result_frame": steps,
                        "planned_primitives": primitives(out),
                        "executed_primitives": len(actual),
                        "net_translation_m": round(translation, 3),
                        "net_yaw_deg": round(yaw, 3),
                        "viewpoint_changed": translation >= 0.15 or yaw >= 30,
                    }
                last_review_resolution = {
                    "checked_before_decision": pending_review[
                        "checked_before_decision"
                    ],
                    "resolution": out["review_response"]["resolution"],
                    "evidence_frames": out["review_response"]["evidence_frames"],
                    "current_test": out["review_response"]["current_test"],
                    "actual_primitives": len(actual),
                    "inspection_count": review_inspection_counts.get(review_id, 0),
                    "inspection": review_inspections.get(review_id) if out['review_response']['resolution'] == 'inspect_suspected_branch' else None,
                    "critic_interrupted": interrupted,
                    "blocked": bool(stall_alert),
                    "requires_followup": (
                        review_remains_open(pending_review, out["review_response"])
                        or pending_node_return is not None
                    ),
                }
            ledger.update(out['events'], allowed)
            if node_return_check and node_return_check['arrived']:
                ledger.invalidate(excluded_frames)
            write(directory / 'clause_states.json', ledger.snapshot())
            front = save_image(
                directory / "front_history" / f"{decision:03d}_{frame:04d}_FRONT.jpg",
                np.asarray(before_obs["rgb"])[..., :3],
                f"PAST frame {frame} FRONT",
                frame,
            )
            history.append(
                {
                    "id": frame,
                    'decision': decision,
                    "path_index": frame_path_index,
                    "stage": out["stage"],
                    "heading": heading,
                    "place": out["place"],
                    "note": out["note"],
                    "events": out["events"],
                    "last_execution": last,
                    "front_image": front,
                    "image": current,
                }
            )
            history[-1]["output"] = out
            memory.remember(history[-1], history[-2] if len(history) > 1 else None)
            write(
                directory / "keyframe_memory.json",
                {
                    "keyframes": list(memory.keyframes.values()),
                    "reviews": memory.reviews,
                },
            )
            trace.append(
                {
                    "decision": decision,
                    "step": frame,
                    "context": ctx,
                    "output": out,
                    "execution": last,
                    "skill_mode": "navigate",
                    "code_critic": critic.packet(),
                    "critic_model_calls_total": review_calls,
                    "semantic_evidence_audit": semantic_audit,
                }
            )
            previous = {
                "decision": decision,
                "images": [current],
                "input_file": str(dp / "evidence.json"),
                "output": out,
                "context": ctx,
            }
            write(directory / "trajectory.partial.json", trace)
            if not self.runtime_unchanged():
                raise CampaignHold("Runtime source changed; no mixed scores")
            if status == "stopped":
                break
        if previous:
            current = save_image(
                directory
                / "images"
                / f"final_{steps:04d}_PANORAMA_FRONT_RIGHT_REAR_LEFT.jpg",
                views(obs)[0][1],
                f"CURRENT frame {steps} PANORAMA_FRONT_RIGHT_REAR_LEFT",
                steps,
            )
            # The regular transition for this decision may already have been
            # sealed when the next decision started. Keep the episode-boundary
            # record separate instead of rewriting an immutable artifact.
            immutable(
                directory / "transitions" / f'{previous["decision"]:03d}_episode_boundary.json',
                {
                    "decision": previous["decision"],
                    "before": previous["images"],
                    "input_file": previous["input_file"],
                    "instruction": task["instruction"],
                    "output": previous["output"],
                    "execution": last,
                    "after": [current],
                    "anomaly_reasons": ["episode_boundary"],
                },
            )
        critic.end(steps)
        if active_return and active_return['status'] != 'arrived':
            active_return['status'] = 'budget_exhausted' if steps >= self.config['episode_budget']['primitives'] else 'episode_ended_before_arrival'
        write(directory / 'backtracking_attempts.json', backtracking_attempts)
        write(directory / "recovery_events.json", critic.episode_events)
        write(directory / "code_critic_audit.json", critic.audit)
        write(directory / "trajectory.json", trace)
        write(
            directory / "goal_distance_checkpoints.json",
            {
                "interval_decisions": goal_checkpoint.interval,
                "regression_tolerance_m": goal_checkpoint.regression_tolerance_m,
                "endpoint_anchor": (
                    None
                    if task_goal_critic.anchor is None
                    else {
                        "description_hypothesis": task_goal_critic.anchor[
                            "description"
                        ],
                        "source_frame": task_goal_critic.anchor["frame"],
                    }
                ),
                "samples": goal_checkpoint.samples,
                "forced_stop": goal_checkpoint_stop,
                "authority": "Distance to a model-identified RGB-D endpoint hypothesis only; unknown when endpoint is not stably observed.",
            },
        )
        if semantic is not None:
            write(
                directory / "semantic_checks.json",
                {
                    "scope": "Audit signatures, not proven root causes or independently labeled perception errors",
                    "records": [r for r in semantic.records if r["issues"]],
                    "full_trace": "trajectory.json",
                    "per_decision_inputs": "decisions/*/evidence.json",
                    "review_note": "Observations and model claims are recorded separately.",
                },
            )
        metrics = jsonable(self.env.get_metrics())
        if task.get("reference_path"):
            reference = np.asarray(task["reference_path"], dtype=float)
            origin = np.asarray(episode.start_position, dtype=float)
            predicted = np.asarray([row["xyz_m"] for row in gtmap.rows], dtype=float)
            metrics["trajectory"] = trajectory_scores(
                predicted,
                reference - origin,
                bool(status == "stopped" and metrics.get("success", 0)),
            )
        return {
            "identity": self.run_identity(version, task),
            "episode_id": task["id"],
            "version": version["id"],
            "status": status,
            "success": float(metrics.get("success", 0)) if status == "stopped" else 0,
            "spl": float(metrics.get("spl", 0)) if status == "stopped" else 0,
            "metrics": metrics,
            "steps": steps,
            "decisions": len(trace),
            "processed_tokens": tokens,
            "sdk_usd": None if cost_missing else usd,
            "recovery": critic.summary(),
            "goal_distance_checkpoint_stop": goal_checkpoint_stop,
            "goal_distance_checkpoint_samples": goal_checkpoint.samples,
            "implementation_unchanged": self.runtime_unchanged(),
            "critic_model_calls": review_calls,
            "scope": "visual route review and local execution checks",
            "elapsed_s": time.monotonic() - started,
        }
