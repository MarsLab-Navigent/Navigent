from collections import Counter
import copy


def active_review_reason(
    decision,
    history,
    last,
    key_events,
    candidate_conflicts,
    stall_alert,
    pending_node_return=None,
    pending_review=None,
):
    """Return a concrete event that makes route review useful now."""
    if decision <= 0 or not history or pending_node_return is not None or pending_review is not None:
        return None
    if candidate_conflicts:
        return "active_branch_conflict"
    if stall_alert or last.get("blocked"):
        return "active_stall"
    if last.get("critic_interrupted"):
        return "active_critic_interruption"
    checkpoint = last.get("goal_distance_checkpoint") or {}
    if checkpoint.get("stop"):
        return "active_goal_regression"
    recent = [event for event in key_events if event.get("frame") == history[-1]["id"]]
    if any(event.get("kind") == "direction_change" for event in recent):
        return "active_direction_change"
    return None


def stage_stall_review_reason(history, last_review_attempt):
    tail = history[-8:]
    if len(tail) != 8 or any(row['stage'] != tail[-1]['stage'] for row in tail):
        return None
    decisions = [row['decision'] for row in tail]
    if decisions != list(range(decisions[0], decisions[0] + 8)):
        return None
    if last_review_attempt is not None and decisions[0] < last_review_attempt:
        return None
    return 'active_stage_stall'


def execution(value):
    rows = value.get("actual", [])
    counts = Counter(r["action"] for r in rows)
    poses = [r.get("relative_pose") for r in rows if r.get("relative_pose")]
    return {
        "first_step": rows[0]["step"] if rows else None,
        "last_step": rows[-1]["step"] if rows else None,
        "executed_forward_primitives": counts[1],
        "executed_turn_left_primitives": counts[2],
        "executed_turn_right_primitives": counts[3],
        "stop_executed": bool(counts[0]),
        "executed_turn_command_deg": sum(r.get("executed_turn_deg", 0) for r in rows),
        "metric_translation_m": None,
        "metric_scale": "unknown",
        "pose_quality_counts": dict(
            Counter(p.get("quality", "unknown") for p in poses)
        ),
        "requested_primitives": value.get("requested_primitives"),
        "executed_primitives": value.get("executed_primitives", len(rows)),
    }


def pose(value):
    p = value.get("last_relative_pose") or {}
    return {
        "source": "RGB relative-direction estimate; no true pose",
        "commanded_heading_deg": value.get("commanded_heading_deg"),
        "last_relative_yaw_deg": p.get("visual_relative_yaw_deg"),
        "translation_direction_camera": p.get("translation_direction_camera"),
        "quality": p.get("quality", "unknown"),
        "inlier_ratio": p.get("inlier_ratio"),
        "metric_translation_m": None,
        "metric_scale": "unknown",
    }


def context(value):
    c = copy.deepcopy(value)
    c.pop("panorama_action_contract", None)
    c.pop("observation_memory", None)
    c.pop("execution_summary", None)
    c["last_execution"] = execution(c.get("last_execution") or {})
    c["pose_evidence"] = pose(c.get("pose_evidence") or {})
    for item in c.get("uniform8_action_history", []):
        item["executed_after_frame"] = execution(item.get("executed_after_frame") or {})
    return c


"""Chronological uniform sampling, matching the original GPT6-VLN strategy."""


def uniform_history_indices(total, count=8):
    if total < 0 or count < 0:
        raise ValueError("total and count must be nonnegative")
    if count == 0 or total == 0:
        return []
    if total <= count:
        return list(range(total))
    if count == 1:
        return [0]
    return [i * (total - 1) // (count - 1) for i in range(count)]


import numpy as np


def node_frame_index(nodes, history, pose_rows, start_position, max_distance_m=0.5):
    if not history:
        return {}
    positions = {
        int(row["step"]): np.asarray(row["xyz_m"], dtype=float) for row in pose_rows
    }
    available = [
        int(record["id"]) for record in history if int(record["id"]) in positions
    ]
    origin = np.asarray(start_position, dtype=float)
    result = {}
    for node in nodes:
        if not available:
            break
        anchor = node.get("anchor") or {}
        target = np.asarray(anchor.get('position', node['position']), dtype=float) - origin
        frame = anchor.get("frame_id")
        if frame in available and np.linalg.norm(positions[frame] - target) <= max_distance_m:
            result[str(node["id"])] = int(frame)
            continue
        nearest = min(
            available,
            key=lambda fid: (float(np.linalg.norm(positions[fid] - target)), -fid),
        )
        if np.linalg.norm(positions[nearest] - target) <= max_distance_m:
            result[str(node['id'])] = nearest
    return result


class RoomStageLedger:
    def __init__(self):
        self.records = {}

    def update(self, node_id, room, relation, stage, evidence_frame, allowed_frames):
        if relation == "none":
            return None
        if evidence_frame not in set(allowed_frames):
            raise ValueError("Room-stage hypothesis must cite a supplied panorama")
        if relation == "confirmed" and (not room.strip() or type(stage) is not int):
            raise ValueError("Confirmed room-stage claim needs a room and stage")
        row = {
            "node_id": node_id,
            "room_hypothesis": room[:80],
            "instruction_relation": relation,
            "stage_hypothesis": stage,
            "evidence_frame": evidence_frame,
            "authority": "Model visual hypothesis, not Matterport ground truth",
        }
        self.records[node_id] = row
        return row

    def context(self):
        return [self.records[key] for key in sorted(self.records)]


import copy


class RouteReviewMemory:
    def __init__(self, interval=10):
        if type(interval) is not int or interval < 1:
            raise ValueError("Review interval must be a positive integer")
        self.interval = interval
        self.keyframes = {}
        self.latest = None
        self.reviews = []

    def due(self, decision):

        return decision > 0 and decision % self.interval == 0

    def remember(self, record, previous=None):
        reasons = []
        if previous is None:
            reasons.append("start")
        if previous is not None and record["stage"] != previous["stage"]:
            reasons.append("actor_claimed_stage_change")
        out = record.get("output", {})
        if out.get("return_anchor_frame") is not None:
            reasons.append("branch_return_reference")
        if out.get("maneuver") in ("climb_up", "climb_down", "return_to_anchor"):
            reasons.append(out["maneuver"])
        if record.get("events"):
            reasons.append("actor_cited_route_event")
        if reasons:
            self.keyframes[record["id"]] = {
                "frame_id": record["id"],
                "reasons": reasons,
                "description_hypothesis": record.get("place", ""),
                "verification": "unverified actor hypothesis; image is recorded evidence",
                "image": copy.deepcopy(record["image"]),
            }

    def select(self, history, anchor=None, limit=6):
        by_id = {r["id"]: r for r in history}

        ids = []
        if anchor in by_id:
            ids.append(anchor)
        proposed = (self.latest or {}).get("suggested_return_anchor_frame")
        if proposed in by_id:
            ids.append(proposed)
        keys = sorted(self.keyframes)
        if keys:
            ids.append(keys[0])
            ids.extend(reversed(keys))
        selected = []
        for fid in ids:
            if fid in by_id and fid not in selected:
                selected.append(fid)
            if len(selected) >= limit:
                break
        return [
            dict(
                by_id[fid]["image"],
                label=f"KEY frame {fid} PANORAMA saved route evidence",
            )
            for fid in sorted(selected)
        ]

    def accept(self, verdict, decision, supplied_frames, history):
        value = copy.deepcopy(verdict)
        if value.get('decision') not in ('continue', 'inspect_required', 'backtrack', 'execution_blocked'):
            raise ValueError("Invalid route review verdict")
        if not set(value.get("frames", [])) <= set(supplied_frames):
            raise ValueError("Reviewer cited an image it did not receive")
        target = value.get("suggested_return_anchor_frame")
        prior = {r["id"] for r in history}
        if target is not None and (
            type(target) is not int
            or target not in prior
        ):
            raise ValueError(
                "Reviewer return target must be an active historical frame"
            )
        if not value.get('frames'):
            raise ValueError('Review decision needs visual citations')
        value.update(
            checked_before_decision=decision,
            authority="Review decision must be resolved by the navigation contract; pose arrival does not certify route recovery",
        )
        self.latest = value
        self.reviews.append(copy.deepcopy(value))

    def context(self, decision, history, anchor=None):
        verdict = copy.deepcopy(self.latest)
        if verdict:
            verdict["age_in_decisions"] = decision - verdict["checked_before_decision"]
        ids = [r["id"] for r in history]
        start = ids.index(anchor) if anchor in ids else max(0, len(history) - 8)

        trail = [
            {"frame": r["id"], "executed_after_frame": execution(r["last_execution"])}
            for r in history[start:]
        ]
        return {
            "interval_decisions": self.interval,
            "latest_review": verdict,
            "keyframe_index": [
                {k: v for k, v in row.items() if k != "image"}
                for row in self.keyframes.values()
            ],
            "return_trail": trail,
            "return_policy": "Unknown is not deviation. Compare supplied anchor RGB, choose short visible segments; never invert commands or claim route recovery from motion alone.",
            "return_verified": False,
        }


def merge_images(*groups):
    result = []
    seen = set()
    for group in groups:
        for im in group:
            key = (im["frame"], im["source"])
            if key not in seen:
                seen.add(key)
                result.append(im)
    return result


"""Public, bounded evidence sent to the navigation model each decision."""


def execution_feedback(last, stall_alert):
    actual = last.get("actual") or []
    forward = [row for row in actual if row.get("action") == 1]
    return {
        "requested_primitives": last.get("requested_primitives", 0),
        "executed_primitives": last.get("executed_primitives", 0),
        "executed_forward_primitives": len(forward),
        "executed_turn_left_primitives": sum(row.get("action") == 2 for row in actual),
        "executed_turn_right_primitives": sum(row.get("action") == 3 for row in actual),
        "measured_forward_path_m": round(
            sum(float(row.get("measured_translation_m") or 0) for row in forward), 3
        ),
        "measured_height_change_m": round(
            sum(float(row.get("gt_height_delta_m") or 0) for row in actual), 3
        ),
        "critic_interrupted": bool(last.get("critic_interrupted")),
        "blocked": bool(stall_alert or last.get("blocked")),
        "measurement_note": "Forward path length is executed motion, not net route progress.",
    }


def review_feedback(latest, decision):
    if latest is None:
        return None
    fields = (
        "checked_before_decision",
        "decision",
        "evidence",
        "prediction_errors",
        "next_check",
        "task_summary",
        "frames",
        "suggested_return_anchor_frame",
    )
    return {
        **{key: latest.get(key) for key in fields},
        "age_in_decisions": decision - latest["checked_before_decision"],
        "authority": "Navigator resolves the review decision using cited observations and measured execution.",
    }


def navigation_context(
    instruction,
    decision,
    frame,
    last,
    stall_alert,
    mapctx,
    key_index,
    pending_node_return,
    pose_history,
    depth_preflight,
    latest_review,
):
    context = {
        "decision_id": decision,
        "frame_id": frame,
        "last_execution": execution_feedback(last, stall_alert),
        "observed_semantic_map": mapctx,
        "keyframe_index": key_index[-16:],
        "pending_node_return": pending_node_return,
        "pose_history": pose_history,
        "depth_preflight": depth_preflight,
        "route_review": review_feedback(latest_review, decision),
    }
    if decision == 0:
        context["instruction"] = instruction
    return context
