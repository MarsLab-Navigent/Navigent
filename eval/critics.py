import math


class ExecutionContractCritic:

    def __init__(self):
        self.flat_climb_steps = 0
        self.last_step = None

    def evaluate(self, intent, measurement):
        if self.last_step == measurement["step"]:
            return {"verdict": "already_checked", "interrupt": False}
        if self.last_step is not None and measurement["step"] != self.last_step + 1:
            self.flat_climb_steps = 0
        self.last_step = measurement["step"]
        out = {
            "step": measurement["step"],
            "source": measurement.get("source"),
            "checks": [],
            "interrupt": False,
            "global_route": "unknown",
            "completion": "unknown",
            "model_calls": 0,
        }
        if not measurement.get("source"):
            self.flat_climb_steps = 0
            out["checks"].append({"kind": "own_motion", "verdict": "unknown"})
            return out
        mode = intent.get("maneuver")
        dy = measurement.get("height_delta_m")
        if mode not in ("climb_up", "climb_down") or measurement.get("action") != 1:
            self.flat_climb_steps = 0
        if mode in ("climb_up", "climb_down") and measurement.get("action") == 1:
            if dy is None:
                self.flat_climb_steps = 0
                out["checks"].append({"kind": "climb_direction", "verdict": "unknown"})
            else:
                sign = 1 if mode == "climb_up" else -1
                signed = sign * dy
                if signed < -0.03:
                    out["checks"].append(
                        {
                            "kind": "climb_direction",
                            "verdict": "contradicted",
                            "evidence": "Measured height changes opposite to intended stair direction",
                            "height_delta_m": dy,
                        }
                    )
                    out["interrupt"] = True
                    self.flat_climb_steps = 0
                elif signed > 0.03:
                    out["checks"].append(
                        {
                            "kind": "climb_direction",
                            "verdict": "supported",
                            "height_delta_m": dy,
                            "limitation": "Correct vertical direction, not one-tread or ordinal completion proof",
                        }
                    )
                    self.flat_climb_steps = 0
                else:
                    self.flat_climb_steps += 1
                    out["checks"].append(
                        {
                            "kind": "climb_direction",
                            "verdict": "unknown",
                            "height_delta_m": dy,
                            "evidence": "No significant vertical progress; may still be approach/landing",
                        }
                    )
                    if self.flat_climb_steps >= 2:
                        out["checks"].append(
                            {
                                "kind": "climb_phase_review",
                                "verdict": "review",
                                "evidence": "Two measured advances without vertical progress: inspect approach/landing before extending climb",
                            }
                        )
                        out["interrupt"] = True
        delta = measurement.get("translation_xz_m")
        heading = intent.get("travel_heading_deg")
        if delta is not None and heading is not None and measurement.get("action") == 1:
            angle = math.radians(heading)
            forward = delta[0] * math.cos(angle) + delta[1] * math.sin(angle)
            lateral = -delta[0] * math.sin(angle) + delta[1] * math.cos(angle)
            length = math.hypot(*delta)
            verdict = (
                "unknown"
                if length < 0.02
                else (
                    "contradicted"
                    if forward < -0.02 or abs(lateral) > 0.15
                    else "supported"
                )
            )
            out["checks"].append(
                {
                    "kind": "local_motion_direction",
                    "verdict": verdict,
                    "forward_projection_m": round(forward, 4),
                    "lateral_m": round(lateral, 4),
                    "limitation": "Local planned segment consistency only; not distance to hidden reference route",
                }
            )
            if verdict == "contradicted":
                out["interrupt"] = True
        if measurement.get("collision") is True:
            out["checks"].append({"kind": "collision", "verdict": "observed"})
            out["interrupt"] = True
        return out


import math


class PoseStallCritic:
    def __init__(self, min_forward_m=0.075, consecutive=2):
        self.min_forward_m = min_forward_m
        self.consecutive = consecutive
        self.low_count = 0

    def check(self, action, delta_xz, heading_deg, step):
        if action != 1:
            self.low_count = 0
            return {
                "step": step,
                "blocked": False,
                "interrupt": False,
                "reason": "not a forward primitive",
                "completion": "unknown",
            }
        angle = math.radians(heading_deg)
        forward = delta_xz[0] * math.cos(angle) + delta_xz[1] * math.sin(angle)
        self.low_count = self.low_count + 1 if forward < self.min_forward_m else 0
        blocked = self.low_count >= self.consecutive
        return {
            "step": step,
            "forward_m": round(forward, 4),
            "low_motion_count": self.low_count,
            "blocked": blocked,
            "interrupt": blocked,
            "reason": (
                "measured forward motion below threshold; choose a new local route"
                if blocked
                else "motion check"
            ),
            "completion": "unknown",
            "limitation": "Blocked motion says nothing about target identity or arrival.",
        }


import copy
import math
from collections import deque
import cv2
import numpy as np

RULES = (
    {
        "id": "unknown_frame_citation",
        "feature": "unknown_frame_citation",
        "equals": True,
        "dwell": 1,
        "cooldown": 4,
        "interrupt": False,
    },
    {
        "id": "completion_after_only_turns",
        "feature": "completion_after_only_turns",
        "equals": True,
        "dwell": 1,
        "cooldown": 4,
        "interrupt": False,
    },
    {
        "id": "unscaled_ordinal_certainty",
        "feature": "unscaled_ordinal_certainty",
        "equals": True,
        "dwell": 1,
        "cooldown": 4,
        "interrupt": False,
    },
    {
        "id": "repeated_branch_without_progress",
        "feature": "repeated_branch_without_progress",
        "equals": True,
        "dwell": 2,
        "cooldown": 12,
        "interrupt": False,
    },
    {
        "id": "verified_collision_sensor",
        "feature": "collision_observed",
        "equals": True,
        "dwell": 2,
        "cooldown": 4,
        "interrupt": True,
    },
    {
        "id": "verified_metric_stall",
        "feature": "metric_stall",
        "equals": True,
        "dwell": 3,
        "cooldown": 8,
        "interrupt": True,
    },
    {
        "id": "return_visual_correspondence",
        "feature": "anchor_visual_match",
        "equals": True,
        "dwell": 2,
        "cooldown": 8,
        "interrupt": False,
    },
)
EXPLANATIONS = {
    "unknown_frame_citation": "Actor cited an unavailable frame; citation provenance is invalid, not proof of task failure.",
    "completion_after_only_turns": "A spatial crossing was claimed with only turn/STOP commands between cited observations. Review the missing crossing evidence; no metric motion is inferred.",
    "unscaled_ordinal_certainty": "Absolute riser-count or zero-climb claim lacks scaled base-height/tread measurements. It is a hypothesis; code cannot prove second stair or overshoot.",
    "repeated_branch_without_progress": "Geometric image correspondence recurred with unchanged unresolved clause states after intervening actions. Possible loop; intentional return is also possible.",
    "verified_collision_sensor": "A declared collision sensor reported consecutive contact. Stop remaining batch and let navigator choose the next action.",
    "verified_metric_stall": "A declared metric odometry source measured repeated forward displacement below its declared stall threshold. Unknown displacement never counts as zero.",
    "return_visual_correspondence": "Current image geometrically matches the stored return-anchor image. This supports visual reidentification only; metric return and correct route are still unverified.",
}


class TemporalRules:
    def __init__(self, rules=RULES):
        self.rules = copy.deepcopy(rules)
        self.states = {r["id"]: {"count": 0, "cooldown": 0} for r in rules}
        self.last = {}

    def evaluate(self, features, step, phase):
        proposals = []
        outcomes = []
        for r in self.rules:
            expected = (
                "proposal"
                if r["id"]
                in (
                    "unknown_frame_citation",
                    "completion_after_only_turns",
                    "unscaled_ordinal_certainty",
                )
                else (
                    "decision"
                    if r["id"] == "return_visual_correspondence"
                    else "primitive"
                )
            )
            if phase != expected:
                continue
            key = (r["id"], phase)
            if self.last.get(key) == step:
                continue
            self.last[key] = step
            s = self.states[r["id"]]
            v = features.get(r["feature"])

            if v is None:
                s["count"] = 0
                outcomes.append({"rule": r["id"], "result": "unknown"})
                continue
            match = v == r["equals"]
            if s["cooldown"]:
                s["cooldown"] -= 1
                s["count"] = 0
            else:
                s["count"] = s["count"] + 1 if match else 0
                if s["count"] >= r["dwell"]:
                    proposals.append(
                        {
                            "rule_id": r["id"],
                            "step": step,
                            "phase": phase,
                            "feature": r["feature"],
                            "observed_value": v,
                            "evidence": EXPLANATIONS.get(r["id"], r["feature"]),
                            "interrupt": r["interrupt"],
                            "environment_write": False,
                        }
                    )
                    s["count"] = 0
                    s["cooldown"] = r["cooldown"]
            outcomes.append(
                {
                    "rule": r["id"],
                    "result": "condition_observed" if match else "not_observed",
                }
            )
        return proposals, outcomes


def visual_correspondence(a, b):

    gray = lambda im: cv2.cvtColor(
        cv2.resize(np.asarray(im)[..., :3], (320, 240)), cv2.COLOR_RGB2GRAY
    )
    a, b = gray(a), gray(b)
    orb = cv2.ORB_create(nfeatures=700)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or min(len(ka), len(kb)) < 30:
        return {"status": "unknown", "reason": "insufficient_features"}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.72 * p[1].distance]
    if len(good) < 30:
        return {"status": "not_supported", "matches": len(good)}
    p = np.float32([ka[m.queryIdx].pt for m in good])
    q = np.float32([kb[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(p, q, cv2.RANSAC, 2.5)
    if H is None or mask is None:
        return {"status": "unknown", "reason": "geometric_fit_failed"}
    keep = mask.ravel().astype(bool)
    n = int(keep.sum())
    ratio = n / len(good)
    hull = cv2.convexHull(p[keep])
    coverage = float(cv2.contourArea(hull) / (320 * 240)) if n >= 3 else 0.0
    error = (
        float(
            np.median(
                np.linalg.norm(
                    cv2.perspectiveTransform(p[None], H)[0][keep] - q[keep], axis=1
                )
            )
        )
        if n
        else None
    )
    supported = (
        n >= 25
        and ratio >= 0.55
        and coverage >= 0.10
        and error is not None
        and error <= 2.5
    )
    return {
        "status": "supported" if supported else "not_supported",
        "inliers": n,
        "ratio": round(ratio, 3),
        "coverage": round(coverage, 3),
        "median_reprojection_px": round(error, 3) if error is not None else None,
        "limitation": "Visual correspondence only; repeated geometry, viewpoint and planar scenes can be ambiguous.",
    }


class NavigationCodeCritic:
    def __init__(self, instruction, clauses=()):
        self.instruction = instruction
        self.clause_text = {row['clause_id']: row['clause'] for row in clauses}
        self.rules = TemporalRules()
        self.actions = []
        self.anchors = {}
        self.latest = []
        self.audit = []
        self.branch_visits = {}
        self.episode_events = []
        self.active = None
        self.recovered_visual = False

    def packet(self):
        return {
            "implementation": "Python temporal feature rules + ORB/RANSAC correspondence",
            "extra_model_calls": 0,
            "proposals": copy.deepcopy(self.latest[-4:]),
            "completion": "unknown; no environment goal predicates or scaled stair geometry available",
            "route": "unknown unless backed by independent declared measurements; actor labels are hypotheses",
            "capabilities_missing": [
                "metric_odometry",
                "collision_sensor",
                "grounded_goal_predicates",
            ],
            "recovery_event": (
                {"id": self.active["id"], "state": self.active["state"]}
                if self.active
                else None
            ),
        }

    def _record(self, features, step, phase, evidence=None):
        proposals, outcomes = self.rules.evaluate(features, step, phase)
        self.latest = [p for p in self.latest if step - p["step"] <= 40]
        self.latest = (self.latest + proposals)[-8:]
        row = {
            "step": step,
            "phase": phase,
            "features": features,
            "rule_outcomes": outcomes,
            "proposals": proposals,
            "visual_evidence": evidence,
            "model_calls": 0,
        }
        self.audit.append(row)
        errors = [
            p
            for p in proposals
            if p["rule_id"]
            in (
                "verified_collision_sensor",
                "verified_metric_stall",
                "repeated_branch_without_progress",
            )
        ]
        if errors and self.active is None:
            self.active = {
                "id": len(self.episode_events),
                "state": "active",
                "start_step": step,
                "actions": [],
                "trigger_rules": [p["rule_id"] for p in errors],
                "movement_restored": False,
                "route_reentry": "unknown",
            }
            self.episode_events.append(self.active)
        if self.active and step - self.active["start_step"] >= 40:
            self.active.update(state="budget_exit_unverified", end_step=step)
            self.active = None
        return row

    def before_decision(self, frame, history, obs, return_anchor=None):
        evidence = None
        features = {}

        if return_anchor in self.anchors:
            old = self.anchors[return_anchor]["rgb"]
            matches = [
                visual_correspondence(old, obs[k])
                for k in ("rgb", "right_rgb", "rear_rgb", "left_rgb")
            ]
            evidence = {"anchor": return_anchor, "views": matches}
            features["anchor_visual_match"] = any(
                m["status"] == "supported" for m in matches
            )
        return self._record(features, frame, "decision", evidence)

    def proposal(self, out, frame, history, allowed_frames):
        events = out.get("events", [])
        known = {r["id"] for r in history} | {frame}
        bad = any(not set(e.get("frames", [])) <= known for e in events)
        crossing_words = ("pass", "cross", "enter", "exit", "through")
        unsupported_cross = False
        for e in events:
            ids = e.get("frames", [])
            if (
                e.get("state") in ("passed", "reached")
                and len(set(ids)) >= 2
                and any(w in self.clause_text.get(e['clause_id'], '').lower() for w in crossing_words)
            ):
                interval = [a for a in self.actions if min(ids) < a["step"] <= max(ids)]
                if interval and not any(a["action"] == 1 for a in interval):
                    unsupported_cross = True
        txt = (
            " ".join(e.get("observation", "") for e in events)
            + " "
            + out.get("note", "")
        )
        absolute = any(
            x in txt.lower()
            for x in (
                "zero riser",
                "zero stair",
                "no riser climbed",
                "no tread climbed",
                "full flight",
                "entire flight",
                "second tread",
                "second stair",
            )
        )
        feats = {
            "unknown_frame_citation": bad,
            "completion_after_only_turns": unsupported_cross,
            "unscaled_ordinal_certainty": (
                absolute
                if any(w in self.instruction.lower() for w in ("stair", "step"))
                else False
            ),
        }
        return self._record(feats, frame, "proposal")

    def after_primitive(self, row, obs, history, branch_anchor=None, route_events=None):
        self.actions.append(copy.deepcopy(row))
        features = {}
        metric = row.get("measured_translation_m")

        if (
            row["action"] == 1
            and row.get("metric_odometry_source")
            and metric is not None
        ):
            features["metric_stall"] = metric < 0.02
            if self.active and metric > 0.12:
                self.active["movement_restored"] = True
        if row.get("collision_sensor_source") and type(row.get("collision")) is bool:
            features["collision_observed"] = row["collision"]
        evidence = None
        if row["action"] == 1:
            candidates = [fid for fid in self.anchors if row["step"] - fid >= 12]
            if candidates:
                fid = branch_anchor if branch_anchor in candidates else candidates[-1]
                old = self.anchors[fid]
                match = visual_correspondence(old["rgb"], obs["rgb"])
                same_progress = old["progress"] == tuple(
                    (e["clause_id"], e["state"]) for e in (route_events or [])
                )
                features["repeated_branch_without_progress"] = (
                    match["status"] == "supported" and same_progress
                )
                evidence = {
                    "past_frame": fid,
                    "correspondence": match,
                    "actor_progress_unchanged": same_progress,
                    "limitation": "Revisit with unchanged actor hypothesis is a possible loop, not proof of deviation.",
                }
        if self.active:
            self.active["actions"].append(
                {"step": row["step"], "action": row["action"]}
            )
        return self._record(features, row["step"], "primitive", evidence)

    def save_observation(self, frame, obs, out):
        self.anchors[frame] = {
            "rgb": cv2.resize(np.asarray(obs["rgb"])[..., :3], (320, 240)),
            "progress": tuple((e["clause_id"], e["state"]) for e in out.get("events", [])),
            "actions": copy.deepcopy(out.get("actions", [])),
        }

    def end(self, step):
        if self.active:
            self.active.update(state="episode_end_unverified", end_step=step)
            self.active = None

    def summary(self):
        return {
            "triggers": len(self.episode_events),
            "executed_interventions": sum(
                bool(e["actions"]) for e in self.episode_events
            ),
            "motion_restored": sum(e["movement_restored"] for e in self.episode_events),
            "route_verified": 0,
            "route_reentry_claims": 0,
            "verified_events": 0,
            "unverified_events": len(self.episode_events),
            "extra_critic_model_calls": 0,
            "interpretation": "Code-only sensor/rule evidence. Unsupported goal and route semantics remain unknown.",
        }


class GoalDistanceCheckpoint:
    def __init__(self, interval=10, regression_tolerance_m=0.5):
        if type(interval) is not int or interval < 1:
            raise ValueError("interval must be a positive integer")
        if regression_tolerance_m < 0:
            raise ValueError("regression tolerance cannot be negative")
        self.interval = interval
        self.regression_tolerance_m = float(regression_tolerance_m)
        self.initial_distance_m = None
        self.initial_decision = None
        self.samples = []

    def set_initial(self, distance_m, completed_decisions=0):
        if self.initial_distance_m is None and distance_m is not None:
            self.initial_distance_m = float(distance_m)
            self.initial_decision = int(completed_decisions)
        return self.initial_distance_m

    def check(self, completed_decisions, anchor_observation):
        if completed_decisions < 1 or completed_decisions % self.interval:
            return None
        if anchor_observation.get("status") != "tracked":
            row = {
                "completed_decisions": completed_decisions,
                "status": "unknown",
                "distance_m": None,
                "regression_m": None,
                "initial_distance_m": self.initial_distance_m,
                "stop": False,
                "reason": "task endpoint has no stable RGB-D anchor",
            }
            self.samples.append(row)
            return row
        distance = float(anchor_observation["distance_xz_m"])
        self.set_initial(distance, completed_decisions)
        regression = distance - self.initial_distance_m
        stop = regression is not None and regression > self.regression_tolerance_m
        row = {
            "completed_decisions": completed_decisions,
            "status": "tracked",
            "distance_m": round(distance, 2),
            "initial_distance_m": round(self.initial_distance_m, 2),
            "regression_m": round(regression, 2),
            "baseline_decision": self.initial_decision,
            "stop": stop,
            "reason": (
                "measured task-endpoint distance increased beyond tolerance"
                if stop
                else "no checkpoint regression beyond tolerance"
            ),
        }
        self.samples.append(row)
        return row


import copy


class SemanticEvidenceAudit:
    def __init__(self, instruction):
        self.instruction = instruction
        self.stage = None
        self.latest = None
        self.records = []
        self.last_visual = None

    def context(self):
        return {
            "tracked_hypothesis_stage": self.stage,
            "last_audit": copy.deepcopy(self.latest),
            "last_visual_hypotheses": copy.deepcopy(self.last_visual),
            "authority": "Model visual hypotheses; Python structural/execution checks only. No independent semantic confirmation.",
        }

    def inspect(self, out, current_frame, visible_frames, history):
        e = out["route_evidence"]
        issues = []
        visible = set(visible_frames)

        def issue(kind, evidence, verdict="unknown"):
            issues.append({"kind": kind, "evidence": evidence, "verdict": verdict})

        def cite(frames, required, label):
            if any(type(f) is not int or f not in visible for f in frames):
                issue("invalid_frame_reference", label, "contradicted")
            if required and not frames:
                issue("missing_visual_evidence", label)

        clause = e["current_clause"]
        if not clause.strip() or clause not in self.instruction:
            issue("instruction_clause_mismatch", clause, "contradicted")
        for label in ("landmarks", "prerequisites"):
            for item in e[label]:
                cite(
                    item["frames"],
                    item["verdict"] == "confirmed",
                    label + ": " + item["description"],
                )
        cite(e["route_frames"], e["route_verdict"] != "unknown", "route judgment")
        stage = out["stage"]
        advance = self.stage is not None and stage > self.stage

        if advance or (self.stage is None and stage > 0):
            if not e["prerequisites"] or any(
                p["verdict"] != "confirmed" or not p["frames"]
                for p in e["prerequisites"]
            ):
                issue(
                    "stage_preconditions_missing",
                    "Cannot trace the claimed new stage to positive prerequisite hypotheses",
                )
        cross = e["door_crossing"]
        cross_check = {"semantic_crossing": "unknown", "executed_forward_count": None}
        if cross["verdict"] != "unknown":
            a, b = cross["before_frame"], cross["after_frame"]
            cite([f for f in (a, b) if f is not None], True, "door crossing")
            if type(a) is not int or type(b) is not int or not a < b <= current_frame:
                issue(
                    "crossing_time_order",
                    "Crossing must refer to an earlier completed interval",
                    "contradicted",
                )
            else:
                rows = {
                    r["step"]: r
                    for h in history
                    for r in h["last_execution"].get("actual", [])
                    if a < r["step"] <= b
                }
                if len(rows) != b - a:
                    issue(
                        "crossing_execution_missing",
                        "Incomplete executed interval; cannot test motion",
                    )
                else:
                    n = sum(r["action"] == 1 for r in rows.values())
                    cross_check["executed_forward_count"] = n
                    if cross["verdict"] == "confirmed" and n == 0:
                        issue(
                            "crossing_without_translation_action",
                            "Completed interval contains only turn/STOP actions",
                            "contradicted",
                        )
                    cross_check["motion_authority"] = (
                        "Executed forward is an attempt, not measured displacement or semantic crossing"
                    )
        if e["route_verdict"] == "contradicted":
            issue(
                "actor_reported_route_conflict",
                "Main navigator reports contradictory visual route evidence; not independently verified",
            )
        if not e["expected_next"].strip():
            issue(
                "missing_expected_effect", "No observable expected post-action effect"
            )
        if type(stage) is not int or stage < 0:
            issue(
                "invalid_stage", "Stage must be a nonnegative integer", "contradicted"
            )
        if not issues:
            self.stage = stage
        audit = {
            "frame": current_frame,
            "claimed_stage": stage,
            "tracked_hypothesis_stage": self.stage,
            "contract_verdict": (
                "contradicted"
                if any(x["verdict"] == "contradicted" for x in issues)
                else "unknown" if issues else "traceable_hypothesis"
            ),
            "issues": issues,
            "crossing_execution_check": cross_check,
            "semantic_route": "unknown",
            "model_calls": 0,
            "action_authority": "advisory; no forced motion or STOP veto",
            "issue_classes": sorted({x["kind"] for x in issues}),
            "false_positive": None,
            "false_negative": None,
            "scoring_note": "Requires independent annotations; unknown is not an error label",
        }
        self.last_visual = copy.deepcopy(e)
        self.latest = copy.deepcopy(audit)
        self.records.append(copy.deepcopy(audit))
        return audit
