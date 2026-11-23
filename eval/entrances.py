import numpy as np

from eval.geometry import TargetAnchorCritic

RELEVANCE = {"unlikely": 0, "uncertain": 1, "potentially_relevant": 2}


class EntranceGraph:
    def __init__(self):
        self.candidates = []
        self.next_id = 0

    def observe_candidates(self, proposals, obs, sim, frame, place_id):
        for proposal in proposals:
            box = proposal["bbox_norm"]
            tracker = TargetAnchorCritic()
            if box is not None:
                point = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                tracker.bind(
                    {
                        "description": proposal["description"],
                        "view": proposal["view"],
                        "point_norm": point,
                        "status": "seen",
                    },
                    obs,
                    sim,
                    frame,
                )
            match = None
            if tracker.anchor is not None:
                for item in reversed(self.candidates):
                    anchor = item["tracker"].anchor
                    if (
                        anchor is not None
                        and np.linalg.norm(
                            (anchor["xyz"] - tracker.anchor["xyz"])[[0, 2]]
                        )
                        < 1.25
                    ):
                        match = item
                        break
            if match is None:
                match = {
                    "id": f"entrance_{self.next_id:02d}",
                    "first_seen_frame": frame,
                    "from_visited_node": place_id,
                    "tracker": tracker,
                    "status": "unverified" if tracker.anchor is not None else "unknown",
                    "last_seen_frame": frame,
                    "active": False,
                    "away_m": 0.0,
                    "alerted": False,
                    "best_relevance": 0,
                }
                self.next_id += 1
                self.candidates.append(match)
            elif tracker.anchor is not None:
                match["tracker"] = tracker
                if match["status"] == "unknown":
                    match["status"] = "unverified"
            match.update(
                description_hypothesis=proposal["description"],
                destination_hypothesis=proposal["possible_destination"],
                instruction_relevance=proposal["instruction_relevance"],
                last_seen_frame=frame,
            )
            match["best_relevance"] = max(
                match["best_relevance"], RELEVANCE[proposal["instruction_relevance"]]
            )
            measured = match["tracker"].observe(sim)
            if measured["status"] == "tracked":
                match["last_distance_m"] = measured["distance_xz_m"]
                match["last_bearing_deg"] = measured["bearing_deg_right"]
        self._select_active(sim, frame)

    def _select_active(self, sim, frame):
        def rank(item):
            measured = item["tracker"].observe(sim)
            distance = measured.get("distance_xz_m", float("inf"))
            return (
                distance <= 2.5,
                item["status"] == "possibly_passed",
                item["best_relevance"],
                item["last_seen_frame"],
            )

        ordered = sorted(self.candidates, key=rank, reverse=True)
        selected = ordered[:2]
        new_near = [
            item
            for item in ordered
            if item not in selected
            and item["last_seen_frame"] == frame
            and item["tracker"].observe(sim).get("distance_xz_m", float("inf")) <= 2.5
        ]
        third = (
            new_near[0]
            if new_near
            else next((item for item in ordered if item not in selected), None)
        )
        if third is not None:
            selected.append(third)
        chosen = {item["id"] for item in selected}
        for item in self.candidates:
            item["active"] = item["id"] in chosen

    def after_primitive(self, sim, action, translation_m):
        alerts = []
        for item in self.candidates:
            if not item["active"] or item["tracker"].anchor is None:
                continue
            measured = item["tracker"].observe(sim)
            if measured["status"] != "tracked":
                continue
            previous = item.get("last_distance_m")
            before_bearing = item.get("last_bearing_deg")
            now_distance = measured["distance_xz_m"]
            item["last_distance_m"] = now_distance
            item["last_bearing_deg"] = measured["bearing_deg_right"]
            if (
                action != 1
                or translation_m < 0.08
                or previous is None
                or item["alerted"]
            ):
                continue
            if min(previous, now_distance) > 2.5 or measured["source_depth_m"] > 4.5:
                continue
            delta = now_distance - previous
            item["away_m"] = max(0.0, item["away_m"] + delta)
            passed = (
                before_bearing is not None
                and abs(before_bearing) < 85
                and abs(measured["bearing_deg_right"]) > 110
            )
            threshold = 0.7 if item["best_relevance"] == 0 else 0.45
            if passed or item["away_m"] >= threshold:
                item["status"] = "possibly_passed"
                item["alerted"] = True
                alerts.append(
                    {
                        "candidate_id": item["id"],
                        "first_seen_frame": item["first_seen_frame"],
                        "measured_away_m": round(item["away_m"], 2),
                        "reason": "A nearby unconfirmed entrance may have been passed; its semantic identity remains unknown.",
                    }
                )
        return alerts

    def context(self, sim, active_only=True):
        rows = []
        selected = [
            item for item in self.candidates if item["active"] or not active_only
        ]
        for item in selected:
            rows.append(
                {
                    "candidate_id": item["id"],
                    "from_visited_node": item["from_visited_node"],
                    "first_seen_frame": item["first_seen_frame"],
                    "description_hypothesis": item["description_hypothesis"],
                    "destination_hypothesis": item["destination_hypothesis"],
                    "instruction_relevance": item["instruction_relevance"],
                    "status": item["status"],
                    "active": item["active"],
                    "geometry": item["tracker"].observe(sim),
                }
            )
        return {
            "predicted_entrances": rows,
            "deferred_count": len(self.candidates)
            - sum(i["active"] for i in self.candidates),
            "confirmed_edges": "Only observed_semantic_map.edges records walked connections",
            "meaning": "Candidate identity and traversability are unverified model hypotheses.",
        }


def route_progress_feedback(previous, execution, current_graph):
    if not previous:
        return None
    output = previous["output"]
    choice = output.get("route_choice") or {}
    candidate_id = choice.get("target_candidate_id")
    old_graph = (previous.get("context") or {}).get("predicted_semantic_graph") or {}

    def candidate(graph):
        return next(
            (
                row
                for row in graph.get("predicted_entrances", [])
                if row["candidate_id"] == candidate_id
            ),
            None,
        )

    old = candidate(old_graph) if candidate_id else None
    current = candidate(current_graph) if candidate_id else None
    old_geometry = (old or {}).get("geometry") or {}
    new_geometry = (current or {}).get("geometry") or {}
    distance_change = None
    if (
        old_geometry.get("status") == "tracked"
        and new_geometry.get("status") == "tracked"
        and old_geometry.get("source_depth_m", 99) <= 4.5
        and new_geometry.get("source_depth_m", 99) <= 4.5
    ):
        distance_change = round(
            new_geometry["distance_xz_m"] - old_geometry["distance_xz_m"], 2
        )
    trend = (
        "unknown"
        if distance_change is None
        else (
            "toward"
            if distance_change < -0.2
            else "away" if distance_change > 0.2 else "stable"
        )
    )
    actions = execution.get("actual", [])
    measured_translation = round(
        sum(row.get("measured_translation_m", 0) or 0 for row in actions), 2
    )
    expected = (output.get("expected_effect") or {}).get(
        "expected_distance_change", "unknown"
    )
    comparison = (
        "unknown"
        if trend == "unknown" or expected == "unknown"
        else (
            "supported"
            if expected == "decrease" and trend == "toward"
            else (
                "contradicted"
                if expected == "decrease" and trend == "away"
                else "unknown"
            )
        )
    )
    return {
        "previous_choice": choice.get("role"),
        "candidate_id": candidate_id,
        "expected_distance_change": expected,
        "measured_distance_change_m": distance_change,
        "measured_trend": trend,
        "expected_effect_comparison": comparison,
        "measured_translation_m": measured_translation,
        "candidate_status": (current or {}).get("status", "unknown"),
        "execution_interrupted": bool(execution.get("critic_interrupted")),
        "limitation": "Geometry tests movement relative to a provisional anchor; it cannot identify the instructed door or prove route completion.",
    }
