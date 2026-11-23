import copy


def validate_action(out, instruction, available_frames=None, history_frames=None):
    actions = out["actions"]
    mode = out["maneuver"]
    if not isinstance(actions, list) or len(actions) > 3:
        raise ValueError("Maximum three commands")
    if out["skill"] == "stop":
        if actions:
            raise ValueError("STOP cannot contain movement")
    elif out["skill"] == "move":
        if not actions and out.get('map_operation') != 'navigate_to_node':
            raise ValueError("No empty/zero observation action")
    else:
        raise ValueError("Unknown skill")
    for a in actions:
        kind = a.get("type")
        if kind in ("turn_left", "turn_right"):
            if (
                set(a) != {"type", "degrees"}
                or type(a["degrees"]) is not int
                or a["degrees"] not in (15, 30, 45, 60, 75, 90)
            ):
                raise ValueError("Turn must be 15..90deg in 15deg increments")
        elif kind == "forward":
            if (
                set(a) != {"type", "centimeters"}
                or type(a["centimeters"]) is not int
                or a["centimeters"] not in (25, 50, 75, 100)
            ):
                raise ValueError("Forward must be 25..100cm in 25cm increments")
        else:
            raise ValueError("Unknown command")
    forward = [a for a in actions if a["type"] == "forward"]
    if actions and not forward:
        clause = out.get("instruction_clause", "")
        if (
            mode != "instruction_orientation"
            or not clause.strip()
            or clause not in instruction
        ):
            raise ValueError(
                "No panoramic scan: pure turn requires original orientation goal"
            )
    if (
        mode in ("climb_up", "climb_down")
        and actions
        and not out.get("evidence_frames")
    ):
        raise ValueError("Climb needs RGB evidence frames")
    anchor = out.get("return_anchor_frame")
    if (
        history_frames is not None
        and anchor is not None
        and (type(anchor) is not int or anchor not in history_frames)
    ):
        raise ValueError("Return anchor must be an actual earlier frame")
    if mode == "return_to_anchor" and anchor is None:
        raise ValueError("Return needs an observed anchor")
    if mode == "return_to_anchor" and (
        out.get('map_operation') != 'navigate_to_node' or type(out.get('map_node_id')) is not int
    ):
        raise ValueError('return_to_anchor must use navigate_to_node with a visited node ID')
    if out.get('map_operation') == 'navigate_to_node' and (
        mode != 'return_to_anchor' or out['skill'] != 'move' or actions != []
    ):
        raise ValueError('Node return requires move, return_to_anchor and actions=[]; executor supplies primitives')
    if available_frames is not None and not set(out.get("evidence_frames", [])) <= set(
        available_frames
    ):
        raise ValueError("Unknown evidence frame")
    return out


def primitives(out):
    if out["skill"] == "stop":
        return [0]
    seq = []
    for a in out["actions"]:
        if a["type"] == "forward":
            seq.extend([1] * (a["centimeters"] // 25))
        else:
            seq.extend([2 if a["type"] == "turn_left" else 3] * (a["degrees"] // 15))
    return seq


import re

_SEGMENT = re.compile(
    r"\bgo past (?P<landmark>[^,.;]+),\s*turn (?P<direction>left|right) "
    r"at (?P<trigger>[^,.;]+?) and (?P<finish>wait|stop) there\b",
    re.IGNORECASE,
)


def compound_segment(instruction):
    match = _SEGMENT.search(instruction)
    if match is None:
        return None
    return {
        "text": match.group(0),
        "landmark": match.group("landmark").strip(),
        "trigger": match.group("trigger").strip(),
        "turn": match.group("direction").lower(),
        "finish": match.group("finish").lower(),
        "relation": "The door must be identified in relation to passing the landmark. "
        "Keep candidate doors unresolved until their route relation is checked.",
    }


def instruction_constraints(instruction):

    turns = []
    for match in re.finditer(r"\bturn\s+(left|right)\b", instruction, re.IGNORECASE):
        turns.append(
            {
                "order": len(turns),
                "direction": match.group(1).lower(),
                "source": instruction[match.start() : match.end()],
            }
        )
    completion = re.search(r"\b(wait|stop)\s+there\b", instruction, re.IGNORECASE)
    segment = compound_segment(instruction)
    return {
        "source": "original instruction; directional words are immutable",
        "explicit_turns_in_order": turns,
        "linked_landmark_turn": segment,
        "completion": completion.group(0) if completion else None,
        "unknowns": "which visible object matches an instruction noun; when a turn is due",
        "authority": "Visual evidence may update object identity and progress, not rewrite an explicit turn.",
    }


def validate_compound_plan(output, context):
    compound = context.get("compound_route")
    if compound is None:
        return
    assessment = output["compound_route_assessment"]
    choice = output["route_choice"]
    if assessment["intended_final_turn"] != compound["turn"]:
        raise ValueError(
            "The final turn must retain the original instruction direction"
        )
    visible = set(context["available_frame_ids"])
    if not set(choice.get("evidence_frames", [])) <= visible:
        raise ValueError("Route choice cites a frame not supplied to this decision")
    if choice["role"] in ("switch_branch", "finish") and not choice.get(
        "evidence_frames"
    ):
        raise ValueError("Route switch or finish needs a supplied panorama")
    if choice["role"] == "switch_branch" and choice["route_support"] != "observed":
        raise ValueError(
            "A branch switch needs positive route evidence, not candidate uncertainty"
        )
    if choice["role"] == "finish" and assessment["completion"] != "satisfied":
        raise ValueError(
            "Cannot finish while the linked landmark-door relation is unresolved"
        )
    if output["skill"] == "stop" and assessment["completion"] != "satisfied":
        raise ValueError(
            "STOP needs an explicit judgment that the linked relation holds"
        )


def pending_route_review(latest, addressed=False):
    if latest is None or addressed:
        return None
    if latest.get('decision') == 'continue':
        return None
    if latest.get('decision') not in ('inspect_required', 'backtrack', 'execution_blocked'):
        raise ValueError('Invalid review decision')
    pending = {
        key: latest.get(key)
        for key in (
            "decision",
            "stage_progress",
            "evidence",
            "prediction_errors",
            "frames",
            "suggested_return_anchor_frame",
            "next_check",
            "checked_before_decision",
            "trigger_type",
        )
    }
    return pending


def repeated_prediction_errors(findings):

    cues = (
        "repeated",
        "consecutive",
        "three times",
        "four times",
        "five times",
        "six times",
        "three decisions",
        "four decisions",
    )
    return any(any(cue in str(item).lower() for cue in cues) for item in findings)


def validate_review_resolution(review, response, output, allowed_frames, current_frame):
    resolution = response["resolution"]
    cited = set(response["evidence_frames"])
    if not cited <= set(allowed_frames):
        raise ValueError(
            "Review response cites a frame not currently supplied or retrieved"
        )
    if not response["current_test"].strip():
        raise ValueError("Review response must state the current visual test")
    if review is None:
        if resolution != "not_applicable":
            raise ValueError("No route review is pending; use not_applicable")
        if output.get('maneuver') == 'return_to_anchor':
            route = output.get('route_evidence', {})
            if route.get('route_verdict') != 'contradicted' or current_frame not in route.get('route_frames', []):
                raise ValueError('Return requires current-frame evidence of wrong direction, not a stall')
        return
    severity = review['decision']
    if resolution == "not_applicable":
        raise ValueError(
            "A route-review warning is pending and must be explicitly resolved"
        )
    unfinished_return = review.get("node_return_in_progress")
    if unfinished_return is not None:
        if (
            resolution != "backtrack_to_node"
            or response["map_node_id"] != unfinished_return["node_id"]
        ):
            raise ValueError(
                "Unfinished map-node return must continue to the same visited node"
            )
    if output["skill"] == "stop" and resolution != "continue_with_counterevidence":
        raise ValueError("Cannot STOP before resolving the pending route warning")
    if severity == 'backtrack' and resolution != 'backtrack_to_node':
        raise ValueError('Confirmed direction error requires recorded-node return')
    if severity in ('inspect_required', 'execution_blocked') and resolution == 'backtrack_to_node':
        route = output.get('route_evidence', {})
        if (route.get('route_verdict') != 'contradicted' or current_frame not in cited
                or current_frame not in route.get('route_frames', [])):
            raise ValueError('Inspection or blockage alone cannot trigger return; cite current route contradiction')
    operation = output["map_operation"]
    if resolution == "backtrack_to_node":
        if (
            operation != "navigate_to_node"
            or output["map_node_id"] != response["map_node_id"]
        ):
            raise ValueError(
                "Backtrack response must navigate to its selected visited map node"
            )
    elif operation == "navigate_to_node":
        raise ValueError("Map-node navigation must be declared as backtrack_to_node")
    if resolution == "inspect_suspected_branch" and (
        output["skill"] != "move" or not output["actions"]
    ):
        raise ValueError(
            "Branch inspection requires an observation-producing movement action"
        )
    if resolution == "inspect_suspected_branch":
        inspection_count = int(review.get("inspection_count", 0))
        if inspection_count >= 2:
            raise ValueError(
                "A pending review allows at most two inspections; "
                "the third response must backtrack or cite current-frame counterevidence"
            )
        forwards = [
            action["centimeters"]
            for action in output["actions"]
            if action["type"] == "forward"
        ]
        if not forwards or sum(forwards) > 50:
            raise ValueError("Review branch inspection must use at most 50 cm forward")
        previous = review.get("last_inspection")
        if previous and previous["planned_primitives"] == primitives(output):
            raise ValueError(
                "Branch inspection must change the verifiable viewpoint; "
                "repeat movement is not allowed"
            )
        if previous and not previous['viewpoint_changed']:
            def approach(sequence):
                turns = sequence[:sequence.index(1)]
                return turns.count(2) - turns.count(3)

            if abs(approach(primitives(output)) - approach(previous['planned_primitives'])) < 2:
                raise ValueError('Ineffective inspection: change the approach by at least 30 degrees before moving forward')
    if resolution == "continue_with_counterevidence":
        if severity == "backtrack" or not cited:
            raise ValueError(
                "Continuing requires a non-confirmed warning and supplied counterevidence frames"
            )
        if current_frame not in cited:
            raise ValueError(
                "Continuing after a review requires current-frame counterevidence"
            )
        route = output.get('route_evidence', {})
        if review.get('inspection_count', 0) >= 2 and (
            route.get('route_verdict') != 'confirmed' or current_frame not in route.get('route_frames', [])
        ):
            raise ValueError('After two inspections, continuing requires confirmed route evidence citing the current frame')


def review_remains_open(review, response):

    return review is not None and response["resolution"] == "inspect_suspected_branch"


def validate_stop_clauses(output):
    if output.get("skill") != "stop":
        return
    events = output.get("events") or []
    if not events:
        raise ValueError("STOP requires the ordered instruction clauses")
    for event in events[:-1]:
        if event.get("state") not in ("passed", "reached"):
            raise ValueError(
                "STOP blocked: earlier instruction clause remains unresolved: "
                + str(event.get("clause_id", "unknown"))
            )


import copy

HORIZONS = {"4_to_3": (4, 3), "3_to_2": (3, 2)}


def crop_horizon(output, mode):
    if not isinstance(output, dict):
        raise ValueError("Navigation output must be an object")
    if mode not in HORIZONS:
        raise ValueError("Unknown action horizon")
    predicted_count, executed_count = HORIZONS[mode]
    planned = output.get("actions")
    if output.get('map_operation') == 'navigate_to_node':
        if output.get('skill') != 'move' or output.get('maneuver') != 'return_to_anchor' or planned != []:
            raise ValueError('Recorded-node return requires move, return_to_anchor and actions=[]')
        return copy.deepcopy(output), {'mode': mode, 'planned': [], 'executed': [], 'executor': 'recorded_path'}
    if output.get("skill") == "stop":
        if planned != []:
            raise ValueError("STOP must have no predicted or executed movement")
        return copy.deepcopy(output), {"mode": mode, "planned": [], "executed": []}
    if output.get("skill") != "move" or not isinstance(planned, list):
        raise ValueError("MOVE needs a predicted action list")
    if len(planned) != predicted_count:
        raise ValueError(
            f"{mode} requires exactly {predicted_count} predicted commands"
        )
    for action in planned:
        if not isinstance(action, dict):
            raise ValueError("Each predicted command must be an object")
        if action.get("type") == "forward":
            if action.get("centimeters") not in (25, 50, 75, 100):
                raise ValueError("Forward distance must be 25, 50, 75 or 100 cm")
        elif action.get("type") in ("turn_left", "turn_right"):
            if action.get("degrees") not in (15, 30, 45, 60, 75, 90):
                raise ValueError("Turn must be in 15 degree increments up to 90")
        else:
            raise ValueError("Unknown predicted command")
    executed = planned[:executed_count]
    if not any(action["type"] == "forward" for action in executed):
        raise ValueError("Executed prefix must contain a forward command")
    if (
        sum(action["centimeters"] for action in executed if action["type"] == "forward")
        > 150
    ):
        raise ValueError("Executed prefix cannot request more than 150 cm forward")
    cropped = copy.deepcopy(output)
    cropped["actions"] = copy.deepcopy(executed)
    return cropped, {
        "mode": mode,
        "planned": copy.deepcopy(planned),
        "executed": copy.deepcopy(executed),
    }
