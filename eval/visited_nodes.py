"""Visited-node memory and return along recorded movement."""

import math
import numpy as np


def _derive_places(path, spacing_m=2.5, snap_frac=0.6, jump_m=0.5, max_nodes=64):

    if not path:
        return {"nodes": [], "edges": [], "current": None}

    def xz(a, b):
        if abs(float(a[1]) - float(b[1])) > 0.6:
            return float("inf")
        return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))

    spacing = float(spacing_m)
    snap = float(snap_frac) * spacing
    jump = float(jump_m)

    nodes = [
        {"id": 0, "position": list(map(float, path[0])), "visits": 1, "path_index": 0}
    ]
    edges = {}
    current = 0
    left_index = 0
    arc = 0.0
    clean = True

    def link(a, b, length, span):
        key = (a, b) if a < b else (b, a)
        prev = edges.get(key)

        if prev is None or length < prev["arc_m"]:
            edges[key] = {"arc_m": round(float(length), 3), "span": list(span)}

    for i in range(1, len(path)):
        seg = xz(path[i - 1], path[i])
        arc += seg
        if seg > jump:
            clean = False

        best, best_d = None, None
        for n in nodes:
            d = xz(path[i], n["position"])
            if best_d is None or d < best_d:
                best, best_d = n, d

        if best_d <= snap and best["id"] != current:
            if clean:
                link(current, best["id"], arc, (left_index, i))
            best["visits"] += 1
            current, left_index, arc, clean = best["id"], i, 0.0, True
        elif best_d > spacing and len(nodes) < int(max_nodes):
            new = {
                "id": len(nodes),
                "position": list(map(float, path[i])),
                "visits": 1,
                "path_index": i,
            }
            nodes.append(new)
            if clean:
                link(current, new["id"], arc, (left_index, i))
            current, left_index, arc, clean = new["id"], i, 0.0, True

    return {
        "nodes": nodes,
        "edges": [{"a": a, "b": b, **v} for (a, b), v in sorted(edges.items())],
        "current": current,
    }


class VisitedNodes:
    def __init__(self, sim):
        self.sim = sim
        self.path = []
        self.nodes = []
        self.edges = []

    def record_pose(self):
        pos = np.asarray(self.sim.get_agent_state().position, dtype=float)
        if not self.path or np.linalg.norm(pos - self.path[-1]) > 0.01:
            self.path.append(pos.copy())
            annotations = {
                n["id"]: (n["name"], n["caption"], n["annotations"], n.get("anchor"))
                for n in self.nodes
            }
            graph = _derive_places(self.path)
            self.nodes = graph["nodes"]
            self.edges = graph["edges"]
            for node in self.nodes:
                node["name"], node["caption"], node["annotations"], node["anchor"] = (
                    annotations.get(node["id"], ("", "", [], None))
                )

    def annotate_node(
        self, node_id, name, caption="", clause=None, frame=None, image=None
    ):
        if type(node_id) is not int or not 0 <= node_id < len(self.nodes):
            raise ValueError("node must already exist")
        if not name.strip():
            raise ValueError("name required")
        node = self.nodes[node_id]
        node["name"] = name[:80]
        node["caption"] = caption[:250]
        pose = self.sim.get_agent_state()
        import quaternion

        forward = quaternion.as_rotation_matrix(pose.rotation) @ np.array(
            [0.0, 0.0, -1.0]
        )
        anchor = {
            "position": np.asarray(pose.position, dtype=float).round(4).tolist(),
            "heading_xz_deg": round(
                math.degrees(math.atan2(forward[2], forward[0])), 3
            ),
            "path_index": len(self.path) - 1,
            "frame_id": frame,
            "image": image,
            "instruction_stage_hypothesis": clause,
        }
        node["anchor"] = anchor
        node["annotations"].append(
            {
                "name": node["name"],
                "caption": node["caption"],
                "path_index": len(self.path) - 1,
                "anchor": anchor,
            }
        )

    def table(self):
        here = np.asarray(self.sim.get_agent_state().position, dtype=float)
        import quaternion

        forward = quaternion.as_rotation_matrix(
            self.sim.get_agent_state().rotation
        ) @ np.array([0.0, 0.0, -1.0])
        facing = math.atan2(float(forward[2]), float(forward[0]))
        out = []
        for n in self.nodes:
            anchor = n.get("anchor")
            target = np.asarray(
                anchor["position"] if anchor else n["position"], dtype=float
            )
            delta = target[[0, 2]] - here[[0, 2]]
            bearing = math.atan2(float(delta[1]), float(delta[0]))
            turn = (math.degrees(bearing - facing) + 180) % 360 - 180
            links = sorted(
                e["b"] if e["a"] == n["id"] else e["a"]
                for e in self.edges
                if n["id"] in (e["a"], e["b"])
            )
            path_index = anchor["path_index"] if anchor else n["path_index"]
            walked = sum(
                float(np.linalg.norm(a[[0, 2]] - b[[0, 2]]))
                for a, b in zip(self.path[path_index:], self.path[path_index + 1 :])
            )
            out.append(
                {
                    "id": n["id"],
                    "xy_m": [
                        round(float(v), 2) for v in np.asarray(n["position"])[[0, 2]]
                    ],
                    "xy_from_start_m": [
                        round(float(v), 2)
                        for v in (np.asarray(n["position"]) - self.path[0])[[0, 2]]
                    ],
                    "name": n["name"],
                    "caption": n["caption"],
                    "visits": n["visits"],
                    "links": links,
                    "straight_m": round(float(np.linalg.norm(delta)), 2),
                    "return_walked_m": round(walked, 2),
                    "turn_deg_left": round(-turn, 1),
                    "path_index": n["path_index"],
                    "annotation_anchor": anchor,
                    "annotations": n["annotations"],
                }
            )
        return out

    def context(self):
        pose = self.sim.get_agent_state()
        import quaternion

        direction = quaternion.as_rotation_matrix(pose.rotation) @ np.array(
            [0.0, 0.0, -1.0]
        )
        return {
            "current_pose": {
                "xyz_from_start_m": (np.asarray(pose.position) - self.path[0])
                .round(3)
                .tolist(),
                "heading_xz_deg": round(
                    math.degrees(math.atan2(direction[2], direction[0])), 1
                ),
            },
            "nodes": self.table(),
            "edges": self.edges,
            "nearest_place": min(self.table(), key=lambda n: n["straight_m"])["id"],
            "unnamed_places": [n["id"] for n in self.table() if not n["name"]],
            "topology": "NavGPT-3 derived place graph; edges are observed traversals",
            "return_walked_m": "length of recorded reverse path to the annotation anchor, or node if unnamed; not a shortest path",
            "turn_deg_left": "relative turn to face the anchor, positive left",
            "unknown_space": "never assumed free",
        }

    def return_target(self, node_id):
        if type(node_id) is not int or not 0 <= node_id < len(self.nodes):
            raise ValueError("node must already exist")
        node = self.nodes[node_id]
        return np.asarray(
            node["anchor"]["position"] if node.get("anchor") else node["position"],
            dtype=float,
        ).copy()

    def return_path_index(self, node_id, anchor_path_index=None):
        target = self.return_target(node_id)
        if anchor_path_index is not None:
            if (type(anchor_path_index) is not int
                or not 0 <= anchor_path_index < len(self.path)
                or np.linalg.norm(self.path[anchor_path_index] - target) > 0.5):
                raise ValueError("Return frame does not identify a recorded visit to this node")
            return anchor_path_index
        node = self.nodes[node_id]
        return (node.get("anchor") or node)["path_index"]

    def actions_to(self, node_id, cap=None, *, anchor_path_index=None):
        stop = self.return_path_index(node_id, anchor_path_index)
        points = self.path[stop:][::-1]
        count = 0
        for point in points:
            while cap is None or count < cap:
                pose = self.sim.get_agent_state()
                here = np.asarray(pose.position, dtype=float)
                delta = np.asarray(point, dtype=float) - here
                planar = float(np.linalg.norm(delta[[0, 2]]))
                if planar < 0.20:
                    break
                import quaternion

                forward = quaternion.as_rotation_matrix(pose.rotation) @ np.array(
                    [0.0, 0.0, -1.0]
                )
                facing = math.atan2(float(forward[2]), float(forward[0]))
                bearing = math.atan2(float(delta[2]), float(delta[0]))
                turn = (math.degrees(bearing - facing) + 180) % 360 - 180
                action = 2 if turn < -7.5 else 3 if turn > 7.5 else 1
                count += 1
                yield action
