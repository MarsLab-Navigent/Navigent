import cv2
import numpy as np


class RGBPose:
    def __init__(self):
        self.commanded_yaw = 0
        self.visual_yaw = 0.0
        self.unknown_translations = 0
        self.last = None

    def update(self, before, after, action):
        self.commanded_yaw = (
            self.commanded_yaw
            + (15 if action == 2 else -15 if action == 3 else 0)
            + 180
        ) % 360 - 180
        result = {
            "rotation_matrix": None,
            "translation_direction_camera": None,
            "translation_scale": "unknown",
            "translation_m": None,
            "quality": "unknown",
            "source": "monocular RGB feature tracking + essential matrix RANSAC",
        }
        if action == 0:
            return result
        a = cv2.cvtColor(np.asarray(before["rgb"])[..., :3], cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(np.asarray(after["rgb"])[..., :3], cv2.COLOR_RGB2GRAY)
        p = cv2.goodFeaturesToTrack(a, 800, 0.01, 8)
        if p is not None and len(p) >= 30:
            q, valid, _ = cv2.calcOpticalFlowPyrLK(
                a, b, p, None, winSize=(31, 31), maxLevel=4
            )
            if q is not None:
                back, ok, _ = cv2.calcOpticalFlowPyrLK(
                    b, a, q, None, winSize=(31, 31), maxLevel=4
                )
                if back is not None:
                    keep = (
                        valid.ravel().astype(bool)
                        & ok.ravel().astype(bool)
                        & (np.linalg.norm(back - p, axis=2).ravel() < 1.0)
                    )
                    p0 = p.reshape(-1, 2)[keep]
                    p1 = q.reshape(-1, 2)[keep]
                    if len(p0) >= 30:
                        flow = float(np.median(np.linalg.norm(p1 - p0, axis=1)))
                        result.update(
                            tracked_features=len(p0), median_pixel_flow=round(flow, 3)
                        )
                        h, w = a.shape
                        f = w / 2
                        K = np.array(
                            [[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]],
                            dtype=float,
                        )

                        if action == 1 and flow >= 0.7:
                            E, mask = cv2.findEssentialMat(
                                p0, p1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
                            )
                            if E is not None and E.shape == (3, 3):
                                n, R, t, inliers = cv2.recoverPose(
                                    E, p0, p1, K, mask=mask
                                )
                                ratio = n / len(p0)
                                if n >= 25 and ratio >= 0.45:
                                    displacement = (-R.T @ t).ravel()
                                    yaw = float(
                                        np.degrees(np.arctan2(R[0, 2], R[2, 2]))
                                    )
                                    self.visual_yaw += yaw
                                    result.update(
                                        rotation_matrix=R.round(5).tolist(),
                                        translation_direction_camera=displacement.round(
                                            5
                                        ).tolist(),
                                        inliers=int(n),
                                        inlier_ratio=round(ratio, 3),
                                        visual_relative_yaw_deg=round(yaw, 3),
                                        quality="relative_direction_only",
                                    )
                        elif action == 1 and flow < 0.7:
                            result["quality"] = "low_parallax_translation_unknown"
        if action == 1:
            self.unknown_translations += 1
        self.last = result
        return result

    def context(self):
        return {
            "source": "RGB-only relative pose estimates, no simulator pose or depth",
            "commanded_heading_deg": self.commanded_yaw,
            "visual_yaw_accumulated_deg": round(self.visual_yaw, 3),
            "last_relative_pose": self.last,
            "metric_position_m": None,
            "metric_scale": "unknown",
            "unknown_translation_scale_intervals": self.unknown_translations,
            "limitations": "Monocular scale, pure rotation, texture and parallax ambiguity. Commanded turn is not measured yaw. No metric trajectory inferred from nominal step length.",
        }


import math, numpy as np


class OwnPoseMap:
    def __init__(self, sim):
        self.sim = sim
        self.origin = np.array(sim.get_agent_state().position)
        self.rows = []
        self.record(0)

    def record(self, step):
        s = self.sim.get_agent_state()
        p = np.array(s.position)
        q = s.rotation
        u = np.array([q.x, q.y, q.z])
        v = np.array([0.0, 0.0, -1.0])
        t = 2 * np.cross(u, v)
        d = v + q.w * t + np.cross(u, t)
        row = {
            "step": step,
            "xyz_m": (p - self.origin).round(4).tolist(),
            "heading_xz_deg": round(math.degrees(math.atan2(d[2], d[0])), 3),
        }
        if not self.rows or self.rows[-1]["step"] != step:
            self.rows.append(row)
        return row

    def context(self, anchor=None):
        current = self.rows[-1]
        a = next((r for r in self.rows if r["step"] == anchor), None)
        selected = (
            self.rows
            if len(self.rows) <= 80
            else [self.rows[round(i * (len(self.rows) - 1) / 79)] for i in range(80)]
        )
        return {
            "source": "simulator GT own pose, origin at episode start; x/z ground plane,y height",
            "current": current,
            "cumulative_map": {
                "type": "executed trajectory and visited points, NOT occupancy/free-space map",
                "points": selected,
                "total_recorded_points": len(self.rows),
                "unvisited_space": "unknown",
                "connections": "chronological executed trail only; no guaranteed reverse passability",
            },
            "return_anchor": a,
            "distance_to_anchor_m": (
                round(float(np.linalg.norm(np.array(current["xyz_m"]) - a["xyz_m"])), 4)
                if a
                else None
            ),
            "limitations": "Own localization does not prove correct room, target completion or stair ordinal. No goal/reference/navmesh access.",
        }

    def history_summary(self, decision_frames, limit=8):
        by_step = {row["step"]: row for row in self.rows}
        frames = [step for step in decision_frames if step in by_step]
        selected = [0] + frames[-(limit - 2) :] + [self.rows[-1]["step"]]
        selected = list(dict.fromkeys(selected))
        rows = [by_step[step] for step in selected]
        return {
            "source": "simulator own pose; episode-relative meters, no target or map",
            "points": [
                {
                    "frame": row["step"],
                    "xyz_m": row["xyz_m"],
                    "heading_xz_deg": row["heading_xz_deg"],
                }
                for row in rows
            ],
            "meaning": "chronological visited poses; straight lines do not imply traversable shortcuts",
        }


"""Conservative near-field depth warning for the next two forward primitives."""

import numpy as np


def forward_clearance(observation, distance_m=0.5, radius_m=0.18):
    depth = observation.get("rgb_map_depth")
    if depth is None:
        return {"status": "unknown", "reason": "front depth unavailable"}
    pixels = np.asarray(depth, dtype=float).squeeze()
    if pixels.ndim != 2 or min(pixels.shape) < 16:
        return {"status": "unknown", "reason": "invalid front depth shape"}
    height, width = pixels.shape
    # The 120-degree camera sees a narrow forward body corridor near its center.
    crop = pixels[
        int(0.38 * height) : int(0.62 * height), int(0.43 * width) : int(0.57 * width)
    ]
    valid = np.isfinite(crop) & (crop > 0.02) & (crop < 10.0)
    valid_fraction = float(valid.mean())
    if valid_fraction < 0.6:
        return {
            "status": "unknown",
            "reason": "insufficient valid depth",
            "valid_fraction": round(valid_fraction, 3),
        }
    clearance = float(np.quantile(crop[valid], 0.25))
    # This is an advisory view check. The horizontal camera cannot certify
    # low steps, glass, or traversability of an unseen floor patch.
    warning = clearance < distance_m + radius_m
    return {
        "status": "possible_obstacle" if warning else "no_near_obstacle_seen",
        "clearance_m": round(clearance, 3),
        "lookahead_m": distance_m,
        "valid_fraction": round(valid_fraction, 3),
        "reason": (
            "front depth corridor within two forward steps"
            if warning
            else "no near surface in sampled corridor; floor remains unverified"
        ),
    }


import math

import numpy as np

VIEWS = {
    "front": (0.0, ""),
    "right": (-90.0, "right_"),
    "rear": (180.0, "rear_"),
    "left": (90.0, "left_"),
}


class TargetAnchorCritic:
    def __init__(self):
        self.anchor = None
        self.last = {"status": "unknown", "reason": "no target bound"}

    def bind(self, target, obs, sim, frame):
        if (
            target["status"] not in ("seen", "approaching")
            or target["point_norm"] is None
        ):
            return self.last
        view = target["view"]
        u, v = target["point_norm"]
        depth = np.asarray(obs[VIEWS[view][1] + "rgb_map_depth"]).squeeze()
        h, w = depth.shape

        x_ray = (2 * u - 1) * math.tan(math.radians(90) / 2)
        y_ray = (1 - 2 * v) * math.tan(math.radians(90) / 2)
        du = (x_ray / math.tan(math.radians(120) / 2) + 1) * w / 2
        dv = (1 - y_ray / math.tan(math.radians(120) / 2)) * h / 2
        cx, cy = int(du), int(dv)
        patch = depth[max(0, cy - 3) : min(h, cy + 4), max(0, cx - 3) : min(w, cx + 4)]
        valid = patch[np.isfinite(patch) & (patch > 0.1) & (patch < 10)]
        if valid.size < 9 or np.percentile(valid, 75) - np.percentile(valid, 25) > 0.6:
            self.last = {"status": "unknown", "reason": "unstable target depth"}
            return self.last
        distance = float(np.median(valid))
        ray = np.array([x_ray, y_ray, -1.0]) * distance
        angle = math.radians(VIEWS[view][0])
        c, s = math.cos(angle), math.sin(angle)
        ray = np.array([c * ray[0] + s * ray[2], ray[1], -s * ray[0] + c * ray[2]])
        pose = sim.get_agent_state()
        world = np.asarray(pose.position, dtype=float) + np.array([0, 1.25, 0])
        world += __import__("quaternion").as_rotation_matrix(pose.rotation) @ ray
        self.anchor = {
            "xyz": world,
            "frame": frame,
            "view": view,
            "description": target["description"],
            "source_depth_m": round(distance, 2),
        }
        return self.observe(sim)

    def observe(self, sim):
        if self.anchor is None:
            return self.last
        pose = sim.get_agent_state()
        delta = (
            self.anchor["xyz"]
            - np.asarray(pose.position, dtype=float)
            - np.array([0, 1.25, 0])
        )
        local = __import__("quaternion").as_rotation_matrix(pose.rotation).T @ delta
        distance = float(np.linalg.norm(delta[[0, 2]]))
        bearing = math.degrees(math.atan2(local[0], -local[2]))
        self.last = {
            "status": "tracked",
            "source_frame": self.anchor["frame"],
            "description_hypothesis": self.anchor["description"],
            "distance_xz_m": round(distance, 2),
            "bearing_deg_right": round(bearing, 1),
            "source_depth_m": self.anchor["source_depth_m"],
            "limitation": "Depth ray may hit a surface behind an opening; geometry cannot verify doorway identity or crossing.",
        }
        return self.last

    def after_primitive(self, sim, action, previous, expected):
        now = self.observe(sim)
        if now["status"] != "tracked" or previous.get("status") != "tracked":
            return {"interrupt": False, "target": now, "verdict": "unknown"}
        delta = now["distance_xz_m"] - previous["distance_xz_m"]
        turned_behind = (
            abs(previous["bearing_deg_right"]) < 90
            and abs(now["bearing_deg_right"]) > 110
        )
        moving_away = action == 1 and expected == "decrease" and delta > 0.35
        interrupt = turned_behind or moving_away
        return {
            "interrupt": interrupt,
            "target": now,
            "verdict": (
                "possibly_passed"
                if turned_behind
                else "inconsistent" if moving_away else "unknown"
            ),
            "measured_distance_change_m": round(delta, 2),
            "reason": (
                "anchor moved behind"
                if turned_behind
                else (
                    "measured distance increased"
                    if moving_away
                    else "no decisive geometric conflict"
                )
            ),
        }


import numpy as np


def return_arrival(target_position, current_position, radius_m=0.5):
    delta = np.asarray(target_position, dtype=float) - np.asarray(current_position, dtype=float)
    distance = float(np.linalg.norm(delta[[0, 2]]))
    return {
        "arrived": float(np.linalg.norm(delta)) <= radius_m,
        "distance_3d_m": round(float(np.linalg.norm(delta)), 3),
        "height_error_m": round(abs(float(delta[1])), 3),
        "distance_xz_m": round(distance, 3),
        "radius_m": radius_m,
        "source": "measured own simulator pose",
        "semantic_route": "unknown",
    }
