import copy
import json
import math
from pathlib import Path
import cv2
import numpy as np


def sensor_rig():
    """Read the camera calibration and view orientations used by Habitat."""
    path = Path(__file__).resolve().parents[1] / "config/sensor_rig.json"
    return json.loads(path.read_text(encoding="utf-8"))


def create_rgb_env(
    split,
    gpu,
    map_depth=False,
    horizontal_only=False,
    *,
    data_path=None,
    scenes_dir=None,
    max_steps=401,
    dataset_kind="rxr",
):
    from eval.cli import require_habitat

    require_habitat()
    from eval import habitat_datasets
    habitat_datasets.register(dataset_kind)
    from eval.cli import ROOT, Env, get_config, habitat
    from habitat.core.registry import registry
    from habitat.sims.habitat_simulator.habitat_simulator import (
        HabitatSimRGBSensor,
        HabitatSimDepthSensor,
    )

    config = get_config(str(ROOT / "config/habitat.yaml"))
    rig = sensor_rig()
    with habitat.config.read_write(config):
        config.habitat.dataset.split = split
        config.habitat.dataset.data_path = str(data_path)
        config.habitat.dataset.scenes_dir = str(scenes_dir)
        config.habitat.dataset.type = "RxRVLN-v1" if dataset_kind == "rxr" else "R2RVLN-v1"
        config.habitat.environment.max_episode_steps = max_steps
        sim = config.habitat.simulator
        sim.habitat_sim_v0.gpu_device_id = gpu
        if sim.forward_step_size != 0.25 or sim.turn_angle != 15:
            raise ValueError(
                "The action contract requires 0.25 m forward and 15 degree turns"
            )
        sensors = sim.agents.main_agent.sim_sensors
        template = copy.deepcopy(sensors.rgb_sensor)
        depth_template = copy.deepcopy(sensors.depth_sensor)
        for key in list(sensors.keys()):
            del sensors[key]
        for key in ("width", "height", "hfov"):
            setattr(template, key, rig["rgb"][key])
        template.position = [0.0, rig["camera_height_m"], 0.0]
        depth_template.position = list(template.position)
        views = [
            (row["uuid"], math.radians(row["pitch_deg"]), math.radians(row["yaw_deg"]))
            for row in rig["horizontal_views"] + rig["downward_views"]
        ]
        if horizontal_only:
            views = views[:4]
        for uuid, pitch, yaw in views:
            name = "NavigationRGB" + uuid.title().replace("_", "")
            if registry.get_sensor(name) is None:
                cls = type(
                    name,
                    (HabitatSimRGBSensor,),
                    {"_get_uuid": lambda self, *args, value=uuid, **kw: value},
                )
                registry.register_sensor(name=name)(cls)
            sensor = copy.deepcopy(template)
            sensor.type = name
            sensor.orientation = [pitch, yaw, 0.0]
            sensors[uuid + "_sensor"] = sensor
        if map_depth:
            for key in ("width", "height", "hfov", "min_depth", "max_depth"):
                setattr(depth_template, key, rig["geometry"][key])
            depth_template.normalize_depth = False
            for uuid, _, yaw in views[:4]:
                color_name = "NavigationMapColor" + uuid.title().replace("_", "")
                if registry.get_sensor(color_name) is None:
                    cls = type(
                        color_name,
                        (HabitatSimRGBSensor,),
                        {
                            "_get_uuid": lambda self, *args, value=uuid + "_map_color", **kw: value
                        },
                    )
                    registry.register_sensor(name=color_name)(cls)
                color_sensor = copy.deepcopy(template)
                for key in ("width", "height", "hfov"):
                    setattr(color_sensor, key, rig["geometry"][key])
                color_sensor.type = color_name
                color_sensor.orientation = [0, yaw, 0]
                sensors[uuid + "_map_color_sensor"] = color_sensor
                name = "NavigationMapDepth" + uuid.title().replace("_", "")
                if registry.get_sensor(name) is None:
                    cls = type(
                        name,
                        (HabitatSimDepthSensor,),
                        {
                            "_get_uuid": lambda self, *args, value=uuid + "_map_depth", **kw: value
                        },
                    )
                    registry.register_sensor(name=name)(cls)
                sensor = copy.deepcopy(depth_template)
                sensor.type = name
                sensor.orientation = [0, yaw, 0]
                sensors[uuid + "_map_depth_sensor"] = sensor
    return Env(config)


def views(obs, horizontal_only=False):
    result = []
    for floor in ((False,) if horizontal_only else (False, True)):
        tiles = []
        for label, key in [
            ("FRONT", "rgb"),
            ("RIGHT", "right_rgb"),
            ("REAR", "rear_rgb"),
            ("LEFT", "left_rgb"),
        ]:
            key = "floor_" + key if floor else key
            tile = cv2.resize(np.asarray(obs[key])[..., :3], (384, 288)).copy()
            cv2.putText(
                tile,
                label + (" DOWN45" if floor else ""),
                (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                4,
            )
            cv2.putText(
                tile,
                label + (" DOWN45" if floor else ""),
                (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            tiles.append(tile)
        result.append(
            (
                "FLOOR4" if floor else "PANORAMA_FRONT_RIGHT_REAR_LEFT",
                np.concatenate(
                    [np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0
                ),
            )
        )
    return result


def visual_change(before, after):

    scores = []
    for key in ("rgb", "right_rgb", "rear_rgb", "left_rgb"):
        a = cv2.resize(np.asarray(before[key])[..., :3], (160, 120)).astype(float)
        b = cv2.resize(np.asarray(after[key])[..., :3], (160, 120)).astype(float)
        scores.append(float(np.mean(np.abs(a - b))))
    return round(float(np.median(scores)), 3)


def integrate(before, after, action, step):
    score = visual_change(before, after) if action == 1 else None
    return {
        "action": action,
        "step": step,
        "estimated_move_m": None,
        "executed_turn_deg": 15 if action == 2 else -15 if action == 3 else 0,
        "rgb_change": score,
        "translation_status": "unknown; RGB appearance change is not metric motion",
    }


class EvidenceLedger:
    def __init__(self, instruction):
        import re
        self.clauses = [dict(clause_id=i, clause=text.strip(), status='unverified', frames=[], completed=[])
                        for i, text in enumerate(filter(str.strip, re.split(
                            r'[,;.!?\u3002\uff0c\uff1b]|\b(?:and then|then)\b', instruction, flags=re.I)), 1)]
        self._by_id = {row['clause_id']: row for row in self.clauses}

    def validate_events(self, events):
        for event in events:
            clause_id = event.get('clause_id')
            if type(clause_id) is not int or clause_id not in self._by_id:
                raise ValueError('Event must reference a valid clause_id')

    def update(self, events, allowed):
        self.validate_events(events)
        for event in events:
            row = self._by_id[event['clause_id']]
            frames = set(event['frames']) & allowed
            if not frames:
                continue
            if event['state'] == 'contradicted':
                row.update(status='contradicted', completed=[], frames=sorted(frames))
                continue
            if event['state'] not in ('reached', 'passed', 'seen'):
                continue
            if row['status'] == 'contradicted':
                row['frames'] = []
            row['frames'] = sorted(set(row['frames']) | frames)
            if event['state'] in ('reached', 'passed'):
                row['completed'] = sorted(set(row['completed']) | frames)
            row['status'] = 'candidate'
        self._order()

    def _order(self):
        prior = True
        for row in self.clauses:
            if row['status'] in ('verified', 'candidate'):
                row['status'] = 'verified' if prior and len(row['completed']) >= 2 else 'candidate'
            prior = prior and row['status'] == 'verified'

    def invalidate(self, excluded):
        for row in self.clauses:
            if excluded.intersection(row['frames']):
                row['frames'] = sorted(set(row['frames']) - excluded)
                row['completed'] = sorted(set(row['completed']) - excluded)
                row['status'] = 'candidate' if row['frames'] else 'contradicted'
        self._order()

    def snapshot(self):
        return copy.deepcopy(self.clauses)
