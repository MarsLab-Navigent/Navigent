"""Habitat dataset registration for the supported R2R and RxR episode files."""

import gzip
import json
import os
from typing import List, Optional

from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode


@registry.register_dataset(name="RxRVLN-v1")
class RxRVLNDatasetV1(Dataset):
    """Load RxR guide/follower episodes without a vocabulary sidecar."""

    episodes: List[VLNEpisode]

    @staticmethod
    def check_config_paths_exist(config) -> bool:
        data_path = config.data_path.format(split=config.split)
        return os.path.exists(data_path) and os.path.exists(config.scenes_dir)

    def __init__(self, config: Optional[object] = None) -> None:
        self.episodes = []
        if config is None:
            return
        path = config.data_path.format(split=config.split)
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as stream:
            rows = json.load(stream)["episodes"]
        for raw in rows:
            episode = dict(raw)
            episode.pop("info", None)
            episode.setdefault("trajectory_id", episode["episode_id"])
            episode["instruction"] = InstructionData(
                instruction_text=episode["instruction"]["instruction_text"],
                instruction_tokens=episode["instruction"].get("instruction_tokens"),
            )
            episode["goals"] = [NavigationGoal(**goal) for goal in episode["goals"]]
            item = VLNEpisode(**episode)
            if item.scene_id.startswith("data/scene_datasets/"):
                item.scene_id = item.scene_id[len("data/scene_datasets/") :]
            if config.scenes_dir:
                item.scene_id = os.path.join(config.scenes_dir, item.scene_id)
            self.episodes.append(item)
        self.episodes = list(filter(self.build_content_scenes_filter(config), self.episodes))


def register(dataset_kind):
    """Register the selected dataset using one public Habitat entry point.

    R2R uses Habitat's maintained loader because its files include the
    instruction vocabulary. RxR uses the local loader because guide/follower
    files omit that vocabulary and carry a different instruction shape.
    """
    if dataset_kind == "r2r":
        try:
            from habitat.datasets.vln.r2r_vln_dataset import VLNDatasetV1  # noqa: F401
        except (ImportError, ValueError) as exc:
            raise RuntimeError(
                "The installed Habitat-Lab must provide the R2R VLN dataset loader"
            ) from exc
    elif dataset_kind != "rxr":
        raise ValueError("Unsupported dataset kind: " + str(dataset_kind))
