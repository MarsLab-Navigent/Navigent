import json
import os
from pathlib import Path
import subprocess
import sys
import time
from PIL import Image

from eval.storage import CampaignHold, immutable, read
from eval.runtime import RGBBackend as BaseBackend


def response_envelope(stdout):
    matches = []
    for line in stdout.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and ('output' in item or 'error' in item):
            matches.append(item)
    if len(matches) != 1:
        raise ValueError('Expected exactly one adapter response envelope')
    return matches[0]


class ModelBackend(BaseBackend):
    def role_request(self, role, body, directory, images=None):
        return super().role_request(role, body, directory, images)

    def request(self, module, payload, directory, timeout=600):
        if payload.get("role") == "periodic_route_review":
            return self._codex_request(
                payload, directory, timeout, "eval/model_reviewer.py"
            )
        if module != "eval/model.py":
            return super().request(module, payload, directory, timeout)
        if self.config.get("pose_without_map"):
            payload = dict(payload)
            payload["frames"] = [
                frame
                for frame in payload.get("frames", [])
                if "OBSERVED RGB-D MAP" not in frame["label"]
            ]
            payload["context"] = dict(payload["context"])
            observed = payload["context"].pop("observed_semantic_map", {})
            payload["context"]["own_pose"] = observed.get("current_pose")
            payload["context"]["visited_nodes"] = [
                {
                    "id": node["id"],
                    "links": node.get("links", []),
                    "return_walked_m": node.get("return_walked_m"),
                    "turn_deg_left": node.get("turn_deg_left"),
                    "pose_from_start_m": node.get("pose_from_start_m"),
                    "return_anchor_frame": node.get("return_anchor_frame"),
                    "semantic_hypothesis": node.get("semantic_hypothesis"),
                }
                for node in observed.get("nodes", [])
            ]
            payload["pose_without_map"] = True
        if self.config.get("no_semantic_map"):
            payload = dict(payload, no_semantic_map=True)
        if self.config.get("execution_feedback_candidate"):
            payload = dict(payload, execution_feedback_candidate=True)
        if self.config.get("adaptive_batching"):
            payload = dict(payload, adaptive_batching=True)
        return self._codex_request(payload, directory, timeout, "eval/model.py")

    def _codex_request(self, payload, directory, timeout, adapter):
        for attempt in range(2):
            response = self._request_once(payload, directory, timeout, adapter)
            if not response.get('error') and response.get('output'):
                return response
            if not response.get('retryable_format') or attempt:
                raise CampaignHold(str(response.get('error') or 'Missing adapter output'))

    def _request_once(self, payload, directory, timeout, adapter):
        directory = Path(directory)
        payload = dict(payload, provider=self.config.get("provider", "codex"))
        immutable(directory / "input.json", payload)
        response_path = directory / "response_0.json"
        try:
            cached = read(response_path) if response_path.exists() else None
        except ValueError:
            cached = None  # The original bytes are archived below.
        if (
            isinstance(cached, dict)
            and not cached.get("error")
            and isinstance(cached.get("output"), dict)
            and cached["output"]
            and (payload.get('role') != 'navigate' or (cached.get('session_id') and
                 cached['session_id'] == (payload.get('session_id') or cached['session_id'])))
        ):
            response = cached
        else:
            if response_path.exists():
                archived = directory / f"response_0_error_{time.time_ns()}.json"
                response_path.rename(archived)
            root = Path(__file__).resolve().parents[1]
            sdk = os.environ.get("NAV_MODEL_PYTHON") or sys.executable
            env = os.environ.copy()
            env["PYTHONPATH"] = str(root)
            env.setdefault("CODEX_HOME", str(Path.home() / ".codex"))
            process = None
            try:
                process = subprocess.run(
                    [sdk, str(root / adapter)],
                    input=json.dumps(payload),
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    env=env,
                )
                response = response_envelope(process.stdout)
                if process.returncode or not isinstance(response, dict):
                    raise ValueError(
                        "Adapter did not return a successful response object"
                    )
                if not response.get("error") and not isinstance(
                    response.get("output"), dict
                ):
                    raise ValueError("Adapter output must be an object")
            except (OSError, subprocess.TimeoutExpired, IndexError, ValueError) as exc:
                label = 'TIMEOUT' if isinstance(exc, subprocess.TimeoutExpired) else 'ADAPTER_ERROR'
                print(f'[{label}] adapter={adapter} detail={exc}', file=sys.stderr, flush=True)
                response = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "usage_unknown": True,
                    'retryable_format': isinstance(exc, ValueError) and process is not None and process.returncode == 0,
                }
            if process is not None and (response.get('error') or not response.get('output')):
                response.update(adapter_stdout=process.stdout, adapter_stderr=process.stderr,
                                adapter_exit=process.returncode)
            from eval.storage import write

            write(response_path, response)
        return response


import copy

import eval.runtime as runtime


class RGBBackend(ModelBackend):
    def _codex_request(self, payload, directory, timeout, adapter):
        if adapter == "eval/model.py":
            payload = dict(payload, horizon_mode=self.config["horizon_mode"])
        return super()._codex_request(payload, directory, timeout, adapter)

    def evaluate(self, version, task, directory):
        if self.config["horizon_mode"] == "free":
            return super().evaluate(version, task, directory)
        original = runtime.validate_action

        def horizon_validate(
            output, instruction, available_frames=None, history_frames=None
        ):
            if output.get("maneuver") not in ("climb_up", "climb_down"):
                return original(output, instruction, available_frames, history_frames)

            ordinary = copy.deepcopy(output)
            ordinary["maneuver"] = "travel"
            original(ordinary, instruction, available_frames, history_frames)
            return output

        runtime.validate_action = horizon_validate
        try:
            return super().evaluate(version, task, directory)
        finally:
            runtime.validate_action = original
