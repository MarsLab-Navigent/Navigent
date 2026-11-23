import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from eval.prompts import REVIEW_PROMPT as PROMPT, REVIEW_SCHEMA as SCHEMA
from eval.model import cli_command


def review(payload, timeout=600):
    try:
        if payload.get("role") != "periodic_route_review":
            raise ValueError("Unexpected review role")
        with tempfile.TemporaryDirectory(prefix="vln_review_") as directory:
            directory = Path(directory)
            content = payload["content"]
            prompt = (
                PROMPT
                + "\nReturn one JSON object, no Markdown, matching this schema:\n"
            )
            prompt += json.dumps(SCHEMA, separators=(",", ":"))
            images = []
            for item in content:
                if item["type"] == "text":
                    prompt += "\n" + item["text"]
                elif item["type"] == "image":
                    path = directory / f"image_{len(images):02d}.jpg"
                    path.write_bytes(
                        base64.b64decode(item["source"]["data"], validate=True)
                    )
                    images.append(path)
            output = directory / "output.json"
            command = cli_command({}, images, output)
            env = os.environ.copy()
            env.setdefault("CODEX_HOME", str(Path.home() / ".codex"))
            body = next(
                (
                    json.loads(item["text"])
                    for item in content
                    if item["type"] == "text" and item["text"].startswith("{")
                ),
                None,
            )
            supplied = set((body or {}).get("available_frame_ids") or [])
            attempts = []
            for attempt in range(2):
                output.unlink(missing_ok=True)
                process = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    cwd=directory,
                    env=env,
                )
                raw = output.read_text() if output.exists() else ""
                try:
                    if process.returncode or not raw:
                        raise ValueError(
                            f"Codex reviewer exit {process.returncode}: {process.stderr[-300:]}"
                        )
                    verdict = json.loads(raw)
                    if not isinstance(verdict, dict):
                        raise ValueError("Reviewer output must be an object")
                    missing = set(SCHEMA["required"]) - set(verdict)
                    if missing:
                        raise ValueError(
                            "Missing reviewer fields: " + ", ".join(sorted(missing))
                        )
                    if verdict.get('decision') not in SCHEMA['properties']['decision']['enum']:
                        raise ValueError('Invalid review decision')
                    if not isinstance(verdict.get('frames'), list) or any(type(f) is not int for f in verdict['frames']):
                        raise ValueError('Review frames must be integer IDs')
                    for key in ('evidence', 'next_check', 'task_summary', 'stage_progress'):
                        if not isinstance(verdict.get(key), str) or not verdict[key].strip():
                            raise ValueError('Review needs a concrete ' + key)
                    verdict = {key: verdict[key] for key in SCHEMA["required"]}
                    if not set(verdict["frames"]) <= supplied:
                        raise ValueError(
                            "Reviewer cited a frame outside supplied images"
                        )
                    if not verdict['frames']:
                        raise ValueError('Review must cite observed frames')
                    anchor = verdict["suggested_return_anchor_frame"]
                    if anchor is not None and type(anchor) is not int:
                        raise ValueError('Return anchor must be an integer frame ID')
                    if verdict['decision'] == 'backtrack':
                        if anchor is None or len(set(verdict['frames'])) < 2 or not verdict['prediction_errors']:
                            raise ValueError('Backtrack requires an anchor, before/after evidence and an explicit contradiction')
                    elif anchor is not None:
                        raise ValueError('Only backtrack may suggest a return anchor')
                    anchors = set((body or {}).get('return_anchor_frames', []))
                    if anchor is not None and anchor not in anchors:
                        raise ValueError('Reviewer return anchor must belong to a supplied visited node')
                    return {
                        "output": verdict,
                        "usage": {},
                        "cost_usd": None,
                        "format_repairs": len(attempts),
                        "attempts": attempts,
                    }
                except (TypeError, ValueError, KeyError) as exc:
                    attempts.append({"error": str(exc), "raw_output": raw,
                                     "stdout": process.stdout, "stderr": process.stderr})
                    prompt += (
                        "\nThe previous response did not match the requested JSON contract: "
                        + str(exc)
                        + ". Return the complete object with exactly the required "
                        "fields, using only supplied frame IDs. No Markdown."
                    )
            return {
                "error": "Reviewer failed the output contract after one repair",
                "attempts": attempts,
                "usage_unknown": True,
            }
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        label = "TIMEOUT" if isinstance(exc, subprocess.TimeoutExpired) else "REVIEW_ERROR"
        print(f"[{label}] detail={exc}", file=sys.stderr, flush=True)
        return {"error": f"{type(exc).__name__}: {exc}", "usage_unknown": True}


if __name__ == "__main__":
    print(json.dumps(review(json.load(sys.stdin))))
