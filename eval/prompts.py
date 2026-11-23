"""Load the navigation and review contracts shipped with the evaluator."""

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "config"
PROMPT = (CONFIG / "navigation_prompt.txt").read_text(encoding="utf-8")
NAV_SCHEMA = json.loads((CONFIG / "navigation_schema.json").read_text(encoding="utf-8"))
REVIEW_PROMPT = (CONFIG / "review_prompt.txt").read_text(encoding="utf-8")
REVIEW_SCHEMA = json.loads((CONFIG / "review_schema.json").read_text(encoding="utf-8"))
