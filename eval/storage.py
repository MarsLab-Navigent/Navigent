"""JSON conversion and atomic artifact storage."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    return value


def write(path, value):
    path = Path(path)
    payload = json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


import hashlib
import json
from pathlib import Path


class CampaignHold(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(
        json.dumps(jsonable(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def immutable(path, value):
    path = Path(path)
    value = jsonable(value)
    if path.exists() and read(path) != value:
        raise ValueError("Saved run identity differs: " + str(path))
    write(path, value)
