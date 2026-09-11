from __future__ import annotations

import json
from pathlib import Path

from controlflow.core.state import atomic_write_json, sha256_file


def test_atomic_json_is_canonical(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"b": 2, "a": 1})
    assert target.read_text(encoding="utf-8") == '{"a":1,"b":2}\n'
    assert sha256_file(target) == "e8d38819d39f705646bfb643368eca78f7db476c16471dbc33b941b27326410d"


def test_atomic_json_replaces_existing(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"old": True})
    atomic_write_json(target, {"new": True})
    assert json.loads(target.read_text()) == {"new": True}
    assert not list(tmp_path.glob("*.tmp"))
