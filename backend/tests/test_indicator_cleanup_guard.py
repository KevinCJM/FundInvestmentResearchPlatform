"""Cleanup safety tests operate ONLY on pytest temporary directories."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("indicator_cleanup", Path(__file__).resolve().parents[2] / "scripts/finalize_indicator_cleanup.py")
cleanup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cleanup)


def fixture_files(root):
    paths = [root / "backend/unused_a.py", root / "frontend/unused_b.ts"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("retired fixture\n", encoding="utf-8")
    manifest = {"schema_version": 1, "files": [{"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]}
    return paths, manifest


def test_read_only_default_and_explicit_idempotent_cleanup(tmp_path, monkeypatch):
    monkeypatch.delenv("CUSTOM_INDICATOR_DATA_DIR", raising=False)
    paths, manifest = fixture_files(tmp_path)
    assert cleanup.finalize(tmp_path, manifest) == 2
    assert all(path.is_file() for path in paths)
    assert cleanup.finalize(tmp_path, manifest, apply=True) == 2
    assert not any(path.exists() for path in paths)
    assert cleanup.finalize(tmp_path, manifest, apply=True) == 0


def test_changed_file_blocks_entire_cleanup(tmp_path, monkeypatch):
    monkeypatch.delenv("CUSTOM_INDICATOR_DATA_DIR", raising=False)
    paths, manifest = fixture_files(tmp_path)
    paths[1].write_text("new human work", encoding="utf-8")
    with pytest.raises(cleanup.CleanupBlocked):
        cleanup.finalize(tmp_path, manifest, apply=True)
    assert all(path.exists() for path in paths)


def test_saved_contract_and_unsafe_path_cannot_be_deleted(tmp_path, monkeypatch):
    monkeypatch.delenv("CUSTOM_INDICATOR_DATA_DIR", raising=False)
    paths, manifest = fixture_files(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    store = data / "custom_indicators.json"
    for retired_contract in (
        {"result_kind": "scalar_bundle"},
        {"scalar_outputs": []},
        {"output_id": None},
    ):
        store.write_text(json.dumps({"items": [retired_contract]}), encoding="utf-8")
        with pytest.raises(cleanup.CleanupBlocked):
            cleanup.finalize(tmp_path, manifest, apply=True)
        assert all(path.exists() for path in paths)
    assert store.exists()
    manifest["files"][0]["path"] = "../outside.py"
    with pytest.raises(cleanup.CleanupBlocked):
        cleanup.planned_files(tmp_path, manifest)
