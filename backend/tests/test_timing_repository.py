import json
from pathlib import Path

import numpy as np
import pytest

from backend.custom_indicators.errors import ConflictError, NotFoundError, ValidationError
from backend.timing_research.repository import TimingRepository, _array_hash, content_hash


@pytest.fixture
def repository(tmp_path):
    return TimingRepository(tmp_path)


def inputs():
    owner = np.arange(60, dtype=np.float64).reshape(10, 6)
    owner.setflags(write=False)
    return {"dates": np.arange(10, dtype=np.int64), "close": owner[:, 0],
            "path": owner, "trades": np.empty((0, 10), dtype=np.float64)}


def result(code="510300.SH", status="ok"):
    return {"product_id": code, "status": status, "summary": {"all": {"total_return": .1}},
            "curve": [{"date": str(i), "close": float(i)} for i in range(10)],
            "trades": [], "channels": [{"id": "close", "values": list(range(10))}],
            "lineage": {"source_hash": "sha256:fixture"}}


def completed(repository):
    run_id = repository.create_run({"name": "可解释择时", "nodes": []},
                                   {"targets": [{"product_id": "510300.SH"}, {"product_id": "510050.SH"}]})
    repository.save_product(run_id, "510300.SH", result(), inputs())
    repository.save_product(run_id, "510050.SH", result("510050.SH"), inputs())
    return repository.finish_run(run_id, {"execution": {"backend": "numba"}, "restrictions": ["仅研究"]})


def test_definition_revisions_are_optimistic_and_historical(repository):
    original = repository.save_definition({"name": "策略1", "nodes": []})
    latest = repository.save_definition({"name": "策略2", "nodes": []}, original["id"], 1)
    assert latest["revision"] == 2
    assert repository.get_definition(original["id"], 1)["name"] == "策略1"
    assert repository.get_definition(original["id"])["name"] == "策略2"
    assert len(repository.list_definitions()) == 1
    with pytest.raises(ConflictError, match="已更新"):
        repository.save_definition({"name": "冲突"}, original["id"], 1)
    with pytest.raises(NotFoundError):
        repository.get_definition(original["id"], 9)


def test_per_product_files_and_first_successful_detail_only(repository):
    run = completed(repository)
    assert len(run["products"][0]["curve"]) == 10
    assert "curve" not in run["products"][1]
    assert run["products"][1]["detail_loaded"] is False
    folder = repository.root / "runs" / run["id"]
    master = json.loads((folder / "manifest.json").read_text())["payload"]
    assert all("curve" not in product and "channels" not in product for product in master["products"])
    assert (folder / "510300.SH.npz").is_file()
    assert (folder / "510050.SH.json").is_file()
    page = repository.get_product(run["id"], "510050.SH", 2, 3)
    assert [p["close"] for p in page["curve"]] == [2., 3., 4.]
    assert page["channels"][0]["values"] == [2, 3, 4]
    assert page["pagination"]["totals"]["curve"] == 10
    listing = repository.list_runs()
    assert listing["total"] == 1
    assert all("curve" not in item for item in listing["items"][0]["products"])


def test_input_hash_strided_readonly_and_empty_2d_arrays(repository):
    arrays = inputs()
    before = {key: value.copy() for key, value in arrays.items()}
    assert _array_hash(arrays) == _array_hash({key: np.ascontiguousarray(value) for key, value in arrays.items()})
    run_id = repository.create_run({"name": "strides"}, {})
    repository.save_product(run_id, "510300.SH", result(), arrays)
    repository.finish_run(run_id)
    restored = repository.load_arrays(run_id, "510300.SH")
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])
        np.testing.assert_array_equal(restored[key], before[key])
        assert not restored[key].flags.writeable
    assert not arrays["close"].flags.writeable
    assert np.shares_memory(arrays["close"], arrays["path"])


def test_interruption_after_npz_preserves_inputs_and_retry_recovers(repository, monkeypatch):
    run_id = repository.create_run({"name": "recover"}, {})
    original = repository._write

    def fail_product(path, payload, **kwargs):
        if path.name == "510300.SH.json":
            raise OSError("interrupted after input snapshot")
        return original(path, payload, **kwargs)

    monkeypatch.setattr(repository, "_write", fail_product)
    with pytest.raises(OSError):
        repository.save_product(run_id, "510300.SH", result(), inputs())
    array_path = repository.root / "runs" / run_id / "510300.SH.npz"
    unchanged = array_path.read_bytes()
    monkeypatch.setattr(repository, "_write", original)
    repository.save_product(run_id, "510300.SH", result(), inputs())
    assert array_path.read_bytes() == unchanged
    reloaded = TimingRepository(repository.root.parent)
    assert reloaded.finish_run(run_id)["products"][0]["status"] == "ok"


def test_frozen_product_run_and_inputs_cannot_be_overwritten(repository):
    run_id = repository.create_run({"name": "freeze"}, {})
    repository.save_product(run_id, "510300.SH", result(), inputs())
    repository.save_product(run_id, "510300.SH", result(), inputs())  # Idempotent recovery.
    changed = inputs()
    changed["dates"] = changed["dates"] + 1
    with pytest.raises(ConflictError):
        repository.save_product(run_id, "510300.SH", result(), changed)
    different_result = {**result(), "status": "error"}
    with pytest.raises(ConflictError):
        repository.save_product(run_id, "510300.SH", different_result)
    repository.finish_run(run_id)
    with pytest.raises(ConflictError):
        repository.save_product(run_id, "510300.SH", result(), inputs())


def test_json_and_input_corruption_fail_closed(repository):
    run = completed(repository)
    directory = repository.root / "runs" / run["id"]
    path = directory / "510300.SH.npz"
    with path.open("wb") as handle:
        arrays = inputs()
        arrays["dates"] = arrays["dates"] + 2
        np.savez(handle, **arrays)
    with pytest.raises(ValidationError, match="校验和"):
        repository.load_arrays(run["id"], "510300.SH")
    product_path = directory / "510300.SH.json"
    product = json.loads(product_path.read_text())
    product["payload"]["summary"] = {}
    product_path.write_text(json.dumps(product))
    with pytest.raises(ValidationError, match="校验失败"):
        repository.get_product(run["id"], "510300.SH")


def test_resealed_product_cannot_bypass_run_manifest(repository):
    run = completed(repository)
    path = repository.root / "runs" / run["id"] / "510300.SH.json"
    item = json.loads(path.read_text())
    item["payload"]["summary"] = {}
    item["sha256"] = content_hash(item["payload"])
    path.write_text(json.dumps(item))
    with pytest.raises(ValidationError, match="运行清单"):
        repository.load_arrays(run["id"], "510300.SH")


def test_release_is_immutable_research_evidence_and_binding_is_idempotent(repository):
    run = completed(repository)
    release = repository.create_release(run["id"], note="保留验证结果")
    assert release["usage"] == "research_only" and release["execution_authorized"] is False
    assert release["definition_hash"] == run["definition_hash"]
    assert release["request_hash"] == run["request_hash"]
    assert release["products"][0]["data_hash"] == "sha256:fixture"
    binding = repository.create_binding(release["id"], "product_research", "research")
    assert repository.create_binding(release["id"], "product_research", "research")["id"] == binding["id"]
    assert binding["context"] == "product_research"
    assert len(repository.list_releases()) == 1
    assert len(repository.list_bindings("product_research", "research")) == 1
    with pytest.raises(ValidationError):
        repository.create_binding(release["id"], "live_trading")


def test_new_binding_revalidates_inputs_after_release(repository):
    run = completed(repository)
    release = repository.create_release(run["id"])
    path = repository.root / "runs" / run["id"] / "510300.SH.npz"
    path.write_bytes(b"corrupt after release")
    with pytest.raises(ValidationError, match="冻结输入损坏"):
        repository.create_binding(release["id"], "pre_investment")
    assert not repository.list_bindings()


def test_failed_products_do_not_become_release_or_first_detail(repository):
    run_id = repository.create_run({"name": "errors"}, {"targets": [{"product_id": "510050.SH"}, {"product_id": "510300.SH"}]})
    repository.save_product(run_id, "510050.SH", {"status": "error", "error": "缺少行情"})
    repository.save_product(run_id, "510300.SH", result(), inputs())
    run = repository.finish_run(run_id)
    assert run["products"][0]["detail_loaded"] is False
    assert run["products"][1]["detail_loaded"] is True
    only_failed = repository.create_run({"name": "all failed"}, {})
    repository.save_product(only_failed, "510050.SH", {"status": "error"})
    repository.finish_run(only_failed)
    with pytest.raises(ValidationError, match="成功"):
        repository.create_release(only_failed)


@pytest.mark.parametrize("bad", ["../outside", "/tmp/outside", "timing-run-../../x", "", "510300.SH/../x"])
def test_object_ids_reject_path_traversal(repository, bad):
    with pytest.raises(ValidationError):
        repository.get_run(bad)
    with pytest.raises(ValidationError):
        repository.get_definition(bad)


def test_symlink_storage_and_invalid_arrays_rejected(repository, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    repository.root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValidationError, match="符号链接"):
        repository.create_run({"name": "bad"}, {})
    assert list(outside.iterdir()) == []
    repository.root.unlink()
    with pytest.raises(ValidationError):
        _array_hash({"price": np.array(["1"], dtype=object)})
    with pytest.raises(ValidationError):
        _array_hash({"price": np.ones(12001)})
    with pytest.raises(ValidationError):
        _array_hash({"../price": np.ones(3)})


def test_missing_requested_product_cannot_finalize_and_summary_cannot_change_frozen_inputs(repository):
    run_id = repository.create_run({"name": "original"}, {"targets": [{"product_id": "510300.SH"}]})
    with pytest.raises(NotFoundError):
        repository.finish_run(run_id)
    repository.save_product(run_id, "510300.SH", result(), inputs())
    run = repository.finish_run(run_id, {"definition_hash": "forged", "definition_snapshot": {"name": "forged"}})
    assert run["definition_snapshot"]["name"] == "original"
    assert run["definition_hash"] == content_hash({"name": "original"})
