"""End-to-end orchestration against deterministic arrays, never production data."""
from dataclasses import replace
import threading
import time
import numpy as np
import pytest

from backend.timing_research.catalog import templates
from backend.timing_research.contracts import Definition, RunRequest
from backend.timing_research.data import ETFBars
from backend.timing_research.service import TimingResearchService


def fixture_bars():
    days = np.arange(np.datetime64("2019-01-01"), np.datetime64("2022-01-01"), dtype="datetime64[D]").astype(np.int64)
    close = 100 + np.arange(days.size, dtype=np.float64) * .02 + np.sin(np.arange(days.size) / 8) * 3
    arrays = dict(dates=days, open=close * .999, high=close * 1.02, low=close * .98, close=close,
                  volume=np.full(days.size, 1000.0), available_days=days)
    arrays.update({"raw_" + name: arrays[name] for name in ("open", "high", "low", "close")})
    for array in arrays.values():
        array.setflags(write=False)
    return ETFBars(**arrays, lineage={"source_hash": "fixture-v1", "price_basis": "hfq"}, warnings=())


@pytest.fixture
def service(tmp_path):
    bars = fixture_bars()
    value = TimingResearchService(tmp_path, loader=lambda *args: bars)
    value.warm()
    yield value
    value.close()


def request_for(service, codes=None):
    definition = Definition.model_validate(templates()[0]["definition"])
    prepared = service.prepare(definition)
    return RunRequest(definition=definition, compile_token=prepared["compile_token"], targets=[{"product_id": code} for code in (codes or ["510300.SH"])],
                      start_date="2020-01-01", end_date="2021-12-31", holdout_start="2021-01-01")


def completed(service, job):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        value = service.job(job["id"])
        if value["status"] in {"completed", "failed"}:
            assert value["status"] == "completed", value
            return service.repository.get_run(value["run_id"])
        time.sleep(.01)
    pytest.fail("research job did not finish")


def test_complete_run_arrays_and_oos_are_frozen(service):
    request = request_for(service)
    run = completed(service, service.submit(request))
    assert run["successful_products"] == 1
    assert run["execution"]["python_fallback"] == 0
    product = run["products"][0]
    assert product["status"] == "ok", product
    assert product["summary"]["out_of_sample"]["observations"] == 365
    assert len(product["monthly"]) == 24
    assert len(product["yearly"]) == 2
    assert len(product["walk_forward"]) == 3
    assert product["diagnostics"]["all"]["total_month_count"] == 24
    assert product["diagnostics"]["out_of_sample"]["total_month_count"] == 12
    assert product["diagnostics"]["all"]["raw_signal_count"] > 0
    assert [row["horizon"] for row in product["signal_quality"]] == [5, 10, 15]
    assert product["curve"][0]["date"] == "2020-01-01"
    assert len(product["channels"][0]["values"]) == len(product["curve"])
    arrays = service.repository.load_arrays(run["id"], "510300.SH")
    assert not arrays["close"].flags.writeable
    np.testing.assert_array_equal(arrays["close"], fixture_bars().close)
    assert arrays["entry"].dtype == np.int64
    request.definition.name = "changed after dispatch"
    assert service.repository.get_run(run["id"])["definition_snapshot"]["name"] != request.definition.name


def test_each_product_is_independent_and_lazy(service):
    original = service.loader
    def loader(base, code, *args):
        if code == "510500.SH":
            raise ValueError("private implementation detail")
        return original()
    service.loader = loader
    run = completed(service, service.submit(request_for(service, ["510500.SH", "510300.SH", "159915.SZ"])))
    assert run["products"][0]["status"] == "error"
    assert "private implementation" not in run["products"][0]["error"]
    assert run["products"][1]["detail_loaded"]
    assert not run["products"][2]["detail_loaded"]
    detail = service.repository.get_product(run["id"], "159915.SZ", offset=10, limit=5)
    assert len(detail["curve"]) == 5
    assert len(detail["channels"][0]["values"]) == 5


def test_missing_price_future_availability_fail_closed(service):
    bars = fixture_bars()
    close = bars.close.copy(); close[500] = np.nan; close.setflags(write=False)
    service.loader = lambda *args: replace(bars, close=close)
    run = completed(service, service.submit(request_for(service)))
    assert run["products"][0]["status"] == "error"
    available = bars.available_days.copy(); available[0] += 1; available.setflags(write=False)
    service.loader = lambda *args: replace(bars, available_days=available)
    run = completed(service, service.submit(request_for(service)))
    assert "晚于行情日期" in run["products"][0]["error"]
    volume = bars.volume.copy(); volume[500] = 0; volume.setflags(write=False)
    service.loader = lambda *args: replace(bars, volume=volume)
    run = completed(service, service.submit(request_for(service)))
    assert "成交量" in run["products"][0]["error"]


def test_prepare_token_queue_and_version_conflict(service):
    request = request_for(service)
    request.compile_token = "wrong" * 16
    with pytest.raises(Exception, match="重新准备"):
        service.submit(request)
    request = request_for(service)
    gate = threading.Event()
    original = service.loader
    service.loader = lambda *args: (gate.wait(5), original())[1]
    try:
        jobs = [service.submit(request) for _ in range(4)]
        with pytest.raises(Exception, match="4 个研究任务"):
            service.submit(request)
    finally:
        gate.set()
    for job in jobs:
        completed(service, job)
    saved = service.save_definition(request.definition)
    updated = service.save_definition(request.definition, saved["id"], saved["revision"])
    assert updated["revision"] == 2
    with pytest.raises(Exception):
        service.save_definition(request.definition, saved["id"], saved["revision"])


def test_comparison_and_research_only_release(service):
    request = request_for(service)
    first = completed(service, service.submit(request))
    second = completed(service, service.submit(request))
    assert len(service.compare([first["id"], second["id"]])["items"]) == 2
    changed = request.model_copy(update={"price_basis": "raw"})
    third = completed(service, service.submit(changed))
    with pytest.raises(Exception, match="相同产品"):
        service.compare([first["id"], third["id"]])
    release = service.repository.create_release(first["id"], "研究版本", "样本仍需独立复核")
    binding = service.repository.create_binding(release["id"], "pre_investment", "workspace")
    assert binding["release_id"] == release["id"]


def test_dates_and_product_caps_rejected():
    with pytest.raises(Exception):
        RunRequest(definition=templates()[0]["definition"], compile_token="0" * 64,
                   targets=[{"kind": "fund", "product_id": "510300.SH"}], start_date="2020-01-01", end_date="2021-12-31", holdout_start="2020-01-01")
