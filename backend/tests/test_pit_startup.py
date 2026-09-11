"""A slow diagnostic scan must not block service readiness or waive PIT checks."""

import ast
from pathlib import Path
from threading import Event, Thread

import pytest

from backend.pit import audit


@pytest.fixture
def scan_threads(monkeypatch):
    threads = []

    def capture_thread(**kwargs):
        thread = Thread(**kwargs)
        threads.append(thread)
        return thread

    audit.clear_cache()
    monkeypatch.setattr(audit.threading, "Thread", capture_thread)
    yield threads
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    audit.clear_cache()


def test_background_scan_returns_before_slow_read_finishes(monkeypatch, tmp_path, scan_threads):
    entered, release = Event(), Event()

    def slow_audit(data_dir, *, scan):
        assert data_dir == tmp_path and scan is True
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(audit, "audit_all", slow_audit)
    try:
        assert audit.start_scan(tmp_path)["state"] == "running"
        assert entered.wait(2)
        assert audit.scan_status()["state"] == "running"
        assert scan_threads[0].daemon is True
        audit.start_scan(tmp_path)
        assert len(scan_threads) == 1  # Startup and the page share one scan.
    finally:
        release.set()
        if scan_threads:
            scan_threads[0].join(2)
    assert audit.scan_status()["state"] == "ready"


def test_background_failure_is_not_reported_as_ready(monkeypatch, tmp_path, scan_threads):
    def fail(*args, **kwargs):
        raise OSError("test disk unavailable")

    monkeypatch.setattr(audit, "audit_all", fail)
    audit.start_scan(tmp_path)
    scan_threads[0].join(2)
    assert audit.scan_status()["state"] == "failed"
    assert audit.scan_status()["error"]


def test_startup_keeps_njit_worker_gate_and_does_not_scan_synchronously():
    module = ast.parse((Path(__file__).parents[1] / "app.py").read_text())
    lifespan = next(node for node in module.body if isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "lifespan")
    calls = {ast.unparse(node.func): node.lineno for node in ast.walk(lifespan)
             if isinstance(node, ast.Call)}
    for warmup in ("indicator_service.start_compute_engine", "warm_resolution_kernels",
                   "factor_service.warm", "regime_graph_v2_service.prewarm_saved_definitions"):
        assert calls[warmup] < calls["start_scan"]
    assert not any(name.endswith("audit_all") or name == "_warm_pit_audit" for name in calls)
    assert calls["start_scan"] < next(node.lineno for node in ast.walk(lifespan)
                                     if isinstance(node, ast.Yield))


def test_health_reports_scan_independently_of_njit(monkeypatch):
    # Execute just the real health handler with controlled state; do not import
    # the production app and its live data store merely to test this response.
    from types import SimpleNamespace
    import sys

    module = ast.parse((Path(__file__).parents[1] / "app.py").read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == "health")
    function.decorator_list = []
    monkeypatch.setitem(sys.modules, "pit.audit", audit)
    monkeypatch.setattr(audit, "scan_status", lambda: {"state": "running"})
    state = SimpleNamespace(numba_warmup={"complete": True})
    scope = {"app": SimpleNamespace(state=state)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "app.health", "exec"), scope)
    result = scope["health"]()
    assert result["ok"] and result["numba_warmup"]["complete"]
    assert result["pit_audit"] == {"state": "running", "complete": False}
    state.pit_audit_start_error = "RuntimeError"
    assert scope["health"]()["pit_audit"]["state"] == "failed"
