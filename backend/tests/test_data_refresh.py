from __future__ import annotations

import io
import json
import stat
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
for path in (ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.services import data_refresh
from backend.services import data_routes
from backend.services.refresh_runtime import InterProcessFileLock, atomic_write_json


TEST_TOKEN = "unit-test-token-1234567890"


@pytest.fixture(autouse=True)
def isolated_refresh_configuration(monkeypatch, tmp_path):
    # Command/fingerprint helpers must never open the user's real SQLite store.
    monkeypatch.setattr(data_refresh, 'DATA_DIR', tmp_path)


def _write_test_token(data_dir: Path) -> None:
    data_refresh.save_local_tushare_token(TEST_TOKEN, data_dir=data_dir)


def test_refresh_defaults_off_in_production_and_on_in_development(monkeypatch) -> None:
    monkeypatch.delenv("DATA_REFRESH_ENABLED", raising=False)
    monkeypatch.setenv("APP_ENV", "production")
    assert data_refresh.data_refresh_enabled() is False
    monkeypatch.setenv("APP_ENV", "development")
    assert data_refresh.data_refresh_enabled() is True


def test_refresh_command_is_fixed_and_uses_selected_incremental_modules(monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_MAX_CALLS_PER_MINUTE", "55")
    monkeypatch.setenv("TUSHARE_MIN_CALL_INTERVAL_SECONDS", "1.0")
    command = data_refresh.build_refresh_command(["base", "fund"], "incremental")
    assert command[0] == sys.executable
    assert "--latest" in command
    assert "--fund-info" in command
    assert "--fund-nav" in command
    assert "--fund-company" in command
    assert "--calendar" in command
    assert "--etf-info" not in command
    assert "--nav" not in command
    assert command[command.index("--max-calls-per-minute") + 1] == "55"
    assert command[command.index("--min-call-interval-sec") + 1] == "1.0"


def test_refresh_command_uses_parallel_safe_defaults(monkeypatch) -> None:
    for name in (
        "TUSHARE_MAX_CALLS_PER_MINUTE",
        "TUSHARE_MIN_CALL_INTERVAL_SECONDS",
        "TUSHARE_MAX_WORKERS",
    ):
        monkeypatch.delenv(name, raising=False)

    command = data_refresh.build_refresh_command(["etf", "fund"], "incremental")

    assert command[command.index("--max-calls-per-minute") + 1] == "450"
    assert command[command.index("--min-call-interval-sec") + 1] == "0.13"
    assert command[command.index("--max-workers") + 1] == "16"
    assert "--etf-share" in command


def test_dataset_summaries_read_parquet_metadata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(data_refresh, "DATA_DIR", tmp_path)
    data_refresh._parquet_summary_cached.cache_clear()
    pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-08-27"), "ts_code": "510050.SH"},
            {"date": pd.Timestamp("2026-08-28"), "ts_code": "510050.SH"},
        ]
    ).to_parquet(tmp_path / "etf_daily_df.parquet", index=False)

    summary = data_refresh.dataset_summaries()["etf_nav"]

    assert summary["exists"] is True
    assert summary["rows"] == 2
    assert summary["latest_date"] == "2026-08-28"


def test_frontend_token_is_local_private_and_environment_is_ignored(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "legacy-environment-token")
    assert data_refresh.tushare_token_configured(tmp_path) is False

    status = data_refresh.save_local_tushare_token(TEST_TOKEN, data_dir=tmp_path)
    credential = tmp_path / data_refresh.TUSHARE_TOKEN_FILENAME

    assert status["token_configured"] is True
    assert status["token_source"] == "frontend_local"
    assert TEST_TOKEN not in json.dumps(status)
    assert credential.read_text(encoding="utf-8").strip() == TEST_TOKEN
    assert stat.S_IMODE(credential.stat().st_mode) == 0o600
    assert data_refresh._sanitise_output(
        f"request failed for {TEST_TOKEN}", data_dir=tmp_path
    ) == "request failed for [REDACTED]"

    cleared = data_refresh.remove_local_tushare_token(data_dir=tmp_path)
    assert cleared["token_configured"] is False
    assert credential.exists() is False


def test_frontend_token_routes_never_return_secret(monkeypatch) -> None:
    captured: list[str] = []
    monkeypatch.setattr(data_routes, "data_refresh_enabled", lambda: True)
    monkeypatch.setattr(data_routes, "tushare_token_configuration_enabled", lambda: True)
    monkeypatch.setattr(
        data_routes,
        "save_local_tushare_token",
        lambda token: captured.append(token) or {"token_configured": True, "token_source": "frontend_local"},
    )
    monkeypatch.setattr(
        data_routes,
        "remove_local_tushare_token",
        lambda: {"token_configured": False, "token_source": "none"},
    )

    saved = data_routes.configure_tushare_token(data_routes.TushareTokenRequest(token=TEST_TOKEN))
    removed = data_routes.clear_tushare_token()

    assert captured == [TEST_TOKEN]
    assert TEST_TOKEN not in json.dumps(saved)
    assert saved["token_configured"] is True
    assert removed["token_configured"] is False


def test_token_cannot_change_while_refresh_lock_is_held(tmp_path: Path) -> None:
    data_refresh.save_local_tushare_token(TEST_TOKEN, data_dir=tmp_path)
    refresh_lock = InterProcessFileLock(tmp_path / data_refresh.REFRESH_LOCK_FILENAME)
    assert refresh_lock.acquire(owner="refresh-running") is True
    try:
        with pytest.raises(RuntimeError, match="正在运行"):
            data_refresh.save_local_tushare_token("replacement-token-123456", data_dir=tmp_path)
    finally:
        refresh_lock.release()

    assert (tmp_path / data_refresh.TUSHARE_TOKEN_FILENAME).read_text(encoding="utf-8").strip() == TEST_TOKEN


def test_start_refresh_enforces_feature_gate_and_single_job(monkeypatch) -> None:
    request = data_routes.DataRefreshRequest(modules=["etf"], mode="incremental")
    monkeypatch.setattr(data_routes, "data_refresh_enabled", lambda: False)
    with pytest.raises(data_routes.HTTPException) as disabled:
        data_routes.start_refresh(request)
    assert disabled.value.status_code == 403

    monkeypatch.setattr(data_routes, "data_refresh_enabled", lambda: True)
    monkeypatch.setattr(data_routes, "tushare_token_configured", lambda: True)

    class BusyManager:
        @staticmethod
        def start(_modules, _mode):
            raise RuntimeError("数据更新任务正在运行，请勿重复启动。")

    monkeypatch.setattr(data_routes, "refresh_manager", BusyManager())
    with pytest.raises(data_routes.HTTPException) as busy:
        data_routes.start_refresh(request)
    assert busy.value.status_code == 409


def test_full_refresh_requires_explicit_gate(monkeypatch) -> None:
    monkeypatch.delenv("DATA_FULL_REFRESH_ENABLED", raising=False)
    with pytest.raises(PermissionError, match="全量更新未开启"):
        data_refresh.build_refresh_command(["fund"], "full")

    monkeypatch.setenv("DATA_FULL_REFRESH_ENABLED", "true")
    command = data_refresh.build_refresh_command(["fund"], "full")
    assert "--latest" not in command
    assert "--fund-info" in command
    assert "--fund-nav" in command
    assert command[command.index("--start-date") + 1] == "20100101"


def test_cross_process_lock_rejects_duplicate_and_stale_state_recovers(tmp_path: Path) -> None:
    manager = data_refresh.DataRefreshManager(data_dir=tmp_path)
    external_lock = InterProcessFileLock(manager.process_lock_path)
    assert external_lock.acquire(owner="external-test") is True
    running_job = {
        **data_refresh._empty_job(),
        "job_id": "external-job",
        "status": "running",
        "started_at": data_refresh.utc_now(),
        "modules": ["fund"],
        "mode": "full",
        "message": "外部全量任务运行中",
    }
    atomic_write_json(manager.state_path, {"schema_version": 1, "job": running_job})

    with pytest.raises(RuntimeError, match="请勿重复启动"):
        manager.start(["fund"], "incremental")
    running_status = manager.snapshot()
    assert running_status["job"]["status"] == "running"
    assert running_status["execution_mode"] == "background"
    assert running_status["refresh_locked"] is True

    external_lock.release()
    recovered = manager.snapshot()["job"]
    assert recovered["status"] == "failed"
    assert "已中断" in recovered["message"]
    assert recovered["resume_available"] is True
    assert recovered["interruption_reason"] == "owner_process_lost"
    persisted = json.loads(manager.state_path.read_text(encoding="utf-8"))
    assert persisted["job"]["status"] == "failed"


def test_manager_start_returns_while_background_worker_keeps_lock(
    monkeypatch, tmp_path: Path
) -> None:
    _write_test_token(tmp_path)
    manager = data_refresh.DataRefreshManager(data_dir=tmp_path)
    worker_started = threading.Event()
    release_worker = threading.Event()

    def wait_in_background(job_id, _modules, _mode, _index_scopes, _module_scopes):
        worker_started.set()
        release_worker.wait(timeout=3)
        manager._finish_job(
            job_id,
            {"status": "succeeded", "message": "后台测试任务完成"},
        )

    monkeypatch.setattr(manager, "_run", wait_in_background)
    started = manager.start(["etf"], "incremental")

    try:
        assert worker_started.wait(timeout=1)
        assert release_worker.is_set() is False
        assert started["job"]["status"] == "running"
        assert started["execution_mode"] == "background"
        assert started["refresh_locked"] is True
        with pytest.raises(RuntimeError, match="请勿重复启动"):
            manager.start(["fund"], "incremental")
    finally:
        release_worker.set()

    deadline = time.monotonic() + 1
    while manager.snapshot()["job"]["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.snapshot()["job"]["status"] == "succeeded"


def test_status_exposes_external_refresh_lock_before_job_state_is_visible(
    tmp_path: Path,
) -> None:
    manager = data_refresh.DataRefreshManager(data_dir=tmp_path)
    external_lock = InterProcessFileLock(manager.process_lock_path)
    assert external_lock.acquire(owner="cli-starting") is True
    try:
        status = manager.snapshot()
        assert status["job"]["status"] == "idle"
        assert status["refresh_locked"] is True
    finally:
        external_lock.release()

    assert manager.snapshot()["refresh_locked"] is False


def test_recent_legacy_validation_checkpoint_blocks_duplicate_web_refresh(tmp_path: Path) -> None:
    checkpoint = (
        tmp_path
        / "tushare_full_validation_20260831"
        / ".fund_nav_df_parts_20100101_20260831"
    )
    checkpoint.mkdir(parents=True)
    manager = data_refresh.DataRefreshManager(data_dir=tmp_path)

    status = manager.snapshot()

    assert status["job"]["status"] == "running"
    assert status["job"]["job_id"].startswith("legacy-")
    with pytest.raises(RuntimeError, match="旧版外部全量任务"):
        manager.start(["fund"], "incremental")


def test_default_manager_resolves_active_manifest_for_dataset_status(monkeypatch, tmp_path: Path) -> None:
    active_dir = tmp_path / "snapshot-v1"
    active_dir.mkdir()
    pd.DataFrame(
        [{"date": pd.Timestamp("2026-08-31"), "ts_code": "510050.SH"}]
    ).to_parquet(active_dir / "etf_daily_df.parquet", index=False)
    (tmp_path / "tushare_active.json").write_text(
        json.dumps({"schema_version": 1, "snapshot_dir": "snapshot-v1"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(data_refresh, "DATA_DIR", tmp_path)
    manager = data_refresh.DataRefreshManager()

    status = manager.snapshot()

    assert Path(status["data_dir"]) == active_dir
    assert status["datasets"]["etf_nav"]["rows"] == 1


def test_full_refresh_staging_is_isolated_from_active_snapshot(tmp_path: Path) -> None:
    active = tmp_path / "active"
    active.mkdir()
    (active / "fund_info_df.parquet").write_bytes(b"active-version")
    (active / data_refresh.REFRESH_STATE_FILENAME).write_text("state", encoding="utf-8")

    staging = data_refresh.prepare_full_refresh_staging(
        data_root=tmp_path,
        source_dir=active,
        job_id="1234567890",
    )
    (staging / "fund_info_df.parquet").write_bytes(b"candidate-version")

    assert (active / "fund_info_df.parquet").read_bytes() == b"active-version"
    assert not (staging / data_refresh.REFRESH_STATE_FILENAME).exists()


def test_full_web_refresh_promotes_only_after_local_validation(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    staging = tmp_path / "candidate"
    staging.mkdir()

    class FakeProcess:
        pid = 43211

        def __init__(self, command, **_kwargs):
            captured["command"] = command
            self.stdout = io.StringIO("[OK] full fetch complete\n")
            self.returncode = None

        def wait(self, timeout=None):
            del timeout
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setenv("DATA_FULL_REFRESH_ENABLED", "true")
    monkeypatch.setattr(data_refresh.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(data_refresh, "prepare_full_refresh_staging", lambda **_kwargs: staging)
    monkeypatch.setattr(
        data_refresh,
        "rebuild_local_analytics_snapshot",
        lambda path: {"rows": 2, "path": str(path / "instrument_metrics_snapshot.parquet")},
    )

    def promote(path, *, data_root):
        captured["promoted"] = (path, data_root)
        return {
            "validation": {"status": "passed"},
            "manifest": {"snapshot_dir": "candidate"},
        }

    monkeypatch.setattr(data_refresh, "validate_and_activate_full_refresh", promote)
    _write_test_token(tmp_path)
    manager = data_refresh.DataRefreshManager(data_dir=tmp_path)
    manager.start(["fund"], "full")
    deadline = time.monotonic() + 3
    status = manager.snapshot()
    while status["job"]["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
        status = manager.snapshot()

    assert status["job"]["status"] == "succeeded"
    assert "原子切换" in status["job"]["message"]
    assert captured["promoted"] == (staging, tmp_path)
    command = captured["command"]
    assert command[command.index("--output-dir") + 1] == str(staging)


def test_full_refresh_reuses_fetch_complete_candidate_without_tushare(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_FULL_REFRESH_ENABLED", "true")
    candidate = tmp_path / "tushare_snapshot_failed_candidate"
    candidate.mkdir()
    fingerprint = data_refresh.refresh_request_fingerprint(["fund"], "full")
    previous = {
        **data_refresh._empty_job(),
        "job_id": "previous",
        "status": "succeeded",
        "modules": ["fund"],
        "mode": "full",
        "request_fingerprint": fingerprint,
        "staging_data_dir": str(candidate),
        "fetch_complete": True,
        "analytics_snapshot": {"status": "failed"},
    }
    atomic_write_json(
        tmp_path / data_refresh.REFRESH_STATE_FILENAME,
        {"schema_version": 1, "job": previous},
    )
    monkeypatch.setattr(
        data_refresh.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("不应再次抓取")),
    )
    rebuilt: list[Path] = []
    monkeypatch.setattr(
        data_refresh,
        "rebuild_local_analytics_snapshot",
        lambda path: rebuilt.append(path) or {"rows": 2},
    )
    monkeypatch.setattr(
        data_refresh,
        "validate_and_activate_full_refresh",
        lambda path, *, data_root: {
            "validation": {"status": "passed"},
            "manifest": {"snapshot_dir": path.name},
        },
    )

    _write_test_token(tmp_path)
    manager = data_refresh.DataRefreshManager(data_dir=tmp_path)
    started = manager.start(["fund"], "full")
    assert started["job"]["resumed"] is True
    deadline = time.monotonic() + 3
    status = manager.snapshot()
    while status["job"]["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
        status = manager.snapshot()

    assert status["job"]["status"] == "succeeded"
    assert rebuilt == [candidate]
    assert "不调用 Tushare" in status["job"]["log_tail"]


def test_incomplete_full_candidate_uses_action_resume_markers(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_FULL_REFRESH_ENABLED", "true")
    candidate = tmp_path / "tushare_snapshot_partial_candidate"
    candidate.mkdir()
    fingerprint = data_refresh.refresh_request_fingerprint(["fund"], "full")
    previous = {
        **data_refresh._empty_job(),
        "job_id": "previous",
        "status": "failed",
        "modules": ["fund"],
        "mode": "full",
        "request_fingerprint": fingerprint,
        "staging_data_dir": str(candidate),
        "fetch_complete": False,
        "analytics_snapshot": None,
    }
    atomic_write_json(
        tmp_path / data_refresh.REFRESH_STATE_FILENAME,
        {"schema_version": 1, "job": previous},
    )
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 43212

        def __init__(self, command, **_kwargs):
            captured["command"] = command
            self.stdout = io.StringIO("[OK] resumed\n")
            self.returncode = None

        def wait(self, timeout=None):
            del timeout
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(data_refresh.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(data_refresh, "rebuild_local_analytics_snapshot", lambda _path: {"rows": 2})
    monkeypatch.setattr(
        data_refresh,
        "validate_and_activate_full_refresh",
        lambda path, *, data_root: {
            "validation": {"status": "passed"},
            "manifest": {"snapshot_dir": path.name},
        },
    )

    _write_test_token(tmp_path)
    manager = data_refresh.DataRefreshManager(data_dir=tmp_path)
    manager.start(["fund"], "full")
    deadline = time.monotonic() + 3
    status = manager.snapshot()
    while status["job"]["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
        status = manager.snapshot()

    command = captured["command"]
    assert "--resume" in command
    assert command[command.index("--output-dir") + 1] == str(candidate)


def test_inherited_lock_descriptor_survives_abrupt_parent_close(tmp_path: Path) -> None:
    lock_path = tmp_path / ".refresh.lock"
    owner = InterProcessFileLock(lock_path)
    assert owner.acquire(owner="parent") is True
    child = data_refresh.subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.3)"],
        pass_fds=(owner.fileno(),),
    )
    # Simulate an abrupt owner-process exit: its descriptor is closed without
    # issuing LOCK_UN, while the refresh child keeps the inherited description.
    owner_handle = owner._handle
    owner._handle = None
    assert owner_handle is not None
    owner_handle.close()
    probe = InterProcessFileLock(lock_path)
    try:
        assert probe.acquire(owner="too-early") is False
        assert child.wait(timeout=2) == 0
        assert probe.acquire(owner="after-child") is True
    finally:
        probe.release()
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=2)


def test_refresh_streams_bounded_logs_and_snapshot_failure_is_nonfatal(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 43210

        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)
            self.stdout = io.StringIO(f"{'x' * 9000}\nTUSHARE_TOKEN=secret-token\n[OK] 数据落盘\n")
            self.returncode = None

        def wait(self, timeout=None):
            del timeout
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(data_refresh.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        data_refresh,
        "rebuild_local_analytics_snapshot",
        lambda _data_dir: (_ for _ in ()).throw(RuntimeError("snapshot unavailable")),
    )
    data_refresh.save_local_tushare_token("secret-token-123456", data_dir=tmp_path)
    manager = data_refresh.DataRefreshManager(data_dir=tmp_path)
    manager.start(["etf"], "incremental")

    deadline = time.monotonic() + 3
    status = manager.snapshot()
    while status["job"]["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
        status = manager.snapshot()

    job = status["job"]
    assert job["status"] == "succeeded"
    assert job["analytics_snapshot"]["status"] == "failed"
    assert job["warnings"][0]["code"] == "ANALYTICS_REBUILD_FAILED"
    assert "无需重新拉取数据" in job["message"]
    assert len(job["log_tail"]) <= data_refresh.MAX_LOG_TAIL_CHARS
    assert "secret-token" not in job["log_tail"]
    assert "[REDACTED]" in job["log_tail"]
    assert captured["stdout"] is data_refresh.subprocess.PIPE
    assert captured["stderr"] is data_refresh.subprocess.STDOUT
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert child_env["PYTHONUNBUFFERED"] == "1"
    assert child_env["TUSHARE_REFRESH_LOCK_HELD_BY_PARENT"] == "1"
    assert "TUSHARE_TOKEN" not in child_env
    assert len(captured["pass_fds"]) == 1
    assert isinstance(captured["pass_fds"][0], int)

    restored = data_refresh.DataRefreshManager(data_dir=tmp_path).snapshot()["job"]
    assert restored["job_id"] == job["job_id"]
    assert restored["status"] == "succeeded"
    probe = InterProcessFileLock(manager.process_lock_path)
    assert probe.acquire(owner="after-refresh") is True
    probe.release()


def test_chunked_refresh_log_exposes_only_latest_complete_progress_line(tmp_path: Path) -> None:
    manager = data_refresh.DataRefreshManager(data_dir=tmp_path)
    job_id = "chunked-progress"
    manager._job.update({"job_id": job_id, "status": "running", "message": "正在更新"})

    manager._append_log(
        job_id,
        "[INFO] index_daily 分段进度 50/9741，异常 0。\n"
        "[INFO] index_daily 分段进度 100/9741，异常 0。\n[INFO",
        force_persist=True,
    )

    assert manager._job["message"] == "[INFO] index_daily 分段进度 100/9741，异常 0。"
    assert manager.snapshot()["job"]["message"] == manager._job["message"]

    manager._append_log(
        job_id,
        "] index_daily 分段进度 150/9741，异常 0。\n",
        force_persist=True,
    )

    assert manager._job["message"] == "[INFO] index_daily 分段进度 150/9741，异常 0。"
    assert manager.snapshot()["job"]["message"] == manager._job["message"]

    manager._append_log(
        job_id,
        "[STAGE] index_daily 已完成数据拉取，正在合并本地增量数据。\n",
        force_persist=True,
    )

    assert manager._job["message"] == "[STAGE] index_daily 已完成数据拉取，正在合并本地增量数据。"
    assert manager.snapshot()["job"]["message"] == manager._job["message"]


def test_refresh_stream_forwards_short_progress_line_without_waiting_for_one_kibibyte(
    tmp_path: Path,
) -> None:
    manager = data_refresh.DataRefreshManager(data_dir=tmp_path)
    job_id = "line-progress"
    manager._job.update({"job_id": job_id, "status": "running", "message": "正在更新"})

    class Process:
        stdout = io.StringIO("[INFO] 指数权重进度 100/9141，workers=16，异常 0。\n")

    manager._stream_process_output(job_id, Process())

    assert manager._job["message"] == "[INFO] 指数权重进度 100/9141，workers=16，异常 0。"
    assert manager._child_output_at[job_id] <= time.monotonic()


def test_refresh_timeout_tracks_inactivity_instead_of_total_elapsed_time() -> None:
    assert data_refresh._next_process_wait_seconds(
        now=2_000.0,
        started_at=0.0,
        last_output_at=1_990.0,
        idle_timeout=1_800,
        max_runtime=0,
    ) == data_refresh.PROCESS_WAIT_POLL_SECONDS

    with pytest.raises(data_refresh._RefreshProcessTimeout) as idle_error:
        data_refresh._next_process_wait_seconds(
            now=2_000.0,
            started_at=0.0,
            last_output_at=199.0,
            idle_timeout=1_800,
            max_runtime=0,
        )
    assert idle_error.value.kind == "idle"

    with pytest.raises(data_refresh._RefreshProcessTimeout) as runtime_error:
        data_refresh._next_process_wait_seconds(
            now=7_201.0,
            started_at=0.0,
            last_output_at=7_200.0,
            idle_timeout=1_800,
            max_runtime=7_200,
        )
    assert runtime_error.value.kind == "max_runtime"


def test_incremental_refresh_reuses_current_snapshot_when_market_data_is_unchanged(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeProcess:
        pid = 43213

        def __init__(self, *_args, **_kwargs):
            self.stdout = io.StringIO("[OK] 增量内容无变化\n")
            self.returncode = None

        def wait(self, timeout=None):
            del timeout
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    pd.DataFrame(
        [{"ts_code": "000001.OF", "date": pd.Timestamp("2026-09-01"), "adj_nav": 1.0}]
    ).to_parquet(tmp_path / "fund_nav_df.parquet", index=False)
    time.sleep(0.01)
    pd.DataFrame(
        [{"instrument_type": "fund", "ts_code": "000001.OF", "as_of": "2026-09-01"}]
    ).to_parquet(tmp_path / "instrument_metrics_snapshot.parquet", index=False)
    monkeypatch.setattr(data_refresh.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        data_refresh,
        "rebuild_local_analytics_snapshot",
        lambda _path: pytest.fail("unchanged market data must reuse current snapshot"),
    )
    _write_test_token(tmp_path)
    manager = data_refresh.DataRefreshManager(data_dir=tmp_path)
    manager.start(["fund"], "incremental")

    deadline = time.monotonic() + 3
    status = manager.snapshot()
    while status["job"]["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
        status = manager.snapshot()

    job = status["job"]
    assert job["status"] == "succeeded"
    assert job["analytics_snapshot"]["status"] == "succeeded"
    assert job["analytics_snapshot"]["reused"] is True
    assert "无变化" in job["message"]


def test_local_analytics_rebuild_route_needs_no_tushare_token(monkeypatch) -> None:
    monkeypatch.setattr(data_routes, "data_refresh_enabled", lambda: True)
    monkeypatch.setattr(
        data_routes,
        "refresh_manager",
        type("Manager", (), {"rebuild_analytics": staticmethod(lambda: {"status": "succeeded", "rows": 7})})(),
    )

    assert data_routes.rebuild_analytics_snapshot() == {"status": "succeeded", "rows": 7}

    class BusyManager:
        @staticmethod
        def rebuild_analytics():
            raise RuntimeError("数据刷新正在运行")

    monkeypatch.setattr(data_routes, "refresh_manager", BusyManager())
    with pytest.raises(data_routes.HTTPException) as busy:
        data_routes.rebuild_analytics_snapshot()
    assert busy.value.status_code == 409


def test_refresh_status_disables_http_caching(monkeypatch) -> None:
    calls: list[bool] = []

    def snapshot(*, include_datasets=True):
        calls.append(include_datasets)
        return {"job": {"status": "idle"}}

    monkeypatch.setattr(
        data_routes,
        "refresh_manager",
        type("Manager", (), {"snapshot": staticmethod(snapshot)})(),
    )
    response = data_routes.Response()

    assert data_routes.refresh_status(response) == {"job": {"status": "idle"}}
    assert data_routes.refresh_status(response, progress_only=True) == {"job": {"status": "idle"}}
    assert calls == [True, False]
    assert response.headers["Cache-Control"] == "no-store"


def test_index_refresh_defaults_and_scope_fingerprint(monkeypatch) -> None:
    command = data_refresh.build_refresh_command(["index"], "incremental")
    assert "--calendar" in command
    assert "--index-catalog" in command
    assert "--index-domestic" in command
    assert "--index-industry" in command
    assert "--index-global" in command
    assert "--index-concept" not in command

    command = data_refresh.build_refresh_command(
        ["index"], "incremental", index_scopes=["concept"]
    )
    assert "--index-catalog" in command
    assert "--index-concept" in command
    assert data_refresh.refresh_request_fingerprint(
        ["index"], "incremental", ["concept"]
    ) != data_refresh.refresh_request_fingerprint(
        ["index"], "incremental", ["global"]
    )


def test_refresh_command_supports_scopes_for_every_module(monkeypatch) -> None:
    monkeypatch.setenv("DATA_FULL_REFRESH_ENABLED", "true")
    selected = {
        "base": ["fund_company"],
        "etf": ["candle"],
        "fund": ["info"],
        "index": ["valuation"],
    }

    command = data_refresh.build_refresh_command(
        ["base", "etf", "fund", "index"],
        "full",
        module_scopes=selected,
    )

    assert "--fund-company" in command
    assert "--calendar" not in command
    assert "--stock-basic" not in command
    assert "--etf-info" in command
    assert "--candle" in command
    assert "--nav" not in command
    assert "--fund-info" in command
    assert "--fund-nav" not in command
    assert "--index-catalog" in command
    assert "--index-valuation" in command
    assert "--index-domestic" not in command

    info_only = data_refresh.build_refresh_command(
        ["etf"], "incremental", module_scopes={"etf": ["info"]}
    )
    assert "--etf-info" in info_only
    assert "--calendar" not in info_only

    normalised = data_refresh.normalise_module_scopes(
        ["etf", "fund", "index"],
        {"etf": ["nav"], "fund": ["nav"], "index": ["global"]},
    )
    assert normalised == {
        "etf": ["info", "nav"],
        "fund": ["info", "nav"],
        "index": ["catalog", "global"],
    }


def test_fund_and_macro_scope_dependencies_build_explicit_cli_flags() -> None:
    command = data_refresh.build_refresh_command(
        ["fund", "macro"],
        "incremental",
        module_scopes={
            "fund": ["scale", "portfolio"],
            "macro": ["cycle", "rates", "release_calendar"],
        },
    )

    assert "--fund-info" in command
    assert "--fund-nav" in command
    assert "--fund-scale" in command
    assert "--fund-portfolio" in command
    assert "--macro-cycle" in command
    assert "--macro-rates" in command
    assert "--macro-release-calendar" in command
    assert "--macro-money-credit" not in command

    normalised = data_refresh.normalise_module_scopes(
        ["fund", "macro"],
        {"fund": ["scale"], "macro": ["money_credit"]},
    )
    assert normalised == {
        "fund": ["info", "nav", "scale"],
        "macro": ["money_credit"],
    }


def test_refresh_module_scopes_reject_invalid_or_unselected_content() -> None:
    with pytest.raises(ValueError, match="不支持下载内容"):
        data_refresh.normalise_module_scopes(["etf"], {"etf": ["unknown"]})
    with pytest.raises(ValueError, match="未选择的数据模块"):
        data_refresh.normalise_module_scopes(["etf"], {"fund": ["info"]})
    with pytest.raises(ValueError, match="至少选择一项"):
        data_refresh.normalise_module_scopes(["etf"], {"etf": []})


def test_refresh_route_forwards_generic_module_scopes(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class CaptureManager:
        @staticmethod
        def start(modules, mode, index_scopes=None, *, module_scopes=None):
            captured.update(
                modules=modules,
                mode=mode,
                index_scopes=index_scopes,
                module_scopes=module_scopes,
            )
            return {"job": {"status": "running"}}

    monkeypatch.setattr(data_routes, "data_refresh_enabled", lambda: True)
    monkeypatch.setattr(data_routes, "tushare_token_configured", lambda: True)
    monkeypatch.setattr(data_routes, "refresh_manager", CaptureManager())
    request = data_routes.DataRefreshRequest(
        modules=["etf", "fund"],
        mode="incremental",
        module_scopes={"etf": ["candle"], "fund": ["info"]},
    )

    assert data_routes.start_refresh(request) == {"job": {"status": "running"}}
    assert captured == {
        "modules": ["etf", "fund"],
        "mode": "incremental",
        "index_scopes": None,
        "module_scopes": {"etf": ["candle"], "fund": ["info"]},
    }


def test_refresh_status_exposes_index_scopes_and_datasets(tmp_path: Path) -> None:
    manager = data_refresh.DataRefreshManager(data_dir=tmp_path)
    status = manager.snapshot()

    assert "index" in status["available_modules"]
    assert "macro" in status["available_modules"]
    assert status["default_module_scopes"]["fund"] == [
        "info", "nav", "manager", "scale", "benchmark"
    ]
    assert status["default_module_scopes"]["macro"] == [
        "cycle", "money_credit", "rates", "release_calendar"
    ]
    assert status["default_index_scopes"] == ["catalog", "domestic", "industry", "global"]
    assert status["default_module_scopes"]["etf"] == ["info", "nav", "share", "candle"]
    assert status["available_module_scopes"]["base"] == [
        "calendar", "stock_basic", "fund_company"
    ]
    assert set(status["available_index_scopes"]) == {
        "catalog", "domestic", "industry", "concept", "global", "futures", "valuation", "constituents"
    }
    assert status["datasets"]["index_catalog"]["status"] == "missing"


def test_analytics_source_state_tracks_etf_share_changes(tmp_path: Path) -> None:
    share_path = tmp_path / "etf_share_size_df.parquet"
    pd.DataFrame(
        [{"ts_code": "510050.SH", "date": pd.Timestamp("2026-08-31"), "total_share": 1.0, "nav": 1.0}]
    ).to_parquet(share_path, index=False)

    before = data_refresh._analytics_source_state(tmp_path)
    pd.DataFrame(
        [{"ts_code": "510050.SH", "date": pd.Timestamp("2026-09-01"), "total_share": 2.0, "nav": 1.0}]
    ).to_parquet(share_path, index=False)
    after = data_refresh._analytics_source_state(tmp_path)

    assert before[0][0] == "etf_share_size_df.parquet"
    assert before != after
