"""Adaptive process, shared-memory, and Numba resources for indicator batches."""

from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import get_context, shared_memory
from pathlib import Path
from typing import Any, Iterator

import numba
import numpy as np
from numba import boolean, float64, int8, njit, prange

from cal_indicators.typed_numeric_backend import warm_typed_numeric_backend
from cal_indicators.builtin_batch_kernel import (
    compute_builtin_batch_parallel,
    compute_builtin_batch_serial,
    warm_builtin_batch_kernels,
)
from custom_indicators.errors import IndicatorDomainError


PARALLEL_ENGINE_VERSION = "indicator-parallel-2"
DEFAULT_QUEUE_WAIT_SECONDS = 1.0
DEFAULT_HARD_TIMEOUT_SECONDS = 600.0
DEFAULT_PRANGE_MIN_ELEMENTS = 300_000
DEFAULT_SHM_THRESHOLD_BYTES = 512 * 1024 * 1024


@njit(
    float64[:, ::1](
        float64[:, ::1],
        float64[::1],
        int8[::1],
        boolean[::1],
        float64[::1],
        float64[::1],
    ),
    cache=True,
    nogil=True,
)
def _score_matrix_serial(
    raw_values: np.ndarray,
    weights: np.ndarray,
    lower_better: np.ndarray,
    complete_mask: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    rows, columns = raw_values.shape
    output = np.empty((rows, columns), dtype=np.float64)
    output[:, :] = np.nan
    for row in range(rows):
        if not complete_mask[row]:
            continue
        for column in range(columns):
            span = upper[column] - lower[column]
            component = (
                50.0
                if span == 0.0
                else (raw_values[row, column] - lower[column]) / span * 100.0
            )
            if lower_better[column] != 0:
                component = 100.0 - component
            output[row, column] = component
    return output


@njit(
    float64[:, ::1](
        float64[:, ::1],
        float64[::1],
        int8[::1],
        boolean[::1],
        float64[::1],
        float64[::1],
    ),
    cache=True,
    nogil=True,
    parallel=True,
)
def _score_matrix_parallel(
    raw_values: np.ndarray,
    weights: np.ndarray,
    lower_better: np.ndarray,
    complete_mask: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    rows, columns = raw_values.shape
    output = np.empty((rows, columns), dtype=np.float64)
    output[:, :] = np.nan
    for row in prange(rows):
        if not complete_mask[row]:
            continue
        for column in range(columns):
            span = upper[column] - lower[column]
            component = (
                50.0
                if span == 0.0
                else (raw_values[row, column] - lower[column]) / span * 100.0
            )
            if lower_better[column] != 0:
                component = 100.0 - component
            output[row, column] = component
    return output


def warm_parallel_numeric_backend() -> None:
    sample = np.ascontiguousarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    weights = np.ascontiguousarray([0.5, 0.5], dtype=np.float64)
    directions = np.ascontiguousarray([0, 1], dtype=np.int8)
    complete = np.ascontiguousarray([True, True], dtype=np.bool_)
    lower = np.ascontiguousarray([1.0, 1.0], dtype=np.float64)
    upper = np.ascontiguousarray([2.0, 2.0], dtype=np.float64)
    _score_matrix_serial(sample, weights, directions, complete, lower, upper)
    _score_matrix_parallel(sample, weights, directions, complete, lower, upper)


def score_matrix(
    raw_values: np.ndarray,
    weights: np.ndarray,
    lower_better: np.ndarray,
    complete_mask: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    min_parallel_elements: int = DEFAULT_PRANGE_MIN_ELEMENTS,
    thread_budget: int = 1,
) -> tuple[np.ndarray, bool]:
    raw = np.ascontiguousarray(raw_values, dtype=np.float64)
    normalized_weights = np.ascontiguousarray(weights, dtype=np.float64)
    directions = np.ascontiguousarray(lower_better, dtype=np.int8)
    complete = np.ascontiguousarray(complete_mask, dtype=np.bool_)
    lower_values = np.ascontiguousarray(lower, dtype=np.float64)
    upper_values = np.ascontiguousarray(upper, dtype=np.float64)
    use_parallel = bool(
        raw.shape[0] >= 32
        and raw.size >= max(1, int(min_parallel_elements))
        and thread_budget > 1
    )
    if use_parallel:
        numba.set_num_threads(max(1, min(int(thread_budget), numba.config.NUMBA_NUM_THREADS)))
        return (
            _score_matrix_parallel(
                raw,
                normalized_weights,
                directions,
                complete,
                lower_values,
                upper_values,
            ),
            True,
        )
    return (
        _score_matrix_serial(
            raw,
            normalized_weights,
            directions,
            complete,
            lower_values,
            upper_values,
        ),
        False,
    )


@dataclass(frozen=True)
class SharedArrayDescriptor:
    backend: str
    location: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


class SharedArrayOwner:
    """Own one immutable array transported by POSIX SHM or a read-only mmap."""

    def __init__(
        self,
        array: np.ndarray,
        runtime_dir: Path,
        *,
        shm_threshold_bytes: int = DEFAULT_SHM_THRESHOLD_BYTES,
    ) -> None:
        source = np.ascontiguousarray(array)
        self._shared: shared_memory.SharedMemory | None = None
        self._mmap_path: Path | None = None
        runtime_dir.mkdir(parents=True, exist_ok=True)
        segment: shared_memory.SharedMemory | None = None
        if source.nbytes <= max(1, int(shm_threshold_bytes)):
            try:
                segment = shared_memory.SharedMemory(
                    name=f"indicator-{os.getpid()}-{uuid.uuid4().hex}",
                    create=True,
                    size=max(1, source.nbytes),
                )
            except (OSError, PermissionError):
                segment = None
        if segment is not None:
            destination = np.ndarray(source.shape, dtype=source.dtype, buffer=segment.buf)
            np.copyto(destination, source)
            destination.flags.writeable = False
            self._shared = segment
            self.descriptor = SharedArrayDescriptor(
                "shm", segment.name, tuple(source.shape), source.dtype.str, source.nbytes
            )
        else:
            handle = tempfile.NamedTemporaryFile(
                prefix=f"indicator-array-{os.getpid()}-",
                suffix=".mmap",
                dir=runtime_dir,
                delete=False,
            )
            handle.close()
            path = Path(handle.name)
            destination = np.memmap(path, mode="w+", dtype=source.dtype, shape=source.shape)
            destination[...] = source
            destination.flush()
            del destination
            self._mmap_path = path
            self.descriptor = SharedArrayDescriptor(
                "mmap", str(path), tuple(source.shape), source.dtype.str, source.nbytes
            )

    def view(self) -> np.ndarray:
        if self._shared is not None:
            view = np.ndarray(
                self.descriptor.shape,
                dtype=np.dtype(self.descriptor.dtype),
                buffer=self._shared.buf,
            )
        else:
            view = np.memmap(
                self.descriptor.location,
                mode="r",
                dtype=np.dtype(self.descriptor.dtype),
                shape=self.descriptor.shape,
            )
        view.flags.writeable = False
        return view

    def close(self) -> None:
        if self._shared is not None:
            self._shared.close()

    def unlink(self) -> None:
        if self._shared is not None:
            try:
                self._shared.unlink()
            except FileNotFoundError:
                pass
        if self._mmap_path is not None:
            try:
                self._mmap_path.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> "SharedArrayOwner":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
        self.unlink()


def attach_shared_array(
    descriptor: SharedArrayDescriptor,
    *,
    writable: bool = False,
) -> tuple[np.ndarray, shared_memory.SharedMemory | np.memmap[Any, Any]]:
    if descriptor.backend == "shm":
        handle = shared_memory.SharedMemory(name=descriptor.location, create=False)
        array = np.ndarray(
            descriptor.shape,
            dtype=np.dtype(descriptor.dtype),
            buffer=handle.buf,
        )
    elif descriptor.backend == "mmap":
        handle = np.memmap(
            descriptor.location,
            mode="r+" if writable else "r",
            dtype=np.dtype(descriptor.dtype),
            shape=descriptor.shape,
        )
        array = np.asarray(handle)
    else:
        raise ValueError(f"未知共享数组后端: {descriptor.backend}")
    array.flags.writeable = writable
    return array, handle


def _close_shared_handle(handle: shared_memory.SharedMemory | np.memmap[Any, Any]) -> None:
    if isinstance(handle, shared_memory.SharedMemory):
        handle.close()
        return
    mmap_handle = getattr(handle, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()


def _worker_initializer(
    max_numba_threads: int,
    startup_barrier: Any,
    startup_timeout_seconds: float,
) -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    numba.set_num_threads(max(1, min(max_numba_threads, numba.config.NUMBA_NUM_THREADS)))
    warm_typed_numeric_backend()
    warm_parallel_numeric_backend()
    warm_builtin_batch_kernels()
    # Every worker waits here only after all of its fixed signatures are ready.
    # Releasing the barrier therefore proves that the complete pool is hot; a
    # failed or timed-out worker prevents the API lifespan from completing.
    startup_barrier.wait(timeout=startup_timeout_seconds)


def _worker_ready() -> int:
    return os.getpid()


def _compute_builtin_shared_task(
    values_descriptor: SharedArrayDescriptor,
    starts_descriptor: SharedArrayDescriptor,
    ends_descriptor: SharedArrayDescriptor,
    codes_descriptor: SharedArrayDescriptor,
    primary_descriptor: SharedArrayDescriptor,
    secondary_descriptor: SharedArrayDescriptor,
    risk_free_descriptor: SharedArrayDescriptor,
    output_descriptor: SharedArrayDescriptor,
    status_descriptor: SharedArrayDescriptor,
    *,
    parallel: bool,
    thread_budget: int,
) -> dict[str, Any]:
    descriptors = (
        values_descriptor,
        starts_descriptor,
        ends_descriptor,
        codes_descriptor,
        primary_descriptor,
        secondary_descriptor,
        risk_free_descriptor,
    )
    attached = [attach_shared_array(descriptor) for descriptor in descriptors]
    output, output_handle = attach_shared_array(output_descriptor, writable=True)
    statuses, status_handle = attach_shared_array(status_descriptor, writable=True)
    try:
        arrays = [item[0] for item in attached]
        numba.set_num_threads(
            max(1, min(int(thread_budget), numba.config.NUMBA_NUM_THREADS))
        )
        function = compute_builtin_batch_parallel if parallel else compute_builtin_batch_serial
        function(*arrays, output, statuses)
        if isinstance(output_handle, np.memmap):
            output_handle.flush()
        if isinstance(status_handle, np.memmap):
            status_handle.flush()
        return {
            "worker_pid": os.getpid(),
            "parallel": parallel,
            "numba_threads": numba.get_num_threads() if parallel else 1,
        }
    finally:
        for _, handle in attached:
            _close_shared_handle(handle)
        _close_shared_handle(output_handle)
        _close_shared_handle(status_handle)


class _WeightedAdmission:
    def __init__(self, tokens: int) -> None:
        self._capacity = max(1, int(tokens))
        self._available = self._capacity
        self._condition = threading.Condition()

    @contextmanager
    def reserve(self, tokens: int, timeout: float) -> Iterator[int]:
        requested = max(1, min(int(tokens), self._capacity))
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self._available < requested:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise IndicatorDomainError(
                        "INDICATOR_ENGINE_BUSY",
                        "指标计算资源繁忙，请稍后重试。",
                        status_code=429,
                    )
                self._condition.wait(remaining)
            self._available -= requested
        try:
            yield requested
        finally:
            with self._condition:
                self._available += requested
                self._condition.notify_all()


class AdaptiveComputeEngine:
    """One persistent worker pool and weighted admission controller per service."""

    def __init__(self, runtime_dir: Path) -> None:
        cpu_count = max(1, os.cpu_count() or 1)
        default_workers = max(1, cpu_count - 1)
        self.worker_count = max(
            1, int(os.getenv("INDICATOR_PROCESS_WORKERS", str(default_workers)))
        )
        self.queue_wait_seconds = max(
            0.0,
            float(
                os.getenv(
                    "INDICATOR_QUEUE_WAIT_SECONDS", str(DEFAULT_QUEUE_WAIT_SECONDS)
                )
            ),
        )
        self.hard_timeout_seconds = max(
            1.0,
            float(
                os.getenv(
                    "INDICATOR_RUN_HARD_TIMEOUT_SECONDS",
                    str(DEFAULT_HARD_TIMEOUT_SECONDS),
                )
            ),
        )
        self.startup_timeout_seconds = max(
            1.0,
            float(
                os.getenv(
                    "INDICATOR_STARTUP_WARMUP_TIMEOUT_SECONDS",
                    str(min(self.hard_timeout_seconds, 120.0)),
                )
            ),
        )
        self.prange_min_elements = max(
            1,
            int(
                os.getenv(
                    "INDICATOR_PRANGE_MIN_ELEMENTS",
                    str(DEFAULT_PRANGE_MIN_ELEMENTS),
                )
            ),
        )
        self.runtime_dir = runtime_dir
        self._admission = _WeightedAdmission(self.worker_count)
        self._pool: ProcessPoolExecutor | None = None
        self._worker_pids: tuple[int, ...] = ()
        self._lock = threading.RLock()

    def _cleanup_orphan_arrays(self) -> None:
        protection_seconds = max(
            1.0,
            float(os.getenv("INDICATOR_ORPHAN_PROTECTION_SECONDS", "60")),
        )
        now = time.time()
        for path in self.runtime_dir.glob("indicator-array-*-*.mmap"):
            parts = path.name.split("-", 3)
            try:
                owner_pid = int(parts[2])
                age = now - path.stat().st_mtime
            except (IndexError, OSError, ValueError):
                continue
            if age < protection_seconds:
                continue
            try:
                os.kill(owner_pid, 0)
            except ProcessLookupError:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            except PermissionError:
                continue

    def start(self) -> None:
        with self._lock:
            if self._pool is not None:
                return
            self.runtime_dir.mkdir(parents=True, exist_ok=True)
            self._cleanup_orphan_arrays()
            process_context = get_context("spawn")
            startup_barrier = process_context.Barrier(self.worker_count)
            self._pool = ProcessPoolExecutor(
                max_workers=self.worker_count,
                mp_context=process_context,
                initializer=_worker_initializer,
                initargs=(1, startup_barrier, self.startup_timeout_seconds),
            )
        warm_parallel_numeric_backend()
        try:
            futures = [self._pool.submit(_worker_ready) for _ in range(self.worker_count)]
            for future in futures:
                future.result(timeout=self.startup_timeout_seconds)
            processes = tuple(getattr(self._pool, "_processes", {}).values())
            pids = tuple(
                sorted(
                    int(process.pid)
                    for process in processes
                    if process.pid is not None and process.is_alive()
                )
            )
            if len(pids) != self.worker_count:
                raise RuntimeError(
                    "indicator worker warmup incomplete: "
                    f"expected {self.worker_count}, ready {len(pids)}"
                )
            self._worker_pids = pids
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        with self._lock:
            pool = self._pool
            self._pool = None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        self._worker_pids = ()

    def restart(self) -> None:
        """Replace a failed pool and terminate workers before shared cleanup."""

        with self._lock:
            pool = self._pool
            self._pool = None
        if pool is not None:
            processes = list(getattr(pool, "_processes", {}).values())
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=1.0)
            pool.shutdown(wait=False, cancel_futures=True)
        self._worker_pids = ()
        self.start()

    @contextmanager
    def admission(self, tokens: int) -> Iterator[int]:
        with self._admission.reserve(tokens, self.queue_wait_seconds) as reserved:
            yield reserved

    @property
    def pool(self) -> ProcessPoolExecutor:
        self.start()
        assert self._pool is not None
        return self._pool

    def status(self) -> dict[str, Any]:
        return {
            "version": PARALLEL_ENGINE_VERSION,
            "started": self._pool is not None,
            "worker_processes": len(self._worker_pids),
            "worker_capacity": self.worker_count,
            "worker_pids": list(self._worker_pids),
            "fully_warmed": bool(
                self._pool is not None
                and len(self._worker_pids) == self.worker_count
            ),
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "queue_wait_seconds": self.queue_wait_seconds,
            "hard_timeout_seconds": self.hard_timeout_seconds,
            "prange_min_elements": self.prange_min_elements,
            "shared_memory_threshold_bytes": int(
                os.getenv(
                    "INDICATOR_SHM_THRESHOLD_BYTES",
                    str(DEFAULT_SHM_THRESHOLD_BYTES),
                )
            ),
        }

    def run_builtin_shared(
        self,
        *,
        values: SharedArrayDescriptor,
        starts: SharedArrayDescriptor,
        ends: SharedArrayDescriptor,
        codes: SharedArrayDescriptor,
        primary_indices: SharedArrayDescriptor,
        secondary_indices: SharedArrayDescriptor,
        risk_free: SharedArrayDescriptor,
        output: SharedArrayDescriptor,
        statuses: SharedArrayDescriptor,
        parallel: bool,
        thread_budget: int,
    ) -> dict[str, Any]:
        future = self.submit_builtin_shared(
            values=values,
            starts=starts,
            ends=ends,
            codes=codes,
            primary_indices=primary_indices,
            secondary_indices=secondary_indices,
            risk_free=risk_free,
            output=output,
            statuses=statuses,
            parallel=parallel,
            thread_budget=thread_budget,
        )
        return self.wait(future)

    def submit_builtin_shared(
        self,
        *,
        values: SharedArrayDescriptor,
        starts: SharedArrayDescriptor,
        ends: SharedArrayDescriptor,
        codes: SharedArrayDescriptor,
        primary_indices: SharedArrayDescriptor,
        secondary_indices: SharedArrayDescriptor,
        risk_free: SharedArrayDescriptor,
        output: SharedArrayDescriptor,
        statuses: SharedArrayDescriptor,
        parallel: bool,
        thread_budget: int,
    ) -> Future[dict[str, Any]]:
        arguments = (
            values,
            starts,
            ends,
            codes,
            primary_indices,
            secondary_indices,
            risk_free,
            output,
            statuses,
        )
        return self.pool.submit(
            _compute_builtin_shared_task,
            *arguments,
            parallel=parallel,
            thread_budget=thread_budget,
        )

    def wait(self, future: Future[dict[str, Any]]) -> dict[str, Any]:
        try:
            return future.result(timeout=self.hard_timeout_seconds)
        except TimeoutError as exc:
            raise IndicatorDomainError(
                "INDICATOR_RUN_TIMEOUT",
                "指标计算超过安全时限，已终止本次运行。",
                status_code=504,
            ) from exc


__all__ = [
    "AdaptiveComputeEngine",
    "PARALLEL_ENGINE_VERSION",
    "SharedArrayDescriptor",
    "SharedArrayOwner",
    "attach_shared_array",
    "score_matrix",
    "warm_parallel_numeric_backend",
]
