# Backend acceptance — 2026-09-14

Final coordinated review: the additional calibration-support defect (ample reference labels but absent or single-class predictions) was reproduced and corrected. The four dedicated reliability modules now pass **30 tests** in **43.36s**. The support counter uses the same fixed-signature, PID-warmed NJIT path; the reliability registry now contains **10 kernels**. Earlier 9-kernel benchmark figures below remain the actual earlier measurement, not a new benchmark claim. See `regime-reference-confidence-final-audit-2026-09-14.md` for the final browser, frontend and implementation-boundary record.

Scope: backend only, current ISSUE2609/BetterSaaTaa. Existing v2_templates.py, peak-trough tests and PNG preserved. No network/data download/commit/push/reset/stash.

## M1

Command: `TASK_TEST_ROOT=$(mktemp -d /private/tmp/regime-reliability-tests.XXXXXX)` then `CUSTOM_INDICATOR_DATA_DIR="$TASK_TEST_ROOT" HISTORICAL_REGIME_DATA_DIR="$TASK_TEST_ROOT" /Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest backend/tests/test_regime_reliability_contracts.py -q`.
Result: **3 passed**, 19.53s. Optional study omitted from old serialization, mode enforcement, recursive reference/mapping rejection, time policy bounds.
Initial collection without store isolation failed on a denied production lock open (no store write succeeded); all subsequent service tests use temporary store environment. Reference publication/tampering integration tests follow before M3 completion.

M2 numerical first run found Numba does not support float(bool); replaced by explicit 1.0/0.0 and reran. No Python fallback was introduced.

## M2

`/Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest backend/tests/test_regime_reliability_math.py -q`: **6 passed**, 22.94s at the first math milestone. Additional boundary cases added during audit.

Actual graph integration initially passed classification/source/frequency tests but exposed a JSON int64 persistence failure (13 passed / 1 failed, 18.66s). Reused `_json_safe` at the persistence/API boundary. A stricter PIT test then exposed that `_execute_graph` returns static capability (`verified=false`), not runtime evidence. Reliability now calls the existing `audit_execution`, retaining its real prefix/tail probes and source checks; it does not lower the gate.

## M3

With fresh temporary `CUSTOM_INDICATOR_DATA_DIR` and `HISTORICAL_REGIME_DATA_DIR`, Python command above plus `-m pytest backend/tests/test_regime_reliability_contracts.py backend/tests/test_regime_reliability_math.py backend/tests/test_regime_reliability_service.py backend/tests/test_regime_reliability_consumer.py -q`: **22 passed**, 47.78s. Includes real uploaded Parquet/graph replay, verified reference publication, full causal probes, expanding GMM folds with frozen training/mapping, API preview/confirm/get/catalog, idempotency/tamper/TTL/prepare gates, both TAA consumers, legacy deterministic replay, chosen-state probability, expiry and actual availability. Previous run's sole remaining failure was missing tactical warmup in the test fixture; service.warm() now runs explicitly before that test.

The whole service path remains offline and uses tmp_path stores. No report supplied by the frontend is accepted. Snapshots remain immutable; calibration attachments on TAA read models are transient.

## Comprehensive audit and regression

Main regression, fresh temporary store environment as above:

```sh
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest backend/tests/test_regime_reliability_*.py backend/tests/test_historical_regime*.py backend/tests/test_tactical_allocation_*.py backend/tests/test_portfolio_regime_backtest.py -q
```

**417 passed**, 315.19s, 6 existing deprecation/date-parsing warnings. This includes the preexisting peak-trough/template changes without editing their algorithms.

Extended routing regression:

```sh
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest backend/tests/test_regime_granularity.py backend/tests/test_regime_segments.py backend/tests/test_regime_authoring.py backend/tests/test_regime_series_builder.py backend/tests/test_regime_math_presentation.py backend/tests/test_regime_node_preview.py backend/tests/test_regime_chart_normalization.py backend/tests/test_regime_indicator_nodes.py backend/tests/test_regime_product_sources.py backend/tests/test_regime_research_versions.py backend/tests/test_regime_event_library.py backend/tests/test_regime_temporal_capability.py backend/tests/test_manual_historical_events.py backend/tests/test_merrill_clock.py backend/tests/test_research_series_numba.py backend/tests/test_research_series_routes.py backend/tests/test_research_series_import.py backend/tests/test_scenario_stress.py backend/tests/test_scenario_stress_routes.py backend/tests/test_research_input_checks.py backend/tests/test_portfolio_research.py backend/tests/test_tactical_walk_forward.py backend/tests/test_bettersaataa_final_audit.py backend/tests/test_indicator_graph.py backend/tests/test_indicator_formula_roundtrip.py -q -o faulthandler_timeout=120
```

**434 passed / 1 failed**, 131.54s. Failure: `test_main_app_mounts_research_series_and_warms_njit_against_index_only_snapshot` tried to open the developer's external data-disk lease before app startup; the sandbox denied it. Fixed that fixture to use a temporary StorageManager and added actual startup assertions for reliability readiness/no compilation/no fallback. No production storage policy was changed. Targeted rerun recorded below.

Post-audit dedicated reliability suite: **26 passed**, 43.40s. Additional TAA shared-import/legacy error-contract checks: `-m pytest backend/tests/test_regime_reliability_consumer.py backend/tests/test_tactical_allocation_service.py backend/tests/test_historical_regime_taa.py -q` → **53 passed**, 75.35s. Counts across commands overlap; they are not a unique-test total.

### Actual numerical path and memory

`PYTHONPATH=.:backend NUMBA_CACHE_DIR=.numba_cache/tests /Users/chenjunming/Desktop/myenv_312/bin/python3.12 backend/tests/benchmark_regime_reliability.py`:

- 20000 observations, 3 classes, 5 repetitions: median **1.028035 seconds** for classification + one-to-one events + 41-candidate temperature fitting + independent test probability scores.
- Python-traced peak **2,574,850 bytes**; whole-process peak RSS **442,843,136 bytes** on macOS includes imports/compilation and is not an isolated native-kernel allocation measurement.
- New calibrated output **480,000 bytes**, small calibration buffers **96 bytes**. JSON decoding/state encoding/probability mapping allocate at the boundary; output/work buffers allocate as needed. No claim of whole-graph zero-copy.
- readonly strided y/p/raw views all share memory with their owners; test-block q slice shares q. Inputs remain unchanged. **9/9 kernels have one fixed readonly arbitrary-stride signature**, compilation disabled; `python_fallback=0`.
- Startup warms/exercises each kernel and records PID. A forked/unwarmed worker fails closed until its own warmup. The app's lifespan exposes this audit alongside existing readiness.

### Remaining boundaries

- No deployment-eligible artifact is issued: historical label/publication availability and model/reference selection history cannot establish a historically deployable calibration. Confirmed reports are immutable experimental evidence. New study TAA falls back with a reason; old no-study deterministic replay and non-regime modes retain their contracts.
- Dependency-aware bootstrap confidence intervals and verified deployment-selection ledger are not implemented; reports explicitly mark unavailable/insufficient. Independent holdout is distinguished from three blocks; no hyperparameter-selection claim is made from final test data.
- Unsupported posterior transformations (component-map/ensemble without verified posterior class axis) use final-state class-average calibration; explicit temperature is blocked. Interval assignment is deterministic greedy maximum overlap, not global optimal assignment.
- 120-second budget is cooperative between phases/folds; it does not interrupt a running NJIT call. Input decoding happens at existing source boundaries before output budgeting.
- No frontend, main design document, AGENTS, routing JSON, production data, existing template/peak-trough algorithm or PNG was edited by this worker. No commit/push/reset/stash/download/network tests.
- Routing evolution check covered implementation paths but flagged newly added tests as absent from routing lists. These JSON changes are outside this worker's ownership and remain for the coordinating worker. No routing facts were silently changed.

### Final completion evidence

Targeted app-startup + reliability + TAA rerun, isolated stores:

```sh
/Users/chenjunming/Desktop/myenv_312/bin/python3.12 -m pytest backend/tests/test_research_series_routes.py::test_main_app_mounts_research_series_and_warms_njit_against_index_only_snapshot backend/tests/test_regime_reliability_*.py backend/tests/test_tactical_allocation_service.py backend/tests/test_historical_regime_taa.py -q -o faulthandler_timeout=120
```

**77 passed**, 284.93s. The 120-second diagnostic traceback showed ongoing preexisting indicator batch-plan startup compilation; the test subsequently completed successfully. App readiness assertions passed. The one failed case from the extended 435-case run is therefore resolved; unrelated extended tests were not needlessly repeated.

The last consumer compatibility adjustment was separately checked by the **53-pass** consumer/TAA command above. Both TAA callers use the same canonical `historical_regimes.reliability.consumer` module so the warmed runtime is shared, rather than accidentally loading an unwarmed duplicate through the backend-prefixed import. Legacy missing-state handling and fallback-reason priority were retained.

Read-only comparison against `git show HEAD:backend/historical_regimes/v2_contracts.py` confirmed the old no-study fixture hash is byte-for-byte unchanged: `e708eaf61fe82cd5c8914d50d75870aff718a5fc8acfab6e39e725389006b814` in both the HEAD baseline and current implementation. The historical implementation was evaluated only in memory, not retained as a second source file.

Final `git diff --check` passed. Current branch remains `ISSUE2609/BetterSaaTaa`. Preexisting template/peak-trough diffs remain 47 and 8 added lines respectively; this worker made no edits to them or the PNG. All spawned test commands finished.

Final routing coverage command (explicit owned paths only) reports **16 covered files**, with these five new test/benchmark files unregistered: `benchmark_regime_reliability.py`, `test_regime_reliability_consumer.py`, `test_regime_reliability_contracts.py`, `test_regime_reliability_math.py`, `test_regime_reliability_service.py` under `backend/tests/`. No routing JSON was changed because it is explicitly outside this worker's ownership.

Changed files: new `backend/historical_regimes/reliability/` subdomain (contracts, verified reference projection, shared graph replay, NJIT statistics/calibration, report assembly, immutable persistence, thin routes, shared TAA gate); small integration changes in `v2_contracts.py`, `v2_service.py`, `taa.py`, `backend/services/historical_regime_routes.py`, `backend/tactical_allocation/service.py`, `backend/app.py`; four dedicated test modules, one reproducible benchmark, and the isolated startup test fixture. API and this acceptance log are the only documents authored by this worker.
