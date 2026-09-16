# Reference confidence API handoff — 2026-09-14

Backend contract v1. All paths are under `/api/historical-regimes`. Existing error envelope/status handling is retained. This document describes the implementation target; actual verification is in the backend acceptance log.

## Definitions and references

Optional `definition.study`: `{purpose:'historical_reference'|'realtime_recognition', family:'market_trend'|'macro_growth_inflation'|'risk'|'financial_conditions'|'custom', reference?:{run_id,publication_id,content_hash}, state_mapping?:Record<realtime_state_id,reference_state_id>, calibration_id?:string}`. Purpose requires retrospective/realtime respectively on execution, and sets default_mode. Historical reference forbids reference/mapping/calibration. Absent study is omitted in old serialization; existing hashes remain unchanged. Mapping defaults only to identical IDs, never order/name. References are evaluation labels, never graph inputs.

`GET /references` → `{items:[{run_id,publication_id,content_hash,definition_id,definition_revision,name,frequency,states,as_of,created_at,series_summary}]}`. Projects verified immutable retrospective published v2 runs (research_display/product_research accepted); manual events excluded. Use existing save + `enable_research_version` to confirm a reference; no new reference POST or second reference store.

## Preview

`POST /reliability/preview`:

```json
{"definition_id":"saved-id","revision":1,"reference":{"run_id":"...","publication_id":"...","content_hash":"..."},"policy":{"calibration_end":"2020-12-31","validation_end":"2022-12-31","test_end":"2024-12-31","minimum_samples":60,"minimum_class_samples":10,"minimum_segments":3,"bins":10,"transition_tolerance":3,"confidence_floor":0.6,"calibration_method":"auto"},"compile_token":"optional prepared graph token"}
```

Policy: ISO dates; calibration_end required. Optional validation_end enables separate calibration/validation/final-test blocks; omitted means calibration + independent holdout, no tuning/model-selection claim. test_end defaults to reference as_of. Thresholds are configurable policy, not statistical guarantees. Bounds: samples 2..20000; class samples 1..20000; segments 1..1000; bins 2..30; tolerance 0..60; floor 0..1. Synchronous bounded execution: <=20000 observations, <=12 states, <=12 expanding folds, one active preview per service. Existing graph executor handles latent train/mapping cutoff and strict PIT. No implicit prepare or request compilation; prepare saved definition explicitly when missing current-worker plan.

Response `{preview_hash,request,report}`. `request` is canonical with policy defaults, no compile token. `report`:

- `schema_version:'1.0'`, `status:'retrospective_only'|'insufficient_evidence'|'eligible'`, `warnings:string[]`.
- `states`: reference state dictionary; `sample`: input/matched/unknown_reference/prediction_abstentions/missing_prediction/invalid_prediction_labels/excluded_dates and per-block sufficiency.
- `classification`: confusion (reference rows, prediction columns + abstention), per_state precision/recall/F1/IoU, accuracy, balanced_accuracy, macro_f1; null when undefined.
- `intervals`, `transitions`: one-to-one matches, misses/false events, observation-step delays; unavailable values null.
- `probability`: raw/calibrated/class_base metrics by time block; Brier=sum over classes, logloss epsilon=1e-12, bins and ECE; missing probability coverage disclosed.
- `calibration`: method, evidence_type, parameters, calibration_end, validation_end, test_end, label_known_at, deployment_eligible, reasons, available_from, expires_on; unavailable result null with reason. Raw onehot is deterministic evidence. Rule calibration is class-average confusion distribution. Temperature only verified graph probability lineage.
- `lineage`: exact definition/reference hashes, source snapshots, state_mapping, probability provenance, fold training cutoffs, execution audit. `stability`/`confidence_interval` explicitly unavailable unless executed.
- `points`: bounded full aligned records (<=20000), `{observation_date,reference_state,predicted_state,block,raw_probabilities,calibrated_probabilities,calibrated_confidence,decision_status}`. Confidence is q[predicted_state], not max(q). Probability axis is reference IDs. No result => null.

Historical full-input labels become known only at full cutoff. Retrospective OOS agreement is not ground truth or deployment validation; confirmation preserves experimental evidence without backdating availability. Unknown reference and prediction abstention are distinct. Same frequency and exact source target identity required; unsupported cross-frequency/target mapping fails, no ffill.

## Confirmation/read

`POST /reliability/confirm` `{request:<canonical preview request>,preview_hash:<hash>}` → `{id,calibration_id,created_at,immutable:true,preview_hash,request,report,content_hash}`. Client reports are forbidden. Server uses immutable in-memory preview (TTL 15 minutes), verifies request and exact current reference/definition/source provenance again; expired or changed inputs return conflict and require new preview. Confirm is idempotent for same preview. GET `/reliability/reports/{id}` returns that immutable object with integrity validation. GET `/reliability/catalog` → `{items:[{id,calibration_id,created_at,definition_id,revision,reference,status,calibration}]}`. Server-generated IDs only, atomic JSON stores, bounded point arrays; no mutable calibrators added to historical runs.

TAA new study protocol uses the shared calibrated-output gate, requiring exact eligible artifact, reference/mapping/model identity, actual availability and expiry. Otherwise explicit SAA fallback. Legacy saved no-study contracts and non-regime modes retain existing behavior. UI must not render fit success as deployment success.

## Final implementation clarifications

- Current reports emit `retrospective_only` or `insufficient_evidence`. `eligible` is reserved: there is no verified model/reference selection-history ledger, so this version deliberately does **not** issue a deployment-eligible calibration. Confirmation saves experimental evidence; new-study TAA falls back to SAA. Existing no-study replay stays unchanged. This is a remaining deployment requirement, not a passed gate.
- Runtime PIT validation reuses existing `audit_execution` (prefix and future-tail perturbation). `stability.status='causal_probes_executed'` includes that temporal audit; parameter sensitivity remains explicitly not executed.
- `selective_classification` adds coverage/error/confusion after calibrated confidence_floor; original `classification` retains original predictions and abstentions. `points` additionally exposes data_available_at/recognized_at. Predictions not available by the end of their block abstain from that block.
- Exact source matching includes frozen source version/checksum and target field. Unsupported posterior lineage (including component-map/ensemble transforms without a verified posterior axis) uses final-state class_frequency; explicit temperature is rejected. Confirmation nodes retain genuine ancestor posterior axes and always use q[final_state].
- `lineage.model_binding_hash` excludes only mutable definition envelope metadata and study.calibration_id, allowing a new immutable revision to attach an artifact without a circular hash. Graph, parameters, data, reference and state mapping remain bound; exact original definition hash/revision is also retained.
- Fixed policy `reference-confidence/1`: expiry 90 calendar days after available_from; earliest availability is the day **after** preview/confirmation because inputs have day precision. No intraday/backdated availability claim.
- One-to-one interval matching greedily consumes greatest overlap first, deterministic ties; unmatched reference segments count zero in equal-segment IoU. Unknown reference spans do not create prediction false-event penalties. It is not a global optimal assignment solver. Transition delay is nearest unused same ordered-pair event, earlier date breaks ties, in observation steps.
- Execution bounds: 64 required nodes, 4 latent models, at most 100 fit iterations/model, 20000 rows/source/output/reference, <=12 folds. Existing temporal probes allow <=4 data sources. Cooperative 120-second budget is checked between phases/folds; it cannot interrupt one running NJIT call. Preview cache: <=8 entries, 15-minute TTL. Reports use a per-server-generated-hash AtomicJsonStore artifact file; catalog keeps summaries. Confirm reruns the exact inputs and compares hashes, excluding wall-clock audit duration from semantic hash.
- Confidence interval/bootstrap and deployment validation are unavailable. No fake CI or validation pass is emitted.
