# Regime completion API handoff — 2026-09-15

Backend/frontend contract. Implementation and test status is recorded separately in
`regime-completion-backend-acceptance-2026-09-15.md`. Existing reliability fields remain.

## Historical quality

- `POST /api/historical-regimes/reference-quality/preview`
- `POST /api/historical-regimes/reference-quality/confirm`
- `GET /api/historical-regimes/reference-quality/reports/{report_id}`
- `GET /api/historical-regimes/reference-quality/catalog` → `{items: [...]}`

Preview request:
```json
{"definition_id":"saved-id","revision":1,"mode":"retrospective","as_of":null,"compile_token":null,"policy":{"stability":{"enabled":true,"max_variants":6,"perturbation":0.1,"parameters":true,"windows":true,"seeds":true,"truncation":true},"include_price_returns":true}}
```
`mode` only accepts retrospective. `policy` is required; its fields have the above defaults.
`as_of` is an optional ISO date. No client graph, labels, report, truth flag, or provenance assertion is accepted.
Confirm request is `{request: <preview.request>, preview_hash: <preview.preview_hash>}`.
Preview returns `{preview_hash, request, report}`; confirm additionally returns
`{id, created_at, immutable:true, content_hash}`. Report id prefix is `reference-quality-`.
Catalog item: `{id,created_at,definition_id,revision,status}`. Preview does not publish,
save reference arrays, or persist business artifacts. Confirmation verifies saved revision,
actual source bytes and report hash again. Existing research publication remains separate.

Report:
- `schema_version: "1.0"`, `kind: "historical_reference_quality"`, `status: "diagnostic_only" | "insufficient_evidence"`.
- `sample: {input,classified,unknown,coverage,head_unknown,tail_unknown,first_date,last_date}`.
- `segments: {total,transitions,per_state:[{state_id,observations,segments,min_length,median_length,max_length,mean_length,price_return_samples,mean_price_return,price_return_reason}]}`.
- `price_returns: {status:"available"|"unavailable"|"disabled",reason,semantics}`.
- `stability`: shared diagnostic schema below.
- `lineage`: exact definition hash, execution snapshots and evaluation snapshot.
- `warnings: string[]`, `execution`: warmed NJIT audit.

Lengths/distances count observations, not calendar days. Returns are decimal simple
endpoint returns within each contiguous state segment, only on a verified price/NAV
source and valid positive finite endpoints. Unknown states split segments. No price
semantics means returns are null with reason. Unknown head/tail count consecutive
unclassified observations; an all-unknown series has both equal to input length.

## Shared stability

Policy `stability` is optional on reliability policy and defaults as in the example.
`max_variants` 1..12; `perturbation` >0..0.5. Flags choose parameter/window/seed/truncation
families. Deterministic declared variants only; seeds fixed server-side. Total budget
120 seconds, 20000 observations/input, 64 required nodes, 4 fitted models, 100 iterations/model.
Each variant obtains a matching preparation plan; realtime fitted variants reuse frozen
baseline expanding-fold cuts. State IDs are never permuted to improve agreement.

Result: `{status:"completed"|"partial"|"disabled"|"not_applicable",variants:[],seed_status,limits,reason}`.
Each variant: `{kind:"parameter"|"window"|"seed"|"truncation",status:"completed"|"failed"|"budget_exceeded",changes:[],graph_hash,reason,agreement,comparable_observations,classification_coverage,boundary_distance,boundary_distance_unit:"observation_steps",...}`.
Changes are `{node_id,parameter,before,after}`. Failed/unavailable numerical results
are null. Agreement denominator is paired classified observations on the same date axis;
coverage denominator is baseline dates retained for that comparison. Boundary distance
is symmetric nearest boundary distance for the same ordered state pair; null if no
matching boundaries. This is sensitivity, not accuracy. Rule-only seed_status is
`not_applicable`; no fabricated seed accuracy. Truncation removes the last 10% of baseline
observation dates and compares only the retained prefix (recorded in variant metadata).

Reliability nests this under `report.stability.parameter_sensitivity`; existing
`report.stability.temporal_audit` remains.

## Paired moving-block bootstrap

Optional reliability `policy.bootstrap` defaults:
```json
{"enabled":true,"replicates":200,"block_length":10,"confidence_level":0.95,"minimum_blocks":5,"minimum_cycles":3,"minimum_valid_replicates":100}
```
Bounds: replicates 20..500; block_length 2..250; confidence_level 0.8..0.99;
minimum_blocks 2..100; minimum_cycles 1..100; minimum_valid_replicates 10..500 and <= replicates.
Fixed server seed 1729. Uses only holdout, or final test when validation_end exists.
No calibration/validation resampling, no fitting in bootstrap; missing observations
retain their positions inside contiguous blocks. All metrics share sampled indices.

`report.confidence_interval`:
`{status:"available"|"partial"|"unavailable"|"disabled",reason,method:"paired_moving_block",scope:"holdout"|"test",conditional_on:"fixed_model_reference_and_calibrator",confidence_level,block_length,replicates,seed:1729,samples,full_blocks,complete_cycles,metrics:{...}}`.
Metrics: `accuracy`, `accepted_coverage`, `accepted_error`, `brier`, `paired_brier_improvement`.
Each is `{estimate,lower,upper,valid_replicates,reason,unit:"fraction"|"brier_score"}`.
Accuracy counts abstention as mismatch against known reference. Accepted coverage/error
use calibrated confidence_floor. Brier uses calibrated probabilities. Paired improvement
is class-baseline Brier minus calibrated Brier on exactly the same valid rows. Insufficient
full blocks/cycles/valid replicates returns null bounds with reason. Finite point estimates
remain visible; unavailable estimates are never coerced to zero.
Intervals describe historical metrics, never today's hidden state or economic truth.

Each reliability point adds `probability_evidence`: null for rules/unverified sources,
or `{selected_probability,top_probability,second_probability,margin,entropy,entropy_unit:"nats"}`
from the verified raw model posterior on the mapped reference axis. This evidence is
independent of fitted calibration and does not imply deployment eligibility.

## Templates and deployment

New editable template IDs: `historical_hmm_risk_v1`, `historical_gmm_volatility_v1`,
`historical_window_mean_change_v1`, `historical_trend_ensemble_v1`, version 1 (existing integer version contract).
Existing template APIs instantiate their full editable graphs, with default
`study.purpose=historical_reference` and `mode=retrospective`.
No BOCPD/PELT claim. Existing PS/CSI300/macro identities remain unchanged.

Deployment qualification remains separate/unproven in this handoff. These historical
quality and reliability APIs do not create eligibility;
client booleans/cutoffs cannot qualify reports. Old immutable reports remain readable.

## Implementation refinements (2026-09-15)

- Template version is integer `1`, following the existing catalog contract.
- Integer diagnostic perturbations move by at least one observation when rounding
  would otherwise erase the change; existing experiment defaults keep their prior rounding.
- Variant agreement/boundary outputs also carry `agreement_reason` and
  `boundary_distance_reason`. Coverage is the fraction of classified candidate rows;
  unpaired/missing rows remain in that denominator. No label permutation is performed.
- Seed perturbations apply only to fitted nodes using `initialization_strategy=random`.
  Quantile/explicit initialization is deterministic and reports `not_applicable`.
- At most 8 expanding folds and 8 evaluation targets. Model dimension/iteration work is
  admitted under a 100,000,000-unit conservative budget, including baseline and variants.
  120 seconds is a cooperative deadline checked between executions; running NJIT calls
  are bounded by input/iteration dimensions and are not forcibly interrupted mid-kernel.
- Bootstrap metric objects additionally expose `samples` (their actual denominator)
  and `full_blocks`. Full blocks are nonoverlapping blocks on the original time axis
  with no missing reference; Brier/improvement additionally require valid probability
  pairs throughout each block. `complete_cycles` counts sequences of fully bounded,
  contiguous state segments visiting every state; missing labels reset a cycle. Segment
  cycles and full blocks are eligibility checks, not a claim of independence.
- Quality segment returns require **every** observation in a segment to have positive
  finite price, and at least two observations. No interior missing price is bridged.
- Preparation uses the existing graph machinery with transient manifests disabled;
  newly created temporary graph plans are released after the diagnostic execution.

## Integration handoff notes

- Variants additionally expose `definition_hash`. Existing `graph_hash` is structural
  and can stay identical across different parameter values; the full definition hash
  and `changes` identify the exact variant. Existing preparation machinery decides
  whether a warmed plan is reusable from its preparation hash; no incompatible plan
  is passed directly from the baseline. Shared temporary plan leases keep concurrent
  consumers alive; explicit user preparation promotes a plan instead of losing it
  when a diagnostic ends.
- Candidate construction shares unchanged source nodes/data and only copies the
  edited parameter dictionary. New diagnostics include active peak/trough windows
  and source.constant rule bounds, and skip deprecated/shadowed parameters. Existing
  experiment defaults retain their original parameter allowlist and rounding.
- Concurrent parent integration added `prospective.py` / `test_regime_prospective.py`
  during backend work. These are outside this backend completion handoff's implemented
  deployment claim. Broad run: 782 passed, 1 latent calibration fixture failure.
  Subsequent integration run against the changing files: forward captures failed
  `SOURCE_FILE_CHECKSUM_MISMATCH` after the source file grew, because `_candidate`
  calls `load_definition` -> `_formal_source_gate` on its original locked checksum.
  Do not bypass that gate with client booleans or by deleting checksum validation.
  The parent subsequently revised that integration; the final standalone
  `test_regime_prospective.py` run passed **21 tests / 49.68 seconds**. Logs: `/tmp/regime-completion-regression.log` and
  `/tmp/regime-completion-integration.log`; latest check:
  `/tmp/regime-completion-prospective-check.log`. This handoff does not certify prospective
  deployment even if separate parent work later makes its tests pass.

- Final owned-scope verification: **206 passed / 87.91 seconds**. Includes every
  reliability/completion suite, v2/v2_p1 graph suites, historical TAA and tactical
  service/bridge suites. `/tmp/regime-completion-owned-final.log`.
