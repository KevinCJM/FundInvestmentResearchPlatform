import type { RegimeResultOverview } from './regimeResultAdapter'

// Captured from build_result_overview via backend/tests/test_historical_regime_overview.py.
export const backendOverviewFixture = {
  "as_of": null,
  "calendar": null,
  "capabilities": {
    "confidence": {
      "available": false
    },
    "effective": {
      "available": false,
      "reason": "未提供完整生效时间轴，当前色带按观测日期展示。"
    },
    "evidence": {
      "available": true
    },
    "observation": {
      "available": true
    },
    "probabilities": {
      "available": false
    }
  },
  "complete": true,
  "created_at": null,
  "data_snapshots": {},
  "date_range": {
    "end": "2026-09-11",
    "start": "2026-09-04"
  },
  "definition_hash": "definition-hash",
  "definition_id": null,
  "definition_revision": null,
  "evaluation_results": {},
  "frequency": null,
  "graph_hash": "graph-hash",
  "interval_convention": "inclusive_observation_indices",
  "mode": "realtime",
  "primary_series": {
    "artifact": null,
    "date_column": "observation_date",
    "date_range": {
      "end": "2026-09-11",
      "start": "2026-09-04"
    },
    "display_source_id": null,
    "endpoint": "/api/historical-regimes/preview-runs/preview-test/series",
    "label": "识别主对照走势",
    "node_id": null,
    "port": "state",
    "price_basis": null,
    "response_field": "items",
    "run_id": "preview-test",
    "source_kind": "final_series",
    "total": 6,
    "unit": null,
    "value_column": "value"
  },
  "revision": null,
  "run_id": "preview-test",
  "run_kind": "preview",
  "schema_version": "2.0",
  "segments": [
    {
      "confirmed_at": "2026-09-05",
      "effective_start": "2026-09-07",
      "end_date": "2026-09-04",
      "end_index": 0,
      "id": "preview-test:0:0",
      "label": "牛市",
      "observations": 1,
      "reasons": [],
      "start_date": "2026-09-04",
      "start_index": 0,
      "state_id": "bull"
    },
    {
      "confirmed_at": null,
      "effective_start": null,
      "end_date": "2026-09-07",
      "end_index": 1,
      "id": "preview-test:1:1",
      "label": "熊市",
      "observations": 1,
      "reasons": [],
      "start_date": "2026-09-07",
      "start_index": 1,
      "state_id": "bear"
    },
    {
      "confirmed_at": null,
      "effective_start": null,
      "end_date": "2026-09-11",
      "end_index": 5,
      "id": "preview-test:5:5",
      "label": "牛市",
      "observations": 1,
      "reasons": [],
      "start_date": "2026-09-11",
      "start_index": 5,
      "state_id": "bull"
    }
  ],
  "states": [
    {
      "color": "#16a34a",
      "id": "bull",
      "label": "牛市",
      "order": 1,
      "role": "positive"
    },
    {
      "color": "#dc2626",
      "id": "bear",
      "label": "熊市",
      "order": 2,
      "role": "negative"
    }
  ],
  "summary": {
    "classified": 3,
    "denominator": "all_observations",
    "state_counts": {
      "bear": 1,
      "bull": 2
    },
    "switch_count": 1,
    "switch_count_basis": "adjacent_classified_observations",
    "total": 6,
    "unknown": 3
  },
  "time_basis": "observation",
  "unknown_intervals": [
    {
      "confirmed_at": null,
      "effective_start": null,
      "end_date": "2026-09-10",
      "end_index": 4,
      "id": "preview-test:2:4",
      "label": "未分类",
      "observations": 3,
      "reason": "unknown",
      "reasons": [],
      "start_date": "2026-09-08",
      "start_index": 2,
      "state_id": "unclassified"
    }
  ],
  "unknown_state": {
    "color": "#94a3b8",
    "id": "unclassified",
    "label": "未分类",
    "reason": "原因未记录，可能为预热、缺失或规则未命中。"
  }
}

export function resultFixture(runId = 'preview-A', count = 3000) {
  const states = [
    { id: 'bull', label: '牛市', color: '#ef4444', order: 0 },
    { id: 'bear', label: '熊市', color: '#22c55e', order: 1 },
  ]
  const rows = Array.from({ length: count }, (_, index) => ({
    observation_date: new Date(Date.UTC(2010, 0, index + 1)).toISOString().slice(0, 10),
    date: new Date(Date.UTC(2010, 0, index + 1)).toISOString().slice(0, 10),
    state_id: index === 0 ? 'unclassified' : index === 500 || index === count - 1 ? 'bear' : 'bull',
    value: index === 500 ? null : 1000 + index,
    recognized_at: null, effective_date: null, confidence: null,
    probabilities: { bull: index === 0 ? null : 1, bear: index === 0 ? null : 0 },
    probability_source: 'deterministic_state',
  }))
  const intervals: RegimeResultOverview['segments'] = []
  for (let index = 0; index < rows.length; index += 1) {
    const previous = intervals[intervals.length - 1]
    const row = rows[index]
    if (previous?.state_id === row.state_id) {
      previous.end_index = index
      previous.end_date = row.observation_date
      previous.observations += 1
    } else intervals.push({
      id: 'segment-' + index, state_id: row.state_id, label: row.state_id === 'bull' ? '牛市' : row.state_id === 'bear' ? '熊市' : '未分类',
      start_date: row.observation_date, end_date: row.observation_date, start_index: index, end_index: index, observations: 1,
      confirmed_at: null, effective_start: null, ...(row.state_id === 'unclassified' ? { reason: 'unknown' } : {}),
    })
  }
  const segments = intervals.filter(interval => interval.state_id !== 'unclassified')
  const overview: RegimeResultOverview = {
    schema_version: '2.0', result_kind: 'regime_states', manual_events: [], manual_event_summary: { event_count: 0, covered_observations: 0, overlap_observations: 0, max_concurrent_events: 0 },
    run_kind: 'preview', run_id: runId, definition_id: null, definition_revision: null,
    definition_hash: runId + '-definition', graph_hash: runId + '-graph', mode: 'retrospective', as_of: null,
    data_snapshots: { prices: 'snapshot-1' }, frequency: 'D', calendar: 'SSE', time_basis: 'observation', complete: true,
    date_range: { start: rows[0]?.observation_date ?? null, end: rows[rows.length - 1]?.observation_date ?? null },
    states, segments, unknown_intervals: intervals.filter(interval => interval.state_id === 'unclassified'),
    summary: { total: count, classified: Math.max(0, count - 1), unknown: Math.min(1, count), state_counts: { bull: rows.filter(row => row.state_id === 'bull').length, bear: rows.filter(row => row.state_id === 'bear').length }, switch_count: Math.max(0, segments.length - 1), denominator: 'all_observations' },
    primary_series: { source_kind: 'final_series', run_id: runId, node_id: null, port: 'state', value_column: 'value', date_column: 'observation_date', unit: null, endpoint: '/api/historical-regimes/preview-runs/' + runId + '/series', label: runId + ' 主对照走势', total: count },
    capabilities: { observation: { available: true }, effective: { available: false, reason: '未提供完整生效时间轴。' }, probabilities: { available: false }, confidence: { available: false }, evidence: { available: true } },
  }
  return { overview, rows }
}
