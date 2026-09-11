import type { LibraryEvent } from '../services/eventLibrary'
import type { TemporalCapability } from '../services/regimeGraph'

export const libraryEventFixture: LibraryEvent = {
  id: 'event-demo', revision: 1, content_hash: 'a'.repeat(64), created_at: '2026-09-10T00:00:00Z', updated_at: '2026-09-10T00:00:00Z',
  name: '全球供应链事件', name_en: 'Supply chain event', description: '待核验的人工研究窗口，不是官方定界。',
  categories: ['supply_chain', 'geopolitical'], regions: ['中东'], fact_start: null, fact_end: null,
  date_precision: 'unknown', status: 'unknown', verification: 'unreviewed', known_at: null, sources: [],
  windows: [{ id: 'acute', label: '急性窗口', start_date: '2020-01-01', end_date: '2020-01-20', rationale: '初始市场压力观察窗口。' },
    { id: 'extended', label: '扩展窗口', start_date: '2020-01-01', end_date: '2020-02-01', rationale: '包含恢复阶段，不等于事件结束日。' }],
  color: '#7c3aed', archived: false, provenance: { kind: 'manual' },
}
export const temporalFixture: TemporalCapability = {
  policy_version: 'regime-temporal/1', status: 'conditional', label: '可按当时信息试算', mode: 'realtime',
  verified: false, realtime_supported: true, semantic_hindsight: false, may_repaint: false,
  reasons: [], outputs: { state: { status: 'conditional', reasons: [], node_ids: ['market', 'threshold'] } },
  nodes: [], runtime_audit: 'not_run', note: '有限探针不是所有输入的数学证明。',
}
