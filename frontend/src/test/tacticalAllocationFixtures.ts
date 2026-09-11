import type { TaaBaseline, TaaCatalog, TaaMetrics, TaaPreview, TaaPreviewRequest, TaaPreflight } from '../services/tacticalAllocation'

export const taaExecution = { backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0, request_time_compilation: 0, kernel_signatures: { allocation: ['fixed'] } }
export const taaBaseline: TaaBaseline = {
  id: 'SAA-1', name: '稳健长期组合', created_at: '2026-09-10T08:00:00Z', content_hash: 'baseline-hash', alloc_name: '股债分类', as_of: '2026-09-10', universe_snapshot_id: 'UNIVERSE-1', data_release_id: 'DATA-1', apply_eligible: true,
  assets: [
    { id: 'equity', name: '权益', base_weight: .6, min_weight: .3, max_weight: .8, max_abs_tilt: .1, products: [{ kind: 'etf', product_id: '510300.SH', name: '沪深300ETF', weight: 1 }] },
    { id: 'bond', name: '债券', base_weight: .4, min_weight: .2, max_weight: .7, max_abs_tilt: .1, products: [{ kind: 'etf', product_id: '511010.SH', name: '国债ETF', weight: 1 }] },
  ], pit: { status: 'research_only', reasons: ['当前修订数据尚未证明完整历史 PIT。'] }, lineage: { frozen_data: true },
}
export const taaCatalog: TaaCatalog = {
  allocations: [{ alloc_name: '股债分类', assets: [{ id: 'equity', name: '权益' }, { id: 'bond', name: '债券' }], as_of: '2026-09-10', coverage: { start_date: '2023-01-03', end_date: '2026-09-10' } }], baselines: [taaBaseline], decisions: [],
}
export const taaRequest: TaaPreviewRequest = { baseline_id: 'SAA-1', start_date: '2023-01-03', end_date: '2026-09-10', as_of: '2026-09-10', train_end_date: '2025-06-22', signal_mode: 'momentum', lookback: 60, manual_tilts: { equity: 0, bond: 0 }, max_abs_tilt: .1, transaction_cost_bps: 10, risk_penalty: 3, max_tracking_error: .1, max_turnover: 1, confidence_floor: .6, max_signal_age_days: 31, search: true, objective: 'active_utility', review_days: 30, note: '' }
const metrics: TaaMetrics = { total_return: .12, baseline_return: .1, excess_return: 1.12 / 1.1 - 1, annual_volatility: .08, max_drawdown: -.03, tracking_error: .02, turnover: .5, cost: .0005, score: .018 }
export const taaPreview: TaaPreview = {
  preview_hash: 'preview-hash', request: taaRequest, baseline: taaBaseline,
  data: { start_date: '2023-01-03', end_date: '2026-09-10', observations: 800, train_observations: 540, validation_observations: 260, lineage: { returns_hash: 'returns-hash' }, pit: taaBaseline.pit },
  candidates: [{ id: 'zero', strength: 0, feasible: true, validation_feasible: true, train: { ...metrics, total_return: .1, excess_return: 0, tracking_error: 0, score: 0 }, validation: { ...metrics, total_return: .1, excess_return: 0, tracking_error: 0, score: 0 } }, { id: 'full', strength: 1, feasible: true, validation_feasible: true, train: metrics, validation: { ...metrics, total_return: .1143, excess_return: 1.1143 / 1.1 - 1 } }], selected_id: 'full',
  recommendation: { is_saa: false, weights: { equity: .65, bond: .35 }, tilts: { equity: .05, bond: -.05 }, trade_deltas: null, reason: '训练区间内该候选兼顾净超额与风险，需继续复核独立验证。', signal_date: '2026-09-09', expires_on: '2099-10-10', confidence: .8, fallback_reason: null },
  chart: [{ date: '2023-01-03', baseline: 1, taa: 1, segment: 'train' }, { date: '2024-06-03', baseline: 1.04, taa: 1.05, segment: 'train' }, { date: '2025-06-20', baseline: 1.1, taa: 1.12, segment: 'train' }, { date: '2025-06-23', baseline: 1, taa: 1, segment: 'validation' }, { date: '2025-09-01', baseline: 1.06, taa: 1.07, segment: 'validation' }, { date: '2026-09-10', baseline: 1.1, taa: 1.1143, segment: 'validation' }], weight_path: [{ date: '2026-09-10', weights: { equity: .65, bond: .35 }, turnover: .1 }], warnings: ['本结果仅用于研究。'], execution: taaExecution, audit: { selection_uses_validation: false },
}

export const taaPreflight: TaaPreflight = {
  coverage: { start_date: '2023-01-03', end_date: '2026-09-10' },
  dates: { start_date: '2023-01-03', end_date: '2026-09-10', as_of: '2026-09-10', train_end_date: '2025-06-22' },
  quality: { status: 'clear', issues: [] },
  training: { eligible: true, train_observations: 540, validation_observations: 260, unavailable_count: 0, unknown_count: 0, earliest_available_date: null, reasons: [] },
  guidance: [], pit: taaBaseline.pit, can_calculate: true,
}
