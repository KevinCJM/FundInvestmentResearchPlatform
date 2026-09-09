import type { ReturnCatalog, ReturnDataset, ReturnPlan, ReturnSource } from '../services/factorResearch'
import { factorAudit, fixtureRun } from './factorFixtures'

// Offline UI fixtures only; never used as production defaults or research evidence.
export const fixtureReturnPlan: ReturnPlan = { id: 'factor-return-plan-test', revision: 1, name: '测试特征收益', method: 'characteristic_spread', source_run_id: fixtureRun.id, source_panel_id: null, factor_key: 'composite', quantiles: 3, cost_bps: 5, output_factor: 'SPREAD' }
export const fixtureReturnSource: ReturnSource = { id: 'factor-return-source-test', name: '测试股票时点面板', market: 'CN', currency: 'CNY', start_date: '2024-06-28', end_date: '2024-09-30', observations: 500, checksum: 'b'.repeat(64) }
export const fixtureReturnCatalog: ReturnCatalog = {
  ready: true,
  methods: [{ id: 'characteristic_spread', name: '特征分组差额', available: true }, { id: 'ff3_2x3', name: 'FF3 风格 2×3', available: true }, { id: 'native_stock_ff3', name: '直接从本地股票数据构建', available: false, reason: '数据尚未验收' }],
  source_template: { name: '', calendar: [], formations: [], returns: [], rf: [] },
  dataset_template: { name: '', factor_names: ['MKT_RF', 'SMB', 'HML'], dependent_return: 'excess', rows: [] },
  source_fields: { calendar: '完整交易日历', formations: 'date, asset, market_cap, december_market_cap, book_equity', returns: 'date, asset, return_value, lagged_market_cap, weight_date', rf: 'date, RF' },
}
export const fixtureReturnDataset: ReturnDataset = {
  id: 'factor-dataset-spread-test', kind: 'dataset', name: '测试特征收益', market: 'CN', currency: 'CNY', created_at: '2026-09-06T00:00:00Z',
  source_url: '', source_method: 'characteristic_spread', source_run_id: fixtureRun.id,
  factor_names: ['SPREAD'], dependent_return: 'total', frequency: 'daily', units: 'decimal_return',
  construction: '离线界面测试，非真实因子收益。', warnings: ['测试夹具，不是真实研究结果。'], input_checksum: 'c'.repeat(64), execution: factorAudit,
  plan_snapshot: fixtureReturnPlan,
  rows: [{ date: '2024-07-01', SPREAD: 0 }, { date: '2024-07-02', SPREAD: .01 }, { date: '2024-07-03', SPREAD: null }],
  diagnostics: { factors: [{ factor: 'SPREAD', observations: 2, mean: .005, std: .007, positive_rate: .5 }], correlation: [[null]], cumulative: [{ date: '2024-07-01', values: [0] }, { date: '2024-07-02', values: [.01] }, { date: '2024-07-03', values: [null] }], cumulative_meaning: '日收益算术累计，不是可投资净值。' },
}
export const fixtureFF3Dataset: ReturnDataset = {
  ...fixtureReturnDataset, id: 'factor-dataset-ff3-test', name: '测试 FF3 数据', source_method: 'ff3_2x3', source_url: 'https://example.org/offline',
  factor_names: ['MKT_RF', 'SMB', 'HML'], dependent_return: 'excess', source_panel_id: fixtureReturnSource.id, source_run_id: undefined,
  rows: [{ date: '2024-07-01', MKT_RF: .001, SMB: .002, HML: .003, RF: .0001 }], diagnostics: undefined,
  formation_evidence: [{ date: '2024-06-28', counts: { SL: 1, SM: 1, SH: 1, BL: 1, BM: 1, BH: 1 }, size_break: 6.5, bm30: 1.5, bm70: 2.5 }],
}
export const fixtureDatasetSummaries = [fixtureReturnDataset, fixtureFF3Dataset].map(dataset => ({ ...dataset, observations: dataset.rows.length, start_date: dataset.rows[0].date, end_date: dataset.rows[dataset.rows.length - 1].date }))
