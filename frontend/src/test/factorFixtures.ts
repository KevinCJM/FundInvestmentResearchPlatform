import type { FactorDefinition, FactorRun, ResearchCatalog, Study } from '../services/factorResearch'
export const factorAudit = { execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0, request_time_compilation: 0, kernel_signatures: { test_fixture: ['(float64[:,::1])'] } }
export const fixtureFactors: FactorDefinition[] = [
  { id: 'factor-test-momentum', revision: 1, read_only: true, name: '测试中期动量', description: '仅用于界面测试的因子定义', operator: 'momentum', window: 126, skip: 21, direction: 1, product_kinds: ['etf', 'fund'] },
  { id: 'factor-test-volatility', revision: 1, read_only: true, name: '测试低波动', description: '仅用于界面测试', operator: 'volatility', window: 63, skip: 0, direction: -1, product_kinds: ['etf', 'fund'] },
  { id: 'factor-test-drawdown', revision: 1, read_only: true, name: '测试回撤控制', description: '仅用于界面测试', operator: 'drawdown', window: 126, skip: 0, direction: 1, product_kinds: ['etf', 'fund'] },
]
export const fixtureStudy: Study = {
  id: 'factor-study-test', revision: 1, created_at: '2026-09-06T00:00:00Z', updated_at: '2026-09-06T00:00:00Z',
  name: '界面测试三因子', product_kind: 'etf', asset_class: 'equity', market: 'CN', currency: 'CNY',
  targets: ['510300.SH', '510500.SH', '512100.SH'], universe_source: 'manual_fixed',
  start_date: '2020-01-01', end_date: '2026-09-03', oos_date: '2024-01-01',
  benchmark: { kind: 'etf', code: '510300.SH', label: '测试基准', return_basis: 'adjusted_nav' },
  factors: fixtureFactors.map((factor, i) => ({ factor_id: factor.id, revision: 1, weight: [.5, .3, .2][i] })),
  normalization: 'rank', horizon: 21, quantiles: 3, top_n: 2, cost_bps: 5,
  model: 'characteristic_composite', dataset: 'active_adjusted_nav',
}
export const fixtureCatalog: ResearchCatalog & { execution: typeof factorAudit } = {
  factors: fixtureFactors, models: [], contexts: [],
  snapshot: { id: 'offline-ui-fixture', latest_date: '2026-09-03' }, ready: true, execution: factorAudit,
  capabilities: [{ kind: 'etf', available: true }, { kind: 'fund', available: true }, { kind: 'stock', available: false }],
  default_study: fixtureStudy,
}
const stats = { observations: 20, mean: .15, std: .3, icir: .5, positive_rate: .6 }
const summary = {
  factors: [...fixtureFactors.map(factor => ({ name: factor.name, factor_id: factor.id, ic: stats, rank_ic: stats })),
    { name: '组合因子', factor_id: 'composite', ic: stats, rank_ic: stats }],
  performance: { days: 200, total_return: .12, annualized_return: .15, annualized_volatility: .18, max_drawdown: -.1,
    benchmark_return: .08, excess_return: .04, turnover: 3.5, fee_sum: .002 },
  group_returns: [stats, stats, stats], factor_correlation: [[1, .2, .3], [.2, 1, .4], [.3, .4, 1]],
}
export const fixtureRun: FactorRun = {
  kind: 'run', id: 'factor-run-test', name: fixtureStudy.name, created_at: fixtureStudy.created_at,
  study_id: fixtureStudy.id, study_revision: 1, study_snapshot: fixtureStudy, factor_snapshots: fixtureFactors,
  as_of: '2026-09-03', input_checksum: 'a'.repeat(64), engine_version: 'offline-ui-fixture', execution: factorAudit,
  summaries: { in_sample: summary, out_of_sample: summary },
  periods: [{ date: '2024-01-31', entry_date: '2024-02-01', label_end: '2024-03-01', sample: 'out_of_sample', ic: [.1, .2, .3, .2], rank_ic: [.1, .2, .3, .2], pair_counts: [3, 3, 3, 3], group_returns: [.01, .02, .03] }],
  curves: [{ date: '2024-01-01', nav: 1, benchmark_nav: 1, turnover: 1, cost: .001 }, { date: '2024-01-02', nav: 1.02, benchmark_nav: 1.01, turnover: 0, cost: 0 }],
  latest_scores: fixtureStudy.targets.map((code, i) => ({ product_id: code, code, name: '测试ETF' + i, kind: 'etf', score: 90 - i * 20, rank: i + 1, percentile: 1 - i / 2, status: 'ranked', exclusion_reason: null,
    factors: fixtureFactors.map(factor => ({ factor_id: factor.id, revision: 1, name: factor.name, raw_value: .2, normalized_value: .8, contribution: 20 })) })),
  warnings: ['界面测试夹具，不是真实研究结果。'], data_quality: [], data_lineage: { snapshot: 'offline-ui-fixture', generation: 'test' },
}
