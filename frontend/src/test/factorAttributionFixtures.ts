import type { AttributionRun, ContributionAnalysis, ContributionSummary } from '../services/factorResearch'
import { factorAudit } from './factorFixtures'

const summary: ContributionSummary = {
  start_date: '2024-01-02', end_date: '2024-01-03', days: 2, valid_days: 2, actual_days: 2,
  total_return: .045, contributions: [.036, .021, -.012], contribution_sum: .045, reconciliation_error: 0,
  model_r2: .9555555556, residual_volatility: .33675, coverage: 1, status: 'complete',
}
const curve = [
  { date: '2024-01-02', contributions: [.08, .01, .01], total_return: .1, contribution_sum: .1, reconciliation_error: 0 },
  { date: '2024-01-03', contributions: [.036, .021, -.012], total_return: .045, contribution_sum: .045, reconciliation_error: 0 },
]
export const fixtureContributionAnalysis: ContributionAnalysis = {
  schema_version: 1, engine_version: 'offline-hand-calculated-fixture', mode: 'fixed', dependent_return: 'total',
  linking_method: 'beginning_wealth_weighted', units: 'decimal_return_contribution', warmup_days: 0,
  evaluation_start: '2024-01-02', summary_basis: 'fixed_in_sample_fit',
  components: [{ id: 'factor_0', label: '测试风格', kind: 'factor' }, { id: 'intercept', label: '模型截距', kind: 'intercept' }, { id: 'residual', label: '未解释残差', kind: 'residual' }],
  products: [{ code: '000001.OF', name: '离线测试基金',
    daily: [
      { date: '2024-01-02', sample: 'out_of_sample', status: 'ok', reason: null, actual_return: .1, exposures: [1], factor_returns: [.08], contributions: [.08, .01, .01], contribution_sum: .1, reconciliation_error: 0, fit_start: '2023-01-01', fit_end: '2023-12-29', fit_observations: 126, fit_r2: .9, exposure_status: 'ok', exposure_basis: 'fixed_training_fit' },
      { date: '2024-01-03', sample: 'out_of_sample', status: 'ok', reason: null, actual_return: -.05, exposures: [1], factor_returns: [-.04], contributions: [-.04, .01, -.02], contribution_sum: -.05, reconciliation_error: 0, fit_start: '2023-01-01', fit_end: '2023-12-29', fit_observations: 126, fit_r2: .9, exposure_status: 'ok', exposure_basis: 'fixed_training_fit' },
    ], summaries: { all: summary, out_of_sample: summary, in_sample: { ...summary, days: 0, valid_days: 0, actual_days: 0, status: 'empty', contributions: [null, null, null], total_return: null, contribution_sum: null, reconciliation_error: null, start_date: null, end_date: null, model_r2: null, residual_volatility: null, coverage: null }, '2024-01': summary },
    curves: { all: curve, in_sample: [], out_of_sample: curve },
  }],
  notes: ['离线手算测试夹具，不是真实基金收益。'],
}
export const fixtureAttributionRun: AttributionRun = {
  id: 'factor-attribution-contribution-test', name: '离线收益贡献检验', created_at: '2026-09-06T00:00:00Z', as_of: '2024-01-03',
  request: { name: '离线收益贡献检验', product_kind: 'fund', targets: ['000001.OF'], model: 'rbsa', indices: ['INDEX0', 'INDEX1'], market: 'CN', currency: 'CNY', start_date: '2023-01-01', end_date: '2024-01-03', oos_date: '2024-01-02', exposure_mode: 'fixed' },
  execution: factorAudit, warnings: ['离线测试。'], attribution: fixtureContributionAnalysis,
  results: [{ name: '离线测试基金', code: '000001.OF', status: 'ok', reason: null, exposures: [{ factor: '测试风格', value: 1 }], train_r2: .9, test_r2: .9555555556, train_observations: 126, test_observations: 2, annualized_intercept: 2.52, train_residual_volatility: .1, test_residual_volatility: .33675 }],
}
