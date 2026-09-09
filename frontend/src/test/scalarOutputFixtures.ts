import type { EvaluationResult, IndicatorDefinition, IndicatorDraft, MetricPresentation } from '../services/customIndicators'

export const scalarDraft: IndicatorDraft = {
  name: '研究摘要', description: '一次计算多个命名结果', expression: 'mean(returns)',
  unit: '%', display_format: 'percent', precision: 2, direction: 'higher_better',
  annual_risk_free_rate_percent: 0, dsl_version: '2.3.0', operator_registry_version: '2.3.0',
  context_kind: 'single_product', result_kind: 'scalar', output_contract: 'scalar',
}

export const scalarBundle: IndicatorDefinition = {
  ...scalarDraft, expression: '', id: 'test-bundle', revision: 1, source: 'custom', read_only: false,
  created_at: '2026-01-02', updated_at: '2026-01-02', result_kind: 'scalar_bundle', output_contract: 'scalar_bundle', direction: 'neutral',
  scalar_outputs: [
    { id: 'mean', label: '平均收益', expression: 'mean(returns)', unit: '%', display_format: 'percent', precision: 2, direction: 'higher_better', display_latex: '\\mu(r)' },
    { id: 'beta', label: 'Beta', expression: '1', unit: '', display_format: 'number', precision: 3, direction: 'neutral', display_latex: '1' },
    { id: 'missing', label: '不可计算结果', expression: 'mean(returns) / 0', unit: '', display_format: 'number', precision: 2, direction: 'neutral' },
  ],
}

export const bundlePresentation: MetricPresentation = {
  indicator_id: scalarBundle.id, revision: 1, name: scalarBundle.name, source: 'custom', category: 'other',
  category_label: '其他指标', context_kind: 'single_product', catalog_status: 'current', result_kind: 'scalar_bundle',
  display_format: 'number', precision: 2, unit: '', notation: 'standard', value_scale: 1, output_measure: 'scalar_bundle',
  direction: 'neutral', description: '', methodology: '', data_basis: '测试夹具', minimum_observations: 1, applicable_product_kinds: ['etf', 'fund'],
}

export const bundleResult: EvaluationResult = {
  indicator_id: scalarBundle.id, indicator_revision: 1, indicator_name: scalarBundle.name,
  target: { kind: 'etf', product_id: '510050.SH', name: '测试产品' }, period: 'ALL', value: null,
  status: 'warning', warnings: [], result_kind: 'scalar_bundle', window_scope: 'common',
  window: { requested_as_of: null, start_date: '2026-01-02', end_date: '2026-01-09', observation_count: 5, data_latest_date: '2026-01-09', effective_as_of: '2026-01-09' },
  presentation: bundlePresentation,
}
bundleResult.outputs = scalarBundle.scalar_outputs!.map((output, index) => ({
  ...bundleResult, outputs: undefined, result_kind: 'scalar', output_id: output.id,
  value: [0.032, 1.1, null][index], status: index === 2 ? 'warning' : 'ok',
  warnings: index === 2 ? [{ code: 'NON_FINITE_RESULT', message: '请检查除零。' }] : [],
  indicator_name: `${scalarBundle.name} · ${output.label}`,
  presentation: { ...bundlePresentation, name: `${scalarBundle.name} · ${output.label}`, result_kind: 'scalar', output_id: output.id,
    output_label: output.label, display_format: output.display_format, precision: output.precision, unit: output.unit,
    value_scale: output.display_format === 'percent' ? 100 : 1, direction: output.direction, output_measure: 'dimensionless' },
}))
