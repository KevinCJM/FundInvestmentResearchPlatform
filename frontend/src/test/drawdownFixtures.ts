import type { IndicatorDraft, IndicatorOperator, IndicatorVariable } from '../services/customIndicators'
import { scalarDraft } from './scalarOutputFixtures'

export const drawdownOperator: IndicatorOperator = {
  name: 'drawdown_analysis', label: '最大回撤分析', signature: 'series → record', return_type: 'record', output_shape: 'record',
  latex_template: String.raw`\operatorname{drawdown_analysis}(p)`,
  parameters: [{ name: 'values', label: '净值', shape: 'series' }],
  parameter_sets: [{ arity: 1, parameters: [{ name: 'values', label: '净值', shape: 'series' }] }],
  output_ports: [
    { id: 'max_drawdown', label: '最大回撤', description: '回撤幅度', type: { kind: 'scalar' }, output_measure: 'return_decimal', unit: '%', display_format: 'percent', precision: 2, direction: 'lower_better' },
    ...[['decline_periods', '最大回撤下跌期数'], ['recovery_periods', '最大回撤恢复期数'], ['longest_underwater_periods', '最长水下期数']].map(([id, label]) => ({
      id, label, description: '观察间隔数', type: { kind: 'scalar' }, output_measure: 'count', unit: '期', display_format: 'number' as const, precision: 0, direction: 'neutral' as const,
    })),
  ],
}
export const drawdownVariables: IndicatorVariable[] = [
  { name: 'adjusted_nav', label: '复权净值', latex: String.raw`\mathbf{p}_{\mathrm{adj}}`, value_type: 'series', dtype: 'float64', shape: 'series', semantic: 'adjusted_nav' },
  { name: 'market_close', label: '收盘价', latex: String.raw`\mathbf{c}`, value_type: 'series', dtype: 'float64', shape: 'series', semantic: 'raw_market_price' },
  { name: 'returns', label: '收益率', latex: String.raw`\mathbf{r}`, value_type: 'series', dtype: 'float64', shape: 'series', semantic: 'return_decimal' },
]
export const drawdownDraft: IndicatorDraft = {
  ...scalarDraft, name: '最大回撤分析', expression: '', result_kind: 'scalar_bundle', output_contract: 'scalar_bundle',
  scalar_outputs: drawdownOperator.output_ports!.map(({ type: _type, ...port }) => ({ ...port, expression: `drawdown_analysis(adjusted_nav).${port.id}` })),
}
