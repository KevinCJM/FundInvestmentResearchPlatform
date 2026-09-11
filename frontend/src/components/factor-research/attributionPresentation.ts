import type { AttributionRun, ContributionProduct, NumberValue } from '../../services/factorResearch'

// Presentation only: financial aggregation is always returned by the backend.
export const contributionText = (value: NumberValue | undefined): string =>
  value == null || !Number.isFinite(value) ? '—' : `${(value * 100).toLocaleString('zh-CN', { maximumFractionDigits: 6 })} 个百分点`
export const contributionStatus = (value: string): string => ({
  complete: '完整对账', incomplete: '数据或拟合有缺口', empty: '无可评价日期', numerical_error: '数值对账失败',
}[value] || value)

function csvCell(value: unknown): string {
  if (value == null || typeof value === 'number' && !Number.isFinite(value)) return ''
  // Keep negative numeric returns numeric, but neutralize spreadsheet formulas in labels.
  let text = String(value)
  if (typeof value === 'string' && /^[\s]*[=+\-@\t\r]/.test(text)) text = "'" + text
  return '"' + text.replace(/"/g, '""') + '"'
}

export function attributionCsv(run: AttributionRun, product: ContributionProduct): string {
  const analysis = run.attribution
  if (!analysis) throw new Error('旧运行未保存逐日贡献，请重新运行。')
  const factorLabels = analysis.components.filter(item => item.kind === 'factor').map(item => item.label)
  const rows: unknown[][] = [[
    'run_id', 'product_code', 'date', 'sample', 'status', 'reason', 'mode', 'units', 'fit_start', 'fit_end', 'fit_observations',
    'actual_return', ...factorLabels.map(name => name + ':beta'), ...factorLabels.map(name => name + ':factor_return'),
    ...analysis.components.map(item => item.label + ':contribution'), 'contribution_sum', 'reconciliation_error',
  ]]
  for (const row of product.daily) rows.push([
    run.id, product.code, row.date, row.sample, row.status, row.reason, analysis.mode, analysis.units,
    row.fit_start, row.fit_end, row.fit_observations, row.actual_return, ...row.exposures, ...row.factor_returns,
    ...row.contributions, row.contribution_sum, row.reconciliation_error,
  ])
  return '\uFEFF' + rows.map(row => row.map(csvCell).join(',')).join('\r\n')
}
