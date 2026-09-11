import type { FactorDataset, ReturnPlan, ReturnPlanDraft } from '../../services/factorResearch'

export const returnMethodName = (method?: string) => ({
  characteristic_spread: '特征分组差额', ff3_2x3: 'FF3 风格 2×3', external_import: '外部导入',
}[method || 'external_import'] || method)

export function isFF3Dataset(dataset: FactorDataset): boolean {
  const names = dataset.factor_names || ['MKT_RF', 'SMB', 'HML']
  return names.length === 3 && ['MKT_RF', 'SMB', 'HML'].every(name => names.includes(name)) && dataset.dependent_return !== 'total'
}

export function returnPlanDraft(plan?: ReturnPlan): ReturnPlanDraft {
  if (!plan) return { name: '特征收益率研究', method: 'characteristic_spread', source_run_id: null, source_panel_id: null, factor_key: 'composite', quantiles: 3, cost_bps: 0, output_factor: 'SPREAD' }
  const { name, method, source_run_id, source_panel_id, factor_key, quantiles, cost_bps, output_factor } = plan
  return { name, method, source_run_id, source_panel_id, factor_key, quantiles, cost_bps, output_factor }
}

export async function readJsonFile(file: File, maxBytes = 8_000_000): Promise<unknown> {
  if (file.size > maxBytes) throw new Error(`文件过大，请缩小研究区间或产品池（上限 ${Math.floor(maxBytes / 1_000_000)} MB）。`)
  try { return JSON.parse(await file.text()) }
  catch { throw new Error('文件不是有效的 JSON，请核对格式后重试。') }
}

export function downloadJson(filename: string, value: unknown): void {
  downloadBlob(filename, new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }))
}

export function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url; anchor.download = filename
  document.body.appendChild(anchor); anchor.click(); anchor.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 0)
}
