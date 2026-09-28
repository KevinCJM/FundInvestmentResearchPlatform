import { systemText, useI18n } from '../../i18n/runtime'
import { Field, inputClass, NumberInput, percentText } from '../risk-models/ResearchUI'
import { percentInputValue } from '../../services/strategicAllocation'

export function riskBudgetError(assets: string[], budget?: Record<string, number> | null): string | null {
  if (budget == null) return null
  if (Object.keys(budget).length !== assets.length || assets.some(a => !Number.isFinite(budget[a]) || budget[a] < 0)) return systemText('preInvestment.riskBudgetEditor.enterANonnegativeRiskBudgetForEach')
  if (Math.abs(Object.values(budget).reduce((sum, n) => sum + n, 0) - 1) > 1e-8) return systemText('preInvestment.riskBudgetEditor.riskBudgetPercentagesMustTotal100')
  return null
}

export default function RiskBudgetEditor({ assets, assetLabels = {}, value, onChange }: { assets: string[]; assetLabels?: Record<string, string>; value?: Record<string, number> | null; onChange: (value: Record<string, number> | null) => void }) {
  useI18n()
  const error = riskBudgetError(assets, value)
  return <div className="space-y-3 border-t border-slate-200 pt-4" aria-label={systemText('preInvestment.riskBudgetEditor.riskBudgetComparisonSettings')}>
    <label className="flex min-h-10 items-center gap-2 text-sm"><input type="checkbox" checked={value != null} onChange={e => onChange(e.target.checked ? Object.fromEntries(assets.map(a => [a, NaN])) : null)} />{systemText('preInvestment.riskBudgetEditor.includeARiskBudgetCandidate')}</label>
    {value != null && <>
      <p className="text-sm leading-6 text-slate-600">{systemText('preInvestment.riskBudgetEditor.specifyEachAssetSDesiredRiskShare')}</p>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{assets.map(a => <Field key={a} label={systemText('preInvestment.riskBudgetEditor.riskBudget', { p0: assetLabels[a] || a })}><NumberInput className={`${inputClass} tabular-nums placeholder:text-slate-600 placeholder:opacity-100`} value={percentInputValue(value[a])} min={0} max={100} onValueChange={n => onChange({ ...value, [a]: n / 100 })} /></Field>)}</div>
      <p className="text-sm tabular-nums text-slate-600">{systemText('preInvestment.riskBudgetEditor.totalRiskBudget')}{percentText(Object.values(value).reduce((sum, n) => sum + n, 0))}</p>
      {error && <p role="status" className="text-sm text-amber-800">{error}</p>}
    </>}
  </div>
}
