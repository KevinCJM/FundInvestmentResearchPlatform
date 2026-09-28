import { systemText, useI18n } from '../../i18n/runtime'
import { Empty, Field } from '../risk-models/ResearchUI'
import { controlClass } from '../risk-scales/shared'
import { CategoryEditor, PercentInput, type CategoryReuseItem } from '../risk-scales/CategoryEditor'
import type { ReferenceInputRequest } from '../../services/riskScales'
import { cashFloorIssue, type UniverseDefinition, type StrategicAsset } from '../../services/strategicScope'
import { WarningMark } from '../WarningMark'

export default function UniverseFields({ value, onChange, cutoff, pitLocked, nameError = '', reuseCategories, cashFloor = 0 }: {
  value: UniverseDefinition; onChange: (value: UniverseDefinition) => void
  cutoff: string; pitLocked: boolean; nameError?: string
  /** 投资目标生效的现金下限；现金大类下限低于它时行内给出红色提醒，不阻断保存。 */
  cashFloor?: number
  /** 已有战略范围里配置过的大类+代理，供「常用大类」下拉一步复用；不传则按原样只显示自由文本输入。 */
  reuseCategories?: CategoryReuseItem[]
}) {
  useI18n()
  const patchAsset = (index: number, patch: Partial<StrategicAsset>) => onChange({ ...value, assets: value.assets.map((a, i) => i === index ? { ...a, ...patch } : a) })
  const setCurrency = (currency: string) => onChange({ ...value, currency, assets: value.assets.map(a => ({ ...a, currency })) })
  const sourceLabels = Object.assign({}, ...value.assets.map(asset => asset.research_proxy?.source_labels ?? {})) as Record<string, string>
  // This adapter reuses the reference editor; it does not publish a risk-scale reference.
  const reference: ReferenceInputRequest = {
    name: value.name, as_of: value.as_of, currency: 'CNY', calendar: 'SSE', frequency: 'daily', periods_per_year: 252,
    return_basis: 'selected_index_and_adjusted_product_total_return', fee_basis: 'source_embedded_no_additional_fee', fx_basis: 'same_currency_no_conversion',
    assets: value.assets.map(asset => ({ id: asset.id, name: asset.name, rationale: asset.rationale,
      asset_type: asset.research_proxy?.asset_type ?? 'market', cash_return: asset.research_proxy?.cash_return ?? null,
      components: asset.research_proxy?.components ?? [], rebalance: asset.research_proxy?.rebalance ?? (asset.research_proxy?.asset_type === 'cash' ? null : 'daily') })),
  }
  const changeReference = (next: ReferenceInputRequest, labels = sourceLabels) => onChange({ ...value, assets: next.assets.map(asset => {
    const previous = value.assets.find(item => item.id === asset.id)
    const templateRoles: Record<string, StrategicAsset['role']> = { cash: 'liquidity', rates: 'rates', credit: 'credit', equity: 'growth', gold: 'inflation' }
    const previousRole = previous?.role ?? templateRoles[asset.id] ?? 'growth'
    return { id: asset.id, name: asset.name, currency: value.currency,
      role: asset.asset_type === 'cash' ? 'liquidity' : previousRole === 'liquidity' ? 'growth' : previousRole,
      liquidity: asset.asset_type === 'cash' ? 'liquid' : previous?.liquidity ?? 'liquid', rationale: asset.rationale ?? '', source: previous?.source ?? '',
      weight_limits: previous?.weight_limits,
      research_proxy: { asset_type: asset.asset_type, cash_return: asset.cash_return, components: asset.components, rebalance: asset.rebalance,
        source_labels: Object.fromEntries(asset.components.filter(item => labels[item.series_id]).map(item => [item.series_id, labels[item.series_id]])) },
    }
  }) })
  return <div className="space-y-3">
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <div className="min-w-0 lg:col-span-2"><Field label={systemText('preInvestment.universeFields.strategicScopeName')} required><input required className={controlClass} value={value.name} aria-invalid={nameError ? true : undefined} aria-describedby={nameError ? 'scope-name-error' : undefined} onChange={e => onChange({ ...value, name: e.target.value })} />{nameError && <span id="scope-name-error" role="alert" className="mt-1 block text-xs text-rose-800">{nameError}</span>}</Field></div>
      <Field label={systemText('preInvestment.universeFields.strategicResearchDate')} required hint={pitLocked ? systemText('preInvestment.universeFields.pitIsEnabledTheResearchDateFollows') : systemText('preInvestment.universeFields.pitIsDisabledYouCanSelectThe')}><input required type="date" className={controlClass} value={value.as_of} max={cutoff} disabled={pitLocked} onChange={e => onChange({ ...value, as_of: e.target.value })} /></Field>
      <Field label={systemText('preInvestment.universeFields.strategicBaseCurrency')} required hint={systemText('preInvestment.universeFields.allAssetsUseThisCurrency')}><input required className={controlClass} value={value.currency} maxLength={3} onChange={e => setCurrency(e.target.value.toUpperCase())} /></Field>
    </div>
    <details className="border-b border-slate-200 pb-3"><summary className="min-h-10 cursor-pointer text-sm font-medium text-slate-700">{systemText('preInvestment.universeFields.additionalInformationOptional')}</summary><div className="mt-2"><Field label={systemText('preInvestment.universeFields.scopeNotes')}><textarea className={controlClass} rows={2} maxLength={2000} value={value.source} onChange={e => onChange({ ...value, source: e.target.value })} /></Field></div></details>
    {value.currency !== 'CNY' && <p className="text-sm text-amber-800">{systemText('preInvestment.universeFields.researchProxiesCurrentlySupportCnyOnlyOther')}</p>}
    <CategoryEditor heading={systemText('preInvestment.universeFields.assetClassesAndResearchProxies')} description={systemText('preInvestment.universeFields.selectIndexOrProductProxiesForNon')}
      value={reference} sourceLabels={sourceLabels} onChange={changeReference} reuseCategories={reuseCategories}
      limitsColumn={{ header: systemText('preInvestment.universeFields.weightLimits'), render: (_, index) => {
        const asset = value.assets[index]
        const limits = asset.weight_limits ?? { min_weight: 0, max_weight: 1 }
        const setLimit = (key: 'min_weight' | 'max_weight', next: number) => patchAsset(index, { weight_limits: { ...limits, [key]: next / 100 } })
        const issue = asset.role === 'liquidity' && asset.liquidity === 'liquid' ? cashFloorIssue([asset], cashFloor) : ''
        return <div className="flex items-center gap-1.5">
          <PercentInput label={systemText('preInvestment.universeFields.minWeight', { p0: asset.name })} value={limits.min_weight * 100} onValueChange={next => setLimit('min_weight', next)} />
          <span aria-hidden="true" className="text-slate-600">{'–'}</span>
          <PercentInput label={systemText('preInvestment.universeFields.maxWeight', { p0: asset.name })} value={limits.max_weight * 100} onValueChange={next => setLimit('max_weight', next)} />
          {issue && <WarningMark label={systemText('preInvestment.scopeCashFloor.label')}>{issue}</WarningMark>}
        </div>
      } }}
      renderAssetDetails={(_, index) => {
      const asset = value.assets[index]
      return <details><summary className="min-h-10 cursor-pointer text-sm font-medium leading-6 text-slate-700">{systemText('preInvestment.universeFields.notesOptional')}</summary><div className="mt-2 space-y-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label={systemText('preInvestment.universeFields.rationaleForChoosingThisAssetClass')}><textarea aria-label={systemText('preInvestment.universeFields.assetNotes', { p0: index + 1 })} rows={2} className={controlClass} maxLength={1000} value={asset.rationale} onChange={e => patchAsset(index, { rationale: e.target.value })} /></Field>
          <Field label={systemText('preInvestment.universeFields.references')}><textarea aria-label={systemText('preInvestment.universeFields.assetReferences', { p0: index + 1 })} rows={2} className={controlClass} maxLength={1000} value={asset.source} onChange={e => patchAsset(index, { source: e.target.value })} /></Field>
        </div>
      </div></details>
    }} />
    {!value.assets.length && <Empty title={systemText('preInvestment.universeFields.noAssetClassesConfigured')}><p>{systemText('preInvestment.universeFields.selectAddReferenceAssetClassToAdd')}</p></Empty>}
    {value.assets.length >= 30 && <p className="text-xs text-slate-600">{systemText('preInvestment.universeFields.aScopeSupportsUpTo30Asset')}</p>}
  </div>
}
