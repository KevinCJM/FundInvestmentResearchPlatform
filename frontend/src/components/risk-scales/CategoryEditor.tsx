import { useId, useMemo, useState, type ReactNode } from 'react'
import { XMarkIcon } from '@heroicons/react/24/outline'
import { Badge, Button } from '../ui'
import { Field, NumberInput } from '../risk-models/ResearchUI'
import type { ReferenceAsset, ReferenceInputRequest } from '../../services/riskScales'
import { controlClass, pct, useRiskText } from './shared'
import { SourcePicker } from './SourcePicker'

/** 已保存范围里配置过的大类，可整条复用（名称 + 代理）。 */
export interface CategoryReuseItem {
  name: string
  summary: string
  proxy: Pick<ReferenceAsset, 'asset_type' | 'cash_return' | 'components' | 'rebalance'>
  sourceLabels: Record<string, string>
}

const SUPPRESSED_KEY = 'betterSaaTaa.reuseCategorySuppressed'
const readSuppressed = (): Set<string> => { try { return new Set(JSON.parse(localStorage.getItem(SUPPRESSED_KEY) ?? '[]')) } catch { return new Set() } }

const REBALANCE_RULES = ['daily', 'monthly', 'quarterly', 'yearly', 'buy_and_hold'] as const
const PALETTE_VISIBLE = 8
// 一行一个大类：名称 / 类型 / 再平衡或现金收益 / 代理与状态 / 操作。窄屏按两列坍缩。
const ROW_GRID = 'grid grid-cols-1 gap-x-3 gap-y-2 sm:grid-cols-2 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,6.5rem)_minmax(0,10rem)_minmax(0,1.5fr)_auto] lg:items-start'
// 02 研究范围多一列权重边界，插在规则与代理之间。
const ROW_GRID_LIMITS = 'grid grid-cols-1 gap-x-3 gap-y-2 sm:grid-cols-2 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,6.5rem)_minmax(0,9rem)_minmax(0,11rem)_minmax(0,1.4fr)_auto] lg:items-start'

/** 百分比输入：可访问名已含（%），可见的单位给用眼睛看的人。 */
export function PercentInput({ label, value, onValueChange }: { label: string; value: number; onValueChange: (value: number) => void }) {
  return <span className="relative block">
    <NumberInput aria-label={label} aria-required className={`${controlClass} pr-7 tabular-nums`} value={value} onValueChange={onValueChange} />
    <span aria-hidden="true" className="pointer-events-none absolute inset-y-0 right-2 mt-1 flex items-center text-xs text-slate-600">%</span>
  </span>
}

/** 窄屏显示字段名，宽屏交给表头一行；控件本身始终带可访问名。 */
function Cell({ label, children, className }: { label: string; children: ReactNode; className?: string }) {
  return <div className={`min-w-0 ${className ?? ''}`}>
    <span aria-hidden="true" className="block text-xs text-slate-600 lg:hidden">{label}</span>
    {children}
  </div>
}

/**
 * 大类与研究代理的共用编辑器：一行一个大类，代理明细与备注收进按需展开区，
 * 常用大类作为一键补齐的调色板放在清单上方。
 * 02 研究范围、08 风险标尺参考输入、05 LTCMA 统计代理共用这一个实现。
 */
export function CategoryEditor({ heading, description, value, sourceLabels = {}, onChange, fixedAssets = false, cashEligibleIds, reuseCategories = [], renderAssetDetails, limitsColumn }: {
  /** 外层已有标题时省略，不重复一层。 */
  heading?: string
  description?: string
  value: ReferenceInputRequest
  sourceLabels?: Record<string, string>
  onChange: (next: ReferenceInputRequest, labels?: Record<string, string>) => void
  /** 大类由上游冻结（05 跟随战略范围）：只配代理，不能增删改名。 */
  fixedAssets?: boolean
  /** 只有这些大类可切成现金；不传则不限制。 */
  cashEligibleIds?: string[]
  reuseCategories?: CategoryReuseItem[]
  /** 不传时展开区给一个「经济定义与依据」输入框。 */
  renderAssetDetails?: (asset: ReferenceAsset, index: number) => ReactNode
  /** 传入时多一列（如 02 的权重边界），由调用方渲染每行内容。 */
  limitsColumn?: { header: string; render: (asset: ReferenceAsset, index: number) => ReactNode }
}) {
  const { t } = useRiskText()
  const domId = useId()
  const [expanded, setExpanded] = useState<string[]>([])
  const [picking, setPicking] = useState<string | null>(null)
  const [suppressVersion, setSuppressVersion] = useState(0)
  const suppressed = useMemo(readSuppressed, [suppressVersion])

  const assets = value.assets
  const cashIndex = assets.findIndex(asset => asset.asset_type === 'cash')
  const update = (index: number, change: Partial<ReferenceAsset>) => onChange({ ...value, assets: assets.map((asset, i) => i === index ? { ...asset, ...change } : asset) })
  const newId = () => `asset-${crypto.randomUUID().slice(0, 8)}`
  const blank = (id: string, name = '', assetType: ReferenceAsset['asset_type'] = 'market'): ReferenceAsset => assetType === 'cash'
    ? { id, name, asset_type: 'cash', rationale: '', cash_return: 0, components: [], rebalance: null }
    : { id, name, asset_type: 'market', rationale: '', cash_return: null, components: [], rebalance: 'daily' }
  const open = (id: string) => setExpanded(current => current.includes(id) ? current : [...current, id])
  const toggle = (id: string) => setExpanded(current => current.includes(id) ? current.filter(item => item !== id) : [...current, id])

  const changeType = (index: number, assetType: ReferenceAsset['asset_type']) => {
    setPicking(null)
    update(index, assetType === 'cash'
      ? { asset_type: 'cash', cash_return: 0, components: [], rebalance: null }
      : { asset_type: 'market', cash_return: null, components: [], rebalance: 'daily' })
  }

  const addBlank = () => { const id = newId(); onChange({ ...value, assets: [...assets, blank(id)] }); open(id) }
  const addReuse = (item: CategoryReuseItem) => {
    const id = newId()
    onChange(
      { ...value, assets: [...assets, { ...blank(id, item.name, item.proxy.asset_type), ...item.proxy }] },
      { ...sourceLabels, ...item.sourceLabels },
    )
    open(id)
  }
  const suppress = (name: string, hidden: boolean) => {
    const next = readSuppressed()
    if (hidden) next.add(name); else next.delete(name)
    try { localStorage.setItem(SUPPRESSED_KEY, JSON.stringify([...next])) } catch { /* 仅为本机偏好，失败时不影响配置 */ }
    setSuppressVersion(version => version + 1)
  }

  const weightOf = (asset: ReferenceAsset) => asset.components.reduce((sum, item) => sum + item.weight, 0)
  const status = (asset: ReferenceAsset): { tone: 'success' | 'warning' | 'danger'; text: string } => {
    if (asset.asset_type === 'cash') {
      const rate = asset.cash_return
      return rate == null || !Number.isFinite(rate) || rate < -0.5 || rate > 1
        ? { tone: 'warning', text: t('statusCashPending') }
        : { tone: 'success', text: t('statusReady') }
    }
    if (!asset.components.length) return { tone: 'warning', text: t('statusProxyPending') }
    return Math.abs(weightOf(asset) - 1) > 1e-10
      ? { tone: 'danger', text: t('statusWeight', { value: pct(weightOf(asset)) }) }
      : { tone: 'success', text: t('statusReady') }
  }
  const readyCount = assets.filter(asset => status(asset).tone === 'success').length
  const rowGrid = limitsColumn ? ROW_GRID_LIMITS : ROW_GRID

  const taken = new Set(assets.map(asset => asset.name.trim()).filter(Boolean))
  const suggestions = reuseCategories.filter(item => !suppressed.has(item.name))
  const hiddenCount = reuseCategories.length - suggestions.length

  const chip = (item: CategoryReuseItem) => {
    const blocked = taken.has(item.name) ? t('alreadyAdded') : item.proxy.asset_type === 'cash' && cashIndex >= 0 ? t('cashExists') : ''
    return <li key={item.name} className="flex max-w-full items-stretch overflow-hidden rounded-lg border border-slate-300 bg-white">
      <button type="button" disabled={Boolean(blocked)} title={blocked || undefined} onClick={() => addReuse(item)}
        className="flex min-h-10 min-w-0 flex-col justify-center px-3 py-1 text-left hover:bg-accent-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 disabled:cursor-not-allowed disabled:bg-slate-50">
        <span className="truncate text-xs font-semibold text-slate-900">{item.name}{blocked && `（${blocked}）`}</span>
        <span className="truncate text-xs text-slate-600 tabular-nums">{item.summary}</span>
      </button>
      <button type="button" aria-label={`${t('hideReuse')}：${item.name}`} onClick={() => suppress(item.name, true)}
        className="flex min-h-10 shrink-0 items-center border-l border-slate-200 px-2 text-slate-600 hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500">
        <XMarkIcon aria-hidden="true" className="h-4 w-4" />
      </button>
    </li>
  }

  return <div className="space-y-3">
    <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
      {(heading || description) && <div className="min-w-0">
        {heading && <h3 className="text-base font-semibold text-slate-900">{heading}</h3>}
        {description && <p className="mt-1 text-sm text-slate-600">{description}</p>}
      </div>}
      <div className="flex flex-wrap items-center gap-2 sm:ml-auto">
        <span aria-live="polite" className="text-xs text-slate-600 tabular-nums">{t('categorySummary', { total: assets.length, ready: readyCount, pending: assets.length - readyCount })}</span>
        {!fixedAssets && <Button tone="primary" disabled={assets.length >= 30} title={assets.length >= 30 ? t('categoryLimit') : undefined} onClick={addBlank}>{t('addAsset')}</Button>}
        {!fixedAssets && !assets.length && <Button onClick={() => onChange({ ...value, assets: (['cash', 'rates', 'credit', 'equity', 'gold'] as const).map((key, index) => blank(key, t(`template.${key}`), index === 0 ? 'cash' : 'market')) })}>{t('useTemplate')}</Button>}
      </div>
    </div>

    {!fixedAssets && (suggestions.length > 0 || hiddenCount > 0) && <div className="border-y border-slate-200 py-3">
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <span className="text-sm font-medium text-slate-800">{t('browseReuse')}</span>
        <span className="text-xs text-slate-600">{t('quickAddHint')}</span>
        {hiddenCount > 0 && <button type="button" onClick={() => suppressed.forEach(name => suppress(name, false))}
          className="min-h-10 text-xs font-semibold text-accent-700 underline focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500">{t('restoreReuse', { count: hiddenCount })}</button>}
      </div>
      {suggestions.length > 0 && <ul aria-label={t('browseReuse')} className="mt-2 flex flex-wrap gap-2">{suggestions.slice(0, PALETTE_VISIBLE).map(chip)}</ul>}
      {suggestions.length > PALETTE_VISIBLE && <details className="mt-2">
        <summary className="min-h-10 cursor-pointer text-xs font-medium leading-10 text-slate-700">{t('moreReuse', { count: suggestions.length - PALETTE_VISIBLE })}</summary>
        <ul className="mt-1 flex flex-wrap gap-2">{suggestions.slice(PALETTE_VISIBLE).map(chip)}</ul>
      </details>}
    </div>}

    <p className="text-xs leading-5 text-slate-600">{t('marketHint')}{t('cashHint')}</p>

    {assets.length > 0 && <div className={`${rowGrid} hidden border-b border-slate-200 pb-2 text-xs font-semibold text-slate-600 lg:grid`} aria-hidden="true">
      <span>{t('assetName')}</span><span>{t('assetType')}</span><span>{t('colRule')}</span>{limitsColumn && <span>{limitsColumn.header}</span>}<span>{t('colProxy')}</span><span className="text-right">{t('colActions')}</span>
    </div>}

    {assets.length > 0 && <ul className="divide-y divide-slate-200 border-b border-slate-200">{assets.map((asset, index) => {
      const detailId = `${domId}-${asset.id}`
      const isOpen = expanded.includes(asset.id)
      const state = status(asset)
      const label = asset.name.trim() || t('untitledCategory', { index: index + 1 })
      return <li key={asset.id} data-testid="risk-reference-asset" className="py-3">
        <div className={rowGrid}>
          <Cell label={t('assetName')}>
            <input required readOnly={fixedAssets} aria-label={t('assetName')} className={controlClass} value={asset.name} onChange={event => update(index, { name: event.target.value })} />
          </Cell>
          <Cell label={t('assetType')}>
            <select required aria-label={t('assetType')} className={controlClass} value={asset.asset_type} onChange={event => changeType(index, event.target.value as ReferenceAsset['asset_type'])}>
              <option value="market">{t('marketAsset')}</option>
              <option value="cash" disabled={cashIndex >= 0 && cashIndex !== index || Boolean(cashEligibleIds && !cashEligibleIds.includes(asset.id))}>{t('cashAsset')}</option>
            </select>
          </Cell>
          <Cell label={t('colRule')}>
            {asset.asset_type === 'cash'
              ? <>
                <PercentInput label={t('cashReturn')} value={(asset.cash_return ?? 0) * 100} onValueChange={next => update(index, { cash_return: next / 100 })} />
                {!Number.isFinite(asset.cash_return ?? NaN) && <span role="alert" className="mt-1 block text-xs text-rose-800">{t('numberRequired')}</span>}
              </>
              : <select required aria-label={t('rebalance')} className={controlClass} value={asset.rebalance ?? 'daily'} onChange={event => update(index, { rebalance: event.target.value as Exclude<ReferenceAsset['rebalance'], null | undefined> })}>
                {REBALANCE_RULES.map(rule => <option key={rule} value={rule}>{t(`frequency.${rule}`)}</option>)}
              </select>}
          </Cell>
          {limitsColumn && <Cell label={limitsColumn.header}>{limitsColumn.render(asset, index)}</Cell>}
          <Cell label={t('colProxy')} className="lg:mt-1">
            <div className="flex min-w-0 flex-wrap items-center gap-2">
              <Badge tone={state.tone}>{state.text}</Badge>
              <span className="min-w-0 flex-1 truncate text-xs text-slate-600 tabular-nums">{asset.asset_type === 'cash'
                ? t('cashReferenceRate')
                : asset.components.map(component => `${sourceLabels[component.series_id] ?? component.series_id.split(':').pop()} ${pct(component.weight)}`).join(' + ') || t('proxyEmpty')}</span>
            </div>
          </Cell>
          <Cell label={t('colActions')} className="lg:mt-1 lg:text-right">
            <div className="flex flex-wrap gap-2 lg:justify-end">
              <Button aria-expanded={isOpen} aria-controls={detailId} aria-label={t('categoryDetail', { name: label })} onClick={() => toggle(asset.id)}>{isOpen ? t('collapseDetail') : t('expandDetail')}</Button>
              {!fixedAssets && <Button aria-label={`${t('removeAsset')}：${label}`} onClick={() => onChange({ ...value, assets: assets.filter((_, i) => i !== index) })}>{t('remove')}</Button>}
            </div>
          </Cell>
        </div>

        <div id={detailId} hidden={!isOpen} className="mt-3 space-y-3 border-l-2 border-slate-200 pl-3">
          {asset.asset_type === 'market' && <>
            {asset.components.map((component, componentIndex) => {
              const parts = component.series_id.split(':'), code = parts[parts.length - 1] || component.series_id
              const name = sourceLabels[component.series_id] ?? code
              return <div key={`${component.series_id}-${componentIndex}`} className="grid items-center gap-2 sm:grid-cols-[minmax(0,1fr)_7rem_auto]">
                <span className="block min-w-0 break-words text-sm">{name}<span className="block text-xs text-slate-600">{code} · {t(`source.${component.kind}`)}</span></span>
                <PercentInput label={`${name} · ${t('componentWeight')}`} value={component.weight * 100}
                  onValueChange={weight => update(index, { components: asset.components.map((member, i) => i === componentIndex ? { ...member, weight: weight / 100 } : member) })} />
                <Button aria-label={`${t('remove')}：${name}`} onClick={() => update(index, { components: asset.components.filter((_, i) => i !== componentIndex) })}>{t('remove')}</Button>
              </div>
            })}
            <div className="flex flex-wrap items-center gap-2">
              <Button aria-expanded={picking === asset.id} onClick={() => setPicking(picking === asset.id ? null : asset.id)}>{t('addProxy')}</Button>
              <span className="text-xs text-slate-600 tabular-nums">{t('weightTotal', { value: pct(weightOf(asset)) })}</span>
            </div>
            <p className="text-xs leading-5 text-slate-600">{t(`rebalance.${asset.rebalance ?? 'daily'}`)}</p>
            {picking === asset.id && <SourcePicker existing={asset.components} onCancel={() => setPicking(null)} onConfirm={items => {
              // One atomic update for the whole batch: the confirm handler reads the
              // current asset once, so a second selection can never overwrite the first.
              const additions = items.filter(item => !asset.components.some(member => member.kind === item.component.kind && member.series_id === item.component.series_id && member.field === item.component.field))
              if (additions.length) onChange(
                { ...value, assets: assets.map((item, i) => i === index ? { ...item, components: [...item.components, ...additions.map(addition => addition.component)] } : item) },
                { ...sourceLabels, ...Object.fromEntries(additions.map(addition => [addition.component.series_id, addition.name])) },
              )
              setPicking(null)
            }} />}
          </>}
          {renderAssetDetails
            ? renderAssetDetails(asset, index)
            : <Field label={t('assetRationale')}><input className={controlClass} value={asset.rationale ?? ''} onChange={event => update(index, { rationale: event.target.value })} /></Field>}
        </div>
      </li>
    })}</ul>}
  </div>
}
