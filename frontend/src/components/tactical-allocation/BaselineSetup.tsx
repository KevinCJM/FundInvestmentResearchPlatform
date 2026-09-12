import { useState } from 'react'
import { Link } from 'react-router-dom'
import { allocationJourneyPath } from '../../app/allocationJourney'
import { createTaaBaseline, type TaaBaseline, type TaaCatalog } from '../../services/tacticalAllocation'
import { buttonClass, Empty, Feedback, Field, inputClass, NumberInput, primaryClass, sectionClass, today } from '../risk-models/ResearchUI'

export default function BaselineSetup({ catalog, selectedId, loading, onSelect, onCreated }: {
  catalog: TaaCatalog | null
  selectedId: string
  loading: boolean
  onSelect: (id: string) => void
  onCreated: (baseline: TaaBaseline) => void
}) {
  const [creating, setCreating] = useState(false)
  const [allocation, setAllocation] = useState('')
  const [name, setName] = useState('')
  const [asOf, setAsOf] = useState(today)
  const [weights, setWeights] = useState<Record<string, number>>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const source = catalog?.allocations.find(item => item.alloc_name === allocation)
  const total = Object.values(weights).reduce((sum, value) => sum + value, 0)
  const valid = Boolean(source?.assets.length) && source!.assets.every(asset => Number.isFinite(weights[asset.id]) && weights[asset.id] >= 0 && weights[asset.id] <= 100) && Math.abs(total - 100) < 0.00001

  function selectAllocation(value: string) {
    const next = catalog?.allocations.find(item => item.alloc_name === value)
    setAllocation(value); setName(next ? `${next.alloc_name} · SAA 基准` : '')
    setAsOf(next?.as_of || today()); setWeights(Object.fromEntries((next?.assets ?? []).map(asset => [asset.id, 0]))); setError('')
  }

  async function save() {
    if (!source || !valid || !name.trim() || !asOf) return
    setBusy(true); setError('')
    try {
      const baseline = await createTaaBaseline({ alloc_name: source.alloc_name, name: name.trim(), as_of: asOf, weights: Object.fromEntries(Object.entries(weights).map(([key, value]) => [key, value / 100])) })
      onCreated(baseline); setCreating(false)
    } catch (failure) { setError(failure instanceof Error ? failure.message : 'SAA 基准保存失败。') }
    finally { setBusy(false) }
  }

  return <section className={sectionClass} aria-label="SAA 基准选择"><details open={!selectedId || creating}><summary className="cursor-pointer text-sm font-medium text-slate-800">{selectedId ? `长期组合：${catalog?.baselines.find(item => item.id === selectedId)?.name ?? '读取中'} · 更换 / 新建` : '选择长期配置基准'}</summary><div className="mt-3">
    <div className="flex flex-wrap items-start justify-between gap-3"><div><h2 className="text-base font-semibold text-slate-950">从哪个长期组合出发？</h2><p className="mt-1 text-sm text-slate-600">锁定 SAA 权重和资产范围，再研究临时调整。</p></div><Link to={allocationJourneyPath('saa')} className="min-h-11 py-2 text-sm font-medium text-accent-800 underline underline-offset-4">去 SAA 选择方案</Link></div>
    {loading ? <p role="status" className="mt-4 text-sm text-slate-600">正在读取 SAA 基准…</p> : <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-end"><div className="min-w-0 flex-1"><Field label="SAA 基准版本"><select className={inputClass} value={selectedId} onChange={event => onSelect(event.target.value)}><option value="">请选择已保存的 SAA 基准</option>{catalog?.baselines.map(item => <option key={item.id} value={item.id}>{item.name} · {item.as_of}</option>)}</select></Field></div><button type="button" className={buttonClass} onClick={() => setCreating(value => !value)}>{creating ? '收起新建基准' : '新建研究基准'}</button></div>}
    {!loading && !catalog?.baselines.length && !creating && <div className="mt-4"><Empty title="先确定长期配置，再讨论偏离"><p>在 SAA 选择组合并带入，或从已有资产分类新建研究基准。这里不会预填合成收益。</p></Empty></div>}
    {creating && <div className="mt-5 space-y-4 border-t border-slate-200 pt-5"><Feedback error={error} /><p className="text-sm text-slate-600">手动确定长期权重。保存后将冻结真实资产数据与产品映射；这一步不代表通过历史 PIT 验证。</p>
      {!catalog?.allocations.length ? <Empty title="暂无可用资产分类"><p>请先在 SAA 中构建资产分类并保存。</p></Empty> : <fieldset disabled={busy} className="min-w-0 space-y-4"><div className="grid gap-4 sm:grid-cols-3"><Field label="资产分类方案"><select className={inputClass} value={allocation} onChange={event => selectAllocation(event.target.value)}><option value="">选择已有分类方案</option>{catalog.allocations.map(item => <option key={item.alloc_name} value={item.alloc_name}>{item.alloc_name}</option>)}</select></Field><Field label="基准名称"><input className={inputClass} value={name} onChange={event => setName(event.target.value)} /></Field><Field label="SAA 方案日期"><input type="date" max={today()} className={inputClass} value={asOf} onChange={event => setAsOf(event.target.value)} /></Field></div>
        {source && <><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{source.assets.map(asset => <Field key={asset.id} label={`${asset.name}长期权重（%）`}><NumberInput className={inputClass} value={weights[asset.id]} onValueChange={value => setWeights(previous => ({ ...previous, [asset.id]: value }))} min={0} max={100} /></Field>)}</div><div className="flex flex-wrap items-center justify-between gap-3"><p role="status" className={`text-sm ${valid ? 'text-accent-800' : 'text-amber-800'}`}>权重合计：{Number.isFinite(total) ? `${total.toFixed(2)}%` : '请填写完整'}，需为 100%。</p><button type="button" className={buttonClass} onClick={() => setWeights(Object.fromEntries(source.assets.map(asset => [asset.id, 100 / source.assets.length])))}>以等权起步</button></div></>}
        <button type="button" className={primaryClass} disabled={!valid || !name.trim() || !asOf || busy} onClick={() => void save()}>{busy ? '正在冻结基准与数据…' : '保存并使用此基准'}</button>
      </fieldset>}
    </div>}
  </div></details>{selectedId && <Link to={allocationJourneyPath('saa')} className="mt-2 inline-block text-xs text-accent-800 underline">返回本次 SAA 方案</Link>}</section>
}
