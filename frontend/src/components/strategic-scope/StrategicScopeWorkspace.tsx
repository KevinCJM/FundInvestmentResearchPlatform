import { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { allocationJourneyPath, readAllocationJourney, updateAllocationJourney, useAllocationDraft } from '../../app/allocationJourney'
import { useResearchDay } from '../../app/ResearchContext'
import { getStrategicCatalog, type StrategicCatalog } from '../../services/strategicAllocation'
import { confirmUniverse, getStrategicUniverse, previewUniverse, type UniverseDefinition, type UniversePreview, type UniverseVersion } from '../../services/strategicScope'
import { Button } from '../ui'
import { Feedback, Field, inputClass, sectionClass, today } from '../risk-models/ResearchUI'
import UniverseFields from './UniverseFields'
import ImplementationMappingEditor from './ImplementationMappingEditor'
import { useScopeOperation } from './useScopeOperation'

export default function StrategicScopeWorkspace() {
  const [params, setParams] = useSearchParams()
  const clock = useResearchDay()
  const previousClock = useRef(clock)
  const initialId = params.get('strategic_universe') ?? readAllocationJourney().strategicUniverseId ?? ''
  const domainId = params.get('universe') ?? readAllocationJourney().universeId ?? ''
  const [draft, setDraft] = useAllocationDraft<UniverseDefinition>('strategic-universe:editor', () => ({ name: '', as_of: clock ?? today(), currency: 'CNY', source: '', assets: [] }))
  const [catalog, setCatalog] = useState<StrategicCatalog | null>(null)
  const [catalogError, setCatalogError] = useState('')
  const [reload, setReload] = useState(0)
  const [loading, setLoading] = useState(true)
  const [saved, setSaved] = useState<UniverseVersion | null>(null)
  const [preview, setPreview] = useState<UniversePreview | null>(null)
  const [mappingId, setMappingId] = useState(params.get('mapping') ?? '')
  const operation = useScopeOperation()
  // Immutable history reads do not depend on the current research clock. Keep
  // them separate so initial PIT settings cannot cancel version restoration.
  const historyRead = useScopeOperation()
  useEffect(() => {
    const controller = new AbortController(); setLoading(true); setCatalogError('')
    getStrategicCatalog(controller.signal).then(value => { if (!controller.signal.aborted) setCatalog(value) })
      .catch(error => { if (!controller.signal.aborted) setCatalogError(error instanceof Error ? error.message : '目录读取失败。') })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [reload])
  function read(id: string, mapping = '') {
    operation.invalidate(); setPreview(null); setMappingId(mapping)
    void historyRead.run(signal => getStrategicUniverse(id, signal), result => {
      if (result.id !== id) throw new Error('读取的战略范围身份不一致。')
      setSaved(result); setDraft(result.definition)
      updateAllocationJourney({ strategicUniverseId: result.id, implementationMappingId: mapping || undefined })
    })
  }
  useEffect(() => { if (initialId) read(initialId, params.get('mapping') ?? readAllocationJourney().implementationMappingId ?? '') }, [initialId])
  useEffect(() => { if (previousClock.current === clock) return; previousClock.current = clock; operation.invalidate(); if (!saved) setPreview(null) }, [clock])
  const change = (value: UniverseDefinition) => { historyRead.invalidate(); operation.invalidate(); setPreview(null); setSaved(null); setDraft(value) }
  const issue = clock === undefined ? '知识截止日尚未确认，暂不能预览或确认；草稿保留。' : !draft.name.trim() || draft.source.trim().length < 3 ? '请填写战略范围名称及来源。' : !draft.as_of || draft.as_of > (clock ?? today()) ? '战略研究日不能晚于平台知识截止日。' : !draft.assets.length ? '请至少增加一个战略资产。' : new Set(draft.assets.map(a => a.id)).size !== draft.assets.length ? '战略资产ID重复，请为不同风险定义唯一ID。' : draft.assets.some(a => !/^[a-z][a-z0-9_-]{0,79}$/.test(a.id) || !a.name.trim() || a.rationale.trim().length < 3 || a.source.trim().length < 3 || a.currency !== draft.currency) ? '逐项填写稳定ID、名称、理由与来源；所有资产须采用相同本位币口径。' : ''
  return <section className={`${sectionClass} space-y-5`} aria-label="独立战略范围">
    <h2 className="text-lg font-semibold">先确定需要的战略资产</h2>
    <p className="text-sm leading-6 text-slate-600">无需先选择产品。经济角色、稳定ID与来源确认后形成独立版本；缺产品的资产仍保留在长期假设和SAA研究中。</p>
    <Feedback error={operation.error || historyRead.error || catalogError} />
    {historyRead.busy && <p role="status" className="text-sm text-slate-600">正在读取不可变战略范围…</p>}
    {saved && <div className="space-y-3"><p className="text-sm text-slate-700">只读战略范围：{saved.name} · {saved.definition.as_of}</p><div className="flex flex-wrap gap-3"><Button onClick={() => { historyRead.invalidate(); operation.invalidate(); setSaved(null); setPreview(null); setMappingId('') }}>复制范围为新研究</Button><Link className="inline-flex min-h-10 items-center text-sm font-semibold text-accent-800 underline" to={allocationJourneyPath('saa', { ...readAllocationJourney(), strategicUniverseId: saved.id, implementationMappingId: mappingId || undefined, baselineId: undefined })}>先做前瞻CMA与SAA研究 →</Link></div></div>}
    <fieldset disabled={Boolean(saved) || historyRead.busy} className="min-w-0"><UniverseFields value={draft} onChange={change} /></fieldset>
    {issue && !saved && <p role="status" className="text-sm text-amber-800">{issue}</p>}
    <div className="flex flex-wrap gap-3"><Button disabled={Boolean(issue) || operation.busy || historyRead.busy || Boolean(saved)} onClick={() => void operation.run(signal => previewUniverse(draft, signal), setPreview)}>预览战略范围</Button><Button tone="primary" disabled={Boolean(issue) || operation.busy || historyRead.busy || !preview || Boolean(saved)} onClick={() => preview && void operation.run(signal => confirmUniverse(draft, preview.preview_hash, signal), result => {
      setSaved(result); setPreview(result); updateAllocationJourney({ strategicUniverseId: result.id, mandateId: params.get('mandate') ?? readAllocationJourney().mandateId }); setReload(n => n + 1)
    })}>确认保存战略范围</Button></div>
    {!saved && !preview && <p className="text-xs text-slate-600">填写并预览后才能确认保存；预览不写入研究库。</p>}
    {operation.busy && <p role="status" className="text-sm text-slate-600">正在核验战略定义…</p>}
    {preview && !saved && <p role="status" className="text-sm text-slate-700">定义已通过校验。尚未匹配产品：{preview.implementation_gaps.join('、')}。确认后可进行纯前瞻研究。</p>}
    <details className="space-y-3 border-t border-slate-200 pt-4"><summary className="min-h-10 cursor-pointer text-sm font-medium">读取历史范围与映射</summary>
      {loading ? <p role="status" className="text-sm text-slate-600">正在读取不可变范围目录…</p> : <Field label="已保存的战略范围"><select className={inputClass} value={saved?.id ?? ''} onChange={e => e.target.value && read(e.target.value)}><option value="">选择历史范围（只读）</option>{catalog?.strategic_universes?.map(v => <option key={v.id} value={v.id}>{v.name} · {v.definition.as_of}</option>)}</select></Field>}
      {saved && <Field label="已保存的实施映射"><select className={inputClass} value={mappingId} onChange={e => { setMappingId(e.target.value); if (!e.target.value) updateAllocationJourney({ implementationMappingId: undefined }) }}><option value="">创建新映射或暂不匹配</option>{catalog?.implementation_maps?.filter(m => m.definition.strategic_universe_id === saved.id).map(m => <option key={m.id} value={m.id}>{m.name} · {m.implementation_status === 'complete' ? '完整覆盖' : '有缺口'}</option>)}</select></Field>}
      <Button disabled={loading} onClick={() => setReload(n => n + 1)}>重试读取范围目录</Button>
    </details>
    {saved && catalog && <ImplementationMappingEditor key={`${saved.id}:${mappingId}`} universe={saved} catalog={catalog} domainId={domainId} requestedMapping={mappingId} onSaved={mapping => { setMappingId(mapping.id); setReload(n => n + 1); setParams(current => { current.set('scope', 'strategic'); current.set('strategic_universe', saved.id); current.set('mapping', mapping.id); return current }, { replace: true }) }} />}
  </section>
}
