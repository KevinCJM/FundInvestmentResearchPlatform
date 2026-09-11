import { useEffect, useRef, useState } from 'react'
import { timingApi, type TimingRelease } from '../../services/timingResearch'
import { timingField } from './TimingRuleEditor'

export default function TimingReleaseLibrary({ onView, disabled }: { onView: (runId: string) => void; disabled?: boolean }) {
  const [releases, setReleases] = useState<TimingRelease[] | null>(null)
  const [selectedId, setSelectedId] = useState('')
  const [note, setNote] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [binding, setBinding] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const mounted = useRef(true)
  const selected = releases?.find(item => item.id === selectedId) || releases?.[0]
  useEffect(() => { mounted.current = true; return () => { mounted.current = false } }, [])
  useEffect(() => {
    const controller = new AbortController()
    setError(''); setReleases(null)
    void timingApi.releases(controller.signal).then(result => {
      if (!controller.signal.aborted) setReleases(result.items)
    }).catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '读取研究版本失败。') })
    return () => controller.abort()
  }, [attempt])
  const bind = async () => {
    if (!selected) return
    setBinding(true); setError(''); setNotice('')
    try {
      await timingApi.bind(selected.id, 'pre_investment', note)
      if (mounted.current) setNotice(`已引用“${selected.name || '所选研究版本'}”到投前研究。组合权重和实际交易均未改变。`)
    } catch (reason) { if (mounted.current) setError(reason instanceof Error ? reason.message : '引用研究版本失败。') }
    finally { if (mounted.current) setBinding(false) }
  }
  return <section aria-label="已有研究版本" className="space-y-4">
    <div><h2 className="text-base font-semibold text-slate-900">选择已经检验的研究版本</h2><p className="mt-2 text-sm leading-6 text-slate-500">先查看适用产品和研究结果，再将固定版本引用到投前。引用不会重新计算或重复发布。</p></div>
    {error && <div role="alert" className="rounded-lg bg-rose-50 p-3 text-sm text-rose-700">{error}{!releases && <button type="button" className="ml-3 min-h-9 underline" onClick={() => setAttempt(value => value + 1)}>重新读取</button>}</div>}
    {notice && <p role="status" className="rounded-lg bg-emerald-50 p-3 text-sm text-emerald-800">{notice}</p>}
    {!releases && !error && <p role="status" className="py-6 text-sm text-slate-500">正在读取已有研究版本…</p>}
    {releases?.length === 0 && <p className="rounded-xl border border-dashed border-slate-300 p-6 text-sm leading-6 text-slate-500">还没有已保存的研究版本。请先在算法步骤中完成研究，再从结果页保存研究版本。</p>}
    {!!releases?.length && <><div role="radiogroup" aria-label="选择已有研究版本" className="grid gap-3 md:grid-cols-2">{releases.map(release => <label key={release.id} className={`flex cursor-pointer items-start gap-3 rounded-xl border p-4 ${selected?.id === release.id ? 'border-indigo-400 bg-indigo-50/40' : 'border-slate-200 bg-white'}`}><input type="radio" name="timing-release" className="mt-1 h-4 w-4 shrink-0" checked={selected?.id === release.id} disabled={binding || disabled} onChange={() => { setSelectedId(release.id); setNotice('') }} /><span className="min-w-0"><strong className="block break-words text-sm text-slate-900">{release.name || '择时研究版本'}</strong><span className="mt-1 block text-xs text-slate-500">{release.created_at?.replace('T', ' ').slice(0, 19) || '已保存'} · 研究用途</span>{release.products?.length ? <span className="mt-2 block break-words text-xs text-slate-600">适用产品：{release.products.map(product => product.product_id).join('、')}</span> : null}{release.note && <span className="mt-2 block text-xs leading-5 text-slate-500">{release.note}</span>}</span></label>)}</div>
      <label className="block text-xs text-slate-600">本次引用说明<input className={timingField} maxLength={500} value={note} onChange={event => setNote(event.target.value)} placeholder="记录使用目的与组合约束" /></label>
      <div className="flex flex-wrap gap-3"><button type="button" className="min-h-10 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm text-slate-700 disabled:opacity-40" disabled={!selected || binding || disabled} onClick={() => selected && onView(selected.run_id)}>查看此版本结果</button><button type="button" className="min-h-10 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-40" disabled={!selected || binding || disabled} onClick={bind}>{binding ? '引用中…' : '引用此版本到投前'}</button></div>
      <p className="text-xs text-slate-500">不自动交易、不改变组合权重；大类配置调整仍需遵守 TAA 约束。</p>
    </>}
  </section>
}
