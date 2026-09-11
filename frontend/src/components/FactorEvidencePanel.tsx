import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { factorApi, numberText, type ContextType, type FactorBinding, type FactorRelease, type FactorRun } from '../services/factorResearch'
import { buttonClass, Field, inputClass, secondaryClass } from './factor-research/shared'
import { releaseState } from './factor-research/ReleaseWorkbench'

export default function FactorEvidencePanel({ contextType, contextId = '', productId }: { contextType: ContextType; contextId?: string; productId?: string }) {
  const [open, setOpen] = useState(false)
  const [releases, setReleases] = useState<FactorRelease[]>([])
  const [selected, setSelected] = useState('')
  const [run, setRun] = useState<FactorRun>()
  const [objectId, setObjectId] = useState(contextId)
  const [note, setNote] = useState('')
  const [bindings, setBindings] = useState<FactorBinding[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  useEffect(() => { setObjectId(contextId); setSelected(''); setRun(undefined); setBindings([]) }, [contextId, contextType, productId])
  useEffect(() => {
    if (!open) return
    let active = true
    setBusy(true); setError('')
    Promise.all([factorApi.releases(productId), contextId ? factorApi.bindings(contextType, contextId) : Promise.resolve({ items: [] })])
      .then(([result, references]) => { if (active) { setReleases(result.items); setBindings(references.items) } })
      .catch(reason => { if (active) setError(reason.message) })
      .finally(() => { if (active) setBusy(false) })
    return () => { active = false }
  }, [open, productId, contextType, contextId])
  const release = releases.find(item => item.id === selected)
  const inspect = async (id: string) => {
    setSelected(id); setRun(undefined); setError('')
    const item = releases.find(value => value.id === id)
    if (!item) return
    setBusy(true)
    try { setRun(await factorApi.getRun(item.run_id)) } catch (reason) { setError((reason as Error).message) } finally { setBusy(false) }
  }
  return <details className="mb-5 min-w-0 rounded-xl border border-indigo-200 bg-white p-4" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary className="cursor-pointer text-sm font-semibold text-indigo-900">因子研究证据</summary>
    {open && <div className="mt-4 space-y-4">
      <p className="text-sm leading-6 text-slate-600">选择已发布研究版本，查看产品得分并登记本环节的使用依据。<Link to="/settings/factor-research" className="ml-2 font-semibold text-indigo-600 underline">进入因子研究中心</Link></p>
      {error && <p role="alert" className="rounded-lg bg-rose-50 p-3 text-sm text-rose-700">{error}</p>}
      {busy && <p role="status" className="text-sm text-slate-500">正在读取因子研究…</p>}
      {message && <p role="status" className="text-sm text-emerald-800">{message}</p>}
      <Field label="引用因子发布"><select className={inputClass} disabled={busy} value={selected} onChange={event => void inspect(event.target.value)}><option value="">请选择发布版本</option>{releases.map(item => <option key={item.id} value={item.id}>{item.name} · {releaseState[item.state]}</option>)}</select></Field>
      {!busy && !releases.length && <p className="text-sm text-slate-500">暂无适用发布。先在因子研究中心完成检验与发布。</p>}
      {run && release && <>
        <p className="text-xs text-slate-500">得分日 {run.as_of} · {releaseState[release.state]} · 方案 v{run.study_revision} · 仅用于研究</p>
        <div className="max-h-64 overflow-auto"><table className="w-full min-w-[360px] text-left text-sm" aria-label="投研因子证据"><thead><tr><th className="p-2">产品</th><th>排名</th><th>组合得分</th></tr></thead><tbody>{run.latest_scores.filter(row => !productId || row.product_id === productId).map(row => <tr className="border-t border-slate-100" key={row.code}><td className="p-2">{row.name}<span className="block text-xs text-slate-500">{row.code}</span></td><td>{numberText(row.rank, 1)}</td><td>{numberText(row.score, 2)}</td></tr>)}</tbody></table></div>
        <form onSubmit={async event => {
          event.preventDefault(); setBusy(true); setError(''); setMessage('')
          try { await factorApi.bind({ release_id: release.id, context_type: contextType, context_id: objectId.trim(), note }); setBindings((await factorApi.bindings(contextType, objectId.trim())).items); setMessage('已登记发布版本及来源运行。') }
          catch (reason) { setError((reason as Error).message) } finally { setBusy(false) }
        }}><fieldset disabled={busy || release.state !== 'active'} className="grid gap-3 md:grid-cols-2">
          <Field label="研究对象 / 组合版本 ID" hint="填写实际研究对象标识；发布引用不会自动改变原配置。"><input className={inputClass} required maxLength={120} value={objectId} onChange={event => setObjectId(event.target.value)} /></Field>
          <Field label="使用依据"><input className={inputClass} maxLength={600} value={note} onChange={event => setNote(event.target.value)} placeholder="例如：检查类内候选产品的风险和动量" /></Field>
          <div className="flex flex-wrap gap-2 md:col-span-2"><button className={buttonClass} type="submit">登记本环节引用</button><Link className={secondaryClass} to={'/settings/factor-research?run=' + encodeURIComponent(run.id)}>查看完整检验</Link></div>
        </fieldset></form>
      </>}
      {bindings.length > 0 && <div className="space-y-2 text-xs text-slate-500">{bindings.map((binding, i) => <p key={i} className="break-all">已引用：{binding.context_id} · {binding.release_id} · {binding.note || '未填写说明'}</p>)}</div>}
    </div>}
  </details>
}
