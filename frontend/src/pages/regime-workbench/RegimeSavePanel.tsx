import { useEffect, useRef, useState, type ReactNode } from 'react'
import {
  createRegimeGraphDefinition, definitionForRequest, enableRegimeResearchVersion,
  prepareRegimeGraph, updateRegimeGraphDefinition,
  type RegimeGraphDefinition, type RegimeMode, type RegimeResearchVersion,
} from '../../services/regimeGraph'

const signature = (definition: RegimeGraphDefinition) => JSON.stringify(definitionForRequest(definition))

export default function RegimeSavePanel({ definition, dirty, valid, mode, asOf, onSaved, onBusy, onViewResult, children }: {
  definition: RegimeGraphDefinition
  dirty: boolean
  valid: boolean
  mode: RegimeMode
  asOf: string
  onSaved: (saved: RegimeGraphDefinition) => void
  onBusy: (busy: boolean) => void
  onViewResult: (runId: string) => void
  children: ReactNode
}) {
  const [name, setName] = useState(definition.name)
  const [description, setDescription] = useState(definition.description)
  const [step, setStep] = useState('')
  const [error, setError] = useState('')
  const [result, setResult] = useState<RegimeResearchVersion | null>(null)
  const [advanced, setAdvanced] = useState(false)
  const saved = useRef<RegimeGraphDefinition | null>(definition.id && !dirty ? definition : null)
  const inFlight = useRef(false)
  const mounted = useRef(true)
  useEffect(() => { mounted.current = true; return () => { mounted.current = false } }, [])
  useEffect(() => {
    if (inFlight.current) return
    setName(definition.name); setDescription(definition.description); setResult(null); setError('')
    saved.current = definition.id && !dirty ? definition : null
  }, [definition, dirty, mode, asOf])

  const save = async () => {
    if (inFlight.current || !valid || !name.trim()) return
    inFlight.current = true; onBusy(true); setError(''); setResult(null)
    let stored = saved.current
    try {
      setStep('正在保存算法…')
      const draft = { ...definition, name: name.trim(), description: description.trim(), default_mode: mode }
      const sameSaved = stored && signature({ ...draft, id: stored.id, revision: stored.revision }) === signature(stored)
      if (!sameSaved) {
        stored = draft.id && draft.revision
          ? await updateRegimeGraphDefinition(draft)
          : await createRegimeGraphDefinition(draft)
        saved.current = stored
        if (mounted.current) onSaved(stored)
      }
      if (!stored || !mounted.current) return
      setStep('正在准备计算…')
      const plan = await prepareRegimeGraph(stored)
      if (!mounted.current) return
      setStep('正在生成研究结果…')
      const version = await enableRegimeResearchVersion(stored, plan.compile_token, mode, asOf)
      if (mounted.current) setResult(version)
    } catch (reason) {
      if (mounted.current) setError(`${stored ? `算法已保存为 v${stored.revision}，尚未完成研究准备。` : ''}${reason instanceof Error ? reason.message : '保存失败，请重试。'}`)
    } finally {
      inFlight.current = false
      if (mounted.current) { setStep(''); onBusy(false) }
    }
  }

  return <section aria-label="保存情景供研究使用" className="min-w-0 space-y-5">
    <p className="text-sm leading-6 text-slate-600">保存这套计算逻辑后，可在单产品等研究页面直接选择它。</p>
    <fieldset disabled={Boolean(step)} className="min-w-0 space-y-4 disabled:opacity-70">
      <label className="block text-sm font-semibold text-slate-700">情景名称<input autoComplete="off" value={name} maxLength={80} onChange={event => { setName(event.target.value); setResult(null) }} className="mt-2 block min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3 font-normal" /></label>
      <label className="block text-sm font-semibold text-slate-700">说明（可选）<textarea value={description} maxLength={1000} rows={2} onChange={event => { setDescription(event.target.value); setResult(null) }} className="mt-2 block w-full resize-y rounded-xl border border-slate-300 bg-white p-3 font-normal" /></label>
    </fieldset>
    <div className="space-y-3 rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600">
      <p><strong className="text-slate-900">{mode === 'retrospective' ? '事后研究' : '实时识别'}</strong>{asOf ? ` · 截至 ${asOf}` : ' · 使用当前可用数据'}</p>
      <div className="flex flex-wrap gap-2">{definition.states.map(state => <span key={state.id} className="rounded-full bg-slate-100 px-3 py-1 text-xs">{state.label}</span>)}</div>
      <p className="text-xs leading-5">{definition.id ? `当前为 v${definition.revision}。修改后会保存为新版本，已有研究继续使用原版本。` : '首次保存为 v1，之后可继续修改并保存新版本。'}</p>
      {mode === 'retrospective' && <p className="text-xs text-amber-800">用于解释历史市场，不作为当时可交易的信号。</p>}
    </div>
    {!valid && !result && !step && <p role="alert" className="text-sm text-amber-800">请先完成公式检查，再保存情景。</p>}
    {error && <p role="alert" className="rounded-xl bg-rose-50 p-3 text-sm leading-6 text-rose-800">{error}</p>}
    {step && <p role="status" className="text-sm font-medium text-accent-700">{step}</p>}
    <button type="button" disabled={!valid || !name.trim() || Boolean(step) || Boolean(result)} onClick={() => void save()} className="min-h-11 w-full rounded-xl bg-accent-600 px-5 py-3 text-sm font-semibold text-white disabled:opacity-40">{step ? '正在保存…' : result ? '已保存，可用于研究' : error && saved.current ? '重试生成研究结果' : '保存并用于研究'}</button>
    {result && <div role="status" className="space-y-3 rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-900">
      <p className="font-semibold">{result.name} · v{result.revision} 已可选用</p>
      <p>{result.series_summary.first_observation_date} — {result.series_summary.last_observation_date} · {result.series_summary.row_count} 个观测</p>
      <p>在产品页的“情景方案”中选择此版本即可开始研究。</p>
      <div className="flex flex-wrap gap-3"><button type="button" onClick={() => onViewResult(result.run_id)} className="min-h-10 rounded-lg border border-emerald-300 bg-white px-3 font-semibold">查看情景结果</button><a href="/product-research/products" className="inline-flex min-h-10 items-center rounded-lg bg-emerald-800 px-3 font-semibold text-white">前往产品研究</a></div>
    </div>}
    <details open={advanced} onToggle={event => setAdvanced(event.currentTarget.open)} className="border-t border-slate-200 pt-4">
      <summary className="min-h-10 cursor-pointer text-sm font-medium text-slate-600">高级管理</summary>
      {advanced && <fieldset disabled={Boolean(step)} className="mt-3 min-w-0 space-y-4">{children}</fieldset>}
    </details>
  </section>
}
