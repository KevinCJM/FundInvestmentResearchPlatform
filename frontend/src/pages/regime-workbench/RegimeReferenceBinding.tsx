import { useEffect, useRef, useState } from 'react'
import { Badge, Button } from '../../components/ui'
import { getRegimeGraphDefinition, listHistoricalReferences, type HistoricalReference, type RegimeGraphDefinition } from '../../services/regimeGraph'
import { bindStudyReference, studyMappingIssue } from './regimeStudy'

export const referenceKey = (reference?: { run_id: string; publication_id: string; content_hash: string }) => reference ? JSON.stringify([reference.run_id, reference.publication_id, reference.content_hash]) : ''
const field = 'mt-1 block min-h-10 w-full min-w-0 rounded-lg border border-slate-300 bg-white px-3 text-sm'
export default function RegimeReferenceBinding({ definition, onChange, onHistorical, onStatus, active = true, disabled = false }: {
  definition: RegimeGraphDefinition; onChange: (next: RegimeGraphDefinition) => void; onHistorical?: () => void; active?: boolean; disabled?: boolean; onStatus?: (status: { key: string; valid: boolean }) => void
}) {
  const [items, setItems] = useState<HistoricalReference[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [source, setSource] = useState<{ key: string; text: string }>({ key: '', text: '' })
  const [retry, setRetry] = useState(0)
  const adoptedReference = useRef('')
  useEffect(() => {
    if (!active) return
    const controller = new AbortController()
    setLoading(true); setError('')
    void listHistoricalReferences(controller.signal).then(next => { if (!controller.signal.aborted) setItems(next) })
      .catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '参考目录读取失败。') })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [active, retry])
  const key = referenceKey(definition.study?.reference)
  const selected = items.find(item => referenceKey(item) === key)
  useEffect(() => {
    if (active && !disabled && selected && adoptedReference.current !== key) {
      adoptedReference.current = key
      if (!definition.id && !definition.graph.nodes.length && JSON.stringify(definition.states) !== JSON.stringify(selected.states)) {
        onChange(bindStudyReference(definition, definition.study?.reference, selected.states))
      }
    }
  }, [active, disabled, selected, key, definition, onChange])
  useEffect(() => {
    if (!selected || !active) return
    const controller = new AbortController()
    setSource({ key, text: '正在读取精确参考的数据来源…' })
    void getRegimeGraphDefinition(selected.definition_id, selected.definition_revision, controller.signal).then(snapshot => {
      if (controller.signal.aborted) return
      const names = snapshot.graph.nodes.filter(node => node.type.startsWith('source.')).map(node => String(node.parameters.name || node.parameters.ts_code || node.label || node.type))
      setSource({ key, text: names.join('、') || '参考定义未声明数据源节点，请查看运行血缘。' })
    }).catch(() => { if (!controller.signal.aborted) setSource({ key, text: '数据来源暂不可读，请刷新参考目录重试。' }) })
    return () => controller.abort()
  }, [selected, active, key])
  const mappingIssue = selected ? studyMappingIssue(definition, selected.states) : ''
  useEffect(() => { onStatus?.({ key, valid: Boolean(selected && !loading && !error && !mappingIssue) }) }, [key, selected, loading, error, mappingIssue, onStatus])
  return <section aria-label="历史参考绑定" className="mb-4 min-w-0 space-y-3 rounded-xl border border-slate-200 bg-white p-4 text-sm tabular-nums">
    <label className="block font-semibold text-slate-700">要识别的历史参考<select aria-label="历史参考版本" className={field} value={key} disabled={disabled || loading || Boolean(error)} onChange={event => {
      const item = items.find(candidate => referenceKey(candidate) === event.target.value)
      onChange(bindStudyReference(definition, item ? { run_id: item.run_id, publication_id: item.publication_id, content_hash: item.content_hash } : undefined, item?.states))
    }}>
      <option value="">请选择已确认的历史参考</option>
      {key && !selected && <option value={key} disabled>当前绑定版本不在可用目录中</option>}
      {items.map(item => <option key={referenceKey(item)} value={referenceKey(item)}>{item.name} · v{item.definition_revision} · {item.as_of || '截至日未提供'} · {item.created_at}</option>)}
    </select></label>
    <Button disabled={loading || disabled} onClick={() => setRetry(value => value + 1)}>刷新参考目录</Button>
    {loading && <div role="status" className="space-y-2 text-slate-600"><div className="h-4 rounded bg-slate-100" />正在读取历史参考…</div>}
    {error && <div role="alert" className="text-rose-800">{error}<Button onClick={() => setRetry(value => value + 1)}>重试读取参考</Button></div>}
    {!loading && !error && !items.length && <div className="space-y-2 text-slate-600"><p>还没有已确认的历史参考。先生成历史区间，再保存为历史参考。</p><Button onClick={onHistorical} disabled={!onHistorical}>前往历史状态定义</Button></div>}
    {!key && <p className="text-slate-600">先选择历史参考，再建立识别同一组状态的实时模型。一份参考可以比较多个实时模型。</p>}
    {key && !loading && !error && !selected && <p role="alert" className="text-amber-900">参考版本不可用，请重新选择；当前不能验证。</p>}
    {selected && <>
      <Badge tone="neutral">已绑定固定参考 · 不代表识别已通过验证</Badge>
      <p className="break-words text-slate-600">{selected.name} · 第 {selected.definition_revision} 版 · {({ daily: '日频', monthly: '月频', weekly: '周频' } as Record<string, string>)[selected.frequency || ''] || selected.frequency || '频率未提供'} · 截至 {selected.as_of || '未提供'}</p>
      <p className="text-xs text-slate-600">观察范围：{selected.series_summary?.first_observation_date || '未提供'} 至 {selected.series_summary?.last_observation_date || '未提供'}。</p>
      <p className="break-words text-xs text-slate-600">数据来源：{source.key === key ? source.text : '正在读取…'}</p>
      <p className="text-xs text-slate-600">更换参考或状态对应关系后，需保存新修订并重新验证、校准。已有版本与报告保留。</p>
      <details><summary className="min-h-10 cursor-pointer text-slate-600">状态对应与来源</summary>
        <div className="grid gap-3 sm:grid-cols-2">{definition.states.map(state => <label key={state.id} className="min-w-0 text-slate-700">{state.label}（{state.id}）<select aria-label={`参考状态：${state.id}`} className={field} disabled={disabled} value={definition.study?.state_mapping ? definition.study.state_mapping[state.id] || '' : selected.states.some(item => item.id === state.id) ? state.id : ''} onChange={event => {
          const { calibration_id: _calibration, qualification_id: _qualification, ...study } = definition.study!
          const initial = Object.fromEntries(definition.states.flatMap(item => selected.states.some(target => target.id === item.id) ? [[item.id, item.id]] : []))
          onChange({ ...definition, study: { ...study, state_mapping: { ...(study.state_mapping || initial), [state.id]: event.target.value } } })
        }}><option value="">请选择对应状态</option>{selected.states.map(target => <option key={target.id} value={target.id}>{target.label}（{target.id}）</option>)}</select></label>)}</div>
        <dl className="mt-3 space-y-2 break-all text-xs text-slate-600"><dt>定义</dt><dd>{selected.definition_id} · v{selected.definition_revision}</dd><dt>运行 / 发布</dt><dd>{selected.run_id} / {selected.publication_id}</dd><dt>内容校验</dt><dd>{selected.content_hash}</dd></dl>
      </details>
      {mappingIssue && <p role="alert" className="text-amber-900">{mappingIssue}</p>}
    </>}
  </section>
}
