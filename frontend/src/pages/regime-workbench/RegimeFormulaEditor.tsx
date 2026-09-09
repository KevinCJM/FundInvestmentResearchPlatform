import { useEffect, useRef, useState } from 'react'
import { definitionForRequest, resolveRegimeAuthoring, type RegimeAuthoringResolution, type RegimeGraphDefinition, type RegimeMode, type RegimeNodeSchema } from '../../services/regimeGraph'
import { RegimeMathDisplay } from './RegimeMathPreview'

interface Props {
  definition: RegimeGraphDefinition
  mode: RegimeMode
  schemas: RegimeNodeSchema[]
  outputId?: string
  onApply: (definition: RegimeGraphDefinition) => void
  onPending: (pending: boolean) => void
  onBusy: (busy: boolean) => void
}

export default function RegimeFormulaEditor({ definition, mode, schemas, outputId, onApply, onPending, onBusy }: Props) {
  const [source, setSource] = useState('')
  const [baseline, setBaseline] = useState('')
  const [resolution, setResolution] = useState<RegimeAuthoringResolution | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const [stale, setStale] = useState(false)
  const [reload, setReload] = useState(0)
  const revision = useRef(0)
  const controller = useRef<AbortController | null>(null)
  const currentKey = `${JSON.stringify(definitionForRequest(definition))}:${mode}`
  const keyRef = useRef(currentKey)
  const dirtyRef = useRef(false)
  const dirty = loaded && source !== baseline
  dirtyRef.current = dirty
  keyRef.current = currentKey

  useEffect(() => { onPending(dirty) }, [dirty, onPending])
  useEffect(() => { onBusy(busy) }, [busy, onBusy])
  useEffect(() => () => { controller.current?.abort(); revision.current += 1; onPending(false); onBusy(false) }, [onPending, onBusy])

  useEffect(() => {
    controller.current?.abort()
    const request = ++revision.current
    if (dirtyRef.current) { setStale(true); setBusy(false); return }
    const abort = new AbortController(); controller.current = abort
    setBusy(true); setError(''); setStale(false)
    void resolveRegimeAuthoring(definition, mode, 'graph', '', abort.signal).then(result => {
      if (abort.signal.aborted || request !== revision.current) return
      setResolution(result); setSource(result.source); setBaseline(result.source); setLoaded(true)
    }).catch(reason => { if (!abort.signal.aborted) setError(reason instanceof Error ? reason.message : '公式载入失败。') })
      .finally(() => { if (!abort.signal.aborted && request === revision.current) setBusy(false) })
    return () => abort.abort()
  }, [currentKey, reload])

  const apply = async () => {
    const key = currentKey
    const request = ++revision.current
    controller.current?.abort(); const abort = new AbortController(); controller.current = abort
    setBusy(true); setError('')
    try {
      const result = await resolveRegimeAuthoring(definition, mode, 'formula', source, abort.signal)
      if (abort.signal.aborted || request !== revision.current || key !== keyRef.current) return
      setResolution(result)
      if (result.valid && result.definition) {
        const positions = new Map(definition.graph.nodes.map(node => [node.id, node.position]))
        const next = { ...result.definition, id: definition.id, revision: definition.revision,
          graph: { ...result.definition.graph, nodes: result.definition.graph.nodes.map(node => ({ ...node, position: positions.get(node.id) })) } }
        dirtyRef.current = false
        setSource(result.source); setBaseline(result.source); onPending(false); setStale(false)
        onApply(next)
      }
    } catch (reason) {
      if (!abort.signal.aborted) setError(reason instanceof Error ? reason.message : '公式校验失败。')
    } finally {
      if (!abort.signal.aborted && request === revision.current) setBusy(false)
    }
  }

  return <section aria-label="情景高级公式编辑器" className="mx-auto max-w-6xl space-y-4 p-4">
    <div><h2 className="text-lg font-bold text-slate-900">高级公式</h2><p className="mt-1 text-xs leading-5 text-slate-600">每行定义一个计算步骤，连接写为“节点.端口”。支持指标通用函数与数学表达式；检查并应用后同步到画布和向导。</p><p className="mt-2 rounded-lg bg-violet-50 p-3 text-xs text-violet-900">例如：<code>smooth = rolling_mean(market.value, 20, 20)</code>。再用分类算子把计算结果映射为市场状态。</p></div>
    <label className="block text-sm font-semibold">情景计算公式<textarea aria-label="情景计算公式" value={source} disabled={!loaded || busy} spellCheck={false} rows={12}
      onChange={event => { setSource(event.target.value); setResolution(null); setError('') }} className="mt-2 w-full rounded-xl border border-slate-300 bg-slate-50 p-4 font-mono text-sm leading-6 text-slate-800" /></label>
    {stale ? <p role="alert" className="text-sm text-amber-800">方案或识别模式已变化。当前公式草稿仍保留；请还原后重新编辑。</p> : null}
    {error ? <p role="alert" className="text-sm text-rose-700">{error}</p> : null}
    {resolution?.diagnostics.length ? <ul aria-label="公式诊断" className="space-y-1 text-sm text-amber-800">{resolution.diagnostics.map((item, index) => <li key={index}>{item.line ? `第 ${item.line} 行：` : ''}{item.message}</li>)}</ul> : null}
    <div className="flex flex-wrap items-center gap-3"><button type="button" disabled={!loaded || busy || stale} onClick={() => void apply()} className="rounded-lg bg-violet-600 px-4 py-2 text-sm font-bold text-white disabled:opacity-40">{busy ? '正在检查…' : '检查并应用公式'}</button>
      <button type="button" disabled={busy} onClick={() => { dirtyRef.current = false; setSource(baseline); onPending(false); setReload(value => value + 1) }} className="rounded-lg border border-slate-300 px-4 py-2 text-sm">还原为当前方案</button>
      <span className="text-xs text-slate-500">{dirty ? '存在未应用公式，保存和运行已暂停。' : '公式与当前方案同步。'}</span></div>
    <RegimeMathDisplay resolution={resolution} outputId={outputId} message={dirty || stale ? '公式尚未应用。检查并应用后更新数学公式。' : busy ? '正在生成数学公式…' : undefined} />
    <details className="rounded-xl border border-slate-200 p-3"><summary className="cursor-pointer text-sm font-semibold">语法与算子参考</summary><p className="my-3 text-xs text-slate-600">output(state=classifier.state, trend=smooth.value) 声明枚举和数值输出；expose(market, change) 指定调试节点。名称与版本随当前方案保留；完整定义可从版本面板导出。窗口等参数使用固定常量，不支持导入或任意程序。</p>
      <div className="grid gap-2 md:grid-cols-2">{schemas.map(schema => <div key={schema.id} className="rounded-lg bg-slate-50 p-2 text-xs"><strong>{schema.label}</strong><code className="mt-1 block break-all">{(schema.id || schema.type || '').replace(/\./g, '_')}({schema.inputs?.map(port => `${port.id}=节点.端口`).join(', ')})</code></div>)}</div>
    </details>
  </section>
}
