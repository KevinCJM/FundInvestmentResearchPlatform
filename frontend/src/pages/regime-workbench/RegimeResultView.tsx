import RegimeTemporalPanel from './RegimeTemporalPanel'
import { useEffect, useRef, useState } from 'react'
import { getRegimeFormalRun, getRegimePreviewOverview, getRegimePreviewSeries } from '../../services/regimeGraph'
import { adaptRegimeFormalOverview, adaptRegimeOverview, adaptRegimeResult, loadCompleteRegimeSeries, type RegimeResultData, type RegimeResultInterval } from './regimeResultAdapter'
import RegimeTimelineChart, { intervalColor } from './RegimeTimelineChart'
import RegimeEvidencePanel from './RegimeEvidencePanel'
import RegimeEvaluationResults from './RegimeEvaluationResults'
import RegimeNumericOutputs from './RegimeNumericOutputs'
import RegimeManualEventResult from './RegimeManualEventResult'

export interface RegimeResultViewProps {
  runId: string
  runKind?: 'preview' | 'formal'
  stale?: boolean
}
type ResultState = { key: string; status: 'loading' | 'ready' | 'error'; data?: RegimeResultData; error?: string }
const percent = (count: number, total: number) => total ? (count / total * 100).toFixed(1) + '%' : '—'
const frequencyLabel = (frequency: string | null | undefined) => ({ daily: '日频', weekly: '周频', monthly: '月频', quarterly: '季频', yearly: '年频', annual: '年频', irregular: '不定期' }[frequency || ''] || frequency || '未提供')

export default function RegimeResultView({ runId, runKind = 'preview', stale = false }: RegimeResultViewProps) {
  const identity = runKind + ':' + runId
  const [state, setState] = useState<ResultState>({ key: identity, status: 'loading' })
  const [retry, setRetry] = useState(0)
  const [tab, setTab] = useState<'intervals' | 'evaluation' | 'evidence' | 'details'>('intervals')
  const [selected, setSelected] = useState<{ key: string; interval: RegimeResultInterval } | null>(null)
  const generation = useRef(0)
  useEffect(() => {
    const current = ++generation.current
    const controller = new AbortController()
    setState({ key: identity, status: 'loading' })
    setSelected(null)
    const load = async () => {
      if (runKind === 'formal') {
        const detail = await getRegimeFormalRun(runId, controller.signal)
        const overview = adaptRegimeFormalOverview(detail, runId)
        return adaptRegimeResult(overview, detail.series ?? [])
      }
      const overview = adaptRegimeOverview(await getRegimePreviewOverview(runId, controller.signal), runId, 'preview')
      const rows = await loadCompleteRegimeSeries(overview, (offset, limit, signal) => getRegimePreviewSeries(runId, { offset, limit }, signal), controller.signal)
      return adaptRegimeResult(overview, rows)
    }
    void load().then(data => {
      if (!controller.signal.aborted && current === generation.current) setState({ key: identity, status: 'ready', data })
    }).catch(error => {
      if (!controller.signal.aborted && current === generation.current) setState({ key: identity, status: 'error', error: error instanceof Error ? error.message : '无法读取本次结果，请重试。' })
    })
    return () => { controller.abort() }
  }, [identity, runId, runKind, retry])

  if (state.key !== identity || state.status === 'loading') return <section role="status" className="grid min-h-80 place-items-center rounded-xl bg-white p-8 text-sm text-slate-600">正在加载完整情景结果与主对照走势…</section>
  if (state.status === 'error' || !state.data) return <section role="alert" className="space-y-3 rounded-xl border border-amber-200 bg-amber-50 p-6"><p className="font-semibold text-amber-950">结果暂不可用</p><p className="text-sm text-amber-900">{state.error}</p><button type="button" onClick={() => setRetry(value => value + 1)} className="rounded-lg border border-amber-400 bg-white px-3 py-2 text-sm font-semibold">重新读取结果</button></section>
  const result = state.data
  const { overview } = result
  if (overview.result_kind === 'manual_events') return <>{overview.temporal_capability && <RegimeTemporalPanel report={overview.temporal_capability} stale={stale} />}<RegimeManualEventResult result={result} stale={stale} /></>
  const interval = selected?.key === identity ? selected.interval : null
  const observationLabel = overview.frequency ? `${frequencyLabel(overview.frequency)}观测` : '观测点'
  const selectInterval = (interval: RegimeResultInterval) => { setSelected({ key: identity, interval }); setTab('evidence') }
  return <section aria-label="完整历史情景结果" className="min-w-0 space-y-4 p-3 sm:p-5" data-run-id={overview.run_id}>
    {overview.temporal_capability && <RegimeTemporalPanel report={overview.temporal_capability} stale={stale} />}
    <header className="flex flex-wrap items-start justify-between gap-3">
      <div><h2 className="text-lg font-bold text-slate-950">历史情景结果</h2><p className="mt-1 text-xs text-slate-600">{overview.date_range.start ?? '无样本'} 至 {overview.date_range.end ?? '无样本'} · 共 {overview.summary.total} 个{observationLabel}{overview.as_of ? ` · 截至 ${overview.as_of}` : ''} · {overview.created_at ? '运行于 ' + overview.created_at : '运行 ' + overview.run_id}</p></div>
      <span className="rounded-full bg-accent-50 px-3 py-1 text-xs font-bold text-accent-800">{overview.mode === 'retrospective' ? '事后解释' : '当时可知模式'} · {frequencyLabel(overview.frequency)}色带</span>
    </header>
    {stale ? <p role="status" className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">此结果与当前配置不同；以下名称、颜色及数据仍使用该次运行的冻结版本。修改后的配置需重新运行。</p> : null}
    <p className="rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600">{overview.capabilities.effective.reason || '当前仅展示观测日期区间，未提供完整生效时间轴。'}{overview.mode === 'retrospective' ? ' 事后色带不能直接当作当时可交易的信号。' : ''}</p>
    {result.points.length ? <RegimeTimelineChart result={result} selectedId={interval?.id ?? null} onSelect={selectInterval} /> : <p className="rounded-xl bg-slate-50 p-8 text-center text-sm text-slate-600">本次运行没有可展示的观测样本。</p>}
    <RegimeNumericOutputs result={result} />
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4" aria-label="全样本情景摘要">
      <Summary label="有效分类覆盖" value={percent(overview.summary.classified, overview.summary.total)} detail={overview.summary.classified + ' / ' + overview.summary.total + ` 个${observationLabel}`} />
      <Summary label="状态切换" value={String(overview.summary.switch_count)} detail="采用本次运行的完整序列统计" />
      <Summary label="未分类" value={String(overview.summary.unknown)} detail={percent(overview.summary.unknown, overview.summary.total) + ' · 不计入任何已命名状态'} />
      <Summary label="冻结版本" value={overview.definition_revision ? '修订 ' + overview.definition_revision : '草稿试算'} detail={overview.as_of ? '截至 ' + overview.as_of : '按本次锁定数据范围'} />
    </div>
    <div className="flex flex-wrap gap-x-4 gap-y-2 text-xs text-slate-600">{overview.states.map(state => <span key={state.id}>{state.label}：{overview.summary.state_counts[state.id] ?? 0} 个（{percent(overview.summary.state_counts[state.id] ?? 0, overview.summary.total)}）</span>)}<span>占比分母：全部 {overview.summary.total} 个{observationLabel}；图表缩放不改变统计。</span></div>
    <div className="flex flex-wrap gap-2 border-b border-slate-200" role="tablist" aria-label="结果分析">
      {([['intervals', '区间明细'], ['evaluation', '条件表现'], ['evidence', '判断依据'], ['details', '运行详情']] as const).map(([id, label]) => <button key={id} type="button" role="tab" id={'regime-result-tab-' + id} aria-selected={tab === id} aria-controls={'regime-result-panel-' + id} onClick={() => setTab(id)} className={'min-h-10 border-b-2 px-3 text-sm font-semibold ' + (tab === id ? 'border-accent-600 text-accent-700' : 'border-transparent text-slate-600')}>{label}</button>)}
    </div>
    <div role="tabpanel" id={'regime-result-panel-' + tab} aria-labelledby={'regime-result-tab-' + tab}>
      {tab === 'intervals' ? <div className="max-h-96 overflow-auto"><table className="min-w-full text-left text-xs" aria-label="完整情景区间明细"><thead className="sticky top-0 bg-slate-50 text-slate-600"><tr>{['状态', '起始日期', '结束日期', '观测数', '首点确认', '首点生效', '依据'].map(label => <th scope="col" key={label} className="whitespace-nowrap px-3 py-2">{label}</th>)}</tr></thead><tbody>{result.intervals.map(item => <tr key={item.id} className={'border-t border-slate-100 ' + (interval?.id === item.id ? 'bg-accent-50' : '')}><td className="whitespace-nowrap px-3 py-2"><span className="mr-2 inline-block h-2.5 w-2.5 rounded-lg" style={{ backgroundColor: intervalColor(result, item) }} />{item.label}</td><td className="whitespace-nowrap px-3 py-2">{item.start_date}</td><td className="whitespace-nowrap px-3 py-2">{item.end_date}</td><td className="px-3 py-2">{item.observations}</td><td className="whitespace-nowrap px-3 py-2">{item.confirmed_at ?? '未提供'}</td><td className="whitespace-nowrap px-3 py-2">{item.effective_start ?? '未提供'}</td><td className="px-3 py-2"><button type="button" onClick={() => selectInterval(item)} className="whitespace-nowrap text-accent-700 underline" aria-label={'查看 ' + item.label + ' ' + item.start_date + ' 的依据'}>查看依据</button></td></tr>)}</tbody></table></div> : null}
      {tab === 'evaluation' ? overview.evaluation_results && Object.keys(overview.evaluation_results).length ? <RegimeEvaluationResults results={overview.evaluation_results} mode={runKind === 'formal' ? 'formal' : 'preview'} /> : <p className="rounded-xl bg-slate-50 p-5 text-sm text-slate-600">本次运行尚未提供评估对象或条件表现。对照走势仅用于观察，不会改变原有分类。</p> : null}
      {tab === 'evidence' ? <RegimeEvidencePanel result={result} interval={interval} /> : null}
      {tab === 'details' ? <dl className="grid gap-3 break-all rounded-xl bg-slate-50 p-4 text-xs sm:grid-cols-2">{[['运行身份', overview.run_kind + ' / ' + overview.run_id], ['识别模式', overview.mode === 'retrospective' ? '事后解释' : '当时可知'], ['定义快照', overview.definition_hash], ['计算图快照', overview.graph_hash], ['频率 / 日历', frequencyLabel(overview.frequency) + ' / ' + (overview.calendar ?? '未提供')], ['时间口径', '真实观测日期双闭区间'], ['数据快照', JSON.stringify(overview.data_snapshots)]].map(([label, value]) => <div key={label}><dt className="text-slate-600">{label}</dt><dd className="mt-1 text-slate-800">{value}</dd></div>)}</dl> : null}
    </div>
  </section>
}

function Summary({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="rounded-xl border border-slate-200 bg-white p-3"><p className="text-xs text-slate-600">{label}</p><p className="mt-1 text-xl font-bold text-slate-950">{value}</p><p className="mt-1 text-xs text-slate-600">{detail}</p></div>
}
