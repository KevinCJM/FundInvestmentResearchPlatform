import type {
  RegimeEvaluationConditionalMetric,
  RegimeEvaluationResults as RegimeEvaluationResultsMap,
} from '../../services/regimeGraph'

function textField(source: Record<string, unknown> | undefined, keys: string[]) {
  for (const key of keys) {
    const value = source?.[key]
    if (typeof value === 'string' && value) return value
    if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  }
  return ''
}

function percentage(value?: number | null) {
  return value == null || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(2)}%`
}

function number(value?: number | null) {
  return value == null || !Number.isFinite(value) ? '—' : value.toFixed(2)
}

function ConditionalMetricsTable({ name, rows }: { name: string; rows: RegimeEvaluationConditionalMetric[] }) {
  return <div className="mt-3 overflow-x-auto rounded-lg border border-slate-200"><table className="min-w-[720px] w-full text-left text-xs"><caption className="sr-only">{name}分状态条件表现</caption><thead className="bg-slate-50 text-slate-600"><tr><th scope="col" className="px-2 py-2">状态</th><th scope="col" className="px-2 py-2 text-right">样本数</th><th scope="col" className="px-2 py-2 text-right">年化收益</th><th scope="col" className="px-2 py-2 text-right">波动率</th><th scope="col" className="px-2 py-2 text-right">最大回撤</th><th scope="col" className="px-2 py-2 text-right">夏普</th><th scope="col" className="px-2 py-2 text-right">胜率</th></tr></thead><tbody>{rows.map((row) => <tr key={row.state_id} className="border-t border-slate-100"><th scope="row" className="px-2 py-2 font-bold text-slate-800">{row.state_label || row.state_id}</th><td className="px-2 py-2 text-right tabular-nums">{row.observations ?? '—'}</td><td className="px-2 py-2 text-right tabular-nums">{percentage(row.annualized_return)}</td><td className="px-2 py-2 text-right tabular-nums">{percentage(row.volatility)}</td><td className="px-2 py-2 text-right tabular-nums text-rose-700">{percentage(row.max_drawdown)}</td><td className="px-2 py-2 text-right tabular-nums">{number(row.sharpe)}</td><td className="px-2 py-2 text-right tabular-nums">{percentage(row.win_rate ?? row.positive_rate)}</td></tr>)}</tbody></table></div>
}

export default function RegimeEvaluationResults({
  results,
  mode,
}: {
  results?: RegimeEvaluationResultsMap
  mode: 'preview' | 'formal'
}) {
  const items = Object.values(results ?? {}).sort((left, right) => Number(right.primary) - Number(left.primary) || left.name.localeCompare(right.name, 'zh-CN'))
  if (!items.length) return null

  return <section className="rounded-xl border border-slate-200 bg-slate-50/70 p-3" aria-label={mode === 'preview' ? '试算评价目标结果' : '正式运行评价目标结果'}><div className="flex flex-wrap items-center justify-between gap-2"><div><h4 className="text-xs font-bold text-slate-900">评价目标</h4><p className="mt-1 text-xs text-slate-600">分类输入与表现评价分离；以下内容直接来自后端运行结果。</p></div><span className="rounded-full bg-white px-2 py-1 text-xs font-bold text-slate-600">{items.length} 个目标</span></div><div className="mt-3 grid gap-2 lg:grid-cols-2">{items.map((item) => {
    const kind = textField(item.source, ['kind']) || textField(item.snapshot, ['kind']) || '—'
    const firstDate = textField(item.snapshot, ['first_observation_date', 'start_date', 'min_date'])
    const lastDate = textField(item.snapshot, ['last_observation_date', 'end_date', 'max_date', 'latest_date'])
    const artifactShape = item.artifact?.shape
    const artifactObservations = Array.isArray(artifactShape) && typeof artifactShape[0] === 'number' ? String(artifactShape[0]) : ''
    const observationCount = textField(item.snapshot, ['selected_observations', 'row_count', 'observations']) || artifactObservations
    const identity = textField(item.snapshot, ['snapshot_id', 'snapshot_generation', 'artifact_id', 'fingerprint', 'file_checksum', 'checksum'])
    const metrics = Array.isArray(item.conditional_metrics) ? item.conditional_metrics : []
    return <article key={item.id} className="min-w-0 rounded-xl border border-slate-200 bg-white p-3"><div className="flex items-start justify-between gap-2"><div className="min-w-0"><p className="truncate text-xs font-bold text-slate-900" title={item.name}>{item.name}</p><p className="mt-1 text-xs text-slate-600">{item.id} · {kind}</p></div>{item.primary ? <span className="shrink-0 rounded-full bg-accent-100 px-2 py-1 text-xs font-bold text-accent-800">主要评价目标</span> : null}</div><dl className="mt-3 grid grid-cols-2 gap-2 text-xs"><div><dt className="text-slate-600">有效区间</dt><dd className="mt-0.5 font-semibold text-slate-700">{firstDate || '—'}{firstDate || lastDate ? ` → ${lastDate || '—'}` : ''}</dd></div><div><dt className="text-slate-600">有效观测</dt><dd className="mt-0.5 font-semibold text-slate-700">{observationCount || '—'}</dd></div><div className="col-span-2"><dt className="text-slate-600">数据版本指纹</dt><dd className="mt-0.5 truncate font-mono text-slate-600" title={identity}>{identity || '—'}</dd></div></dl>{metrics.length ? <details className="mt-3"><summary className="cursor-pointer text-xs font-bold text-accent-700">查看分状态条件表现（{metrics.length} 个状态）</summary><ConditionalMetricsTable name={item.name} rows={metrics} /></details> : mode === 'preview' ? <div className="mt-3 rounded-lg bg-slate-50 px-2 py-2 text-xs leading-4 text-slate-600"><p>试算仅返回评价目标血缘；分状态条件表现由正式运行在服务端计算。</p><a href="#regime-formal-experiments" className="mt-1 inline-block font-bold text-accent-700 underline underline-offset-2">进入正式实验生成条件表现</a></div> : <p className="mt-3 rounded-lg bg-slate-50 px-2 py-2 text-xs leading-4 text-slate-600">服务端未返回该目标的分状态条件表现。</p>}</article>
  })}</div></section>
}
