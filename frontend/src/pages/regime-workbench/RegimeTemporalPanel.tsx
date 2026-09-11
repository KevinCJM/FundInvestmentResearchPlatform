import type { TemporalCapability } from '../../services/regimeGraph'

export default function RegimeTemporalPanel({ report, stale = false, busy = false, disabled = false, onAudit, onNode }: {
  report?: TemporalCapability | null; stale?: boolean; busy?: boolean; disabled?: boolean
  onAudit?: () => void; onNode?: (id: string) => void
}) {
  const verified = report?.verified === true && !stale
  const hindsight = report?.status === 'retrospective_required'
  const title = stale ? '配置已变更，旧审计不再适用' : verified ? '本次时点审计通过' : report?.label || '时点能力尚未验证'
  return <section aria-label="时点能力与因果性审计" className={'min-w-0 space-y-3 rounded-xl border p-4 ' + (verified ? 'border-emerald-200 bg-emerald-50/40' : 'border-slate-200 bg-slate-50')}>
    <div className="flex flex-wrap items-center justify-between gap-3"><div><h3 className="text-sm font-semibold">{title}</h3><p className="mt-1 text-xs leading-5 text-slate-600">{hindsight ? '这是事后研究能力，不是算法错误；不能用它重建当时的交易信号。' : verified ? '仅验证本次定义、模式与数据；正式发布还需通过完整门禁。' : '先检查实际依赖与时间契约，再用真实计算检验未来数据是否改变过去。'}</p></div>{onAudit && <button type="button" disabled={busy || disabled} onClick={onAudit} className="min-h-10 rounded-lg border border-violet-200 bg-white px-3 text-sm font-semibold text-violet-700 disabled:opacity-40">{busy ? '计算与审计中…' : '因果性审计'}</button>}</div>
    {!stale && report?.reasons?.slice(0, 2).map((r, i) => <div key={r.node_id + i} className="flex flex-wrap items-start justify-between gap-2 text-xs leading-5"><span>{r.message}</span>{onNode && <button type="button" onClick={() => onNode(r.node_id)} className="min-h-9 shrink-0 text-violet-700 underline">定位节点</button>}</div>)}
    {report && <details className="rounded-lg border border-slate-200 bg-white p-3"><summary className="cursor-pointer text-xs font-semibold">查看审计依据与限制</summary>
      <div className="mt-3 grid gap-3 text-xs sm:grid-cols-2"><div><p className="text-slate-500">计算因果性</p><p>{report.numerical_verdict === 'causal' ? '已覆盖探针未发现未来依赖' : report.numerical_verdict === 'leak' ? '发现未来依赖反例' : '数值证据尚未充分验证'}</p></div><div><p className="text-slate-500">数据与信息时点</p><p>{report.data_checks ? report.data_checks.passed ? '本次来源时点契约已检查（日期级，不证明日内可交易）。' : '来源时点尚未充分核验，不能认定为实时可用。' : '运行后核对真实数据快照、公布时间与版本。'}</p></div><div><p className="text-slate-500">历史重绘</p><p>{report.may_repaint ? '存在后续确认或全样本回标依赖' : '未声明重绘；仍需前缀重放核对'}</p></div><div><p className="text-slate-500">事后语义</p><p>{report.semantic_hindsight ? '人工事后认定：必须用于事后研究' : '没有人工事后标签强制限制'}</p></div></div>
      {report.data_checks?.sources.filter(source => !source.passed).map(source => <p key={source.node_id} className="mt-2 text-xs text-amber-800">{source.node_id}：{source.message}</p>)}
      {report.coverage && <p className="mt-3 text-xs text-slate-600">实际重算 {report.coverage.executions} 次，比较 {report.coverage.comparisons} 项；未覆盖 {report.coverage.untested.length} 项。未覆盖不会当作通过。</p>}
      {report.warmup && <p className="mt-2 text-xs text-slate-600">预热检查：{report.warmup.sensitive ? '对历史长度敏感，需要固定预热策略；这不等于未来泄漏。' : report.warmup.status === 'tested' ? '本次未发现显著预热差异。' : '不适用或证据不足。'}</p>}
      {(report.findings || []).slice(0, 8).map((finding, i) => <div key={i} className="mt-2 break-words rounded-lg bg-amber-50 p-2 text-xs text-amber-950">{finding.node_id}.{finding.port}：{finding.message}{finding.first_mismatch_date && ` 首个差异：${finding.first_mismatch_date}`}{onNode && <button type="button" onClick={() => onNode(finding.node_id)} className="ml-2 min-h-9 underline">定位</button>}</div>)}
      {report.errors?.map((error, i) => <p key={i} className="mt-2 text-xs text-amber-800">{error}</p>)}
      <div className="mt-3 space-y-2">{Object.entries(report.outputs || {}).map(([id, output]) => <div key={id} className="break-words text-xs"><strong>输出 {id}</strong> · {output.status === 'conditional' ? '按当时信息计算的候选' : output.status === 'retrospective_required' ? '仅事后' : '未验证'}{output.reasons.slice(0, 2).map((r, i) => <p key={i} className="mt-1 text-slate-500">{r.path.join(' → ')}</p>)}</div>)}</div>
      <p className="mt-3 text-[11px] leading-5 text-slate-500">{report.note} · 规则 {report.policy_version}</p>
    </details>}
  </section>
}
