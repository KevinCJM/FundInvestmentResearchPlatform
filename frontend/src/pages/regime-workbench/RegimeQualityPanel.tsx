import { useEffect, useRef, useState } from 'react'
import { Badge, Button, Card, SectionHeader } from '../../components/ui'
import { confirmRegimeQuality, definitionForRequest, getRegimeQuality, listRegimeQuality, prepareRegimeGraph, previewRegimeQuality, type RegimeGraphDefinition } from '../../services/regimeGraph'
import type { RegimeQualityCatalogItem, RegimeQualityPreview, RegimeQualityRequest, SavedRegimeQuality } from '../../services/regimeDiagnostics'
import RegimeDiagnosticControls, { defaultStability } from './RegimeDiagnosticControls'
import { diagnosticNumber, diagnosticPercent, diagnosticReason, RegimeStabilityView } from './RegimeDiagnosticsView'

export default function RegimeQualityPanel({ definition, dirty, valid, contextKey, asOf, active, onSave }: {
  definition: RegimeGraphDefinition; dirty: boolean; valid: boolean; contextKey: string; asOf: string; active: boolean; onSave: () => void
}) {
  const [policy, setPolicy] = useState<RegimeQualityRequest['policy']>({ stability: defaultStability, include_price_returns: true, minimum_state_episodes_for_estimation: 3 })
  const [result, setResult] = useState<{ key: string; value: RegimeQualityPreview | SavedRegimeQuality } | null>(null)
  const [busy, setBusy] = useState(''), [error, setError] = useState('')
  const [catalog, setCatalog] = useState<RegimeQualityCatalogItem[]>([]), [selected, setSelected] = useState('')
  const [catalogError, setCatalogError] = useState(''), [catalogLoading, setCatalogLoading] = useState(false), [refresh, setRefresh] = useState(0)
  const operation = useRef<AbortController | null>(null)
  const identity = JSON.stringify([definitionForRequest(definition), dirty, valid, contextKey, asOf, policy, active])
  const latest = useRef(identity); latest.current = identity
  useEffect(() => {
    operation.current?.abort(); setResult(null); setBusy(''); setError(''); setSelected('')
    return () => operation.current?.abort()
  }, [identity])
  useEffect(() => {
    if (!active || !definition.id) return
    const controller = new AbortController()
    setCatalogLoading(true); setCatalogError('')
    void listRegimeQuality(controller.signal).then(items => { if (!controller.signal.aborted) setCatalog(items) })
      .catch(reason => { if (!controller.signal.aborted) setCatalogError(diagnosticReason(reason instanceof Error ? reason.message : null) || '报告目录读取失败，请刷新重试。') })
      .finally(() => { if (!controller.signal.aborted) setCatalogLoading(false) })
    return () => controller.abort()
  }, [active, definition.id, definition.revision, refresh])
  const current = result?.key === identity ? result.value : null
  const saved = current && 'id' in current ? current as SavedRegimeQuality : null
  const report = current?.report
  const matching = catalog.filter(item => item.definition_id === definition.id && item.revision === definition.revision)
  const blocked = dirty || !definition.id || !definition.revision ? '请先保存当前历史状态定义，再检查这个精确修订。' : !valid ? '请先修复计算图或输入数据的问题。' : !active ? '返回历史状态定义后再检查。' : ''
  const execute = async (label: string, action: (signal: AbortSignal) => Promise<RegimeQualityPreview | SavedRegimeQuality>) => {
    operation.current?.abort()
    const controller = new AbortController(); operation.current = controller
    const key = identity
    setBusy(label); setError(''); setResult(null)
    try {
      const value = await action(controller.signal)
      if (controller.signal.aborted || key !== latest.current) return
      if (value.request.definition_id !== definition.id || value.request.revision !== definition.revision || value.request.mode !== 'retrospective' || (value.request.as_of || '') !== asOf) throw new Error('报告的定义修订或截至日不同，请选择对应版本后重新检查。')
      setResult({ key, value })
      if ('id' in value) { setSelected(value.id); setRefresh(n => n + 1) }
    } catch (reason) {
      if (!controller.signal.aborted && key === latest.current) setError(diagnosticReason(reason instanceof Error ? reason.message : null) || '质量检查失败，请检查输入后重试。')
    } finally { if (!controller.signal.aborted && key === latest.current) setBusy('') }
  }
  const preview = () => {
    if (blocked) return
    const request: RegimeQualityRequest = { definition_id: definition.id!, revision: definition.revision!, mode: 'retrospective', as_of: asOf || null, policy }
    void execute('正在检查划分质量…', async signal => {
      const plan = await prepareRegimeGraph(definition, signal)
      if (signal.aborted || latest.current !== identity) throw new DOMException('Cancelled', 'AbortError')
      return previewRegimeQuality(request, plan.compile_token, signal)
    })
  }
  const labels = Object.fromEntries(definition.states.map(state => [state.id, state.label]))
  return <Card className="mb-5 min-w-0 space-y-4">
    <SectionHeader title="检查划分质量" description="按需检查已保存的历史状态定义，确认保存后可重载同一份质量报告。" />
    <label className="block max-w-sm text-sm text-slate-700">LTCMA研究样本门槛：每状态最少独立完整区间<input aria-label="LTCMA研究样本门槛：每状态最少独立完整区间" type="number" min={1} max={100} step={1} value={policy.minimum_state_episodes_for_estimation ?? 3} onChange={event => setPolicy(p => ({ ...p, minimum_state_episodes_for_estimation: Math.min(100, Math.max(1, Number(event.target.value))) }))} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm tabular-nums" /></label>
    <RegimeDiagnosticControls stability={policy.stability || defaultStability} onStability={stability => setPolicy(p => ({ ...p, stability }))} />
    <div className="flex flex-wrap gap-2">
      <Button tone="primary" disabled={Boolean(blocked || busy)} onClick={preview}>检查划分质量</Button>
      {(dirty || !definition.id) && <Button onClick={onSave}>先保存历史状态定义</Button>}
      {busy && <Button onClick={() => { operation.current?.abort(); setBusy(''); setError('已取消检查，可重新运行。') }}>取消质量检查</Button>}
      {current && !saved && <Button disabled={Boolean(blocked || busy)} onClick={() => void execute('正在保存质量报告…', signal => confirmRegimeQuality({ request: current.request, preview_hash: current.preview_hash }, signal))}>确认保存质量报告</Button>}
      {saved && <Badge>质量报告已保存</Badge>}
    </div>
    {blocked && <p className="text-sm text-slate-600">{blocked}</p>}
    {!report && !busy && !blocked && !error && <p className="text-sm text-slate-600">尚未检查当前修订。点击检查划分质量，查看覆盖与边界敏感性。</p>}
    {busy && <div role="status" className="space-y-2 text-sm text-slate-600"><div className="h-4 rounded bg-slate-100" /><div className="h-4 w-2/3 rounded bg-slate-100" />{busy}</div>}
    {error && <p role="alert" className="text-sm text-rose-800">{error}</p>}
    {report && <section aria-label="历史划分质量报告" className="min-w-0 space-y-3 border-t border-slate-200 pt-4 text-sm tabular-nums">
      <Badge tone="warning">{report.status === 'insufficient_evidence' ? '划分证据不足' : '划分诊断，不代表标签真值'}</Badge>
      <p className="text-slate-700">分类覆盖 {diagnosticPercent(report.sample.coverage)} · 已分类 {report.sample.classified} / {report.sample.input} 个观测 · 未分类 {report.sample.unknown}。</p>
      <p className="text-slate-600">首端未分类 {report.sample.head_unknown} · 尾端未分类 {report.sample.tail_unknown} 个观测。未完成的首尾不能当作完整状态区间；全未分类时首尾计数重叠。</p>
      <p className="text-slate-600">{report.sample.first_date || '无起始日期'} 至 {report.sample.last_date || '无结束日期'} · 状态区间 {report.segments.total} · 转折 {report.segments.transitions}。</p>
      {report.horizon_profile && <section aria-label="市场状态实测时间尺度" className="space-y-2 rounded-xl border border-slate-200 p-3">
        <p className="font-medium text-slate-800">实测状态持续期</p>
        <p className="text-xs text-slate-600">这不是研究员填写的 Horizon 标签。系统只使用两侧边界都已完成的独立状态区间；开放首尾与未知标签隔开的区间不进入分位数。{report.horizon_profile.calendar_boundary === 'observation_inclusive_next_observation_exclusive' ? '日历持续期包含起始观测日，持续到下一状态的首个观测日（不含）；月末单次观测因此覆盖至下一月末。' : '此旧报告按首末样本点跨度统计日历天数，未计入下一状态边界；重新检查可生成新版持续期证据。'}</p>
        <p className="text-xs text-slate-600">观测频率 {report.horizon_profile.observation_frequency} · 年化状态转换 {diagnosticNumber(report.horizon_profile.transitions_per_year)} 次 · 已分类覆盖 {diagnosticPercent(report.horizon_profile.classified_coverage)}。</p>
        <div className="overflow-x-auto"><table aria-label="市场状态持续期统计" className="w-full text-xs"><thead><tr>{['状态', '完整区间', '观测期 P25 / 中位 / P75', '日历天 P25 / 中位 / P75', '已分类时间占比'].map(title => <th scope="col" key={title} className="p-2 text-left">{title}</th>)}</tr></thead><tbody>{report.horizon_profile.per_state.map(state => <tr key={state.state_id} className="border-b border-slate-100"><th scope="row" className="p-2 text-left">{labels[state.state_id] || state.state_id}</th><td className="p-2 text-right">{state.independent_complete_episodes}</td><td className="p-2 text-right">{[state.duration_observations_p25, state.duration_observations_median, state.duration_observations_p75].map(diagnosticNumber).join(' / ')}</td><td className="p-2 text-right">{[state.duration_calendar_days_p25, state.duration_calendar_days_median, state.duration_calendar_days_p75].map(diagnosticNumber).join(' / ')}</td><td className="p-2 text-right">{diagnosticPercent(state.classified_occupancy)}</td></tr>)}</tbody></table></div>
      </section>}
      <p className="text-slate-600">仅检查独立区间数量，未估计长期收益、协方差、估计误差或 LTCMA 分布。LTCMA研究样本：{report.conditional_estimation.status === 'ready' ? '全部状态达到区间数量门槛' : report.conditional_estimation.status === 'partially_ready' ? '部分状态达到区间数量门槛' : '独立状态区间不足'}。</p>
      <RegimeStabilityView result={report.stability} />
      <details><summary className="min-h-10 cursor-pointer text-slate-600">状态支持、持续时间与区间收益</summary>
        <p className="text-xs text-slate-600">持续时间按输入观测数计，不是日历天数。收益只在经核验的价格或净值区间上计算。</p>
        {report.price_returns.status !== 'available' && <p className="text-slate-600">{diagnosticReason(report.price_returns.reason) || '当前没有可用的区间收益证据。'}</p>}
        <div className="overflow-x-auto"><table aria-label="历史状态区间统计" className="w-full text-xs"><thead><tr>{['状态', '观测', '区间', '独立完整区间', '区间样本门槛', '最短', '中位', '最长', '平均持续', '平均区间收益', '收益样本'].map(title => <th scope="col" key={title} className="p-2 text-left">{title}</th>)}</tr></thead><tbody>{report.segments.per_state.map(state => <tr key={state.state_id} className="border-b border-slate-100"><th scope="row" className="p-2 text-left">{labels[state.state_id] || state.state_id}</th><td className="p-2 text-right">{state.observations}</td><td className="p-2 text-right">{state.segments}</td><td className="p-2 text-right">{state.independent_complete_episodes}</td><td className="p-2">{state.conditional_estimation_status === 'ready' ? '数量达标' : '证据不足'}</td>{[state.min_length, state.median_length, state.max_length].map((n, i) => <td className="p-2 text-right" key={i}>{n ?? '无法估计'}</td>)}<td className="p-2 text-right">{diagnosticNumber(state.mean_length)}</td><td className="p-2 text-right">{diagnosticPercent(state.mean_price_return)}{state.price_return_reason && <span className="block text-slate-600">{diagnosticReason(state.price_return_reason)}</span>}</td><td className="p-2 text-right">{state.price_return_samples}</td></tr>)}</tbody></table></div>
      </details>
      {report.warnings.map((warning, i) => <p key={i} className="text-xs text-slate-600">{diagnosticReason(warning)}</p>)}
      <p className="text-slate-600">保存质量报告不会发布历史参考。需要供实时识别使用时，在保存流程中单独确认历史参考。</p>
      <Button onClick={onSave}>前往保存历史参考</Button>
      <details><summary className="min-h-10 cursor-pointer text-slate-600">精确版本、参数与快照来源</summary><p className="text-xs text-slate-600">{saved ? `不可变报告，保存于 ${saved.created_at}` : '当前为未保存预览'}。以下为这份报告实际执行的参数与来源。</p><pre className="max-h-64 overflow-auto whitespace-pre-wrap break-all text-xs text-slate-600">{JSON.stringify({ request: current.request, preview_hash: current.preview_hash, ...(saved ? { id: saved.id, content_hash: saved.content_hash } : {}), lineage: report.lineage, stability: report.stability }, null, 2)}</pre></details>
    </section>}
    <details><summary className="min-h-10 cursor-pointer text-sm text-slate-600">已保存质量报告</summary>
      {catalogLoading && <p role="status" className="text-sm text-slate-600">正在读取质量报告目录…</p>}
      {catalogError && <p role="alert" className="text-sm text-rose-800">{catalogError}</p>}
      {!catalogLoading && !catalogError && !matching.length && <p className="text-sm text-slate-600">当前修订尚无质量报告，检查后确认保存即可重载。</p>}
      <div className="flex min-w-0 flex-wrap items-end gap-2"><label className="min-w-0 flex-1 text-sm text-slate-700">精确修订的质量报告<select aria-label="已保存质量报告" disabled={Boolean(blocked || busy || catalogLoading)} value={selected} onChange={e => setSelected(e.target.value)} className="mt-1 min-h-10 w-full min-w-0 rounded-lg border border-slate-300 bg-white px-2"><option value="">选择报告</option>{matching.map(item => <option key={item.id} value={item.id}>{item.created_at} · {item.status === 'insufficient_evidence' ? '划分证据不足' : item.conditional_estimation?.status === 'ready' ? 'LTCMA研究样本数量达标' : item.conditional_estimation?.status === 'partially_ready' ? '部分状态样本数量达标' : item.conditional_estimation ? 'LTCMA研究样本不足' : '划分诊断'}</option>)}</select></label><Button disabled={!selected || Boolean(blocked || busy)} onClick={() => void execute('正在读取质量报告…', signal => getRegimeQuality(selected, signal))}>载入质量报告</Button><Button onClick={() => setRefresh(n => n + 1)}>刷新质量报告目录</Button></div>
    </details>
  </Card>
}
