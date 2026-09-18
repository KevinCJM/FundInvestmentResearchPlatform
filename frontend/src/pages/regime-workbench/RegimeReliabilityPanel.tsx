import RegimeProspectivePanel from './RegimeProspectivePanel'
import { diagnosticReason } from './RegimeDiagnosticsView'
import RegimeDiagnosticControls, { defaultBootstrap, defaultStability } from './RegimeDiagnosticControls'
import { useEffect, useRef, useState } from 'react'
import { Badge, Button, Card, SectionHeader } from '../../components/ui'
import {
  confirmRegimeReliability, definitionForRequest, getRegimeReliability, listRegimeReliability,
  prepareRegimeGraph, previewRegimeReliability,
  type RegimeGraphDefinition, type RegimeReliabilityCatalogItem, type RegimeReliabilityPolicy,
  type RegimeReliabilityPreview, type SavedRegimeReliability,
} from '../../services/regimeGraph'
import { referenceKey } from './RegimeReferenceBinding'
import RegimeReliabilityReportView from './RegimeReliabilityReportView'

const field = 'mt-1 block min-h-10 w-full min-w-0 rounded-lg border border-slate-300 bg-white px-3 text-sm tabular-nums'
const sameReference = (a: RegimeReliabilityCatalogItem['reference'], b: RegimeGraphDefinition['study']) => referenceKey(a) === referenceKey(b?.reference)
const isBoundCalibration = (definition: RegimeGraphDefinition, reportId: string | null | undefined) => Boolean(reportId && definition.study?.calibration_id === reportId)
const belongsToCurrentModel = (definition: RegimeGraphDefinition, report: Pick<RegimeReliabilityCatalogItem, 'id' | 'definition_id' | 'revision' | 'reference'>) =>
  report.definition_id === definition.id
  && sameReference(report.reference, definition.study)
  && (report.revision === definition.revision || isBoundCalibration(definition, report.id))
const catalogLabel = (item: RegimeReliabilityCatalogItem) => item.verification?.status === 'verified' ? '状态验证通过'
  : item.verification?.status === 'partially_verified' ? '部分状态验证通过'
    : item.verification?.status === 'failed' ? '状态验证未通过'
      : item.verification?.status === 'insufficient_evidence' ? '状态证据不足'
        : item.status === 'eligible' && item.calibration?.deployment_eligible ? '校准可用'
          : item.status === 'retrospective_only' ? '仅回顾评分' : item.status === 'eligible' ? '研究诊断' : '证据不足'
export default function RegimeReliabilityPanel({ definition, dirty, valid, contextKey, onSave, onApplyCalibration, onApplyQualification, onInvalidateCalibration, active = true }: {
  definition: RegimeGraphDefinition; dirty: boolean; valid: boolean; contextKey: string; onSave: () => void; onApplyCalibration?: (id: string) => void; onApplyQualification?: (calibrationId: string, qualificationId: string, sources?: import('../../services/regimeProspective').IndexSnapshotBindings) => void; onInvalidateCalibration?: () => void; active?: boolean
}) {
  const [policy, setPolicy] = useState<RegimeReliabilityPolicy>({ calibration_end: '', minimum_samples: 60, minimum_class_samples: 10, minimum_segments: 3, minimum_state_episodes: 3, minimum_state_predictions: 5, minimum_state_precision: 0.65, bins: 10, transition_tolerance: 3, confidence_floor: 0.6, calibration_method: 'auto', stability: defaultStability, bootstrap: defaultBootstrap })
  const [result, setResult] = useState<{ key: string; value: RegimeReliabilityPreview | SavedRegimeReliability } | null>(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [catalogError, setCatalogError] = useState('')
  const [catalog, setCatalog] = useState<RegimeReliabilityCatalogItem[]>([])
  const [catalogLoading, setCatalogLoading] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const [selected, setSelected] = useState('')
  const operation = useRef<AbortController | null>(null)
  const identity = JSON.stringify([definitionForRequest(definition), contextKey, policy, dirty, valid, active])
  const latest = useRef(identity); latest.current = identity
  const mounted = useRef(true)
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; operation.current?.abort() } }, [])
  useEffect(() => { if (!active) { operation.current?.abort(); setBusy('') } }, [active])
  useEffect(() => { operation.current?.abort(); setBusy(''); setError(''); setResult(null); setSelected('') }, [identity])
  useEffect(() => {
    if (!active) return
    const controller = new AbortController()
    setCatalogLoading(true); setCatalogError('')
    void listRegimeReliability(controller.signal).then(items => {
      if (controller.signal.aborted) return
      setCatalog(items)
      if (definition.study?.calibration_id && items.some(item => belongsToCurrentModel(definition, item) && item.id === definition.study?.calibration_id)) {
        setSelected(definition.study.calibration_id)
      }
    })
      .catch(reason => { if (!controller.signal.aborted) setCatalogError(reason instanceof Error ? diagnosticReason(reason.message) : '报告目录读取失败。') })
      .finally(() => { if (!controller.signal.aborted) setCatalogLoading(false) })
    return () => controller.abort()
  }, [active, refresh, definition.id, definition.revision])

  const current = result?.key === identity ? result.value : null
  const saved = current && 'id' in current ? current as SavedRegimeReliability : null
  const matching = catalog.filter(item => belongsToCurrentModel(definition, item))
  const blocked = !definition.study?.reference ? '选择历史参考后才能验证。'
    : dirty || !definition.id || !definition.revision ? '请先保存当前模型，再验证这个精确修订。'
      : !valid ? '请先修复定义或状态对应问题。'
        : !policy.calibration_end ? '请选择校准截止日。'
          : policy.validation_end && policy.validation_end <= policy.calibration_end ? '验证截止日须晚于校准截止日。'
            : policy.test_end && policy.test_end <= (policy.validation_end || policy.calibration_end) ? '测试截止日须晚于前一时间块。' : ''
  const execute = async (label: string, action: (signal: AbortSignal) => Promise<RegimeReliabilityPreview | SavedRegimeReliability>) => {
    if (operation.current && !operation.current.signal.aborted && busy) return
    operation.current?.abort()
    const controller = new AbortController(); operation.current = controller
    const key = identity
    setBusy(label); setError('')
    try {
      const value = await action(controller.signal)
      if (!mounted.current || controller.signal.aborted || latest.current !== key) return
      const exactRevision = value.request.revision === definition.revision
      const explicitlyBound = 'id' in value && isBoundCalibration(definition, value.id)
      if (value.request.definition_id !== definition.id || (!exactRevision && !explicitlyBound) || !sameReference(value.request.reference, definition.study)) throw new Error('报告不属于当前模型和历史参考，请重新验证。')
      setResult({ key, value })
      if ('id' in value) { setSelected(value.id); setRefresh(value => value + 1) }
    } catch (reason) {
      if (mounted.current && !controller.signal.aborted && latest.current === key) setError(reason instanceof Error ? diagnosticReason(reason.message) : '验证失败，请重试。')
    } finally {
      if (mounted.current && !controller.signal.aborted && latest.current === key) setBusy('')
    }
  }
  const preview = () => {
    if (blocked || !definition.study?.reference || !definition.id || !definition.revision) return
    const request = { definition_id: definition.id, revision: definition.revision, reference: definition.study.reference, policy }
    setResult(null)
    void execute('正在验证识别能力…', async signal => {
      const plan = await prepareRegimeGraph(definition, signal)
      if (signal.aborted || latest.current !== identity) throw new DOMException('Cancelled', 'AbortError')
      return previewRegimeReliability(request, plan.compile_token, signal)
    })
  }
  const patchPolicy = (patch: Partial<RegimeReliabilityPolicy>) => {
    if (patch.calibration_method && patch.calibration_method !== policy.calibration_method && (definition.study?.calibration_id || definition.study?.qualification_id)) onInvalidateCalibration?.()
    setPolicy(current => ({ ...current, ...patch }))
  }
  return <Card className="mb-5 min-w-0 space-y-4">
    <SectionHeader title="验证识别能力" description="评价相对所选历史参考的表现，校准与后续测试使用独立时间段。" />
    <p className="text-sm leading-6 text-slate-600">先固定参考与评估时间范围，再比较模型。模型原始概率、相对参考校准后的匹配概率、前瞻验证资格是三种不同证据；规则输出的 0/1 不代表预测正确率。</p>
    <div className="grid min-w-0 gap-3 sm:grid-cols-3">
      {([['calibration_end', '校准截止日'], ['validation_end', '验证截止日（可选）'], ['test_end', '测试截止日（可选）']] as const).map(([key, label]) => <label key={key} className="min-w-0 text-sm text-slate-700">{label}<input aria-label={label} type="date" className={field} value={policy[key] || ''} onChange={event => patchPolicy({ [key]: event.target.value || null })} /></label>)}
    </div>
    <details><summary className="min-h-10 cursor-pointer text-sm text-slate-600">样本门槛与校准设置</summary>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{([
        ['minimum_samples', '最少样本数', 2, 20000, 1], ['minimum_class_samples', '每类最少样本', 1, 20000, 1], ['minimum_segments', '最少参考区间', 1, 1000, 1],
        ['minimum_state_episodes', '每状态最少独立完整区间', 1, 100, 1], ['minimum_state_predictions', '每状态最少高置信预测', 1, 20000, 1], ['minimum_state_precision', '状态验证最低 Precision', 0, 1, 0.01],
        ['bins', '可靠性分箱数', 2, 30, 1], ['transition_tolerance', '转折容忍（观测间隔）', 0, 60, 1], ['confidence_floor', '接受判断的最低匹配概率', 0, 1, 0.01],
      ] as const).map(([key, label, min, max, step]) => <label key={key} className="min-w-0 text-sm text-slate-700">{label}<input aria-label={label} className={field} type="number" min={min} max={max} step={step} value={policy[key]} onChange={event => patchPolicy({ [key]: Math.min(max, Math.max(min, Number(event.target.value))) })} /></label>)}
        <label className="text-sm text-slate-700">校准方法<select className={field} aria-label="校准方法" value={policy.calibration_method} onChange={event => patchPolicy({ calibration_method: event.target.value as RegimeReliabilityPolicy['calibration_method'] })}><option value="auto">按证据类型选择</option><option value="temperature">温度校准（模型概率）</option><option value="class_frequency">同类历史平均（规则）</option></select></label>
      </div>
    </details>
    <RegimeDiagnosticControls stability={policy.stability || defaultStability} bootstrap={policy.bootstrap || defaultBootstrap} onStability={stability => patchPolicy({ stability })} onBootstrap={bootstrap => patchPolicy({ bootstrap })} />
    <div className="flex flex-wrap gap-2">
      <Button tone="primary" disabled={Boolean(blocked || busy)} onClick={preview}>验证识别能力</Button>
      <Button disabled={Boolean(blocked || busy || policy.stability?.enabled === false)} onClick={preview}>检查稳定性</Button>
      {(dirty || !definition.id) && <Button onClick={onSave}>先保存当前模型</Button>}
      {busy && <Button onClick={() => { operation.current?.abort(); setBusy(''); setError('已取消本次请求。') }}>取消验证请求</Button>}
      {current && !saved && <Button disabled={Boolean(busy || blocked)} onClick={() => void execute('正在保存报告…', signal => confirmRegimeReliability(current, signal))}>保存验证报告</Button>}
      {saved && <Badge>已保存的不可变报告</Badge>}
      {saved?.calibration_id && <Button disabled={!saved.report.calibration?.deployment_eligible || !onApplyCalibration || dirty} onClick={() => { if (saved.calibration_id && saved.report.calibration?.deployment_eligible) onApplyCalibration?.(saved.calibration_id) }}>采用此校准结果</Button>}
    </div>
    {saved?.calibration_id && !saved.report.calibration?.deployment_eligible && <p className="text-sm text-slate-600">这个历史报告尚不能直接部署；拟合完成并保存后，可登记独立前瞻验证。</p>}
    {policy.stability?.enabled === false && <p className="text-sm text-slate-600">稳定性检查已关闭；可在高级检查设置中启用。</p>}
    {blocked && <p className="text-sm text-slate-600">{blocked}</p>}
    {busy && <div role="status" className="space-y-2 text-sm text-slate-600"><div className="h-4 rounded bg-slate-100" /><div className="h-4 w-2/3 rounded bg-slate-100" />{busy}</div>}
    {error && <p role="alert" className="text-sm text-rose-800">{error}</p>}
    {current && <><RegimeReliabilityReportView report={current.report} savedAt={saved?.created_at} />
      {saved && <RegimeProspectivePanel definition={definition} saved={saved} dirty={dirty} valid={valid} contextKey={identity} active={active} onApplyQualification={onApplyQualification} />}
      <details><summary className="min-h-10 cursor-pointer text-sm text-slate-600">报告实际参数与保存凭据</summary>
        <p className="text-xs text-slate-600">{saved ? '当前载入的是不可变历史报告，使用保存时的策略与数据；上方设置用于下一次验证。' : '当前预览使用以下服务器规范请求；确认仅提交请求和预览凭据。'}</p>
        <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-all text-xs text-slate-600">{JSON.stringify({ request: current.request, preview_hash: current.preview_hash, ...(saved ? { id: saved.id, content_hash: saved.content_hash } : {}) }, null, 2)}</pre>
      </details></>}
    <details><summary className="min-h-10 cursor-pointer text-sm text-slate-600">已保存报告</summary>
      {catalogLoading && <p role="status" className="text-sm text-slate-600">正在读取报告目录…</p>}
      {catalogError && <p role="alert" className="text-sm text-rose-800">{catalogError}</p>}
      {!catalogLoading && !catalogError && !matching.length && <p className="text-sm text-slate-600">这个模型修订与参考还没有保存报告。完成验证后可保存。</p>}
      <div className="flex min-w-0 flex-wrap items-end gap-2"><label className="min-w-0 flex-1 text-sm text-slate-700">精确版本的报告<select className={field} aria-label="已保存验证报告" disabled={dirty || !valid || !active || Boolean(busy) || catalogLoading} value={selected} onChange={event => setSelected(event.target.value)}><option value="">选择报告</option>{matching.map(item => <option key={item.id} value={item.id}>{item.created_at} · {catalogLabel(item)}{item.verification?.recognition_ready ? ' · 识别验证可用' : ''}</option>)}</select></label><Button disabled={!selected || dirty || !valid || !active || Boolean(busy) || catalogLoading} onClick={() => void execute('正在读取报告…', signal => getRegimeReliability(selected, signal))}>载入报告</Button><Button onClick={() => setRefresh(value => value + 1)}>刷新报告目录</Button></div>
    </details>
  </Card>
}
