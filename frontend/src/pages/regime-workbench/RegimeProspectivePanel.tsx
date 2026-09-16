import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Badge, Button, SectionHeader } from '../../components/ui'
import { definitionForRequest, listHistoricalReferences, type HistoricalReference, type RegimeGraphDefinition, type SavedRegimeReliability } from '../../services/regimeGraph'
import { assessRegimeProspective, captureRegimeProspective, getRegimeProspectiveProgress, getRegimeProspectiveQualification, listRegimeProspective, registerRegimeProspective, previewRegimeSourceVersion, confirmRegimeSourceVersion, type IndexSnapshotBindings, type ProspectiveSourcePreview, type ProspectiveAssessment, type ProspectiveProgress, type ProspectiveProtocol } from '../../services/regimeProspective'
import { referenceKey } from './RegimeReferenceBinding'
import { diagnosticPercent } from './RegimeDiagnosticsView'

const messages: Record<string, string> = {
  no_post_registration_day: '登记后尚无可记录日期。请在后续 UTC 日期有新观测后再记录。',
  no_post_registration_observation: '没有登记后的新观测。请更新同一数据来源后记录当前判断。',
  no_eligible_observation: '没有时点合格的新观测。请检查数据可得时间后重试。',
  observation_too_old: '最新观测已过期。请更新同一数据来源后再记录。',
  reference_label_already_published: '该日参考标签已发布，不能再补录预测。请等待尚未发布标签的新观测。',
  unknown_state_or_invalid_probability: '当前状态未知或概率无效，未写入记录。请检查模型输入后重试。',
  future_observation_window_incomplete: '固定前瞻窗口尚未完整。请继续实际记录新观测，待参考成熟后再检验。',
  reference_labels_not_mature: '参考标签尚未成熟。请选择同一定义修订的后续发布版本再检验。',
  no_forward_captures: '尚无实际前瞻记录。请先记录当前判断。',
  insufficient_paired_observations: '有效配对观测不足，未达到冻结门槛。',
  insufficient_reference_or_prediction_class_support: '参考或预测的逐类样本不足，未达到冻结门槛。',
  insufficient_complete_class_regimes: '各类完整状态区间不足，未达到冻结门槛。',
  insufficient_capture_coverage: '实际捕获覆盖不足，未达到冻结门槛。',
  insufficient_complete_time_blocks: '完整时间块不足，未达到冻结门槛。',
  complete_blocks_do_not_all_improve_class_base: '部分完整块未达到相对类别基准的改善门槛。',
  reference_agreement_below_policy: '相对参考的一致率未达到冻结门槛。',
  forward_evidence_expired: '前瞻证据已过期，不能采用。请重新研究并保存新的候选。',
  no_state_has_sufficient_forward_evidence: '各状态的独立前瞻证据仍不足，暂不能取得状态资格。',
  no_state_qualified: '已有充分证据的状态未达到验证门槛，请重新研究新的候选。',
  protocol_deadline: '登记协议已到期。请重新研究并保存新的候选。',
  fixed_window_assessed: '固定窗口已完成终局检验。请刷新进度查看结果。',
  CANDIDATE_SOURCE_CHANGED: '候选保存后输入已变化。请重新验证并保存校准候选。',
  captured: '已记录当前判断；稍后可刷新进度查看。',
  duplicate: '当前观测已记录，无需重复写入。请等待新观测。',
  pending: '证据尚未齐备。请记录新观测，并等待同一定义的后续参考发布。',
  stale: '观测已过期。请更新同一数据来源后重试。',
  unavailable: '当前证据不可用。请检查数据与模型后重试。',
}
export function prospectiveMessage(value: unknown): string {
  if (typeof value === 'string' && messages[value]) return messages[value]
  const sourceErrors = new Set(['当前快照没有新增的已知观测，无需续接；不要覆盖旧文件。', '历史输入发生修订；该协议不得继续累积证据。', '冻结输入前缀已缺失。', '续接预览已过期或不属于当前协议，请重新检查。', '数据、参考或实际捕获已变化，请重新检查再确认。', '历史参考算法已修改，请恢复原定义或重新研究候选，不能覆盖现有编辑。', '数据续接目前仅支持单一指数来源；上传、多源及宏观定义不能自动续接。', '前瞻协议已过期，不能续接数据。', '固定窗口已检验失败，不能通过续接数据重考。'])
  if (typeof value === 'string' && sourceErrors.has(value)) return value
  return '前瞻请求未完成。请检查当前模型、数据与参考后重试；必要时重新载入报告。'
}
export function prospectiveReferences(items: HistoricalReference[], protocol: ProspectiveProtocol, approved?: ProspectiveProtocol['reference_definition'][]) {
  const original = items.find(item => referenceKey(item) === referenceKey(protocol.reference))
  const definitions = approved || [protocol.reference_definition]
  return items.filter(item => definitions.some(d => item.definition_id === d?.definition_id && item.definition_revision === d?.revision)
    && referenceKey(item) !== referenceKey(protocol.reference)
    && Date.parse(item.created_at) > Date.parse(protocol.recorded_at)
    && (!original || Date.parse(item.created_at) > Date.parse(original.created_at)))
}
interface Props {
  definition: RegimeGraphDefinition; saved: SavedRegimeReliability | null; contextKey: string
  active?: boolean; dirty: boolean; valid: boolean
  onApplyQualification?: (calibrationId: string, qualificationId: string, sources?: IndexSnapshotBindings) => void
}
export default function RegimeProspectivePanel(props: Props) {
  const { definition, saved, contextKey, active = true, dirty, valid } = props
  const key = JSON.stringify([definitionForRequest(definition), saved, contextKey, active, dirty, valid])
  const blocked = !saved?.id || !saved.immutable || !saved.calibration_id ? '请先保存拟合完成的验证报告，再登记前瞻验证。'
    : !saved.report.calibration?.fitted ? '校准尚未拟合完成，请补充有效样本后重新验证并保存报告。'
      : dirty || !definition.id || !definition.revision ? '模型已变更，请先保存并重新验证。'
        : !active || !valid ? '当前模型或时点条件不可用，请先修复后重试。'
          : saved.request.definition_id !== definition.id || saved.request.revision !== definition.revision || referenceKey(saved.request.reference) !== referenceKey(definition.study?.reference) ? '报告与当前模型或参考不匹配，请重新载入报告。' : ''
  return <section aria-label="前瞻验证" className="min-w-0 space-y-3 border-t border-slate-200 pt-4 text-sm tabular-nums">
    <SectionHeader title="前瞻验证" description="冻结当前校准候选，用登记后的实际判断与后续历史参考检验。" />
    <p className="text-slate-600">默认日频窗口为 252 个位置（至少 120 个配对），周频 120 个，月频 60 个；按输出频率设置间隔与期限，覆盖须达 95%，并检查逐类区间和完整时间块。月频至少需要 60 个月的前向观测及后续成熟标签。只有登记后的实际捕获计入证据；不会自动监测或补录历史。</p>
    {blocked ? <><p className="text-slate-600">{blocked}</p><Button disabled>登记前瞻验证</Button></> : <ProspectiveSession key={key} {...props} saved={saved!} />}
  </section>
}
function ProspectiveSession({ definition, saved, onApplyQualification }: Props & { saved: SavedRegimeReliability }) {
  const [progress, setProgress] = useState<ProspectiveProgress | null>(null)
  const [references, setReferences] = useState<HistoricalReference[]>([])
  const [selected, setSelected] = useState('')
  const [busy, setBusy] = useState('正在读取前瞻进度…')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [loaded, setLoaded] = useState(false)
  const [adopted, setAdopted] = useState(false)
  const [sourcePreview, setSourcePreview] = useState<ProspectiveSourcePreview | null>(null)
  const alive = useRef(true)
  const operation = useRef<AbortController | null>(null)
  useLayoutEffect(() => { alive.current = true; return () => { alive.current = false; operation.current?.abort(); operation.current = null } }, [])
  const matches = (protocol: ProspectiveProtocol) => typeof protocol?.id === 'string' && protocol.id && typeof protocol.model_binding_hash === 'string' && protocol.model_binding_hash && protocol.calibration_id === saved.calibration_id && protocol.definition_id === definition.id && protocol.revision === definition.revision && referenceKey(protocol.reference) === referenceKey(saved.request.reference)
  const validateProgress = (value: ProspectiveProgress) => {
    if (!value || !matches(value.protocol)) throw new Error('mismatch')
    const assessment = value.latest_assessment
    if (assessment && (!assessment.id || !['pending', 'rejected', 'qualified'].includes(assessment.status) || assessment.protocol_id !== value.protocol.id || assessment.calibration_id !== saved.calibration_id || assessment.model_binding_hash !== value.protocol.model_binding_hash)) throw new Error('mismatch')
    const source = value.current_source_version
    if (source && (source.protocol_id !== value.protocol.id || source.kind !== 'source_version' || source.status !== 'accepted' || !source.prefix_unchanged || source.reference_definition.definition_id !== value.protocol.reference_definition.definition_id)) throw new Error('mismatch')
    return value
  }
  const execute = async (label: string, action: (signal: AbortSignal, current: () => boolean) => Promise<void>) => {
    if (!alive.current || operation.current) return
    const controller = new AbortController(); operation.current = controller
    const current = () => alive.current && !controller.signal.aborted && operation.current === controller
    setBusy(label); setError('')
    try { await action(controller.signal, current) }
    catch (reason) { if (current()) setError(prospectiveMessage(reason instanceof Error ? reason.message : reason)) }
    finally { if (current()) { operation.current = null; setBusy('') } }
  }
  const reload = () => void execute('正在读取前瞻进度…', async (signal, current) => {
    const [catalog, items] = await Promise.all([listRegimeProspective(signal), listHistoricalReferences(signal)])
    if (!current()) return
    const protocol = catalog.find(matches)
    const value = protocol ? validateProgress(await getRegimeProspectiveProgress(protocol.id, signal)) : null
    if (!current()) return
    setProgress(value); setReferences(items || []); setLoaded(true); setSourcePreview(null)
  })
  useEffect(() => { reload() }, []) // Session is remounted for every model/report/PIT identity.
  const protocol = progress?.protocol
  const assessment = progress?.latest_assessment
  const options = protocol ? prospectiveReferences(references, protocol, progress?.reference_definitions) : []
  const reference = options.find(item => referenceKey(item) === selected)
  const expired = Boolean(assessment?.expires_at && Date.parse(assessment.expires_at) <= Date.now())
  const terminal = assessment?.status === 'qualified' || assessment?.status === 'rejected'
  const readProgress = async (id: string, signal: AbortSignal, current: () => boolean) => {
    const value = validateProgress(await getRegimeProspectiveProgress(id, signal))
    if (current()) setProgress(value)
  }
  const qualified = assessment?.status === 'qualified' && !expired
  const verifyAdoption = (value: ProspectiveAssessment) => {
    if (!value || !protocol || !assessment || value.status !== 'qualified' || value.id !== assessment.id || value.protocol_id !== protocol.id || value.calibration_id !== saved.calibration_id || value.model_binding_hash !== protocol.model_binding_hash || !value.expires_at || !(Date.parse(value.expires_at) > Date.now())) throw new Error('forward_evidence_expired')
  }
  return <div className="min-w-0 space-y-3">
    <div className="flex flex-wrap gap-2">
      {!protocol && <Button tone="primary" disabled={Boolean(busy) || !loaded} onClick={() => void execute('正在登记前瞻验证…', async (signal, current) => {
        const value = await registerRegimeProspective({ calibration_id: saved.calibration_id! }, signal)
        if (!current()) return
        if (!matches(value)) throw new Error('mismatch')
        setProgress({ protocol: value, observations: null, last_observation_date: null, latest_assessment: null })
        setNotice('已冻结登记。请在后续有新观测时记录当前判断。')
        await readProgress(value.id, signal, current)
      })}>登记前瞻验证</Button>}
      <Button disabled={Boolean(busy)} onClick={reload}>刷新前瞻进度</Button>
      {protocol && <Button disabled={Boolean(busy) || terminal} onClick={() => void execute('正在记录当前判断…', async (signal, current) => {
        setSourcePreview(null)
        const value = await captureRegimeProspective(protocol.id, {}, signal)
        if (!current()) return
        setNotice(prospectiveMessage(value?.reason || value?.status))
        await readProgress(protocol.id, signal, current)
      })}>记录当前判断</Button>}
    </div>
    {busy && <div role="status" className="space-y-2 text-slate-600"><div className="h-4 rounded bg-slate-100" /><div className="h-4 w-2/3 rounded bg-slate-100" />{busy}</div>}
    {error && <p role="alert" className="text-rose-800">{error}</p>}
    {notice && <p role="status" className="text-slate-700">{notice}</p>}
    {loaded && !protocol && <p className="text-slate-600">该报告尚未登记。登记将冻结模型、校准、参考与默认门槛。</p>}
    {protocol && <>
      <div className="flex flex-wrap gap-2"><Badge tone={qualified ? 'success' : assessment?.status === 'rejected' && assessment?.outcome !== 'insufficient_evidence' || expired ? 'danger' : 'warning'}>{expired ? '资格已过期' : qualified ? assessment?.outcome === 'partially_qualified' ? '部分状态前瞻通过' : '前瞻检验通过' : assessment?.status === 'rejected' ? assessment?.outcome === 'insufficient_evidence' ? '前瞻证据不足' : '前瞻检验未通过' : '前瞻验证待完成'}</Badge></div>
      <dl className="grid min-w-0 gap-3 break-words sm:grid-cols-3 text-slate-700"><div><dt>冻结登记时间（UTC）</dt><dd>{protocol.recorded_at || '未提供'}</dd></div><div><dt>实际捕获数</dt><dd>{progress.observations ?? '未提供'}</dd></div><div><dt>最近观测日</dt><dd>{progress.last_observation_date || '尚无记录'}</dd></div></dl>
      <p className="text-slate-600">本协议窗口：{protocol.policy?.observation_window ?? '未提供'} 个参考观测位置。记录操作由服务器选择不晚于 UTC 昨天的最新合格观测，且必须严格晚于登记日。</p>
      {protocol.forward_source_mode === 'immutable_source_pending_new_observations' && <p className="text-amber-900">模型使用固定快照。单一指数来源可在下方检查并确认新版本；旧文件不能覆盖，上传或多源定义暂不支持续接。</p>}
      {assessment?.status !== 'rejected' && <div aria-label="前瞻数据版本续接" className="min-w-0 space-y-2 border-l-2 border-slate-300 pl-3">
        <p className="text-slate-700">接入新数据，保持算法和历史证据不变。</p>
        <Button disabled={Boolean(busy) || expired} onClick={() => void execute('正在核对新数据与历史前缀…', async (signal, current) => {
          setSourcePreview(null)
          const value = await previewRegimeSourceVersion(protocol.id, signal)
          if (!current()) return
          if (value.protocol_id !== protocol.id || !value.prefix_unchanged || !value.preview_hash) throw new Error('mismatch')
          setSourcePreview(value)
        })}>检查新数据版本</Button>
        {sourcePreview && <>
          <p className="text-slate-700">历史前缀一致：{sourcePreview.old_observations} → {sourcePreview.new_observations} 个观测，新增 {sourcePreview.added_observations} 个。</p>
          <p className="break-all text-xs text-slate-600">新快照：{Object.values(sourcePreview.model_bindings).map(binding => binding.snapshot_id).join('、')}</p>
          <p className="text-slate-600">确认后将记录数据版本并创建对应历史参考修订；不会发布标签或补录预测。</p>
          <Button tone="primary" disabled={Boolean(busy)} onClick={() => void execute('正在确认数据续接…', async (signal, current) => {
            const value = await confirmRegimeSourceVersion(protocol.id, sourcePreview.preview_hash, signal)
            if (!current()) return
            if (value.protocol_id !== protocol.id || value.status !== 'accepted') throw new Error('mismatch')
            setSourcePreview(null); setNotice('已续接新数据。可记录当前判断；后续参考仍需单独生成并确认发布。')
            await readProgress(protocol.id, signal, current)
          })}>确认续接数据</Button>
        </>}
        {progress.current_source_version && <>
          <p className="text-xs text-slate-600">已确认数据版本：{Object.values(progress.current_source_version.model_bindings).map(binding => binding.snapshot_id).join('、')}</p>
          <Link className="inline-flex min-h-10 items-center text-accent-700 underline" target="_blank" rel="noopener noreferrer" to={`/settings/scenario-algorithms?center=historical&definition=${encodeURIComponent(progress.current_source_version.reference_definition.definition_id)}&revision=${progress.current_source_version.reference_definition.revision}`}>打开后续参考修订（新页面）</Link>
        </>}
      </div>}
      {progress.observations === 0 && <p className="text-slate-600">没有登记后的实际记录。请等待新观测后点击“记录当前判断”。</p>}
      {!terminal && <><label className="block min-w-0 text-slate-700">用于检验的后续历史参考<select aria-label="前瞻检验参考" disabled={Boolean(busy)} className="mt-1 block min-h-10 w-full min-w-0 max-w-full rounded-lg border border-slate-300 bg-white px-3 text-sm" value={reference ? selected : ''} onChange={event => { setSelected(event.target.value); setNotice('') }}><option value="">选择原定义或已确认数据修订的后续发布</option>{options.map(item => <option key={referenceKey(item)} value={referenceKey(item)}>{item.name} · v{item.definition_revision} · {item.created_at}</option>)}</select></label>
        {!options.length && <p className="text-slate-600">尚无允许使用的后续参考。请生成并发布原定义或已确认数据修订的参考，再刷新进度；不能更改算法重考。</p>}
        <Button disabled={Boolean(busy) || !reference} onClick={() => void execute('正在检验前瞻结果…', async (signal, current) => {
          if (!reference) return
          const value = await assessRegimeProspective(protocol.id, { reference: { run_id: reference.run_id, publication_id: reference.publication_id, content_hash: reference.content_hash } }, signal)
          if (!current()) return
          setProgress(validateProgress({ ...progress, latest_assessment: value }))
          await readProgress(protocol.id, signal, current)
        })}>检验前瞻结果</Button>
        {!reference && options.length > 0 && <p className="text-slate-600">请选择后续参考再检验；服务端会核验标签成熟时间与完整来源。</p>}
      </>}
      {assessment && <div className="space-y-2">
        <p className="text-slate-700">有效配对覆盖：{diagnosticPercent(assessment.metrics?.coverage)} · 参考一致率：{diagnosticPercent(assessment.metrics?.classification?.accuracy)}</p>
        {assessment.state_evidence?.length ? <div className="overflow-x-auto"><table aria-label="前瞻逐状态资格" className="w-full text-xs"><thead><tr>{['状态', '完整区间', '高置信预测', 'Precision', 'Recall', '结论'].map(title => <th key={title} scope="col" className="p-2 text-left">{title}</th>)}</tr></thead><tbody>{assessment.state_evidence.map(row => <tr key={row.state_id} className="border-b border-slate-100"><th scope="row" className="p-2 text-left">{definition.states.find(state => state.id === row.state_id)?.label || row.state_id}</th><td className="p-2 text-right">{row.complete_regimes}</td><td className="p-2 text-right">{row.accepted_predictions}</td><td className="p-2 text-right">{diagnosticPercent(row.precision)}</td><td className="p-2 text-right">{diagnosticPercent(row.recall)}</td><td className="p-2">{row.status === 'qualified' ? '已通过' : row.status === 'insufficient_evidence' ? '证据不足' : '未通过'}</td></tr>)}</tbody></table></div> : null}
        {assessment.qualified_states?.length ? <p className="text-slate-600">已取得资格：{assessment.qualified_states.map(id => definition.states.find(state => state.id === id)?.label || id).join('、')}。{assessment.fallback_states?.length ? '其余状态不能授权 Regime 信号，回退原有非 Regime 决策逻辑。' : ''}</p> : null}
        <div className="grid gap-2 sm:grid-cols-2">{(assessment.reasons || []).map((reason, index) => <p key={index} className="text-slate-600">{prospectiveMessage(reason)}</p>)}</div>
        {terminal && <p className="text-slate-600">固定窗口已关闭，不能更换参考或门槛重考。{qualified ? '可显式采用已通过状态的资格，并保存模型新修订；未通过状态继续回退。' : assessment.outcome === 'insufficient_evidence' ? '当前只代表证据不足，不等同于模型失败。' : '请依据证据重新研究新的候选。'}</p>}
        <p className="break-words text-xs text-slate-600">最早可用日：{assessment.available_from_date || '未提供'} · 到期：{assessment.expires_at || '未提供'}。采用后仍由服务端核验模型、时点和资格。</p>
        {qualified && <Button tone="primary" disabled={Boolean(busy) || !onApplyQualification || adopted} onClick={() => void execute('正在核验校准资格…', async (signal, current) => {
          const value = await getRegimeProspectiveQualification(assessment.id, signal)
          if (!current()) return
          verifyAdoption(value)
          setAdopted(true)
          if (progress.current_source_version) onApplyQualification?.(saved.calibration_id!, value.id, progress.current_source_version.model_bindings)
          else onApplyQualification?.(saved.calibration_id!, value.id)
        })}>{adopted ? '已采用，请保存模型' : '采用已验证校准'}</Button>}
      </div>}
      <details><summary className="min-h-10 cursor-pointer text-slate-600">前瞻证据与冻结门槛</summary><pre className="whitespace-pre-wrap break-all text-xs text-slate-600">{JSON.stringify({ protocol_id: protocol.id, calibration_id: protocol.calibration_id, model_binding_hash: protocol.model_binding_hash, policy: protocol.policy, assessment_id: assessment?.id, reference: assessment?.reference, metrics: assessment?.metrics }, null, 2)}</pre></details>
    </>}
    <p className="text-xs text-slate-600">检验衡量相对指定历史参考的表现，不证明真实市场准确率，也不声明统计显著性或自动取得交易权限。</p>
  </div>
}
