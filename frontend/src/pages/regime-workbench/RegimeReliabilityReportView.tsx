import { RegimeConfidenceView, RegimeStabilityView, diagnosticStatus } from './RegimeDiagnosticsView'
import { useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { Badge } from '../../components/ui'
import type { RegimeReliabilityReport } from '../../services/regimeGraph'

const percent = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? '无法估计' : `${(value * 100).toFixed(1)}%`
const number = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? '无法估计' : value.toFixed(3)
const BLOCKS: Record<string, string> = { calibration: '校准段', validation: '验证段', test: '最终测试段', holdout: '独立留出段' }
const REASONS: Record<string, string> = {
  reference_labels_unavailable_at_calibration_end: '历史参考在校准截止时尚不可得，仅能做回顾评分。',
  insufficient_samples_classes_or_complete_segments: '样本、类别或完整参考区间不足。',
  independent_holdout_does_not_improve_class_base_brier: '独立测试未优于历史类别基准。',
  model_and_reference_selection_history_unverified: '尚未验证模型与参考的选择历史，不能用于部署。',
  insufficient_calibration_evidence: '校准样本不足，不能估计匹配概率。',
  insufficient_usable_calibration_predictions: '可用于校准的有效预测或类别样本不足；参考标签数量不能代替有效预测数量。',
  'Reference agreement is not ground truth.': '一致程度仅相对所选历史参考。',
  'Dependent daily observations are not independent regimes.': '每日观测相互依赖，样本天数不等于独立周期数。',
  'Retrospective calibration is diagnostic, never a historically deployable probability.': '回顾性校准仅用于诊断，不能倒填为历史可用信号。',
}
export const reliabilityReason = (reason: string) => REASONS[reason] || (/[\u3400-\u9fff]/.test(reason) ? reason : '服务端未提供可解释的诊断说明，请检查报告来源或重新验证。')
const EVIDENCE: Record<string, string> = { deterministic_state: '确定性状态编码，未校准', deterministic: '规则判定，未校准', deterministic_one_hot: '规则判定，未校准', deterministic_onehot: '规则判定，未校准', model_posterior: '模型内部概率，未校准', posterior: '模型内部概率，未校准', ensemble_vote: '模型一致票占比，未校准', vote_strength: '模型一致票占比，未校准' }
const VERIFY_LABEL: Record<string, string> = { verified: '已验证', partially_verified: '部分验证通过', insufficient_evidence: '证据不足', failed: '验证未通过' }
const VERIFY_TONE = (status: string): 'success' | 'warning' | 'danger' => status === 'verified' ? 'success' : status === 'failed' ? 'danger' : 'warning'
const recognitionReady = (verification: NonNullable<RegimeReliabilityReport['verification']>) => verification.recognition_ready ?? verification.cma_research_ready ?? false
export default function RegimeReliabilityReportView({ report, savedAt }: { report: RegimeReliabilityReport; savedAt?: string }) {
  const [block, setBlock] = useState('test')
  const labels = Object.fromEntries(report.states.map(state => [state.id, state.label]))
  const label = (id: string | null) => id === null ? '参考未分类' : id === 'abstained' ? '预测拒识' : labels[id] || id
  const blockIds = Object.keys(report.probability.blocks)
  const activeBlock = blockIds.includes(block) ? block : blockIds.includes('holdout') ? 'holdout' : blockIds[0]
  const metrics = report.probability.blocks[activeBlock]
  const last = report.points[report.points.length - 1]
  const unavailable = report.status !== 'eligible' || !report.calibration?.deployment_eligible
  const sufficient = Object.values(report.sample.blocks).every(item => item.sufficient)
  return <section aria-label="识别能力验证报告" className="min-w-0 space-y-4 border-t border-slate-200 pt-4 text-sm tabular-nums">
    <div className="flex flex-wrap gap-2"><Badge tone={unavailable ? 'warning' : 'success'}>{report.status === 'retrospective_only' ? '仅回顾性评分' : report.status === 'insufficient_evidence' ? '证据不足' : '验证报告'}</Badge>{!sufficient && <Badge tone="warning">样本不足</Badge>}{unavailable && <Badge tone="warning">不可部署</Badge>}{savedAt && <span className="text-slate-600">保存于 {savedAt}</span>}</div>
    <p className="text-slate-600">{EVIDENCE[report.probability.raw_type] || '证据类型未确认，不能解释为概率'}。原始数值为 1 不代表 100% 可信。</p>
    {report.verification && <section aria-label="状态级验证" className="space-y-2 rounded-xl border border-slate-200 p-3">
      <div className="flex flex-wrap gap-2"><Badge tone={VERIFY_TONE(report.verification.status)}>{VERIFY_LABEL[report.verification.status] || report.verification.status}</Badge><Badge tone={recognitionReady(report.verification) ? 'success' : 'warning'}>{recognitionReady(report.verification) ? '识别验证可用' : '识别验证暂不可用'}</Badge></div>
      <p className="text-slate-600">逐状态使用独立完整区间与高置信预测验证。样本不足是证据不足，不是算法失败；未验证状态不能授权下游实时 Regime 决策。实时识别证据不作为 LTCMA 输入。</p>
      <div className="overflow-x-auto"><table aria-label="逐状态验证结果" className="w-full text-xs"><thead><tr>{['状态','独立完整区间','高置信预测','Precision','Recall','结论'].map(title => <th scope="col" className="p-2 text-left" key={title}>{title}</th>)}</tr></thead><tbody>{report.verification.states.map(row => <tr className="border-b border-slate-100" key={row.state_id}><th scope="row" className="p-2 text-left">{label(row.state_id)}</th><td className="p-2 text-right">{row.independent_complete_episodes}</td><td className="p-2 text-right">{row.accepted_predictions}</td><td className="p-2 text-right">{percent(row.precision)}</td><td className="p-2 text-right">{percent(row.recall)}</td><td className="p-2"><Badge tone={VERIFY_TONE(row.status)}>{VERIFY_LABEL[row.status] || row.status}</Badge></td></tr>)}</tbody></table></div>
      <p className="text-xs text-slate-600">已验证状态：{report.verification.verified_states.map(label).join('、') || '无'}。回退状态：{report.verification.fallback_states.map(label).join('、') || '无'}。</p>
    </section>}
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      <div><p className="text-xs text-slate-600">参考一致率（全样本，含拒识）</p><p className="mt-1 font-semibold">{percent(report.classification.accuracy)}</p></div>
      <div><p className="text-xs text-slate-600">区间等权重合 IoU</p><p className="mt-1 font-semibold">{percent(report.intervals.equal_reference_segment_iou)}</p></div>
      <div><p className="text-xs text-slate-600">转折延迟中位数（观测间隔）</p><p className="mt-1 font-semibold">{report.transitions.delay_median ?? '无法估计'}</p></div>
      <div><p className="text-xs text-slate-600">历史留出指标区间</p><p className="mt-1 font-semibold">{diagnosticStatus(report.confidence_interval?.status)}</p></div>
    </div>
    <p className="text-slate-700">输入 {report.sample.input} · 日期匹配 {report.sample.matched} · 参考未分类 {report.sample.unknown_reference} · 预测拒识 {report.sample.prediction_abstentions} · 缺失预测 {report.sample.missing_prediction} · 无效状态 {report.sample.invalid_prediction_labels}</p>
    <p className="text-slate-600">排除：截止日后参考 {report.sample.excluded_dates.reference_after_cutoff}、无对应参考的预测 {report.sample.excluded_dates.prediction_without_reference}。完整参考区间 {report.intervals.complete_reference_segments} / 全部 {report.intervals.reference_segments}。</p>
    <p className="text-slate-600">转折：匹配 {report.transitions.matches} · 漏检 {report.transitions.misses} · 误报 {report.transitions.false_events} · P90 延迟 {report.transitions.delay_p90 ?? '无法估计'} 个观测间隔。</p>
    {report.selective_classification && <p className="text-slate-600">应用校准门槛后：接受覆盖 {percent(report.selective_classification.accepted_coverage)} · 接受后错误率 {percent(report.selective_classification.accepted_error)}。原始全样本指标保留在上方。</p>}
    {last && <div className="space-y-1 border-l-2 border-slate-300 pl-3">
      <p>{last.observation_date} · 识别 {last.predicted_state === null ? '拒识' : label(last.predicted_state)} · 参考 {label(last.reference_state)}</p>
      {last.reference_state === null && <p className="text-xs text-slate-600">参考在此日未分类，无法判断当天是否匹配。</p>}
      <p>校准后的匹配概率（相对所选历史参考）：<strong>{percent(last.calibrated_confidence)}</strong></p>
      {last.probability_evidence && <p className="text-xs text-slate-600">模型对所选状态的原始概率 {percent(last.probability_evidence.selected_probability)} · 前两名间距 {percent(last.probability_evidence.margin)} · 概率熵 {number(last.probability_evidence.entropy)}（自然对数单位）。这是模型内部证据，不是校准后的匹配概率。</p>}
      <p className="text-xs text-slate-600">数据可得日 {last.data_available_at || '未提供'} · 识别确认日 {last.recognized_at || '未提供'}</p>
      <p className="text-xs text-slate-600">{report.calibration?.evidence_type === 'class_average' ? '来源：同类历史平均，不是当天独立概率。' : report.calibration?.evidence_type === 'model_reference_probability' ? '来源：模型概率经参考校准。' : '未提供校准来源。'}{last.decision_status === 'below_floor' ? '低于接受门槛，当前判断不接受。' : last.decision_status === 'abstained' ? '当前拒识，不等同于震荡。' : last.decision_status === 'uncalibrated' ? '尚无可用校准结果。' : '仅供研究诊断。'}</p>
    </div>}
    <RegimeStabilityView result={report.stability?.parameter_sensitivity} />
    <RegimeConfidenceView result={report.confidence_interval} />
    {report.calibration?.reasons.length ? <div className="space-y-1 text-amber-900">{report.calibration.reasons.map(reason => <p key={reason}>{reliabilityReason(reason)}</p>)}</div> : null}
    <details><summary className="min-h-10 cursor-pointer text-slate-600">样本分段、状态表现与混淆矩阵</summary>
      <div className="overflow-x-auto"><table className="w-full text-xs" aria-label="验证样本分段"><thead><tr>{['时间段', '范围', '样本', '各状态样本', '完整区间', '样本状态'].map(item => <th scope="col" className="p-2 text-left" key={item}>{item}</th>)}</tr></thead><tbody>{Object.entries(report.sample.blocks).map(([id, item]) => <tr key={id} className="border-b border-slate-100"><th scope="row" className="p-2 text-left">{BLOCKS[id] || id}</th><td className="p-2">{item.start || '无样本'} 至 {item.end || '无样本'}</td><td className="p-2 text-right">{item.samples}</td><td className="p-2">{Object.entries(item.per_class).map(([id, count]) => label(id) + ' ' + count).join('、')}</td><td className="p-2 text-right">{item.complete_segments}</td><td className="p-2">{item.sufficient ? '充足' : '不足'}</td></tr>)}</tbody></table></div>
      <div className="overflow-x-auto"><table className="w-full text-xs" aria-label="各状态表现"><thead><tr>{['状态', '参考样本', '精确率', '召回率', 'F1', '逐日 IoU'].map(item => <th scope="col" className="p-2 text-left" key={item}>{item}</th>)}</tr></thead><tbody>{report.classification.per_state.map(item => <tr key={item.state_id} className="border-b border-slate-100"><th scope="row" className="p-2 text-left">{label(item.state_id)}</th><td className="p-2 text-right">{item.support}</td>{[item.precision, item.recall, item.f1, item.iou].map((value, i) => <td key={i} className="p-2 text-right">{percent(value)}</td>)}</tr>)}</tbody></table></div>
      <div className="overflow-x-auto"><table className="w-full text-xs" aria-label="混淆矩阵：参考行与预测列"><thead><tr><th scope="col" className="p-2 text-left">参考 / 预测</th>{report.classification.columns.map(id => <th scope="col" key={id} className="p-2 text-right">{label(id)}</th>)}</tr></thead><tbody>{report.classification.rows.map((id, i) => <tr key={id} className="border-b border-slate-100"><th scope="row" className="p-2 text-left">{label(id)}</th>{report.classification.confusion[i].map((value, j) => <td className="p-2 text-right" key={j}>{value}</td>)}</tr>)}</tbody></table></div>
      <p className="mt-2 text-xs text-slate-600">平衡准确率 {percent(report.classification.balanced_accuracy)} · Macro F1 {percent(report.classification.macro_f1)} · 接受覆盖 {percent(report.classification.accepted_coverage)} · 接受后错误率 {percent(report.classification.accepted_error)}</p>
    </details>
    <details><summary className="min-h-10 cursor-pointer text-slate-600">概率评分与可靠性图</summary>
      <label className="block text-slate-700">评分时间段<select aria-label="概率评分时间段" className="ml-2 min-h-10 max-w-full rounded-lg border border-slate-300 bg-white px-2" value={activeBlock || ''} onChange={event => setBlock(event.target.value)}>{blockIds.map(id => <option key={id} value={id}>{BLOCKS[id] || id}</option>)}</select></label>
      {metrics && <><div className="overflow-x-auto"><table className="mt-3 w-full text-xs" aria-label="概率评分"><thead><tr>{['来源', '有效样本', 'Brier', 'LogLoss', 'ECE'].map(item => <th scope="col" key={item} className="p-2 text-left">{item}</th>)}</tr></thead><tbody>{([['raw', '原始证据'], ['calibrated', '校准结果'], ['class_base', '历史类别基准']] as const).map(([key, label]) => <tr key={key} className="border-b border-slate-100"><th scope="row" className="p-2 text-left">{label}</th><td className="p-2 text-right">{metrics[key].samples}</td>{[metrics[key].brier, metrics[key].logloss, metrics[key].ece].map((value, i) => <td className="p-2 text-right" key={i}>{number(value)}</td>)}</tr>)}</tbody></table></div>
        {metrics.calibrated.bins.some(bin => bin.samples > 0) ? <ReactECharts style={{ height: 280, width: '100%' }} option={{ aria: { enabled: true, description: '校准结果的可靠性图，横轴为预测匹配概率，纵轴为实际参考匹配率。详细数据见下表。' }, tooltip: { trigger: 'item' }, grid: { left: 50, right: 20, bottom: 55, top: 20 }, xAxis: { type: 'value', min: 0, max: 1, name: '预测匹配概率', nameLocation: 'middle', nameGap: 30, axisLabel: { fontSize: 12 } }, yAxis: { type: 'value', min: 0, max: 1, axisLabel: { fontSize: 12 } }, series: [{ type: 'scatter', name: '实际参考匹配率', data: metrics.calibrated.bins.filter(bin => bin.samples > 0).map(bin => [bin.mean_confidence, bin.match_rate, bin.samples]) }] }} /> : <p className="mt-3 text-slate-600">没有可用的校准概率样本，无法绘制可靠性图。</p>}
        <div className="overflow-x-auto"><table className="w-full text-xs" aria-label="校准可靠性分箱"><thead><tr>{['分箱', '预测匹配概率', '实际参考匹配率', '样本'].map(item => <th scope="col" className="p-2 text-left" key={item}>{item}</th>)}</tr></thead><tbody>{metrics.calibrated.bins.map(bin => <tr key={bin.index} className="border-b border-slate-100"><th scope="row" className="p-2 text-left">{bin.index + 1}</th><td className="p-2 text-right">{percent(bin.mean_confidence)}</td><td className="p-2 text-right">{percent(bin.match_rate)}</td><td className="p-2 text-right">{bin.samples}</td></tr>)}</tbody></table></div></>}
    </details>
    <details><summary className="min-h-10 cursor-pointer text-slate-600">报告来源与限制</summary>
      <p className="text-slate-600">稳定性：{report.stability?.status === 'causal_probes_executed' ? '已执行因果时点探测' : report.stability?.status === 'not_executed' ? '尚未执行新检验' : '未提供检验证据'}。</p>
      <p className="text-slate-600">参考标签可得日 {report.calibration?.label_known_at || '未知'} · 校准制品最早可得日 {report.calibration?.available_from || '未知'} · 到期 {report.calibration?.expires_on || '未知'}。</p>
      {report.warnings.map((warning, i) => <p key={i} className="mt-1 text-xs text-slate-600">{reliabilityReason(warning)}</p>)}
      <pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap break-all text-xs text-slate-600">{JSON.stringify({ ...report.lineage, stability: report.stability }, null, 2)}</pre>
    </details>
  </section>
}
