import type { RegimeConfidenceInterval, RegimeStabilityResult } from '../../services/regimeDiagnostics'

export const diagnosticPercent = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? '无法估计' : `${(value * 100).toFixed(1)}%`
export const diagnosticNumber = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? '无法估计' : value.toFixed(2)
const reasons: Record<string, string> = {
  insufficient_dependency_aware_evidence: '考虑时序相关性后，部分或全部指标证据不足；展开详情查看时间块、周期与有效重采样。',
  insufficient_full_blocks: '完整时间块不足；请扩大留出样本或调整块长度。',
  insufficient_blocks: '完整时间块不足；请扩大留出样本或调整块长度。',
  insufficient_complete_cycles: '完整状态周期不足；请扩大留出样本。',
  insufficient_cycles: '完整状态周期不足；请扩大留出样本。',
  insufficient_valid_replicates: '有效重采样次数不足，无法估计区间。',
  no_valid_probability_samples: '没有有效概率样本，无法估计该指标。',
  no_applicable_parameters: '当前模型没有可扰动的参数。',
  no_applicable_variants: '当前模型没有适用的稳定性变体。',
  no_stochastic_model: '规则模型没有随机种子，种子检查不适用。',
  no_price_source: '没有可解释的价格或净值输入，不计算区间收益。',
  no_verified_price_source: '没有经核验的价格或净值输入，不计算区间收益。',
  disabled: '本次策略未启用。',
  disabled_by_policy: '本次策略未启用。',
  no_verified_price_semantics: '没有经核验的价格或净值输入，不计算区间收益。',
  no_valid_price_segments: '区间价格端点不完整或无效，无法计算收益。',
  'Historical segmentation diagnostics are not ground truth.': '历史划分诊断不代表标签真值。',
  'This report does not publish a reference or establish historical label availability.': '本报告不发布参考，也不证明历史标签在当时可得。',
  no_valid_segment_price_returns: '区间价格端点不完整或无效，无法计算收益。',
  insufficient_prefix: '截尾后可比较的历史观测不足。',
  diagnostic_budget_exceeded: '模型规模超过检查预算，请缩小输入或变体范围。',
  diagnostic_time_budget_exceeded: '本次检查超过时间预算，可减少变体后重试。',
  invalid_variant: '变体无法执行，请检查参数与输入。',
  'Dependency-aware interval estimator not implemented': '这份旧报告未执行相关时序区间估计；重新验证可生成新报告。',
  not_applicable: '当前模型不适用此项检查。',
  not_implemented: '旧报告未执行此项计算；可重新检查生成新报告。',
  'not implemented': '旧报告未执行此项计算；可重新检查生成新报告。',
  'not run': '旧报告未运行此项检查。',
}
export const diagnosticReason = (value?: string | null) => !value ? '' : reasons[value] || (/[\u3400-\u9fff]/.test(value) ? value : '本次未获得可用证据；请检查输入与策略后重试。')
const statuses: Record<string, string> = { completed: '已完成', partial: '部分完成', disabled: '未启用', not_applicable: '不适用', failed: '执行失败', budget_exceeded: '超过计算预算', not_executed: '尚未执行', available: '已估计', unavailable: '无法估计' }
export const diagnosticStatus = (status?: string) => statuses[status || ''] || '未提供检验证据'
const kinds: Record<string, string> = { parameter: '参数扰动', window: '窗口变化', seed: '随机种子', truncation: '截尾修订' }
export function RegimeStabilityView({ result }: { result?: RegimeStabilityResult }) {
  return <section aria-label="划分稳定性" className="min-w-0 space-y-2 text-sm tabular-nums">
    <p className="font-semibold text-slate-800">稳定性：{diagnosticStatus(result?.status)}</p>
    {!result && <p className="text-slate-600">旧报告没有参数稳定性证据；点击检查稳定性可生成新报告。</p>}
    {result?.reason && <p className="text-slate-600">{diagnosticReason(result.reason)}</p>}
    {result?.seed_status === 'not_applicable' && <p className="text-slate-600">随机种子：不适用，当前定义没有可变的随机拟合。</p>}
    <details><summary className="min-h-10 cursor-pointer text-slate-600">参数、窗口与边界变化详情</summary>
      <p className="text-xs text-slate-600">一致率仅比较同日已分类观测，不是准确率。边界移动按观测间隔计；没有可比较边界时显示无法估计。</p>
      {!result?.variants?.length ? <p className="py-2 text-slate-600">没有已执行变体。启用适用的检查后再运行。</p> : <div className="overflow-x-auto"><table className="w-full text-xs" aria-label="稳定性变体"><thead><tr>{['检查', '状态', '一致率', '分类覆盖', '可比较观测', '边界移动（观测间隔）', '变更与原因'].map(title => <th key={title} scope="col" className="p-2 text-left">{title}</th>)}</tr></thead><tbody>{result.variants.map((variant, index) => <tr key={index} className="border-b border-slate-100"><th scope="row" className="p-2 text-left">{kinds[variant.kind] || '模型变体'}</th><td className="p-2">{diagnosticStatus(variant.status)}</td><td className="p-2 text-right">{diagnosticPercent(variant.agreement)}</td><td className="p-2 text-right">{diagnosticPercent(variant.classification_coverage)}</td><td className="p-2 text-right">{variant.comparable_observations ?? '无法估计'}</td><td className="p-2 text-right">{diagnosticNumber(variant.boundary_distance)}</td><td className="max-w-xs break-words p-2">{variant.changes.map((change, i) => <p key={i}>{change.node_id} · {change.parameter}：{JSON.stringify(change.before)} → {JSON.stringify(change.after)}</p>)}{diagnosticReason(variant.reason)}</td></tr>)}</tbody></table></div>}
    </details>
  </section>
}
const metricNames: Record<string, string> = { accuracy: '参考一致率', accepted_coverage: '接受覆盖', accepted_error: '接受后错误率', brier: 'Brier 分数', paired_brier_improvement: '相对类别基准的 Brier 改善' }
export function RegimeConfidenceView({ result }: { result?: RegimeConfidenceInterval }) {
  return <section aria-label="历史指标置信区间" className="min-w-0 space-y-2 text-sm tabular-nums">
    <p className="font-semibold text-slate-800">历史留出指标区间：{diagnosticStatus(result?.status)}</p>
    <p className="text-slate-600">区间衡量历史总体指标的不确定性，不是今天状态的概率范围；条件是固定模型、历史参考与已拟合校准器。</p>
    {result?.reason && <p className="text-slate-600">{diagnosticReason(result.reason)}</p>}
    {!result && <p className="text-slate-600">旧报告未记录区间估计；重新验证可生成新报告。</p>}
    <details><summary className="min-h-10 cursor-pointer text-slate-600">置信区间与重采样详情</summary>
      {result?.method && <p className="text-xs text-slate-600">配对移动块重采样 · {result.scope === 'test' ? '最终测试段' : '独立留出段'} · 置信水平 {diagnosticPercent(result.confidence_level)} · 每块 {result.block_length ?? '未知'} 个观测 · {result.replicates ?? '未知'} 次重采样 · 随机种子 {result.seed ?? '未知'}。完整时间块 {result.full_blocks ?? '未知'}，完整状态周期 {result.complete_cycles ?? '未知'}；时间块不等于状态区间。</p>}
      {result?.metrics && <div className="overflow-x-auto"><table aria-label="历史指标区间" className="w-full text-xs"><thead><tr>{['指标', '估计', '下界', '上界', '有效重采样', '说明'].map(title => <th scope="col" key={title} className="p-2 text-left">{title}</th>)}</tr></thead><tbody>{Object.entries(result.metrics).map(([key, metric]) => {
        const format = metric.unit === 'fraction' ? diagnosticPercent : diagnosticNumber
        return <tr className="border-b border-slate-100" key={key}><th scope="row" className="p-2 text-left">{metricNames[key] || '其他历史指标'}</th><td className="p-2 text-right">{format(metric.estimate)}</td><td className="p-2 text-right">{format(metric.lower)}</td><td className="p-2 text-right">{format(metric.upper)}</td><td className="p-2 text-right">{metric.valid_replicates}</td><td className="p-2">{diagnosticReason(metric.reason) || (metric.unit === 'fraction' ? '比例' : 'Brier 分数')}</td></tr>
      })}</tbody></table></div>}
    </details>
  </section>
}
