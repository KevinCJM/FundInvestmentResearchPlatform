import type { TaaBaseline, TaaPreflight, TaaPreview, TaaPreviewRequest } from '../../services/tacticalAllocation'
import { buttonClass, sectionClass } from '../risk-models/ResearchUI'
import { Link } from 'react-router-dom'
import { allocationJourneyPath } from '../../app/allocationJourney'

export default function TaaResearchContext({ baseline, request, preview, preflight, checking, error, contextIssue, onChange, onEdit, onRetry }: {
  baseline: TaaBaseline; request: TaaPreviewRequest; preview: TaaPreview | null; preflight: TaaPreflight | null
  checking: boolean; error: string; contextIssue?: string; onRetry: () => void; onChange: (patch: Partial<TaaPreviewRequest>) => void; onEdit: () => void
}) {
  const names = Object.fromEntries(baseline.assets.map(asset => [asset.id, asset.name]))
  const signalTraining = preview ? preview.data.training : preflight?.training
  const blocked = preflight?.quality.status === 'blocked'
  const reasons = [...new Set([...(preflight?.training.reasons ?? []), ...(preflight?.pit.reasons ?? [])])]
  return <section className={`${sectionClass} space-y-3`} aria-label="本次研究条件与资格">
    <div className="flex flex-wrap items-center justify-between gap-2"><h2 className="text-sm font-semibold">本次研究条件</h2><button type="button" className="min-h-9 text-sm font-medium text-accent-800 underline" onClick={onEdit}>修改日期与费用</button></div>
    <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-xs sm:grid-cols-3 lg:grid-cols-6">
      {[
        ['SAA 方案日期', baseline.as_of.slice(0, 10)],
        ['行情截至', preview?.data.end_date ?? preflight?.coverage.end_date ?? '正在核验'],
        ['研究日 / 知识截止', request.as_of],
        ['训练区间', `${request.start_date} — ${request.train_end_date}`],
        ['独立验证', `${request.train_end_date} 之后至 ${request.end_date}`],
        ['单边费用', `${request.transaction_cost_bps} 基点（${(request.transaction_cost_bps / 100).toFixed(2)}%）`],
      ].map(([label, value]) => <div key={label} className="min-w-0"><dt className="text-slate-600">{label}</dt><dd className="mt-1 break-words font-medium leading-5">{value}</dd></div>)}
    </dl>
    <div role="status" className={`rounded-lg px-3 py-2 text-xs leading-5 ${blocked || error ? 'bg-rose-50 text-rose-900' : preflight?.can_calculate ? 'bg-accent-50 text-accent-900' : 'bg-amber-50 text-amber-900'}`}>
      {contextIssue ? '当前研究日期与平台 PIT 冲突，请先处理上方日期提示。' : checking ? '正在检查数据质量、共同区间与训练可得时间…' : error || (blocked ? '数据质量待处理：当前结果不能用于配置判断。' : !preflight ? '等待研究条件预检。' : preflight.can_calculate ? `${request.search ? '可进行训练期强度搜索' : '可做固定假设比较，不参与强度选优'} · 训练 ${preflight.training.train_observations} 期 / 验证 ${preflight.training.validation_observations} 期 · 历史 PIT 尚未认证` : '当前条件不能计算，请处理以下事项。')}
    </div>
    {signalTraining?.train_signal_observations != null && <p className="text-xs leading-5 text-slate-600">有效趋势信号：训练 {signalTraining.train_signal_observations} / {signalTraining.train_observations} 期，验证 {signalTraining.validation_signal_observations} / {signalTraining.validation_observations} 期。没有信号的期间维持 SAA；有数据不代表有可用信号。</p>}
    {preview && request.signal_mode === 'momentum' && !preview.data.training && <p className="text-xs text-slate-600">此结果未记录有效信号期数；重新计算会采用当前趋势口径。</p>}
    {error && <button type="button" className={buttonClass} onClick={onRetry}>重试条件检查</button>}
    {preflight && (!preflight.can_calculate || preflight.training.unavailable_count > 0) && <div className="space-y-2 text-xs leading-5">
      {preflight.training.train_signal_observations === 0 && <p className="text-amber-900">训练期没有有效趋势信号，无法据此挑选最优偏离。请调整观察窗口、信号有效期或检查公布日期。</p>}
      {preflight.quality.issues.map((issue, index) => <p key={`${issue.asset_id}-${issue.date}-${index}`} className="text-rose-900">{names[issue.asset_id] ?? issue.asset_id} · {issue.date}：{issue.message}</p>)}
      {preflight.training.unavailable_count > 0 && <p className="text-amber-900">训练截止时尚不可得 {preflight.training.unavailable_count} 条；可得时间未知 {preflight.training.unknown_count} 条。{preflight.training.earliest_available_date ? `最早可得日期：${preflight.training.earliest_available_date}。` : ''}</p>}
      {preflight.guidance.map(item => <div key={item.code} className="flex flex-wrap items-center gap-2"><p className="min-w-0 flex-1 basis-64">{item.message}</p>{item.action === 'review_data' ? <Link className={buttonClass} to={allocationJourneyPath('classes')}>返回大类检查产品</Link> : <button type="button" className={buttonClass} onClick={() => onChange(item.action === 'fixed_comparison' ? { search: false } : item.patch ?? preflight.dates)}>{item.action === 'fixed_comparison' ? '改为固定假设比较' : '采用建议日期'}</button>}</div>)}
    </div>}
    <details className="text-xs leading-5 text-slate-600"><summary className="cursor-pointer">日期含义、继承范围与证据限制</summary><div className="mt-2 space-y-1"><p>继承 SAA 权重、资产范围和约束。以下回测按本页日期、日频目标再平衡和费用重算，不直接沿用 SAA 页的历史曲线。</p><p>研究日限制本次可知信息；行情日期不是信息首次可得日期，当前修订数据的历史回放不等于正式 PIT 业绩。</p>{reasons.map(reason => <p key={reason}>{reason}</p>)}</div></details>
  </section>
}
