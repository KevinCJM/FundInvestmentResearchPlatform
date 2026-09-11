import type { RegimeResultData, RegimeResultInterval } from './regimeResultAdapter'
import { regimeFeatureNumber } from './regimeResultAdapter'

function reasonText(value: unknown): string {
  if (Array.isArray(value)) return value.filter(item => typeof item === 'string').join('；')
  return typeof value === 'string' ? value : ''
}

export default function RegimeEvidencePanel({ result, interval }: { result: RegimeResultData; interval: RegimeResultInterval | null }) {
  if (!interval) return <p className="rounded-xl bg-slate-50 p-5 text-sm text-slate-600">点击上方色带或区间明细，查看该段何时识别、何时生效及判断依据。</p>
  const first = result.points[interval.start_index]
  const reasons = reasonText(first?.raw.reasons)
  const confirmed = interval.confirmed_at ?? first?.confirmed_at
  const effective = interval.effective_start ?? first?.effective_from
  const phase = regimeFeatureNumber(first, 'phase')
  const phaseLabel = phase == null ? null : ['牛市上行', '牛市回调', '熊市下行', '熊市反弹', '震荡'][phase]
  const risk = regimeFeatureNumber(first, 'risk')
  const hasTrend = regimeFeatureNumber(first, 'distance') != null
  const peakStart = regimeFeatureNumber(first, 'phase_start_index')
  const peakEnd = regimeFeatureNumber(first, 'phase_end_index')
  const peakReturn = regimeFeatureNumber(first, 'phase_return')
  const sidewaysStart = regimeFeatureNumber(first, 'sideways_start_index')
  const sidewaysEnd = regimeFeatureNumber(first, 'sideways_end_index')
  return <section aria-label="区间判断依据" className="space-y-4 rounded-xl border border-slate-200 p-4">
    <div><h3 className="font-bold text-slate-900">{interval.label} · {interval.start_date} 至 {interval.end_date}</h3><p className="mt-1 text-xs text-slate-500">本段 {interval.observations} 个真实观测日；以下时点对应区间首个观测日，不代表整段均已可交易。</p></div>
    <dl className="grid gap-3 text-sm sm:grid-cols-3">
      <div><dt className="text-slate-500">首个观测日</dt><dd className="mt-1 font-medium">{interval.start_date}</dd></div>
      <div><dt className="text-slate-500">首点识别 / 确认日</dt><dd className="mt-1 font-medium">{confirmed ?? '未提供'}</dd></div>
      <div><dt className="text-slate-500">首点生效日</dt><dd className="mt-1 font-medium">{effective ?? '尚未提供或未生效'}</dd></div>
    </dl>
    {peakStart != null && peakStart >= 0 && peakEnd != null && peakEnd >= 0 ? <div className="space-y-2 rounded-lg bg-violet-50 p-3 text-sm" aria-label="峰谷定界依据">
      <p>起始拐点：{result.points[peakStart]?.observation_date ?? '未知'}；结束拐点：{result.points[peakEnd]?.observation_date ?? '未知'}。</p>
      <p>{sidewaysStart != null && sidewaysStart >= 0 ? '原始完整峰谷涨跌幅' : '完整峰谷涨跌幅'}：{peakReturn == null ? '未知' : new Intl.NumberFormat('zh-CN', { style: 'percent', maximumFractionDigits: 2 }).format(peakReturn)}。</p>
      <p className="text-xs text-slate-600">色带包含起始拐点、不含结束拐点；结束拐点属于下一阶段。识别时间按整段样本数据可得时间保守记录，不提供交易生效日。</p>
    </div> : null}
    {(['segment_amplitude', 'segment_volatility', 'segment_duration', 'segment_efficiency'] as const).some(key => regimeFeatureNumber(first, key) != null) ? <dl aria-label="独立区间统计" className="grid gap-3 rounded-lg bg-slate-50 p-3 text-sm sm:grid-cols-2">{([['segment_amplitude', '区间振幅'], ['segment_volatility', '区间收益波动率（未年化）'], ['segment_duration', '区间长度（观测间隔）'], ['segment_efficiency', '区间方向效率']] as const).map(([key, label]) => { const value = regimeFeatureNumber(first, key); return value == null ? null : <div key={key}><dt className="text-slate-500">{label}</dt><dd>{key === 'segment_duration' ? value : key === 'segment_efficiency' ? value.toFixed(4) : new Intl.NumberFormat('zh-CN', { style: 'percent', maximumFractionDigits: 2 }).format(value)}</dd></div> })}</dl> : null}
    {sidewaysStart != null && sidewaysStart >= 0 && sidewaysEnd != null && sidewaysEnd >= 0 ? <div className="space-y-2 rounded-lg bg-slate-100 p-3 text-sm" aria-label="峰谷震荡依据">
      <p>合并范围：{result.points[sidewaysStart]?.observation_date ?? '未知'} 至 {result.points[sidewaysEnd]?.observation_date ?? '未知'}（不含结束点）；合并波段数：{regimeFeatureNumber(first, 'sideways_swing_count') ?? '未知'}。</p>
      <p>整段振幅：{regimeFeatureNumber(first, 'sideways_range') == null ? '未知' : new Intl.NumberFormat('zh-CN', { style: 'percent', maximumFractionDigits: 2 }).format(regimeFeatureNumber(first, 'sideways_range')!)}；方向效率 ER：{regimeFeatureNumber(first, 'sideways_efficiency')?.toFixed(4) ?? '未知'}。</p>
      <p className="text-xs text-slate-600">按整段原始价格路径确认震荡；上方原始峰谷是合并前的波段。尾部未完成区间不参与合并。</p>
    </div> : null}
    {hasTrend ? <div className="space-y-2 rounded-lg bg-amber-50 p-3 text-sm" aria-label="趋势判断证据">
      <p>趋势内部阶段：{phaseLabel ?? '尚未分类'}；待确认连续期数：{regimeFeatureNumber(first, 'pending_count') ?? '未知'}。</p>
      <dl className="grid gap-2 sm:grid-cols-3">
        {([['distance', '标准化价格偏离'], ['slope', '标准化趋势斜率'], ['efficiency', '方向效率']] as const).map(([key, label]) => <div key={key}><dt className="text-slate-500">{label}</dt><dd>{regimeFeatureNumber(first, key)?.toFixed(4) ?? '未知'}</dd></div>)}
      </dl>
      <p>{risk === 1 ? '原始指数已触发回撤或急跌警报，短信号过滤不会抹去这项风险。' : risk === 0 ? '原始指数未触发配置的回撤或急跌警报。' : '原始指数风险数据不足。'}</p>
      {result.overview.mode === 'retrospective' ? <p className="text-xs text-slate-600">以上为合并前的趋势证据；事后合并可能改变最终色带。</p> : null}
    </div> : null}
    <div className="rounded-lg bg-slate-50 p-3 text-sm text-slate-700">
      {interval.state_id === 'unclassified'
        ? (interval.reason && interval.reason !== 'unknown' ? interval.reason : '未分类：当前输出未区分预热、数据缺失或规则未命中，不能将其解释为震荡。')
        : reasons || (interval.reason && interval.reason !== 'unknown' ? interval.reason : '当前输出未提供具体规则证据，可返回构建视图查看对应节点。')}
      {first?.raw.reason_code != null ? <p className="mt-2 text-xs text-slate-500">服务端原因码：{String(first.raw.reason_code)}</p> : null}
    </div>
    <p className="text-xs text-slate-500">{result.overview.mode === 'retrospective' ? '本次为事后解释；分类可能使用之后才可得的信息，不能直接用作当时的交易信号。' : '本次采用当时可知的识别模式；当前图仍按观测日展示，交易时点需以服务端可得性与发布门禁为准。'}</p>
  </section>
}
