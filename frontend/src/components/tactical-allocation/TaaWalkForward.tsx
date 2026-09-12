import type { TaaWalkForwardConfig, TaaWalkForwardResult } from '../../services/tacticalAllocation'
import { Field, inputClass, NumberInput, percentText, sectionClass } from '../risk-models/ResearchUI'

export default function TaaWalkForward({ value, result, disabled, onChange }: {
  value?: TaaWalkForwardConfig | null; result?: TaaWalkForwardResult; disabled: boolean
  onChange: (value: TaaWalkForwardConfig | null) => void
}) {
  return <section className={`${sectionClass} space-y-4`} aria-label="多段样本外检验">
    <div><h2 className="text-lg font-semibold">不同阶段是否依然有效？</h2><p className="mt-1 text-sm leading-6 text-slate-600">可增加滚动或扩展训练，逐段验证。每段只按其训练期选择强度，不用本段验证结果重新挑赢家。</p></div>
    <label className="flex items-center gap-3 text-sm font-medium"><input type="checkbox" checked={Boolean(value)} disabled={disabled} onChange={e => onChange(e.target.checked ? { window_mode: 'rolling', training_periods: 126, validation_periods: 63 } : null)} />同时运行多段样本外检验</label>
    {value && <div className="grid gap-4 sm:grid-cols-3">
      <Field label="训练窗口方式"><select className={inputClass} value={value.window_mode} disabled={disabled} onChange={e => onChange({ ...value, window_mode: e.target.value as TaaWalkForwardConfig['window_mode'] })}><option value="rolling">滚动：只用最近一段训练</option><option value="expanding">扩展：累计使用已知历史</option></select></Field>
      <Field label="初始训练期数" hint="至少 20 个共同收益观察期；趋势窗口还需要足够预热。"><NumberInput className={inputClass} value={value.training_periods} min={20} max={2500} disabled={disabled} onValueChange={number => onChange({ ...value, training_periods: number })} /></Field>
      <Field label="每段验证期数" hint="至少 20 期，单次最多 40 段。"><NumberInput className={inputClass} value={value.validation_periods} min={20} max={1000} disabled={disabled} onValueChange={number => onChange({ ...value, validation_periods: number })} /></Field>
    </div>}
    {value && !result && <p className="text-xs leading-5 text-slate-600">设置完成后点击“运行回测与候选比较”。本诊断不改变主研究所选方案；修改参数后旧结果需要重算。</p>}
    {result && <>
      <p role="status" className="text-sm text-slate-700">完成 {result.completed_folds} 段，阻断 {result.blocked_folds} 段；末尾不足验证长度而未使用 {result.excluded_tail_observations} 期。</p>
      <div className="overflow-x-auto"><table aria-label="分段样本外结果" className="w-full min-w-[760px] text-sm"><thead><tr>{['分段', '实际训练区间', '验证区间', '强度', '验证净超额', '跟踪误差', '换手', '资格'].map((label, i) => <th key={label} scope="col" className={`whitespace-nowrap p-2 ${i >= 3 && i <= 6 ? 'text-right' : 'text-left'}`}>{label}</th>)}</tr></thead><tbody>{result.folds.map(fold => <tr key={fold.fold} className="border-t border-slate-200"><th scope="row" className="p-2 text-left font-medium">{fold.fold}</th><td className="p-2 text-xs leading-5">{fold.train_start}<br />{fold.train_end ?? '无成熟样本'}{fold.purged_training_periods > 0 && <span className="block text-amber-800">已剔除尾部 {fold.purged_training_periods} 期</span>}</td><td className="p-2 text-xs leading-5">{fold.validation_start}<br />{fold.validation_end}</td><td className="p-2 text-right tabular-nums">{fold.strength ?? '—'}</td><td className="p-2 text-right tabular-nums">{percentText(fold.validation?.excess_return)}</td><td className="p-2 text-right tabular-nums">{percentText(fold.validation?.tracking_error)}</td><td className="p-2 text-right tabular-nums">{percentText(fold.validation?.turnover)}</td><td className="max-w-64 p-2 text-xs leading-5">{fold.status === 'blocked' ? fold.reasons.join('；') : fold.validation_feasible ? '验证未超限' : fold.baseline_fallback_exemption ? '保留 SAA；基准换手豁免' : '验证风险或换手超限'}</td></tr>)}</tbody></table></div>
      {result.warnings.map(warning => <p key={warning} className="text-xs leading-5 text-slate-600">{warning}</p>)}
    </>}
  </section>
}
