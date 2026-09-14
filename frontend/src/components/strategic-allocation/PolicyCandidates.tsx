import RiskBudgetEditor, { riskBudgetError } from './RiskBudgetEditor'
import { GoalCandidateSummary } from '../investment-mandate/MandateResults'
import { Button } from '../ui'
import { Field, inputClass, NumberInput, percentText, sectionClass } from '../risk-models/ResearchUI'
import { percentInputValue, type PolicyCandidate, type PolicyPreview, type PolicyRequest } from '../../services/strategicAllocation'

export default function PolicyCandidates({ value, assets, result, busy, compareDisabled, onChange, onCompare, onSelect }: {
  value: PolicyRequest; assets: string[]; result: PolicyPreview | null; busy: boolean; compareDisabled: boolean
  onChange: (value: PolicyRequest) => void; onCompare: () => void; onSelect: (candidate: PolicyCandidate) => void
}) {
  const budgetIssue = riskBudgetError(assets, value.risk_budget)
  return <div className="space-y-5">
    <section className={`${sectionClass} space-y-4`} aria-label="政策资金与偏离边界">
      <h2 className="text-lg font-semibold">允许配置多少？</h2>
      <p className="text-sm leading-6 text-slate-600">目标收益、波动、流动性和主动风险预算由所选投资目标提供。这里设置资产权重区间与今后的战术偏离幅度，不能突破上游目标。</p>
      <div className="overflow-x-auto"><table aria-label="政策资产约束" className="w-full min-w-[460px] text-sm"><thead><tr><th scope="col" className="p-2 text-left">大类</th>{['最低权重（%）', '最高权重（%）', '最大战术偏离（百分点）'].map(label => <th scope="col" className="p-2 text-right" key={label}>{label}</th>)}</tr></thead><tbody>{assets.map(asset => <tr key={asset} className="border-t border-slate-200"><th scope="row" className="p-2 text-left font-medium">{asset}</th>{(['min_weight', 'max_weight', 'max_abs_tilt'] as const).map((field, i) => <td className="p-2" key={field}><NumberInput aria-label={`${asset}${['最低权重', '最高权重', '最大战术偏离'][i]}`} className={`${inputClass} text-right tabular-nums`} value={percentInputValue(value.constraints[asset]?.[field])} onValueChange={number => onChange({ ...value, constraints: { ...value.constraints, [asset]: { ...value.constraints[asset], [field]: number / 100 } } })} /></td>)}</tr>)}</tbody></table></div>
      <details className="border-t border-slate-200 pt-4"><summary className="cursor-pointer text-sm font-medium">联合约束与候选搜索设置</summary>
        <div className="mt-4 space-y-4">
          {value.group_limits.map((group, index) => <fieldset className="min-w-0 space-y-3 border-b border-slate-200 pb-4" key={index}><legend className="text-sm font-medium">联合约束 {index + 1}</legend>
            <div className="grid gap-3 sm:grid-cols-3"><Field label={`联合约束 ${index + 1}名称`}><input className={inputClass} value={group.id} onChange={e => onChange({ ...value, group_limits: value.group_limits.map((g, i) => i === index ? { ...g, id: e.target.value } : g) })} /></Field>
              {(['lo', 'hi'] as const).map(field => <Field key={field} label={`联合约束 ${index + 1}${field === 'lo' ? '最低' : '最高'}比例（%）`}><NumberInput className={inputClass} value={percentInputValue(group[field])} onValueChange={number => onChange({ ...value, group_limits: value.group_limits.map((g, i) => i === index ? { ...g, [field]: number / 100 } : g) })} /></Field>)}</div>
            <div className="flex flex-wrap gap-4">{assets.map(asset => <label className="flex items-center gap-2 text-sm" key={asset}><input type="checkbox" checked={group.assets.includes(asset)} onChange={e => onChange({ ...value, group_limits: value.group_limits.map((g, i) => i === index ? { ...g, assets: e.target.checked ? [...g.assets, asset] : g.assets.filter(name => name !== asset) } : g) })} />{asset}</label>)}</div>
            <Button onClick={() => onChange({ ...value, group_limits: value.group_limits.filter((_, i) => i !== index) })}>移除此联合约束</Button>
          </fieldset>)}
          <Button disabled={value.group_limits.length >= 28} onClick={() => onChange({ ...value, group_limits: [...value.group_limits, { id: `联合约束 ${value.group_limits.length + 1}`, assets: [], lo: 0, hi: 1 }] })}>增加联合约束</Button>
          <div className="grid gap-3 sm:grid-cols-3">
            <Field label="均值不确定性惩罚倍数"><NumberInput className={inputClass} value={value.uncertainty_penalty} min={0} max={5} onValueChange={number => onChange({ ...value, uncertainty_penalty: number })} /></Field>
            <Field label="随机候选数"><NumberInput className={inputClass} value={value.candidate_count} min={200} max={5000} onValueChange={number => onChange({ ...value, candidate_count: number })} /></Field>
            <Field label="随机种子"><NumberInput className={inputClass} value={value.seed} min={0} onValueChange={number => onChange({ ...value, seed: number })} /></Field>
          </div>
        </div>
      </details>
      <RiskBudgetEditor assets={assets} value={value.risk_budget} onChange={risk_budget => onChange({ ...value, risk_budget })} />
      <Button tone="primary" disabled={busy || Boolean(budgetIssue) || compareDisabled || !assets.length || !value.cma_id || !value.mandate_id} onClick={onCompare}>{busy ? '正在比较…' : '比较符合目标的政策候选'}</Button>
    </section>
    {result && <section className={`${sectionClass} space-y-4`} aria-label="政策候选结果">
      <h2 className="text-lg font-semibold">先比较，再决定采用哪个</h2>
      {result.current_application_eligible === false && <div className="space-y-1 text-sm text-amber-800"><p>当前应用条件尚未满足，可以继续研究；保存版本不等于通过应用检查。</p>{result.application_blockers?.map(reason => <p key={reason}>{reason}</p>)}</div>}
      {result.candidates.some(c => c.goal_check || c.benchmark_check) && <div className="space-y-3" aria-label="投资目标检查">{result.candidates.map(c => <div className="border-b border-slate-200 pb-3" key={c.id}><h3 className="text-sm font-medium">{c.name}</h3><GoalCandidateSummary candidate={c} /></div>)}</div>}
      {result.unavailable_candidates?.map(c => <p role="status" className="text-sm text-amber-800" key={c.id}>{c.name}不可用：{c.unavailable_reason}</p>)}
      <p className="text-sm leading-6 text-slate-600">以下来自同一组可行候选，不保证全局最优。保守收益按声明的均值不确定区间扣减，不是概率分位数。</p>
      <div className="overflow-x-auto"><table aria-label="长期政策候选比较" className="w-full min-w-[600px] text-sm"><thead><tr>{['候选方法', '预期年收益', '预期年波动', '保守年收益', ...assets, '下一步'].map(label => <th key={label} scope="col" className={`whitespace-nowrap p-3 ${label === '候选方法' || label === '下一步' ? 'text-left' : 'text-right'}`}>{label}</th>)}</tr></thead><tbody>{result.candidates.map(candidate => <tr className="border-t border-slate-200" key={candidate.id}><th scope="row" className="whitespace-nowrap p-3 text-left font-medium">{candidate.name}</th>{[candidate.metrics.expected_return, candidate.metrics.volatility, candidate.metrics.conservative_return, ...assets.map(asset => candidate.weights[asset])].map((number, i) => <td key={i} className="whitespace-nowrap p-3 text-right tabular-nums">{percentText(number)}</td>)}<td className="p-3"><Button disabled={busy} onClick={() => onSelect(candidate)}>复核此候选</Button></td></tr>)}</tbody></table></div>
      <details className="border-t border-slate-200 pt-4"><summary className="cursor-pointer text-sm font-medium">风险贡献与证据限制</summary><div className="mt-3 space-y-3"><p className="text-sm text-slate-600">风险贡献保留正负号，负值表示分散作用；零方差时贡献未定义。风险预算距离是有符号贡献与目标的平方距离，有限搜索不保证精确匹配。</p>{result.candidates.map(candidate => <p className="text-sm leading-6 text-slate-600" key={candidate.id}>{candidate.name}：{assets.map(asset => `${asset} ${percentText(candidate.risk_contributions[asset])}`).join('；')}{candidate.risk_budget_distance !== undefined && `；风险预算平方距离 ${candidate.risk_budget_distance.toPrecision(5)}`}</p>)}{result.warnings.map((warning, index) => <p key={index} className="text-xs leading-5 text-slate-600">{warning}</p>)}</div></details>
    </section>}
  </div>
}
