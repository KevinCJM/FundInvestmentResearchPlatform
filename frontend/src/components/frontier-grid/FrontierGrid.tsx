import { Button } from '../ui'
import { Field, NumberInput, inputClass, percentText } from '../risk-models/ResearchUI'

export interface FrontierGridSettings {
  enabled: boolean
  point_count: number
  max_iterations: number
  accept_continuous_weights: boolean
}
export const defaultFrontierGrid: FrontierGridSettings = {
  enabled: false, point_count: 20, max_iterations: 300, accept_continuous_weights: false,
}
export interface FrontierGridPoint {
  target_index: number; target: number | null; value: [number | null, number | null]
  weights: Array<number | null>; status: string; iterations: number
  optimality_residual: number | null; constraint_violation: number | null
  duplicate_of: number | null; candidate_index: number | null; on_frontier: boolean
}
export interface FrontierGridResult {
  requested_points: number; attempted_points: number; successful_points: number; failed_points: number
  unattempted_points: number; duplicate_targets: number; duplicate_solutions: number; added_candidates: number
  max_iterations: number; risk_solver: string; optimality_scope: string
  points: FrontierGridPoint[]; curve: Array<FrontierGridPoint | null>
  endpoints: Array<{ kind: string; status: string; iterations: number }>
}
const statusText: Record<string, string> = {
  converged: '已收敛', max_iterations: '达到迭代上限', infeasible_target: '目标不可行',
  numerical_failure: '数值求解失败', line_search_failed: '线搜索未收敛', range_unresolved: '端点未收敛，未开始',
}
export function frontierGridIssue(value: FrontierGridSettings, quantized: boolean): string {
  if (!value.enabled) return ''
  if (!Number.isInteger(value.point_count) || value.point_count < 2 || value.point_count > 200) return '前沿目标点数须为 2 至 200 的整数。'
  if (!Number.isInteger(value.max_iterations) || value.max_iterations < 1 || value.max_iterations > 1000) return '单点最大迭代次数须为 1 至 1000 的整数。'
  if (quantized && !value.accept_continuous_weights) return '请确认：散点采样取整，目标网格使用连续权重；两者不是同一离散可行域。'
  return ''
}

export function FrontierGridControls({ value, quantized, busy, onChange }: {
  value: FrontierGridSettings; quantized: boolean; busy: boolean
  onChange: (next: FrontierGridSettings) => void
}) {
  const issue = frontierGridIssue(value, quantized)
  return <div className="space-y-4" aria-label="整条前沿求解设置">
    <label className="flex min-h-10 items-center gap-2 text-sm font-medium">
      <input type="checkbox" checked={value.enabled} disabled={busy}
        onChange={event => onChange({ ...value, enabled: event.target.checked })} />按目标网格加密整条前沿
    </label>
    <p className="text-xs leading-5 text-slate-600">在当前资产和单项／分组边界内，先求解前沿端点，再沿收益轴逐目标求最小风险。20 或 200 控制目标数量，不是迭代次数，也不是增加随机散点。</p>
    {value.enabled && <>
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="前沿目标点数" hint="2–200 个收益目标。增加点数会加密整条曲线。">
          <NumberInput aria-label="前沿目标点数" className={inputClass} value={value.point_count} min={2} max={200} disabled={busy}
            onValueChange={point_count => onChange({ ...value, point_count })} />
        </Field>
        <Field label="单点最大迭代次数" hint="每个优化问题单独的预算；默认 300，端点求解使用相同预算。">
          <NumberInput aria-label="单点最大迭代次数" className={inputClass} value={value.max_iterations} min={1} max={1000} disabled={busy}
            onValueChange={max_iterations => onChange({ ...value, max_iterations })} />
        </Field>
      </div>
      <p className="text-xs leading-5 text-slate-600">实际算法：固定签名 NJIT。波动风险使用主动集二次规划；其他风险使用局部 BFGS-SQP，不称为 SLSQP，也不保证非光滑风险的全局最优。失败目标不会伪装成成功点。</p>
      {quantized && <label className="flex items-start gap-2 text-sm leading-6">
        <input className="mt-1" type="checkbox" disabled={busy} checked={value.accept_continuous_weights}
          onChange={event => onChange({ ...value, accept_continuous_weights: event.target.checked })} />
        我确认网格采用连续权重，取整步长只用于散点采样；采用优化权重时不再强制取整。
      </label>}
      {issue && <p role="status" className="text-sm text-amber-800">{issue}</p>}
    </>}
  </div>
}

export function FrontierGridResults({ result, assetNames, riskLabel, returnLabel, onAdopt }: {
  result: FrontierGridResult; assetNames: string[]; riskLabel: string; returnLabel: string
  onAdopt: (point: FrontierGridPoint) => void
}) {
  const unresolved = result.endpoints.filter(endpoint => endpoint.status !== 'converged')
  return <section className="mt-4 space-y-3" aria-label="逐目标前沿求解结果">
    <h3 className="text-base font-semibold">整条前沿求解结果</h3>
    <p role="status" className="text-sm leading-6">目标 {result.requested_points} 个 · 成功 {result.successful_points} 个 · 失败 {result.failed_points} 个 · 未开始 {result.unattempted_points} 个 · 重复目标 {result.duplicate_targets} 个 · 重复解 {result.duplicate_solutions} 个。</p>
    {unresolved.length > 0 && <p className="text-sm text-amber-800">前沿端点未完成：{unresolved.map(endpoint => `${endpoint.kind === 'minimum_risk' ? '最低风险' : '最高收益'}：${statusText[endpoint.status] ?? endpoint.status}`).join('；')}。请增加单点迭代预算或复核约束，不将未求解目标画成曲线。</p>}
    <p className="text-xs leading-5 text-slate-600">曲线按收益目标顺序连接真实求解结果，失败处断开；重复解不增加有效密度。代表点从采样、局部精炼和成功网格解的共同集合重选。{result.optimality_scope === 'local_numerical_stationarity' ? '当前风险只能报告局部数值收敛，不能据此证明全局最优。' : '当前使用凸二次规划及数值 KKT 检查。'}所有网格权重均为连续权重。</p>
    <details>
      <summary className="min-h-10 cursor-pointer py-2 text-sm font-medium">逐目标状态、权重与采用（{result.points.length} 项）</summary>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] text-sm" aria-label="前沿目标求解明细">
          <thead><tr>
            <th scope="col" className="p-2 text-left">目标</th>
            <th scope="col" className="p-2 text-right">目标{returnLabel}</th>
            <th scope="col" className="p-2 text-right">实际{returnLabel}</th>
            <th scope="col" className="p-2 text-right">{riskLabel}</th>
            <th scope="col" className="p-2 text-right">迭代次数</th>
            <th scope="col" className="p-2 text-left">状态</th>
            <th scope="col" className="p-2 text-left">权重</th>
            <th scope="col" className="p-2 text-left">操作</th>
          </tr></thead>
          <tbody>{result.points.map(point => <tr key={point.target_index} className="border-b border-slate-200">
            <th scope="row" className="p-2 text-left font-medium">{point.target_index + 1}</th>
            <td className="p-2 text-right tabular-nums">{percentText(point.target)}</td>
            <td className="p-2 text-right tabular-nums">{percentText(point.value[1])}</td>
            <td className="p-2 text-right tabular-nums">{percentText(point.value[0])}</td>
            <td className="p-2 text-right tabular-nums">{point.iterations}</td>
            <td className="p-2 text-xs leading-5">{statusText[point.status] ?? point.status}{point.duplicate_of != null ? `；重复目标 ${point.duplicate_of + 1} 的解` : ''}{point.status === 'converged' && !point.on_frontier ? '；未进入最终前沿' : ''}</td>
            <td className="p-2 text-xs leading-5">{assetNames.map((name, index) => <span className="block whitespace-nowrap" key={name}>{name}：{percentText(point.weights[index])}</span>)}</td>
            <td className="p-2"><Button disabled={point.status !== 'converged' || point.candidate_index == null}
              aria-label={`采用前沿目标 ${point.target_index + 1}`} onClick={() => onAdopt(point)}>采用权重</Button></td>
          </tr>)}</tbody>
        </table>
      </div>
    </details>
  </section>
}
