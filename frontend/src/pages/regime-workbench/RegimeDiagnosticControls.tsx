import type { RegimeBootstrapPolicy, RegimeStabilityPolicy } from '../../services/regimeDiagnostics'
export const defaultStability: Required<RegimeStabilityPolicy> = { enabled: true, max_variants: 6, perturbation: 0.1, parameters: true, windows: true, seeds: true, truncation: true }
export const defaultBootstrap: Required<RegimeBootstrapPolicy> = { enabled: true, replicates: 200, block_length: 10, confidence_level: 0.95, minimum_blocks: 5, minimum_cycles: 3, minimum_valid_replicates: 100 }
const field = 'mt-1 block min-h-10 w-full min-w-0 rounded-lg border border-slate-300 bg-white px-3 text-sm tabular-nums'
export default function RegimeDiagnosticControls({ stability, bootstrap, onStability, onBootstrap }: {
  stability: RegimeStabilityPolicy; bootstrap?: RegimeBootstrapPolicy
  onStability: (next: RegimeStabilityPolicy) => void; onBootstrap?: (next: RegimeBootstrapPolicy) => void
}) {
  const s = { ...defaultStability, ...stability }, b = { ...defaultBootstrap, ...bootstrap }
  return <details><summary className="min-h-10 cursor-pointer text-sm text-slate-600">高级检查设置</summary>
    <div className="space-y-3 py-2 text-sm text-slate-700">
      <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">{([['enabled', '检查稳定性'], ['parameters', '参数扰动'], ['windows', '窗口变化'], ['seeds', '随机种子'], ['truncation', '截尾修订']] as const).map(([key, label]) => <label className="flex min-h-10 items-center gap-2" key={key}><input type="checkbox" checked={s[key]} onChange={e => onStability({ ...s, [key]: e.target.checked })} />{label}</label>)}
        <label>最多变体数<input className={field} type="number" min={1} max={12} value={s.max_variants} onChange={e => onStability({ ...s, max_variants: Math.min(12, Math.max(1, Math.trunc(Number(e.target.value)))) })} /></label>
        <label>参数扰动比例<input className={field} type="number" min={0.01} max={0.5} step={0.01} value={s.perturbation} onChange={e => onStability({ ...s, perturbation: Math.min(0.5, Math.max(0.01, Number(e.target.value))) })} /></label>
      </div>
      {bootstrap && onBootstrap && <>
        <label className="flex min-h-10 items-center gap-2"><input type="checkbox" checked={b.enabled} onChange={e => onBootstrap({ ...b, enabled: e.target.checked })} />估计历史留出指标区间</label>
        <p className="text-xs text-slate-600">每块长度按输入观测数计算，不是状态区间数或固定天数。默认使用配对移动块重采样；种子由服务器固定。</p>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{([
          ['replicates', '重采样次数', 20, 500, 1], ['block_length', '每块观测数', 2, 250, 1], ['confidence_level', '置信水平', 0.8, 0.99, 0.01],
          ['minimum_blocks', '最少完整时间块', 2, 100, 1], ['minimum_cycles', '最少完整状态周期', 1, 100, 1], ['minimum_valid_replicates', '最少有效重采样', 10, b.replicates, 1],
        ] as const).map(([key, label, min, max, step]) => <label key={key}>{label}<input className={field} type="number" min={min} max={max} step={step} value={b[key]} onChange={e => {
          const parsed = Number(e.target.value)
          const value = Math.min(max, Math.max(min, step === 1 ? Math.trunc(parsed) : parsed))
          onBootstrap({ ...b, [key]: value, ...(key === 'replicates' ? { minimum_valid_replicates: Math.min(value, b.minimum_valid_replicates) } : {}) })
        }} /></label>)}</div>
      </>}
    </div>
  </details>
}
