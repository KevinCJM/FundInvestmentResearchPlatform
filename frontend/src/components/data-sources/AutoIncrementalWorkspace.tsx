import { useMemo, useRef, useState } from 'react'
import type { EtlTaskSpec, SourceCatalog } from '../../services/dataSources'
import { blankStep, runEtl, validateEtl, type EtlDefinition, type EtlRun, type EtlValidation } from '../../services/etl'
import { buttonClass, primaryClass } from './EditorFields'
import AutoPlanSummary from './AutoPlanSummary'

export function autoDefinition(tasks: EtlTaskSpec[], selected: string[]): EtlDefinition {
  const ordered: EtlTaskSpec[] = [], visited = new Set<string>(), active = new Set<string>()
  const visit = (id: string) => {
    if (visited.has(id)) return
    if (active.has(id)) throw new Error('任务目录存在循环依赖。')
    const spec = tasks.find(t => t.id === id)
    if (!spec?.auto_incremental_supported) throw new Error('所选任务或其依赖尚不支持自动增量。')
    active.add(id)
    for (const capability of spec.requires) {
      const dependency = tasks.find(t => t.provides.includes(capability))
      if (!dependency) throw new Error('任务目录缺少前置数据集。')
      visit(dependency.id)
    }
    active.delete(id); visited.add(id); ordered.push(spec)
  }
  selected.forEach(visit)
  return { name: '自动增量更新', description: '依据活跃快照自动推断区间；数据集及依赖来自服务端目录。', max_runtime_seconds: 86400,
    steps: ordered.map((task, i) => ({ ...blankStep('task'), id: `auto_${i}`, name: task.name, task_id: task.id, source_id: task.requires_source ? 'tushare' : null, inputs: i ? [`auto_${i - 1}`] : [] })) }
}

export default function AutoIncrementalWorkspace({ catalog, canRun, onStarted }: { catalog: SourceCatalog; canRun: boolean; onStarted: (run: EtlRun) => void }) {
  const tasks = useMemo(() => (catalog.etl_tasks ?? []).filter(t => !t.requires_source || t.source_ids.includes('tushare')), [catalog])
  const [selected, setSelected] = useState<string[]>([])
  const [preview, setPreview] = useState<EtlValidation | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const pending = useRef<{ plan: string; id: string } | null>(null)
  const definition = useMemo(() => {
    try { return selected.length ? autoDefinition(tasks, selected) : null } catch { return null }
  }, [tasks, selected])
  const options = { mode: 'auto_incremental' as const, parameters: {} }
  const inspect = async () => {
    if (!definition) return
    setBusy(true); setError(''); setPreview(null)
    try { setPreview(await validateEtl(definition, options)) }
    catch (reason) { setError(reason instanceof Error ? reason.message : '计划暂时不可用。') }
    finally { setBusy(false) }
  }
  const launch = async () => {
    if (!definition || !preview?.valid || !preview.auto_plan?.ready || busy) return
    if (!window.confirm('按已显示的自动增量计划开始下载？仅写入候选数据，不覆盖正式快照。')) return
    setBusy(true); setError('')
    if (pending.current?.plan !== preview.auto_plan.plan_id) pending.current = { plan: preview.auto_plan.plan_id, id: crypto.randomUUID() }
    try { onStarted(await runEtl(definition, pending.current.id, options, preview.auto_plan.plan_id)) }
    catch (reason) { setError(reason instanceof Error ? reason.message : '启动失败，请重新预览。'); setPreview(null) }
    finally { setBusy(false) }
  }
  return <section className="space-y-4 rounded-xl border border-slate-200 bg-white p-4 sm:p-5">
    <h2 className="text-lg font-bold">自动增量下载</h2>
    <p className="text-sm leading-6 text-slate-600">只选择要更新的数据，无需填写起止日期。系统使用当前活跃 Tushare 快照，自动补入必要的目录和日历依赖。</p>
    {!tasks.length ? <p role="status">当前服务尚未提供数据集目录；如刚升级后端，请在正在运行的下载结束后重启服务。</p> : null}
    <fieldset disabled={busy} className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3" aria-label="自动增量数据集">
      {tasks.map(task => <label key={task.id} className="flex items-start gap-2 rounded-lg bg-slate-50 p-3 text-sm">
        <input type="checkbox" className="mt-1" disabled={!task.auto_incremental_supported} checked={selected.includes(task.id)} onChange={event => { setSelected(value => event.target.checked ? [...value, task.id] : value.filter(id => id !== task.id)); setPreview(null); setError('') }} />
        <span>{task.name}{!task.auto_incremental_supported ? '（请使用手动更新）' : ''}</span>
      </label>)}
    </fieldset>
    {selected.length && !definition ? <p role="alert">任务目录尚未就绪或依赖不完整，请重新加载页面。</p> : null}
    {definition ? <p className="text-xs leading-6 text-slate-500">含依赖的执行顺序：{definition.steps.map(s => s.name).join(' → ')}</p> : null}
    <button type="button" className={buttonClass} disabled={busy || !definition} onClick={() => void inspect()}>{busy ? '正在处理…' : '分析快照并预览区间'}</button>
    {error ? <p role="alert" className="text-sm text-rose-700">{error}</p> : null}
    {preview?.auto_plan ? <AutoPlanSummary plan={preview.auto_plan} /> : preview?.errors.map((e, i) => <p role="alert" key={i} className="text-sm text-rose-700">{e.message}</p>)}
    {preview?.valid ? <button type="button" className={primaryClass} disabled={busy || !canRun || !preview.auto_plan?.ready} onClick={() => void launch()}>确认计划并开始自动增量</button> : null}
    {!canRun ? <p className="text-xs text-amber-800">可以只读预览；有其他任务执行、状态未确认或环境只读时，不允许启动新下载。</p> : null}
  </section>
}
