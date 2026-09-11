import type { SourceCatalog } from '../../services/dataSources'
import type { EtlParameter, EtlStep } from '../../services/etl'
import { inputClass } from './EditorFields'
import RequestParameterFields from './RequestParameterFields'

export default function EtlTaskFields({ step, catalog, parameters, onChange }: {
  step: EtlStep; catalog: SourceCatalog; parameters: EtlParameter[]; onChange: (value: EtlStep) => void
}) {
  const tasks = catalog.etl_tasks ?? []
  const spec = tasks.find(t => t.id === step.task_id)
  return <section className="space-y-3" aria-label="通用数据集任务设置">
    <label className="block text-xs font-semibold">任务类型<select aria-label="任务类型" className={inputClass} value={step.task_id ?? ''} onChange={event => {
      const selected = tasks.find(t => t.id === event.target.value)
      if (selected) onChange({ ...step, task_id: selected.id, name: selected.name, source_id: selected.source_ids[0] ?? null, params: Object.fromEntries(selected.parameters.filter(p => p.default).map(p => [p.name, p.default])), parameter_bindings: {} })
    }}><option value="">选择数据集任务</option>{[...new Set(tasks.map(t => t.category))].map(category => <optgroup key={category} label={category}>{tasks.filter(t => t.category === category).map(t => <option key={t.id} value={t.id}>{t.name}</option>)}</optgroup>)}</select></label>
    {spec?.requires_source ? <label className="block text-xs font-semibold">任务数据源<select aria-label="任务数据源" className={inputClass} value={step.source_id ?? ''} onChange={event => onChange({ ...step, source_id: event.target.value })}><option value="">选择来源</option>{catalog.sources.filter(s => spec.source_ids.includes(s.config.id)).map(s => <option key={s.config.id} value={s.config.id}>{s.config.name}</option>)}</select></label> : null}
    {spec ? <p className="text-xs leading-5 text-slate-600">{spec.description}</p> : <p className="text-xs text-amber-800">请选择已登记的任务；不会执行任意脚本或命令。</p>}
    <label className="block text-xs font-semibold">数据集更新策略<select aria-label="数据集更新策略" className={inputClass} value={step.mode} onChange={event => onChange({ ...step, mode: event.target.value as EtlStep['mode'] })}><option value="inherit">跟随本次运行模式</option><option value="full">每次全量刷新</option><option value="incremental">固定增量</option></select></label>
    <RequestParameterFields fields={spec?.parameters ?? []} values={step.params} bindings={step.parameter_bindings} onChange={params => onChange({ ...step, params })} />
    {spec?.parameters.length ? <fieldset className="space-y-2 rounded-lg bg-slate-50 p-3"><legend className="text-xs font-semibold">参数取值方式</legend>{spec.parameters.map(field => <label key={field.name} className="block text-xs">{field.label}取值方式<select aria-label={`${field.label}取值方式`} className={inputClass} value={step.parameter_bindings?.[field.name] ?? ''} onChange={event => {
      const bindings = { ...step.parameter_bindings }
      if (event.target.value) bindings[field.name] = event.target.value; else delete bindings[field.name]
      onChange({ ...step, parameter_bindings: bindings })
    }}><option value="">使用上方固定值</option>{parameters.map(p => <option key={p.id} value={p.id}>运行参数：{p.label}</option>)}</select></label>)}</fieldset> : null}
  </section>
}
