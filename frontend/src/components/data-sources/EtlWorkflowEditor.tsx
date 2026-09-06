import type { SourceCatalog } from '../../services/dataSources'
import { blankStep, stepLabels, type EtlDefinition, type EtlKind, type EtlParameter } from '../../services/etl'
import { buttonClass, inputClass, JsonField } from './EditorFields'
import EtlDownloadFields from './EtlDownloadFields'

export default function EtlWorkflowEditor({ definition, catalog, onChange }: { definition: EtlDefinition; catalog: SourceCatalog; onChange: (value: EtlDefinition) => void }) {
  const patch = (index: number, value: EtlDefinition['steps'][number]) => onChange({ ...definition, steps: definition.steps.map((s, i) => i === index ? value : s) })
  const move = (index: number, offset: number) => { const next = [...definition.steps]; [next[index], next[index + offset]] = [next[index + offset], next[index]]; onChange({ ...definition, steps: next }) }
  const add = (kind: EtlKind) => {
    const step = blankStep(kind)
    const expected = { map: 'download', resolve: 'map', snapshot: 'resolve' }[kind as 'map' | 'resolve' | 'snapshot']
    const prior = definition.steps.filter(s => s.kind === expected)
    step.inputs = kind === 'snapshot' ? prior.map(s => s.id) : prior.slice(-1).map(s => s.id)
    if (kind === 'download') step.source_id = catalog.sources[0]?.config.id
    if (kind === 'resolve') {
      const input = prior[prior.length - 1]
      const download = definition.steps.find(s => s.id === input?.inputs[0])
      step.table_id = catalog.interfaces.find(i => i.config.id === download?.interface_id)?.config.mappings.find(m => m.enabled)?.target_table
    }
    onChange({ ...definition, steps: [...definition.steps, step] })
  }
  return <section className="space-y-4" aria-label="ETL 流程编辑器">
    <div className="grid gap-3 sm:grid-cols-[1fr_200px]"><label className="text-sm font-semibold">流程名称<input className={inputClass} value={definition.name} required onChange={e => onChange({ ...definition, name: e.target.value })} /></label><label className="text-sm font-semibold">最长运行时间（秒）<input className={inputClass} type="number" min={10} max={86400} value={definition.max_runtime_seconds} onChange={e => onChange({ ...definition, max_runtime_seconds: Number(e.target.value) })} /></label></div>
    <label className="block text-sm font-semibold">流程说明<textarea className={inputClass} value={definition.description} onChange={event => onChange({ ...definition, description: event.target.value })} /></label>
    <p className="text-sm leading-6 text-slate-600">流程定义处理顺序，本次运行决定全量或增量。基础信息可固定每次全量刷新；其他下载默认跟随运行模式。移动步骤后请重新校验依赖。</p>
    <details className="rounded-lg border border-slate-200 p-3"><summary className="cursor-pointer text-xs font-semibold">高级：运行参数定义</summary><JsonField label="运行参数定义 JSON" objectOnly={false} value={definition.parameters ?? []} onChange={value => {
      if (!Array.isArray(value) || value.some(item => !item || typeof item !== 'object' || typeof item.id !== 'string' || typeof item.label !== 'string' || !['text', 'date'].includes(item.data_type))) throw new Error('请提供含 id、label、data_type 的参数数组。')
      onChange({ ...definition, parameters: value as EtlParameter[] })
    }} /></details>
    {definition.steps.map((step, index) => {
      const expected = { map: 'download', resolve: 'map', snapshot: 'resolve' }[step.kind as 'map' | 'resolve' | 'snapshot']
      const candidates = definition.steps.slice(0, index).filter(s => s.kind === expected)
      const unresolved = step.inputs.filter(id => !candidates.some(s => s.id === id))
      return <article key={step.id} aria-label={`步骤 ${index + 1} ${step.name}`} className="min-w-0 space-y-3 rounded-xl border border-slate-200 bg-white p-4">
        <div className="flex flex-wrap items-center gap-2"><h3 className="mr-auto text-sm font-bold">{index + 1}. {stepLabels[step.kind]}</h3><button type="button" className={buttonClass} aria-label={`步骤 ${index + 1} 上移`} disabled={index === 0} onClick={() => move(index, -1)}>上移</button><button type="button" className={buttonClass} aria-label={`步骤 ${index + 1} 下移`} disabled={index === definition.steps.length - 1} onClick={() => move(index, 1)}>下移</button><button type="button" className={buttonClass} aria-label={`删除步骤 ${index + 1}`} onClick={() => { if (!definition.steps.some(s => s.inputs.includes(step.id)) || window.confirm('后续步骤引用了此步骤。删除后需要重新选择输入，继续？')) onChange({ ...definition, steps: definition.steps.filter(s => s.id !== step.id) }) }}>删除</button></div>
        <label className="block text-xs font-semibold">步骤名称<input className={inputClass} value={step.name} onChange={e => patch(index, { ...step, name: e.target.value })} /></label>
        {step.kind === 'download' ? <EtlDownloadFields step={step} catalog={catalog} onChange={value => patch(index, value)} /> : <fieldset className="space-y-2"><legend className="text-xs font-semibold">使用哪些前置结果</legend>{candidates.map(s => <label key={s.id} className="flex min-h-10 items-center gap-2 rounded-lg bg-slate-50 px-3 text-sm"><input type={step.kind === 'map' ? 'radio' : 'checkbox'} name={`input-${step.id}`} checked={step.inputs.includes(s.id)} onChange={e => patch(index, { ...step, inputs: step.kind === 'map' ? [s.id] : e.target.checked ? [...step.inputs, s.id] : step.inputs.filter(id => id !== s.id) })} />{s.name}</label>)}{!candidates.length || unresolved.length ? <p role="alert" className="text-xs text-rose-700">缺少有效前置结果。先添加对应步骤，或调整顺序后重新选择输入。</p> : null}</fieldset>}
        {step.kind === 'resolve' ? <>
          <label className="block text-xs font-semibold">取值业务表<select className={inputClass} value={step.table_id ?? ''} onChange={e => patch(index, { ...step, table_id: e.target.value })}><option value="">请选择</option>{catalog.targets.categories.map(c => <optgroup key={c.category_id} label={c.label}>{catalog.targets.tables.filter(t => t.source_mappable && t.category_id === c.category_id).map(t => <option key={t.table_id} value={t.table_id}>{t.label}</option>)}</optgroup>)}</select></label>
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={step.include_history} onChange={e => patch(index, { ...step, include_history: e.target.checked })} />包含启动时锁定的历史候选</label>
          {step.include_history ? <label className="block text-xs font-semibold">历史数据范围<select className={inputClass} value={step.history_scope ?? 'table'} onChange={event => patch(index, { ...step, history_scope: event.target.value as 'table' | 'matching_inputs' })}><option value="table">本表全部历史候选</option><option value="matching_inputs">仅同接口、同产品及同口径历史</option></select></label> : null}
          <details><summary className="cursor-pointer text-xs font-semibold">取值日期范围与历史时点</summary><div className="mt-3 grid gap-3 sm:grid-cols-3">{(['start_date', 'end_date', 'as_of'] as const).map(key => <label key={key} className="text-xs">{({ start_date: '取值开始日期', end_date: '取值结束日期', as_of: '历史可得截止时点' })[key]}<input className={inputClass} type={key === 'as_of' ? 'text' : 'date'} value={step[key] ?? ''} placeholder={key === 'as_of' ? '带时区的 ISO 时间，可留空' : undefined} onChange={e => patch(index, { ...step, [key]: e.target.value || null })} /></label>)}</div></details>
        </> : null}
        {step.kind === 'snapshot' ? <p className="rounded-lg bg-indigo-50 p-3 text-xs leading-6 text-indigo-900">需要选择已取值的产品信息和基金净值；日行情、交易日历可选。使用本次输入和锁定指标配置计算，不读取旧活跃价格，不自动发布。</p> : null}
      </article>
    })}
    <div className="flex flex-wrap gap-2">{(Object.keys(stepLabels) as EtlKind[]).map(kind => <button type="button" key={kind} className={buttonClass} disabled={definition.steps.length >= 40} onClick={() => add(kind)}>＋{stepLabels[kind]}</button>)}</div>
  </section>
}
