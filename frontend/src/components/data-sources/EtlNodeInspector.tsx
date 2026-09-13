import type { SourceCatalog } from '../../services/dataSources'
import type { EtlDefinition, EtlStep } from '../../services/etl'
import type { CanvasConnection, GraphNodeSchema } from '../computation-graph/types'
import { buttonClass, inputClass } from './EditorFields'
import EtlDownloadFields from './EtlDownloadFields'
import EtlTaskFields from './EtlTaskFields'
import { connectionProblem, etlEdges } from './etlGraphAdapter'

export default function EtlNodeInspector({ step, definition, catalog, schemas, onPatch, onConnect, onDisconnect, onRemove, onClose, readOnly }: {
  step: EtlStep; definition: EtlDefinition; catalog: SourceCatalog; schemas: GraphNodeSchema[]; readOnly: boolean
  onPatch: (step: EtlStep) => void; onConnect: (edge: CanvasConnection) => void
  onDisconnect: (id: string) => void; onRemove: () => void; onClose: () => void
}) {
  const schema = schemas.find(item => item.id === step.kind)
  const edges = etlEdges(definition).filter(edge => edge.target === step.id)
  return <aside aria-label="ETL 节点检查器" className="min-w-0 space-y-4 rounded-xl border border-accent-200 bg-white p-4 shadow-lg">
    <div className="flex items-center justify-between gap-2"><h3 className="font-bold">节点设置</h3><button type="button" className={buttonClass} onClick={onClose}>收起检查器</button></div>
    <fieldset disabled={readOnly} className="min-w-0 space-y-4">
      <label className="block text-xs font-semibold">步骤名称<input className={inputClass} maxLength={100} value={step.name} onChange={e => onPatch({ ...step, name: e.target.value })} /></label>
      <p className="break-all text-xs text-slate-600">{schema?.label} · {step.id}</p>
      {step.kind === 'download' ? <EtlDownloadFields step={step} catalog={catalog} onChange={onPatch} /> : null}
      {step.kind === 'task' ? <EtlTaskFields step={step} catalog={catalog} parameters={definition.parameters ?? []} onChange={onPatch} /> : null}
      <section className="space-y-3" aria-label="节点连接"><h4 className="text-sm font-semibold">输入与执行依赖</h4>
        {(schema?.inputs ?? []).map(port => {
          const connections = edges.filter(edge => edge.targetPort === port.id)
          return <div key={port.id} className="space-y-2 rounded-lg border border-slate-200 p-3">
            <p className="text-xs font-bold">{port.label}{port.required ? '（必需）' : '（可选）'}</p>
            {connections.map(edge => {
              const name = definition.steps.find(s => s.id === edge.source)?.name ?? edge.source
              const dataPort = schema?.inputs.find(p => p.id === 'data')
              const dataAllowed = Boolean(dataPort && (dataPort.multiple || !step.inputs.some(id => id !== edge.source)) && schemas.find(s => s.id === definition.steps.find(s => s.id === edge.source)?.kind)?.outputs.find(p => p.id === 'data')?.value_type === dataPort.value_type)
              return <div key={edge.id} className="space-y-2 text-xs"><div className="flex items-start justify-between gap-2"><span>{name}</span><button type="button" className="shrink-0 text-rose-700 underline" aria-label={`断开${port.label}：${name}`} onClick={() => onDisconnect(edge.id)}>断开</button></div>
                <label className="block">依赖关系<select aria-label={`与${name}的依赖关系`} className={inputClass} value={edge.kind} onChange={event => {
                  const inputs = step.inputs.filter(id => id !== edge.source), after = (step.after ?? []).filter(id => id !== edge.source)
                  if (event.target.value === 'data') inputs.push(edge.source); else after.push(edge.source)
                  onPatch({ ...step, inputs, after })
                }}><option value="data" disabled={!dataAllowed}>必须依赖其数据（失败时阻断）</option><option value="control">仅执行顺序（失败后仍继续）</option></select></label>
              </div>
            })}
            {port.required && !connections.length ? <p className="text-xs text-amber-800">尚未连接，运行前需要补齐。</p> : null}
            <label className="block text-xs">添加{port.label}<select aria-label={`添加${port.label}`} className={inputClass} value="" disabled={!port.multiple && connections.length > 0} onChange={e => { if (e.target.value) onConnect({ source: e.target.value, sourcePort: port.id === 'after' ? 'done' : 'data', target: step.id, targetPort: port.id }) }}>
              <option value="">选择上游节点</option>
              {definition.steps.filter(s => s.id !== step.id).map(source => {
                const connection = { source: source.id, sourcePort: port.id === 'after' ? 'done' : 'data', target: step.id, targetPort: port.id }
                const problem = connectionProblem(definition, connection, schemas)
                return <option key={source.id} value={source.id} disabled={Boolean(problem)} title={problem ?? undefined}>{source.name}{problem ? '（不可连接）' : ''}</option>
              })}
            </select></label>
          </div>
        })}
        <p className="text-xs leading-5 text-slate-600">仅执行顺序不会读取上游输出。改为顺序后如缺少必需数据，运行前校验会明确提示；不能用此选项绕过数据完整性检查。</p>
      </section>
      {step.kind === 'resolve' ? <>
        <label className="block text-xs font-semibold">取值业务表<select className={inputClass} value={step.table_id ?? ''} onChange={e => onPatch({ ...step, table_id: e.target.value })}><option value="">请选择</option>{catalog.targets.categories.map(category => <optgroup key={category.category_id} label={category.label}>{catalog.targets.tables.filter(t => t.source_mappable && t.category_id === category.category_id).map(t => <option key={t.table_id} value={t.table_id}>{t.label}</option>)}</optgroup>)}</select></label>
        <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={step.include_history} onChange={e => onPatch({ ...step, include_history: e.target.checked })} />包含启动时锁定的历史候选</label>
        {step.include_history ? <label className="block text-xs">历史数据范围<select className={inputClass} value={step.history_scope ?? 'table'} onChange={e => onPatch({ ...step, history_scope: e.target.value as 'table' | 'matching_inputs' })}><option value="table">本表全部历史候选</option><option value="matching_inputs">仅同接口、同产品及同口径历史</option></select></label> : null}
        <details><summary className="cursor-pointer text-xs">取值日期范围与历史时点</summary><div className="mt-3 space-y-2">{(['start_date', 'end_date', 'as_of'] as const).map(key => <label key={key} className="block text-xs">{({ start_date: '取值开始日期', end_date: '取值结束日期', as_of: '历史可得截止时点' })[key]}<input className={inputClass} type={key === 'as_of' ? 'text' : 'date'} value={step[key] ?? ''} onChange={e => onPatch({ ...step, [key]: e.target.value || null })} /></label>)}</div></details>
      </> : null}
      {step.kind === 'map' ? <p className="text-xs leading-6 text-slate-600">使用上游下载节点冻结的字段映射，不再次访问数据源。需要修改映射时进入数据源与接口映射中心。</p> : null}
      {step.kind === 'snapshot' ? <p className="rounded-lg bg-accent-50 p-3 text-xs leading-6 text-accent-900">需要已取值的产品信息和基金净值；行情与日历可选。快照只使用连接的数据，不读取旧活跃价格、不自动发布。</p> : null}
      {step.kind !== 'download' ? <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={step.allow_empty} onChange={e => onPatch({ ...step, allow_empty: e.target.checked })} />允许空结果后继续</label> : null}
      <button type="button" className={`${buttonClass} text-rose-700`} onClick={onRemove}>删除此节点</button>
    </fieldset>
  </aside>
}
