import { Link } from 'react-router-dom'
import type { SourceCatalog } from '../../services/dataSources'
import { downloadStep, type EtlStep } from '../../services/etl'
import { inputClass, JsonField } from './EditorFields'
import RequestParameterFields from './RequestParameterFields'

export default function EtlDownloadFields({ step, catalog, onChange, chooseInterface = true }: {
  step: EtlStep; catalog: SourceCatalog; onChange: (step: EtlStep) => void; chooseInterface?: boolean
}) {
  const interfaces = catalog.interfaces.filter(i => i.config.source_id === step.source_id)
  const record = interfaces.find(i => i.config.id === step.interface_id)
  const fields = record?.config.request_fields ?? record?.request_fields ?? []
  return <div className="min-w-0 space-y-3">
    {chooseInterface ? <div className="grid gap-3 sm:grid-cols-2">
      <label className="text-xs font-semibold">数据源<select aria-label="数据源" className={inputClass} value={step.source_id ?? ''} onChange={e => onChange({ ...step, source_id: e.target.value, interface_id: null, interface_revision: null, params: {}, parameter_bindings: {} })}><option value="">选择来源</option>{catalog.sources.map(s => <option key={s.config.id} value={s.config.id}>{s.config.name}{s.config.enabled ? '' : '（已停用）'}</option>)}</select></label>
      <label className="text-xs font-semibold">下载数据<select aria-label="下载数据" className={inputClass} value={step.interface_id ?? ''} onChange={e => { const item = interfaces.find(i => i.config.id === e.target.value); if (item) onChange({ ...downloadStep(item, step.mode), id: step.id }) }}><option value="">选择接口支持的数据</option>{interfaces.map(i => <option key={i.config.id} value={i.config.id} disabled={!i.config.enabled || !i.validation?.ready}>{i.config.name}{!i.config.enabled ? '（未启用）' : !i.validation?.ready ? '（映射待完善）' : ''}</option>)}</select></label>
    </div> : null}
    {chooseInterface ? <label className="block text-xs font-semibold">更新方式<select aria-label="更新方式" className={inputClass} value={step.mode} onChange={e => onChange({ ...step, mode: e.target.value as EtlStep['mode'] })}><option value="inherit">跟随本次运行模式（默认）</option><option value="incremental">固定增量：仅特殊步骤使用</option><option value="full">每次全量刷新：适合基础信息</option></select></label> : null}
    <RequestParameterFields fields={fields} values={step.params} bindings={step.parameter_bindings} onChange={params => onChange({ ...step, params })} />
    {!fields.length && record ? <p className="text-xs text-slate-500">此接口尚未声明请求字段；可在下方填写参数，或在接口配置中补充请求字段定义。</p> : null}
    {record ? <p className="text-xs leading-5 text-slate-500">接口修订 {step.interface_revision} · 目标：{record.config.mappings.filter(m => m.enabled).map(m => catalog.targets.tables.find(t => t.table_id === m.target_table)?.label ?? m.target_table).join('、')}。<Link className="text-indigo-700 underline" to={`/settings/source-center?source=${step.source_id}&interface=${step.interface_id}`}>配置来源与映射</Link></p> : null}
    <details className="rounded-lg border border-slate-200 p-3"><summary className="cursor-pointer text-xs font-semibold">其他参数与空结果处理</summary><div className="mt-3 space-y-3"><JsonField label="本步骤请求参数" value={step.params} onChange={params => onChange({ ...step, params: params as Record<string, unknown> })} />{chooseInterface ? <JsonField label="运行参数绑定（接口字段 → 参数 ID）" value={step.parameter_bindings ?? {}} onChange={value => onChange({ ...step, parameter_bindings: value as Record<string, string> })} /> : null}<label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={step.allow_empty} onChange={e => onChange({ ...step, allow_empty: e.target.checked })} />允许来源返回空结果后继续（默认停止）</label></div></details>
  </div>
}
