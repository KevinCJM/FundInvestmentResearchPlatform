import type { DataModelCategory, DataModelTable } from '../../services/dataModel'
import type { DatasetMapping, FieldMapping, IdentityBinding, InterfaceConfig } from '../../services/dataSources'
import { buttonClass, inputClass, JsonField, TextField } from './EditorFields'

const operations: [FieldMapping['operation'], string][] = [['copy', '直接取值'], ['scale', '单位换算（乘系数）'], ['constant', '固定值'], ['enum', '枚举转换'], ['date', '日期转换'], ['timestamp', '时间与时区转换'], ['period_end', '月末 / 季末日期'], ['capture_date', '当前元数据采集日期']]
const generated = new Set(['source_id', 'source_batch_id', 'source_record_hash', 'revision', 'ingested_at', 'recorded_at', 'vintage_id'])
const newField = (name: string, operation: FieldMapping['operation'], source: string | null): FieldMapping => ({ target_field: name, source_field: source, operation, factor: 1, constant: null, enum_map: {}, timezone: 'Asia/Shanghai', date_format: null })

function IdentityEditor({ value, fields, sources, onChange }: {
  value: IdentityBinding; fields: DataModelTable['fields']; sources: InterfaceConfig['source_fields']; onChange: (next: IdentityBinding) => void
}) {
  return <div className="space-y-3 rounded-lg border border-slate-200 p-3">
    <div className="grid gap-3 sm:grid-cols-2">
      <label className="text-xs font-semibold text-slate-600">内部身份字段<select className={inputClass} value={value.target_field} onChange={e => onChange({ ...value, target_field: e.target.value })}>{fields.map(f => <option key={f.name} value={f.name}>{f.label} · {f.name}</option>)}</select></label>
      <label className="text-xs font-semibold text-slate-600">身份解析方式<select className={inputClass} value={value.resolution} onChange={e => onChange({ ...value, resolution: e.target.value as IdentityBinding['resolution'] })}><option value="namespace">同一代码空间生成稳定 ID</option><option value="lookup">人工确认的身份对照</option></select></label>
      <TextField label="代码空间" value={value.namespace} required onChange={namespace => onChange({ ...value, namespace })} />
      <label className="text-xs font-semibold text-slate-600">代码标准化<select className={inputClass} value={value.key_transform ?? 'none'} onChange={event => onChange({ ...value, key_transform: event.target.value as IdentityBinding['key_transform'] })}><option value="none">保持原代码</option><option value="cn_etf_code">ETF 六位代码 → 带交易所后缀的代码</option><option value="cn_fund_code">场外基金六位代码 → 带 OF 后缀的代码</option></select></label>
      <label className="text-xs font-semibold text-slate-600">外部身份键<select className={inputClass} value={value.constant !== null ? '__constant' : value.key_fields.length ? '__composite' : value.source_field ?? ''} onChange={e => {
        const selected = e.target.value
        onChange({ ...value, source_field: selected.startsWith('__') ? null : selected, key_fields: selected === '__composite' ? [sources[0]?.name ?? ''] : [], constant: selected === '__constant' ? '' : null })
      }}><option value="">请选择来源字段</option>{sources.map(f => <option key={f.name} value={f.name}>{f.name}</option>)}<option value="__constant">固定代码</option><option value="__composite">多字段组合键</option></select></label>
    </div>
    {value.constant !== null ? <TextField label="固定身份代码" value={value.constant} required onChange={constant => onChange({ ...value, constant })} /> : null}
    {value.key_fields.length > 0 ? <TextField label="组合键字段（逗号分隔）" value={value.key_fields.join(',')} required onChange={text => onChange({ ...value, key_fields: text.split(',').map(s => s.trim()) })} /> : null}
    {value.resolution === 'lookup' ? <JsonField label="身份对照（外部代码 → 已确认内部 ID）" value={value.value_map} onChange={data => onChange({ ...value, value_map: data as Record<string, string> })} /> : <p className="text-xs leading-5 text-amber-800">只有确认同一标识体系后才复用代码空间。姓名相同、供应商代码相同不代表同一实体。</p>}
  </div>
}

function DatasetEditor({ value, table, sources, onChange }: {
  value: DatasetMapping; table: DataModelTable; sources: InterfaceConfig['source_fields']; onChange: (value: DatasetMapping) => void
}) {
  const fields = table.fields.filter(f => f.source_mappable)
  const identityFields = table.fields.filter(f => !f.source_mappable && f.data_type === 'string' && ['primary_key', 'foreign_key'].includes(f.role) && !generated.has(f.name))
  const patch = (name: string, binding: FieldMapping | null) => onChange({ ...value, fields: [...value.fields.filter(f => f.target_field !== name), ...(binding ? [binding] : [])] })
  return <div className="space-y-4">
    <p className="text-xs leading-6 text-slate-500">{table.description} 非空字段须由来源或固定值提供；内部 ID 单独解析，批次与哈希由系统维护。</p>
    <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={value.enabled} onChange={e => onChange({ ...value, enabled: e.target.checked })} />启用这张表的映射</label>
    <div className="overflow-x-auto rounded-lg border border-slate-200"><table className="w-full min-w-[700px] text-left text-xs">
      <thead className="bg-slate-50"><tr><th className="p-3">标准字段</th><th className="p-3">转换方式</th><th className="p-3">来源 / 转换参数</th></tr></thead>
      <tbody>{fields.map(field => {
        const binding = value.fields.find(f => f.target_field === field.name)
        return <tr key={field.name} className="border-t border-slate-100 align-top">
          <td className="p-3"><strong>{field.label}</strong><code className="mt-1 block text-indigo-700">{field.name}</code><p className="mt-1 text-slate-500">{field.data_type} · {field.nullable ? '可空' : '非空'}{field.unit ? ` · ${field.unit}` : ''}</p><p className="mt-1 max-w-xs leading-5 text-slate-500">{field.description}</p></td>
          <td className="p-3"><select className={inputClass} aria-label={`${field.name} 转换方式`} value={binding?.operation ?? ''} onChange={e => patch(field.name, e.target.value ? newField(field.name, e.target.value as FieldMapping['operation'], sources.some(s => s.name === field.name) ? field.name : sources[0]?.name ?? null) : null)}><option value="">暂不映射</option>{operations.filter(([op]) => op !== 'capture_date' || ['valid_from', 'effective_from'].includes(field.name)).map(([op, label]) => <option key={op} value={op}>{label}</option>)}</select></td>
          <td className="space-y-2 p-3">{binding ? <>
            {!['constant', 'capture_date'].includes(binding.operation) ? <select className={inputClass} aria-label={`${field.name} 来源字段`} value={binding.source_field ?? ''} required onChange={e => patch(field.name, { ...binding, source_field: e.target.value })}><option value="">选择来源字段</option>{sources.map(s => <option key={s.name} value={s.name}>{s.name}{s.unit ? ` (${s.unit})` : ''}</option>)}</select> : null}
            {binding.operation === 'constant' ? <JsonField label={`${field.name} 固定值`} value={binding.constant} objectOnly={false} onChange={constant => patch(field.name, { ...binding, constant: constant as FieldMapping['constant'] })} /> : null}
            {binding.operation === 'scale' ? <label className="block">乘数<input className={inputClass} type="number" step="any" required value={binding.factor} onChange={e => patch(field.name, { ...binding, factor: Number(e.target.value) })} /></label> : null}
            {binding.operation === 'enum' ? <JsonField label={`${field.name} 枚举对应`} value={binding.enum_map} onChange={enum_map => patch(field.name, { ...binding, enum_map: enum_map as FieldMapping['enum_map'] })} /> : null}
            {['date', 'timestamp'].includes(binding.operation) ? <TextField label={`${field.name} 日期格式（可留空）`} value={binding.date_format ?? ''} placeholder="%Y%m%d" onChange={date_format => patch(field.name, { ...binding, date_format: date_format || null })} /> : null}
            {binding.operation === 'timestamp' ? <TextField label={`${field.name} 来源时区`} value={binding.timezone} required onChange={timezone => patch(field.name, { ...binding, timezone })} /> : null}
            {binding.operation === 'capture_date' ? <p className="leading-5 text-amber-800">仅标记本次元数据版本起点，不代表历史首次生效或可得日期。</p> : null}
          </> : <span className="text-slate-400">不提供此字段</span>}</td>
        </tr>
      })}</tbody>
    </table></div>
    <details className="rounded-xl border border-slate-200 p-3" open={value.identities.length > 0}>
      <summary className="cursor-pointer text-sm font-semibold">外部代码与内部身份解析 · {value.identities.length} 项</summary>
      <div className="mt-3 space-y-3">{value.identities.map((binding, index) => <div key={index}><IdentityEditor value={binding} fields={identityFields} sources={sources} onChange={next => onChange({ ...value, identities: value.identities.map((item, i) => i === index ? next : item) })} /><button type="button" className="mt-1 text-xs text-rose-700" onClick={() => onChange({ ...value, identities: value.identities.filter((_, i) => i !== index) })}>删除身份解析 {index + 1}</button></div>)}
        <button type="button" className={buttonClass} disabled={!identityFields.some(f => !value.identities.some(i => i.target_field === f.name))} onClick={() => {
          const next = identityFields.find(f => !value.identities.some(i => i.target_field === f.name))
          if (next) onChange({ ...value, identities: [...value.identities, { target_field: next.name, source_field: sources[0]?.name ?? null, key_fields: [], constant: null, namespace: 'CONFIRMED_CODE_SPACE', resolution: 'lookup', value_map: {} }] })
        }}>添加身份解析</button>
      </div>
    </details>
  </div>
}

export default function MappingEditor({ config, tables, categories, version, onChange }: {
  config: InterfaceConfig; tables: DataModelTable[]; categories: DataModelCategory[]; version: string; onChange: (config: InterfaceConfig) => void
}) {
  const targets = tables.filter(t => t.source_mappable && t.usage === 'external_import')
  const targetGroups = [
    ...categories.flatMap(category => {
      const grouped = targets.filter(table => table.category_id === category.category_id)
      return grouped.length ? [{ id: category.category_id, label: category.label, tables: grouped }] : []
    }),
    ...(() => {
      const known = new Set(categories.map(category => category.category_id))
      const other = targets.filter(table => !known.has(table.category_id))
      return other.length ? [{ id: 'other', label: '其他业务数据', tables: other }] : []
    })(),
  ]
  return <section className="space-y-4" aria-label="标准表字段映射">
    {config.mappings.map((mapping, index) => {
      const table = targets.find(t => t.table_id === mapping.target_table)
      return <article key={index} className="space-y-3 rounded-xl border border-slate-200 p-4">
        <div className="flex flex-wrap items-end gap-3"><label className="min-w-0 flex-1 text-xs font-semibold text-slate-600">目标标准表<select aria-label="目标标准表" className={inputClass} value={mapping.target_table} onChange={e => onChange({ ...config, mappings: config.mappings.map((item, i) => i === index ? { target_table: e.target.value, contract_version: version, enabled: true, fields: [], identities: [] } : item) })}>{!table ? <option value={mapping.target_table}>{mapping.target_table}（目标不可用）</option> : null}{targetGroups.map(group => <optgroup key={group.id} label={`${group.label}（${group.tables.length}）`}>{group.tables.map(t => <option key={t.table_id} value={t.table_id}>{t.label} · {t.table_id}</option>)}</optgroup>)}</select></label><button type="button" className={buttonClass} onClick={() => onChange({ ...config, mappings: config.mappings.filter((_, i) => i !== index) })}>删除映射 {index + 1}</button></div>
        <p className="text-xs text-slate-500">合同版本 {mapping.contract_version}</p>
        {table ? <DatasetEditor value={mapping} table={table} sources={config.source_fields} onChange={next => onChange({ ...config, mappings: config.mappings.map((item, i) => i === index ? next : item) })} /> : <p role="alert">只能映射到外部导入表，请重新选择目标。</p>}
      </article>
    })}
    <button type="button" className={buttonClass} disabled={!targets.length || config.mappings.length >= 30} onClick={() => onChange({ ...config, mappings: [...config.mappings, { target_table: targets[0].table_id, contract_version: version, enabled: true, fields: [], identities: [] }] })}>添加目标表映射</button>
  </section>
}
