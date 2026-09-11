import { useState } from 'react'
import type { DataModelCategory, DataModelTable } from '../../services/dataModel'
import type { DatasetMapping, FieldMapping, IdentityBinding, InterfaceConfig } from '../../services/dataSources'
import { buttonClass, inputClass, JsonField, TextField } from './EditorFields'
import ParameterEditor from './ParameterEditor'

const operations: [FieldMapping['operation'], string][] = [['copy', '直接取值'], ['scale', '单位换算（乘系数）'], ['constant', '固定值'], ['enum', '枚举转换'], ['date', '日期转换'], ['timestamp', '时间与时区转换'], ['period_end', '月末 / 季末日期'], ['capture_date', '当前元数据采集日期']]
const generated = new Set(['source_id', 'source_batch_id', 'source_record_hash', 'revision', 'ingested_at', 'recorded_at', 'vintage_id'])
const newField = (name: string, operation: FieldMapping['operation'], source: string | null): FieldMapping => ({ target_field: name, source_field: ['constant', 'capture_date'].includes(operation) ? null : source, operation, factor: 1, constant: null, enum_map: {}, timezone: 'Asia/Shanghai', date_format: null })

function ConstantField({ field, value, onChange }: {
  field: DataModelTable['fields'][number]; value: FieldMapping['constant']; onChange: (value: FieldMapping['constant']) => void
}) {
  const label = `${field.name} 固定值`
  if (field.enum_values.length) return <label className="block">固定值<select aria-label={label} className={inputClass} value={String(value ?? '')} required={!field.nullable} onChange={event => onChange(event.target.value || null)}><option value="">选择系统允许的值</option>{field.enum_values.map(item => <option key={String(item)} value={String(item)}>{String(item)}</option>)}</select></label>
  if (field.data_type === 'bool') return <label className="block">固定值<select aria-label={label} className={inputClass} value={value === null ? '' : String(value)} required={!field.nullable} onChange={event => onChange(event.target.value === '' ? null : event.target.value === 'true')}><option value="">未设置</option><option value="true">是</option><option value="false">否</option></select></label>
  if (field.data_type === 'json' || field.data_type.startsWith('list<')) return <JsonField label={label} value={value} objectOnly={false} onChange={item => onChange(item as FieldMapping['constant'])} />
  // Numeric text is intentionally preserved: decimal / int64 constants may exceed JS precision.
  return <label className="block">固定值<input aria-label={label} className={inputClass} type={field.data_type === 'date32' ? 'date' : 'text'} required={!field.nullable} value={String(value ?? '')} onChange={event => onChange(event.target.value === '' ? null : event.target.value)} /><span className="mt-1 block text-slate-500">直接填写，无需 JSON 引号；类型由系统校验。</span></label>
}

function IdentityEditor({ value, fields, sources, onChange }: {
  value: IdentityBinding; fields: DataModelTable['fields']; sources: InterfaceConfig['source_fields']; onChange: (next: IdentityBinding) => void
}) {
  return <div className="space-y-3 rounded-lg border border-slate-200 p-3">
    <div className="grid gap-3 sm:grid-cols-2">
      <label className="text-xs font-semibold text-slate-600">内部身份字段<select className={inputClass} value={value.target_field} onChange={event => onChange({ ...value, target_field: event.target.value })}>{fields.map(field => <option key={field.name} value={field.name}>{field.label} · {field.name}</option>)}</select></label>
      <label className="text-xs font-semibold text-slate-600">身份解析方式<select className={inputClass} value={value.resolution} onChange={event => onChange({ ...value, resolution: event.target.value as IdentityBinding['resolution'] })}><option value="namespace">已确认同一套代码规则，生成稳定 ID</option><option value="lookup">逐项填写已确认的代码对照</option></select></label>
      <TextField label="代码空间" value={value.namespace} placeholder="填写已确认的代码体系名称" required onChange={namespace => onChange({ ...value, namespace })} />
      <label className="text-xs font-semibold text-slate-600">代码标准化<select className={inputClass} value={value.key_transform ?? 'none'} onChange={event => onChange({ ...value, key_transform: event.target.value as IdentityBinding['key_transform'] })}><option value="none">保持原代码</option><option value="cn_etf_code">ETF 六位代码 → 带交易所后缀的代码</option><option value="cn_fund_code">场外基金六位代码 → 带 OF 后缀的代码</option></select></label>
      <label className="text-xs font-semibold text-slate-600">外部身份键<select className={inputClass} required value={value.constant !== null ? '__constant' : value.key_fields.length ? '__composite' : value.source_field ?? ''} onChange={event => {
        const selected = event.target.value
        onChange({ ...value, source_field: selected.startsWith('__') ? null : selected || null, key_fields: selected === '__composite' ? [''] : [], constant: selected === '__constant' ? '' : null })
      }}><option value="">选择代表产品或实体的来源字段</option>{sources.map(field => <option key={field.name} value={field.name}>{field.description || field.name} · {field.name}</option>)}<option value="__constant">固定代码</option><option value="__composite">多字段组合键</option></select></label>
    </div>
    {value.constant !== null ? <TextField label="固定身份代码" value={value.constant} required onChange={constant => onChange({ ...value, constant })} /> : null}
    {value.key_fields.length > 0 ? <TextField label="组合键字段（逗号分隔）" value={value.key_fields.join(',')} required onChange={text => onChange({ ...value, key_fields: text.split(',').map(item => item.trim()) })} /> : null}
    {value.resolution === 'lookup' ? <ParameterEditor label="代码对照" stringOnly value={value.value_map} onDraftChange={() => onChange({ ...value })} onChange={data => onChange({ ...value, value_map: data as Record<string, string> })} /> : <p className="text-xs leading-5 text-amber-800">只有确认同一标识体系后才复用代码空间。姓名相同、供应商代码相同不代表同一实体。</p>}
  </div>
}

function DatasetEditor({ value, table, sources, disabled, onChange }: {
  value: DatasetMapping; table: DataModelTable; sources: InterfaceConfig['source_fields']; disabled: boolean; onChange: (value: DatasetMapping) => void
}) {
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState('essential')
  const fields = table.fields.filter(field => field.source_mappable)
  const identityFields = table.fields.filter(field => !field.source_mappable && field.data_type === 'string' && ['primary_key', 'foreign_key'].includes(field.role) && !generated.has(field.name))
  const missing = fields.filter(field => !field.nullable && !value.fields.some(binding => binding.target_field === field.name))
  const missingIds = identityFields.filter(field => !field.nullable && !value.identities.some(binding => binding.target_field === field.name))
  const keyword = query.trim().toLowerCase()
  const visible = new Set(fields.filter(field => {
    const binding = value.fields.find(item => item.target_field === field.name)
    if (filter === 'essential' && field.nullable && !binding) return false
    if (filter === 'required' && field.nullable) return false
    if (filter === 'unmapped' && binding) return false
    return `${field.label} ${field.name} ${field.description} ${binding?.source_field ?? ''}`.toLowerCase().includes(keyword)
  }).map(field => field.name))
  const patch = (name: string, binding: FieldMapping | null) => onChange({ ...value, fields: [...value.fields.filter(field => field.target_field !== name), ...(binding ? [binding] : [])] })

  return <div className="space-y-4" onInvalidCapture={() => { setFilter('all'); setQuery('') }}>
    <p className="text-sm leading-6 text-slate-600">{table.description}</p>
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg bg-slate-50 p-3">
      <label className="flex items-center gap-2 text-sm"><input type="checkbox" disabled={disabled} checked={value.enabled} onChange={event => onChange({ ...value, enabled: event.target.checked })} />启用这张表的映射</label>
      <span className="text-xs text-slate-600">已配置 {value.fields.length} 个业务字段 · {value.identities.length} 项身份对应</span>
    </div>
    {missing.length || missingIds.length ? <p role="status" className="rounded-lg bg-amber-50 p-3 text-sm leading-6 text-amber-900">还需对应必填项：{[...missing, ...missingIds].map(field => field.label).join('、')}。先补齐，再到第 3 步检查。</p> : <p className="text-xs text-slate-500">必填项已分配；类型、取值和身份对照是否正确，仍以第 3 步的后端校验为准。</p>}
    <div data-editor-ui className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_220px]">
      <label className="text-xs font-semibold text-slate-600">搜索字段<input className={inputClass} aria-label={`${table.label} 搜索字段`} value={query} onChange={event => setQuery(event.target.value)} placeholder="例如：收盘价、日期、close" /></label>
      <label className="text-xs font-semibold text-slate-600">显示哪些字段<select className={inputClass} aria-label={`${table.label} 字段筛选`} value={filter} onChange={event => setFilter(event.target.value)}><option value="essential">必填和已配置字段</option><option value="required">只看必填字段</option><option value="unmapped">只看未配置字段</option><option value="all">全部字段（含可选）</option></select></label>
    </div>
    <fieldset disabled={disabled} className="min-w-0"><div className="overflow-x-auto rounded-lg border border-slate-200"><table className="w-full min-w-[680px] text-left text-xs">
      <caption className="sr-only">{table.label} 来源与系统字段对应</caption>
      <thead className="bg-slate-50"><tr><th scope="col" className="p-3">供应商的哪一列</th><th scope="col" className="p-3">→ 系统存到哪一列</th><th scope="col" className="p-3">需要怎样处理</th></tr></thead>
      <tbody>{fields.map(field => {
        const binding = value.fields.find(item => item.target_field === field.name)
        return <tr key={field.name} hidden={!visible.has(field.name)} className="border-t border-slate-100 align-top">
          <td className="max-w-xs space-y-2 p-3">
            {binding && !['constant', 'capture_date'].includes(binding.operation) ? <select className={inputClass} aria-label={`${field.name} 来源字段`} value={binding.source_field ?? ''} required onChange={event => patch(field.name, { ...binding, source_field: event.target.value || null })}><option value="">请选择，不会自动猜测</option>{sources.map(source => <option key={source.name} value={source.name}>{source.description || source.name} · {source.name}{source.unit ? ` (${source.unit})` : ''}</option>)}</select>
              : <span className="block pt-2 text-slate-500">{binding?.operation === 'constant' ? '不读取来源列，使用固定值' : binding?.operation === 'capture_date' ? '使用本次采集日期' : '先在右侧选择处理方式'}</span>}
            {binding?.source_field ? <code className="block break-all text-slate-500">{binding.source_field}</code> : null}
          </td>
          <td className="p-3"><strong className="text-sm">{field.label}</strong>{!field.nullable ? <span className="ml-2 text-rose-700">必填</span> : <span className="ml-2 text-slate-400">可选</span>}<code className="mt-1 block text-indigo-700">{field.name}</code>{field.unit ? <p className="mt-1 text-slate-500">系统单位：{field.unit}</p> : null}<p className="mt-1 max-w-xs leading-5 text-slate-500">{field.description}</p></td>
          <td className="max-w-xs space-y-2 p-3">
            <select className={inputClass} aria-label={`${field.name} 转换方式`} value={binding?.operation ?? ''} onChange={event => patch(field.name, event.target.value ? newField(field.name, event.target.value as FieldMapping['operation'], binding?.source_field ?? (sources.some(source => source.name === field.name) ? field.name : null)) : null)}><option value="">暂不映射</option>{operations.filter(([operation]) => operation !== 'capture_date' || ['valid_from', 'effective_from'].includes(field.name)).map(([operation, label]) => <option key={operation} value={operation}>{label}</option>)}</select>
            {binding?.operation === 'constant' ? <ConstantField field={field} value={binding.constant} onChange={constant => patch(field.name, { ...binding, constant })} /> : null}
            {binding?.operation === 'scale' ? <label className="block">乘数<input className={inputClass} type="number" step="any" required value={binding.factor} onChange={event => patch(field.name, { ...binding, factor: Number(event.target.value) })} /><span className="mt-1 block text-slate-500">系统值 = 来源值 × 乘数。请确认双方单位。</span></label> : null}
            {binding?.operation === 'enum' ? <ParameterEditor label={`${field.name} 枚举对应`} value={binding.enum_map} onDraftChange={() => onChange({ ...value })} onChange={enumMap => patch(field.name, { ...binding, enum_map: enumMap as FieldMapping['enum_map'] })} /> : null}
            {binding && ['date', 'timestamp'].includes(binding.operation) ? <TextField label={`${field.name} 日期格式（可留空）`} value={binding.date_format ?? ''} placeholder="例如 %Y%m%d 对应 20260908" onChange={dateFormat => patch(field.name, { ...binding, date_format: dateFormat || null })} /> : null}
            {binding?.operation === 'timestamp' ? <TextField label={`${field.name} 来源时区`} value={binding.timezone} required onChange={timezone => patch(field.name, { ...binding, timezone })} /> : null}
            {binding?.operation === 'capture_date' ? <p className="leading-5 text-amber-800">只标记本次元数据版本起点，不代表历史首次生效或可得日期。</p> : null}
          </td>
        </tr>
      })}</tbody>
    </table></div></fieldset>
    {!visible.size ? <p className="text-sm text-slate-500">当前筛选下没有字段。<button type="button" className="ml-2 font-semibold text-indigo-700 underline" onClick={() => { setFilter('all'); setQuery('') }}>显示全部字段</button></p> : null}
    <details className="rounded-xl border border-slate-200 p-3" open={missingIds.length > 0 || value.identities.length > 0}>
      <summary className="cursor-pointer text-sm font-semibold">产品 / 实体代码对应 · {value.identities.length} 项{missingIds.length ? `，还缺 ${missingIds.length} 项必填` : ''}</summary>
      <p className="mt-3 text-sm leading-6 text-slate-600">这一步告诉系统“这条记录属于哪个产品或实体”。不要把供应商代码直接当成内部 ID；先确认代码体系或逐项对照。</p>
      <fieldset disabled={disabled} className="mt-3 space-y-3">{value.identities.map((binding, index) => <div key={index}><IdentityEditor value={binding} fields={identityFields.filter(field => field.name === binding.target_field || !value.identities.some(item => item.target_field === field.name))} sources={sources} onChange={next => onChange({ ...value, identities: value.identities.map((item, i) => i === index ? next : item) })} /><button type="button" className="mt-2 text-xs text-rose-700 underline" onClick={() => { if (window.confirm('删除这项代码对应？必填身份缺失时无法用于正式导入。')) onChange({ ...value, identities: value.identities.filter((_, i) => i !== index) }) }}>删除身份解析 {index + 1}</button></div>)}
        <button type="button" className={buttonClass} disabled={!identityFields.some(field => !value.identities.some(item => item.target_field === field.name))} onClick={() => {
          const next = identityFields.find(field => !value.identities.some(item => item.target_field === field.name))
          if (next) onChange({ ...value, identities: [...value.identities, { target_field: next.name, source_field: null, key_fields: [], constant: null, namespace: '', resolution: 'lookup', value_map: {} }] })
        }}>添加身份解析</button>
      </fieldset>
    </details>
    <p className="text-xs leading-5 text-slate-500">批次、哈希、修订号等审计字段由系统自动填写。这里的数量只表示是否已分配，不代替后端校验。</p>
  </div>
}

export default function MappingEditor({ config, tables, categories, version, disabled = false, onChange }: {
  config: InterfaceConfig; tables: DataModelTable[]; categories: DataModelCategory[]; version: string; disabled?: boolean; onChange: (config: InterfaceConfig) => void
}) {
  const targets = tables.filter(table => table.source_mappable && table.usage === 'external_import')
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
  const nextTarget = targets.find(table => !config.mappings.some(mapping => mapping.target_table === table.table_id))
  return <section className="space-y-4" aria-label="标准表字段映射">
    {!config.mappings.length ? <p className="rounded-xl bg-indigo-50 p-4 text-sm leading-6 text-indigo-900">还没有选择数据存放位置。点击“添加目标表映射”，选择日行情、基金净值等系统数据表，再逐列对应。</p> : null}
    {config.mappings.map((mapping, index) => {
      const table = targets.find(item => item.table_id === mapping.target_table)
      return <article key={index} className="space-y-3 rounded-xl border border-slate-200 p-4">
        <div className="flex flex-wrap items-end gap-3"><label className="min-w-0 flex-1 text-xs font-semibold text-slate-600">目标标准表<select aria-label="目标标准表" disabled={disabled} className={inputClass} value={mapping.target_table} onChange={event => {
          if (event.target.value === mapping.target_table) return
          if ((mapping.fields.length || mapping.identities.length) && !window.confirm('更换目标表会清空这张表已配置的字段与身份对应。继续更换？')) return
          onChange({ ...config, mappings: config.mappings.map((item, i) => i === index ? { target_table: event.target.value, contract_version: version, enabled: true, fields: [], identities: [] } : item) })
        }}>{!table ? <option value={mapping.target_table}>{mapping.target_table}（目标不可用）</option> : null}{targetGroups.map(group => <optgroup key={group.id} label={`${group.label}（${group.tables.length}）`}>{group.tables.map(item => <option key={item.table_id} value={item.table_id}>{item.label} · {item.table_id}</option>)}</optgroup>)}</select></label><button type="button" disabled={disabled} className={buttonClass} onClick={() => { if (window.confirm('删除这张表的字段与身份对应？已下载的数据不会删除。')) onChange({ ...config, mappings: config.mappings.filter((_, i) => i !== index) }) }}>删除映射 {index + 1}</button></div>
        <p className="text-xs text-slate-500">标准表版本 {mapping.contract_version}</p>
        {table ? <DatasetEditor key={table.table_id} value={mapping} table={table} sources={config.source_fields} disabled={disabled} onChange={next => onChange({ ...config, mappings: config.mappings.map((item, i) => i === index ? next : item) })} /> : <p role="alert">只能映射到外部导入表，请重新选择目标。</p>}
      </article>
    })}
    <button type="button" className={buttonClass} disabled={disabled || !nextTarget || config.mappings.length >= 30} onClick={() => { if (nextTarget) onChange({ ...config, mappings: [...config.mappings, { target_table: nextTarget.table_id, contract_version: version, enabled: true, fields: [], identities: [] }] }) }}>添加目标表映射</button>
  </section>
}
