import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import type { DataModelCatalog } from '../../services/dataModel'
import type { ConfigRecord, InterfaceConfig, MappingPreview, MappingValidation, SourceConfig } from '../../services/dataSources'
import { deleteSourceConfig, previewSourceMapping, sampleSourceInterface, saveInterface, validateSourceMapping } from '../../services/dataSources'
import MappingEditor from './MappingEditor'
import SourceSyncPanel from './SourceSyncPanel'
import ParameterEditor from './ParameterEditor'
import RequestParameterFields from './RequestParameterFields'
import { buttonClass, inputClass, JsonField, PolicyEditor, primaryClass, TextField } from './EditorFields'

const steps = [
  ['endpoint-settings', '1. 数据与接口'],
  ['mapping-settings', '2. 字段对应'],
  ['mapping-validation', '3. 验证与使用'],
] as const
type EditorStep = typeof steps[number][0] | 'limit-settings'

function ValidationReport({ result, targets }: { result: MappingValidation; targets: DataModelCatalog }) {
  const isPreview = 'tables' in result;
  const [showSystemFields, setShowSystemFields] = useState(false);
  const preview = isPreview ? result as MappingPreview : null
  const samplePassed = Boolean(preview && preview.source_rows > 0 && preview.tables.length && preview.tables.every(table => table.accepted_rows > 0 && table.rejected_rows === 0) && result.valid && result.ready)
  const message = !result.valid ? '验证未通过，请按下方原因修正'
    : preview ? !preview.source_rows ? '样本为空，尚未验证实际数据' : samplePassed ? '样本转换通过（仅预览）' : '样本尚未完整通过，请检查映射与拒绝记录'
      : result.ready ? '映射定义已通过验证' : '配置可保存，仍需补齐映射'
  return <section className="space-y-2 rounded-xl border border-slate-200 bg-slate-50 p-4" aria-label="映射验证结果">
    <p role="status" className={`text-sm font-bold ${(preview ? samplePassed : result.valid && result.ready) ? 'text-emerald-800' : 'text-amber-900'}`}>{message}</p>
    {[...result.errors, ...result.warnings].slice(0, 30).map((item, i) => <p key={i} className="text-xs leading-5 text-slate-700">{item.field ? `${item.field}：` : ''}{item.message}</p>)}
    <p className="text-xs leading-5 text-slate-500">定义校验检查配置；样本预览检查字段转换。两者都不会下载完整历史或发布数据。</p>
    {isPreview ? <button type="button" className={buttonClass} aria-pressed={showSystemFields} onClick={() => setShowSystemFields(value => !value)}>{showSystemFields ? '隐藏系统维护字段' : '高级：显示系统维护字段'}</button> : null}
    {isPreview ? (result as MappingPreview).tables.map((table, index) => {
      const definition = targets.tables.find(item => item.table_id === table.table_id);
      const columns = table.columns.filter(name => showSystemFields || !definition || definition.fields.some(field => field.name === name && field.source_mappable));
      return <details key={`${table.table_id}-${index}`} className="min-w-0 rounded-lg border border-slate-200 bg-white p-3" open>
        <summary className="cursor-pointer text-sm font-semibold">{definition?.label ?? table.table_id} · 接受 {table.accepted_rows} 行 / 拒绝 {table.rejected_rows} 行</summary>
        <div className="mt-3 max-h-80 overflow-auto"><table className="text-left text-xs"><caption className="sr-only">{definition?.label ?? table.table_id} 映射预览</caption><thead><tr>{columns.map(name => <th key={name} scope="col" className="whitespace-nowrap border-b px-3 py-2">{definition?.fields.find(field => field.name === name)?.label ?? name}<code className="mt-1 block font-normal text-slate-500">{name}</code></th>)}</tr></thead><tbody>{table.rows.map((row, i) => <tr key={i}>{columns.map(name => <td key={name} className="whitespace-nowrap border-b px-3 py-2">{row[name] == null ? '空值' : typeof row[name] === 'object' ? JSON.stringify(row[name]) : String(row[name])}</td>)}</tr>)}</tbody></table></div>
      </details>
    }) : null}
    {'source_preview' in result ? <details><summary className="cursor-pointer text-xs">查看来源样本（最多 20 行）</summary><pre className="mt-2 max-h-64 overflow-auto text-xs">{JSON.stringify((result as MappingPreview).source_preview, null, 2)}</pre></details> : null}
  </section>
}

export default function InterfaceEditor({ record, source, targets, editingEnabled, credentialConfigured = false, onSaved, onDirty, onBusy }: {
  record: ConfigRecord<InterfaceConfig>; source: SourceConfig; targets: DataModelCatalog; editingEnabled: boolean; credentialConfigured?: boolean; onSaved: (id?: string) => void; onDirty: (dirty: boolean) => void; onBusy?: (busy: boolean) => void
}) {
  const [config, setConfig] = useState(record.config)
  const [sampleText, setSampleText] = useState('')
  const [sampleParams, setSampleParams] = useState<Record<string, unknown>>({})
  const [result, setResult] = useState<MappingValidation | null>(record.validation ?? null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const form = useRef<HTMLFormElement>(null)
  const [step, setStep] = useState<EditorStep>(record.revision ? 'mapping-settings' : 'endpoint-settings')
  const [draftTouched, setDraftTouched] = useState(false)
  const submitting = useRef(false)
  const dirty = draftTouched || JSON.stringify(config) !== JSON.stringify(record.config)
  const needsCredential = source.transport === 'tushare' || source.auth_mode !== 'none'
  const missingCredential = needsCredential && !credentialConfigured
  const requestFields = config.request_fields ?? record.request_fields ?? []
  useEffect(() => () => onBusy?.(false), [onBusy])
  const revealStep = (id: EditorStep) => {
    setStep(id)
    requestAnimationFrame(() => {
      const section = form.current?.querySelector<HTMLElement>(`#${id}`)
      if (section instanceof HTMLDetailsElement) section.open = true
      section?.focus()
    })
  }
  const markDirty = () => { setDraftTouched(true); onDirty(true); setResult(null); setError('') }
  const patch = (next: InterfaceConfig) => { setConfig(next); markDirty() }
  const checkInputs = (includeSample: boolean) => {
    const controls = form.current?.querySelectorAll<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>('input, textarea, select')
    const invalid = controls && Array.from(controls).find(input => input.willValidate && (includeSample || !input.closest('[data-sample-params]')) && !input.checkValidity())
    if (!invalid) return true
    const section = invalid.closest<HTMLElement>('[data-editor-step]')
    if (section) revealStep(section.id as EditorStep)
    let parent = invalid.parentElement
    while (parent && parent !== form.current) { if (parent instanceof HTMLDetailsElement) parent.open = true; parent = parent.parentElement }
    requestAnimationFrame(() => { invalid.focus(); invalid.reportValidity() })
    return false
  }
  const perform = async (operation: () => Promise<void>, validate = true, includeSample = false) => {
    if (submitting.current) return
    if (validate && !checkInputs(includeSample)) { setError('请修正标出的输入后再操作。'); return }
    submitting.current = true; setBusy(true); onBusy?.(true); setError('')
    try { await operation() } catch (reason) { setError(reason instanceof Error ? reason.message : '接口操作未完成。') } finally { submitting.current = false; setBusy(false); onBusy?.(false) }
  }
  return <form ref={form} noValidate className="min-w-0 space-y-4 rounded-2xl border border-slate-200 bg-white p-5" onSubmit={event => {
    event.preventDefault()
    if (!busy && editingEnabled) void perform(async () => { const saved = await saveInterface(config, record.revision); onDirty(false); onSaved(saved.config.id) })
  }}>
    <div className="flex flex-wrap justify-between gap-2"><div><h2 className="text-lg font-bold text-slate-950">{record.revision ? config.name : '新建接口'}</h2><p className="mt-1 text-xs text-slate-500">{source.name} · {record.revision ? `修订 ${record.revision}` : '未保存'}</p></div><button type="submit" disabled={!editingEnabled || busy} className={primaryClass}>{busy ? '处理中…' : '保存接口配置'}</button></div>
    <p className="text-xs leading-5 text-slate-500">{dirty ? '有未保存修改：先保存，再使用真实接口采样。' : record.revision ? '当前显示已保存的配置。修改后保存并重新验证。' : '先配置接口与返回结构，再映射标准字段，最后验证。'}</p>
    <nav aria-label="接口配置步骤" className="grid gap-2 sm:grid-cols-3">{steps.map(([id, label]) => <button type="button" key={id} disabled={busy} aria-current={step === id ? 'step' : undefined} className={`${buttonClass} ${step === id ? 'border-indigo-400 bg-indigo-50 text-indigo-800' : ''}`} onClick={() => revealStep(id)}>{label}</button>)}</nav>
    <button type="button" className="text-sm font-semibold text-slate-600 underline disabled:opacity-40" disabled={busy} aria-pressed={step === 'limit-settings'} onClick={() => revealStep('limit-settings')}>高级：下载限制与分页</button>
    {error ? <p role="alert" className="rounded-lg bg-rose-50 p-3 text-sm text-rose-800">{error}</p> : null}
    <fieldset disabled={busy} onChangeCapture={event => { if (!(event.target as HTMLElement).closest('[data-editor-ui]')) markDirty() }} className="min-w-0 space-y-4">
      <details id="endpoint-settings" data-editor-step hidden={step !== 'endpoint-settings'} tabIndex={-1} open className="scroll-mt-6 rounded-xl border border-slate-200 p-4"><summary className="cursor-pointer text-sm font-bold">数据与接口</summary><fieldset disabled={!editingEnabled} className="mt-4 space-y-4"><p className="text-sm leading-6 text-slate-600">这里决定读取哪一类数据。已有接口通常只需核对名称和参数；不确定的地址、响应路径和权限设置不要随意改动。</p>
        <div className="grid gap-3 sm:grid-cols-2">
          <TextField label="接口 ID" value={config.id} required disabled={record.revision > 0} onChange={id => patch({ ...config, id })} />
          <TextField label="接口名称" value={config.name} required onChange={name => patch({ ...config, name })} />
          {source.transport !== 'http' ? <TextField label="接口 API 名称" value={config.api_name} required onChange={api_name => patch({ ...config, api_name })} /> : null}
          {source.transport !== 'akshare' ? <TextField label="接口相对路径" value={config.path} placeholder="/v1/prices；根路径留空" onChange={path => patch({ ...config, path })} /> : <p className="text-xs leading-5 text-slate-500">SDK 接口名称填写函数名；请求方法由 SDK 执行，网络调用仍受共享限额与超时控制。</p>}
          <label className="text-xs font-semibold text-slate-600">请求方式<select className={inputClass} disabled={source.transport === 'akshare'} value={config.method} onChange={e => patch({ ...config, method: e.target.value as InterfaceConfig['method'] })}><option>GET</option><option>POST</option></select></label>
          <label className="text-xs font-semibold text-slate-600">响应数据格式<select className={inputClass} value={config.response.format} onChange={e => patch({ ...config, response: { ...config.response, format: e.target.value as InterfaceConfig['response']['format'] } })}><option value="json_records">JSON 对象数组</option><option value="json_columns">JSON 字段数组 + 数据行</option><option value="csv">CSV 文本</option></select></label>
          <TextField label="数据数组路径（顶层留空）" value={config.response.records_path} disabled={config.response.format === 'csv'} placeholder="data.items" onChange={records_path => patch({ ...config, response: { ...config.response, records_path } })} />
          <TextField label="字段名数组路径" value={config.response.columns_path} disabled={config.response.format !== 'json_columns'} onChange={columns_path => patch({ ...config, response: { ...config.response, columns_path } })} />
          <TextField label="CSV 分隔符" value={config.response.delimiter} disabled={config.response.format !== 'csv'} onChange={delimiter => patch({ ...config, response: { ...config.response, delimiter } })} />
        </div>
        <div className="flex flex-wrap gap-5 text-sm"><label className="flex items-center gap-2"><input type="checkbox" checked={config.enabled} onChange={e => patch({ ...config, enabled: e.target.checked })} />启用接口</label>{source.transport === 'tushare' ? <label className="flex items-center gap-2"><input type="checkbox" checked={config.entitlement_confirmed} onChange={e => patch({ ...config, entitlement_confirmed: e.target.checked })} />已核实该接口的账户权限</label> : null}</div>
        <ParameterEditor label="默认请求参数" value={config.params} onDraftChange={markDirty} onChange={params => patch({ ...config, params: params as InterfaceConfig['params'] })} />
        <details className="rounded-lg bg-slate-50 p-3"><summary className="cursor-pointer text-sm font-semibold">高级：请求头与原始 JSON</summary><div className="mt-3 space-y-3"><p className="text-xs text-slate-500">不要在这里填写 Token 或密钥；凭据在数据源设置中单独保存。</p><ParameterEditor label="附加请求头" stringOnly value={config.headers} onDraftChange={markDirty} onChange={headers => patch({ ...config, headers: headers as InterfaceConfig['headers'] })} /><JsonField label="默认请求参数（JSON，不含凭据）" value={config.params} onChange={params => patch({ ...config, params: params as InterfaceConfig['params'] })} /><JsonField label="附加请求头（JSON，不含凭据）" value={config.headers} onChange={headers => patch({ ...config, headers: headers as InterfaceConfig['headers'] })} /></div></details>
        <details className="rounded-lg bg-slate-50 p-3"><summary className="cursor-pointer text-sm font-semibold">请求参数表单定义</summary><p className="mt-2 text-xs text-slate-500">下载页按此合同显示参数，不从返回字段猜测代码或日期。每项包含 name、label、data_type，可配置 required、date_format 和 description。</p><JsonField label="请求字段定义 JSON" objectOnly={false} value={config.request_fields ?? record.request_fields ?? []} onChange={value => { if (!Array.isArray(value)) throw new Error('请求字段定义必须是数组。'); patch({ ...config, request_fields: value as NonNullable<InterfaceConfig['request_fields']> }) }} /></details>
        <details className="rounded-lg bg-slate-50 p-3"><summary className="cursor-pointer text-sm font-semibold">来源字段定义 · {config.source_fields.length} 个</summary><div className="mt-3 space-y-2">{config.source_fields.map((field, index) => <div key={index} className="grid gap-2 rounded-lg border border-slate-200 bg-white p-2 sm:grid-cols-[1fr_120px_1fr_auto]">
          <input aria-label={`来源字段 ${index + 1} 名称`} className={inputClass} required value={field.name} onChange={e => patch({ ...config, source_fields: config.source_fields.map((item, i) => i === index ? { ...item, name: e.target.value } : item) })} />
          <select aria-label={`来源字段 ${index + 1} 类型`} className={inputClass} value={field.data_type} onChange={e => patch({ ...config, source_fields: config.source_fields.map((item, i) => i === index ? { ...item, data_type: e.target.value as typeof field.data_type } : item) })}>{['string', 'number', 'integer', 'boolean', 'date', 'datetime', 'json'].map(type => <option key={type}>{type}</option>)}</select>
          <input aria-label={`来源字段 ${index + 1} 单位`} className={inputClass} placeholder="来源单位" value={field.unit} onChange={e => patch({ ...config, source_fields: config.source_fields.map((item, i) => i === index ? { ...item, unit: e.target.value } : item) })} />
          <button type="button" className={buttonClass} onClick={() => patch({ ...config, source_fields: config.source_fields.filter((_, i) => i !== index) })}>删除字段 {index + 1}</button>
        </div>)}<button type="button" className={buttonClass} onClick={() => patch({ ...config, source_fields: [...config.source_fields, { name: '', data_type: 'string', description: '', unit: '' }] })}>添加来源字段</button></div></details>
      </fieldset></details>
      <details id="mapping-settings" data-editor-step hidden={step !== 'mapping-settings'} tabIndex={-1} open className="min-w-0 scroll-mt-6 rounded-xl border border-slate-200 p-4"><summary className="cursor-pointer text-sm font-bold">字段对应：来源里的哪一列，放进系统哪一列</summary><p className="mt-3 text-sm leading-6 text-slate-600">先选择系统数据表，再检查必填字段。名称不同可以手动对应；单位不同要换算，不能只因名称相似就直接取值。</p><div className="mt-4"><MappingEditor config={config} tables={targets.tables} categories={targets.categories} version={targets.schema_version} disabled={!editingEnabled} onChange={patch} /></div></details>
      <details id="limit-settings" data-editor-step hidden={step !== 'limit-settings'} open tabIndex={-1} className="scroll-mt-6 rounded-xl border border-slate-200 p-4"><summary className="cursor-pointer text-sm font-bold">高级：下载限制与分页设置</summary><fieldset disabled={!editingEnabled} className="mt-4 space-y-4">
        <PolicyEditor value={config.policy} onChange={policy => patch({ ...config, policy })} effective={record.effective_policy} />
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          <label className="text-xs font-semibold text-slate-600">分页方式<select className={inputClass} disabled={source.transport === 'akshare'} value={config.pagination.mode} onChange={e => patch({ ...config, pagination: { ...config.pagination, mode: e.target.value as InterfaceConfig['pagination']['mode'], cursor_param: e.target.value === 'page' ? 'page' : 'offset' } })}><option value="none">不分页（达到行数上限即报错）</option><option value="offset">Offset 分页</option><option value="page">页码分页</option></select></label>
          <TextField label="分页游标参数" value={config.pagination.cursor_param} onChange={cursor_param => patch({ ...config, pagination: { ...config.pagination, cursor_param } })} />
          <TextField label="每页行数参数" value={config.pagination.limit_param} onChange={limit_param => patch({ ...config, pagination: { ...config.pagination, limit_param } })} />
          <label className="text-xs font-semibold text-slate-600">每页行数<input className={inputClass} type="number" min={1} max={config.policy.max_rows_per_request} required value={config.pagination.page_size} onChange={e => patch({ ...config, pagination: { ...config.pagination, page_size: Number(e.target.value) } })} /></label>
          <label className="text-xs font-semibold text-slate-600">最大分页数<input className={inputClass} type="number" min={1} max={1000} required value={config.pagination.max_pages} onChange={e => patch({ ...config, pagination: { ...config.pagination, max_pages: Number(e.target.value) } })} /></label>
          <TextField label="增量日期字段" value={config.incremental_field ?? ''} onChange={incremental_field => patch({ ...config, incremental_field: incremental_field || null })} />
          <TextField label="开始日期参数" value={config.start_param} onChange={start_param => patch({ ...config, start_param })} />
          <TextField label="结束日期参数" value={config.end_param} onChange={end_param => patch({ ...config, end_param })} />
        </div>
        <p className="text-xs leading-5 text-slate-500">Tushare 全量 / 增量仍由原下载器按代码和日期分片。本页的接口限频、重试、超时与支持的分页参数会实际生效。</p>
      </fieldset></details>
      <fieldset disabled={!editingEnabled} hidden={step !== 'endpoint-settings'}><TextField label="接口备注与权限说明" value={config.notes} onChange={notes => patch({ ...config, notes })} /></fieldset>
    </fieldset>
    <section id="mapping-validation" data-editor-step hidden={step !== 'mapping-validation'} tabIndex={-1} className="scroll-mt-6 space-y-3 rounded-xl border border-slate-200 p-4" aria-label="验证与预览">
      <h3 className="text-sm font-bold">验证与使用</h3><p className="text-xs leading-5 text-slate-500">离线预览不访问外网、不写入数据。真实采样最多请求一次，不表示已完成全量下载。</p>
      <button type="button" className={buttonClass} disabled={busy} onClick={() => void perform(async () => setResult(await validateSourceMapping(config)))}>校验映射定义</button>
      <label className="block text-xs font-semibold text-slate-600">粘贴来源样本（按接口响应格式）<textarea className={`${inputClass} min-h-32 font-mono text-xs`} value={sampleText} onChange={e => { setSampleText(e.target.value); setResult(null) }} disabled={busy} /></label>
      <button type="button" className={buttonClass} disabled={busy || !sampleText.trim()} onClick={() => void perform(async () => setResult(await previewSourceMapping(config, sampleText)))}>离线映射预览</button>
      <details className="rounded-lg border border-amber-200 p-3"><summary className="cursor-pointer text-sm font-semibold text-amber-900">真实接口采样（消耗配额）</summary><fieldset disabled={busy} className="mt-3 space-y-3" data-sample-params onChangeCapture={() => setResult(null)}>{requestFields.length ? <RequestParameterFields fields={requestFields} values={{ ...config.params, ...sampleParams }} onChange={params => { setSampleParams(params); setResult(null) }} /> : <ParameterEditor label="本次采样参数" value={sampleParams} onDraftChange={() => setResult(null)} onChange={params => { setSampleParams(params); setResult(null) }} />}<details><summary className="cursor-pointer text-xs">高级：采样参数 JSON</summary><JsonField label="本次采样参数（覆盖默认参数）" value={sampleParams} onChange={params => { setSampleParams(params as Record<string, unknown>); setResult(null) }} /></details><p className="text-xs text-slate-500">{dirty ? '先保存上方修改，采样必须使用已保存版本。' : record.revision === 0 ? '请先保存接口配置。' : missingCredential ? '请先在数据源设置中保存认证凭据。' : !source.enabled ? '数据源已停用，请先启用来源。' : !config.enabled ? '接口已停用，请勾选“启用接口”并保存。' : '按接口要求填写代码或日期，避免无条件请求大型历史接口。'} 采样最多发起一次请求，不代表下载完成。</p><button type="button" className={buttonClass} disabled={busy || !editingEnabled || dirty || missingCredential || record.revision === 0 || !config.enabled || !source.enabled} onClick={() => {
        if (window.confirm('本次最多发起 1 次真实请求并消耗数据源配额，结果只作预览。继续？')) void perform(async () => setResult(await sampleSourceInterface(record.config.id, record.revision, sampleParams)), true, true)
      }}>确认并采样一次</button></fieldset></details>
      {result ? <><ValidationReport result={result} targets={targets} />{!result.ready ? <button type="button" className={buttonClass} onClick={() => revealStep('mapping-settings')}>返回字段对应，修正缺项</button> : null}</> : null}
      <div className="rounded-xl bg-indigo-50 p-4"><h4 className="text-sm font-bold">下一步：到下载工作区选择范围</h4><p className="mt-2 text-sm leading-6 text-slate-600">保存只保存配置。下载后还需检查候选数据和发布状态，才能用于正式研究。</p>{record.revision > 0 && !dirty && !busy ? <Link className={`${primaryClass} mt-3 inline-flex`} to={`/settings/data-sources?source=${encodeURIComponent(source.id)}`}>进入下载工作区 →</Link> : <p className="mt-2 text-xs text-amber-900">请先保存接口配置。</p>}</div>
      {record.revision > 0 ? <details className="rounded-xl border border-slate-200 p-3"><summary className="cursor-pointer text-sm text-slate-600">高级：只下载当前接口的数据</summary><SourceSyncPanel record={record} source={source} disabled={busy || !editingEnabled || dirty || missingCredential || !source.enabled || !config.enabled} /></details> : null}
    </section>
    <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 pt-4"><p className="text-xs text-slate-500">{step === 'mapping-validation' ? '定义校验 → 样本验证 → 保存配置 → 下载；各步骤的结果独立。' : '切换步骤不会丢失输入；离开当前接口前请保存。'}</p><button type="button" className={primaryClass} disabled={busy} onClick={() => revealStep(step === 'endpoint-settings' || step === 'mapping-validation' ? 'mapping-settings' : 'mapping-validation')}>{step === 'mapping-validation' ? '上一步：检查字段对应' : step === 'endpoint-settings' ? '下一步：对应系统字段' : '下一步：验证与使用'}</button></div>
    {record.revision > 0 ? <button type="button" className={buttonClass} disabled={busy || !editingEnabled} onClick={() => {
      if (window.confirm('删除此接口配置？已下载数据不会删除。')) void perform(async () => { await deleteSourceConfig('interface', record.config.id, record.revision); onDirty(false); onSaved() }, false)
    }}>删除接口</button> : null}
  </form>
}
