import { useEffect, useRef, useState } from 'react'
import type { ResolutionConfig, ResolutionPolicyRecord, ResolutionRun, ResolutionTableRule, SourceCatalog } from '../../services/dataSources'
import { getResolutionPolicy, previewResolution, runResolution, saveResolutionPolicy } from '../../services/dataSources'
import { buttonClass, inputClass, primaryClass, validateEditor } from './EditorFields'

const labels: Record<string, string> = {
  SELECTED: '使用首选来源', FALLBACK: '已使用备用来源', WARNING: '按优先级选取并告警', CONFLICT: '数据冲突，等待处理', BLOCKED: '没有合格数据',
  REQUIRED_VALUE_MISSING: '缺少必需值', VALUE_OUT_OF_RANGE: '数值不合法或超出范围', OHLC_INCONSISTENT: '开高低收关系异常', SUSPICIOUS_JUMP: '变动幅度超过阈值',
  PRIMARY_RECORD_MISSING: '该来源无同日同口径记录', SAME_SOURCE_CONFLICT: '同来源同版本不一致', SOURCE_DISABLED: '来源已停用',
  LATEST_SOURCE_BATCH_REJECTED: '最近批次映射失败', TYPE_INVALID: '字段类型错误', NOT_KNOWN_AS_OF: '在指定历史时点尚不可得',
  LEGACY_CANDIDATE_UNVERIFIED: '旧候选缺少校验和，请重新下载；其值不参与选取',
}
const blankRule = (table_id: string): ResolutionTableRule => ({ table_id, source_priority: [], fallback_on_missing: true, fallback_on_invalid: false, conflict_action: 'quarantine', required_fields: [], compare_fields: [], absolute_tolerance: 0.000001, relative_tolerance: 0.0001, max_relative_jump: null, field_rules: [] })

function PriorityEditor({ value, catalog, onChange, label }: { value: string[]; catalog: SourceCatalog; onChange: (value: string[]) => void; label: string }) {
  const move = (index: number, step: number) => { const next = [...value]; [next[index], next[index + step]] = [next[index + step], next[index]]; onChange(next) }
  return <section aria-label={label} className="space-y-2">
    <h3 className="text-sm font-bold">{label}</h3>
    {value.map((id, index) => <div key={id} className="flex flex-wrap items-center gap-2 rounded-xl border border-slate-200 bg-white p-2">
      <span className="mr-auto text-sm"><strong>{index + 1}. {catalog.sources.find(s => s.config.id === id)?.config.name ?? id}</strong>{catalog.sources.find(s => s.config.id === id)?.config.enabled === false ? '（已停用）' : ''}</span>
      <button type="button" className={buttonClass} aria-label={`${label} ${id} 上移`} disabled={index === 0} onClick={() => move(index, -1)}>上移</button>
      <button type="button" className={buttonClass} aria-label={`${label} ${id} 下移`} disabled={index === value.length - 1} onClick={() => move(index, 1)}>下移</button>
      <button type="button" className={buttonClass} aria-label={`${label} ${id} 不参与`} disabled={value.length === 1} onClick={() => onChange(value.filter(s => s !== id))}>不参与</button>
    </div>)}
    {catalog.sources.filter(s => !value.includes(s.config.id)).map(s => <button key={s.config.id} type="button" className={buttonClass} onClick={() => onChange([...value, s.config.id])}>加入 {s.config.name}</button>)}
  </section>
}

function RunResult({ result, catalog }: { result: ResolutionRun; catalog: SourceCatalog }) {
  const sourceName = (id: string | null) => catalog.sources.find(s => s.config.id === id)?.config.name ?? id ?? '未选取'
  return <section className="space-y-3 rounded-xl border border-slate-200 bg-slate-50 p-4" aria-label="多源取值结果">
    <h3 className="font-bold">取值结果 · 未发布</h3>
    <p className="text-sm">选中 {result.summary?.selected_rows ?? 0} 行；备用替代 {result.summary?.FALLBACK ?? 0} 行；冲突 {result.summary?.CONFLICT ?? 0} 行；阻断 {result.summary?.BLOCKED ?? 0} 行。</p>
    {Boolean(result.summary?.unverified_input_batches) ? <p className="text-xs text-amber-800">包含 {result.summary?.unverified_input_batches} 个旧格式批次，其值不会被采用；同业务键的新验证记录可替代旧批次。</p> : null}
    {result.decisions?.slice(0, 30).map((decision, index) => <details key={index} className="rounded-xl border border-slate-200 bg-white p-3"><summary className="cursor-pointer break-words text-sm">{labels[decision.status] ?? decision.status} · {sourceName(decision.selected_source)}</summary>
      <p className="mt-2 break-all text-xs">业务键：{JSON.stringify(decision.key)}</p>
      {decision.selected_batch ? <p className="mt-1 break-all text-xs text-slate-600">采用批次：{decision.selected_batch}</p> : null}
      {decision.skipped.map((item, n) => <p className="mt-1 text-xs" key={n}>{sourceName(item.source_id)}：{item.reasons.map(r => labels[r] ?? r).join('、')}</p>)}
      {decision.conflicts.map((item, n) => <p className="mt-1 break-words text-xs text-amber-900" key={n}>与 {sourceName(item.source_id)} 的 {item.fields.join('、')} 不一致；所选候选 {JSON.stringify(item.selected_values)}，对照 {JSON.stringify(item.other_values)}。</p>)}
    </details>)}
    <p className="text-xs text-slate-600">只展示前 30 条决策。实际合并的完整决策随不可变候选保存；原始来源数据不被覆盖。</p>
  </section>
}

export default function ResolutionPanel({ catalog, onDirty }: { catalog: SourceCatalog; onDirty: (dirty: boolean) => void }) {
  const [saved, setSaved] = useState<ResolutionPolicyRecord | null>(null)
  const [config, setConfig] = useState<ResolutionConfig | null>(null)
  const [selected, setSelected] = useState('market.quote_daily')
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(false)
  const [retry, setRetry] = useState(0)
  const [result, setResult] = useState<ResolutionRun | null>(null)
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [asOf, setAsOf] = useState('')
  const [sample, setSample] = useState('')
  const form = useRef<HTMLFormElement>(null)
  useEffect(() => {
    let cancelled = false
    getResolutionPolicy().then(value => { if (!cancelled) { setSaved(value); setConfig(value.config); setError('') } }).catch(reason => { if (!cancelled) setError(reason instanceof Error ? reason.message : '规则加载失败。') })
    return () => { cancelled = true }
  }, [retry])
  if (!saved || !config) return <section className="rounded-xl bg-white p-5">{error ? <p role="alert">{error}<button type="button" className={buttonClass} onClick={() => setRetry(v => v + 1)}>重新加载规则</button></p> : <p>正在读取多源取值规则…</p>}</section>
  const dirty = JSON.stringify(config) !== JSON.stringify(saved.config)
  const tables = catalog.targets.tables.filter(t => t.source_mappable && t.usage === 'external_import')
  const table = tables.find(t => t.table_id === selected) ?? tables[0]
  const rule = config.tables.find(t => t.table_id === table?.table_id) ?? blankRule(table?.table_id ?? '')
  const fields = table?.fields.filter(f => f.source_mappable) ?? []
  const patch = (next: ResolutionConfig) => { setConfig(next); onDirty(JSON.stringify(next) !== JSON.stringify(saved.config)); setResult(null); setMessage(''); setError('') }
  const patchRule = (next: ResolutionTableRule) => patch({ ...config, tables: [...config.tables.filter(t => t.table_id !== next.table_id), next] })
  const perform = async (operation: () => Promise<void>) => {
    if (busy || !validateEditor(form.current)) return
    setBusy(true); setError(''); setMessage('')
    try { await operation() } catch (reason) { setError(reason instanceof Error ? reason.message : '取值规则操作未完成。') } finally { setBusy(false) }
  }
  return <form ref={form} noValidate className="space-y-5" onSubmit={event => { event.preventDefault(); if (catalog.editing_enabled) void perform(async () => { const value = await saveResolutionPolicy(config, saved.revision); setSaved(value); setConfig(value.config); onDirty(false); setMessage('取值规则已保存；已有研究数据不会被覆盖。') }) }}>
    <header className="rounded-xl border border-slate-200 bg-white p-5"><h2 className="text-xl font-bold">多源优先级与异常处理</h2><p className="mt-2 text-sm leading-6 text-slate-600">先对齐产品、日期、币种与复权口径，再按规则选取整条记录。不混拼开高低收，不用单位净值替代复权净值，也不对冲突值擅自求平均。</p><p className="mt-2 text-xs text-slate-600">规则修订 {saved.revision} · 先保存，再应用到已下载候选。当前正式研究仍使用原活跃数据。</p></header>
    {error ? <p role="alert" className="rounded-lg bg-rose-50 p-3 text-sm text-rose-800">{error}</p> : null}
    {message ? <p role="status" className="rounded-lg bg-emerald-50 p-3 text-sm text-emerald-800">{message}</p> : null}
    <fieldset disabled={!catalog.editing_enabled || busy} className="space-y-5">
      <div className="rounded-xl border border-slate-200 bg-white p-5"><PriorityEditor label="全局来源优先级" value={config.default_source_priority} catalog={catalog} onChange={value => patch({ ...config, default_source_priority: value })} /></div>
      <section className="space-y-4 rounded-xl border border-slate-200 bg-white p-5">
        <label className="block text-sm font-bold">选择业务表<select className={inputClass} value={table?.table_id ?? ''} onChange={e => { setSelected(e.target.value); setResult(null) }}>{catalog.targets.categories.map(category => <optgroup key={category.category_id} label={category.label}>{tables.filter(t => t.category_id === category.category_id).map(t => <option key={t.table_id} value={t.table_id}>{t.label}</option>)}</optgroup>)}</select></label>
        <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={rule.source_priority.length > 0} onChange={e => patchRule({ ...rule, source_priority: e.target.checked ? [...config.default_source_priority] : [] })} />为本表单独设置来源顺序</label>
        {rule.source_priority.length ? <PriorityEditor label="本表来源优先级" value={rule.source_priority} catalog={catalog} onChange={value => patchRule({ ...rule, source_priority: value })} /> : <p className="text-xs text-slate-600">沿用全局来源顺序。</p>}
        <div className="grid gap-4 md:grid-cols-2">
          <label className="flex items-start gap-2 rounded-lg bg-slate-50 p-3 text-sm"><input className="mt-1" type="checkbox" checked={rule.fallback_on_missing} onChange={e => patchRule({ ...rule, fallback_on_missing: e.target.checked })} /><span>主来源缺失时使用下一来源<span className="mt-1 block text-xs text-slate-600">备用记录必须属于同一天、同口径，且满足必需字段。</span></span></label>
          <label className="flex items-start gap-2 rounded-lg bg-slate-50 p-3 text-sm"><input className="mt-1" type="checkbox" checked={rule.fallback_on_invalid} onChange={e => patchRule({ ...rule, fallback_on_invalid: e.target.checked })} /><span>主来源异常时使用下一来源<span className="mt-1 block text-xs text-slate-600">备用记录也要通过质量检查；会保存替代原因。</span></span></label>
        </div>
        <label className="block text-sm font-semibold">双方合法但数值不一致时<select className={inputClass} value={rule.conflict_action} onChange={e => patchRule({ ...rule, conflict_action: e.target.value as typeof rule.conflict_action })}><option value="quarantine">隔离冲突，不自动采用（建议）</option><option value="prefer_priority">采用优先来源，同时记录告警</option></select></label>
        <div className="grid gap-3 sm:grid-cols-3">
          <label className="text-xs font-semibold">绝对差异容差<input className={inputClass} type="number" min="0" step="any" required value={rule.absolute_tolerance} onChange={e => patchRule({ ...rule, absolute_tolerance: Number(e.target.value) })} /></label>
          <label className="text-xs font-semibold">相对差异容差（小数）<input className={inputClass} type="number" min="0" max="1" step="any" required value={rule.relative_tolerance} onChange={e => patchRule({ ...rule, relative_tolerance: Number(e.target.value) })} /></label>
          <label className="text-xs font-semibold">可疑跳变阈值（留空关闭）<input className={inputClass} type="number" min="0.000001" max="100" step="any" value={rule.max_relative_jump ?? ''} onChange={e => patchRule({ ...rule, max_relative_jump: e.target.value === '' ? null : Number(e.target.value) })} /></label>
        </div>
        <p className="text-xs leading-6 text-slate-600">0.01 表示 1%。超过绝对与相对容差中的较大值才算冲突。跳变只代表可疑，分红、拆分也可能造成真实变化。</p>
        <details className="rounded-lg border border-slate-200 p-3"><summary className="cursor-pointer text-sm font-semibold">必需字段、对比字段与值域</summary><p className="mt-2 text-xs leading-6 text-slate-600">要求复权净值时应勾选 adjusted_nav 为必需字段；缺少该值的备用来源不能替代。留空对比字段表示检查全部业务维度/数值字段。</p>
          <div className="mt-3 grid gap-2 md:grid-cols-2">{fields.map(field => <div key={field.name} className="space-y-2 rounded-lg border border-slate-200 p-3 text-xs"><strong>{field.label} · {field.name}</strong><div className="flex gap-4">{(['required_fields', 'compare_fields'] as const).map(kind => <label key={kind} className="flex items-center gap-1"><input type="checkbox" checked={rule[kind].includes(field.name)} onChange={e => patchRule({ ...rule, [kind]: e.target.checked ? [...rule[kind], field.name] : rule[kind].filter(n => n !== field.name) })} />{kind === 'required_fields' ? '必须有值' : '跨来源对比'}</label>)}</div>
            {field.data_type === 'float64' ? <div className="grid grid-cols-2 gap-2">{(['minimum', 'maximum', 'absolute_tolerance', 'relative_tolerance'] as const).map(key => { const existing = rule.field_rules.find(f => f.field === field.name); return <label key={key}>{({ minimum: '最小值', maximum: '最大值', absolute_tolerance: '本字段绝对容差', relative_tolerance: '本字段相对容差' })[key]}<input aria-label={`${field.name} ${key}`} className={inputClass} type="number" step="any" min={key.includes('tolerance') ? 0 : undefined} max={key === 'relative_tolerance' ? 1 : undefined} value={existing?.[key] ?? ''} onChange={e => patchRule({ ...rule, field_rules: [...rule.field_rules.filter(f => f.field !== field.name), { field: field.name, minimum: null, maximum: null, absolute_tolerance: null, relative_tolerance: null, ...existing, [key]: e.target.value === '' ? null : Number(e.target.value) }] })} /></label> })}</div> : null}
          </div>)}</div>
        </details>
        <button type="submit" className={primaryClass} disabled={!dirty}>{busy ? '处理中…' : '保存取值规则'}</button>
      </section>
    </fieldset>
    <section className="space-y-4 rounded-xl border border-slate-200 bg-white p-5"><h3 className="font-bold">检查并合并已下载候选</h3><p className="text-xs leading-6 text-slate-600">不访问外部接口。按已保存规则合并所选表，生成独立候选及审计记录；冲突记录不进入选中结果。</p>
      <div className="grid gap-3 sm:grid-cols-3"><label className="text-xs">候选开始日期<input type="date" className={inputClass} value={start} onChange={e => setStart(e.target.value)} /></label><label className="text-xs">候选结束日期<input type="date" className={inputClass} value={end} onChange={e => setEnd(e.target.value)} /></label><label className="text-xs">历史可得截止时间（可留空）<input className={inputClass} value={asOf} placeholder="2026-09-05T16:00:00+08:00" onChange={e => setAsOf(e.target.value)} /></label></div>
      <button type="button" className={buttonClass} disabled={busy || dirty || !catalog.editing_enabled || !table} onClick={() => void perform(async () => { const value = await runResolution(table.table_id, saved.revision, start, end, asOf); setResult(value); setSaved(current => current ? { ...current, runs: [value, ...current.runs.filter(run => run.run_id !== value.run_id)].slice(0, 20) } : current) })}>应用规则到已下载候选</button>
      {dirty ? <p className="text-xs text-amber-800">有未保存修改，请先保存规则。离线样本可以测试当前草稿。</p> : null}
      <details><summary className="cursor-pointer text-sm">高级：使用标准记录离线试算</summary><label className="mt-3 block text-xs">标准记录 JSON 数组<textarea className={`${inputClass} min-h-32 font-mono text-xs`} value={sample} onChange={e => setSample(e.target.value)} /></label><p className="mt-2 text-xs text-slate-600">每条记录应符合标准表全部必填字段，包括业务键、source_id、revision、source_record_hash、ingested_at；可使用已下载候选记录。最多 1000 条，只预览，不保存或发布。</p><button type="button" className={`${buttonClass} mt-2`} disabled={busy || !sample.trim()} onClick={() => void perform(async () => { const rows: unknown = JSON.parse(sample); if (!Array.isArray(rows)) throw new Error('请填写标准记录数组。'); setResult(await previewResolution(table.table_id, config, rows, asOf)) })}>离线试算当前规则</button></details>
    </section>
    {result ? <RunResult result={result} catalog={catalog} /> : null}
    {saved.runs.length ? <details className="rounded-xl border border-slate-200 bg-white p-4"><summary className="cursor-pointer text-sm">最近取值记录 · {saved.runs.length}</summary>{saved.runs.map((run, i) => <button type="button" key={run.run_id ?? i} className={`${buttonClass} mt-2 block w-full text-left`} onClick={() => setResult(run)}>{catalog.targets.tables.find(t => t.table_id === run.table_id)?.label ?? run.table_id} · {run.created_at} · 未发布</button>)}</details> : null}
  </form>
}
