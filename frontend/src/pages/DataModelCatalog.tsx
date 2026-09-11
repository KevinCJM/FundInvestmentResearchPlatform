import { useEffect, useMemo, useState } from 'react'
import {
  fetchDataModelCatalog,
  type DataModelCatalog,
  type DataModelField,
  type DataModelTable,
  type DataModelScope,
} from '../services/dataModel'

const layerLabels: Record<DataModelTable['layer'], string> = {
  control: '控制面',
  master: '主数据',
  canonical: '标准事实',
  mart: '研究派生',
}

const layerStyles: Record<DataModelTable['layer'], string> = {
  control: 'bg-slate-100 text-slate-700',
  master: 'bg-indigo-100 text-indigo-800',
  canonical: 'bg-emerald-100 text-emerald-800',
  mart: 'bg-violet-100 text-violet-800',
}

const phaseLabels: Record<DataModelTable['delivery_phase'], string> = {
  core: '核心模型',
  next: '下一阶段',
}

const updateLabels: Record<string, string> = {
  append: '追加历史',
  upsert: '更新合并',
  snapshot: '版本快照',
  scd2: '保留历史版本',
  derived: '系统派生',
}

const roleLabels: Record<string, string> = {
  primary_key: '主键',
  foreign_key: '外键',
  dimension: '维度',
  measure: '数值',
  observation_time: '观测时间',
  available_time: '可得时间',
  effective_time: '生效时间',
  audit: '审计',
  configuration: '配置',
}

const textList = (items: string[]) => items.length ? items.join('、') : '无'

function SummaryCard({ label, value, description }: { label: string; value: number; description: string }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-xs font-semibold text-slate-500">{label}</p>
      <p className="mt-2 text-2xl font-bold tabular-nums text-slate-950">{value}</p>
      <p className="mt-1 text-xs leading-5 text-slate-500">{description}</p>
    </div>
  )
}

function TableBadge({ table }: { table: DataModelTable }) {
  return (
    <div className="flex flex-wrap gap-2">
      <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${layerStyles[table.layer]}`}>
        {layerLabels[table.layer]}
      </span>
      <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-700">
        {table.storage_engine === 'parquet' ? 'Parquet' : 'SQLite'}
      </span>
      <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${table.delivery_phase === 'core' ? 'bg-sky-100 text-sky-800' : 'bg-amber-100 text-amber-900'}`}>
        {phaseLabels[table.delivery_phase]}
      </span>
      {table.pit_supported ? (
        <span className="rounded-full bg-cyan-100 px-2.5 py-1 text-xs font-semibold text-cyan-800">支持 PIT</span>
      ) : null}
      <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${table.source_mappable ? 'bg-fuchsia-100 text-fuchsia-800' : 'bg-slate-100 text-slate-500'}`}>
        {table.source_mappable ? '外部导入表' : '系统内部表 · 不接受外部导入'}
      </span>
    </div>
  )
}

function FieldRow({ field }: { field: DataModelField }) {
  return (
    <tr data-field-kind={field.source_mappable ? 'source' : 'system'} className={`border-t border-slate-200 align-top ${field.source_mappable ? 'bg-white' : 'bg-slate-100 text-slate-600'}`}>
      <td className="px-4 py-3">
        <p className="font-semibold text-slate-900">{field.label}{!field.source_mappable ? <span className="ml-2 inline-block rounded border border-slate-300 bg-slate-200 px-2 py-0.5 text-xs font-medium text-slate-700">系统维护</span> : null}</p>
        <code className={`mt-1 block break-all text-xs ${field.source_mappable ? 'text-indigo-700' : 'text-slate-600'}`}>{field.name}</code>
      </td>
      <td className="px-4 py-3">
        <code className="break-all text-xs text-slate-700">{field.data_type}</code>
        {field.unit ? <p className="mt-1 text-xs text-slate-500">单位：{field.unit}</p> : null}
      </td>
      <td className="px-4 py-3 text-xs text-slate-600">
        <span className="font-semibold text-slate-800">{roleLabels[field.role] ?? field.role}</span>
        <p className="mt-1">{field.nullable ? '允许空值' : field.source_mappable ? '必填' : '系统必填'}</p>
        <p className="mt-1">{field.source_mappable ? '可配置来源字段映射' : '系统维护，不需来源映射'}</p>
      </td>
      <td className="px-4 py-3 text-sm leading-6 text-slate-600">
        <p>{field.description}</p>
        {field.reference ? <p className="mt-1 break-all text-xs text-slate-500">引用：{field.reference}</p> : null}
        {field.enum_values.length ? <p className="mt-1 break-all text-xs text-slate-500">枚举：{field.enum_values.join(' / ')}</p> : null}
      </td>
    </tr>
  )
}

function TableDetail({ table }: { table: DataModelTable }) {
  const [showSystemFields, setShowSystemFields] = useState(false)
  const importFields = table.fields.filter((field) => field.source_mappable)
  const systemFields = table.fields.filter((field) => !field.source_mappable)
  const fields = table.source_mappable && !showSystemFields ? importFields : table.fields

  return (
    <article className="min-w-0 rounded-2xl border border-slate-200 bg-white shadow-sm">
      <header className="border-b border-slate-100 p-5 sm:p-6">
        <TableBadge table={table} />
        <h2 className="mt-4 text-xl font-bold text-slate-950">{table.label}</h2>
        <code className="mt-1 block break-all text-sm font-semibold text-indigo-700">{table.table_id}</code>
        <p className="mt-3 text-sm leading-6 text-slate-600">{table.description}</p>
      </header>

      <section className="grid gap-3 border-b border-slate-100 p-5 sm:grid-cols-2 xl:grid-cols-3 sm:p-6" aria-label="表结构信息">
        {[
          ['数据粒度', table.grain],
          ['主键', textList(table.primary_key)],
          ['更新方式', updateLabels[table.update_strategy] ?? table.update_strategy],
          ['物理位置', table.storage_location],
          ['分区字段', textList(table.partition_by)],
          ['排序字段', textList(table.sort_by)],
        ].map(([label, value]) => (
          <div key={label} className="min-w-0 rounded-xl bg-slate-50 p-3">
            <p className="text-xs font-semibold text-slate-500">{label}</p>
            <p className="mt-1 break-words text-sm font-medium leading-6 text-slate-800">{value}</p>
          </div>
        ))}
      </section>

      <section className="p-5 sm:p-6" aria-labelledby="field-list-title">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h3 id="field-list-title" className="text-base font-bold text-slate-950">{table.source_mappable ? '导入字段' : '系统内部字段'}</h3>
            <p className="mt-1 text-xs leading-5 text-slate-500">
              {table.source_mappable
                ? `可映射 ${importFields.length} 个字段；另有 ${systemFields.length} 个字段由系统生成或解析，无需用户填写。`
                : '以下字段由系统配置、任务或计算流程维护，不是数据源字段映射的目标。'}
            </p>
          </div>
        </div>
        {table.source_mappable && systemFields.length > 0 ? (
          <button
            type="button"
            aria-expanded={showSystemFields}
            aria-controls="data-model-fields"
            onClick={() => setShowSystemFields((current) => !current)}
            className="mt-3 rounded-lg border border-slate-200 px-3 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            {showSystemFields ? '收起系统维护字段' : '高级：查看系统维护字段'}
          </button>
        ) : null}
        <p className="mt-3 flex flex-wrap gap-4 text-xs text-slate-600"><span><span aria-hidden="true" className="mr-1 inline-block h-3 w-3 rounded border border-slate-300 bg-white" />白色：来源业务字段</span><span><span aria-hidden="true" className="mr-1 inline-block h-3 w-3 rounded border border-slate-300 bg-slate-200" />灰蓝色：系统维护字段</span></p>
        <div id="data-model-fields" className="mt-4 overflow-x-auto rounded-xl border border-slate-200">
          <table className="min-w-[760px] w-full border-collapse text-left">
            <thead className="bg-slate-50 text-xs font-semibold text-slate-600">
              <tr>
                <th className="px-4 py-3">字段</th>
                <th className="px-4 py-3">类型与单位</th>
                <th className="px-4 py-3">规则</th>
                <th className="px-4 py-3">说明</th>
              </tr>
            </thead>
            <tbody>
              {fields.map((field) => <FieldRow key={field.name} field={field} />)}
            </tbody>
          </table>
        </div>
      </section>
    </article>
  )
}

export default function DataModelCatalog() {
  const [scope, setScope] = useState<DataModelScope>('external')

  return (
    <div className="space-y-5">
      <header className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <h1 className="text-2xl font-bold text-slate-950">系统数据模型</h1>
        <p className="mt-2 text-sm leading-6 text-slate-600">这里查看可从外部导入的业务数据：产品、机构、行情、净值、持仓等。字段映射和代码对照通过配置管理，内部表默认隐藏。</p>
        <div className="mt-4 flex flex-wrap items-center gap-3" aria-label="数据字典范围">
          <button
            type="button"
            aria-pressed={scope === 'external'}
            onClick={() => setScope('external')}
            className={`min-h-11 rounded-xl px-4 py-2 text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-indigo-500 ${scope === 'external' ? 'bg-indigo-700 text-white' : 'bg-slate-100 text-slate-700'}`}
          >
            外部导入表
          </button>
          <button
            type="button"
            aria-pressed={scope === 'internal'}
            onClick={() => setScope(scope === 'internal' ? 'external' : 'internal')}
            className={`min-h-11 rounded-xl border px-4 py-2 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-indigo-500 ${scope === 'internal' ? 'border-slate-700 bg-slate-800 text-white' : 'border-slate-200 text-slate-500 hover:bg-slate-50'}`}
          >
            {scope === 'internal' ? '收起系统内部表' : '高级：系统内部表'}
          </button>
        </div>
      </header>
      {/* Switching scope resets search/selection and aborts the old request. */}
      <CatalogContent key={scope} scope={scope} />
    </div>
  )
}

function CatalogContent({ scope }: { scope: DataModelScope }) {
  const [catalog, setCatalog] = useState<DataModelCatalog | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [retryCount, setRetryCount] = useState(0)
  const [categoryId, setCategoryId] = useState('all')
  const [phase, setPhase] = useState<'all' | DataModelTable['delivery_phase']>('all')
  const [query, setQuery] = useState('')
  const [selectedTableId, setSelectedTableId] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    fetchDataModelCatalog(controller.signal, scope)
      .then((payload) => {
        if (controller.signal.aborted) return
        setCatalog(payload)
        setSelectedTableId((current) => current ?? payload.tables[0]?.table_id ?? null)
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted || (reason instanceof DOMException && reason.name === 'AbortError')) return
        setError(reason instanceof Error ? reason.message : '无法读取系统数据模型。')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [scope, retryCount])

  const keyword = query.trim().toLocaleLowerCase('zh-CN')
  const eligibleTables = useMemo(() => {
    if (!catalog) return []
    return catalog.tables.filter((table) => {
      if (table.source_mappable !== (scope === 'external')) return false
      if (phase !== 'all' && table.delivery_phase !== phase) return false
      if (!keyword) return true
      const haystack = [
        table.table_id,
        table.label,
        table.description,
        table.grain,
        ...table.fields
          .filter((field) => scope === 'internal' || field.source_mappable)
          .flatMap((field) => [field.name, field.label, field.description]),
      ].join(' ').toLocaleLowerCase('zh-CN')
      return haystack.includes(keyword)
    })
  }, [catalog, keyword, phase, scope])
  const filteredTables = useMemo(
    () => categoryId === 'all' ? eligibleTables : eligibleTables.filter((table) => table.category_id === categoryId),
    [categoryId, eligibleTables],
  )
  const categoryCounts = useMemo(() => eligibleTables.reduce<Record<string, number>>((result, table) => {
    result[table.category_id] = (result[table.category_id] ?? 0) + 1
    return result
  }, {}), [eligibleTables])
  const groupedTables = useMemo(() => {
    if (!catalog) return []
    return catalog.categories.flatMap((category) => {
      const tables = filteredTables.filter((table) => table.category_id === category.category_id)
      return tables.length ? [{ category, tables }] : []
    })
  }, [catalog, filteredTables])
  const matchedFields = (table: DataModelTable) => keyword
    ? table.fields
      .filter((field) => scope === 'internal' || field.source_mappable)
      .filter((field) => [field.name, field.label, field.description].join(' ').toLocaleLowerCase('zh-CN').includes(keyword))
      .slice(0, 3)
    : []

  const effectiveSelectedTableId = (
    selectedTableId && filteredTables.some((table) => table.table_id === selectedTableId)
      ? selectedTableId
      : filteredTables[0]?.table_id ?? null
  )
  const selectedTable = filteredTables.find((table) => table.table_id === effectiveSelectedTableId) ?? null

  if (loading) {
    return <div className="rounded-2xl border border-slate-200 bg-white p-8 text-sm text-slate-500">正在读取系统数据模型…</div>
  }

  if (error || !catalog) {
    return (
      <div role="alert" className="rounded-2xl border border-rose-200 bg-rose-50 p-6 text-sm text-rose-800">
        <p>{error ?? '系统数据模型不可用。'}</p>
        <button type="button" onClick={() => setRetryCount((current) => current + 1)} className="mt-3 rounded-lg border border-rose-300 px-3 py-2 font-semibold">重试</button>
      </div>
    )
  }

  return (
    <div className="space-y-5">
      <header className="rounded-2xl border border-indigo-200 bg-gradient-to-br from-indigo-950 via-slate-900 to-slate-950 p-5 text-white shadow-sm sm:p-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-indigo-200">Canonical data contracts</p>
            <h2 className="mt-2 text-xl font-bold">{scope === 'external' ? '外部数据导入标准' : '系统内部结构'}</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-200">
              {scope === 'external' ? catalog.description : '这里仅供查看代码对照、配置、任务和计算结果等内部结构，不需要用户单独导入。'}
            </p>
            <p className="mt-2 text-xs leading-5 text-slate-300">这里展示标准结构，不表示已接入数据。数据源与字段映射可在设置中的“数据源与接口映射”配置；定时调度尚未开放。</p>
          </div>
          <span className="w-fit rounded-lg border border-white/20 bg-white/10 px-3 py-2 text-xs font-semibold">Schema v{catalog.schema_version}</span>
        </div>
      </header>

      <section className="grid gap-3 sm:grid-cols-3" aria-label="数据模型摘要">
        <SummaryCard label="业务分类" value={catalog.summary.category_count} description="仅统计当前范围" />
        <SummaryCard label={scope === 'external' ? '导入表' : '系统内部表'} value={catalog.summary.table_count} description={scope === 'external' ? '可由外部提供的业务数据表' : '系统维护，不接受外部导入'} />
        <SummaryCard label={scope === 'external' ? '可映射字段' : '内部字段'} value={scope === 'external' ? catalog.summary.mapping_target_field_count : catalog.summary.field_count} description="含类型、单位与空值规则" />
      </section>

      <details className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <summary className="cursor-pointer text-sm font-semibold text-slate-900">查看全平台数据合同原则</summary>
        <div className="mt-4 grid gap-3 lg:grid-cols-2">
          <ul className="space-y-2 text-sm leading-6 text-slate-600">
            {catalog.principles.map((principle) => <li key={principle} className="rounded-xl bg-slate-50 px-3 py-2">{principle}</li>)}
          </ul>
          <div className="space-y-2">
            {catalog.type_conventions.map((item) => (
              <div key={item.logical_type} className="rounded-xl border border-slate-100 p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <code className="text-xs font-semibold text-indigo-700">{item.logical_type}</code>
                  <span className="text-slate-300">→</span>
                  <code className="text-xs text-slate-700">{item.physical_type}</code>
                </div>
                <p className="mt-1 text-xs leading-5 text-slate-500">{item.rule}</p>
              </div>
            ))}
          </div>
        </div>
      </details>

      <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm" aria-label="数据表筛选">
        <div className={`grid gap-3 ${scope === 'internal' ? 'lg:grid-cols-[minmax(0,1fr)_190px]' : ''}`}>
          <label className="text-sm font-medium text-slate-700">
            搜索业务表或字段
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="例如：基金净值、收盘价、adjusted_nav、可得时间"
              className="mt-1 block min-h-11 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
            />
            <span className="mt-1 block text-xs font-normal leading-5 text-slate-500">支持中文表名、英文 table_id、字段中文名和字段名。</span>
          </label>
          {scope === 'internal' ? <label className="text-sm font-medium text-slate-700">
            交付阶段
            <select
              value={phase}
              onChange={(event) => setPhase(event.target.value as typeof phase)}
              className="mt-1 block min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
            >
              <option value="all">全部</option>
              <option value="core">核心模型</option>
              <option value="next">下一阶段</option>
            </select>
          </label> : null}
        </div>
        <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-4" aria-label="数据模型分类">
          <button
            type="button"
            aria-pressed={categoryId === 'all'}
            onClick={() => setCategoryId('all')}
            className={`rounded-xl border p-3 text-left transition focus:outline-none focus:ring-2 focus:ring-indigo-500 ${categoryId === 'all' ? 'border-slate-900 bg-slate-900 text-white' : 'border-slate-200 bg-slate-50 text-slate-700 hover:border-slate-300'}`}
          >
            <span className="block text-sm font-bold">全部业务数据</span>
            <span className={`mt-1 block text-xs ${categoryId === 'all' ? 'text-slate-300' : 'text-slate-500'}`}>当前可见 {eligibleTables.length} 张表</span>
          </button>
          {catalog.categories.map((category) => {
            const count = categoryCounts[category.category_id] ?? 0
            const selected = categoryId === category.category_id
            return <button
              key={category.category_id}
              type="button"
              aria-pressed={selected}
              disabled={count === 0}
              onClick={() => setCategoryId(category.category_id)}
              className={`rounded-xl border p-3 text-left transition focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:cursor-not-allowed disabled:opacity-40 ${selected ? 'border-indigo-600 bg-indigo-700 text-white' : 'border-indigo-100 bg-indigo-50 text-indigo-950 hover:border-indigo-200'}`}
            >
              <span className="flex items-start justify-between gap-2"><span className="text-sm font-bold">{category.label}</span><span className={`shrink-0 rounded-full px-2 py-0.5 text-[11px] font-bold ${selected ? 'bg-white/15 text-white' : 'bg-white text-indigo-700'}`}>{count}</span></span>
              <span className={`mt-1 line-clamp-2 block text-xs leading-5 ${selected ? 'text-indigo-100' : 'text-slate-500'}`}>{category.description}</span>
            </button>
          })}
        </div>
      </section>

      <div className="grid min-w-0 gap-5 xl:grid-cols-[360px_minmax(0,1fr)]">
        <aside className="min-w-0 rounded-2xl border border-slate-200 bg-white p-3 shadow-sm xl:max-h-[calc(100vh-170px)] xl:overflow-y-auto" aria-label={scope === 'external' ? '外部导入表列表' : '系统内部表列表'}>
          <div className="px-2 py-2">
            <p className="text-sm font-bold text-slate-900">{scope === 'external' ? '外部导入表' : '系统内部表'}</p>
            <p className="mt-1 text-xs text-slate-500">当前筛选 {filteredTables.length} 张；按业务分类展示。</p>
          </div>
          <div className="mt-2 space-y-4">
            {groupedTables.map(({ category, tables }) => <section key={category.category_id} aria-label={`${category.label}表`}>
              <div className="sticky top-0 z-10 flex items-center justify-between gap-2 rounded-lg bg-slate-100 px-3 py-2">
                <span className="text-xs font-bold text-slate-700">{category.label}</span>
                <span className="text-[11px] font-semibold text-slate-500">{tables.length} 张</span>
              </div>
              <div className="mt-2 grid gap-2">{tables.map((table) => {
                const fieldMatches = matchedFields(table)
                return <button
                  key={table.table_id}
                  type="button"
                  aria-pressed={effectiveSelectedTableId === table.table_id}
                  onClick={() => setSelectedTableId(table.table_id)}
                  className={`rounded-xl border p-3 text-left transition focus:outline-none focus:ring-2 focus:ring-indigo-500 ${effectiveSelectedTableId === table.table_id ? 'border-indigo-300 bg-indigo-50' : 'border-slate-100 hover:border-slate-300 hover:bg-slate-50'}`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="font-semibold text-slate-900">{table.label}</span>
                    <span className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold ${layerStyles[table.layer]}`}>{layerLabels[table.layer]}</span>
                  </div>
                  <code className="mt-1 block break-all text-xs text-slate-500">{table.table_id}</code>
                  <p className="mt-2 line-clamp-2 text-xs leading-5 text-slate-500">{table.description}</p>
                  {fieldMatches.length ? <div className="mt-2 flex flex-wrap gap-1" aria-label="搜索命中字段">{fieldMatches.map((field) => <span key={field.name} className="rounded bg-amber-50 px-1.5 py-1 text-[10px] font-semibold text-amber-800">{field.label} · {field.name}</span>)}</div> : null}
                </button>
              })}</div>
            </section>)}
          </div>
          {!filteredTables.length ? <p className="px-3 py-8 text-center text-sm text-slate-500">没有符合条件的表或字段。可以清空搜索或切回“全部业务数据”。</p> : null}
        </aside>

        {selectedTable ? <TableDetail key={selectedTable.table_id} table={selectedTable} /> : (
          <div className="rounded-2xl border border-slate-200 bg-white p-8 text-center text-sm text-slate-500">请选择一张数据表。</div>
        )}
      </div>
    </div>
  )
}
