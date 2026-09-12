import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { fetchSourceCatalog, type ConfigRecord, type InterfaceConfig, type SourceCatalog, type SourceConfig } from '../services/dataSources'
import InterfaceEditor from '../components/data-sources/InterfaceEditor'
import SourceEditor from '../components/data-sources/SourceEditor'
import SourceOverview from '../components/data-sources/SourceOverview'
import ResolutionPanel from '../components/data-sources/ResolutionPanel'
import DataWorkspaceNav from '../components/data-sources/DataWorkspaceNav'
import { buttonClass, inputClass } from '../components/data-sources/EditorFields'

export default function DataSourceCenter() {
  const [params] = useSearchParams()
  const [tab, setTab] = useState<'sources' | 'resolution'>(params.get('view') === 'resolution' ? 'resolution' : 'sources')
  const [catalog, setCatalog] = useState<SourceCatalog | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [refresh, setRefresh] = useState(0)
  const [sourceId, setSourceId] = useState(params.get('source') ?? 'tushare')
  const [interfaceId, setInterfaceId] = useState<string | null>(params.get('interface'))
  const [editingSource, setEditingSource] = useState(false)
  const [newKind, setNewKind] = useState<'source' | 'interface' | null>(null)
  const [dirty, setDirty] = useState(false)
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<'all' | 'incomplete' | 'disabled' | 'ready'>('all')
  const [sort, setSort] = useState<'name' | 'status'>('name')
  const [operationBusy, setOperationBusy] = useState(false)
  const [category, setCategory] = useState('all')
  const [notice, setNotice] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true); setError('')
    fetchSourceCatalog(controller.signal)
      .then(value => { if (!controller.signal.aborted) setCatalog(value) })
      .catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '数据源中心加载失败。') })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [refresh])

  useEffect(() => {
    if (!dirty) return
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = '' }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [dirty])

  useEffect(() => {
    if (!loading && catalog && params.get('view') === 'results') {
      document.getElementById('mapping-results')?.scrollIntoView?.({ block: 'start' })
    }
  }, [loading, catalog, params])

  const allowLeave = () => {
    if (operationBusy) { setNotice('当前操作尚未结束，请勿切换配置。'); return false }
    return !dirty || window.confirm('当前修改尚未保存，放弃修改并切换？')
  }
  const navigateEditor = (action: () => void) => {
    if (!allowLeave()) return
    setDirty(false); setNotice(''); action()
  }
  const source = catalog?.sources.find(item => item.config.id === sourceId) ?? catalog?.sources[0]
  const sourceInterfaces = useMemo(() => catalog?.interfaces.filter(item => item.config.source_id === source?.config.id) ?? [], [catalog, source?.config.id])
  const interfaces = useMemo(() => {
    const keyword = query.trim().toLowerCase()
    return sourceInterfaces.filter(item => {
      if (filter === 'incomplete' && item.validation?.ready) return false
      if (filter === 'disabled' && item.config.enabled) return false
      if (filter === 'ready' && (!item.config.enabled || !item.validation?.ready)) return false
      const targets = catalog?.targets.tables.filter(table => item.config.mappings.some(mapping => mapping.target_table === table.table_id)) ?? []
      if (category !== 'all' && !targets.some(table => table.category_id === category)) return false
      return `${item.config.name} ${item.config.api_name} ${item.config.id} ${targets.map(table => `${table.label} ${table.table_id}`).join(' ')}`.toLowerCase().includes(keyword)
    }).sort((a, b) => {
      if (sort === 'status') {
        const order = Number(Boolean(a.validation?.ready)) - Number(Boolean(b.validation?.ready))
        if (order) return order
      }
      return a.config.name.localeCompare(b.config.name, 'zh-CN') || a.config.id.localeCompare(b.config.id)
    })
  }, [catalog, sourceInterfaces, query, category, filter, sort])

  const openInterface = (id: string) => navigateEditor(() => {
    const item = catalog?.interfaces.find(entry => entry.config.id === id)
    if (item) setSourceId(item.config.source_id)
    setInterfaceId(id); setNewKind(null); setEditingSource(false)
  })
  const addInterface = () => navigateEditor(() => { setNewKind('interface'); setInterfaceId(null); setEditingSource(false) })
  const editSource = () => navigateEditor(() => { setInterfaceId(null); setNewKind(null); setEditingSource(true) })

  if (loading) return <p role="status" className="rounded-xl bg-white p-6 text-sm">正在读取数据源与映射配置…</p>
  if (error || !catalog) return <section role="alert" className="rounded-xl bg-rose-50 p-6 text-rose-800"><p>{error || '配置不可用。'}</p><button type="button" className={`${buttonClass} mt-3`} onClick={() => setRefresh(v => v + 1)}>重新加载</button></section>

  const sourceRecord: ConfigRecord<SourceConfig> | undefined = newKind === 'source'
    ? { config: { ...catalog.templates.source, id: '' }, revision: 0, builtin: false, updated_at: '' } : source
  const interfaceRecord: ConfigRecord<InterfaceConfig> | undefined = newKind === 'interface' && source
    ? { config: { ...catalog.templates.interface, source_id: source.config.id, id: '', method: source.config.transport === 'tushare' ? 'POST' : 'GET' }, revision: 0, builtin: false, updated_at: '' }
    : catalog.interfaces.find(item => item.config.id === interfaceId && item.config.source_id === source?.config.id)
  const batches = catalog.batches?.filter(batch => batch.source_id === source?.config.id) ?? []
  const saved = (kind: 'source' | 'interface', id?: string) => {
    setDirty(false); setNewKind(null)
    setNotice(id ? '配置已保存。映射验证和真实采样不会自动启动下载。' : '配置已删除，已下载数据未受影响。')
    if (kind === 'source') { setSourceId(id ?? 'tushare'); setInterfaceId(null); setEditingSource(Boolean(id)) } else setInterfaceId(id ?? null)
    setRefresh(v => v + 1)
  }

  return <div className="space-y-5">
    <DataWorkspaceNav beforeNavigate={allowLeave} />
    <header className="rounded-xl border border-slate-200 bg-white p-5">
      <h1 className="text-2xl font-bold text-slate-950">数据源与接口映射</h1>
      <p className="mt-2 text-sm leading-6 text-slate-600">告诉系统：数据从哪里来、每一列对应什么。已有接口直接选择，不必重新配置。</p>
      <p className="mt-3 rounded-lg bg-slate-50 px-3 py-2 text-sm text-slate-700">选需要的数据 → 确认连接 → 对应字段 → 验证样本 → 去下载</p>
      <details className="mt-3 text-sm text-slate-600"><summary className="cursor-pointer font-semibold">第一次使用？三个概念就够了</summary><p className="mt-2 leading-6">数据源是供应商，例如 Tushare；接口是一类数据，例如基金净值；字段映射是把供应商的列对应到系统的列，例如 close → 收盘价。保存配置不等于下载，样本通过也不等于数据已发布。</p></details>
      <div className="mt-3 flex flex-wrap gap-4 text-sm font-semibold text-accent-700">
        <Link to="/settings/data-model" onClick={event => { if (!allowLeave()) event.preventDefault() }}>查看系统标准表 →</Link>
        {interfaceRecord && source?.config.transport === 'tushare' ? <Link to="/settings/data-sources" onClick={event => { if (!allowLeave()) event.preventDefault() }}>进入 Tushare 下载与更新 →</Link> : null}
      </div>
    </header>
    {notice ? <p role="status" className="rounded-xl bg-emerald-50 p-3 text-sm text-emerald-900">{notice}</p> : null}
    {!catalog.editing_enabled ? <p role="status" className="rounded-xl bg-amber-50 p-4 text-sm text-amber-900">当前环境为只读：可查看合同和离线验证，不允许修改配置或真实采样。</p> : null}
    <nav aria-label="数据接入功能" className="flex flex-wrap gap-2"><button type="button" aria-pressed={tab === 'sources'} className={`${buttonClass} ${tab === 'sources' ? 'bg-accent-50' : ''}`} onClick={() => navigateEditor(() => setTab('sources'))}>数据源与接口</button><button type="button" aria-pressed={tab === 'resolution'} className={`${buttonClass} ${tab === 'resolution' ? 'bg-accent-50' : ''}`} onClick={() => navigateEditor(() => setTab('resolution'))}>多源取值规则</button></nav>
    {tab === 'sources' ? <>
    <div className="grid min-w-0 gap-5 xl:grid-cols-[280px_minmax(0,1fr)]">
      <aside className="min-w-0 space-y-4 rounded-xl border border-slate-200 bg-white p-4" aria-label="数据源与接口列表">
        <label className="block text-xs font-semibold text-slate-600">当前数据源<select className={inputClass} value={source?.config.id ?? ''} onChange={e => navigateEditor(() => { setSourceId(e.target.value); setInterfaceId(null); setNewKind(null); setEditingSource(false); setQuery(''); setFilter('all'); setCategory('all') })}>{catalog.sources.map(item => <option key={item.config.id} value={item.config.id}>{item.config.name}{item.config.enabled ? '' : '（已停用）'}</option>)}</select></label>
        <div className="flex flex-wrap gap-2">
          <button type="button" className={buttonClass} onClick={() => navigateEditor(() => { setInterfaceId(null); setNewKind(null); setEditingSource(false) })}>来源概览</button>
          <button type="button" className={buttonClass} onClick={editSource}>编辑数据源</button>
          <button type="button" className={buttonClass} disabled={!catalog.editing_enabled} onClick={() => navigateEditor(() => { setNewKind('source'); setInterfaceId(null); setEditingSource(true) })}>新建数据源</button>
        </div>
        <label className="block text-xs font-semibold text-slate-600">搜索接口<input className={inputClass} value={query} onChange={e => setQuery(e.target.value)} placeholder="例如：ETF 行情、基金净值" /></label>
        <label className="block text-xs font-semibold text-slate-600">接口状态<select className={inputClass} value={filter} onChange={e => setFilter(e.target.value as typeof filter)}><option value="all">全部接口</option><option value="incomplete">映射待完善</option><option value="ready">已启用且定义已校验</option><option value="disabled">已停用</option></select></label>
        <label className="block text-xs font-semibold text-slate-600">数据分类<select className={inputClass} value={category} onChange={e => setCategory(e.target.value)}><option value="all">全部分类</option>{catalog.targets.categories.map(item => <option key={item.category_id} value={item.category_id}>{item.label}</option>)}</select></label>
        <label className="block text-xs font-semibold text-slate-600">接口排序<select className={inputClass} value={sort} onChange={e => setSort(e.target.value as typeof sort)}><option value="name">按数据名称</option><option value="status">待完善优先</option></select></label>
        <button type="button" className={buttonClass} disabled={!catalog.editing_enabled || !source} onClick={addInterface}>新建接口</button>
        <p className="text-xs text-slate-600">显示 {interfaces.length} / {sourceInterfaces.length} 个接口</p>
        {interfaceRecord || editingSource || newKind ? <><label className="block text-xs font-semibold text-slate-600 xl:hidden">选择接口<select aria-label="选择接口" className={inputClass} value={interfaces.some(item => item.config.id === interfaceId) ? interfaceId ?? '' : ''} onChange={event => { if (event.target.value) openInterface(event.target.value) }}><option value="">选择接口查看映射</option>{interfaces.map(item => <option key={item.config.id} value={item.config.id}>{item.config.name}{item.validation?.ready ? '' : '（待完善）'}</option>)}</select></label>
        <div className="hidden max-h-[60vh] space-y-2 overflow-y-auto xl:block">{interfaces.map(item => <button type="button" key={item.config.id} aria-pressed={interfaceId === item.config.id && !newKind} onClick={() => openInterface(item.config.id)} className={`block w-full rounded-xl border p-3 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 ${interfaceId === item.config.id && !newKind ? 'border-accent-400 bg-accent-50' : 'border-slate-200 hover:bg-slate-50'}`}>
          <span className="block text-sm font-semibold">{item.config.name}</span><code className="mt-1 block break-all text-xs text-slate-600">{item.config.api_name || item.config.id}</code>
          <span className={`mt-2 block text-xs ${item.config.enabled && item.validation?.ready ? 'text-slate-600' : 'text-amber-800'}`}>{item.config.enabled ? '已启用' : '已停用'} · {item.validation?.ready ? '定义已校验' : '映射待完善'}</span>
        </button>)}{!interfaces.length ? <p className="p-3 text-sm text-slate-600">没有符合条件的接口。可切换为“全部接口”或清空搜索。</p> : null}</div></> : null}
      </aside>
      <div className="min-w-0">{interfaceRecord && source && newKind !== 'source'
        ? <InterfaceEditor key={`${interfaceRecord.config.id}:${interfaceRecord.revision}:${refresh}`} record={interfaceRecord} source={source.config} targets={catalog.targets} editingEnabled={catalog.editing_enabled} credentialConfigured={source.credential_configured === true} onSaved={id => saved('interface', id)} onDirty={setDirty} onBusy={setOperationBusy} />
        : sourceRecord && (editingSource || newKind === 'source')
          ? <SourceEditor key={`${sourceRecord.config.id}:${sourceRecord.revision}:${refresh}`} record={sourceRecord} editingEnabled={catalog.editing_enabled} onSaved={id => saved('source', id)} onDirty={setDirty} onBusy={setOperationBusy} onCredentialChanged={configured => setCatalog(current => current ? { ...current, sources: current.sources.map(item => item.config.id === sourceRecord.config.id ? { ...item, credential_configured: configured } : item) } : current)} onNext={() => navigateEditor(() => { setEditingSource(false); setNewKind(null); setInterfaceId(null) })} />
          : source ? <SourceOverview key={`${source.config.id}:${query}:${filter}:${category}:${sort}`} source={source} interfaces={interfaces} total={sourceInterfaces.length} targets={catalog.targets} editingEnabled={catalog.editing_enabled} onConnect={editSource} onAddInterface={addInterface} onOpen={openInterface} onClear={() => { setFilter('all'); setQuery(''); setCategory('all') }} onReview={() => { setFilter('incomplete'); setQuery(''); setCategory('all') }} />
            : <p>请先新建数据源。</p>}
      </div>
    </div>
    <details id="mapping-results" open={params.get('view') === 'results'} className="scroll-mt-6 rounded-xl border border-slate-200 bg-white p-4">
      <summary className="cursor-pointer py-2 text-sm font-semibold">标准表映射结果 · {batches.length} 个最近批次</summary>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3"><p className="max-w-3xl text-sm leading-6 text-slate-600">这里只显示 {source?.config.name} 的候选结果。字段校验通过后仍需跨批次去重、身份核验和版本发布，不能直接用于正式研究。</p><button type="button" className={buttonClass} onClick={() => navigateEditor(() => setRefresh(value => value + 1))}>刷新映射结果</button></div>
      <div className="mt-3 space-y-2">{batches.map(batch => <details key={batch.batch_id} className="rounded-lg border border-slate-200 p-3">
        <summary className="cursor-pointer text-sm"><strong>{catalog.interfaces.find(item => item.config.id === batch.interface_id)?.config.name ?? batch.interface_id}</strong> · {batch.status === 'REJECTED' ? '映射需要修复' : batch.status === 'EMPTY' ? '来源没有返回数据' : '字段校验通过，尚未发布'} · {batch.source_rows} 行</summary>
        <p className="mt-2 break-all text-xs text-slate-600">记录时间：{new Date(batch.created_at).toLocaleString('zh-CN')} · 批次 {batch.batch_id}</p>
        {batch.tables.map((table, index) => <div key={index} className="mt-2 text-xs leading-5"><strong>{catalog.targets.tables.find(item => item.table_id === table.table_id)?.label ?? table.table_id}</strong> · 接受 {table.rows ?? 0} 行 / 拒绝 {table.rejected_rows ?? 0} 行{table.errors.map((issue, i) => <p key={i} className="text-rose-700">{issue.message}</p>)}</div>)}
        {batch.status === 'REJECTED' ? <button type="button" className={`${buttonClass} mt-3`} onClick={() => { openInterface(batch.interface_id); window.scrollTo?.({ top: 0, behavior: 'smooth' }) }}>修复此接口映射</button> : null}
      </details>)}</div>
      {!batches.length ? <p className="mt-3 text-sm text-slate-600">暂无候选批次。保存配置、离线预览和单次采样都不会生成下载结果。</p> : null}
    </details>
    </> : <ResolutionPanel catalog={catalog} onDirty={setDirty} />}
  </div>
}
