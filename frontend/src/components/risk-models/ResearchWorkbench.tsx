import { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import {
  getRiskRun, previewRiskModel, publishRiskPreview, retireRiskRelease, riskCatalog, riskReleases,
  searchRiskProducts,
  type ResearchDomain, type RiskCatalog, type RiskModelFields, type RiskProduct,
  type RiskRelease, type RiskRun, type RiskVariable,
} from '../../services/riskModels'
import CashflowResearch from './CashflowResearch'
import RiskRunView from './RiskRunView'
import VariableImport from './VariableImport'
import { buttonClass, Empty, Feedback, Field, frequencyLabels, inputClass, primaryClass, sectionClass, statusLabels, Steps, today, NumberInput } from './ResearchUI'

function initialFields(domain: ResearchDomain, target?: RiskProduct): RiskModelFields {
  const date = new Date(); const end = today()
  const earlier = (years: number) => new Date(Date.UTC(date.getUTCFullYear() - years, date.getUTCMonth(), 1)).toISOString().slice(0, 10)
  return { name: '', stage: domain === 'product' ? 'product' : 'macro_market', method: 'ols', inputs: [], outputs: [], targets: target ? [target] : [], frequency: 'monthly', start_date: earlier(5), end_date: end, validation_start: earlier(1), as_of: end, lags: 0, min_train: 36, min_validation: 12, minimum_validation_r2: 0, refit_after_validation: true }
}
function VariablePicker({ title, variables, selected, onChange }: { title: string; variables: RiskVariable[]; selected: string[]; onChange: (keys: string[]) => void }) {
  const [query, setQuery] = useState('')
  const filtered = variables.filter(item => `${item.name} ${item.id}`.toLowerCase().includes(query.toLowerCase()))
  return <fieldset className="min-w-0"><legend className="text-sm font-semibold text-slate-900">{title} <span className="font-normal text-slate-600">已选 {selected.length} 项</span></legend><input className={`${inputClass} mb-2`} aria-label={`搜索${title}`} placeholder="搜索名称或代码" value={query} onChange={event => setQuery(event.target.value)} /><div className="max-h-64 space-y-1 overflow-y-auto rounded-lg border border-slate-200 p-2">{filtered.map(variable => <label key={variable.id} className="flex cursor-pointer items-start gap-2 rounded-lg px-2 py-2 text-sm hover:bg-slate-50"><input className="mt-1" type="checkbox" checked={selected.includes(variable.id)} disabled={(!variable.availability?.available && !selected.includes(variable.id)) || (selected.length >= 8 && !selected.includes(variable.id))} onChange={event => onChange(event.target.checked ? [...selected, variable.id] : selected.filter(key => key !== variable.id))} /><span className="min-w-0"><span className="font-medium text-slate-800">{variable.name}</span><span className="ml-2 text-xs text-slate-600">{variable.unit_label} · {frequencyLabels[variable.frequency]}</span>{!variable.availability?.available && <span className="mt-1 block text-xs text-amber-800">{variable.availability?.reason ?? '数据状态未知，请刷新。'}</span>}</span></label>)}{!filtered.length && <p className="p-3 text-sm text-slate-600">没有匹配变量。缺少自有数据时，可到「数据与因子」导入。</p>}</div></fieldset>
}

function PublishedProductLinks({ releaseId, targets }: { releaseId: string; targets: RiskRun['targets'] }) {
  const products = targets.filter(target => target.kind === 'etf' || target.kind === 'fund')
  if (!releaseId || !products.length) return null
  return <nav aria-label="到产品查看这份成果" className="space-y-2 border-t border-slate-100 pt-3">
    <p className="text-xs text-slate-600">跳转后锁定这份发布版本，不会换成其他模型或重新训练。</p>
    <div className="flex max-h-60 flex-wrap gap-2 overflow-y-auto">{products.map(target => <Link
      key={target.key} className={buttonClass}
      to={`/product-research/products/${encodeURIComponent(target.product_id)}?${new URLSearchParams({ kind: target.kind, tab: 'risk', exposure_release_id: releaseId })}`}
    >到{target.name}查看</Link>)}</div>
  </nav>
}

export default function ResearchWorkbench({ domain = 'product' }: { domain?: ResearchDomain }) {
  const [params] = useSearchParams()
  const key = params.get('product_key') ?? ''
  const split = key.indexOf(':'); const kind = key.slice(0, split)
  const preselected = (kind === 'etf' || kind === 'fund') && split >= 0 ? { kind, product_id: key.slice(split + 1), name: params.get('product_name') ?? key.slice(split + 1) } as RiskProduct : undefined
  const [pane, setPane] = useState<'research' | 'published' | 'data'>(domain === 'product' && !preselected ? 'published' : 'research')
  const [method, setMethod] = useState<'ols' | 'cashflow'>('ols')
  const [catalog, setCatalog] = useState<RiskCatalog | null>(null)
  const [releases, setReleases] = useState<RiskRelease[]>([])
  const [draft, setDraft] = useState(() => initialFields(domain, preselected))
  const [step, setStep] = useState(0)
  const [run, setRun] = useState<RiskRun | null>(null)
  const [runSignature, setRunSignature] = useState('')
  const [inspected, setInspected] = useState<RiskRun | null>(null)
  const [inspectedReleaseId, setInspectedReleaseId] = useState('')
  const [query, setQuery] = useState('')
  const [productKind, setProductKind] = useState<'etf' | 'fund'>('etf')
  const [products, setProducts] = useState<Array<{ ts_code: string; name: string }>>([])
  const [searching, setSearching] = useState(false)
  const [searched, setSearched] = useState(false)
  const [busy, setBusy] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [refresh, setRefresh] = useState(0)
  const [acknowledged, setAcknowledged] = useState(false)
  const [validDays, setValidDays] = useState(domain === 'product' ? 30 : 180)
  const [note, setNote] = useState('')
  const [publishedId, setPublishedId] = useState('')
  const alive = useRef(true); const searchToken = useRef(0); const operation = useRef(0)
  const signature = JSON.stringify(draft); const stale = runSignature !== signature
  useEffect(() => { alive.current = true; return () => { alive.current = false; operation.current++; searchToken.current++ } }, [])
  useEffect(() => {
    const controller = new AbortController(); setLoading(true)
    Promise.all([riskCatalog(domain, controller.signal), riskReleases(domain, {}, controller.signal)])
      .then(([nextCatalog, nextReleases]) => { if (!controller.signal.aborted) { setCatalog(nextCatalog); setReleases(nextReleases) } })
      .catch(caught => { if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : '研究目录加载失败。') })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [domain, refresh])
  function patch(fields: Partial<RiskModelFields>) { setDraft(old => ({ ...old, ...fields })); setAcknowledged(false); setPublishedId(''); setNotice(''); setError('') }
  function resetResearch() { operation.current++; setDraft(initialFields(domain, preselected)); setRun(null); setRunSignature(''); setStep(0); setPublishedId(''); setAcknowledged(false); setError(''); setNotice(''); setMethod('ols'); setPane('research') }
  async function search() {
    const token = ++searchToken.current; setSearching(true); setProducts([]); setError('')
    try { const next = await searchRiskProducts(productKind, query); if (alive.current && token === searchToken.current) { setProducts(next); setSearched(true) } }
    catch (caught) { if (alive.current && token === searchToken.current) setError(caught instanceof Error ? caught.message : '产品搜索失败。') }
    finally { if (alive.current && token === searchToken.current) setSearching(false) }
  }
  async function calculate() {
    const token = ++operation.current; setBusy('run'); setError(''); setNotice(''); setAcknowledged(false); setPublishedId('')
    try {
      if (!draft.name.trim() || !draft.inputs.length || (domain === 'product' ? !draft.targets.length : !draft.outputs.length)) throw new Error('请先填写研究名称，选择输入变量和研究对象。')
      const result = await previewRiskModel(domain, draft)
      if (alive.current && token === operation.current) { setRun(result); setRunSignature(signature); setStep(1); setNotice('计算完成。当前结果只保留在本页面；只有确认发布后才会写入磁盘。') }
    } catch (caught) { if (alive.current && token === operation.current) setError(caught instanceof Error ? caught.message : '敏感性研究未完成。') }
    finally { if (alive.current && token === operation.current) setBusy('') }
  }
  async function publish() {
    if (!run || stale || !run.publishable || !run.preview_hash || !acknowledged) return
    const token = ++operation.current; setBusy('publish'); setError('')
    try {
      const release = await publishRiskPreview(domain, draft, run.preview_hash, validDays, note)
      if (alive.current && token === operation.current) { setPublishedId(release.id); setNotice('已确认发布并写入统一数据磁盘。产品与组合页面可直接读取，不需要重新训练。'); setRefresh(old => old + 1) }
    } catch (caught) { if (alive.current && token === operation.current) setError(caught instanceof Error ? caught.message : '发布失败。') }
    finally { if (alive.current && token === operation.current) setBusy('') }
  }
  async function inspect(id: string, releaseId = '') {
    const token = ++operation.current; setBusy('read'); setError(''); setInspected(null); setInspectedReleaseId('')
    try {
      const result = await getRiskRun(domain, id)
      if (alive.current && token === operation.current) { setInspected(result); setInspectedReleaseId(releaseId) }
    } catch (caught) { if (alive.current && token === operation.current) setError(caught instanceof Error ? caught.message : '结果读取失败。') }
    finally { if (alive.current && token === operation.current) setBusy('') }
  }
  async function retire(release: RiskRelease) {
    if (!window.confirm(`停用「${release.name}」后，新研究不能再使用。历史记录会保留。确定停用吗？`)) return
    setBusy('retire'); setError('')
    try { await retireRiskRelease(domain, release.id, '研究者在成果库中确认停用。'); if (alive.current) { setRefresh(old => old + 1); setNotice('已停用，历史研究记录保留。') } }
    catch (caught) { if (alive.current) setError(caught instanceof Error ? caught.message : '停用失败。') }
    finally { if (alive.current) setBusy('') }
  }
  const inputRole = draft.stage === 'event_macro' ? 'driver' : domain === 'product' ? 'market' : 'macro'
  const outputRole = draft.stage === 'event_macro' ? 'macro' : 'market'
  return <div className="min-w-0 space-y-5" data-testid={`${domain}-risk-workbench`}>
    <div className="flex flex-wrap items-start justify-between gap-3"><div><h2 className="text-xl font-semibold text-slate-950">{domain === 'product' ? '风险模型中心' : '宏观传导研究'}</h2><p className="mt-2 text-sm text-slate-600">{domain === 'product' ? '研究产品怕什么，验证并发布后，供产品和组合页面直接使用。' : '研究事件、经济与市场之间的条件关系；传导强度由数据估计，不手填系数。'}</p></div><button type="button" className={buttonClass} disabled={Boolean(busy)} onClick={resetResearch}>新建研究</button></div>
    <nav aria-label="研究中心工作区" className="flex flex-wrap gap-2 border-b border-slate-200 pb-3">{([['research', '研究方案'], ['published', '已发布成果'], ['data', '数据与因子']] as const).map(([id, label]) => <button type="button" key={id} aria-current={pane === id ? 'page' : undefined} disabled={Boolean(busy)} className={pane === id ? primaryClass : buttonClass} onClick={() => setPane(id)}>{label}</button>)}</nav>
    <Feedback error={error} notice={notice} />
    {loading && !catalog ? <div role="status" className="rounded-xl bg-white p-6 text-sm text-slate-600">正在读取本地研究目录…</div> : !catalog ? <Empty title="研究目录暂不可用"><p>检查数据磁盘与后端连接后重试。</p><button type="button" className={`${buttonClass} mt-3`} onClick={() => setRefresh(old => old + 1)}>重新加载</button></Empty> : <>
      {pane === 'research' && <>
        {domain === 'product' && <div className="flex flex-wrap gap-2" aria-label="研究方法"><button type="button" disabled={Boolean(busy)} aria-pressed={method === 'ols'} className={buttonClass} onClick={() => setMethod('ols')}>基金 / ETF 敏感度</button><button type="button" disabled={Boolean(busy)} aria-pressed={method === 'cashflow'} className={buttonClass} onClick={() => setMethod('cashflow')}>债券现金流估值</button></div>}
        {method === 'cashflow' ? <CashflowResearch onPublished={() => setRefresh(old => old + 1)} /> : <div className="min-w-0 space-y-4"><Steps labels={['选对象和数据', '计算与验证', '发布成果']} active={step} onChange={setStep} disabled={Boolean(busy)} />
            {step === 0 && <section className={`${sectionClass} space-y-5`}><fieldset disabled={Boolean(busy)} className="min-w-0 space-y-5"><div className="grid gap-4 sm:grid-cols-2"><Field label="研究名称"><input className={inputClass} value={draft.name} onChange={event => patch({ name: event.target.value })} placeholder={domain === 'product' ? '例如：核心基金市场敏感度' : '例如：通胀与市场响应'} /></Field>{domain === 'transmission' && <Field label="研究哪一段传导"><select className={inputClass} value={draft.stage} onChange={event => patch({ stage: event.target.value as RiskModelFields['stage'], inputs: [], outputs: [] })}><option value="event_macro">事件驱动 → 宏观变量</option><option value="macro_market">宏观变量 → 市场风险因子</option></select></Field>}</div>
              {domain === 'product' && <div className="space-y-3"><h3 className="text-sm font-semibold">先选择基金或 ETF</h3><div className="grid items-end gap-2 sm:grid-cols-[120px_minmax(0,1fr)_auto]"><Field label="产品类型"><select className={inputClass} value={productKind} disabled={searching} onChange={event => { searchToken.current++; setProductKind(event.target.value as 'etf' | 'fund'); setProducts([]); setSearched(false) }}><option value="etf">ETF</option><option value="fund">公募基金</option></select></Field><Field label="名称或代码"><input className={inputClass} value={query} onChange={event => setQuery(event.target.value)} onKeyDown={event => { if (event.key === 'Enter') { event.preventDefault(); void search() } }} /></Field><button type="button" className={buttonClass} disabled={searching} onClick={() => void search()}>{searching ? '搜索中…' : '搜索产品'}</button></div>{products.length > 0 && <div className="max-h-44 overflow-auto rounded-lg border border-slate-200 p-2">{products.map(product => { const selected = draft.targets.some(target => target.kind === productKind && target.product_id === product.ts_code); return <label key={product.ts_code} className="flex min-h-10 items-center gap-2 px-2 text-sm"><input type="checkbox" checked={selected} disabled={!selected && draft.targets.length >= 64} onChange={event => patch({ targets: event.target.checked ? [...draft.targets, { kind: productKind, product_id: product.ts_code, name: product.name }] : draft.targets.filter(target => target.kind !== productKind || target.product_id !== product.ts_code) })} />{product.name} <span className="text-xs text-slate-600">{product.ts_code}</span></label> })}</div>}{searched && !products.length && <p className="text-sm text-amber-800">没有匹配的本地产品，请调整搜索词或先同步产品资料。</p>}<div className="flex flex-wrap gap-2">{draft.targets.map(target => <button type="button" key={`${target.kind}:${target.product_id}`} className="rounded-lg border border-accent-200 bg-accent-50 px-3 py-2 text-xs text-accent-900" onClick={() => patch({ targets: draft.targets.filter(item => item !== target) })} aria-label={`移除研究对象${target.name}`}>{target.name} · 移除</button>)}</div></div>}
              <div className={`grid gap-5 ${domain === 'transmission' ? 'lg:grid-cols-2' : ''}`}><VariablePicker title={domain === 'product' ? '风险因子' : '上游输入变量'} variables={catalog.variables.filter(item => item.roles.includes(inputRole))} selected={draft.inputs} onChange={inputs => patch({ inputs })} />{domain === 'transmission' && <VariablePicker title="下游响应变量" variables={catalog.variables.filter(item => item.roles.includes(outputRole) && !draft.inputs.includes(item.id))} selected={draft.outputs} onChange={outputs => patch({ outputs })} />}</div>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3"><Field label="研究频率" hint="低频数据不能自动变成高频；季度模型通常需要更长历史。"><select className={inputClass} value={draft.frequency} onChange={event => patch({ frequency: event.target.value as RiskModelFields['frequency'] })}>{catalog.frequencies.filter(item => domain === 'product' || ['monthly', 'quarterly'].includes(item.id)).map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></Field><Field label="研究截止日"><input className={inputClass} type="date" max={today()} value={draft.as_of} onChange={event => patch({ as_of: event.target.value })} /></Field><Field label="历史开始日"><input className={inputClass} type="date" value={draft.start_date} onChange={event => patch({ start_date: event.target.value })} /></Field><Field label="历史结束日"><input className={inputClass} type="date" max={draft.as_of} value={draft.end_date} onChange={event => patch({ end_date: event.target.value })} /></Field><Field label="从哪天开始留作验证" hint="这一天之后的数据不会用于验证模型的训练。"><input className={inputClass} type="date" value={draft.validation_start} onChange={event => patch({ validation_start: event.target.value })} /></Field></div>
              <details className="border-t border-slate-100 pt-3"><summary className="cursor-pointer text-sm font-medium text-slate-700">高级参数</summary><div className="mt-4 grid gap-4 sm:grid-cols-2">{domain === 'transmission' && <Field label="最多考虑几期滞后"><input className={inputClass} type="number" min="0" max="6" value={draft.lags} onChange={event => patch({ lags: Number(event.target.value) })} /></Field>}<Field label="最少训练样本"><input className={inputClass} type="number" min="30" value={draft.min_train} onChange={event => patch({ min_train: Number(event.target.value) })} /></Field><Field label="最少验证样本"><input className={inputClass} type="number" min="10" value={draft.min_validation} onChange={event => patch({ min_validation: Number(event.target.value) })} /></Field><Field label="最低验证 R²" hint="模型解释力门槛，不是准确率或收益承诺。"><NumberInput className={inputClass} min="-1" max="0.99" value={draft.minimum_validation_r2} onValueChange={number => patch({ minimum_validation_r2: number })} /></Field></div><label className="mt-4 flex items-start gap-2 text-sm text-slate-600"><input className="mt-1" type="checkbox" checked={draft.refit_after_validation} onChange={event => patch({ refit_after_validation: event.target.checked })} />验证通过后，再用全部截至日数据重估发布系数；独立验证结果仍保留。</label><p className="mt-3 text-xs text-slate-600">当前使用多因子 OLS；宏观模型是有限滞后的条件统计传导，不冒充结构因果模型。用户不能修改 Beta。</p></details>
            </fieldset><div className="rounded-lg bg-slate-50 p-3 text-xs leading-5 text-slate-600">计算和验证阶段不会保存方案、系数或冻结数组。关闭页面即丢弃；只有点击“确认发布”后才写入统一数据磁盘。</div><button type="button" className={primaryClass} disabled={Boolean(busy)} onClick={() => void calculate()}>{busy === 'run' ? '正在读取数据并计算…' : '计算敏感度'}</button></section>}
            {step === 1 && <>{run ? <><RiskRunView run={run} />{stale && <p className="rounded-lg bg-amber-50 p-3 text-sm text-amber-900">此运行与当前参数不同，不能直接发布当前方案。请重新计算。</p>}<div className="flex flex-wrap gap-3"><button type="button" className={buttonClass} disabled={Boolean(busy)} onClick={() => setStep(0)}>调整研究设置</button><button type="button" className={primaryClass} disabled={Boolean(busy) || stale || !run.publishable} onClick={() => setStep(2)}>确认并发布</button></div></> : <Empty title="还没有计算结果"><p>先选择研究对象、变量和日期，再计算敏感度。</p><button type="button" className={`${primaryClass} mt-3`} onClick={() => setStep(0)}>去选择数据</button></Empty>}</>}
            {step === 2 && <section className={`${sectionClass} space-y-4`}><h3 className="font-semibold">发布当前研究成果</h3>{!run || stale || !run.publishable ? <Empty title="当前没有可发布的结果"><p>需要当前参数对应的真实运行通过验证。</p><button type="button" className={`${buttonClass} mt-3`} onClick={() => setStep(1)}>查看计算与验证</button></Empty> : <><p className="text-sm text-slate-600">{run.name} · 数据截至 {run.data_as_of ?? run.as_of}。发布后保留本地不可变结果；修改模型只会产生新版本。</p><Field label="数据有效天数" hint="从数据截至日计算，不因重复发布而刷新；过期后需重新研究。"><input type="number" min="1" max="365" className={inputClass} value={validDays} disabled={Boolean(busy) || Boolean(publishedId)} onChange={event => setValidDays(Number(event.target.value))} /></Field><Field label="发布说明"><textarea className={inputClass} rows={2} value={note} disabled={Boolean(busy) || Boolean(publishedId)} onChange={event => setNote(event.target.value)} /></Field><label className="flex items-start gap-2 text-sm leading-6 text-slate-600"><input className="mt-1" type="checkbox" checked={acknowledged} disabled={Boolean(busy) || Boolean(publishedId)} onChange={event => setAcknowledged(event.target.checked)} />我已查看验证及数据来源，理解这些成果仅用于研究，不认证经济因果或历史时点可交易性。</label><div className="flex flex-wrap items-center gap-3"><button type="button" className={primaryClass} disabled={Boolean(busy) || !acknowledged || Boolean(publishedId)} onClick={() => void publish()}>{busy === 'publish' ? '发布中…' : publishedId ? '已发布' : '确认发布成果'}</button>{publishedId && <button type="button" className={buttonClass} onClick={() => setPane('published')}>查看已发布成果</button>}</div>{domain === 'product' && publishedId && <PublishedProductLinks releaseId={publishedId} targets={run.targets} />}</>}</section>}
          </div>}
      </>}
      {pane === 'published' && <div className="space-y-4">{!releases.length ? <Empty title="还没有已发布成果"><p>计算并验证研究后，在「发布成果」中确认发布。这里不会生成示例数据。</p><button type="button" className={`${primaryClass} mt-3`} onClick={() => setPane('research')}>开始研究</button></Empty> : <section className={sectionClass}><div className="overflow-x-auto"><table className="w-full min-w-[680px] text-sm"><caption className="mb-3 text-left font-semibold">已发布敏感性成果</caption><thead className="text-left text-xs text-slate-600"><tr><th scope="col" className="p-2">成果</th><th scope="col" className="p-2">数据截至</th><th scope="col" className="p-2">有效状态</th><th scope="col" className="p-2">操作</th></tr></thead><tbody className="divide-y divide-slate-100">{releases.map(release => <tr key={release.id}><td className="p-3"><p className="font-medium">{release.name}</p><p className="mt-1 text-xs text-slate-600">{frequencyLabels[release.frequency]} · {release.method === 'cashflow' ? '现金流定价' : '已训练系数'} · {release.target_keys.length ? `${release.target_keys.length} 个产品` : '宏观传导'}</p></td><td className="p-3">{release.data_as_of ?? release.as_of}</td><td className="p-3"><span>{statusLabels[release.status] ?? release.status}</span><p className="mt-1 text-xs text-slate-600">到期：{release.expires_at.slice(0, 10)}</p></td><td className="p-3"><div className="flex flex-wrap gap-2"><button type="button" className={buttonClass} disabled={Boolean(busy)} onClick={() => void inspect(release.run_id, release.id)}>查看成果</button>{release.status === 'active' && <>{domain === 'product' ? <Link className={buttonClass} to={`/settings/scenario-algorithms/apply?${new URLSearchParams({ exposure_release_id: release.id })}`}>应用压测</Link> : <Link className={buttonClass} to="/settings/scenario-algorithms?center=simulation">去构建情景</Link>}<button type="button" className="min-h-11 px-2 text-xs text-rose-700" disabled={Boolean(busy)} onClick={() => void retire(release)}>停用</button></>}</div></td></tr>)}</tbody></table></div></section>}{inspected && <><RiskRunView run={inspected} published releaseId={inspectedReleaseId} />{domain === 'product' && <PublishedProductLinks releaseId={inspectedReleaseId} targets={inspected.targets} />}</>}</div>}
      {pane === 'data' && <div className="space-y-4"><section className={sectionClass}><h3 className="font-semibold">共享变量目录</h3><p className="mt-2 text-sm text-slate-600">宏观传导和产品风险模型引用同一套变量身份。因子名称相似，不代表单位和来源相同。</p><div className="mt-4 max-h-96 overflow-auto"><table className="w-full min-w-[600px] text-sm"><thead className="text-left text-xs text-slate-600"><tr><th scope="col" className="p-2">变量</th><th scope="col" className="p-2">单位 / 频率</th><th scope="col" className="p-2">数据状态</th></tr></thead><tbody className="divide-y divide-slate-100">{catalog.variables.map(variable => <tr key={variable.id}><td className="p-3"><p className="font-medium">{variable.name}</p><p className="mt-1 break-all text-xs text-slate-600">{variable.id}</p></td><td className="p-3">{variable.unit_label} / {frequencyLabels[variable.frequency]}</td><td className="p-3 text-xs text-slate-600">{variable.availability?.reason}</td></tr>)}</tbody></table></div><details className="mt-3 text-xs text-slate-600"><summary className="cursor-pointer">本地存储位置</summary><p className="mt-2 break-all">{catalog.storage.logical_path}</p><p className="mt-1">跟随统一数据磁盘；磁盘离线时不会回落到另一个目录。</p></details></section><VariableImport domain={domain} onImported={() => setRefresh(old => old + 1)} /></div>}
    </>}
  </div>
}
