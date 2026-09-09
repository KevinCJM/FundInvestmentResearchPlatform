import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import FactorLibrary from '../components/factor-research/FactorLibrary'
import StudyEditor from '../components/factor-research/StudyEditor'
import ResearchResults from '../components/factor-research/ResearchResults'
import AttributionWorkbench from '../components/factor-research/AttributionWorkbench'
import ReleaseWorkbench from '../components/factor-research/ReleaseWorkbench'
import ReturnConstructionWorkbench from '../components/factor-research/ReturnConstructionWorkbench'
import ReturnDatasetWorkspace from '../components/factor-research/ReturnDatasetWorkspace'
import { buttonClass, inputClass, secondaryClass, type Action } from '../components/factor-research/shared'
import { factorApi, type FactorRelease, type FactorRun, type ResearchCatalog, type RunRecord, type Study } from '../services/factorResearch'

type Module = 'characteristics' | 'returns'
type Tab = 'library' | 'workbench' | 'releases' | 'construct' | 'datasets' | 'attribution'
const tabs: Record<Module, ReadonlyArray<readonly [Tab, string]>> = {
  characteristics: [['library', '因子库'], ['workbench', '研究工作台'], ['releases', '发布与应用']],
  returns: [['construct', '构建工作台'], ['datasets', '收益率数据集'], ['attribution', '收益归因']],
}
const modules = [
  { id: 'characteristics', name: '产品特征因子', description: '计算特征 → 横截面检验 → 产品筛选' },
  { id: 'returns', name: '因子收益率', description: '构建收益序列 → 数据集 → 收益归因' },
] as const
function tabKeyDown(event: KeyboardEvent<HTMLButtonElement>, index: number) {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
  const buttons = Array.from(event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('[role="tab"]') || [])
  if (!buttons.length) return
  event.preventDefault()
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length
  buttons[next].click(); buttons[next].focus()
}

export default function FactorResearchCenter() {
  const [params, setParams] = useSearchParams()
  const module: Module = params.get('module') === 'returns' ? 'returns' : 'characteristics'
  const requestedTab = params.get('tab')
  const tab: Tab = tabs[module].some(([id]) => id === requestedTab) ? requestedTab as Tab : module === 'returns' ? 'construct' : 'workbench'
  const [catalog, setCatalog] = useState<ResearchCatalog>()
  const [studies, setStudies] = useState<Study[]>([])
  const [study, setStudy] = useState<Study>()
  const [runs, setRuns] = useState<RunRecord[]>([])
  const [run, setRun] = useState<FactorRun>()
  const [releases, setReleases] = useState<FactorRelease[]>([])
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const pending = useRef(0)
  const action: Action = useCallback(async (label, work) => {
    pending.current += 1
    setBusy(label); setError(''); setMessage('')
    try { const value = await work(); setMessage(label + '已完成'); return value }
    catch (reason) { setError(reason instanceof Error ? reason.message : '操作失败，请重试。'); return undefined }
    finally { pending.current -= 1; if (!pending.current) setBusy('') }
  }, [])
  const refresh = useCallback(async () => {
    const results = await Promise.all([factorApi.catalog(), factorApi.studies(), factorApi.runs(), factorApi.releases()])
    setCatalog(results[0]); setStudies(results[1].items); setRuns(results[2].items); setReleases(results[3].items)
  }, [])
  useEffect(() => { void action('加载因子研究中心', refresh) }, [action, refresh])
  const runId = params.get('run')
  useEffect(() => {
    if (!runId) return
    let active = true
    void action('加载历史研究运行', async () => { const result = await factorApi.getRun(runId); if (active) setRun(result) })
    return () => { active = false }
  }, [runId, action])
  const navigate = (nextModule: Module, nextTab: Tab, patch: Record<string, string> = {}) => {
    const next = new URLSearchParams(params)
    next.set('module', nextModule); next.set('tab', nextTab)
    Object.entries(patch).forEach(([key, value]) => value ? next.set(key, value) : next.delete(key))
    setParams(next); setMessage(''); setError('')
  }
  const saved = (value: Study) => { setStudy(value); setRun(undefined); setStudies(previous => [value, ...previous.filter(item => item.id !== value.id)]) }
  const completed = (value: FactorRun) => {
    setRun(value); setRuns(previous => [{ id: value.id, name: value.name, study_id: value.study_id, study_revision: value.study_revision, created_at: value.created_at }, ...previous])
    setMessage('研究完成，结果已保存。')
  }
  const viewRun = (id: string) => navigate('characteristics', 'workbench', { run: id })
  const attribute = (id: string) => navigate('returns', 'attribution', { dataset: id })
  return <div className="min-w-0 space-y-5" data-testid="factor-research-center">
    <header className="rounded-2xl border border-slate-200 bg-white p-5 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-4"><div><p className="text-xs font-semibold tracking-widest text-indigo-600">共享研究能力</p><h2 className="mt-2 text-2xl font-bold text-slate-950">因子研究中心</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">按计算逻辑与研究产物区分：特征用于比较产品，收益序列用于研究收益来源。两者都可服务投前、投中与投后。</p></div><div className="rounded-lg bg-slate-50 px-3 py-2 text-xs leading-6 text-slate-600">数据截至 {catalog?.snapshot.latest_date || '待核对'}<br />{catalog?.ready ? '研究计算已就绪' : '正在检查研究环境'}</div></div>
      <div role="tablist" aria-label="因子研究模块" className="mt-5 grid gap-3 md:grid-cols-2">{modules.map((item, index) => <button key={item.id} type="button" role="tab" id={'factor-module-' + item.id} aria-label={item.name} aria-selected={module === item.id} aria-controls="factor-module-panel" tabIndex={module === item.id ? 0 : -1} disabled={Boolean(busy)} onKeyDown={event => tabKeyDown(event, index)} onClick={() => navigate(item.id, item.id === 'returns' ? 'construct' : 'workbench')} className={'min-h-24 rounded-xl border p-4 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ' + (module === item.id ? 'border-slate-900 bg-slate-900 text-white' : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50')}><span className="block text-base font-bold">{item.name}</span><span className={'mt-2 block text-xs leading-5 ' + (module === item.id ? 'text-slate-300' : 'text-slate-500')}>{item.description}</span></button>)}</div>
    </header>
    {busy && <p className="rounded-lg border border-indigo-200 bg-indigo-50 p-3 text-sm text-indigo-900" role="status">{busy}…{busy.includes('运行') ? '正在校验数据并计算，请保留此页面。' : ''}</p>}
    {error && <div role="alert" className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">{error}{!catalog && <button className="ml-3 underline" onClick={() => void action('加载因子研究中心', refresh)}>重试</button>}</div>}
    {message && !busy && <p role="status" className="text-sm text-emerald-800">{message}</p>}
    <section id="factor-module-panel" role="tabpanel" aria-labelledby={'factor-module-' + module} className="min-w-0 space-y-5">
      <nav role="tablist" aria-label="因子研究功能" className="flex flex-wrap gap-2">{tabs[module].map(([id, label], index) => <button role="tab" id={'factor-tab-' + id} aria-selected={tab === id} aria-controls="factor-tab-panel" tabIndex={tab === id ? 0 : -1} key={id} disabled={Boolean(busy)} className={tab === id ? buttonClass : secondaryClass} onKeyDown={event => tabKeyDown(event, index)} onClick={() => navigate(module, id)}>{label}</button>)}</nav>
      {catalog && <div id="factor-tab-panel" role="tabpanel" aria-labelledby={'factor-tab-' + tab} className="min-w-0">
        {tab === 'library' && <FactorLibrary factors={catalog.factors} action={action} busy={Boolean(busy)} onChanged={refresh} />}
        {tab === 'workbench' && <div className="space-y-5">
          <div className="grid gap-3 rounded-xl border border-slate-200 bg-white p-4 md:grid-cols-2"><label className="text-sm font-medium text-slate-700">研究方案<select aria-label="选择研究方案" className={inputClass + ' mt-2'} disabled={Boolean(busy)} value={study?.id || ''} onChange={event => { setStudy(studies.find(item => item.id === event.target.value)); setRun(undefined); navigate('characteristics', 'workbench', { run: '' }) }}><option value="">新建 · ETF 三因子模板</option>{studies.map(item => <option key={item.id} value={item.id}>{item.name} · v{item.revision}</option>)}</select></label><label className="text-sm font-medium text-slate-700">历史运行<select aria-label="选择历史研究运行" className={inputClass + ' mt-2'} disabled={Boolean(busy)} value={run?.id || ''} onChange={event => { if (event.target.value) viewRun(event.target.value) }}><option value="">选择已保存结果</option>{runs.map(item => <option key={item.id} value={item.id}>{item.name} · {item.created_at.slice(0, 16)}</option>)}</select></label></div>
          <details open={!run} className="rounded-xl border border-slate-200 bg-slate-50 p-3 sm:p-4"><summary className="mb-4 cursor-pointer text-sm font-semibold text-slate-800">研究方案设置</summary><StudyEditor seed={catalog.default_study} current={study} factors={catalog.factors} action={action} busy={Boolean(busy) || !catalog.ready} onSaved={saved} onRun={completed} /></details>
          {run ? <ResearchResults run={run} onPublish={() => navigate('characteristics', 'releases')} onBuildReturns={() => navigate('returns', 'construct', { source: run.id })} /> : <div className="rounded-xl border border-dashed border-slate-300 p-8 text-center text-sm leading-6 text-slate-500">尚未运行检验。设置研究对象、比较基准和因子权重后，点击“保存并运行检验”。</div>}
        </div>}
        {tab === 'releases' && <ReleaseWorkbench run={run} runs={runs} releases={releases} action={action} busy={Boolean(busy)} refresh={refresh} onViewRun={viewRun} />}
        {tab === 'construct' && <ReturnConstructionWorkbench runs={runs} action={action} busy={Boolean(busy)} sourceRunId={params.get('source') || undefined} onAttribute={attribute} />}
        {tab === 'datasets' && <ReturnDatasetWorkspace action={action} busy={Boolean(busy)} datasetId={params.get('dataset') || undefined} onSelect={id => navigate('returns', 'datasets', { dataset: id })} onAttribute={attribute} />}
        {tab === 'attribution' && <AttributionWorkbench seed={catalog.default_study} action={action} busy={Boolean(busy)} preferredDatasetId={params.get('dataset') || undefined} />}
      </div>}
    </section>
  </div>
}
