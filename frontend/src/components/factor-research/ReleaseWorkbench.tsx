import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { attachEvaluationPlan, createProductPool, getProductPool, listProductPools, type ProductPool } from '../../services/productPools'
import { factorApi, numberText, percentText, type FactorRelease, type FactorRun, type PortfolioFactorProfile, type ReleaseMonitor, type RunRecord } from '../../services/factorResearch'
import { buttonClass, Card, Field, inputClass, secondaryClass, type Action } from './shared'

export function todayIso(offset = 0) {
  const value = new Date()
  value.setDate(value.getDate() + offset)
  return [value.getFullYear(), String(value.getMonth() + 1).padStart(2, '0'), String(value.getDate()).padStart(2, '0')].join('-')
}
export const releaseState = { active: '可引用 · 研究用途', scheduled: '待生效', expired: '已到期', retired: '已停用', stale: '信号已过期' }
export default function ReleaseWorkbench({ run, runs, releases, action, busy, refresh, onViewRun }: {
  run?: FactorRun; runs: RunRecord[]; releases: FactorRelease[]; action: Action; busy: boolean; refresh: () => Promise<void>; onViewRun: (id: string) => void
}) {
  const [runId, setRunId] = useState(run?.id || '')
  const [name, setName] = useState(run ? run.name + ' · 研究版' : '')
  const [from, setFrom] = useState(todayIso())
  const [to, setTo] = useState(todayIso(30))
  const [note, setNote] = useState('用于研究复核与候选筛选；保留样本选择和净值模拟限制。')
  const [selected, setSelected] = useState(releases[0]?.id || '')
  const [pools, setPools] = useState<ProductPool[]>([])
  const [poolId, setPoolId] = useState('')
  const [poolName, setPoolName] = useState('因子研究候选池')
  const [topN, setTopN] = useState(4)
  const [imported, setImported] = useState<ProductPool>()
  const [monitor, setMonitor] = useState<ReleaseMonitor>()
  const [holdings, setHoldings] = useState('')
  const [asOf, setAsOf] = useState(todayIso())
  const [profile, setProfile] = useState<PortfolioFactorProfile>()
  useEffect(() => { if (run) { setRunId(run.id); setName(run.name + ' · 研究版') } }, [run?.id])
  useEffect(() => { if (!selected && releases.length) setSelected(releases[0].id) }, [releases, selected])
  useEffect(() => { void action('加载产品池', async () => setPools((await listProductPools()).items)) }, [])
  const release = releases.find(value => value.id === selected)
  return <div className="space-y-5">
    <Card title="发布不可变研究版本">
      <form onSubmit={event => { event.preventDefault(); void action('发布因子研究版本', async () => { const value = await factorApi.publish({ run_id: runId, name, effective_from: from, effective_to: to, note }); await refresh(); setSelected(value.id); return value }) }}>
        <fieldset disabled={busy} className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2"><Field label="发布来源运行"><select className={inputClass} required value={runId} onChange={e => setRunId(e.target.value)}><option value="">选择成功运行</option>{runs.map(item => <option key={item.id} value={item.id}>{item.name} · v{item.study_revision} · {item.created_at.slice(0, 10)}</option>)}</select></Field><Field label="发布名称"><input className={inputClass} required value={name} maxLength={80} onChange={e => setName(e.target.value)} /></Field></div>
          <div className="grid gap-4 sm:grid-cols-2"><Field label="发布生效日"><input className={inputClass} type="date" required value={from} onChange={e => setFrom(e.target.value)} /></Field><Field label="发布失效日"><input className={inputClass} type="date" required value={to} onChange={e => setTo(e.target.value)} /></Field></div>
          <Field label="研究结论与适用范围"><textarea className={inputClass} rows={3} value={note} onChange={e => setNote(e.target.value)} /></Field>
          <p className="text-sm leading-6 text-slate-600">发布锁定本次运行、因子修订和输入数据。发布表示结果可供研究引用；产品准入仍需要审核。</p>
          <button className={buttonClass} type="submit">确认发布研究版</button>
        </fieldset>
      </form>
    </Card>
    <Card title="发布目录与应用">
      <Field label="选择因子发布"><select className={inputClass} value={selected} onChange={e => { setSelected(e.target.value); setMonitor(undefined); setProfile(undefined); setImported(undefined) }}><option value="">选择发布版本</option>{releases.map(item => <option key={item.id} value={item.id}>{item.name} · {releaseState[item.state]}</option>)}</select></Field>
      {release ? <div className="mt-4 space-y-4">
        <div className="rounded-lg bg-indigo-50 p-3 text-sm leading-6 text-indigo-950"><strong>{release.name}</strong><br />得分日 {release.as_of} · 有效期 {release.effective_from} 至 {release.effective_to}<br />{releaseState[release.state]}<p className="mt-1">{release.note}</p></div>
        <div className="flex flex-wrap gap-2"><button className={secondaryClass} onClick={() => onViewRun(release.run_id)}>查看来源检验</button><button className={secondaryClass} disabled={busy} onClick={() => void action('检查因子发布', async () => setMonitor(await factorApi.monitor(release.id)))}>检查更新与漂移</button>{release.state !== 'retired' && <button className={secondaryClass} disabled={busy} onClick={() => void action('停用因子发布', async () => { await factorApi.retire(release.id); await refresh() })}>停用此发布</button>}</div>
        {monitor && <div className="rounded-lg border border-slate-200 p-3 text-sm leading-6" role="status"><p>{monitor.data_changed_since_run ? '数据已更新，请重新运行研究方案。' : '最近运行与当前数据快照一致。'}</p><p>最新运行时间：{monitor.latest_run_at}</p>{monitor.comparable ? <p>同口径得分相关性 {numberText(monitor.drift?.score_correlation)} · 平均绝对变化 {numberText(monitor.drift?.mean_absolute_score_change)} · 当前覆盖 {percentText(monitor.drift?.coverage)}</p> : <p>方案或因子版本已变化，两个运行不能直接比较得分。</p>}<p>已登记 {monitor.bindings.length} 个投研引用。</p>{monitor.bindings.map((binding, i) => <p className="break-all text-xs text-slate-500" key={i}>{binding.context_type} · {binding.context_id} · {binding.note}</p>)}</div>}
        <fieldset disabled={busy || release.state !== 'active'} className="space-y-3 rounded-lg border border-slate-200 p-4">
          <h4 className="font-semibold text-slate-800">进入产品池审核</h4>
          <Field label="导入产品池"><select className={inputClass} value={poolId} onChange={e => setPoolId(e.target.value)}><option value="">新建研究候选池</option>{pools.filter(pool => pool.state !== 'archived').map(pool => <option key={pool.id} value={pool.id}>{pool.name}</option>)}</select></Field>
          {!poolId && <Field label="新候选池名称"><input className={inputClass} value={poolName} onChange={e => setPoolName(e.target.value)} /></Field>}
          <Field label="导入排名前 N 个"><input className={inputClass} type="number" min={1} max={120} value={topN} onChange={e => setTopN(Number(e.target.value))} /></Field>
          <button className={buttonClass} type="button" onClick={() => void action('导入产品池候选', async () => {
            if (!Number.isInteger(topN) || topN < 1 || topN > 120) throw new Error('导入数量须为1–120之间的整数。')
            if (!poolId && !poolName.trim()) throw new Error('请输入新候选池名称。')
            const pool = poolId ? await getProductPool(poolId) : await createProductPool({ name: poolName.trim(), description: release.name, purpose: '因子研究候选审核', owner: '' })
            const result = await attachEvaluationPlan(pool.id, { revision: pool.revision, plan_id: release.id, selection_mode: 'top_n', selection_value: topN })
            setImported(result); setPoolId(result.id); setPools((await listProductPools()).items)
          })}>导入待审核候选</button>
        </fieldset>
        {imported && <p className="rounded-lg bg-emerald-50 p-3 text-sm text-emerald-900" role="status">已导入“{imported.name}”，保留人工准入审核。<Link className="ml-2 font-semibold underline" to="/product-research/pools">前往产品池</Link></p>}
      </div> : <p className="mt-4 text-sm text-slate-500">完成研究检验后发布，即可被各投研环节引用。</p>}
    </Card>
    {release && <Card title="持仓因子画像">
      <p className="mb-4 text-sm text-slate-600">输入真实研究持仓权重，查看覆盖部分的加权特征；缺失持仓权重会明确显示。</p>
      <form onSubmit={event => { event.preventDefault(); void action('计算持仓因子画像', async () => {
        const rows = holdings.trim().split(/\n/).filter(Boolean).map(line => { const [product_id, weight] = line.trim().split(/[,，\s]+/); return { product_id, weight: Number(weight) } })
        if (rows.some(row => !Number.isFinite(row.weight))) throw new Error('每行填写完整产品代码和小数权重。')
        setProfile(await factorApi.profile(release.id, rows, asOf))
      }) }}><fieldset disabled={busy || release.state !== 'active'} className="space-y-3"><Field label="持仓代码与权重" hint="每行：产品代码,小数权重；总权重不得超过1。"><textarea className={inputClass} required rows={4} placeholder={'510300.SH,0.6\n510500.SH,0.4'} value={holdings} onChange={e => setHoldings(e.target.value)} /></Field><Field label="持仓研究日"><input className={inputClass} type="date" required value={asOf} onChange={e => setAsOf(e.target.value)} /></Field><button type="submit" className={buttonClass}>计算加权因子画像</button></fieldset></form>
      {profile && <div className="mt-4 space-y-2">{profile.factors.map(item => <p key={item.name} className="rounded-lg bg-slate-50 p-3 text-sm">{item.name}：{numberText(item.value)} · 覆盖持仓权重 {percentText(item.covered_weight)}</p>)}<p className="text-xs text-slate-500">{profile.meaning}</p></div>}
    </Card>}
  </div>
}
