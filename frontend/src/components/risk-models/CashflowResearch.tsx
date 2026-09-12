import { useRef, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { previewCashflow, publishCashflowPreview, type CashflowStudy, type RiskRun } from '../../services/riskModels'
import RiskRunView from './RiskRunView'
import { buttonClass, Feedback, Field, inputClass, primaryClass, sectionClass, today } from './ResearchUI'

export default function CashflowResearch({ onPublished }: { onPublished: () => void }) {
  const [name, setName] = useState('')
  const [productId, setProductId] = useState('')
  const [asOf, setAsOf] = useState(today)
  const [yieldText, setYieldText] = useState('')
  const [compounding, setCompounding] = useState<CashflowStudy['compounding']>(2)
  const [source, setSource] = useState('')
  const [flows, setFlows] = useState([{ years: '', amount: '' }])
  const [run, setRun] = useState<RiskRun | null>(null)
  const [computedSignature, setComputedSignature] = useState('')
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [acknowledged, setAcknowledged] = useState(false)
  const [releaseId, setReleaseId] = useState('')
  const alive = useRef(true)
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  const signature = JSON.stringify([name, productId, asOf, yieldText, compounding, source, flows])
  const stale = signature !== computedSignature
  const study = (): CashflowStudy => ({ name, product_id: productId, as_of: asOf, source_label: source, yield_factor_id: 'cn-gov-yield-bp', yield_percent: Number(yieldText), compounding, cashflows: flows.map(flow => ({ years: Number(flow.years), amount: Number(flow.amount) })) })
  async function calculate() {
    setBusy('run'); setError(''); setNotice(''); setReleaseId(''); setAcknowledged(false)
    try {
      if (!name.trim() || !productId.trim() || !source.trim() || yieldText === '' || flows.some(flow => !flow.years || !flow.amount)) throw new Error('请填写名称、债券标识、来源、收益率和每笔现金流。')
      const result = await previewCashflow(study())
      if (alive.current) { setRun(result); setComputedSignature(signature); setNotice('计算完成。当前结果只保留在本页面；确认发布后才会写入磁盘。') }
    } catch (caught) { if (alive.current) setError(caught instanceof Error ? caught.message : '现金流研究失败。') }
    finally { if (alive.current) setBusy('') }
  }
  async function publish() {
    if (!run || stale || !run.preview_hash || !acknowledged) return
    setBusy('publish'); setError('')
    try {
      const release = await publishCashflowPreview(study(), run.preview_hash, 30, '固定现金流、单期收益率冲击；研究者确认现金流与估值假设。')
      if (alive.current) { setReleaseId(release.id); setNotice('已确认发布并写入磁盘；后续压测直接使用冻结现金流，不会重新研究。'); onPublished() }
    } catch (caught) { if (alive.current) setError(caught instanceof Error ? caught.message : '发布失败。') }
    finally { if (alive.current) setBusy('') }
  }
  return <div className="space-y-4"><section className={`${sectionClass} space-y-4`}><div><h3 className="font-semibold">根据现金流计算久期与凸度</h3><p className="mt-1 text-sm leading-6 text-slate-600">适用于已知、确定的债券现金流。不能只凭基金名称计算债券基金久期，也不包含提前赎回或违约。</p></div><Feedback error={error} notice={notice} /><fieldset disabled={Boolean(busy)} className="min-w-0 space-y-4"><div className="grid gap-4 sm:grid-cols-2"><Field label="研究名称"><input className={inputClass} value={name} onChange={event => setName(event.target.value)} /></Field><Field label="债券标识" hint="自有债券代码，不与 ETF 或基金代码混用。"><input className={inputClass} value={productId} onChange={event => setProductId(event.target.value)} /></Field><Field label="现金流估值日"><input className={inputClass} type="date" value={asOf} max={today()} onChange={event => setAsOf(event.target.value)} /></Field><Field label="基准到期收益率（%）"><input className={inputClass} type="number" step="0.01" value={yieldText} onChange={event => setYieldText(event.target.value)} /></Field><Field label="每年复利次数"><select className={inputClass} value={compounding} onChange={event => setCompounding(Number(event.target.value) as CashflowStudy['compounding'])}>{[1, 2, 4, 12].map(value => <option key={value} value={value}>{value} 次</option>)}</select></Field><Field label="现金流来源"><input className={inputClass} value={source} onChange={event => setSource(event.target.value)} /></Field></div><div className="overflow-x-auto"><table className="w-full text-sm"><caption className="mb-2 text-left font-medium">未来现金流（相同计价单位）</caption><thead className="text-left text-xs text-slate-600"><tr><th scope="col" className="p-2">距估值日的年数</th><th scope="col" className="p-2">收到的金额</th><th scope="col" className="p-2">操作</th></tr></thead><tbody>{flows.map((flow, index) => <tr key={index}><td className="p-1"><input aria-label={`第${index + 1}笔现金流年数`} className={inputClass} type="number" min="0" step="0.01" value={flow.years} onChange={event => setFlows(old => old.map((item, position) => position === index ? { ...item, years: event.target.value } : item))} /></td><td className="p-1"><input aria-label={`第${index + 1}笔现金流金额`} className={inputClass} type="number" min="0" step="0.01" value={flow.amount} onChange={event => setFlows(old => old.map((item, position) => position === index ? { ...item, amount: event.target.value } : item))} /></td><td className="p-1"><button type="button" className={buttonClass} disabled={flows.length === 1} onClick={() => setFlows(old => old.filter((_, position) => position !== index))}>移除</button></td></tr>)}</tbody></table></div><button type="button" className={buttonClass} disabled={flows.length >= 600} onClick={() => setFlows(old => [...old, { years: '', amount: '' }])}>添加现金流</button></fieldset><p className="rounded-lg bg-slate-50 p-3 text-xs leading-5 text-slate-600">这里的计算结果不会自动保存。只有确认发布后，现金流、估值结果和计算证据才写入统一数据磁盘。</p><button type="button" className={primaryClass} disabled={Boolean(busy)} onClick={() => void calculate()}>{busy === 'run' ? '计算中…' : '计算价格、久期与凸度'}</button></section>{run && <><RiskRunView run={run} /><section className={`${sectionClass} space-y-3`}>{stale ? <p className="text-sm text-amber-800">输入已变化，请重新计算后发布。</p> : <><label className="flex items-start gap-2 text-sm text-slate-600"><input className="mt-1" type="checkbox" checked={acknowledged} disabled={Boolean(busy) || Boolean(releaseId)} onChange={event => setAcknowledged(event.target.checked)} />我已核对现金流、收益率及估值日期，理解这只支持确定现金流的单期压测。</label><button type="button" className={primaryClass} disabled={Boolean(busy) || !acknowledged || Boolean(releaseId)} onClick={() => void publish()}>{busy === 'publish' ? '发布中…' : releaseId ? '已发布' : '确认发布估值成果'}</button>{releaseId && <Link className="ml-3 text-sm text-accent-800 underline" to={`/settings/scenario-algorithms/apply?${new URLSearchParams({ product_key: `bond:${productId}`, exposure_release_id: releaseId })}`}>使用情景测试这只债券</Link>}</>}</section></>}</div>
}
