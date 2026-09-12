import { useEffect, useRef, useState } from 'react'
import { factorApi, type FactorRun, type ReturnCatalog, type ReturnDataset, type ReturnPlan, type ReturnPlanDraft, type ReturnSource, type RunRecord } from '../../services/factorResearch'
import { buttonClass, Card, Field, inputClass, secondaryClass, type Action } from './shared'
import { downloadJson, readJsonFile, returnPlanDraft } from './returnShared'
import ReturnDatasetView from './ReturnDatasetView'

interface Props { runs: RunRecord[]; action: Action; busy: boolean; sourceRunId?: string; onAttribute: (id: string) => void }
export default function ReturnConstructionWorkbench({ runs, action, busy, sourceRunId, onAttribute }: Props) {
  const [catalog, setCatalog] = useState<ReturnCatalog>()
  const [plans, setPlans] = useState<ReturnPlan[]>([])
  const [sources, setSources] = useState<ReturnSource[]>([])
  const [current, setCurrent] = useState<ReturnPlan>()
  const [draft, setDraft] = useState<ReturnPlanDraft>(() => ({ ...returnPlanDraft(), source_run_id: sourceRunId || null }))
  const [parent, setParent] = useState<FactorRun>()
  const [result, setResult] = useState<ReturnDataset>()
  const form = useRef<HTMLFormElement>(null)
  useEffect(() => {
    let active = true
    void action('加载收益率构建目录', async () => {
      const [info, saved, panels] = await Promise.all([factorApi.returnCatalog(), factorApi.returnPlans(), factorApi.returnSources()])
      if (active) { setCatalog(info); setPlans(saved.items); setSources(panels.items) }
    })
    return () => { active = false }
  }, [action])
  useEffect(() => {
    let active = true
    setParent(undefined)
    if (draft.source_run_id) void action('读取冻结特征运行', async () => {
      const value = await factorApi.getRun(draft.source_run_id!)
      if (active) setParent(value)
    })
    return () => { active = false }
  }, [draft.source_run_id, action])
  const change = <K extends keyof ReturnPlanDraft>(key: K, value: ReturnPlanDraft[K]) => {
    setResult(undefined); setDraft(previous => ({ ...previous, [key]: value }))
  }
  const selectPlan = (id: string) => {
    const selected = plans.find(plan => plan.id === id)
    setCurrent(selected); setDraft(returnPlanDraft(selected)); setResult(undefined)
  }
  const save = (calculate: boolean) => {
    if (!form.current?.reportValidity()) return
    void action(calculate ? '运行因子收益率构建' : '保存收益率方案', async () => {
      const saved = await factorApi.saveReturnPlan(draft, current)
      setCurrent(saved); setPlans(previous => [saved, ...previous.filter(plan => plan.id !== saved.id)])
      if (calculate) setResult(await factorApi.runReturnPlan(saved))
      return saved
    })
  }
  const panel = sources.find(source => source.id === draft.source_panel_id)
  return <div className="min-w-0 space-y-5">
    <Card title="构建因子收益率">
      <p className="mb-4 text-sm leading-6 text-slate-600">先选构造算法，再锁定原始数据，生成逐日因子收益。保存方案方便修订；每次运行产生独立数据集，不覆盖历史。</p>
      <Field label="收益率构建方案"><select className={inputClass} disabled={busy} value={current?.id || ''} onChange={event => selectPlan(event.target.value)}><option value="">新建构建方案</option>{plans.map(plan => <option key={plan.id} value={plan.id}>{plan.name} · v{plan.revision}</option>)}</select></Field>
    </Card>
    <form ref={form} onSubmit={event => { event.preventDefault(); save(false) }}>
      <fieldset disabled={busy || !catalog} className="min-w-0 space-y-5">
        <Card title="1 · 算法与产物">
          <div className="grid gap-4 md:grid-cols-2"><Field label="构建方案名称"><input className={inputClass} required maxLength={80} value={draft.name} onChange={e => change('name', e.target.value)} /></Field><Field label="收益率构造算法"><select className={inputClass} value={draft.method} onChange={e => { setResult(undefined); setDraft(previous => ({ ...previous, method: e.target.value as ReturnPlanDraft['method'], source_run_id: null, source_panel_id: null, factor_key: 'composite', cost_bps: 0 })) }}><option value="characteristic_spread">特征分组 · 高分收益 − 低分收益</option><option value="ff3_2x3">FF3 风格 · 规模 × 价值双排序</option></select></Field></div>
          <p className="mt-4 rounded-lg bg-accent-50 p-3 text-sm leading-6 text-accent-900">{draft.method === 'characteristic_spread' ? '产物：一条特征收益差额序列。使用当时已计算的特征分组，下一交易日收盘进入，按每日真实净值变化计算持有收益，不拼接重叠的远期标签。' : '产物：MKT_RF、SMB、HML、RF。使用上一财年账面权益、上一年12月市值及6月市值独立双排序，形成六组；这是自定义市场与参考池的 FF3 风格构造。'}</p>
        </Card>
        {draft.method === 'characteristic_spread' ? <Card title="2 · 冻结特征数据与分组">
          <div className="space-y-4">
            <Field label="来源特征运行" hint="只读取已完成运行的冻结数据，不读取后续替换的行情快照。"><select className={inputClass} required value={draft.source_run_id || ''} onChange={e => { setResult(undefined); setDraft(previous => ({ ...previous, source_run_id: e.target.value || null, factor_key: 'composite' })) }}><option value="">选择已完成的特征研究</option>{runs.map(run => <option value={run.id} key={run.id}>{run.name} · {run.created_at.slice(0, 16)}</option>)}</select></Field>
            {!runs.length && <p className="text-sm text-slate-600">还没有特征研究运行。请先在“产品特征因子 → 研究工作台”完成一次检验。</p>}
            <div className="grid gap-4 sm:grid-cols-2"><Field label="用于分组的特征"><select className={inputClass} value={draft.factor_key} onChange={e => change('factor_key', e.target.value)}><option value="composite">综合得分</option>{parent?.factor_snapshots.map(factor => <option value={factor.id} key={factor.id}>{factor.name} · v{factor.revision}</option>)}</select></Field><Field label="输出因子列名" hint="大写英文字母、数字、下划线；不要用 SMB/HML 冒充 FF3。"><input className={inputClass} required pattern="[A-Z][A-Z0-9_]{0,31}" maxLength={32} value={draft.output_factor} onChange={e => change('output_factor', e.target.value.toUpperCase())} /></Field><Field label="收益率分组数" hint="末组减首组；边界并列不按代码拆开，空组不伪造收益。"><input className={inputClass} type="number" required min={2} max={10} step={1} value={draft.quantiles} onChange={e => change('quantiles', Number(e.target.value))} /></Field><Field label="扣费诊断 · 单边费用（bp）" hint="因子列保留毛收益；另列双边换手和扣费差额。"><input className={inputClass} type="number" required min={0} max={500} step="any" value={draft.cost_bps} onChange={e => change('cost_bps', Number(e.target.value))} /></Field></div>
            {parent && <p className="break-all text-xs leading-6 text-slate-600">来源：{parent.study_snapshot.start_date} — {parent.as_of}；{parent.latest_scores.length} 个产品；信号频率：{({ daily: '日频', weekly: '周频', monthly: '月频' })[parent.study_snapshot.signal_frequency || 'monthly']}。<br />冻结输入：{parent.input_checksum}</p>}
            <p className="text-xs leading-6 text-slate-600">多头名义 1、空头名义 1。场外基金可能不可做空，未计借券、融资和容量；这是一条研究因子，不是自动交易策略。</p>
          </div>
        </Card> : <Card title="2 · 股票时点面板与无风险收益">
          <div className="space-y-4">
            <p className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm leading-6 text-amber-900">本地股票行情、财务时点和退市覆盖尚未验收，因此不自动取本地股票构建。请导入有明确来源的股票面板；没有原始数据时，不生成虚假结果。</p>
            <Field label="FF3 原始面板"><select className={inputClass} required value={draft.source_panel_id || ''} onChange={e => change('source_panel_id', e.target.value || null)}><option value="">选择已导入的股票时点面板</option>{sources.map(source => <option key={source.id} value={source.id}>{source.name} · {source.market}/{source.currency} · {source.start_date}—{source.end_date}</option>)}</select></Field>
            <Field label="导入 FF3 股票时点 JSON" hint="最多32MB；导入验证公告日期、前日市值、完整 RF 和交易日历。"><input className="block w-full text-sm" type="file" accept=".json,application/json" onChange={event => { const file = event.target.files?.[0]; event.target.value = ''; if (!file) return; void action('校验并导入股票时点面板', async () => { const source = await factorApi.importReturnSource(await readJsonFile(file, 32_000_000)); setSources(previous => [source, ...previous]); change('source_panel_id', source.id) }) }} /></Field>
            {panel && <p className="text-xs leading-6 text-slate-600">{panel.observations} 条股票收益记录；{panel.market}/{panel.currency}。输入哈希：<span className="break-all">{panel.checksum}</span></p>}
            <details className="rounded-lg border border-slate-200 p-3"><summary className="cursor-pointer text-sm font-semibold text-slate-700">查看输入字段与下载空模板</summary><div className="mt-3 space-y-3">{catalog && Object.entries(catalog.source_fields).map(([field, description]) => <p className="break-words text-xs leading-6 text-slate-600" key={field}><strong>{field}</strong>：{description}</p>)}<p className="text-xs leading-6 text-slate-600">B/M = 上一财年账面权益 ÷ 上年12月市值。形成日是6月最后交易日，财报须提前公告。RF 使用实际日收益，小数 0.01 表示 1%，不可直接填年化利率。收益应包含分红与退市损失。空模板不含模拟行情。</p><button className={secondaryClass} type="button" onClick={() => downloadJson('ff3-source-template.json', catalog?.source_template)}>下载股票面板空模板</button></div></details>
          </div>
        </Card>}
        <div className="flex flex-wrap items-center gap-3"><button type="submit" className={secondaryClass}>保存构建方案</button><button type="button" className={buttonClass} disabled={!catalog?.ready} onClick={() => save(true)}>保存并生成收益率</button>{current && <span className="text-xs text-slate-600">当前 v{current.revision}；再次保存会生成新修订。</span>}</div>
      </fieldset>
    </form>
    {result ? <ReturnDatasetView dataset={result} onAttribute={onAttribute} busy={busy} /> : <p className="rounded-xl border border-dashed border-slate-300 p-6 text-sm leading-6 text-slate-600">尚未生成本方案结果。生成后可查看逐日收益、统计、分组证据，并直接用于基金或 ETF 收益归因。</p>}
    <details className="rounded-xl border border-slate-200 bg-white p-4"><summary className="cursor-pointer text-sm font-semibold text-slate-700">其他算法与当前边界</summary>{catalog?.methods.filter(method => !method.available).map(method => <p className="mt-3 text-sm leading-6 text-slate-600" key={method.id}><strong>{method.name}</strong>：{method.reason}</p>)}</details>
  </div>
}
