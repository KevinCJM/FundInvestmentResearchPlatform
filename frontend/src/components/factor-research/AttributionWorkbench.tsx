import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { factorApi, parseCodes, type AttributionRequest, type AttributionRun, type FactorDataset, type StudyDraft } from '../../services/factorResearch'
import { buttonClass, Card, Field, inputClass, type Action } from './shared'
import { isFF3Dataset, returnMethodName } from './returnShared'
import AttributionResults from './AttributionResults'

export default function AttributionWorkbench({ seed, action, busy, preferredDatasetId }: { seed: StudyDraft; action: Action; busy: boolean; preferredDatasetId?: string }) {
  const [model, setModel] = useState<AttributionRequest['model']>('rbsa')
  const [exposureMode, setExposureMode] = useState<'fixed' | 'rolling'>('fixed')
  const [rolling, setRolling] = useState({ rolling_window: 126, min_observations: 60, refit_step: 21 })
  const [kind, setKind] = useState<'etf' | 'fund'>('fund')
  const [name, setName] = useState('基金收益风格研究')
  const [targets, setTargets] = useState('')
  const [indices, setIndices] = useState('000300.SH, 000905.SH, 000852.SH')
  const [datasetId, setDatasetId] = useState(preferredDatasetId || '')
  const [datasets, setDatasets] = useState<FactorDataset[]>([])
  const [dates, setDates] = useState({ start_date: seed.start_date, end_date: seed.end_date, oos_date: seed.oos_date })
  const [result, setResult] = useState<AttributionRun>()
  const [history, setHistory] = useState<Array<{ id: string; name: string }>>([])
  useEffect(() => {
    let active = true
    void action('加载归因数据', async () => {
      const [data, runs] = await Promise.all([factorApi.datasets(), factorApi.attributions()])
      if (!active) return
      setDatasets(data.items); setHistory(runs.items)
      if (preferredDatasetId) {
        const selected = data.items.find(item => item.id === preferredDatasetId)
        if (!selected) throw new Error('所选因子收益数据集不存在，请返回数据集目录核对。')
        setDatasetId(selected.id); setModel(isFF3Dataset(selected) ? 'ff3' : 'factor_regression'); setResult(undefined)
      }
    })
    return () => { active = false }
  }, [action, preferredDatasetId])
  const request: AttributionRequest = { name, product_kind: kind, targets: parseCodes(targets), model, indices: model === 'rbsa' ? parseCodes(indices) : [], ...(model !== 'rbsa' ? { dataset_id: datasetId } : {}), market: 'CN', currency: 'CNY', exposure_mode: exposureMode, ...rolling, ...dates }
  const selectedDataset = datasets.find(dataset => dataset.id === datasetId)
  const compatible = (dataset: FactorDataset) => dataset.market === 'CN' && dataset.currency === 'CNY' && (model !== 'ff3' || isFF3Dataset(dataset))
  return <div className="min-w-0 space-y-5">
    <Card title="风格与收益贡献研究">
      <p className="mb-4 text-sm leading-6 text-slate-600">复用同一套因子收益与暴露，分解各因子、截距和残差的收益贡献，并与实际收益对账。暴露不等同持仓，截距和残差不直接等同经理能力。</p>
      <form onSubmit={event => { event.preventDefault(); setResult(undefined); void action('运行收益归因', async () => { const value = await factorApi.attribution(request); setResult(value); setHistory((await factorApi.attributions()).items); return value }) }}>
        <fieldset disabled={busy} className="space-y-4" onChange={() => setResult(undefined)}>
          <div className="grid gap-4 md:grid-cols-2"><Field label="归因研究名称"><input className={inputClass} required maxLength={80} value={name} onChange={e => setName(e.target.value)} /></Field><Field label="归因模型"><select className={inputClass} value={model} onChange={e => { setModel(e.target.value as AttributionRequest['model']); setDatasetId('') }}><option value="rbsa">RBSA · 非负、权重和为1的收益风格分析</option><option value="ff3">FF3 · 市场、规模、价值三因子</option><option value="factor_regression">通用因子收益回归 · 自定义因子列</option></select></Field></div>
          <div className="grid gap-4 md:grid-cols-2"><Field label="归因产品类型"><select className={inputClass} value={kind} onChange={e => setKind(e.target.value as 'etf' | 'fund')}><option value="fund">场外基金</option><option value="etf">ETF</option></select></Field><Field label="归因产品代码" hint="最多40个；使用带市场后缀的完整代码。"><input className={inputClass} required placeholder="例如 000001.OF" value={targets} onChange={e => setTargets(e.target.value)} /></Field></div>
          {model === 'rbsa' ? <Field label="风格代理指数" hint="2–8个本地指数代码。优先选择覆盖充分、风格区分清楚的代理。"><input className={inputClass} required value={indices} onChange={e => setIndices(e.target.value)} /></Field> : <>
            <Field label={model === 'ff3' ? 'FF3 因子收益数据集' : '回归因子收益数据集'} hint={model === 'ff3' ? '必须包含 MKT_RF、SMB、HML、RF，且与产品同市场、同币种。' : '1–8个解释因子。数据集明确决定解释总收益还是超额收益，不能静默更换口径。'}><select className={inputClass} required value={datasetId} onChange={e => setDatasetId(e.target.value)}><option value="">选择已构建或导入的数据集</option>{datasets.map(data => <option key={data.id} value={data.id} disabled={!compatible(data)}>{data.name} · {data.market}/{data.currency} · {returnMethodName(data.source_method)}{!compatible(data) ? ' · 不匹配当前模型' : ''}</option>)}</select></Field>
            {selectedDataset && <p className="rounded-lg bg-accent-50 p-3 text-sm leading-6 text-accent-900">因子列：{(selectedDataset.factor_names || ['MKT_RF', 'SMB', 'HML']).join('、')}。{selectedDataset.dependent_return === 'total' ? '本次解释产品总收益，没有 RF；截距不是风险调整 Alpha。' : '本次解释产品扣除 RF 后的超额收益。'} 数据覆盖 {selectedDataset.start_date} — {selectedDataset.end_date}。</p>}
            <p className="text-xs leading-6 text-slate-600">没有合适数据时，先在“构建工作台”生成，或在 <Link className="text-accent-700 underline" to="?module=returns&tab=datasets">收益率数据集</Link> 导入；ETF 动量差额不能作为 FF3 使用。</p>
          </>}
          <Field label="暴露估计方式"><select className={inputClass} value={exposureMode} onChange={e => setExposureMode(e.target.value as 'fixed' | 'rolling')}><option value="fixed">固定暴露 · 样本内拟合，样本外保持不变</option><option value="rolling">滚动暴露 · 前期估计，逐步前推</option></select></Field>
          {exposureMode === 'rolling' ? <div className="rounded-lg border border-accent-200 bg-accent-50/30 p-4">
            <div className="grid gap-4 sm:grid-cols-3">{([['rolling_window', '滚动窗口（交易日）', 30, 1260], ['min_observations', '最少有效配对日数', 30, rolling.rolling_window], ['refit_step', '重新估计间隔（交易日）', 1, 252]] as const).map(([key, label, min, max]) => <Field key={key} label={label}><input className={inputClass} required type="number" step={1} min={min} max={max} value={rolling[key]} onChange={e => setRolling({ ...rolling, [key]: Number(e.target.value) })} /></Field>)}</div>
            <p className="mt-3 text-xs leading-6 text-slate-600">积累完整窗口后，只用此前的收益估计；缺失日不压缩，拟合失败不沿用旧模型。前推样本外会使用此前已过去的样本外数据，结果不保证实时 PIT 可交易性。</p>
          </div> : <p className="text-xs leading-6 text-slate-600">样本内是一组事后拟合暴露，不代表当时已经知道；样本外使用固定系数。两段贡献分别独立对账。</p>}
          <div className="grid gap-4 sm:grid-cols-3">{([['start_date', '归因起始日'], ['end_date', '归因截止日'], ['oos_date', '归因样本外起始日']] as const).map(([key, label]) => <Field key={key} label={label}><input className={inputClass} type="date" required value={dates[key]} onChange={e => setDates({ ...dates, [key]: e.target.value })} /></Field>)}</div>
          <button className={buttonClass} type="submit" disabled={model !== 'rbsa' && (!selectedDataset || !compatible(selectedDataset))}>运行归因研究</button>
        </fieldset>
      </form>
    </Card>
    <Field label="历史归因运行"><select className={inputClass} disabled={busy} value={result?.id || ''} onChange={e => { if (e.target.value) void action('加载归因结果', async () => setResult(await factorApi.getAttribution(e.target.value))) }}><option value="">选择已保存归因运行</option>{history.map(item => <option key={item.id} value={item.id}>{item.name} · {item.id.slice(-8)}</option>)}</select></Field>
    {result && <AttributionResults key={result.id} run={result} />}
  </div>
}
