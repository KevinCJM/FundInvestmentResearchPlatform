import { useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { factorApi, numberText, percentText, type ReturnDataset } from '../../services/factorResearch'
import { buttonClass, Card, secondaryClass } from './shared'
import { downloadBlob, downloadJson, returnMethodName } from './returnShared'

export default function ReturnDatasetView({ dataset, onAttribute, busy = false }: { dataset: ReturnDataset; onAttribute: (id: string) => void; busy?: boolean }) {
  const [exporting, setExporting] = useState(false)
  const [exportError, setExportError] = useState('')
  const exportCsv = async () => {
    setExporting(true); setExportError('')
    try { downloadBlob(dataset.id + '.csv', await factorApi.exportReturnCsv(dataset.id)) }
    catch (reason) { setExportError(reason instanceof Error ? reason.message : 'CSV 导出失败。') }
    finally { setExporting(false) }
  }
  const names = dataset.factor_names
  const columns = [...names, ...(dataset.dependent_return === 'excess' ? ['RF'] : [])]
  const diagnostics = dataset.diagnostics
  const number = (value: unknown) => typeof value === 'number' ? value : null
  const portable = {
    name: dataset.name, source_url: dataset.source_url, market: dataset.market, currency: dataset.currency,
    frequency: dataset.frequency, units: dataset.units, construction: dataset.construction,
    factor_names: names, dependent_return: dataset.dependent_return,
    rows: dataset.rows.map(row => ({ date: row.date, values: Object.fromEntries(columns.map(name => [name, row[name]])) })),
  }
  return <div className="min-w-0 space-y-5" aria-label="因子收益率结果">
    <Card title={dataset.name + ' · 因子收益率'}>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 text-sm leading-6 text-slate-600"><p>{returnMethodName(dataset.source_method)} · {dataset.market}/{dataset.currency} · {dataset.rows.length} 个收益日</p><p>{dataset.rows[0]?.date} — {dataset.rows[dataset.rows.length - 1]?.date}</p><p className="mt-2">{dataset.dependent_return === 'excess' ? '含逐日 RF，可解释产品超额收益。' : '不含 RF：解释产品总收益，不将截距称为风险调整 Alpha。'}</p></div>
        <div className="flex flex-wrap gap-2"><button className={buttonClass} disabled={busy} onClick={() => onAttribute(dataset.id)}>用于收益归因</button><button className={secondaryClass} disabled={busy || exporting} onClick={() => void exportCsv()}>{exporting ? '正在导出…' : '导出 CSV'}</button><button className={secondaryClass} onClick={() => downloadJson(dataset.id + '.json', portable)}>导出 JSON</button></div>
      </div>
      {exportError && <p role="alert" className="mt-3 text-sm text-rose-700">{exportError}</p>}
      <p className="mt-4 rounded-lg bg-slate-50 p-3 text-sm leading-6 text-slate-600">{dataset.construction}</p>
      {!dataset.source_url && <p className="mt-2 text-xs text-slate-500">内部构建的 JSON 保留空外部来源；重新外部导入时需填写可核查来源。内部归因直接引用此不可变数据集，无需导出再导入。</p>}
    </Card>
    {diagnostics && <>
      <Card title="收益率统计与相关性">
        <div className="overflow-x-auto"><table className="w-full min-w-[500px] text-left text-sm" aria-label="因子收益统计"><thead className="text-xs text-slate-500"><tr><th className="p-2">因子</th><th>有效天数</th><th>日均收益</th><th>日收益标准差</th><th>正收益比例</th></tr></thead><tbody>{diagnostics.factors.map(row => <tr key={row.factor} className="border-t border-slate-100"><th className="p-2">{row.factor}</th><td>{row.observations}</td><td>{percentText(row.mean)}</td><td>{percentText(row.std)}</td><td>{percentText(row.positive_rate)}</td></tr>)}</tbody></table></div>
        <details className="mt-4"><summary className="cursor-pointer text-sm font-semibold text-slate-700">查看因子收益相关矩阵</summary><div className="mt-2 overflow-x-auto"><table className="w-full text-left text-xs"><thead><tr><th className="p-2">因子</th>{names.map(name => <th className="p-2" key={name}>{name}</th>)}</tr></thead><tbody>{names.map((name, i) => <tr className="border-t border-slate-100" key={name}><th className="p-2">{name}</th>{diagnostics.correlation[i].map((value, j) => <td className="p-2" key={j}>{numberText(value)}</td>)}</tr>)}</tbody></table></div></details>
      </Card>
      <Card title="收益率累计观察">
        <ReactECharts style={{ height: 290, width: '100%' }} notMerge option={{
          tooltip: { trigger: 'axis' }, legend: { type: 'scroll', top: 0 },
          grid: { left: 55, right: 18, top: 45, bottom: 55 },
          xAxis: { type: 'category', boundaryGap: false, data: diagnostics.cumulative.map(row => row.date) },
          yAxis: { type: 'value', scale: true }, dataZoom: [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 8 }],
          series: names.map((name, i) => ({ name, type: 'line', showSymbol: false, connectNulls: false, data: diagnostics.cumulative.map(row => row.values[i]) })),
        }} />
        <p className="text-xs leading-5 text-slate-500">{diagnostics.cumulative_meaning} 纵轴为小数收益累计值；不展示未经证明的可投资净值或显著性。</p>
      </Card>
    </>}
    <Card title="逐日因子收益">
      <p className="mb-3 text-xs text-slate-500">预览前 80 行；导出包含全部日期。缺失值显示“—”，不补零。</p>
      <div className="max-h-80 overflow-auto"><table className="w-full min-w-[420px] text-left text-sm" aria-label="因子收益序列"><thead><tr><th className="p-2">日期</th>{columns.map(name => <th className="p-2" key={name}>{name}</th>)}</tr></thead><tbody>{dataset.rows.slice(0, 80).map(row => <tr className="border-t border-slate-100" key={row.date}><td className="whitespace-nowrap p-2">{row.date}</td>{columns.map(name => <td className="whitespace-nowrap p-2 tabular-nums" key={name}>{percentText(number(row[name]))}</td>)}</tr>)}</tbody></table></div>
    </Card>
    {dataset.formation_evidence && <Card title="FF3 分组证据"><div className="overflow-x-auto"><table className="w-full min-w-[640px] text-left text-xs"><thead><tr><th className="p-2">形成日</th><th>规模断点</th><th>B/M 30% / 70%</th><th>六组数量</th></tr></thead><tbody>{dataset.formation_evidence.map(row => <tr key={row.date} className="border-t border-slate-100"><td className="p-2">{row.date}</td><td>{numberText(row.size_break)}</td><td>{numberText(row.bm30)} / {numberText(row.bm70)}</td><td>{Object.entries(row.counts).map(([key, value]) => `${key}: ${value}`).join(' · ')}</td></tr>)}</tbody></table></div><p className="mt-3 text-xs text-slate-500">独立双排序；不是将 ETF 风格标签命名为 SMB/HML。完整成员证据保存在本次不可变运行中。</p></Card>}
    {dataset.leg_returns?.length ? <details className="rounded-xl border border-slate-200 bg-white p-4"><summary className="cursor-pointer text-sm font-semibold text-slate-700">查看组合腿收益、费用与换手</summary><div className="mt-3 max-h-72 overflow-auto"><table className="w-full text-left text-xs"><thead><tr>{Object.keys(dataset.leg_returns[0]).map(key => <th className="whitespace-nowrap p-2" key={key}>{key}</th>)}</tr></thead><tbody>{dataset.leg_returns.slice(0, 80).map(row => <tr key={row.date} className="border-t border-slate-100">{Object.entries(row).map(([key, value]) => <td className="whitespace-nowrap p-2" key={key}>{key === 'date' ? String(value) : numberText(number(value), 6)}</td>)}</tr>)}</tbody></table></div><p className="mt-3 text-xs leading-5 text-slate-500">收益、费用为小数；换手为双边绝对权重变化。特征差额展示毛因子收益及另列的扣费诊断，未计借券融资。</p></details> : null}
    <Card title="来源与适用范围"><p className="mb-3 break-all text-xs leading-5 text-slate-500">数据集：{dataset.id}<br />输入校验：{dataset.input_checksum || dataset.checksum || '历史版本未记录'}<br />{dataset.source_run_id && <>父特征运行：{dataset.source_run_id}<br /></>}{dataset.source_panel_id && <>原始时点面板：{dataset.source_panel_id}<br /></>}{dataset.source_url && <a className="text-indigo-700 underline" href={dataset.source_url} target="_blank" rel="noreferrer">查看原始来源</a>}</p>{dataset.warnings.map(warning => <p className="mt-2 text-sm leading-6 text-slate-600" key={warning}>{warning}</p>)}</Card>
  </div>
}
