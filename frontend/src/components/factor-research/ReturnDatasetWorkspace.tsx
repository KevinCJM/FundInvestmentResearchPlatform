import { useCallback, useEffect, useState } from 'react'
import { factorApi, type FactorDataset, type ReturnCatalog, type ReturnDataset } from '../../services/factorResearch'
import { Card, Field, inputClass, secondaryClass, type Action } from './shared'
import { downloadJson, readJsonFile, returnMethodName } from './returnShared'
import ReturnDatasetView from './ReturnDatasetView'

export default function ReturnDatasetWorkspace({ action, busy, datasetId, onSelect, onAttribute }: { action: Action; busy: boolean; datasetId?: string; onSelect: (id: string) => void; onAttribute: (id: string) => void }) {
  const [datasets, setDatasets] = useState<FactorDataset[]>([])
  const [catalog, setCatalog] = useState<ReturnCatalog>()
  const [selected, setSelected] = useState<ReturnDataset>()
  const refresh = useCallback(async () => {
    const [list, info] = await Promise.all([factorApi.datasets(), factorApi.returnCatalog()])
    setDatasets(list.items); setCatalog(info)
  }, [])
  useEffect(() => { void action('加载收益率数据集', refresh) }, [action, refresh])
  useEffect(() => {
    let active = true
    setSelected(undefined)
    if (datasetId) void action('读取收益率数据集', async () => {
      const value = await factorApi.getReturnDataset(datasetId)
      if (active) setSelected(value)
    })
    return () => { active = false }
  }, [datasetId, action])
  return <div className="min-w-0 space-y-5">
    <Card title="收益率数据集">
      <p className="mb-4 text-sm leading-6 text-slate-600">统一管理自行构建和外部导入的因子收益。每份数据集有独立 ID、来源和回归口径，不与产品评分混用。</p>
      <Field label="选择因子收益数据集"><select className={inputClass} disabled={busy} value={datasetId || ''} onChange={event => onSelect(event.target.value)}><option value="">选择数据集查看收益与证据</option>{datasets.map(dataset => <option key={dataset.id} value={dataset.id}>{dataset.name} · {returnMethodName(dataset.source_method)} · {dataset.market}/{dataset.currency}</option>)}</select></Field>
      {datasets.length ? <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[630px] text-left text-sm" aria-label="收益率数据集目录"><thead className="text-xs text-slate-600"><tr><th scope="col" className="p-2">数据集</th><th scope="col">算法 / 来源</th><th scope="col">市场</th><th scope="col">因子列</th><th scope="col">覆盖区间</th><th scope="col">天数</th></tr></thead><tbody>{datasets.map(dataset => <tr key={dataset.id} className="border-t border-slate-100"><th scope="row" className="p-2 font-medium"><button className="text-left text-accent-700 hover:underline disabled:opacity-50" disabled={busy} onClick={() => onSelect(dataset.id)}>{dataset.name}</button></th><td className="p-2">{returnMethodName(dataset.source_method)}</td><td className="p-2">{dataset.market}/{dataset.currency}</td><td className="p-2">{(dataset.factor_names || ['MKT_RF', 'SMB', 'HML']).join(', ')}</td><td className="whitespace-nowrap p-2 text-xs">{dataset.start_date}<br />{dataset.end_date}</td><td className="p-2">{dataset.observations}</td></tr>)}</tbody></table></div> : <p className="mt-4 text-sm text-slate-600">尚无数据集。可在“构建工作台”生成，或在下方导入已有的真实因子收益。</p>}
    </Card>
    <Card title="导入已有因子收益">
      <p className="mb-4 text-sm leading-6 text-slate-600">支持原有 FF3 平铺 JSON，也支持自定义列的通用格式。日收益 1% 填 0.01；超额收益回归必须包含逐日 RF，不能用年化利率替代。</p>
      <Field label="导入因子收益 JSON" hint="最多8MB、30–6000行、1–8个因子；日期升序且唯一。"><input type="file" accept=".json,application/json" className="block w-full text-sm" disabled={busy} onChange={event => {
        const file = event.target.files?.[0]; event.target.value = ''; if (!file) return
        void action('导入因子收益数据集', async () => {
          const value = await readJsonFile(file)
          const generic = typeof value === 'object' && value !== null && 'factor_names' in value
          const saved = generic ? await factorApi.importReturnDataset(value) : await factorApi.importDataset(value)
          await refresh(); onSelect(saved.id)
        })
      }} /></Field>
      <details className="mt-4"><summary className="cursor-pointer text-sm font-semibold text-slate-700">查看通用导入格式</summary><p className="mt-3 text-xs leading-6 text-slate-600">factor_names 指定列名；dependent_return 为 excess 时每行必须另外包含 RF，为 total 时只提供因子列。rows 的结构为：date 与 values，values 内放因子名对应的小数收益。缺失可填 null，不自动填零。旧 FF3 的 rows 直接包含 MKT_RF、SMB、HML、RF，也可继续导入。</p><pre className="mt-3 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-slate-50 p-3 text-xs">{JSON.stringify(catalog?.dataset_template || {}, null, 2)}</pre><button className={secondaryClass + ' mt-3'} disabled={!catalog || busy} onClick={() => downloadJson('factor-return-template.json', catalog?.dataset_template)}>下载收益率空模板</button></details>
    </Card>
    {selected && <ReturnDatasetView dataset={selected} onAttribute={onAttribute} busy={busy} />}
  </div>
}
