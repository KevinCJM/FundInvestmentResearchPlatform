import { useState } from 'react'
import { importRiskSeries, type ResearchDomain, type RiskVariable, type SeriesImport } from '../../services/riskModels'
import { buttonClass, Feedback, Field, inputClass, primaryClass, sectionClass } from './ResearchUI'

const meanings = {
  price_return: { name: '价格或净值 → 每期收益率', unit: 'return', hint: 'value 填原始正价格或净值，系统计算相邻完整周期的收益率。' },
  percent_rate_change: { name: '百分数利率 → 每期基点变化', unit: 'bp', hint: '例如 2.5 表示利率 2.5%；系统将变动转换为 bp。' },
  difference: { name: '经济指标水平 → 每期变化', unit: 'pp', hint: '例如 CPI 同比从 2.5 到 3.0，变化是 +0.5 个百分点。PMI 则选择“点”。' },
  identity: { name: '已经计算好的每期变化', unit: 'return', hint: '直接输入时，收益率用小数：0.01 表示 1%；利率用 bp，宏观变量用百分点或点。不得把累计变化当单期变化。' },
} as const

export default function VariableImport({ domain, onImported }: { domain: ResearchDomain; onImported: (variable: RiskVariable) => void }) {
  const [fields, setFields] = useState<SeriesImport>({ name: '', roles: [domain === 'product' ? 'market' : 'macro'], unit: 'return', frequency: 'monthly', transform: 'price_return', source_label: '', csv_text: '', category: 'other' })
  const [fileName, setFileName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const patch = (value: Partial<SeriesImport>) => { setFields(old => ({ ...old, ...value })); setNotice(''); setError('') }
  async function selectFile(file?: File) {
    if (!file) return
    setError(''); setNotice('')
    if (file.size > 2_000_000) { setError('文件超过 2MB，请缩小序列范围。'); return }
    try { patch({ csv_text: await file.text() }); setFileName(file.name) } catch { setError('无法读取所选文件。') }
  }
  async function submit() {
    setBusy(true); setError(''); setNotice('')
    try {
      const variable = await importRiskSeries(domain, fields)
      onImported(variable)
      setNotice(`已保存「${variable.name}」。现在可以在研究方案中选择；未覆盖任何系统变量。`)
      setFields(old => ({ ...old, name: '', csv_text: '' })); setFileName('')
    } catch (caught) { setError(caught instanceof Error ? caught.message : '导入失败。') }
    finally { setBusy(false) }
  }
  return <section className={`${sectionClass} space-y-4`}>
    <div><h3 className="font-semibold">导入自有因子或宏观时序</h3><p className="mt-1 text-sm text-slate-500">只有缺少所需标准时序时才需要导入。结果保存到统一数据磁盘，原始数值不变。</p></div>
    <Feedback error={error} notice={notice} />
    <fieldset disabled={busy} className="grid min-w-0 gap-4 sm:grid-cols-2">
      <Field label="时序名称"><input className={inputClass} value={fields.name} onChange={event => patch({ name: event.target.value })} /></Field>
      <Field label="来源说明" hint="填写供应商、指标定义或内部资料名称；不要填写账号密码。"><input className={inputClass} value={fields.source_label} onChange={event => patch({ source_label: event.target.value })} /></Field>
      <Field label="原始数值是什么"><select className={inputClass} value={fields.transform} onChange={event => { const transform = event.target.value as SeriesImport['transform']; patch({ transform, unit: meanings[transform].unit }) }}>{Object.entries(meanings).map(([id, item]) => <option key={id} value={id}>{item.name}</option>)}</select></Field>
      <Field label="原始数据频率"><select className={inputClass} value={fields.frequency} onChange={event => patch({ frequency: event.target.value as SeriesImport['frequency'] })}><option value="daily">日频</option><option value="weekly">周频</option><option value="monthly">月频</option><option value="quarterly">季频</option></select></Field>
      {(fields.transform === 'difference' || fields.transform === 'identity') && <Field label="变动量单位"><select className={inputClass} value={fields.unit} onChange={event => patch({ unit: event.target.value as SeriesImport['unit'] })}>{fields.transform === 'identity' && <option value="return">简单收益率（小数）</option>}<option value="bp">利率基点（bp）</option><option value="pp">百分点</option><option value="points">点</option></select></Field>}
      <fieldset className="min-w-0"><legend className="text-sm font-medium text-slate-800">用于哪一层</legend><div className="mt-2 flex flex-wrap gap-3">{([['driver', '事件驱动'], ['macro', '宏观变量'], ['market', '市场风险因子']] as const).map(([id, label]) => <label key={id} className="flex min-h-10 items-center gap-2 text-sm"><input type="checkbox" checked={fields.roles.includes(id)} onChange={event => patch({ roles: event.target.checked ? [...fields.roles, id] : fields.roles.filter(role => role !== id) })} />{label}</label>)}</div></fieldset>
      <div className="sm:col-span-2"><p className="rounded-lg bg-slate-50 p-3 text-xs leading-5 text-slate-600">{meanings[fields.transform].hint}<br />CSV 表头：<code>date,value,available_at</code>。日期使用 YYYY-MM-DD。available_at 可省略，但省略后只视为导入时才可得，不伪造历史发布时间。自有数据不认证为完整历史 PIT。</p></div>
      <Field label="CSV 时序文件"><input type="file" accept=".csv,text/csv" className={`${inputClass} file:mr-3 file:rounded file:border-0 file:bg-slate-100 file:px-3 file:py-1`} onChange={event => void selectFile(event.target.files?.[0])} />{fileName && <span className="mt-1 block break-all text-xs text-slate-500">已选择：{fileName}</span>}</Field>
    </fieldset>
    <div className="flex flex-wrap gap-3"><button type="button" className={primaryClass} disabled={busy || !fields.csv_text || !fields.name.trim() || !fields.source_label.trim() || !fields.roles.length} onClick={() => void submit()}>{busy ? '校验并保存中…' : '校验并保存时序'}</button>{fields.csv_text && <button type="button" className={buttonClass} disabled={busy} onClick={() => { patch({ csv_text: '' }); setFileName('') }}>清除选择</button>}</div>
  </section>
}
