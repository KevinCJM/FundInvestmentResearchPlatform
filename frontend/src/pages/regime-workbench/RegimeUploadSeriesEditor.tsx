import { useEffect, useRef, useState } from 'react'
import { getResearchSeriesProfile, parseResearchFile, type ResearchImportFile, type ResearchInlineRow } from '../../services/researchSeries'
import type { RegimeGraphNode } from '../../services/regimeGraph'

const inputClass = 'mt-1 min-h-10 w-full min-w-0 rounded-xl border border-slate-300 bg-white px-2 text-xs'
const columnLabels = { date: '日期列', value: '数值列', available_at: '可得日期列', vintage: '修订批次列', revision: '修订序号列' }
type Mapping = Record<keyof typeof columnLabels, string>

export default function RegimeUploadSeriesEditor({ node, onPatchNode }: {
  node: RegimeGraphNode; onPatchNode: (patch: Partial<RegimeGraphNode>) => void
}) {
  const [file, setFile] = useState<File>()
  const [parsed, setParsed] = useState<ResearchImportFile>()
  const [mapping, setMapping] = useState<Mapping>({ date: '', value: '', available_at: '', vintage: '', revision: '' })
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const pending = useRef<AbortController>()
  const latestNode = useRef(node); latestNode.current = node
  useEffect(() => () => pending.current?.abort(), [])
  const parse = async (next: File, sheet?: string) => {
    pending.current?.abort()
    const controller = new AbortController(); pending.current = controller
    setBusy(true); setError(''); setParsed(undefined); setFile(next)
    try {
      const result = await parseResearchFile(next, sheet, controller.signal)
      if (controller.signal.aborted) return
      setParsed(result)
      const match = (...keys: string[]) => result.columns.find(column => keys.includes(column.toLowerCase())) || ''
      setMapping({ date: match('date', 'observation_date', 'trade_date', '日期', '交易日期'), value: match('value', 'close', '数值', '净值', '收盘价'), available_at: match('available_at', '可得日期', '发布日期'), vintage: match('vintage', '修订批次'), revision: match('revision', '修订序号') })
      if (!sheet) setName(next.name.replace(/\.[^.]+$/, ''))
    } catch (reason) { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '文件读取失败。') }
    finally { if (!controller.signal.aborted) setBusy(false) }
  }
  const save = async () => {
    if (!parsed || !name.trim() || !mapping.date || !mapping.value) return
    pending.current?.abort(); const controller = new AbortController(); pending.current = controller
    setBusy(true); setError('')
    try {
      const rows = parsed.rows.map(row => Object.fromEntries(Object.entries(mapping).filter(([, column]) => column).map(([key, column]) => [key, key === 'date' || key === 'available_at' ? String(row[column] ?? '') : row[column] === '' || row[column] === undefined ? null : row[column]]))) as unknown as ResearchInlineRow[]
      const profile = await getResearchSeriesProfile({ series_id: 'upload:time_series', inline_rows: rows, name: name.trim(), frequency: (node.parameters.frequency || 'daily') as 'daily', availability_mode: (node.parameters.availability_mode || 'point_in_time') as 'point_in_time', register_artifact: true }, controller.signal)
      if (controller.signal.aborted) return
      if (JSON.stringify(latestNode.current) !== JSON.stringify(node)) throw new Error('节点配置已改变，请重新点击保存并使用。')
      if (!profile.binding_parameters?.artifact_id || !profile.binding_parameters.checksum) throw new Error('文件尚未保存，请重试。')
      onPatchNode({ label: !node.label || node.label === node.parameters.name || ['上传时序', '不可变上传时序'].includes(node.label) ? name.trim() : node.label, parameters: profile.binding_parameters })
      setParsed(undefined); setFile(undefined)
    } catch (reason) { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '保存失败，请重试。') }
    finally { if (!controller.signal.aborted) setBusy(false) }
  }
  return <details open={!node.parameters.artifact_id || Boolean(file)} className="rounded-xl border border-accent-200 bg-accent-50/40 p-3">
    <summary className="cursor-pointer text-sm font-semibold text-accent-950">上传新文件</summary>
    <div className="mt-3 space-y-3">
      <p className="text-xs leading-5 text-slate-600">选择文件 → 确认日期和数值列 → 保存并使用。支持 CSV、Excel（.xlsx）和 JSON，最多 20,000 行、8 MB。</p>
      <label className="inline-flex min-h-10 cursor-pointer items-center rounded-lg border border-accent-300 bg-white px-3 text-xs font-semibold text-accent-700">选择时序文件<input aria-label="选择时序文件" type="file" accept=".csv,.xlsx,.json" disabled={busy} className="sr-only" onChange={event => { const next = event.target.files?.[0]; if (next) void parse(next); event.target.value = '' }} /></label>
      {file && <p className="break-all text-xs text-slate-600">已选择：{file.name}</p>}
      {parsed && <>
        {parsed.sheets.length > 1 && <label className="block text-xs">工作表<select aria-label="工作表" value={parsed.sheet || ''} className={inputClass} disabled={busy} onChange={event => file && void parse(file, event.target.value)}>{parsed.sheets.map(sheet => <option key={sheet}>{sheet}</option>)}</select></label>}
        <label className="block text-xs">序列名称<input aria-label="上传序列名称" value={name} maxLength={160} disabled={busy} onChange={event => setName(event.target.value)} className={inputClass} /></label>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">{(['date', 'value'] as const).map(key => <label key={key} className="text-xs">{columnLabels[key]} *<select aria-label={columnLabels[key]} className={inputClass} value={mapping[key]} disabled={busy} onChange={event => setMapping({ ...mapping, [key]: event.target.value })}><option value="">请选择</option>{parsed.columns.map(column => <option key={column}>{column}</option>)}</select></label>)}</div>
        <details><summary className="cursor-pointer text-xs text-slate-600">发布时间与修订列（可选）</summary><div className="mt-2 space-y-2">{(['available_at', 'vintage', 'revision'] as const).map(key => <label key={key} className="block text-xs">{columnLabels[key]}<select aria-label={columnLabels[key]} className={inputClass} value={mapping[key]} disabled={busy} onChange={event => setMapping({ ...mapping, [key]: event.target.value })}><option value="">不指定</option>{parsed.columns.map(column => <option key={column}>{column}</option>)}</select></label>)}</div></details>
        <p className="text-xs text-slate-600">未指定可得日期列时，按观察日已知处理；修订数据请指定实际发布日期。</p>
        <div className="overflow-auto rounded-xl border border-slate-200 bg-white"><table aria-label="文件列预览" className="w-full text-left text-xs"><thead><tr><th scope="col" className="p-2">日期</th><th scope="col" className="p-2">数值</th></tr></thead><tbody>{parsed.rows.slice(0, 5).map((row, i) => <tr key={i}><td className="p-2">{String(row[mapping.date] ?? '—')}</td><td className="p-2">{String(row[mapping.value] ?? '—')}</td></tr>)}</tbody></table></div>
        <p className="text-xs text-slate-600">共 {parsed.rows.length} 行，以上展示前 5 行。保存为固定版本后可重复使用。</p>
        <button type="button" onClick={() => void save()} disabled={busy || !name.trim() || !mapping.date || !mapping.value || mapping.date === mapping.value} className="min-h-10 rounded-lg bg-accent-600 px-3 text-xs font-semibold text-white disabled:opacity-40">保存并使用</button>
      </>}
      {busy && <p role="status" className="text-xs text-accent-700">正在处理文件…</p>}
      {error && <p role="alert" className="break-words text-xs text-rose-700">{error}</p>}
      {file && <button type="button" onClick={() => { pending.current?.abort(); setBusy(false); setFile(undefined); setParsed(undefined); setError('') }} className="min-h-9 text-xs text-slate-600">取消本次上传</button>}
    </div>
  </details>
}
