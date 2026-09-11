import RegimeNodeSeriesChart from './RegimeNodeSeriesChart'
import type { RegimeGraphNode, RegimeNodeSchema, RegimePreviewRun, RegimeSeriesPage, RegimeSeriesRow } from '../../services/regimeGraph'
import RegimeEvaluationResults from './RegimeEvaluationResults'
import RegimeHelpTip from './RegimeHelpTip'

function valueText(value: unknown) {
  if (value == null) return '—'
  if (typeof value === 'number') return Number.isFinite(value) ? new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 6 }).format(value) : '—'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

const COLUMN_LABELS: Record<string, string> = {
  date: '观测日',
  observation_date: '观测日',
  recognized_at: '识别日',
  effective_date: '生效日',
  state_id: '状态编号',
  state_label: '状态',
  value: '数值',
  confidence: '置信度',
  recognition_index: '识别索引',
  effective_index: '生效索引',
  reason_code: '原因码',
  reasons: '原因',
  available_at: '可得日',
}

function columnsFor(rows: RegimeSeriesRow[]) {
  const keys = new Set<string>()
  rows.slice(0, 20).forEach((row) => Object.keys(row).forEach((key) => { if (!['date', 'observation_date'].includes(key)) keys.add(key) }))
  const preferred = ['recognized_at', 'effective_date', 'state_label', 'state_id', 'value', 'confidence', 'reason_code', 'reasons', 'recognition_index', 'effective_index', 'available_at']
  const ordered = preferred.filter((key) => keys.delete(key))
  return ['date', ...ordered, ...keys].slice(0, 12)
}

function SeriesTable({ page }: { page: RegimeSeriesPage }) {
  const columns = columnsFor(page.items)
  return <section aria-label="节点预览数据表" className="overflow-hidden rounded-xl border border-slate-200 bg-white"><div className="space-y-2 p-3 md:hidden" data-testid="regime-result-mobile-list">{page.items.map((row) => <article key={`${row.date}-${String(row.state_id ?? '')}`} className="rounded-lg bg-slate-50 p-3"><p className="text-xs font-bold text-slate-800">{row.date}</p><dl className="mt-2 grid grid-cols-2 gap-2">{columns.slice(1).map((column) => <div key={column} className="min-w-0"><dt className="truncate text-[10px] text-slate-500">{COLUMN_LABELS[column] || column}</dt><dd className="truncate text-xs font-semibold text-slate-800">{valueText(row[column])}</dd></div>)}</dl></article>)}</div><div className="hidden max-h-64 overflow-auto md:block"><table className="min-w-full text-left text-xs" aria-label="节点预览抽样数据"><thead className="sticky top-0 bg-slate-50 text-slate-500"><tr>{columns.map((column) => <th key={column} className="whitespace-nowrap px-3 py-2 font-semibold">{COLUMN_LABELS[column] || column}</th>)}</tr></thead><tbody>{page.items.map((row, index) => <tr key={`${row.date}-${index}`} className="border-t border-slate-100">{columns.map((column) => <td key={column} className="max-w-64 truncate whitespace-nowrap px-3 py-2 text-slate-700">{valueText(row[column])}</td>)}</tr>)}</tbody></table></div></section>
}

export interface RegimeResultDockProps {
  nodeOnly?: boolean
  run: RegimePreviewRun | null
  page: RegimeSeriesPage | null
  nodes: RegimeGraphNode[]
  schemas: RegimeNodeSchema[]
  previewNodeId: string
  loadingSeries: boolean
  onPreviewNode: (id: string) => void
  onLoadSeries: () => void
  onSelectComparisons?: () => void
}

export default function RegimeResultDock({ nodeOnly = false, run, page, nodes, schemas, previewNodeId, loadingSeries, onPreviewNode, onLoadSeries, onSelectComparisons }: RegimeResultDockProps) {
  // Use the same custom-name / catalog-name precedence as the canvas.
  const nodeOptions = nodes.map((node) => {
    const schema = schemas.find((item) => item.id === node.type || item.type_id === node.type || item.type === node.type)
    return { id: node.id, label: node.label?.trim() || schema?.label?.trim() || node.id }
  })
  return (
    <section className="flex min-w-0 flex-1 flex-col rounded-2xl border border-slate-200 bg-white shadow-lg" aria-label={nodeOnly ? "节点预览结果区" : "试算结果区"}>
      <header className="shrink-0 flex flex-col gap-3 border-b border-slate-200 px-4 py-3 sm:flex-row sm:items-center sm:justify-between"><div><div className="flex flex-wrap items-center gap-2"><h3 className="text-sm font-bold text-slate-950">{nodeOnly ? '节点预览结果' : '试算结果'} <RegimeHelpTip label="试算结果说明" text="试算用于随时查看当前草稿的节点时序和最终状态。修改图谱后旧结果会标记过期，必须重新试算。" /></h3><span className={`rounded-full px-2 py-1 text-[10px] font-bold ${run?.status === 'completed' ? 'bg-emerald-100 text-emerald-800' : run?.status === 'failed' ? 'bg-rose-100 text-rose-800' : run?.status === 'cancelled' ? 'bg-slate-100 text-slate-600' : 'bg-indigo-100 text-indigo-800'}`}>{run ? ({ queued: '排队中', preparing: '准备中', running: '计算中', completed: '已完成', failed: '失败', cancelled: '已取消' } as const)[run.status] : '尚未运行'}</span>{run?.stage ? <span className="text-[11px] text-slate-500">{run.stage} · {Math.round((run.progress ?? 0) * 100)}%</span> : null}</div><p className="mt-1 text-[11px] text-slate-500">{run?.message || '运行完成后可选择任意节点读取真实序列；不会在浏览器生成模拟结果。'}</p></div></header>
      <div className="flex flex-1 flex-col p-4" data-testid="regime-result-content">
        {run?.status === 'completed' ? <div className="flex flex-1 flex-col gap-4"><RegimeEvaluationResults results={run.result?.evaluation_results} mode="preview" />{!nodeOnly && <div className="flex flex-col gap-2 sm:flex-row sm:items-end"><label className="min-w-0 flex-1 text-xs font-bold text-slate-600">预览节点<RegimeHelpTip label="预览节点说明" text="选择任意中间节点查看它的真实时序，便于判断滤波、特征、模型或稳定化处理是否符合预期。" /><select aria-label="预览节点" value={previewNodeId} onChange={(event) => onPreviewNode(event.target.value)} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 bg-white px-2 font-normal"><option value="">最终输出</option>{nodeOptions.map((node) => <option key={node.id} value={node.id}>{node.label}</option>)}</select></label><button type="button" disabled={loadingSeries} onClick={onLoadSeries} title="从后端读取所选节点的真实计算序列" className="min-h-10 rounded-xl bg-indigo-600 px-4 text-sm font-bold text-white disabled:opacity-40">{loadingSeries ? '读取节点序列…' : '查看节点结果'}</button></div>}{page ? <RegimeNodeSeriesChart key={`${page.run_id}:${page.node_id}:${page.port}:${page.offset}:${page.items.length}`} page={page} onSelectComparisons={onSelectComparisons} /> : null}{page ? <><p className="text-xs text-slate-500">已返回 {page.items.length} / {page.total} 条节点结果</p>{nodeOnly && page.items.length > 100 ? <p className="text-xs text-slate-500">图表覆盖全部数据，表格展示最近 100 条。</p> : null}{nodeOnly ? <details className="rounded-lg border border-slate-200 p-3"><summary className="cursor-pointer text-xs font-semibold text-slate-600">查看节点数据（原值）</summary><div className="mt-3"><SeriesTable page={{ ...page, items: page.items.slice(-100) }} /></div></details> : <SeriesTable page={page} />}</> : <div className="grid min-h-36 place-items-center rounded-xl border border-dashed border-slate-300 text-center text-sm text-slate-500">选择节点并读取后端序列。</div>}<p className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-semibold text-emerald-900">试算的固定签名 NJIT 执行链已通过校验。</p></div> : <div className="grid min-h-48 place-items-center rounded-xl border border-dashed border-slate-300 p-8 text-center"><div><p className="font-bold text-slate-800">{run ? '试算尚未完成' : '等待一次真实试算'}</p><p className="mt-2 text-sm text-slate-500">{run?.error ? typeof run.error === 'string' ? run.error : run.error.message : nodeOnly ? '选择节点后点击预览；其他未完成的步骤不会阻止预览。' : '完成计算图并通过检查后，从顶部命令栏启动试算。'}</p></div></div>}
      </div>
    </section>
  )
}
