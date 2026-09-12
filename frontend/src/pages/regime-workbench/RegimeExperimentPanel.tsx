import { useEffect, useMemo, useState } from 'react'
import {
  getRegimeBatchExperiment,
  listRegimeBatchExperiments,
  prepareRegimeGraph,
  runRegimeBatchExperiment,
  type PreparedRegimeGraph,
  type RegimeBatchExperiment,
  type RegimeExperimentCandidate,
  type RegimeGraphDefinition,
  type RegimeMode,
  type RegimeNodeSchema,
} from '../../services/regimeGraph'
import RegimeHelpTip from './RegimeHelpTip'
import { regimeParameterHelp, regimeParameterLabel } from './regimeDisplay'

type DraftDimension = { nodeId: string; parameter: string; values: string }

function schemaId(schema: RegimeNodeSchema) {
  return schema.id || schema.type || ''
}

function parameters(schema?: RegimeNodeSchema) {
  return schema?.parameter_schema?.properties ?? schema?.parameters ?? {}
}

function nodeLabel(node: RegimeGraphDefinition['graph']['nodes'][number], schemas: RegimeNodeSchema[]) {
  if (node.label?.trim()) return node.label
  return schemas.find((schema) => schemaId(schema) === node.type || schema.type === node.type)?.label || '计算节点'
}

function text(reason: unknown, fallback: string) {
  return reason instanceof Error ? reason.message : fallback
}

function percent(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—'
}

function number(value: unknown, digits = 2) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—'
}

function parseValues(raw: string, type?: string) {
  const values = raw.split(',').map((value) => value.trim()).filter(Boolean)
  if (!values.length) throw new Error('每个参数维度至少填写一个候选值。')
  if (values.length > 12) throw new Error('每个参数维度最多 12 个候选值。')
  if (type === 'number' || type === 'integer') return values.map((value) => {
    const parsed = Number(value)
    if (!Number.isFinite(parsed)) throw new Error(`“${value}”不是有效数值。`)
    return type === 'integer' ? Math.trunc(parsed) : parsed
  })
  if (type === 'boolean') return values.map((value) => {
    if (value === 'true') return true
    if (value === 'false') return false
    throw new Error('布尔候选值只能是 true 或 false。')
  })
  return values
}

function CandidateSummary({ candidate }: { candidate: RegimeExperimentCandidate }) {
  return <article className="rounded-xl border border-slate-200 bg-white p-3"><div className="flex items-center justify-between gap-2"><h4 className="text-xs font-bold text-slate-950">#{candidate.rank} · {candidate.candidate_id}</h4><span className="rounded-full bg-accent-50 px-2 py-1 text-xs font-bold text-accent-700">排名值 {number(candidate.rank_value, 4)}</span></div><p className="mt-2 text-xs leading-4 text-slate-600">{candidate.parameter_differences.map((item) => `${item.node_id}.${item.parameter}: ${String(item.baseline)} → ${String(item.candidate)}`).join('；')}</p><div className="mt-2 grid grid-cols-2 gap-1 text-xs sm:grid-cols-4"><span>一致率 <b>{percent(candidate.metrics.agreement)}</b></span><span>分类率 <b>{percent(candidate.metrics.classified_ratio)}</b></span><span>翻转率 <b>{percent(candidate.metrics.flip_rate)}</b></span><span>边界距离 <b>{number(candidate.metrics.mean_boundary_distance_observations)}</b></span></div>{candidate.disagreement_intervals?.length ? <details className="mt-2"><summary className="cursor-pointer text-xs font-bold text-amber-800">分歧区间 {candidate.disagreement_intervals.length} 段{candidate.disagreement_intervals_truncated ? '（已截断）' : ''}</summary><ul className="mt-1 grid gap-1 sm:grid-cols-2">{candidate.disagreement_intervals.slice(0, 12).map((interval, index) => <li key={`${interval.start_date}-${index}`} className="rounded-lg bg-amber-50 px-2 py-1 text-xs text-amber-950">{interval.start_date || interval.start_index} → {interval.end_date || interval.end_index} · {interval.observations ?? '—'} 期</li>)}</ul></details> : <p className="mt-2 text-xs text-emerald-700">与基准没有连续分歧区间。</p>}</article>
}

export default function RegimeExperimentPanel({ definition, schemas, dirty, valid, mode, asOf, preparedPlan, onPrepared, onError, onNotice }: {
  definition: RegimeGraphDefinition
  schemas: RegimeNodeSchema[]
  dirty: boolean
  valid: boolean
  mode: RegimeMode
  asOf: string
  preparedPlan: PreparedRegimeGraph | null
  onPrepared: (plan: PreparedRegimeGraph) => void
  onError: (value: string) => void
  onNotice: (value: string) => void
}) {
  const eligibleNodes = useMemo(() => definition.graph.nodes.filter((node) => {
    const schema = schemas.find((item) => schemaId(item) === node.type || item.type === node.type)
    return schema && !['source', 'alignment'].includes(schema.category) && node.type !== 'feature.formula' && Object.keys(parameters(schema)).length > 0
  }), [definition.graph.nodes, schemas])
  const firstNode = eligibleNodes[0]
  const firstSchema = schemas.find((schema) => firstNode && (schemaId(schema) === firstNode.type || schema.type === firstNode.type))
  const [dimensions, setDimensions] = useState<DraftDimension[]>([{ nodeId: firstNode?.id || '', parameter: Object.keys(parameters(firstSchema))[0] || '', values: '' }])
  const [rankingMetric, setRankingMetric] = useState<RegimeBatchExperiment['ranking_metric']>('agreement')
  const [experiments, setExperiments] = useState<RegimeBatchExperiment[]>([])
  const [result, setResult] = useState<RegimeBatchExperiment | null>(null)
  const [busy, setBusy] = useState<'prepare' | 'run' | 'load' | ''>('')

  const eligibleIdentity = eligibleNodes.map((node) => node.id).join('|')
  useEffect(() => {
    if (!firstNode) {
      if (dimensions.length !== 1 || dimensions[0].nodeId || dimensions[0].parameter || dimensions[0].values) setDimensions([{ nodeId: '', parameter: '', values: '' }])
      return
    }
    if (dimensions.every((dimension) => eligibleNodes.some((node) => node.id === dimension.nodeId))) return
    setDimensions([{ nodeId: firstNode.id, parameter: Object.keys(parameters(firstSchema))[0] || '', values: '' }])
  }, [dimensions, eligibleIdentity, eligibleNodes, firstNode, firstSchema])
  useEffect(() => {
    if (!definition.id) { setExperiments([]); return }
    const controller = new AbortController()
    void listRegimeBatchExperiments(definition.id, controller.signal).then(setExperiments).catch((reason) => { if (!controller.signal.aborted) onError(text(reason, '批量实验目录加载失败。')) })
    return () => controller.abort()
  }, [definition.id, definition.revision, onError])

  const patchDimension = (index: number, patch: Partial<DraftDimension>) => setDimensions((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, ...patch } : item))
  const prepare = async () => {
    setBusy('prepare'); onError(''); onNotice('')
    try { const plan = await prepareRegimeGraph(definition); onPrepared(plan); onNotice(`批量实验计划已显式预热 · ${plan.plan_id}。`) } catch (reason) { onError(text(reason, '批量实验计划预热失败。')) } finally { setBusy('') }
  }
  const run = async () => {
    if (!preparedPlan) { onError('请先显式预热当前已保存版本。'); return }
    setBusy('run'); onError(''); onNotice('')
    try {
      const grid = dimensions.map((dimension) => {
        const node = definition.graph.nodes.find((item) => item.id === dimension.nodeId)
        const schema = schemas.find((item) => node && (schemaId(item) === node.type || item.type === node.type))
        const parameter = parameters(schema)[dimension.parameter]
        return { node_id: dimension.nodeId, parameter: dimension.parameter, values: parseValues(dimension.values, parameter?.type) }
      })
      const next = await runRegimeBatchExperiment({ definition, compileToken: preparedPlan.compile_token, mode, asOf: asOf || undefined, parameterGrid: grid, rankingMetric })
      setResult(next); setExperiments(await listRegimeBatchExperiments(definition.id))
      onNotice(`批量实验完成：${next.candidate_count} 个候选已由服务端 NJIT 链路排序。`)
    } catch (reason) { onError(text(reason, '批量实验失败。')) } finally { setBusy('') }
  }
  const load = async (experiment: RegimeBatchExperiment) => {
    setBusy('load'); onError('')
    try { setResult(await getRegimeBatchExperiment(experiment.id)) } catch (reason) { onError(text(reason, '批量实验详情加载失败。')) } finally { setBusy('') }
  }
  const runnable = Boolean(definition.id && definition.revision && !dirty && valid && preparedPlan && dimensions.length && dimensions.every((item) => item.nodeId && item.parameter && item.values.trim()))

  return <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm" aria-label="历史情景批量实验">
    <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between"><div><h3 className="text-sm font-bold text-slate-950">批量参数实验 <RegimeHelpTip label="批量参数实验说明" text="围绕当前已保存版本同时尝试多组窗口、阈值或状态数，比较分类覆盖、翻转率和边界变化，帮助判断算法是否稳健。" /></h3><p className="mt-1 text-xs leading-4 text-slate-500">以已保存的精确版本为基准；服务端复用同一数据扫描与预热计划，计算排名和连续分歧区间。</p></div><div className="flex flex-wrap gap-2"><button type="button" disabled={!definition.id || !definition.revision || dirty || !valid || busy !== ''} onClick={() => void prepare()} title="为当前精确版本编译并锁定固定签名 NJIT 计划" className="min-h-9 rounded-lg border border-emerald-300 px-3 text-xs font-bold text-emerald-800 disabled:opacity-40">{busy === 'prepare' ? '预热中…' : '显式预热批量实验'}</button><button type="button" disabled={!runnable || busy !== ''} onClick={() => void run()} title="运行所有候选参数组合并按所选指标排序" className="min-h-9 rounded-lg bg-accent-600 px-3 text-xs font-bold text-white disabled:opacity-40">{busy === 'run' ? '实验运行中…' : '运行参数网格'}</button></div></div>
    <div className="mt-3 flex flex-wrap gap-2 text-xs"><span className="rounded-full bg-slate-100 px-2 py-1 font-bold text-slate-700">基准 {definition.id ? `${definition.id} · r${definition.revision}` : '未保存'}</span><span className={`rounded-full px-2 py-1 font-bold ${preparedPlan ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-900'}`}>{preparedPlan ? `计划 ${preparedPlan.plan_id}` : '计划未预热'}</span></div>
    <div className="mt-4 space-y-2">{dimensions.map((dimension, index) => {
      const node = definition.graph.nodes.find((item) => item.id === dimension.nodeId)
      const schema = schemas.find((item) => node && (schemaId(item) === node.type || item.type === node.type))
      const parameterEntries = Object.entries(parameters(schema))
      const activeParameter = parameters(schema)[dimension.parameter]
      return <div key={index} className="grid gap-2 rounded-xl bg-slate-50 p-2 sm:grid-cols-[minmax(140px,1fr)_minmax(120px,1fr)_minmax(180px,1.6fr)_auto]"><label className="text-xs font-bold text-slate-600">节点<RegimeHelpTip label={`实验维度${index + 1}节点说明`} text="选择要做敏感性测试的计算节点；数据源、对齐和公式结构参数不允许在批量实验中改动。" /><select aria-label={`实验维度${index + 1}节点`} value={dimension.nodeId} onChange={(event) => { const nextNode = definition.graph.nodes.find((item) => item.id === event.target.value); const nextSchema = schemas.find((item) => nextNode && (schemaId(item) === nextNode.type || item.type === nextNode.type)); patchDimension(index, { nodeId: event.target.value, parameter: Object.keys(parameters(nextSchema))[0] || '' }) }} className="mt-1 min-h-9 w-full rounded-xl border border-slate-300 bg-white px-2 text-xs font-normal"><option value="">选择节点</option>{eligibleNodes.map((item) => <option key={item.id} value={item.id}>{nodeLabel(item, schemas)}</option>)}</select></label><label className="text-xs font-bold text-slate-600">参数<RegimeHelpTip label={`实验维度${index + 1}参数说明`} text={dimension.parameter ? regimeParameterHelp(dimension.parameter, activeParameter) : '选择该节点中要批量改变的参数。'} /><select aria-label={`实验维度${index + 1}参数`} value={dimension.parameter} onChange={(event) => patchDimension(index, { parameter: event.target.value })} className="mt-1 min-h-9 w-full rounded-xl border border-slate-300 bg-white px-2 text-xs font-normal"><option value="">选择参数</option>{parameterEntries.map(([key, meta]) => <option key={key} value={key}>{regimeParameterLabel(key, meta)}</option>)}</select></label><label className="text-xs font-bold text-slate-600">候选值（逗号分隔，最多 12 个）<RegimeHelpTip label={`实验维度${index + 1}候选值说明`} text="填写要比较的多个参数值，例如 10, 20, 40。系统会与其他维度组合，最多展开为 64 个候选。" /><input aria-label={`实验维度${index + 1}候选值`} value={dimension.values} onChange={(event) => patchDimension(index, { values: event.target.value })} className="mt-1 min-h-9 w-full rounded-lg border border-slate-300 px-2 text-xs font-normal" placeholder="10, 20, 40" /></label><button type="button" disabled={dimensions.length === 1} onClick={() => setDimensions((current) => current.filter((_, itemIndex) => itemIndex !== index))} className="min-h-9 self-end rounded-lg px-2 text-xs font-bold text-rose-600 disabled:opacity-30">删除</button></div>
    })}</div>
    <div className="mt-2 flex flex-wrap items-end gap-2"><button type="button" disabled={dimensions.length >= 6} onClick={() => setDimensions((current) => [...current, { nodeId: firstNode?.id || '', parameter: Object.keys(parameters(firstSchema))[0] || '', values: '' }])} className="min-h-9 rounded-lg border border-slate-300 px-3 text-xs font-bold text-slate-700 disabled:opacity-40">添加参数维度</button><label className="text-xs font-bold text-slate-600">排名目标<RegimeHelpTip label="排名目标说明" text="决定候选方案的排序方式。一致率衡量对基准的偏离，分类覆盖和翻转率更关注结果可用性与稳定性。" /><select aria-label="批量实验排名目标" value={rankingMetric} onChange={(event) => setRankingMetric(event.target.value as RegimeBatchExperiment['ranking_metric'])} className="mt-1 block min-h-9 rounded-xl border border-slate-300 bg-white px-2 text-xs font-normal"><option value="agreement">与基准一致率</option><option value="classified_ratio">分类覆盖率</option><option value="low_flip_rate">低翻转率</option><option value="boundary_distance">低边界距离</option></select></label><span className="pb-2 text-xs text-slate-600">最多 6 维、展开后最多 64 个候选；数据源、对齐和公式结构参数由后端禁止。</span></div>
    {experiments.length ? <div className="mt-4 border-t border-slate-200 pt-3"><h4 className="text-xs font-bold text-slate-900">历史实验</h4><div className="mt-2 flex gap-2 overflow-x-auto pb-1">{experiments.map((experiment) => <button key={experiment.id} type="button" disabled={busy !== ''} onClick={() => void load(experiment)} className="min-h-9 shrink-0 rounded-lg border border-slate-200 px-2 text-xs font-bold text-slate-700">{experiment.id} · {experiment.candidate_count} 候选</button>)}</div></div> : null}
    {result ? <div className="mt-4 rounded-xl border border-accent-200 bg-accent-50/30 p-3"><div className="flex flex-wrap items-center justify-between gap-2"><h4 className="text-xs font-bold text-accent-950">实验结果 · {result.id}</h4><span className="text-xs font-bold text-accent-700">不可变快照 · {result.ranking_metric}</span></div><div className="mt-2 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4"><span className="rounded-lg bg-white p-2">基准分类率 <b>{percent(result.baseline.classified_ratio)}</b></span><span className="rounded-lg bg-white p-2">基准翻转率 <b>{percent(result.baseline.flip_rate)}</b></span><span className="rounded-lg bg-white p-2">基准切换 <b>{String(result.baseline.state_switches ?? '—')}</b></span><span className="rounded-lg bg-white p-2">候选数 <b>{result.candidate_count}</b></span></div><div className="mt-3 space-y-2">{result.ranking.map((candidate) => <CandidateSummary key={candidate.candidate_id} candidate={candidate} />)}</div></div> : null}
  </section>
}
