import { useEffect, useId, useMemo, useRef, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import HistoricalRegimeWorkbench from './HistoricalRegimeWorkbench'
import { copyHistoricalRegimeDefinitionToV2, type RegimeGraphDefinition } from '../services/regimeGraph'
import {
  compareHistoricalRegimeRuns,
  createHistoricalRegimeDefinition,
  getHistoricalRegimeRun,
  getHistoricalRegimeMeta,
  listHistoricalRegimeDefinitions,
  listHistoricalRegimeRuns,
  publishHistoricalRegimeRun,
  runHistoricalRegime,
  updateHistoricalRegimeDefinition,
  type AlgorithmFamily,
  type HistoricalRegimeDefinition,
  type HistoricalRegimeMeta,
  type HistoricalRegimeRun,
  type PublicationUsage,
  type RegimeComparison,
  type RegimeMode,
  type RegimeSegment,
  type RegimeStateDefinition,
  type RegimeTemplate,
} from '../services/historicalRegimes'

type PipelineStep = 'data' | 'features' | 'model' | 'states' | 'validation' | 'publish'
type ResultTab = 'segments' | 'conditional' | 'transition' | 'stability' | 'audit' | 'comparison'

const pipelineSteps: Array<{ id: PipelineStep; label: string; helper: string }> = [
  { id: 'data', label: '数据', helper: '对象、频率与可用时间' },
  { id: 'features', label: '指标 / 特征', helper: '变换、窗口与因果性' },
  { id: 'model', label: '模型', helper: '算法族与参数' },
  { id: 'states', label: '标签规则', helper: '状态语义与颜色' },
  { id: 'validation', label: '验证', helper: '稳定性与实时回放' },
  { id: 'publish', label: '发布', helper: '用途门禁与应用绑定' },
]

const algorithmNames: Record<AlgorithmFamily, string> = {
  causal_filter: '因果低滞后滤波',
  turning_point: '峰谷与转折点识别',
  hmm: '隐马尔可夫模型（HMM）',
  markov: 'Markov 状态切换',
  gmm: '高斯混合模型（GMM）',
  change_point: '变化点检测',
  merrill_clock: '美林时钟规则',
  relative_strength: '相对强弱轮动',
  ensemble: '候选算法集成',
}

const parameterMeta: Record<string, { label: string; unit?: string }> = {
  bull_enter: { label: '进入牛市阈值' },
  bull_exit: { label: '退出牛市阈值' },
  bear_enter: { label: '进入熊市阈值' },
  bear_exit: { label: '退出熊市阈值' },
  upper: { label: '上穿阈值' },
  lower: { label: '下穿阈值' },
  confirmation: { label: '连续确认期', unit: '期' },
  min_duration: { label: '最短持续期', unit: '期' },
  positive_exit: { label: '退出正向状态阈值' },
  negative_exit: { label: '退出负向状态阈值' },
  states: { label: '状态数量', unit: '类' },
  initial_train_size: { label: '初始训练窗', unit: '期' },
  iterations: { label: '最大迭代次数', unit: '次' },
  volatility_window: { label: '波动率窗口', unit: '期' },
  window: { label: '识别窗口', unit: '期' },
  min_move: { label: '最小阶段涨跌幅' },
  threshold: { label: '变点判定阈值' },
  growth_field: { label: '增长指标字段' },
  inflation_field: { label: '通胀指标字段' },
  feature_fields: { label: '模型特征字段' },
  consensus_threshold: { label: '最低共识阈值' },
  members: { label: '候选算法与权重' },
}

const defaultAlgorithmParameters: Record<AlgorithmFamily, HistoricalRegimeDefinition['algorithm']['parameters']> = {
  causal_filter: { upper: 0.001, lower: -0.001, positive_exit: 0, negative_exit: 0, confirmation: 3, min_duration: 5 },
  turning_point: { window: 20, min_move: 0.08 },
  merrill_clock: { growth_field: 'growth', inflation_field: 'inflation', confirmation: 2 },
  relative_strength: { upper: 0.001, lower: -0.001, confirmation: 3, min_duration: 5 },
  hmm: { states: 3, initial_train_size: 120, iterations: 60, volatility_window: 20, feature_fields: '' },
  markov: { states: 3, initial_train_size: 120, iterations: 60, volatility_window: 20, feature_fields: '' },
  gmm: { states: 3, initial_train_size: 120, iterations: 60, volatility_window: 20, feature_fields: '' },
  change_point: { window: 20, threshold: 1.5, confirmation: 2 },
  ensemble: {
    consensus_threshold: 0.6,
    members: [
      { family: 'causal_filter', weight: 0.6, parameters: { bull_enter: 0.001, bear_enter: -0.001, confirmation: 3 } },
      { family: 'change_point', weight: 0.4, parameters: { window: 20, threshold: 1.5, confirmation: 2 } },
    ],
  },
}

const usageMeta: Record<PublicationUsage, { name: string; description: string }> = {
  research_display: { name: '研究展示', description: '历史图表、区间说明与研究复盘' },
  product_research: { name: '产品研究', description: '净值区间底色与分情景表现' },
  formal_backtest: { name: '正式回测', description: '只允许逐日可获得的状态信号' },
  taa: { name: 'TAA', description: '映射受约束的战术权重偏移' },
}

const fallbackStateColors = ['#16a34a', '#dc2626', '#64748b', '#7c3aed', '#0284c7']

function cx(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(' ')
}

function cloneDefinition(definition: HistoricalRegimeDefinition): HistoricalRegimeDefinition {
  return JSON.parse(JSON.stringify(definition)) as HistoricalRegimeDefinition
}

function definitionSignature(definition: HistoricalRegimeDefinition | null): string {
  if (!definition) return ''
  const copy = cloneDefinition(definition)
  delete copy.updated_at
  delete copy.created_at
  delete copy.status
  return JSON.stringify(copy)
}

function signatureForRun(run: HistoricalRegimeRun, draft: HistoricalRegimeDefinition): string {
  if (run.definition) return definitionSignature(run.definition)
  if (run.definition_id === draft.id && run.definition_revision === draft.revision) return definitionSignature(draft)
  return '__different-run__:' + run.id
}

function normalizeTemplateDefinition(template: RegimeTemplate): HistoricalRegimeDefinition | null {
  const candidate = template.definition ?? template.default_definition
  if (!candidate) return null
  const definition = cloneDefinition(candidate)
  definition.template_id = definition.template_id || template.id
  definition.name = definition.name || template.name
  definition.description = definition.description || template.description
  const target = definition.target ?? definition.data as HistoricalRegimeDefinition['target'] | undefined
  definition.target = {
    ...target,
    kind: target?.kind || 'index',
    series_id: target?.series_id || target?.ts_code || target?.code || target?.kind || '',
    name: target?.name || target?.ts_code || target?.code || '',
    frequency: target?.frequency || 'daily',
    availability_mode: target?.availability_mode || 'point_in_time',
  }
  definition.features = definition.features && !Array.isArray(definition.features) ? definition.features : {}
  definition.algorithm = {
    family: definition.algorithm?.family || 'causal_filter',
    parameters: definition.algorithm?.parameters ?? definition.algorithm?.params ?? {},
  }
  definition.states = Array.isArray(definition.states) ? definition.states : []
  definition.validation = definition.validation ?? { walk_forward: true, sensitivity_pct: 10, minimum_segment: 20 }
  return definition
}

function formatPercent(value: number | null | undefined, digits = 2) {
  return typeof value === 'number' && Number.isFinite(value) ? new Intl.NumberFormat('zh-CN', { style: 'percent', maximumFractionDigits: digits }).format(value) : '—'
}

function formatNumber(value: unknown, digits = 2) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : typeof value === 'boolean' ? (value ? '是' : '否') : typeof value === 'string' ? value : '—'
}

const validationMetricLabels: Record<string, string> = {
  status: '验证状态',
  agreement_rate: '状态一致率',
  revision_rate: '历史标签修订率',
  revisions: '发生修订的观测数',
  state_switches: '状态切换次数',
  classified_ratio: '有效分类覆盖率',
  revision_observations: '原始数据修订观测数',
  stability_perturbation: '参数扰动幅度',
  'prefix_invariance.agreement_rate': '截断样本标签一致率',
  'prefix_invariance.revision_rate': '截断样本标签修订率',
  'prefix_invariance.revisions': '截断样本修订数',
  'parameter_sensitivity.agreement_rate': '参数扰动标签一致率',
  'parameter_sensitivity.revision_rate': '参数扰动标签修订率',
  'parameter_sensitivity.perturbation': '敏感性扰动幅度',
  'realtime_monitoring.prefix_revisions': '实时回放修订数',
  'realtime_monitoring.prefix_revision_rate': '实时回放修订率',
  'realtime_monitoring.label_flips': '标签翻转次数',
  'realtime_monitoring.label_flip_rate': '标签翻转率',
  fold_count: 'Walk-forward 折数',
  classified_observations: '验证覆盖观测数',
  state_agreement: '逐期状态一致率',
}

function formatMetricValue(key: string, value: number | string | boolean) {
  if (typeof value === 'string') {
    const statusNames: Record<string, string> = { passed: '通过', not_requested: '未启用', insufficient_data: '样本不足', unavailable: '不可用' }
    return statusNames[value] || value
  }
  if (typeof value === 'number' && /(rate|ratio|agreement|perturbation)/.test(key)) return formatPercent(value)
  return formatNumber(value)
}

function stateColor(states: RegimeStateDefinition[], id: string, index = 0) {
  return states.find((state) => state.id === id)?.color || fallbackStateColors[index % fallbackStateColors.length]
}

function metricEntries(value: Record<string, unknown> | undefined) {
  if (!value) return []
  const entries: Array<{ key: string; label: string; value: string }> = []
  const append = (raw: unknown, key: string, depth: number) => {
    if (typeof raw === 'number' || typeof raw === 'string' || typeof raw === 'boolean') {
      entries.push({ key, label: validationMetricLabels[key] || key, value: formatMetricValue(key, raw) })
      return
    }
    if (depth > 0 || !raw || Array.isArray(raw) || typeof raw !== 'object') return
    Object.entries(raw as Record<string, unknown>).forEach(([child, item]) => append(item, key + '.' + child, depth + 1))
  }
  Object.entries(value).forEach(([key, raw]) => append(raw, key, 0))
  return entries
}

function SectionHeading({ eyebrow, title, detail }: { eyebrow: string; title: string; detail: string }) {
  return (
    <div>
      <p className="text-[11px] font-bold uppercase tracking-[0.16em] text-indigo-600">{eyebrow}</p>
      <h3 className="mt-1 text-lg font-bold text-slate-950">{title}</h3>
      <p className="mt-1 text-xs leading-5 text-slate-500">{detail}</p>
    </div>
  )
}

function CausalityBadge({ run }: { run: HistoricalRegimeRun | null }) {
  if (!run) return <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-bold text-slate-600">尚未运行</span>
  if (run.causality.uses_future_data) return <span className="rounded-full bg-rose-100 px-3 py-1 text-xs font-bold text-rose-800">含未来信息 · 仅事后</span>
  if (run.causality.realtime_eligible) return <span className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-bold text-emerald-800">单边因果 · 实时可用</span>
  return <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-bold text-amber-900">研究用途 · 不可实时</span>
}

function PipelineNavigation({ active, onChange }: { active: PipelineStep; onChange: (step: PipelineStep) => void }) {
  return (
    <nav aria-label="历史情景识别研究管线" className="rounded-2xl border border-slate-200 bg-white p-3 shadow-sm">
      <p className="px-2 pb-2 text-xs font-bold text-slate-500">研究管线</p>
      <ol className="space-y-1">
        {pipelineSteps.map((step, index) => (
          <li key={step.id}>
            <button
              type="button"
              aria-current={active === step.id ? 'step' : undefined}
              onClick={() => onChange(step.id)}
              className={cx(
                'flex min-h-14 w-full items-center gap-3 rounded-xl px-3 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500',
                active === step.id ? 'bg-slate-950 text-white' : 'text-slate-700 hover:bg-slate-50',
              )}
            >
              <span className={cx('grid h-7 w-7 shrink-0 place-items-center rounded-full text-xs font-bold', active === step.id ? 'bg-indigo-500 text-white' : 'bg-slate-100 text-slate-600')}>{index + 1}</span>
              <span>
                <span className="block text-sm font-bold">{step.label}</span>
                <span className={cx('block text-[11px]', active === step.id ? 'text-slate-300' : 'text-slate-400')}>{step.helper}</span>
              </span>
            </button>
          </li>
        ))}
      </ol>
    </nav>
  )
}

function RegimeChart({ run, onSelectSegment }: { run: HistoricalRegimeRun; onSelectSegment: (segment: RegimeSegment) => void }) {
  const chartId = useId()
  const dates = run.series.map((point) => point.date || point.observation_date || '')
  const stateIds = run.states.length ? run.states.map((state) => state.id) : Array.from(new Set(run.series.map((point) => point.state_id)))
  const option = useMemo<EChartsOption>(() => ({
    animationDuration: 280,
    aria: {
      enabled: true,
      decal: { show: true },
      description: run.name + '历史状态图。上方展示原始序列、滤波趋势和状态区间，下方展示状态概率。',
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' },
    },
    legend: { top: 0, data: ['原始序列', '滤波趋势', ...stateIds.map((id) => run.states.find((state) => state.id === id)?.label || id)] },
    grid: [
      { left: 56, right: 28, top: 54, height: '53%' },
      { left: 56, right: 28, top: '75%', height: '15%' },
    ],
    xAxis: [
      { type: 'category', data: dates, boundaryGap: false, axisLabel: { color: '#64748b', hideOverlap: true } },
      { type: 'category', gridIndex: 1, data: dates, boundaryGap: false, axisLabel: { color: '#64748b', hideOverlap: true } },
    ],
    yAxis: [
      { type: 'value', scale: true, axisLabel: { color: '#64748b' }, splitLine: { lineStyle: { color: '#e2e8f0' } } },
      { type: 'value', gridIndex: 1, min: 0, max: 1, axisLabel: { color: '#64748b', formatter: '{value}' }, splitLine: { show: false } },
    ],
    dataZoom: [
      { type: 'inside', xAxisIndex: [0, 1] },
      { type: 'slider', xAxisIndex: [0, 1], bottom: 2, height: 18 },
    ],
    series: [
      {
        name: '原始序列',
        type: 'line',
        data: run.series.map((point) => point.value),
        showSymbol: false,
        lineStyle: { color: '#0f172a', width: 2 },
        markArea: {
          silent: false,
          label: { show: true, color: '#334155', fontSize: 10 },
          data: run.segments.map((segment, index) => ([
            {
              name: segment.state_label,
              xAxis: segment.start_date,
              itemStyle: { color: stateColor(run.states, segment.state_id, index) + '22' },
            },
            { xAxis: segment.end_date },
          ])),
        },
      },
      {
        name: '滤波趋势',
        type: 'line',
        data: run.series.map((point) => point.filtered_value ?? null),
        showSymbol: false,
        lineStyle: { color: '#6366f1', width: 2 },
      },
      ...stateIds.map((stateId, index) => ({
        name: run.states.find((state) => state.id === stateId)?.label || stateId,
        type: 'line' as const,
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: run.series.map((point) => point.probabilities?.[stateId] ?? null),
        showSymbol: false,
        lineStyle: { color: stateColor(run.states, stateId, index), width: 1.5 },
        areaStyle: { opacity: 0.05 },
      })),
    ],
  }) as EChartsOption, [dates, run, stateIds])

  const handleChartClick = (params: { dataIndex?: number }) => {
    if (typeof params.dataIndex !== 'number') return
    const date = dates[params.dataIndex]
    const segment = run.segments.find((item) => date >= item.start_date && date <= item.end_date)
    if (segment) onSelectSegment(segment)
  }

  return (
    <figure aria-labelledby={chartId} className="min-w-0">
      <figcaption id={chartId} className="sr-only">{run.name}历史情景识别结果</figcaption>
      <ReactECharts option={option} style={{ height: 500 }} notMerge lazyUpdate onEvents={{ click: handleChartClick }} />
      <details className="mt-2 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm">
        <summary className="cursor-pointer font-semibold text-indigo-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500">查看图表数据表</summary>
        <div className="mt-3 max-h-72 overflow-auto">
          <table className="min-w-full text-left text-xs">
            <caption className="sr-only">历史状态逐期数据</caption>
            <thead className="sticky top-0 bg-slate-100 text-slate-500"><tr><th className="px-2 py-2">日期</th><th className="px-2 py-2 text-right">原始值</th><th className="px-2 py-2 text-right">趋势值</th><th className="px-2 py-2">状态</th><th className="px-2 py-2">识别日</th></tr></thead>
            <tbody className="divide-y divide-slate-200">
              {run.series.map((point) => <tr key={point.date}><td className="px-2 py-2">{point.date}</td><td className="px-2 py-2 text-right tabular-nums">{formatNumber(point.value)}</td><td className="px-2 py-2 text-right tabular-nums">{formatNumber(point.filtered_value)}</td><td className="px-2 py-2">{point.state_label}</td><td className="px-2 py-2">{point.recognized_at || '—'}</td></tr>)}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  )
}

function EvidenceInspector({ run, segment }: { run: HistoricalRegimeRun; segment: RegimeSegment | null }) {
  const current = run.series[run.series.length - 1]
  const selected = segment ?? run.segments[run.segments.length - 1] ?? null
  const evidencePoint = selected
    ? [...run.series].reverse().find((point) => point.date >= selected.start_date && point.date <= selected.end_date)
    : current
  const probabilities = evidencePoint?.probabilities ?? {}
  return (
    <aside aria-label="状态证据检查器" className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
      <p className="text-[11px] font-bold uppercase tracking-[0.16em] text-indigo-600">Evidence inspector</p>
      <h4 className="mt-2 text-lg font-bold text-slate-950">{selected?.state_label ?? current?.state_label ?? '未识别'}</h4>
      <p className="mt-1 text-xs text-slate-500">{selected ? selected.start_date + ' 至 ' + selected.end_date : '暂无区间'}</p>
      <dl className="mt-4 space-y-3 text-sm">
        <div><dt className="text-xs text-slate-500">置信度</dt><dd className="mt-1 font-bold text-slate-950">{formatPercent(selected?.confidence ?? evidencePoint?.confidence)}</dd></div>
        <div><dt className="text-xs text-slate-500">事后区间起点</dt><dd className="mt-1 font-semibold">{selected?.start_date ?? '—'}</dd></div>
        <div><dt className="text-xs text-slate-500">数据当时可用日</dt><dd className="mt-1 font-semibold">{evidencePoint?.data_available_at ?? '—'}</dd></div>
        <div><dt className="text-xs text-slate-500">模型识别日</dt><dd className="mt-1 font-semibold">{selected?.recognized_at ?? evidencePoint?.recognized_at ?? '—'}</dd></div>
        <div><dt className="text-xs text-slate-500">信号生效日</dt><dd className="mt-1 font-semibold">{evidencePoint?.effective_date ?? '—'}</dd></div>
        <div><dt className="text-xs text-slate-500">持续观测数</dt><dd className="mt-1 font-semibold tabular-nums">{selected?.duration_observations ?? '—'}</dd></div>
      </dl>
      {evidencePoint?.executable === false ? <p className="mt-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs font-bold text-amber-950">待下一交易日生效 / 不可执行</p> : evidencePoint?.executable === true ? <p className="mt-4 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-bold text-emerald-900">信号已生效 / 可用于当日决策</p> : null}
      {Object.keys(probabilities).length ? (
        <div className="mt-5 border-t border-slate-200 pt-4">
          <p className="text-xs font-bold text-slate-700">各状态概率</p>
          <div className="mt-3 space-y-3">{Object.entries(probabilities).map(([stateId, probability], index) => {
            const state = run.states.find((item) => item.id === stateId)
            const safeProbability = typeof probability === 'number' && Number.isFinite(probability) ? Math.min(1, Math.max(0, probability)) : 0
            return <div key={stateId}><div className="flex items-center justify-between text-[11px]"><span className="font-semibold text-slate-700">{state?.label || stateId}</span><span className="tabular-nums text-slate-500">{formatPercent(probability)}</span></div><div className="mt-1 h-2 overflow-hidden rounded-full bg-slate-200"><span className="block h-full rounded-full" style={{ width: safeProbability * 100 + '%', backgroundColor: stateColor(run.states, stateId, index) }} /></div></div>
          })}</div>
        </div>
      ) : null}
      <div className="mt-5 border-t border-slate-200 pt-4">
        <p className="text-xs font-bold text-slate-700">识别依据</p>
        <ul className="mt-2 space-y-2 text-xs leading-5 text-slate-600">
          {[...(selected?.reasons ?? []), ...(evidencePoint?.reasons ?? [])].filter((reason, index, all) => all.indexOf(reason) === index).slice(0, 6).map((reason) => <li key={reason} className="rounded-lg bg-white px-3 py-2 ring-1 ring-slate-200">{reason}</li>)}
          {!selected?.reasons?.length && !evidencePoint?.reasons?.length ? <li>API 未返回该区间的解释证据。</li> : null}
        </ul>
      </div>
      {evidencePoint && Object.keys(evidencePoint.features ?? {}).length ? (
        <div className="mt-5 border-t border-slate-200 pt-4">
          <p className="text-xs font-bold text-slate-700">特征快照</p>
          <dl className="mt-2 grid grid-cols-2 gap-2 text-xs">
            {Object.entries(evidencePoint.features).map(([name, value]) => <div key={name} className="rounded-lg bg-white p-2 ring-1 ring-slate-200"><dt className="truncate text-slate-500">{name}</dt><dd className="mt-1 font-bold tabular-nums text-slate-900">{formatNumber(value, 3)}</dd></div>)}
          </dl>
        </div>
      ) : null}
    </aside>
  )
}

function parseInlineRows(text: string): Array<Record<string, unknown>> {
  const source = text.trim()
  if (!source) throw new Error('请粘贴 CSV 或 JSON 数据。')
  let rows: Array<Record<string, unknown>>
  if (source.startsWith('[')) {
    const parsed = JSON.parse(source) as unknown
    if (!Array.isArray(parsed)) throw new Error('JSON 顶层必须是数组。')
    rows = parsed.map((item) => {
      if (!item || typeof item !== 'object' || Array.isArray(item)) throw new Error('JSON 数组中的每一项都必须是对象。')
      return item as Record<string, unknown>
    })
  } else {
    const lines = source.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
    if (lines.length < 2) throw new Error('CSV 至少需要表头和一行数据。')
    const headers = lines[0].split(',').map((header) => header.trim())
    rows = lines.slice(1).map((line, rowIndex) => {
      const values = line.split(',').map((value) => value.trim())
      if (values.length !== headers.length) throw new Error('CSV 第 ' + (rowIndex + 2) + ' 行列数与表头不一致。')
      return headers.reduce<Record<string, unknown>>((result, header, index) => {
        const raw = values[index]
        const numeric = raw !== '' && Number.isFinite(Number(raw)) ? Number(raw) : raw
        result[header] = numeric
        return result
      }, {})
    })
  }
  if (!rows.length) throw new Error('数据中没有可运行的观测。')
  rows.forEach((row, index) => {
    const date = row.observation_date ?? row.date
    if (typeof date !== 'string' || !date) throw new Error('第 ' + (index + 1) + ' 行缺少 date 或 observation_date。')
    const hasValue = typeof row.value === 'number'
    const hasClockFields = typeof row.growth === 'number' && typeof row.inflation === 'number'
    if (!hasValue && !hasClockFields) throw new Error('第 ' + (index + 1) + ' 行需要数值 value，或同时提供 growth 与 inflation。')
    if (row.available_at != null && typeof row.available_at !== 'string') throw new Error('第 ' + (index + 1) + ' 行 available_at 必须是日期字符串。')
  })
  return rows
}

function InlineDataEditor({ rows, onApply }: { rows: Array<Record<string, unknown>>; onApply: (rows: Array<Record<string, unknown>>) => void }) {
  const [text, setText] = useState(() => rows.length ? JSON.stringify(rows, null, 2) : 'date,value,available_at,vintage\n2024-01-31,100,2024-01-31,initial')
  const [error, setError] = useState('')
  const [parsedCount, setParsedCount] = useState(rows.length)
  const apply = (nextText = text) => {
    try {
      const parsed = parseInlineRows(nextText)
      onApply(parsed)
      setParsedCount(parsed.length)
      setError('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '数据解析失败。')
    }
  }
  const loadFile = async (file: File | undefined) => {
    if (!file) return
    const nextText = await file.text()
    setText(nextText)
    apply(nextText)
  }
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 sm:col-span-2">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div><p className="text-xs font-bold text-slate-700">粘贴或上传原始数据</p><p className="mt-1 text-[11px] text-slate-500">支持 CSV / JSON；字段可包含 date、value、available_at、vintage、growth、inflation。</p></div>
        <label className="min-h-9 cursor-pointer rounded-lg border border-indigo-200 bg-white px-3 py-2 text-xs font-bold text-indigo-700 hover:bg-indigo-50">选择 CSV / JSON 文件<input aria-label="选择原始数据文件" type="file" accept=".csv,.json,text/csv,application/json" className="sr-only" onChange={(event) => void loadFile(event.target.files?.[0])} /></label>
      </div>
      <textarea aria-label="原始数据内容" value={text} onChange={(event) => setText(event.target.value)} rows={8} spellCheck={false} className="mt-3 w-full rounded-lg border border-slate-300 bg-white p-3 font-mono text-xs leading-5 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200" />
      <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
        <span className={cx('text-xs font-semibold', error ? 'text-rose-700' : 'text-emerald-700')}>{error || '已解析 ' + parsedCount + ' 条观测'}</span>
        <button type="button" onClick={() => apply()} className="min-h-9 rounded-lg bg-indigo-600 px-3 text-xs font-bold text-white">解析并应用</button>
      </div>
    </div>
  )
}

function DataPanel({
  meta,
  draft,
  onChange,
  onTemplate,
}: {
  meta: HistoricalRegimeMeta
  draft: HistoricalRegimeDefinition
  onChange: (next: HistoricalRegimeDefinition) => void
  onTemplate: (template: RegimeTemplate) => void
}) {
  const updateTarget = (patch: Partial<HistoricalRegimeDefinition['target']>) => onChange({ ...draft, target: { ...draft.target, ...patch } })
  const selectTargetKind = (kind: HistoricalRegimeDefinition['target']['kind']) => {
    if (kind !== 'indicator') {
      updateTarget({ kind })
      return
    }
    const indicator = meta.indicator_catalog?.[0]
    const productKind = indicator?.applicable_product_kinds.includes('etf') ? 'etf' : indicator?.applicable_product_kinds[0] ?? 'etf'
    onChange({
      ...draft,
      target: {
        kind: 'indicator',
        indicator_id: indicator?.id ?? '',
        indicator_revision: indicator?.revision,
        indicator_name: indicator?.name,
        product_kind: productKind,
        product_id: '',
        period: indicator?.periods[0] ?? meta.indicator_periods?.[0]?.value ?? '1M',
        frequency: 'daily',
        availability_mode: 'point_in_time',
        start_date: draft.target.start_date,
        end_date: draft.target.end_date,
      },
    })
  }
  const selectedIndicator = meta.indicator_catalog?.find((item) => item.id === draft.target.indicator_id && item.revision === draft.target.indicator_revision)
  return (
    <section className="space-y-5" aria-labelledby="data-panel-title">
      <SectionHeading eyebrow="01 / Data" title="选择原始数据与研究口径" detail="先确定要拆分的序列；宏观变量必须保留发布日期与修订口径。" />
      <div>
        <p id="data-panel-title" className="mb-2 text-xs font-bold text-slate-600">研究模板</p>
        <div role="radiogroup" aria-label="历史情景模板" className="grid gap-2 sm:grid-cols-2">
          {meta.templates.map((template) => (
            <button
              key={template.id}
              type="button"
              role="radio"
              aria-checked={draft.template_id === template.id}
              onClick={() => onTemplate(template)}
              className={cx(
                'min-h-24 rounded-xl border p-3 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500',
                draft.template_id === template.id ? 'border-indigo-500 bg-indigo-50' : 'border-slate-200 bg-white hover:border-slate-400',
              )}
            >
              <span className="text-[10px] font-bold uppercase tracking-wide text-indigo-600">{template.category || '研究模板'}</span>
              <span className="mt-1 block text-sm font-bold text-slate-950">{template.name}</span>
              <span className="mt-1 block text-[11px] leading-4 text-slate-500">{template.description}</span>
            </button>
          ))}
        </div>
      </div>
      <label className="block text-sm font-semibold text-slate-700">研究名称
        <input value={draft.name} onChange={(event) => onChange({ ...draft, name: event.target.value })} className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 px-3 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200" />
      </label>
      <label className="block text-sm font-semibold text-slate-700">研究说明
        <textarea value={draft.description} onChange={(event) => onChange({ ...draft, description: event.target.value })} rows={2} className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200" />
      </label>
      <div className="grid gap-4 sm:grid-cols-2">
        <label className="text-sm font-semibold text-slate-700 sm:col-span-2">数据结构
          <select aria-label="数据结构" value={draft.target.kind} onChange={(event) => selectTargetKind(event.target.value as HistoricalRegimeDefinition['target']['kind'])} className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200">
            {meta.data_sources.map((source) => <option key={source.id} value={source.id}>{source.label || source.name || source.id}</option>)}
          </select>
        </label>
        {draft.target.kind === 'index' ? (
          <>
            <label className="text-sm font-semibold text-slate-700">指数代码
              <input aria-label="指数代码" value={draft.target.ts_code || draft.target.code || ''} onChange={(event) => updateTarget({ ts_code: event.target.value, series_id: event.target.value, name: event.target.value })} className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 px-3 font-mono text-sm" />
            </label>
            <label className="text-sm font-semibold text-slate-700">原始字段
              <select aria-label="原始字段" value={draft.target.field || 'close'} onChange={(event) => updateTarget({ field: event.target.value })} className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3"><option value="close">收盘价</option><option value="pct_chg">涨跌幅</option><option value="total_return">全收益</option></select>
            </label>
          </>
        ) : null}
        {draft.target.kind === 'relative' ? (
          <>
            <label className="text-sm font-semibold text-slate-700">分子序列
              <input aria-label="分子序列" value={String(draft.target.numerator?.ts_code ?? draft.target.numerator?.code ?? '')} onChange={(event) => updateTarget({ numerator: { ...(draft.target.numerator ?? {}), kind: 'index', source_api: 'index_daily', ts_code: event.target.value, field: 'close' } })} className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 px-3 font-mono text-sm" />
            </label>
            <label className="text-sm font-semibold text-slate-700">分母序列
              <input aria-label="分母序列" value={String(draft.target.denominator?.ts_code ?? draft.target.denominator?.code ?? '')} onChange={(event) => updateTarget({ denominator: { ...(draft.target.denominator ?? {}), kind: 'index', source_api: 'index_daily', ts_code: event.target.value, field: 'close' } })} className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 px-3 font-mono text-sm" />
            </label>
          </>
        ) : null}
        {draft.target.kind === 'indicator' ? (
          <>
            <label className="text-sm font-semibold text-slate-700 sm:col-span-2">指标中心版本
              <select
                aria-label="指标中心版本"
                value={draft.target.indicator_id || ''}
                onChange={(event) => {
                  const indicator = meta.indicator_catalog?.find((item) => item.id === event.target.value)
                  if (!indicator) return
                  const productKind = indicator.applicable_product_kinds.includes(draft.target.product_kind || 'etf') ? draft.target.product_kind : indicator.applicable_product_kinds[0]
                  updateTarget({ indicator_id: indicator.id, indicator_revision: indicator.revision, indicator_name: indicator.name, indicator_dsl_version: indicator.dsl_version, indicator_compiled_plan_id: indicator.compiled_plan_id, product_kind: productKind, period: indicator.periods.includes(draft.target.period || '') ? draft.target.period : indicator.periods[0] })
                }}
                className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3 text-sm"
              >
                {!meta.indicator_catalog?.length ? <option value="">当前没有可用的 typed NJIT 单产品指标</option> : null}
                {meta.indicator_catalog?.map((item) => <option key={item.id + '@' + item.revision} value={item.id}>{item.name} · R{item.revision} · {item.dsl_version}</option>)}
              </select>
            </label>
            <label className="text-sm font-semibold text-slate-700">产品类型
              <select aria-label="指标产品类型" value={draft.target.product_kind || 'etf'} onChange={(event) => updateTarget({ product_kind: event.target.value as 'etf' | 'fund' })} className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3">
                {(selectedIndicator?.applicable_product_kinds ?? ['etf', 'fund']).map((kind) => <option key={kind} value={kind}>{kind === 'etf' ? 'ETF' : '场外公募基金'}</option>)}
              </select>
            </label>
            <label className="text-sm font-semibold text-slate-700">产品编号
              <input aria-label="指标产品编号" value={draft.target.product_id || ''} onChange={(event) => updateTarget({ product_id: event.target.value })} placeholder={draft.target.product_kind === 'fund' ? '例如 000001.OF' : '例如 510300.SH'} className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 px-3 font-mono text-sm" />
            </label>
            <label className="text-sm font-semibold text-slate-700 sm:col-span-2">逐期评价窗口
              <select aria-label="指标评价周期" value={draft.target.period || ''} onChange={(event) => updateTarget({ period: event.target.value })} className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3">
                {(meta.indicator_periods ?? []).filter((item) => !selectedIndicator || selectedIndicator.periods.includes(item.value)).map((item) => <option key={item.value} value={item.value}>{item.label} · {item.value}</option>)}
              </select>
            </label>
            <div className="sm:col-span-2 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-xs leading-5 text-emerald-950">
              将锁定 <strong>{selectedIndicator?.name || '所选指标'} R{draft.target.indicator_revision ?? '—'}</strong>，逐日用当时及此前数据执行已预热的 typed AST → DAG → NJIT 固定签名计划；legacy/Python 或未预热版本会被后端拒绝。
            </div>
          </>
        ) : null}
        {draft.target.kind === 'inline' ? <InlineDataEditor key={draft.template_id} rows={draft.target.rows ?? draft.target.points ?? draft.target.series ?? []} onApply={(rows) => updateTarget({ rows, points: undefined, series: undefined })} /> : null}
        <label className="text-sm font-semibold text-slate-700">开始日期
          <input type="date" value={draft.target.start_date || ''} onChange={(event) => updateTarget({ start_date: event.target.value })} className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 px-3 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200" />
        </label>
        <label className="text-sm font-semibold text-slate-700">结束日期
          <input type="date" value={draft.target.end_date || ''} onChange={(event) => updateTarget({ end_date: event.target.value })} className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 px-3 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200" />
        </label>
        <label className="text-sm font-semibold text-slate-700">频率
          <select disabled={draft.target.kind === 'indicator'} value={draft.target.frequency || 'daily'} onChange={(event) => updateTarget({ frequency: event.target.value })} className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3 disabled:bg-slate-100">
            <option value="daily">日频</option><option value="weekly">周频</option><option value="monthly">月频</option>
          </select>
        </label>
        <label className="text-sm font-semibold text-slate-700">数据可用口径
          <select disabled={draft.target.kind === 'indicator'} value={draft.target.availability_mode || 'point_in_time'} onChange={(event) => updateTarget({ availability_mode: event.target.value as 'point_in_time' | 'latest' })} className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3 disabled:bg-slate-100">
            <option value="point_in_time">当时可获得（PIT）</option><option value="latest">最新修订值</option>
          </select>
        </label>
      </div>
      <div className={cx('rounded-xl border px-4 py-3 text-xs leading-5', draft.target.availability_mode === 'point_in_time' ? 'border-emerald-200 bg-emerald-50 text-emerald-900' : 'border-amber-200 bg-amber-50 text-amber-950')}>
        {draft.target.availability_mode === 'point_in_time' ? '运行时按 available_at 截断数据，可用于逐日回放。' : '最新修订值适合事后解释；发布到正式回测或 TAA 前仍需通过因果性检查。'}
      </div>
    </section>
  )
}

function FeaturePanel({ meta, draft, onChange }: { meta: HistoricalRegimeMeta; draft: HistoricalRegimeDefinition; onChange: (next: HistoricalRegimeDefinition) => void }) {
  const updateFeature = (key: string, value: number | string | boolean | null) => onChange({ ...draft, features: { ...draft.features, [key]: value } })
  const updateFormula = (value: string) => {
    const features = { ...draft.features }
    if (value.trim()) features.formula = value
    else delete features.formula
    onChange({ ...draft, features })
  }
  const filterId = typeof draft.features.filter === 'string' ? draft.features.filter : 'ema'
  const activeFilter = meta.feature_catalog.find((item) => item.id === filterId)
  const formula = typeof draft.features.formula === 'string' ? draft.features.formula : ''
  const formulaFunctions = (meta.formula_language?.functions ?? ['log', 'abs', 'sqrt', 'clip', 'lag', 'difference', 'cumulative_sum', 'cumulative_max']).map((item) => typeof item === 'string' ? item : item.signature || item.name || item.id || '').filter(Boolean)
  const formulaExamples = ['log(value)', 'growth - inflation', 'difference(log(value), 20)']
  return (
    <section className="space-y-5">
      <SectionHeading eyebrow="02 / Features" title="构建指标与特征管线" detail="配置原始变换、滤波和斜率窗口，也可引用指标中心中的版本化指标。" />
      <div>
        <p className="mb-2 text-xs font-bold text-slate-600">滤波方法</p>
        <div role="radiogroup" aria-label="滤波方法" className="grid gap-2 sm:grid-cols-2">
          {meta.feature_catalog.map((feature) => {
            const active = feature.id === filterId
            return <button key={feature.id} type="button" role="radio" aria-checked={active} onClick={() => updateFeature('filter', feature.id)} className={cx('min-h-20 rounded-xl border p-3 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500', active ? 'border-indigo-500 bg-indigo-50' : 'border-slate-200 bg-white hover:border-slate-400')}><span className="block text-sm font-bold text-slate-950">{feature.label || feature.name || feature.id}</span><span className={cx('mt-2 inline-flex rounded-full px-2 py-0.5 text-[10px] font-bold', feature.causal === false ? 'bg-rose-100 text-rose-800' : 'bg-emerald-100 text-emerald-800')}>{feature.causal === false ? '双边 / 会重绘' : '单边 / 因果'}</span></button>
          })}
        </div>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="text-xs font-semibold text-slate-600">原始变换
          <select aria-label="原始变换" value={String(draft.features.transform ?? 'log')} onChange={(event) => updateFeature('transform', event.target.value)} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm"><option value="identity">原值</option><option value="log">对数价格</option><option value="return">收益率</option><option value="zscore">滚动标准化</option></select>
        </label>
        <label className="text-xs font-semibold text-slate-600">平滑窗口
          <input aria-label="平滑窗口" type="number" min={1} value={Number(draft.features.window ?? 20)} onChange={(event) => updateFeature('window', Number(event.target.value))} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-3 text-sm" />
        </label>
        <label className="text-xs font-semibold text-slate-600">趋势斜率窗口
          <input aria-label="趋势斜率窗口" type="number" min={1} value={Number(draft.features.slope_window ?? 5)} onChange={(event) => updateFeature('slope_window', Number(event.target.value))} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-3 text-sm" />
        </label>
        <label className="text-xs font-semibold text-slate-600">波动率窗口
          <input aria-label="波动率窗口" type="number" min={2} value={Number(draft.features.volatility_window ?? 20)} onChange={(event) => updateFeature('volatility_window', Number(event.target.value))} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-3 text-sm" />
        </label>
      </div>
      <div className="rounded-xl border border-indigo-200 bg-indigo-50/50 p-4">
        <div className="flex flex-wrap items-start justify-between gap-2"><div><p className="text-xs font-bold text-slate-800">自定义因果公式（可选）</p><p className="mt-1 text-[11px] leading-5 text-slate-600">从数据源的数值字段生成一条新序列，再依次进入滤波、斜率计算与识别模型。留空时直接使用统一的 value 序列。</p></div>{meta.formula_language?.allowlist_version ? <span className="rounded-full bg-white px-2 py-1 text-[10px] font-bold text-indigo-700 ring-1 ring-indigo-200">白名单 {meta.formula_language.allowlist_version}</span> : null}</div>
        <label className="mt-3 block text-xs font-semibold text-slate-700">公式表达式
          <textarea aria-label="自定义因果公式" value={formula} onChange={(event) => updateFormula(event.target.value)} rows={3} spellCheck={false} placeholder="例如：difference(log(value), 20)" className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 font-mono text-sm leading-6 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200" />
        </label>
        <div className="mt-2 flex flex-wrap gap-2" aria-label="公式示例">{formulaExamples.map((example) => <button key={example} type="button" onClick={() => updateFormula(example)} className="rounded-lg border border-indigo-200 bg-white px-2.5 py-1.5 font-mono text-[11px] text-indigo-700 hover:bg-indigo-50">{example}</button>)}</div>
        <details className="mt-3 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs">
          <summary className="cursor-pointer font-semibold text-slate-700">查看公式规则与允许函数</summary>
          <div className="mt-2 space-y-2 leading-5 text-slate-600"><p>可用变量：数据源返回的数值列；统一别名为 <code>value</code>，相对序列另可用 <code>numerator</code>、<code>denominator</code>。</p><p>运算符：{meta.formula_language?.operators?.join('  ') || '+  -  *  /'}。窗口与滞后参数必须是正整数常量；不允许属性访问、下标、关键字参数、未来函数或全样本归约。</p><div className="flex flex-wrap gap-1.5">{formulaFunctions.map((name) => <code key={name} className="rounded bg-slate-100 px-2 py-0.5 text-[10px] text-slate-700">{name}</code>)}</div>{meta.formula_language?.variable_rule ? <p>{meta.formula_language.variable_rule}</p> : null}<p className="font-medium text-emerald-700">所有可用算子均走 typed AST → DAG → NJIT 固定签名内核；无 Python 回退。</p></div>
        </details>
      </div>
      <div className="rounded-xl border border-dashed border-slate-300 bg-white px-4 py-3 text-xs leading-5 text-slate-600"><strong className="text-slate-800">指标中心版本引用：</strong>已作为独立的数据源接入。请在“数据”步骤选择“指标中心版本”，锁定指标修订、产品和逐期窗口；旧的 <code>features.indicator_ref</code> 不会被静默执行。</div>
      <div className={cx('rounded-xl border px-4 py-3 text-xs leading-5', activeFilter?.causal === false ? 'border-rose-200 bg-rose-50 text-rose-900' : 'border-emerald-200 bg-emerald-50 text-emerald-900')}>
        {activeFilter?.causal === false ? '该滤波使用未来样本或会重绘，只能运行事后划分，不能发布到正式回测或 TAA。' : '该滤波按单边序列计算；最终实时资格仍以后端逐日可用性诊断为准。'}
      </div>
    </section>
  )
}

function ModelPanel({ meta, draft, onChange }: { meta: HistoricalRegimeMeta; draft: HistoricalRegimeDefinition; onChange: (next: HistoricalRegimeDefinition) => void }) {
  const members = draft.algorithm.parameters.members
  const [membersText, setMembersText] = useState(() => Array.isArray(members) ? JSON.stringify(members, null, 2) : '')
  const [membersError, setMembersError] = useState('')
  useEffect(() => {
    setMembersText(Array.isArray(members) ? JSON.stringify(members, null, 2) : '')
    setMembersError('')
  }, [draft.algorithm.family, members])
  const selectFamily = (family: AlgorithmFamily) => {
    const option = meta.algorithm_families.find((item) => (item.id ?? item.family) === family)
    const configured = (option?.parameters ?? []).reduce<Record<string, number | string | boolean | null>>((result, parameter) => {
      result[parameter.key] = parameter.default ?? null
      return result
    }, {})
    const parameters = Object.keys(configured).length ? configured : { ...defaultAlgorithmParameters[family] }
    onChange({ ...draft, algorithm: { family, parameters } })
  }
  const updateParameter = (key: string, raw: string | boolean) => {
    const current = draft.algorithm.parameters[key]
    const value = typeof current === 'number' ? Number(raw) : typeof current === 'boolean' ? Boolean(raw) : raw
    onChange({ ...draft, algorithm: { ...draft.algorithm, parameters: { ...draft.algorithm.parameters, [key]: value } } })
  }
  const applyMembers = () => {
    try {
      const parsed = JSON.parse(membersText) as unknown
      if (!Array.isArray(parsed) || parsed.length < 2 || parsed.length > 8) throw new Error('请配置 2 至 8 个候选算法。')
      const validFamilies = new Set(meta.algorithm_families.map((item) => item.id ?? item.family).filter(Boolean))
      const invalid = parsed.find((item) => {
        if (!item || typeof item !== 'object' || Array.isArray(item)) return true
        const member = item as Record<string, unknown>
        return typeof member.family !== 'string' || member.family === 'ensemble' || !validFamilies.has(member.family as AlgorithmFamily) || (member.weight != null && (typeof member.weight !== 'number' || member.weight <= 0))
      })
      if (invalid) throw new Error('每个成员都需要有效算法族和正权重，且不能嵌套集成模型。')
      setMembersError('')
      onChange({ ...draft, algorithm: { ...draft.algorithm, parameters: { ...draft.algorithm.parameters, members: parsed } } })
    } catch (reason) {
      setMembersError(reason instanceof SyntaxError ? '成员配置不是有效 JSON。' : reason instanceof Error ? reason.message : '成员配置无法解析。')
    }
  }
  return (
    <section className="space-y-5">
      <SectionHeading eyebrow="03 / Model" title="选择识别算法并调整参数" detail="算法输出状态概率或规则信号，业务标签在下一步单独定义。" />
      <div role="radiogroup" aria-label="算法族" className="grid gap-2 sm:grid-cols-2">
        {meta.algorithm_families.map((option) => {
          const family = option.id ?? option.family
          if (!family) return null
          const active = draft.algorithm.family === family
          const realtimeCapable = option.causal ?? option.supports_realtime
          return <button key={family} type="button" role="radio" aria-checked={active} onClick={() => selectFamily(family)} className={cx('min-h-24 rounded-xl border p-3 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500', active ? 'border-indigo-500 bg-indigo-50' : 'border-slate-200 hover:border-slate-400')}><span className="block text-sm font-bold text-slate-950">{option.label || option.name || algorithmNames[family]}</span><span className="mt-1 block text-[11px] leading-4 text-slate-500">{option.description || '由后端算法注册表提供。'}</span><span className={cx('mt-2 inline-flex rounded-full px-2 py-0.5 text-[10px] font-bold', realtimeCapable ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-900')}>{realtimeCapable ? '支持实时推断' : '仅适合事后识别'}</span></button>
        })}
      </div>
      <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
        <div className="flex items-center justify-between gap-3"><div><p className="text-xs font-bold text-slate-500">当前算法</p><h4 className="mt-1 font-bold text-slate-950">{algorithmNames[draft.algorithm.family]}</h4></div><code className="rounded bg-white px-2 py-1 text-xs text-indigo-700 ring-1 ring-slate-200">{draft.algorithm.family}</code></div>
        {draft.algorithm.family === 'ensemble' ? (
          <div className="mt-4 rounded-xl border border-indigo-200 bg-white p-3">
            <label className="text-xs font-semibold text-slate-700">候选算法与权重（JSON）
              <textarea aria-label="候选算法与权重 JSON" value={membersText} onChange={(event) => setMembersText(event.target.value)} rows={10} spellCheck={false} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 font-mono text-xs leading-5" />
            </label>
            <div className="mt-2 flex flex-wrap items-center justify-between gap-2"><p className="text-[11px] leading-5 text-slate-500">每个成员配置 family、weight 和自己的 parameters；冲突低于共识阈值时拒绝分类。</p><button type="button" onClick={applyMembers} className="min-h-9 rounded-lg border border-indigo-200 px-3 text-xs font-bold text-indigo-700 hover:bg-indigo-50">校验并应用成员</button></div>
            {membersError ? <p role="alert" className="mt-2 text-xs text-rose-700">{membersError}</p> : null}
          </div>
        ) : null}
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          {Object.entries(draft.algorithm.parameters).filter(([key]) => key !== 'members').map(([key, value]) => (
            <label key={key} className="text-xs font-semibold text-slate-600">{parameterMeta[key]?.label || key}{parameterMeta[key]?.unit ? '（' + parameterMeta[key].unit + '）' : ''}
              {typeof value === 'boolean'
                ? <span className="mt-2 flex min-h-10 items-center gap-2 rounded-lg border border-slate-300 bg-white px-3"><input type="checkbox" checked={value} onChange={(event) => updateParameter(key, event.target.checked)} /><span>{value ? '启用' : '停用'}</span></span>
                : <><input aria-label={'算法参数 ' + (parameterMeta[key]?.label || key)} type={typeof value === 'number' ? 'number' : 'text'} step="any" value={value == null ? '' : String(value)} placeholder={key === 'feature_fields' ? '例如：growth,inflation' : undefined} onChange={(event) => updateParameter(key, event.target.value)} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm" />{key === 'feature_fields' ? <span className="mt-1 block text-[11px] font-normal leading-4 text-slate-500">逗号分隔；留空时使用统一特征管线的趋势斜率与滚动波动率。</span> : null}</>}
            </label>
          ))}
          {!Object.keys(draft.algorithm.parameters).length ? <p className="text-xs text-slate-500 sm:col-span-2">该算法没有暴露可编辑参数。</p> : null}
        </div>
      </div>
      <div className="rounded-xl border border-indigo-200 bg-indigo-50 px-4 py-3 text-xs leading-5 text-indigo-950">
        “零相位”或双边滤波可能利用未来样本。系统以运行结果中的因果诊断为准；含未来信息的结果只允许发布为研究展示。
      </div>
    </section>
  )
}

function StatePanel({ draft, onChange }: { draft: HistoricalRegimeDefinition; onChange: (next: HistoricalRegimeDefinition) => void }) {
  const updateState = (index: number, patch: Partial<RegimeStateDefinition>) => onChange({ ...draft, states: draft.states.map((state, itemIndex) => itemIndex === index ? { ...state, ...patch } : state) })
  const addState = () => {
    const index = draft.states.length
    onChange({ ...draft, states: [...draft.states, { id: 'state_' + (index + 1), label: '新状态 ' + (index + 1), color: fallbackStateColors[index % fallbackStateColors.length], description: '' }] })
  }
  return (
    <section className="space-y-5">
      <SectionHeading eyebrow="04 / State schema" title="定义状态标签与业务语义" detail="模型状态 ID 与牛熊等业务标签分离，避免重训后的标签交换。" />
      <div className="space-y-3">
        {draft.states.map((state, index) => (
          <article key={state.id} className="rounded-xl border border-slate-200 bg-white p-4">
            <div className="grid gap-3 sm:grid-cols-[52px_1fr_1fr_auto] sm:items-end">
              <label className="text-xs font-semibold text-slate-600">颜色<input aria-label={state.label + '颜色'} type="color" value={state.color} onChange={(event) => updateState(index, { color: event.target.value })} className="mt-1 h-10 w-12 rounded border border-slate-300 bg-white p-1" /></label>
              <label className="text-xs font-semibold text-slate-600">状态 ID<input aria-label={'状态 ' + (index + 1) + ' ID'} value={state.id} onChange={(event) => updateState(index, { id: event.target.value })} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-3 text-sm" /></label>
              <label className="text-xs font-semibold text-slate-600">业务标签<input aria-label={'状态 ' + (index + 1) + ' 标签'} value={state.label} onChange={(event) => updateState(index, { label: event.target.value })} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-3 text-sm" /></label>
              <button type="button" disabled={draft.states.length <= 2} onClick={() => onChange({ ...draft, states: draft.states.filter((_, itemIndex) => itemIndex !== index) })} className="min-h-10 rounded-lg px-3 text-xs font-bold text-rose-700 hover:bg-rose-50 disabled:opacity-30">移除</button>
            </div>
            <label className="mt-3 block text-xs font-semibold text-slate-600">解释说明<input aria-label={state.label + '说明'} value={state.description || ''} onChange={(event) => updateState(index, { description: event.target.value })} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-3 text-sm" /></label>
          </article>
        ))}
      </div>
      <button type="button" onClick={addState} className="min-h-10 rounded-xl border border-indigo-200 px-4 text-sm font-bold text-indigo-700 hover:bg-indigo-50">添加状态</button>
    </section>
  )
}

function ValidationPanel({ draft, run, onChange }: { draft: HistoricalRegimeDefinition; run: HistoricalRegimeRun | null; onChange: (next: HistoricalRegimeDefinition) => void }) {
  const validation = draft.validation
  const update = (patch: Partial<HistoricalRegimeDefinition['validation']>) => onChange({ ...draft, validation: { ...validation, ...patch } })
  return (
    <section className="space-y-5">
      <SectionHeading eyebrow="05 / Validation" title="配置实时回放与稳定性验证" detail="区分全样本解释效果和当时真正可获得的识别结果。" />
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="text-xs font-semibold text-slate-600">训练开始<input type="date" value={validation.train_start || ''} onChange={(event) => update({ train_start: event.target.value })} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-3" /></label>
        <label className="text-xs font-semibold text-slate-600">训练结束<input type="date" value={validation.train_end || ''} onChange={(event) => update({ train_end: event.target.value })} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-3" /></label>
        <label className="text-xs font-semibold text-slate-600">样本外开始<input type="date" value={validation.test_start || ''} onChange={(event) => update({ test_start: event.target.value })} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-3" /></label>
        <label className="text-xs font-semibold text-slate-600">样本外结束<input type="date" value={validation.test_end || ''} onChange={(event) => update({ test_end: event.target.value })} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-3" /></label>
        <label className="text-xs font-semibold text-slate-600">参数敏感性范围<input aria-label="参数敏感性范围" type="number" min={1} max={50} value={validation.sensitivity_pct ?? 10} onChange={(event) => update({ sensitivity_pct: Number(event.target.value) })} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-3" /></label>
        <label className="text-xs font-semibold text-slate-600">最短区间观测数<input aria-label="最短区间观测数" type="number" min={1} value={validation.minimum_segment ?? 1} onChange={(event) => update({ minimum_segment: Number(event.target.value) })} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-3" /></label>
      </div>
      <label className="flex min-h-12 items-center gap-3 rounded-xl border border-slate-200 bg-slate-50 px-4 text-sm font-semibold text-slate-700">
        <input type="checkbox" checked={validation.walk_forward ?? false} onChange={(event) => update({ walk_forward: event.target.checked })} />
        启用 walk-forward 逐期训练与样本外验证
      </label>
      {run ? (
        <div className="space-y-3">
          <CausalityBadge run={run} />
          {run.diagnostics.map((diagnostic, index) => <div key={(diagnostic.code || 'diagnostic') + index} data-diagnostic-code={diagnostic.code} className={cx('rounded-xl border px-4 py-3 text-xs leading-5', diagnostic.level === 'error' ? 'border-rose-200 bg-rose-50 text-rose-900' : diagnostic.level === 'warning' ? 'border-amber-200 bg-amber-50 text-amber-950' : 'border-slate-200 bg-slate-50 text-slate-700')}>{diagnostic.message}</div>)}
          {!run.diagnostics.length ? <p className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-xs text-emerald-900">本次运行未返回阻断性诊断。</p> : null}
        </div>
      ) : <p className="rounded-xl border border-dashed border-slate-300 p-6 text-center text-sm text-slate-500">运行后显示因果性、稳定性和 walk-forward 诊断。</p>}
    </section>
  )
}

function PublishPanel({
  draft,
  run,
  stale,
  usage,
  publishing,
  onUsage,
  onPublish,
}: {
  draft: HistoricalRegimeDefinition
  run: HistoricalRegimeRun | null
  stale: boolean
  usage: PublicationUsage
  publishing: boolean
  onUsage: (usage: PublicationUsage) => void
  onPublish: () => void
}) {
  const hasSavedRun = Boolean(draft.id && run && run.definition_id === draft.id && (run.definition_revision ?? 0) > 0)
  const eligible = Boolean(run && hasSavedRun && run.causality.publish_eligible_usages.includes(usage) && !stale)
  const bindings = run?.application_bindings ?? draft.application_bindings ?? []
  return (
    <section className="space-y-5">
      <SectionHeading eyebrow="06 / Publish" title="发布不可变情景版本" detail="同一运行可按不同用途发布；正式回测和 TAA 受因果门禁约束。" />
      <div role="radiogroup" aria-label="发布用途" className="space-y-2">
        {(Object.keys(usageMeta) as PublicationUsage[]).map((item) => {
          const allowed = run?.causality.publish_eligible_usages.includes(item) ?? false
          return (
            <button key={item} type="button" role="radio" aria-checked={usage === item} onClick={() => onUsage(item)} className={cx('flex min-h-16 w-full items-center justify-between gap-3 rounded-xl border p-3 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500', usage === item ? 'border-indigo-500 bg-indigo-50' : 'border-slate-200 bg-white')}>
              <span><span className="block text-sm font-bold text-slate-950">{usageMeta[item].name}</span><span className="mt-1 block text-[11px] text-slate-500">{usageMeta[item].description}</span></span>
              <span className={cx('shrink-0 rounded-full px-2 py-1 text-[10px] font-bold', allowed ? 'bg-emerald-100 text-emerald-800' : 'bg-slate-100 text-slate-500')}>{allowed ? '可发布' : '待门禁'}</span>
            </button>
          )
        })}
      </div>
      {run?.causality.blockers?.length ? <ul className="space-y-2 rounded-xl border border-rose-200 bg-rose-50 p-4 text-xs leading-5 text-rose-900">{run.causality.blockers.map((blocker) => <li key={blocker}>• {blocker}</li>)}</ul> : null}
      {run && !hasSavedRun ? <p className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs leading-5 text-amber-950">当前是模板草稿试算，不能作为可追溯版本发布。请先保存定义，再重新运行识别。</p> : null}
      {stale ? <p className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-950">配置或模式已经变化，请重新运行后再发布。</p> : null}
      <button type="button" disabled={!eligible || publishing} onClick={onPublish} className="min-h-12 w-full rounded-xl bg-indigo-600 px-4 text-sm font-bold text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-40">{publishing ? '正在发布…' : '发布到' + usageMeta[usage].name}</button>
      <div>
        <p className="text-xs font-bold text-slate-600">已发布与应用绑定</p>
        <div className="mt-2 space-y-2">
          {run?.publications.map((publication) => <div key={publication.id} className="rounded-xl border border-slate-200 bg-white px-3 py-3 text-xs"><div className="flex items-center justify-between gap-2"><strong>{usageMeta[publication.usage]?.name || publication.usage}</strong><span className="text-slate-500">{publication.published_at}</span></div><p className="mt-1 text-slate-500">{publication.id} · 定义 R{publication.definition_revision}</p></div>)}
          {bindings.map((binding, index) => <div key={(binding.publication_id || binding.usage) + index} className="rounded-xl border border-slate-200 bg-white px-3 py-3 text-xs"><strong>{binding.name || usageMeta[binding.usage]?.name || binding.usage}</strong><p className="mt-1 text-slate-500">{binding.status || (binding.run_id ? '已锁定运行版本' : '待绑定')}{binding.revision ? ' · R' + binding.revision : ''}</p></div>)}
          {!run?.publications.length && !bindings.length ? <p className="rounded-xl border border-dashed border-slate-300 p-5 text-center text-xs text-slate-500">尚无应用引用。</p> : null}
        </div>
      </div>
    </section>
  )
}

function SegmentTable({ run, selected, onSelect }: { run: HistoricalRegimeRun; selected: RegimeSegment | null; onSelect: (segment: RegimeSegment) => void }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[920px] text-left text-xs">
        <caption className="sr-only">历史情景区间明细</caption>
        <thead className="bg-slate-50 text-slate-500"><tr><th className="px-3 py-3">状态</th><th className="px-3 py-3">区间</th><th className="px-3 py-3">当时识别日</th><th className="px-3 py-3 text-right">持续期</th><th className="px-3 py-3 text-right">累计收益</th><th className="px-3 py-3 text-right">波动率</th><th className="px-3 py-3 text-right">最大回撤</th><th className="px-3 py-3 text-right">置信度</th></tr></thead>
        <tbody className="divide-y divide-slate-100">
          {run.segments.map((segment, index) => (
            <tr key={segment.state_id + segment.start_date} className={selected?.start_date === segment.start_date ? 'bg-indigo-50' : undefined}>
              <td className="px-3 py-3"><button type="button" onClick={() => onSelect(segment)} className="flex items-center gap-2 font-bold text-slate-900 hover:text-indigo-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"><span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: stateColor(run.states, segment.state_id, index) }} />{segment.state_label}</button></td>
              <td className="px-3 py-3 text-slate-600">{segment.start_date} → {segment.end_date}</td>
              <td className="px-3 py-3 text-slate-600">{segment.recognized_at || '—'}</td>
              <td className="px-3 py-3 text-right tabular-nums">{segment.duration_observations}</td>
              <td className="px-3 py-3 text-right tabular-nums">{formatPercent(segment.return)}</td>
              <td className="px-3 py-3 text-right tabular-nums">{formatPercent(segment.volatility)}</td>
              <td className="px-3 py-3 text-right tabular-nums text-rose-700">{formatPercent(segment.max_drawdown)}</td>
              <td className="px-3 py-3 text-right font-bold tabular-nums">{formatPercent(segment.confidence)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ConditionalTable({ run }: { run: HistoricalRegimeRun }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] text-left text-xs">
        <caption className="sr-only">各历史状态条件表现</caption>
        <thead className="bg-slate-50 text-slate-500"><tr><th className="px-3 py-3">状态</th><th className="px-3 py-3 text-right">样本数</th><th className="px-3 py-3 text-right">年化收益</th><th className="px-3 py-3 text-right">波动率</th><th className="px-3 py-3 text-right">最大回撤</th><th className="px-3 py-3 text-right">夏普</th><th className="px-3 py-3 text-right">胜率</th></tr></thead>
        <tbody className="divide-y divide-slate-100">{run.conditional_stats.map((item) => <tr key={item.state_id}><th className="px-3 py-3 font-bold text-slate-900">{item.state_label}</th><td className="px-3 py-3 text-right tabular-nums">{item.observations ?? '—'}</td><td className="px-3 py-3 text-right tabular-nums">{formatPercent(item.annualized_return ?? item.return)}</td><td className="px-3 py-3 text-right tabular-nums">{formatPercent(item.volatility)}</td><td className="px-3 py-3 text-right tabular-nums text-rose-700">{formatPercent(item.max_drawdown)}</td><td className="px-3 py-3 text-right tabular-nums">{formatNumber(item.sharpe)}</td><td className="px-3 py-3 text-right tabular-nums">{formatPercent(item.win_rate)}</td></tr>)}</tbody>
      </table>
    </div>
  )
}

function TransitionMatrix({ run }: { run: HistoricalRegimeRun }) {
  const labels = run.transition.states
  return (
    <div className="overflow-x-auto">
      <table className="min-w-[520px] text-center text-xs">
        <caption className="mb-3 text-left font-semibold text-slate-700">状态转移概率（行：当前，列：下一期）</caption>
        <thead><tr><th className="px-3 py-2 text-left text-slate-500">状态</th>{labels.map((label) => <th key={label} className="px-3 py-2 text-slate-500">{run.states.find((state) => state.id === label)?.label || label}</th>)}</tr></thead>
        <tbody>{labels.map((label, row) => <tr key={label}><th className="px-3 py-3 text-left font-bold text-slate-900">{run.states.find((state) => state.id === label)?.label || label}</th>{labels.map((column, col) => { const value = run.transition.probabilities?.[row]?.[col]; return <td key={column} className="px-3 py-3 tabular-nums" style={{ backgroundColor: typeof value === 'number' ? 'rgba(99, 102, 241, ' + Math.max(0.05, value * 0.45) + ')' : undefined }}>{formatPercent(value, 1)}</td> })}</tr>)}</tbody>
      </table>
    </div>
  )
}

function ValidationResults({ run }: { run: HistoricalRegimeRun }) {
  const stability = metricEntries(run.stability)
  const walkForward = metricEntries(run.walk_forward)
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <section className="rounded-xl border border-slate-200 p-4"><h4 className="font-bold text-slate-950">参数稳定性</h4><dl className="mt-3 grid grid-cols-2 gap-2">{stability.map((item) => <div key={item.key} className="rounded-lg bg-slate-50 p-3"><dt className="text-[11px] leading-4 text-slate-500">{item.label}</dt><dd className="mt-1 font-bold tabular-nums">{item.value}</dd></div>)}</dl>{!stability.length ? <p className="mt-3 text-xs text-slate-500">API 未返回稳定性指标。</p> : null}</section>
      <section className="rounded-xl border border-slate-200 p-4"><h4 className="font-bold text-slate-950">Walk-forward</h4><dl className="mt-3 grid grid-cols-2 gap-2">{walkForward.map((item) => <div key={item.key} className="rounded-lg bg-slate-50 p-3"><dt className="text-[11px] leading-4 text-slate-500">{item.label}</dt><dd className="mt-1 font-bold tabular-nums">{item.value}</dd></div>)}</dl>{!walkForward.length ? <p className="mt-3 text-xs text-slate-500">API 未返回逐期验证指标。</p> : null}</section>
    </div>
  )
}

function ComparisonPanel({
  runs,
  currentRun,
}: {
  runs: HistoricalRegimeRun[]
  currentRun: HistoricalRegimeRun
}) {
  const [selectedIds, setSelectedIds] = useState<string[]>([currentRun.id])
  const [comparison, setComparison] = useState<RegimeComparison | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const toggle = (id: string) => setSelectedIds((current) => current.includes(id) ? current.filter((item) => item !== id) : current.length < 3 ? [...current, id] : current)
  const compare = async () => {
    setLoading(true); setError('')
    try { setComparison(await compareHistoricalRegimeRuns(selectedIds, currentRun.id)) } catch (reason) { setError(reason instanceof Error ? reason.message : '模型对比失败。') } finally { setLoading(false) }
  }
  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
        <p className="text-xs font-bold text-slate-700">选择 2–3 个不可变运行版本</p>
        <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
          {runs.map((run) => <label key={run.id} className={cx('flex min-h-14 items-center gap-3 rounded-lg border bg-white px-3 text-xs', selectedIds.includes(run.id) ? 'border-indigo-400' : 'border-slate-200', !selectedIds.includes(run.id) && selectedIds.length >= 3 && 'opacity-50')}><input type="checkbox" checked={selectedIds.includes(run.id)} disabled={!selectedIds.includes(run.id) && selectedIds.length >= 3} onChange={() => toggle(run.id)} /><span><strong className="block text-slate-900">{run.name}</strong><span className="text-slate-500">{run.id} · {run.mode === 'realtime' ? '实时' : '事后'}</span></span></label>)}
        </div>
        <button type="button" disabled={selectedIds.length < 2 || loading} onClick={compare} className="mt-3 min-h-10 rounded-lg bg-indigo-600 px-4 text-xs font-bold text-white disabled:opacity-40">{loading ? '正在比较…' : '运行版本对比'}</button>
        {error ? <p role="alert" className="mt-3 text-xs text-rose-700">{error}</p> : null}
      </div>
      {comparison ? (
        <div className="space-y-4">
          <div className="rounded-xl bg-slate-950 p-4 text-white"><p className="text-xs text-slate-400">总体状态一致率</p><p className="mt-1 text-3xl font-bold">{formatPercent(comparison.agreement_rate)}</p></div>
          <div className="overflow-x-auto"><table className="w-full min-w-[620px] text-left text-xs"><thead className="bg-slate-50 text-slate-500"><tr><th className="px-3 py-2">版本 A</th><th className="px-3 py-2">版本 B</th><th className="px-3 py-2 text-right">一致率</th><th className="px-3 py-2 text-right">平均边界距离</th></tr></thead><tbody>{comparison.pairwise?.map((item) => <tr key={item.left_run_id + item.right_run_id}><td className="px-3 py-3">{item.left_run_id}</td><td className="px-3 py-3">{item.right_run_id}</td><td className="px-3 py-3 text-right">{formatPercent(item.agreement_rate)}</td><td className="px-3 py-3 text-right">{formatNumber(item.boundary_distance)} 期</td></tr>)}</tbody></table></div>
          {comparison.disagreement_periods?.length ? <div><p className="text-xs font-bold text-slate-700">分歧区间</p><ul className="mt-2 space-y-2 text-xs">{comparison.disagreement_periods.map((period) => <li key={period.start_date + period.end_date} className="rounded-lg border border-amber-200 bg-amber-50 p-3"><strong>{period.start_date} → {period.end_date}</strong><span className="mt-1 block text-amber-950">{Object.entries(period.states).map(([id, state]) => id + ': ' + state).join('；')}</span></li>)}</ul></div> : null}
        </div>
      ) : <p className="rounded-xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500">选择至少两个运行版本，比较状态一致率与边界分歧。</p>}
    </div>
  )
}

function CalculationAuditPanel({ run }: { run: HistoricalRegimeRun }) {
  const audits = run.calculation_audits?.length
    ? run.calculation_audits
    : run.calculation_audit
      ? [run.calculation_audit]
      : []
  if (!audits.length) return <p className="rounded-xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500">本次运行没有公式或指标中心计算计划。</p>
  return (
    <div className="space-y-5" aria-label="计算计划审计">
      {audits.map((audit, auditIndex) => {
        const dag = audit.dag ?? {}
        const typedAst = audit.typed_ast
        const nodes = typedAst?.nodes ?? dag.nodes ?? []
        const edges = typedAst?.edges ?? dag.edges ?? []
        const root = typedAst?.root ?? dag.roots?.result
        const plan = audit.plan ?? audit
        const kernels = Array.isArray(audit.kernels)
          ? audit.kernels
          : Array.isArray(plan.kernels)
            ? plan.kernels
            : []
        const signatures = Array.isArray(plan.compiled_signatures)
          ? plan.compiled_signatures
          : kernels.flatMap((kernel) => kernel.compiled_signatures ?? [])
        const sourceLabel = audit.source_kind === 'indicator'
          ? 'Indicator revision'
          : audit.source_kind === 'formula'
            ? 'Causal formula'
            : audit.family === 'analytics'
              ? 'Regime analytics'
              : 'Regime algorithm'
        return (
          <article key={(audit.source_kind || 'calculation') + auditIndex} className="overflow-hidden rounded-xl border border-slate-200 bg-white">
            <div className="flex flex-col gap-3 border-b border-slate-200 bg-slate-950 p-4 text-white lg:flex-row lg:items-start lg:justify-between">
              <div><p className="text-[10px] font-bold uppercase tracking-[0.16em] text-indigo-300">{sourceLabel}</p><h4 className="mt-1 text-base font-bold">{nodes.length ? 'typed AST → DAG → NJIT 固定签名' : '固定签名 NJIT 内核链'}</h4><p className="mt-1 break-all font-mono text-[10px] text-slate-400">{String(plan.compiled_plan_id || '未返回 plan id')}</p></div>
              <div className="flex flex-wrap gap-2 text-[10px] font-bold"><span className="rounded-full bg-emerald-400/15 px-2 py-1 text-emerald-200">{plan.compile_status === 'compiled' ? '已编译' : String(plan.compile_status || '状态未知')}</span><span className="rounded-full bg-indigo-400/15 px-2 py-1 text-indigo-200">NJIT {plan.njit_required === false ? '非必需' : '必需'}</span><span className={cx('rounded-full px-2 py-1', plan.python_fallback === 0 && plan.python_operator_calls === 0 ? 'bg-emerald-400/15 text-emerald-200' : 'bg-rose-400/20 text-rose-200')}>Python fallback {String(plan.python_fallback ?? '—')}</span></div>
            </div>
            <div className="grid gap-3 border-b border-slate-200 p-4 sm:grid-cols-2 xl:grid-cols-4">
              <div className="rounded-lg bg-slate-50 p-3"><p className="text-[10px] font-bold text-slate-500">{nodes.length ? '根节点' : '算法族'}</p><p className="mt-1 font-mono text-sm font-bold text-slate-950">{String(root ?? audit.family ?? '—')}</p></div>
              <div className="rounded-lg bg-slate-50 p-3"><p className="text-[10px] font-bold text-slate-500">节点 / 边</p><p className="mt-1 text-sm font-bold text-slate-950">{nodes.length} / {edges.length}</p></div>
              <div className="rounded-lg bg-slate-50 p-3"><p className="text-[10px] font-bold text-slate-500">Kernel</p><p className="mt-1 font-mono text-xs font-bold text-slate-950">{String(plan.kernel_version || '—')}</p></div>
              <div className="rounded-lg bg-slate-50 p-3"><p className="text-[10px] font-bold text-slate-500">Engine</p><p className="mt-1 font-mono text-xs font-bold text-slate-950">{String(plan.engine_version || '—')}</p></div>
            </div>
            {kernels.length ? <div className="overflow-x-auto border-b border-slate-200 p-4"><table className="w-full min-w-[760px] text-left text-xs"><caption className="mb-2 text-left font-bold text-slate-700">预热内核与可复现指纹</caption><thead className="bg-slate-50 text-slate-500"><tr><th className="px-3 py-2">Kernel ID</th><th className="px-3 py-2">状态</th><th className="px-3 py-2">固定签名</th><th className="px-3 py-2">Fingerprint</th></tr></thead><tbody className="divide-y divide-slate-100">{kernels.map((kernel) => <tr key={kernel.kernel_id}><td className="px-3 py-2 font-mono font-bold text-slate-900">{kernel.kernel_id}</td><td className="px-3 py-2 text-emerald-700">{kernel.compile_status === 'compiled' ? '已预热' : kernel.compile_status || '未知'}</td><td className="max-w-[360px] break-all px-3 py-2 font-mono text-[10px] text-slate-600">{kernel.compiled_signatures?.join(' · ') || '—'}</td><td className="max-w-[260px] break-all px-3 py-2 font-mono text-[10px] text-slate-500">{kernel.kernel_fingerprint || '—'}</td></tr>)}</tbody></table></div> : null}
            <div className="grid gap-4 p-4 xl:grid-cols-[minmax(0,1fr)_280px]">
              <div className="overflow-x-auto">
                <table className="w-full min-w-[600px] text-left text-xs">
                  <caption className="mb-2 text-left font-bold text-slate-700">AST / DAG 节点</caption>
                  <thead className="bg-slate-50 text-slate-500"><tr><th className="px-3 py-2">ID</th><th className="px-3 py-2">类型</th><th className="px-3 py-2">算子 / 变量</th><th className="px-3 py-2">输入</th><th className="px-3 py-2">输出类型</th></tr></thead>
                  <tbody className="divide-y divide-slate-100">{nodes.map((node, index) => {
                    const id = node.id ?? node.node_id ?? index
                    const inputs = Array.isArray(node.inputs) ? node.inputs.join(', ') : '—'
                    const inferred = node.inferred_type
                    const output = inferred && typeof inferred === 'object' ? String((inferred as Record<string, unknown>).display ?? (inferred as Record<string, unknown>).kind ?? 'typed') : String(inferred ?? node.value_type ?? '—')
                    return <tr key={String(id)} className={String(id) === String(root) ? 'bg-indigo-50' : undefined}><td className="px-3 py-2 font-mono font-bold">{String(id)}{String(id) === String(root) ? ' · root' : ''}</td><td className="px-3 py-2">{String(node.kind ?? '—')}</td><td className="px-3 py-2 font-mono">{String(node.operator_id ?? node.operator ?? node.label ?? '—')}</td><td className="px-3 py-2 font-mono text-slate-500">{inputs}</td><td className="px-3 py-2 text-slate-600">{output}</td></tr>
                  })}</tbody>
                </table>
              </div>
              <div className="space-y-3">
                <div><p className="text-xs font-bold text-slate-700">DAG 连线</p><div className="mt-2 flex max-h-40 flex-wrap gap-1.5 overflow-y-auto">{edges.map((edge, index) => <code key={String(edge.source) + '-' + String(edge.target) + index} className="rounded bg-slate-100 px-2 py-1 text-[10px] text-slate-700">{String(edge.source)} → {String(edge.target)}</code>)}{!edges.length ? <span className="text-xs text-slate-400">单节点计划，无连线</span> : null}</div></div>
                <div><p className="text-xs font-bold text-slate-700">已编译签名</p><ul className="mt-2 space-y-1.5">{signatures.map((signature) => <li key={signature} className="break-all rounded bg-emerald-50 px-2 py-1.5 font-mono text-[10px] text-emerald-900">{signature}</li>)}{!signatures.length ? <li className="text-xs text-slate-400">未返回签名</li> : null}</ul></div>
              </div>
            </div>
          </article>
        )
      })}
    </div>
  )
}

function ResultWorkspace({
  run,
  runs,
  stale,
  onSelectRun,
}: {
  run: HistoricalRegimeRun | null
  runs: HistoricalRegimeRun[]
  stale: boolean
  onSelectRun: (run: HistoricalRegimeRun) => void
}) {
  const [selectedSegment, setSelectedSegment] = useState<RegimeSegment | null>(null)
  const [tab, setTab] = useState<ResultTab>('segments')
  useEffect(() => { setSelectedSegment(run?.segments[run.segments.length - 1] ?? null) }, [run?.id])
  if (!run) return (
    <section className="grid min-h-[620px] place-items-center rounded-2xl border border-dashed border-slate-300 bg-white p-8 text-center shadow-sm">
      <div className="max-w-md"><p className="text-sm font-bold text-indigo-600">等待一次真实运行</p><h3 className="mt-2 text-xl font-bold text-slate-950">配置数据、特征与模型后运行识别</h3><p className="mt-2 text-sm leading-6 text-slate-500">结果区将展示原始序列、趋势线、状态背景带、识别日、概率、区间表与验证结果。页面不会生成前端伪结果。</p></div>
    </section>
  )
  const tabs: Array<{ id: ResultTab; label: string }> = [
    { id: 'segments', label: '区间明细' }, { id: 'conditional', label: '条件收益' }, { id: 'transition', label: '转移矩阵' }, { id: 'stability', label: '稳定性 / Walk-forward' }, { id: 'audit', label: '计算审计' }, { id: 'comparison', label: '版本对比' },
  ]
  const exportRun = () => {
    const blob = new Blob([JSON.stringify(run, null, 2)], { type: 'application/json;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'historical-regime-' + run.id + '.json'
    anchor.click()
    URL.revokeObjectURL(url)
  }
  return (
    <div className="space-y-5">
      <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" aria-label="历史情景识别结果">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div><p className="text-[11px] font-bold uppercase tracking-[0.16em] text-indigo-600">Immutable run</p><h3 className="mt-1 text-lg font-bold text-slate-950">{run.name}</h3><p className="mt-1 text-xs text-slate-500">{run.id} · {run.definition_revision == null ? '未版本化试算' : '定义 R' + run.definition_revision} · {run.mode === 'realtime' ? '实时识别' : '事后划分'} · {run.created_at}</p></div>
          <div className="flex flex-wrap items-center gap-2">
            <CausalityBadge run={run} />
            {stale ? <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-bold text-amber-900">配置已变化 · 结果过期</span> : <span className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-bold text-emerald-800">结果与配置一致</span>}
            <button type="button" onClick={exportRun} className="min-h-9 rounded-lg border border-slate-300 bg-white px-3 text-xs font-bold text-slate-700 hover:border-indigo-300 hover:text-indigo-700">导出完整 JSON</button>
            <label className="text-xs font-bold text-slate-600">运行版本
              <select aria-label="运行版本" value={run.id} onChange={(event) => { const selected = runs.find((item) => item.id === event.target.value); if (selected) onSelectRun(selected) }} className="ml-2 min-h-9 rounded-lg border border-slate-300 bg-white px-2 font-normal">
                {runs.map((item) => <option key={item.id} value={item.id}>{item.id} · {item.mode === 'realtime' ? '实时' : '事后'}</option>)}
              </select>
            </label>
          </div>
        </div>
        <div className="mt-5 grid gap-4 2xl:grid-cols-[minmax(0,1fr)_280px]">
          <RegimeChart run={run} onSelectSegment={setSelectedSegment} />
          <EvidenceInspector run={run} segment={selectedSegment} />
        </div>
      </section>
      <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
        <div role="tablist" aria-label="历史识别结果分析" className="flex overflow-x-auto border-b border-slate-200 p-2">
          {tabs.map((item) => <button key={item.id} type="button" role="tab" aria-selected={tab === item.id} onClick={() => setTab(item.id)} className={cx('min-h-10 shrink-0 rounded-lg px-3 text-xs font-bold focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500', tab === item.id ? 'bg-slate-950 text-white' : 'text-slate-600 hover:bg-slate-50')}>{item.label}</button>)}
        </div>
        <div role="tabpanel" className="p-4">
          {tab === 'segments' ? <SegmentTable run={run} selected={selectedSegment} onSelect={setSelectedSegment} /> : null}
          {tab === 'conditional' ? <ConditionalTable run={run} /> : null}
          {tab === 'transition' ? <TransitionMatrix run={run} /> : null}
          {tab === 'stability' ? <ValidationResults run={run} /> : null}
          {tab === 'audit' ? <CalculationAuditPanel run={run} /> : null}
          {tab === 'comparison' ? <ComparisonPanel runs={runs} currentRun={run} /> : null}
        </div>
      </section>
    </div>
  )
}

export default function HistoricalRegimeCenter() {
  const [workbenchOpen, setWorkbenchOpen] = useState(false)
  const [workbenchInitialDefinition, setWorkbenchInitialDefinition] = useState<RegimeGraphDefinition | undefined>()
  const [meta, setMeta] = useState<HistoricalRegimeMeta | null>(null)
  const [definitions, setDefinitions] = useState<HistoricalRegimeDefinition[]>([])
  const [runs, setRuns] = useState<HistoricalRegimeRun[]>([])
  const [draft, setDraft] = useState<HistoricalRegimeDefinition | null>(null)
  const [savedSignature, setSavedSignature] = useState('')
  const [runSignature, setRunSignature] = useState('')
  const [currentRun, setCurrentRun] = useState<HistoricalRegimeRun | null>(null)
  const [runDetailLoading, setRunDetailLoading] = useState(false)
  const [mode, setMode] = useState<RegimeMode>('realtime')
  const [activeStep, setActiveStep] = useState<PipelineStep>('data')
  const [usage, setUsage] = useState<PublicationUsage>('research_display')
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [saving, setSaving] = useState(false)
  const [publishing, setPublishing] = useState(false)
  const [convertingV2, setConvertingV2] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const runDetailRequest = useRef(0)

  useEffect(() => {
    if (workbenchOpen) return undefined
    let cancelled = false
    const load = async () => {
      setLoading(true); setError('')
      try {
        const [nextMeta, nextDefinitions, nextRuns] = await Promise.all([
          getHistoricalRegimeMeta(),
          listHistoricalRegimeDefinitions(),
          listHistoricalRegimeRuns(),
        ])
        if (cancelled) return
        setMeta(nextMeta)
        setDefinitions(nextDefinitions)
        setRuns(nextRuns)
        const initialDefinition = nextDefinitions[0] ?? (nextMeta.templates[0] ? normalizeTemplateDefinition(nextMeta.templates[0]) : null)
        if (!initialDefinition) {
          setError('接口未返回可编辑的历史情景定义或模板。')
          return
        }
        const nextDraft = cloneDefinition(initialDefinition)
        const signature = definitionSignature(nextDraft)
        setDraft(nextDraft)
        setSavedSignature(nextDraft.id ? signature : '')
        const initialRunSummary = nextDraft.id ? nextRuns.find((item) => item.definition_id === nextDraft.id) ?? null : null
        setCurrentRun(null)
        if (initialRunSummary) {
          setRunDetailLoading(true)
          try {
            const initialRun = await getHistoricalRegimeRun(initialRunSummary.id)
            if (cancelled) return
            setCurrentRun(initialRun)
            setMode(initialRun.mode)
            setRunSignature(signatureForRun(initialRun, nextDraft))
          } catch (reason) {
            if (!cancelled) setError(reason instanceof Error ? reason.message : '历史情景运行详情加载失败。')
          } finally {
            if (!cancelled) setRunDetailLoading(false)
          }
        }
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : '历史情景工作台加载失败。')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    void load()
    return () => { cancelled = true; runDetailRequest.current += 1 }
  }, [workbenchOpen])

  const loadRunDetail = async (summary: HistoricalRegimeRun, signatureDraft: HistoricalRegimeDefinition) => {
    const requestId = ++runDetailRequest.current
    setRunDetailLoading(true); setCurrentRun(null); setRunSignature(''); setError('')
    try {
      const detail = await getHistoricalRegimeRun(summary.id)
      if (requestId !== runDetailRequest.current) return
      setCurrentRun(detail)
      setMode(detail.mode)
      setRunSignature(signatureForRun(detail, signatureDraft))
    } catch (reason) {
      if (requestId === runDetailRequest.current) setError(reason instanceof Error ? reason.message : '历史情景运行详情加载失败。')
    } finally {
      if (requestId === runDetailRequest.current) setRunDetailLoading(false)
    }
  }

  if (workbenchOpen) return <HistoricalRegimeWorkbench initialDefinition={workbenchInitialDefinition} onExit={() => { setWorkbenchOpen(false); setWorkbenchInitialDefinition(undefined) }} />

  if (loading) return <div role="status" className="grid min-h-[520px] place-items-center rounded-2xl border border-slate-200 bg-white"><div className="text-center"><span className="mx-auto block h-8 w-8 animate-spin rounded-full border-4 border-indigo-200 border-t-indigo-600" /><p className="mt-3 text-sm font-semibold text-slate-600">正在加载历史情景元数据与版本…</p></div></div>

  if (!meta || !draft) return (
    <section role="alert" className="rounded-2xl border border-rose-200 bg-rose-50 p-6">
      <h2 className="text-lg font-bold text-rose-950">历史情景识别暂不可用</h2>
      <p className="mt-2 text-sm text-rose-900">{error || '接口未返回工作台所需数据。'}</p>
      <button type="button" onClick={() => window.location.reload()} className="mt-4 min-h-10 rounded-xl bg-rose-800 px-4 text-sm font-bold text-white">重新加载</button>
    </section>
  )

  const signature = definitionSignature(draft)
  const saveDirty = signature !== savedSignature
  const resultStale = Boolean(currentRun && (signature !== runSignature || currentRun.mode !== mode))

  const changeDraft = (next: HistoricalRegimeDefinition) => {
    setDraft(next)
    setNotice('')
  }

  const copyCurrentToV2 = async () => {
    if (!draft.id || !draft.revision) { setError('请先选择或保存一个经典版定义，再复制为 V2 图谱。'); return }
    setConvertingV2(true); setError(''); setNotice('')
    try {
      const converted = await copyHistoricalRegimeDefinitionToV2(draft.id, draft.revision)
      setWorkbenchInitialDefinition(converted.definition)
      setWorkbenchOpen(true)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '经典定义复制为 V2 图谱失败。')
    } finally { setConvertingV2(false) }
  }

  const selectTemplate = (template: RegimeTemplate) => {
    const next = normalizeTemplateDefinition(template)
    if (!next) {
      setError('模板“' + template.name + '”没有返回完整定义，无法载入。')
      return
    }
    setDraft(next)
    setSavedSignature('')
    runDetailRequest.current += 1
    setCurrentRun(null)
    setRunDetailLoading(false)
    setRunSignature('')
    setMode('realtime')
    setNotice('已载入“' + template.name + '”，请检查参数后运行。')
    setError('')
  }

  const selectDefinition = (id: string) => {
    const selected = definitions.find((definition) => definition.id === id)
    if (!selected) return
    const next = cloneDefinition(selected)
    const nextSignature = definitionSignature(next)
    setDraft(next)
    setSavedSignature(nextSignature)
    const linkedRun = runs.find((run) => run.definition_id === id) ?? null
    if (linkedRun) void loadRunDetail(linkedRun, next)
    else { runDetailRequest.current += 1; setCurrentRun(null); setRunDetailLoading(false); setRunSignature('') }
    setNotice('')
    setError('')
  }

  const saveDefinition = async () => {
    setSaving(true); setError(''); setNotice('')
    try {
      const saved = draft.id ? await updateHistoricalRegimeDefinition(draft) : await createHistoricalRegimeDefinition(draft)
      const next = cloneDefinition(saved)
      setDraft(next)
      setDefinitions((current) => {
        const without = current.filter((item) => item.id !== next.id)
        return [next, ...without]
      })
      setSavedSignature(definitionSignature(next))
      setNotice('定义已保存为可追溯修订版。')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '保存历史情景定义失败。')
    } finally {
      setSaving(false)
    }
  }

  const runDefinition = async () => {
    setRunning(true); setError(''); setNotice('')
    try {
      let requestedDefinition: HistoricalRegimeDefinition | { id: string; revision: number }
      if (draft.id && draft.revision && !saveDirty) {
        requestedDefinition = { id: draft.id, revision: draft.revision }
      } else {
        const trialDefinition = cloneDefinition(draft)
        delete trialDefinition.id
        delete trialDefinition.revision
        delete trialDefinition.created_at
        delete trialDefinition.updated_at
        delete trialDefinition.status
        requestedDefinition = trialDefinition
      }
      const nextRun = await runHistoricalRegime(requestedDefinition, mode)
      runDetailRequest.current += 1
      setRuns((current) => [nextRun, ...current.filter((item) => item.id !== nextRun.id)])
      setCurrentRun(nextRun)
      setRunDetailLoading(false)
      setRunSignature(signature)
      setNotice(nextRun.definition_revision == null ? '草稿试算已完成；保存定义并重新运行后才能发布。' : '识别运行已完成；结果来自后端锁定数据与算法快照。')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '历史情景识别运行失败。')
    } finally {
      setRunning(false)
    }
  }

  const publishRun = async () => {
    if (!currentRun) return
    setPublishing(true); setError(''); setNotice('')
    try {
      const result = await publishHistoricalRegimeRun(currentRun.id, usage)
      const updated = { ...currentRun, publications: result.publications, application_bindings: result.application_bindings ?? currentRun.application_bindings }
      setCurrentRun(updated)
      setRuns((current) => current.map((run) => run.id === updated.id ? updated : run))
      setNotice('运行已发布到“' + usageMeta[usage].name + '”，下游将锁定本次版本。')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '发布历史情景运行失败。')
    } finally {
      setPublishing(false)
    }
  }

  const editor = activeStep === 'data'
    ? <DataPanel meta={meta} draft={draft} onChange={changeDraft} onTemplate={selectTemplate} />
    : activeStep === 'features'
      ? <FeaturePanel meta={meta} draft={draft} onChange={changeDraft} />
      : activeStep === 'model'
        ? <ModelPanel meta={meta} draft={draft} onChange={changeDraft} />
        : activeStep === 'states'
          ? <StatePanel draft={draft} onChange={changeDraft} />
          : activeStep === 'validation'
            ? <ValidationPanel draft={draft} run={currentRun} onChange={changeDraft} />
            : <PublishPanel draft={draft} run={currentRun} stale={resultStale} usage={usage} publishing={publishing} onUsage={setUsage} onPublish={publishRun} />

  return (
    <div className="space-y-5" data-testid="historical-regime-center">
      <header className="rounded-2xl border border-slate-800 bg-slate-950 px-5 py-5 text-white shadow-sm sm:px-6">
        <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
          <div><p className="text-xs font-bold uppercase tracking-[0.2em] text-indigo-300">Historical regime research</p><h2 className="mt-1 text-2xl font-bold">历史情景识别</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">从真实原始数据出发，自定义指标与模型，把历史拆成牛熊震荡、经济周期或风格轮动区间，并发布为可复用版本。</p></div>
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" disabled={!draft.id || convertingV2} onClick={() => void copyCurrentToV2()} className="min-h-9 rounded-lg border border-indigo-300 px-3 text-xs font-bold text-indigo-100 disabled:opacity-40">{convertingV2 ? '正在复制…' : '复制当前 V1 为 V2 图谱'}</button>
            <button type="button" onClick={() => { setWorkbenchInitialDefinition(undefined); setWorkbenchOpen(true) }} className="min-h-9 rounded-lg bg-indigo-500 px-3 text-xs font-bold text-white shadow hover:bg-indigo-400">进入 V2 自由工作台</button>
            <CausalityBadge run={currentRun} />
            <span className={cx('rounded-full px-3 py-1 text-xs font-bold', resultStale ? 'bg-amber-300 text-amber-950' : 'bg-white/10 text-slate-200')}>{resultStale ? '结果已过期' : currentRun ? '结果已同步' : '待运行'}</span>
          </div>
        </div>
      </header>

      <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm" aria-label="研究运行控制">
        <div className="grid gap-3 lg:grid-cols-[minmax(230px,1fr)_auto_auto_auto] lg:items-end">
          <label className="text-xs font-bold text-slate-600">已保存定义
            <select aria-label="已保存定义" value={draft.id || ''} onChange={(event) => selectDefinition(event.target.value)} className="mt-1 min-h-11 w-full rounded-xl border border-slate-300 bg-white px-3 text-sm font-normal">
              {!draft.id ? <option value="">当前模板草稿 · 尚未保存</option> : null}
              {definitions.map((definition) => <option key={definition.id} value={definition.id}>{definition.name} · R{definition.revision ?? 1}</option>)}
            </select>
          </label>
          <div role="radiogroup" aria-label="识别模式" className="flex rounded-xl border border-slate-300 bg-slate-50 p-1">
            <button type="button" role="radio" aria-checked={mode === 'realtime'} onClick={() => setMode('realtime')} className={cx('min-h-9 rounded-lg px-3 text-xs font-bold', mode === 'realtime' ? 'bg-emerald-600 text-white' : 'text-slate-600')}>实时识别</button>
            <button type="button" role="radio" aria-checked={mode === 'retrospective'} onClick={() => setMode('retrospective')} className={cx('min-h-9 rounded-lg px-3 text-xs font-bold', mode === 'retrospective' ? 'bg-amber-500 text-slate-950' : 'text-slate-600')}>事后划分</button>
          </div>
          <button type="button" disabled={!saveDirty || saving || running} onClick={saveDefinition} className="min-h-11 rounded-xl border border-indigo-200 px-4 text-sm font-bold text-indigo-700 hover:bg-indigo-50 disabled:opacity-40">{saving ? '保存中…' : '保存定义'}</button>
          <button type="button" disabled={running || saving} onClick={runDefinition} className="min-h-11 rounded-xl bg-indigo-600 px-5 text-sm font-bold text-white shadow-sm hover:bg-indigo-500 disabled:opacity-50">{running ? '正在运行识别…' : '运行历史识别'}</button>
        </div>
        {running ? <div role="progressbar" aria-label="历史情景识别运行进度" aria-valuetext="后端正在锁定数据并计算" className="mt-3 h-2 overflow-hidden rounded-full bg-indigo-100"><span className="block h-full w-2/3 animate-pulse rounded-full bg-indigo-600" /></div> : null}
        {error ? <p role="alert" className="mt-3 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">{error}</p> : null}
        {notice ? <p role="status" className="mt-3 rounded-xl border border-indigo-200 bg-indigo-50 px-4 py-3 text-sm text-indigo-950">{notice}</p> : null}
      </section>

      <div className="grid gap-5 xl:grid-cols-[200px_minmax(350px,0.78fr)_minmax(0,1.35fr)]">
        <PipelineNavigation active={activeStep} onChange={setActiveStep} />
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">{editor}</section>
        {runDetailLoading ? <section role="status" className="grid min-h-[620px] place-items-center rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-sm"><div><span className="mx-auto block h-8 w-8 animate-spin rounded-full border-4 border-indigo-200 border-t-indigo-600" /><p className="mt-3 text-sm font-semibold text-slate-600">正在按需读取所选运行的完整序列与分析结果…</p></div></section> : <ResultWorkspace run={currentRun} runs={runs} stale={resultStale} onSelectRun={(run) => { void loadRunDetail(run, draft) }} />}
      </div>
    </div>
  )
}
