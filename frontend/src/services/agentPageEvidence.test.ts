import { afterEach, describe, expect, it, vi } from 'vitest'
import { MAX_PAGE_EVIDENCE_CHARS, buildIndicatorStudioEvidence, freezePageSnapshot, newPageSnapshotId, submittedParameterOverrides, type IndicatorStudioEvidenceInput } from './agentPageEvidence'
import type { AgentPreview, PageEvidenceSnapshot } from './agent'
import type { EvaluationResult, IndicatorDraft, TimeSeriesIndicatorResult } from './customIndicators'

const draft = { name: '波动率', description: 'd', expression: 'std(returns, 1)', unit: '%', display_format: 'percent', precision: 2, direction: 'lower_better', annual_risk_free_rate_percent: 1.5, parameter_contract_version: '1.0', parameter_schema: [{ id: 'window', label: '窗口', type: 'integer', default: 20, minimum: 1, maximum: 200, step: 1 }] }
const previewDraft = { ...draft, name: '滚动夏普试算', expression: 'mean(rolling_window(returns, window))', result_kind: 'time_series', axis_anchor: 'adjusted_nav', history_policy: 'window', lookback_observations: 37, minimum_observations: 37, dsl_version: '2.1.0', fixed_parameters: [{ id: 'window', value: 37 }] }

const scalarResult = (value: unknown, target = '510300.SH'): EvaluationResult => ({
  indicator_id: null, indicator_revision: null, indicator_name: '波动率', target: { kind: 'etf', product_id: target, name: target },
  period: '1Y', value: value as number, value_type: 'number', status: 'ok', warnings: [],
  window: { requested_as_of: null, effective_as_of: '2026-09-18', start_date: '2025-09-01', end_date: '2026-09-18', observation_count: 240, data_latest_date: '2026-09-18' },
  presentation: { unit: '%', precision: 2 } as EvaluationResult['presentation'],
  parameters: { window: 20 }, parameter_hash: 'ph-1',
  input_requirements: { status: 'ready', required_count: 1, available_count: 1, items: [{ variable_id: 'returns', status: 'available', source_dataset: 'fund_nav' }], blocking_inputs: [], partial_inputs: [] },
  series: null,
}) as unknown as EvaluationResult

const seriesResult = (product = '510300.SH', count = 12): TimeSeriesIndicatorResult => ({
  indicator_id: null, indicator_revision: null, instance_key: 'i', indicator_name: '滚动波动率', result_kind: 'time_series',
  target: { kind: 'etf', product_id: product, name: product }, period: '1Y', parameters: { window: 20 }, parameter_hash: 'ph-2',
  axis_anchor: 'adjusted_nav', history_policy: 'window', status: 'ok', warnings: [], window: { requested_as_of: null, effective_as_of: '2026-09-18', observation_count: count },
  dates: Array.from({ length: count }, (_, index) => `2026-09-${String(index + 1).padStart(2, '0')}`),
  channels: [
    { id: 'vol', label: '年化波动率', unit: '%', display_format: 'percent', precision: 2, output_measure: 'volatility',
      values: Array.from({ length: count }, (_, index) => index === 4 ? null : index === 7 ? 0 : index), null_count: 1 },
    { id: 'level', label: '净值', unit: '元', display_format: 'number', precision: 4, output_measure: 'adjusted_nav',
      values: Array.from({ length: count }, (_, index) => 100 + index), null_count: 0 },
  ],
}) as unknown as TimeSeriesIndicatorResult

const preview = (overrides: Partial<AgentPreview> = {}): AgentPreview => ({
  preview_id: 'preview-1', run_id: 'run-1', session_id: 's-1', definition_hash: 'h'.repeat(64), target: { kind: 'etf', product_id: '510300.SH' },
  period: '6M', as_of: '2019-12-30', result_kind: 'time_series', created_at: '2026-09-21T01:00:00+00:00', expires_at: '2026-09-22T01:00:00+00:00',
  definition: previewDraft as unknown as AgentPreview['definition'], result: { results: [seriesResult()] },
  context_hash: 'ctx-1', data_generation: 'gen-1', effective_context: { as_of: '2019-12-27', run_mode: 'pit', data_release_id: 'rel-1' },
  ...overrides,
} as AgentPreview)

const frozenRequest = (overrides: Partial<IndicatorStudioEvidenceInput['manual_request']> = {}) => ({
  definition: { expression: 'std(returns, 1)', result_kind: 'scalar', parameter_contract_version: '1.0' },
  parameters: { window: 20 }, parameters_submitted: true, targets: [{ kind: 'etf', product_id: '510300.SH' }],
  period: '1Y', as_of: null, requested_at: '2026-09-21T02:00:00+00:00', completed_at: '2026-09-21T02:00:03+00:00', ...overrides,
})

const input = (overrides: Partial<IndicatorStudioEvidenceInput> = {}): IndicatorStudioEvidenceInput => ({
  selection: { indicator_id: 'indicator-1', indicator_revision: 3, name: '波动率' },
  draft: draft as unknown as IndicatorStudioEvidenceInput['draft'],
  active_preview_definition: null,
  definition_dirty: false, validation: { valid: true, diagnostics: [] },
  canvas_pending: false, parameter_pending: false, previewing: false,
  targets: [{ kind: 'etf', product_id: '510300.SH', name: '沪深300ETF' }], period: '1Y', as_of: '',
  runtime_parameters: { window: 20 },
  manual_results: [], manual_series_results: [], manual_request: null, agent_preview: null,
  ...overrides,
})

afterEach(() => { vi.unstubAllGlobals() })

describe('buildIndicatorStudioEvidence', () => {
  it('保留零值、空值和缺失的差别，并把结果分区与当前编辑定义分开冻结', () => {
    const snapshot = buildIndicatorStudioEvidence(input({
      manual_results: [scalarResult(0), scalarResult(null, '512960.SH')],
      manual_request: frozenRequest(),
      // 用户在结果之后改了公式和参数：编辑分区必须是最新输入，结果分区必须保留原请求口径。
      draft: { ...draft, expression: 'mean(returns)' } as unknown as IndicatorStudioEvidenceInput['draft'],
      runtime_parameters: { window: 60 }, definition_dirty: true,
    }))
    const editing = snapshot.sections.editing as any
    const results = snapshot.sections.results as any
    expect(editing.definition.expression).toBe('mean(returns)')
    expect(editing.definition.source).toBe('editor')
    expect(editing.runtime_inputs.runtime_parameters).toEqual({ window: 60 })
    expect(editing.definition.definition_dirty).toBe(true)
    expect(results.displayed_source).toBe('manual_preview')
    expect(results.frozen_request.definition.expression).toBe('std(returns, 1)')
    expect(results.frozen_request.parameters).toEqual({ window: 20 })
    expect(results.groups[0].value).toBe(0)
    expect(results.groups[1].value).toBeNull()
    expect(results.groups[0].window.effective_as_of).toBe('2026-09-18')
  })

  it('采纳的 AI 试算与编辑器定义分开陈述，并保留完整定义契约字段', () => {
    const snapshot = buildIndicatorStudioEvidence(input({
      agent_preview: preview(),
      active_preview_definition: previewDraft as unknown as Record<string, unknown>,
      targets: [{ kind: 'etf', product_id: '510300.SH', name: '510300.SH' }],
      period: '6M', as_of: '2019-12-30', runtime_parameters: { window: 20 },
    }))
    const editing = snapshot.sections.editing as any
    const results = snapshot.sections.results as any
    // 编辑器定义保持原样，采纳的试算定义单独列出，两者公式明显不同。
    expect(editing.definition.source).toBe('editor')
    expect(editing.definition.expression).toBe('std(returns, 1)')
    expect(editing.active_preview.source).toBe('adopted_agent_preview')
    expect(editing.active_preview.definition.expression).toBe('mean(rolling_window(returns, window))')
    expect(editing.active_preview.definition.axis_anchor).toBe('adjusted_nav')
    expect(editing.active_preview.definition.history_policy).toBe('window')
    expect(editing.active_preview.definition.lookback_observations).toBe(37)
    expect(editing.active_preview.definition.dsl_version).toBe('2.1.0')
    expect(editing.active_preview.definition.fixed_parameters).toEqual([{ id: 'window', value: 37 }])
    expect(editing.active_preview.result_adopted).toBe(true)
    expect(editing.active_preview.runtime_inputs).toMatchObject({ period: '6M', as_of: '2019-12-30', runtime_parameters: { window: 20 } })
    expect(editing.active_preview.definition.annual_risk_free_rate_percent).toBe(1.5)
    expect(editing.active_preview.definition.parameter_contract_version).toBe('1.0')
    expect(editing.active_preview.definition.parameter_schema[0]).toMatchObject({ id: 'window', minimum: 1, maximum: 200, step: 1 })
    expect(results.displayed_source).toBe('agent_preview')
    expect(results.frozen_definition.expression).toBe('mean(rolling_window(returns, window))')
    expect(results.provenance).toMatchObject({ preview_id: 'preview-1', run_id: 'run-1', context_hash: 'ctx-1', data_generation: 'gen-1', effective_context: { as_of: '2019-12-27' } })
  })

  it('采纳试算结果被清除后，活动定义与当前输入仍是试算口径，且不编造已显示结果', () => {
    const snapshot = buildIndicatorStudioEvidence(input({
      active_preview_definition: previewDraft as unknown as Record<string, unknown>,
      agent_preview: null,
      targets: [{ kind: 'etf', product_id: '510300.SH', name: '510300.SH' }],
      period: '3M', as_of: '2019-11-29', runtime_parameters: { window: 15 },
    }))
    const editing = snapshot.sections.editing as any
    const results = snapshot.sections.results as any
    expect(editing.definition.source).toBe('editor')
    expect(editing.definition.expression).toBe('std(returns, 1)')
    expect(editing.active_preview.source).toBe('adopted_agent_preview')
    expect(editing.active_preview.definition.expression).toBe('mean(rolling_window(returns, window))')
    expect(editing.active_preview.result_adopted).toBe(false)
    expect(editing.active_preview.preview_id).toBeNull()
    expect(editing.active_preview.runtime_inputs).toMatchObject({ period: '3M', as_of: '2019-11-29', runtime_parameters: { window: 15 } })
    expect(results.displayed_source).toBeNull()
    expect(results.groups).toEqual([])
    expect((snapshot.sections.series as any).displayed_source).toBeNull()
  })

  it('series 分区保留页面上实际存在的完整数组，包括中间零值与缺失', () => {
    const snapshot = buildIndicatorStudioEvidence(input({
      targets: [{ kind: 'etf', product_id: '510300.SH', name: '沪深300ETF' }, { kind: 'etf', product_id: '512960.SH', name: '证券ETF' }],
      manual_series_results: [seriesResult(), seriesResult('512960.SH')],
    }))
    const series = snapshot.sections.series as any
    expect(series.displayed_source).toBe('manual_preview')
    expect(series.groups).toHaveLength(2)
    expect(series.groups[0].dates).toHaveLength(12)
    expect(series.groups[0].channels.map((channel: any) => channel.id)).toEqual(['vol', 'level'])
    expect(series.groups[0].channels[0].values[7]).toBe(0)
    expect(series.groups[0].channels[0].values[4]).toBeNull()
    expect(series.groups[0].channels[1].values[5]).toBe(105)
    expect(series.groups[1].target.product_id).toBe('512960.SH')
    // 摘要仍保留首尾采样与省略计数，和完整数组互补。
    const summary = (snapshot.sections.results as any).groups[0].channels[0]
    expect(summary.sample.head).toHaveLength(3)
    expect(summary.sample.tail).toHaveLength(3)
    expect(summary.sample.omitted_values).toBe(6)
  })

  it('标量结果未包含曲线时 series 分区显式说明，不伪造数据', () => {
    const snapshot = buildIndicatorStudioEvidence(input({ manual_results: [scalarResult(0)], manual_request: frozenRequest() }))
    const series = snapshot.sections.series as any
    expect(series.groups[0].dates).toEqual([])
    expect(series.groups[0].channels).toEqual([])
    expect(series.groups[0].unavailable.code).toBe('series_not_on_page')
    expect((snapshot.sections.results as any).groups[0].value).toBe(0)
  })

  it('无结果、画布未应用和正在计算都显式说明，不伪造数值', () => {
    const empty = buildIndicatorStudioEvidence(input({ canvas_pending: true, previewing: true }))
    const results = empty.sections.results as any
    expect(results.displayed_source).toBeNull()
    expect(results.groups).toEqual([])
    expect(results.pending.join('')).toContain('画布修改尚未应用')
    expect(results.pending.join('')).toContain('正在计算中')
    expect(results.pending.join('')).not.toContain('尚未预览')
    expect((empty.sections.series as any).displayed_source).toBeNull()
  })

  it('时序摘要采样有界并说明省略数量', () => {
    const snapshot = buildIndicatorStudioEvidence(input({ manual_series_results: [seriesResult()] }))
    const group = (snapshot.sections.results as any).groups[0]
    const channel = group.channels[0]
    expect(channel.point_count).toBe(12)
    expect(channel.null_count).toBe(1)
    expect(channel.sample.head).toHaveLength(3)
    expect(channel.sample.tail).toHaveLength(3)
    expect(channel.sample.omitted_values).toBe(6)
    expect(channel.sample.head[0]).toEqual({ date: '2026-09-01', value: 0 })
    expect(channel.sample.tail[2]).toEqual({ date: '2026-09-12', value: 11 })
    // 中间缺失和中间零值落在采样窗口之外，只有 series 分区能看到。
    expect([...channel.sample.head, ...channel.sample.tail].map((point: any) => point.value)).toEqual([0, 1, 2, 9, 10, 11])
    expect(group.date_range).toEqual({ start: '2026-09-01', end: '2026-09-12' })

    const small = buildIndicatorStudioEvidence(input({ manual_series_results: [{ ...seriesResult('510300.SH', 4) } as TimeSeriesIndicatorResult] }))
    const smallChannel = (small.sections.results as any).groups[0].channels[0]
    expect(smallChannel.sample.head).toHaveLength(4)
    expect(smallChannel.sample.tail).toEqual([])
    expect(smallChannel.sample.omitted_values).toBe(0)
  })

  it('多产品分别成组，状态与数据来源保持原始值', () => {
    const snapshot = buildIndicatorStudioEvidence(input({
      targets: [{ kind: 'etf', product_id: '510300.SH', name: '沪深300ETF' }, { kind: 'fund', product_id: '000001.OF', name: '示例基金' }],
      manual_results: [scalarResult(0), scalarResult(0.5, '000001.OF')],
    }))
    const results = snapshot.sections.results as any
    expect(results.groups.map((row: any) => row.target.product_id)).toEqual(['510300.SH', '000001.OF'])
    expect(results.groups[0].status).toBe('ok')
    expect(results.groups[0].data_context).toBeNull()
    expect(results.groups[0].input_requirements).toMatchObject({ status: 'ready', omitted_items: 0 })
    expect((snapshot.sections.editing as any).runtime_inputs.targets).toHaveLength(2)
  })

  it('极端时序只显式省略 series，结果摘要与定义参数仍保留', () => {
    const huge = buildIndicatorStudioEvidence(input({
      manual_series_results: [seriesResult('510300.SH', 40_000)],
    }))
    const series = huge.sections.series as any
    expect(series.omitted).toMatchObject({ code: 'page_evidence_series_too_large', groups: 1, channels: 2, points: 80_000 })
    const summary = (huge.sections.results as any).groups[0]
    expect(summary.status).toBe('ok')
    expect(summary.parameters).toEqual({ window: 20 })
    expect(summary.channels[0].sample.head).toHaveLength(3)
    expect(summary.channels[0].point_count).toBe(40_000)
    expect((huge.sections.editing as any).definition.expression).toBe('std(returns, 1)')
    expect(JSON.stringify(huge).length).toBeLessThanOrEqual(MAX_PAGE_EVIDENCE_CHARS)
  })

  it('只有连摘要都超限时才逐级省略 results 和 editing，并显式说明', () => {
    const snapshot = buildIndicatorStudioEvidence(input({ draft: { ...draft, description: 'x'.repeat(600_000) } as unknown as IndicatorStudioEvidenceInput['draft'] }))
    expect((snapshot.sections.series as any).omitted.code).toBe('page_evidence_series_too_large')
    expect((snapshot.sections.results as any).omitted.code).toBe('page_evidence_too_large')
    expect((snapshot.sections.editing as any).omitted.code).toBe('page_evidence_too_large')
    expect(JSON.stringify(snapshot).length).toBeLessThanOrEqual(MAX_PAGE_EVIDENCE_CHARS)
  })

  it('只记录请求实际提交的参数覆盖，并区分请求与完成时间', () => {
    const notSubmitted = buildIndicatorStudioEvidence(input({
      runtime_parameters: { window: 60 }, manual_results: [scalarResult(0)],
      manual_request: frozenRequest({ definition: { expression: 'x' }, parameters: null, parameters_submitted: false }),
    }))
    expect((notSubmitted.sections.results as any).frozen_request).toMatchObject({ parameters: null, parameters_submitted: false, requested_at: '2026-09-21T02:00:00+00:00', completed_at: '2026-09-21T02:00:03+00:00' })
    expect((notSubmitted.sections.editing as any).runtime_inputs.runtime_parameters).toEqual({ window: 60 })
    expect(submittedParameterOverrides({ parameter_contract_version: '1.0' }, { window: 60 })).toEqual({ window: 60 })
    expect(submittedParameterOverrides({ parameter_contract_version: null }, { window: 60 })).toBeNull()
    expect(submittedParameterOverrides({}, { window: 60 })).toBeNull()
  })

  it('快照是版本化信封且标识可被服务端校验', () => {
    const snapshot = buildIndicatorStudioEvidence(input())
    expect(snapshot).toMatchObject({ version: 1, page: 'indicator-studio' })
    expect(snapshot.captured_at).toMatch(/^\d{4}-\d{2}-\d{2}T/)
    expect(newPageSnapshotId()).toMatch(/^snap-[0-9a-f]{32}$/)
    expect(newPageSnapshotId()).not.toBe(newPageSnapshotId())
  })
})

describe('freezePageSnapshot', () => {
  const source = (): PageEvidenceSnapshot => ({ version: 1, snapshot_id: newPageSnapshotId(), captured_at: 'now', page: 'indicator-studio', sections: { editing: { marker: 'live' } } })

  it('正常克隆后与页面对象完全脱钩', () => {
    const live = source()
    const frozen = freezePageSnapshot(live)!
    live.sections.editing = { marker: 'mutated' }
    expect(frozen.sections.editing).toEqual({ marker: 'live' })
  })

  it('structuredClone 失败时退回 JSON 安全副本，不返回可变页面引用', () => {
    const live = source()
    vi.stubGlobal('structuredClone', () => { throw new Error('unsupported') })
    const frozen = freezePageSnapshot(live)!
    expect(frozen).not.toBe(live)
    live.sections.editing = { marker: 'mutated' }
    expect(frozen.sections.editing).toEqual({ marker: 'live' })
  })

  it('无法安全序列化时显式声明冻结失败，不静默发送共享引用', () => {
    const live = source()
    ;(live.sections.editing as any).self = live
    vi.stubGlobal('structuredClone', () => { throw new Error('unsupported') })
    const frozen = freezePageSnapshot(live)!
    expect(frozen.sections.editing).toMatchObject({ omitted: { code: 'page_evidence_freeze_failed' } })
    expect(frozen.snapshot_id).toMatch(/^snap-[0-9a-f]{32}$/)
  })
})
