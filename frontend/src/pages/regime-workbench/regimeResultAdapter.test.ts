import { describe, expect, it, vi } from 'vitest'
import { init, setPlatformAPI } from 'echarts'
import { adaptRegimeFormalOverview, adaptRegimeOverview, adaptRegimeResult, loadCompleteRegimeSeries } from './regimeResultAdapter'
import { buildRegimeProbabilityOption, buildRegimeTimelineOption } from './RegimeTimelineChart'
import { buildManualEventLaneOption } from './RegimeManualEventResult'
import { backendOverviewFixture, resultFixture } from './regimeResultFixtures'

describe('完整情景结果契约与图层', () => {
  it('直接接受后端真实serializer响应，不猜测版本字段和区间时点', () => {
    const overview = adaptRegimeOverview(backendOverviewFixture, 'preview-test', 'preview')
    const dates = ['2026-09-04', '2026-09-07', '2026-09-08', '2026-09-09', '2026-09-10', '2026-09-11']
    const states = ['bull', 'bear', 'unclassified', 'unclassified', 'unclassified', 'bull']
    const values = [100, null, null, null, 99, 102]
    const result = adaptRegimeResult(overview, dates.map((date, index) => ({ observation_date: date, state_id: states[index], value: values[index] })))
    expect(result.overview.summary).toMatchObject({ total: 6, classified: 3, unknown: 3, switch_count: 1 })
    expect(result.intervals.map(interval => [interval.start_index, interval.end_index])).toEqual([[0, 0], [1, 1], [2, 4], [5, 5]])
    expect(result.intervals[0]).toMatchObject({ confirmed_at: '2026-09-05', effective_start: '2026-09-07' })
    expect(result.points[1].value).toBeNull()
  })

  it('3000日完整加载，保留第500/501日切换、NaN断点以及末日单日色带', async () => {
    const { overview, rows } = resultFixture()
    const read = vi.fn(async (offset: number, limit: number) => ({ run_id: overview.run_id, items: rows.slice(offset, Math.min(offset + limit, offset + 500)), total: rows.length, offset, limit }))
    const loaded = await loadCompleteRegimeSeries(overview, read, new AbortController().signal)
    expect(loaded).toHaveLength(3000)
    expect(read.mock.calls).toHaveLength(6)
    expect(read.mock.calls.every(call => call[1] === 5000)).toBe(true)
    const result = adaptRegimeResult(adaptRegimeOverview(overview, overview.run_id, 'preview'), loaded)
    expect(result.points[500].value).toBeNull()
    const option = buildRegimeTimelineOption(result)
    expect(option.xAxis).toMatchObject({ type: 'value', min: -0.5, max: 2999.5 })
    expect(option.series).toMatchObject([{
      data: expect.arrayContaining([[500, null], [2999, 3999]]),
      markArea: { data: expect.arrayContaining([
        [{ name: '熊市', xAxis: 499.5, itemStyle: expect.objectContaining({ color: '#22c55e' }) }, { xAxis: 500.5 }],
        [{ name: '熊市', xAxis: 2998.5, itemStyle: expect.objectContaining({ color: '#22c55e' }) }, { xAxis: 2999.5 }],
      ]) },
    }])
    expect(buildRegimeProbabilityOption(result)).toBeNull()
  })

  it('6001行按5000上限读取两页，拒绝空页和总数漂移', async () => {
    const { overview, rows } = resultFixture('large', 6001)
    const read = vi.fn(async (offset: number, limit: number) => ({ run_id: 'large', items: rows.slice(offset, offset + limit), total: rows.length, offset, limit }))
    expect(await loadCompleteRegimeSeries(overview, read, new AbortController().signal)).toHaveLength(6001)
    expect(read.mock.calls.map(call => call[0])).toEqual([0, 5000])
    await expect(loadCompleteRegimeSeries(overview, async (offset, limit) => ({ run_id: 'large', items: [], total: rows.length, offset, limit }), new AbortController().signal)).rejects.toThrow('分页缺失')
    await expect(loadCompleteRegimeSeries(overview, async (offset, limit) => ({ run_id: 'large', items: rows.slice(0, 1), total: 5, offset, limit }), new AbortController().signal)).rejects.toThrow('数量发生变化')
  })

  it('人工事件允许区间重叠，并保留独立事件轨道与重叠摘要', () => {
    const { overview, rows } = resultFixture('manual-events', 10)
    overview.result_kind = 'manual_events'
    overview.manual_events = [
      { id: 'a', label: '事件 A', start_date: rows[2].observation_date, end_date: rows[5].observation_date, color: '#7c3aed', covered_observations: 4, first_observation_index: 2, last_observation_index: 5, first_observation_date: rows[2].observation_date, last_observation_date: rows[5].observation_date },
      { id: 'b', label: '事件 B', start_date: rows[4].observation_date, end_date: rows[7].observation_date, color: '#dc2626', covered_observations: 4, first_observation_index: 4, last_observation_index: 7, first_observation_date: rows[4].observation_date, last_observation_date: rows[7].observation_date },
    ]
    overview.manual_event_summary = { event_count: 2, covered_observations: 6, overlap_observations: 2, max_concurrent_events: 2 }
    const adapted = adaptRegimeOverview(overview, 'manual-events', 'preview')
    expect(adapted.manual_events.map(event => [event.id, event.first_observation_index, event.last_observation_index])).toEqual([['a', 2, 5], ['b', 4, 7]])
    expect(adapted.manual_event_summary).toEqual({ event_count: 2, covered_observations: 6, overlap_observations: 2, max_concurrent_events: 2 })
    const result = adaptRegimeResult(adapted, rows)
    const option = buildManualEventLaneOption(result)
    expect(option.yAxis).toMatchObject({ data: ['事件 A', '事件 B'] })
    expect(option.series).toMatchObject([
      { data: [2, 4] },
      { data: [{ value: 4, itemStyle: { color: '#7c3aed' } }, { value: 4, itemStyle: { color: '#dc2626' } }] },
    ])
  })

  it('正式详情按实际overview/series适配，拒绝错run、局部series和重叠区间', () => {
    const { overview, rows } = resultFixture('formal-1', 600)
    const saved = { ...overview, run_kind: 'saved' }
    const normalized = adaptRegimeFormalOverview({ id: 'formal-1', overview: saved, series: rows }, 'formal-1')
    expect(adaptRegimeResult(normalized, rows).points).toHaveLength(600)
    expect(() => adaptRegimeFormalOverview({ id: 'other', overview: saved }, 'formal-1')).toThrow('身份')
    expect(() => adaptRegimeResult(normalized, rows.slice(0, 500))).toThrow('完整加载')
    expect(() => adaptRegimeOverview({ ...overview, segments: [...overview.segments, overview.segments[0]] }, 'formal-1', 'preview')).toThrow('缺口或重叠')
    expect(() => adaptRegimeOverview({ ...overview, summary: { ...overview.summary, state_counts: { bull: 598, bear: 1 } } }, 'formal-1', 'preview')).toThrow('区间覆盖与统计')
  })

  it('色带使用冻结字典；未分类单独着色；12个状态均不丢失', () => {
    const { overview, rows } = resultFixture('twelve', 12)
    overview.states = rows.map((_, index) => ({ id: 'custom-' + index, label: '状态' + index, color: '#' + (0x110000 + index * 0x1111).toString(16), order: index }))
    overview.segments = rows.map((row, index) => ({ id: 'i' + index, state_id: 'custom-' + index, label: '状态' + index, start_date: row.observation_date, end_date: row.observation_date, start_index: index, end_index: index, observations: 1, confirmed_at: null, effective_start: null }))
    overview.unknown_intervals = []
    overview.summary = { total: 12, classified: 12, unknown: 0, state_counts: Object.fromEntries(overview.states.map(state => [state.id, 1])), switch_count: 11, denominator: 'all_observations' }
    rows.forEach((row, index) => { row.state_id = 'custom-' + index })
    const result = adaptRegimeResult(adaptRegimeOverview(overview, 'twelve', 'preview'), rows)
    expect(result.intervals).toHaveLength(12)
    expect(buildRegimeTimelineOption(result).series).toMatchObject([{ markArea: { data: overview.states.map((state, index) => [{ name: state.label, xAxis: index - .5, itemStyle: expect.objectContaining({ color: state.color }) }, { xAxis: index + .5 }]) } }])
  })

  it('真实ECharts SVG为单日与末日产生非零宽色块，跨周末不生成额外数据', () => {
    const { overview, rows } = resultFixture('svg', 3)
    const dates = ['2026-09-04', '2026-09-07', '2026-09-08']
    rows.forEach((row, index) => { row.observation_date = dates[index]; row.date = dates[index] })
    overview.date_range = { start: dates[0], end: dates[2] }
    for (const segment of [...overview.segments, ...overview.unknown_intervals]) { segment.start_date = dates[segment.start_index]; segment.end_date = dates[segment.end_index] }
    const result = adaptRegimeResult(adaptRegimeOverview(overview, 'svg', 'preview'), rows)
    setPlatformAPI({ measureText: text => ({ width: String(text).length * 7 }) })
    const chart = init(null, undefined, { renderer: 'svg', ssr: true, width: 900, height: 400 })
    try {
      chart.setOption(buildRegimeTimelineOption(result))
      const svg = chart.renderToSVGString()
      for (const color of ['#ef4444', '#22c55e', '#94a3b8']) {
        const tag = svg.match(new RegExp('<polygon[^>]*fill="' + color + '"[^>]*>'))?.[0]
        expect(tag).toBeDefined()
        const coordinates = tag?.match(/points="([^"]+)"/)?.[1].split(/\s+/).map(Number) ?? []
        const xValues = coordinates.filter((_, index) => index % 2 === 0)
        expect(Math.max(...xValues) - Math.min(...xValues)).toBeGreaterThan(0)
      }
      expect(result.points).toHaveLength(3)
    } finally { chart.dispose() }
  })
})
