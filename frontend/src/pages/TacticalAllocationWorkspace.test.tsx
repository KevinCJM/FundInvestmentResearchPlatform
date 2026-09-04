import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import TacticalAllocationWorkspace from './TacticalAllocationWorkspace'

vi.mock('echarts-for-react', () => ({
  default: ({ option }: { option: { series: Array<{ name: string }> } }) => <div data-testid="taa-chart">{option.series.map((item) => item.name).join('、')}</div>,
}))

const fixedExecution = {
  engine: 'numba_njit_fixed_signature',
  backend: 'numba_njit_fixed_signature',
  execution_backend: 'numba_njit_fixed_signature',
  typed_indicator_dag: true,
  nopython: true,
  object_mode: 0,
  python_fallback: 0,
  request_time_compilation: 0,
  kernel_signatures: { taa_backtest_kernel: ['fixed'] },
}

const run = {
  id: 'RUN-REGIME-1',
  definition_id: 'REGIME-1',
  definition_revision: 3,
  name: '沪深300 牛熊状态',
  mode: 'realtime',
  created_at: '2026-09-03T00:00:00Z',
  immutable: true,
  content_hash: 'run-hash',
  states: [
    { id: 'bull', label: '牛市', color: '#16a34a' },
    { id: 'bear', label: '熊市', color: '#dc2626' },
  ],
  series: [],
  segments: [],
  conditional_stats: [],
  transition: { states: ['bull', 'bear'], counts: [[1, 1], [1, 1]], probabilities: [[0.5, 0.5], [0.5, 0.5]] },
  causality: { classification: 'causal', is_causal: true, uses_future_data: false, repaints: false, realtime_eligible: true, publish_eligible_usages: ['taa'], blockers: [], warnings: [] },
  stability: {},
  walk_forward: {},
  diagnostics: [],
  calculation_audits: [{ source_kind: 'historical_regime_algorithm', ...fixedExecution }],
  publications: [{ id: 'PUB-1', usage: 'taa', published_at: '2026-09-03', definition_revision: 3, run_id: 'RUN-REGIME-1', run_content_hash: 'run-hash', gate: 'causality_passed' }],
}

const result = {
  schema_version: '1.0',
  run_id: 'RUN-REGIME-1',
  definition_id: 'REGIME-1',
  definition_revision: 3,
  execution: fixedExecution,
  gate: { passed: true, immutable: true, mode: 'realtime', causal: true, publication_usages: ['taa'], publication_ids: ['PUB-1'], run_content_hash: 'run-hash' },
  timing_policy: { return_timestamp: 'period_end', signal_rule: 'regime.effective_date <= asset_return.period_start', same_day_signal_allowed: false, allocation_source: 'probability_weighted_state_tilts', period_start_policy: 'previous period end', max_signal_age_days: 31 },
  input_snapshot: { observations: 10, start_date: '2024-01-02', end_date: '2024-01-15', assets: ['中债综合', '沪深300'], regime_run_content_hash: 'run-hash', asset_returns_hash: 'returns-hash', parameters_hash: 'params-hash' },
  baseline: { nav: [{ date: '2024-01-02', value: 1.001 }], metrics: { total_return: 0.01, annualized_return: 0.1, annualized_volatility: 0.08, sharpe: 1.25, max_drawdown: -0.02 }, policy: 'daily_constant_weight_no_cost' },
  taa: { nav: [{ date: '2024-01-02', value: 1.002 }], metrics: { total_return: 0.02, annualized_return: 0.2, annualized_volatility: 0.09, sharpe: 2.22, max_drawdown: -0.018 }, policy: 'daily_target_weight_with_pretrade_cost' },
  weights: [{ date: '2024-01-02', regime_observation_date: '2024-01-01', regime_effective_date: '2024-01-01', probabilities: { bull: 0.7, bear: 0.3 }, allocation_probabilities: { bull: 0.7, bear: 0.3 }, confidence: 0.7, fallback_to_base: false, fallback_reason: null, tilt_scale: 1, weights: { '中债综合': 0.36, '沪深300': 0.64 }, turnover: 0.04, transaction_cost_amount: 0.00002, baseline_return: 0.001, gross_taa_return: 0.002, net_taa_return: 0.00198 }],
  turnover_and_cost: { total_turnover: 0.04, average_turnover: 0.004, total_transaction_cost: 0.00002 },
  excess: { total_return_difference: 0.01, relative_total_return: 0.0099, gross_active_return_sum: 0.011 },
  state_contributions: [{ state_id: 'bull', probability_weight: 7, gross_excess_return_contribution: 0.012, active_periods: 10 }, { state_id: 'bear', probability_weight: 3, gross_excess_return_contribution: -0.001, active_periods: 10 }],
  fallbacks: { periods: 0, reasons: {} },
  snapshot_hash: 'snapshot-hash-abcdef',
}

function ok(body: unknown) {
  return { ok: true, status: 200, json: async () => body } as Response
}

describe('TacticalAllocationWorkspace', () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

  it('只加载通过发布与因果门禁的版本，并执行真实 TAA 回测', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input)
      if (path === '/api/historical-regimes/runs') return ok({ items: [run, { ...run, id: 'RETRO', mode: 'retrospective', publications: [] }] })
      if (path.endsWith('/RUN-REGIME-1/taa-backtest') && init?.method === 'POST') return ok(result)
      throw new Error(`unexpected ${path}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<MemoryRouter><TacticalAllocationWorkspace /></MemoryRouter>)

    expect(await screen.findByRole('heading', { name: '战术资产配置工作台' })).toBeInTheDocument()
    expect(await screen.findByRole('option', { name: /沪深300 牛熊状态 · R3/ })).toBeInTheDocument()
    expect(screen.queryByRole('option', { name: /RETRO/ })).not.toBeInTheDocument()
    expect((await screen.findByLabelText('状态权重偏移') as HTMLTextAreaElement).value).toContain('bull')

    await act(async () => { await user.click(screen.getByRole('button', { name: '运行 TAA 回测' })) })

    expect(await screen.findByText('SAA 与情景驱动 TAA 对照')).toBeInTheDocument()
    expect(screen.getByText('2.00%')).toBeInTheDocument()
    expect(screen.getByTestId('taa-chart')).toHaveTextContent('SAA 基准、情景驱动 TAA')
    const call = fetchMock.mock.calls.find(([path]) => String(path).endsWith('/taa-backtest'))
    const payload = JSON.parse(String(call?.[1]?.body))
    expect(payload.base_weights).toEqual({ '沪深300': 0.6, '中债综合': 0.4 })
    expect(payload.state_tilts.bull).toEqual({ '沪深300': 0, '中债综合': 0 })

    await act(async () => { await user.click(screen.getByRole('tab', { name: '时序与审计' })) })
    const timingCard = (await screen.findByText('严格时序')).parentElement
    expect(timingCard).toHaveTextContent('regime.effective_date <= asset_return.period_start')
    expect(screen.getByText(/returns-hash/)).toBeInTheDocument()
  })

  it('没有合格的发布版本时阻止运行并解释门禁', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ok({ items: [{ ...run, mode: 'retrospective', publications: [] }] })))
    render(<MemoryRouter><TacticalAllocationWorkspace /></MemoryRouter>)
    expect(await screen.findByText(/暂无可用版本/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '运行 TAA 回测' })).toBeDisabled()
  })

  it('轻量摘要缺少状态字典时只按需读取所选运行详情', async () => {
    const summary = { ...run, states: undefined, series: undefined, segments: undefined, series_included: false, series_detail_endpoint: `/api/historical-regimes/runs/${run.id}` }
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input)
      if (path === '/api/historical-regimes/runs') return ok({ items: [summary] })
      if (path === `/api/historical-regimes/runs/${run.id}`) return ok(run)
      throw new Error(`unexpected ${path}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<MemoryRouter><TacticalAllocationWorkspace /></MemoryRouter>)

    expect((await screen.findByLabelText('状态权重偏移') as HTMLTextAreaElement).value).toContain('bull')
    expect(fetchMock.mock.calls.filter(([path]) => String(path) === `/api/historical-regimes/runs/${run.id}`)).toHaveLength(1)
  })

  it('参数变化和失败运行不会继续展示旧结果', async () => {
    let backtestCalls = 0
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input)
      if (path === '/api/historical-regimes/runs') return ok({ items: [run] })
      backtestCalls += 1
      if (backtestCalls === 1) return ok(result)
      return { ok: false, status: 422, json: async () => ({ detail: { message: '成本参数不合法。' } }) } as Response
    }))
    const user = userEvent.setup()
    render(<MemoryRouter><TacticalAllocationWorkspace /></MemoryRouter>)
    await screen.findByRole('option', { name: /沪深300 牛熊状态/ })

    await act(async () => { await user.click(screen.getByRole('button', { name: '运行 TAA 回测' })) })
    expect(await screen.findByText('SAA 与情景驱动 TAA 对照')).toBeInTheDocument()

    await act(async () => {
      await user.clear(screen.getByLabelText('单边交易成本'))
      await user.type(screen.getByLabelText('单边交易成本'), '10')
    })
    expect(screen.queryByText('SAA 与情景驱动 TAA 对照')).not.toBeInTheDocument()

    await act(async () => { await user.click(screen.getByRole('button', { name: '运行 TAA 回测' })) })
    expect(await screen.findByRole('alert')).toHaveTextContent('成本参数不合法')
    expect(screen.queryByText('SAA 与情景驱动 TAA 对照')).not.toBeInTheDocument()
  })

  it('执行证明不合规时失败关闭，不展示回测数值', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input)
      if (path === '/api/historical-regimes/runs') return ok({ items: [run] })
      return ok({ ...result, execution: { ...fixedExecution, object_mode: 1 } })
    }))
    const user = userEvent.setup()
    render(<MemoryRouter><TacticalAllocationWorkspace /></MemoryRouter>)
    await screen.findByRole('option', { name: /沪深300 牛熊状态/ })

    await act(async () => { await user.click(screen.getByRole('button', { name: '运行 TAA 回测' })) })
    expect(await screen.findByRole('alert')).toHaveTextContent('TAA 回测未提供有效的固定签名 NJIT 执行证明')
    expect(screen.queryByText('SAA 与情景驱动 TAA 对照')).not.toBeInTheDocument()
  })
})
