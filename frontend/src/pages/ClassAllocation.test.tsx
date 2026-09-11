import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ClassAllocation from './ClassAllocation'
import { readAllocationDraft, readAllocationJourney, updateAllocationJourney, writeAllocationDraft } from '../app/allocationJourney'

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="chart" /> }))

const allocation = [
  { name: '权益', etfs: [{ code: '510300.SH', name: '沪深300ETF', weight: 1 }] },
  { name: '固收', etfs: [{ code: '511010.SH', name: '国债ETF', weight: 1 }] },
  { name: '商品', etfs: [{ code: '518880.SH', name: '黄金ETF', weight: 1 }] },
]

const execution = {
  backend: 'numba_njit_fixed_signature',
  execution_backend: 'numba_njit_fixed_signature',
  kernel_version: '2.0.0',
  kernel_coverage: '5/5',
  kernel_signatures: { equal_weights_kernel: ['(int64,) -> tuple'] },
  fingerprint: 'strategy-kernel-fingerprint',
  nopython: true,
  njit_required: true,
  object_mode: 0,
  python_fallback: 0,
  request_time_compilation: 0,
}

const response = (payload: unknown, ok = true, status = ok ? 200 : 400) =>
  Promise.resolve({ ok, status, json: async () => payload } as Response)

function installFetch(
  equalExecution = execution,
  numericResponses: Record<string, unknown> = {},
  historicalRuns: unknown[] = [],
) {
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    if (url === '/api/list-allocations') return response(['测试方案'])
    if (url.startsWith('/api/strategy/default-start')) {
      return response({ default_start: '2024-01-02', count: 30 })
    }
    if (url.startsWith('/api/load-allocation')) return response(allocation)
    if (url === '/api/historical-regimes/runs') return response({ items: historicalRuns })
    if (url === '/api/strategy/equal-weights') {
      expect(init?.method).toBe('POST')
      expect(JSON.parse(String(init?.body))).toEqual({ asset_count: 3, max_leverage: 0 })
      return response({ weights: [33.34, 33.33, 33.33], execution: equalExecution })
    }
    if (Object.prototype.hasOwnProperty.call(numericResponses, url)) {
      return response(numericResponses[url])
    }
    throw new Error(`unexpected fetch: ${url}`)
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

async function selectAllocation(user: ReturnType<typeof userEvent.setup>) {
  await screen.findByRole('button', { name: '选择该方案' })
  await user.click(screen.getByRole('button', { name: '选择该方案' }))
  await screen.findByText('沪深300ETF')
}

async function loadAllocation(user: ReturnType<typeof userEvent.setup>) {
  await selectAllocation(user)
  await user.click(screen.getByRole('button', { name: '+ 添加新的组合策略' }))
}

describe('ClassAllocation fixed-signature equal weights', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    sessionStorage.clear()
    vi.stubGlobal('alert', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('从后端固定签名 NJIT 接口创建三资产等权策略', async () => {
    const fetchMock = installFetch()
    const user = userEvent.setup()
    render(<MemoryRouter><ClassAllocation /></MemoryRouter>)

    await loadAllocation(user)
    await user.click(screen.getByRole('button', { name: '固定比例' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/strategy/equal-weights',
      expect.objectContaining({ method: 'POST' }),
    ))
    expect(await screen.findByDisplayValue('固定比例策略')).toBeInTheDocument()
    expect(screen.getByText('各大类平均分配资金。')).toBeInTheDocument()
    expect(screen.queryByText(/等权来源/)).not.toBeInTheDocument()
    expect(screen.getByDisplayValue('33.34')).toBeDisabled()
    expect(screen.getAllByDisplayValue('33.33')).toHaveLength(2)
  })

  it('执行证明不合规时失败关闭，不在浏览器生成等权结果', async () => {
    installFetch({ ...execution, python_fallback: 1 })
    const user = userEvent.setup()
    render(<MemoryRouter><ClassAllocation /></MemoryRouter>)

    await loadAllocation(user)
    await user.click(screen.getByRole('button', { name: '固定比例' }))

    expect(await screen.findByText('等权计算服务未通过固定签名 NJIT 执行校验')).toBeInTheDocument()
    expect(screen.queryByDisplayValue('固定比例策略')).not.toBeInTheDocument()
  })

  it('锁定所选 SAA 权重进入战术研究，不再丢失预算导入等权产品', async () => {
    updateAllocationJourney({ universeId: 'range-test', name: '期初产品范围', researchDate: '2025-12-31' })
    const fetchMock = installFetch(execution, {
      '/api/tactical-allocation/baselines': { id: 'saa-frozen', name: '已冻结基线' },
    })
    const user = userEvent.setup()
    render(<MemoryRouter><ClassAllocation /></MemoryRouter>)
    await loadAllocation(user)
    fireEvent.change(screen.getByLabelText('研究区间结束'), { target: { value: '2026-09-04' } })
    await user.click(screen.getByRole('button', { name: '固定比例' }))
    await screen.findByDisplayValue('固定比例策略')
    await user.click(screen.getByRole('button', { name: '以此为 SAA，研究战术偏离' }))
    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([url]) => url === '/api/tactical-allocation/baselines')
      expect(call).toBeTruthy()
      const body = JSON.parse(String(call?.[1]?.body))
      expect(body.as_of).toBe('2026-09-04')
      expect(body.weights.权益).toBeCloseTo(.3334, 10)
      expect(body.weights.固收).toBeCloseTo(.3333, 10)
      expect(body.weights.商品).toBeCloseTo(.3333, 10)
      expect(body.constraints.权益).toEqual({ min_weight: 0, max_weight: 1, max_abs_tilt: .1 })
      expect(body.group_limits).toEqual([])
    })
    expect(sessionStorage.getItem('portfolioResearchImport')).toBeNull()
    expect(readAllocationJourney().researchDate).toBe('2025-12-31')
  })

  it('有效前沿响应缺少合规证明时拒绝展示结果', async () => {
    const fetchMock = installFetch(execution, {
      '/api/efficient-frontier': {
        scatter: [],
        frontier: [],
        execution: { ...execution, request_time_compilation: 1 },
      },
    })
    const user = userEvent.setup()
    render(<MemoryRouter><ClassAllocation /></MemoryRouter>)

    await selectAllocation(user)
    await user.click(screen.getByRole('button', { name: '比较收益与风险候选' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/efficient-frontier',
      expect.objectContaining({ method: 'POST' }),
    ))
    expect(await screen.findByRole('alert')).toHaveTextContent('大类配置有效前沿未提供有效的固定签名 NJIT 执行证明')
  })

  it('单点权重响应缺少合规证明时拒绝写入策略', async () => {
    installFetch(execution, {
      '/api/strategy/compute-weights': {
        weights: [.3, .3, .4],
        execution: { ...execution, python_fallback: 1 },
      },
    })
    const user = userEvent.setup()
    render(<MemoryRouter><ClassAllocation /></MemoryRouter>)

    await loadAllocation(user)
    await user.click(screen.getByRole('button', { name: '风险预算' }))
    await user.click(screen.getByRole('button', { name: '反推资金权重' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('大类权重求解未提供有效的固定签名 NJIT 执行证明')
  })

  it('调仓权重响应缺少合规证明时拒绝写入策略', async () => {
    installFetch(execution, {
      '/api/strategy/compute-schedule-weights': {
        dates: ['2024-01-02'],
        weights: [[.3, .3, .4]],
        execution: { ...execution, object_mode: 1 },
      },
    })
    const user = userEvent.setup()
    render(<MemoryRouter><ClassAllocation /></MemoryRouter>)

    await loadAllocation(user)
    await user.click(screen.getByRole('button', { name: '风险预算' }))
    await user.click(screen.getByRole('checkbox', { name: '是否启用再平衡' }))
    await user.click(screen.getByRole('checkbox', { name: '再平衡时是否重新模型计算' }))
    await user.click(screen.getByRole('button', { name: '反推资金权重' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('批量调仓权重计算未提供有效的固定签名 NJIT 执行证明')
  })

  it('回测响应缺少合规证明时拒绝展示结果', async () => {
    const fetchMock = installFetch(execution, {
      '/api/strategy/backtest': {
        dates: ['2024-01-02'],
        series: { 固定比例策略: [1] },
        metrics: [],
        execution: { ...execution, kernel_signatures: {} },
      },
    })
    const user = userEvent.setup()
    render(<MemoryRouter><ClassAllocation /></MemoryRouter>)

    await loadAllocation(user)
    await user.click(screen.getByRole('button', { name: '固定比例' }))
    await screen.findByDisplayValue('固定比例策略')
    await user.click(screen.getByRole('button', { name: '开始策略回测' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/strategy/backtest',
      expect.objectContaining({ method: 'POST' }),
    ))
    expect(await screen.findByRole('alert')).toHaveTextContent('大类配置策略回测未提供有效的固定签名 NJIT 执行证明')
  })

  it('策略回测携带精确的正式历史情景发布引用', async () => {
    const historicalRun = {
      id: 'regime-run-1', schema_version: '2.0', definition_id: 'definition-1', definition_revision: 3,
      definition_snapshot_hash: 'b'.repeat(64), name: '牛熊状态', mode: 'realtime', created_at: '2026-09-04', immutable: true, content_hash: 'a'.repeat(64),
      causality: { is_causal: true, uses_future_data: false, repaints: false, realtime_eligible: true },
      governance: { formal_gate_passed: true, publish_eligible_usages: ['formal_backtest'] },
      publications: [{ id: 'regime-publication-1', usage: 'formal_backtest', published_at: '2026-09-04', definition_revision: 3, run_id: 'regime-run-1', run_content_hash: 'a'.repeat(64), gate: 'comprehensive_formal_gate_passed' }],
    }
    const fetchMock = installFetch(execution, {
      '/api/strategy/backtest': { dates: ['2024-01-02'], series: { 固定比例策略: [1] }, metrics: [], execution },
    }, [historicalRun])
    const user = userEvent.setup()
    render(<MemoryRouter><ClassAllocation /></MemoryRouter>)

    await loadAllocation(user)
    await user.click(screen.getByRole('button', { name: '固定比例' }))
    await screen.findByDisplayValue('固定比例策略')
    await user.selectOptions(await screen.findByLabelText('历史情景条件化（可选）'), 'regime-run-1|regime-publication-1')
    await user.click(screen.getByRole('button', { name: '开始策略回测' }))

    await waitFor(() => {
      const request = fetchMock.mock.calls.find(([url]) => url === '/api/strategy/backtest')
      expect(JSON.parse(String(request?.[1]?.body)).historical_regime).toEqual({ run_id: 'regime-run-1', publication_id: 'regime-publication-1' })
    })
  })
})

describe('ClassAllocation continuous research journey', () => {
  beforeEach(() => { vi.clearAllMocks(); localStorage.clear(); sessionStorage.clear() })
  afterEach(() => { vi.unstubAllGlobals() })

  it('从大类 URL 自动加载并恢复输入，计算结果不跨数据口径复用', async () => {
    installFetch()
    const user = userEvent.setup()
    const route = '/pre-investment/saa/allocation-lab?alloc=测试方案&universe=universe-1'
    const view = render(<MemoryRouter initialEntries={[route]}><ClassAllocation /></MemoryRouter>)
    await screen.findByText('沪深300ETF')
    await user.click(screen.getByRole('button', { name: '填写一组长期权重' }))
    fireEvent.change(await screen.findByLabelText('固定比例策略 权益 权重 (%)'), { target: { value: '60' } })
    fireEvent.change(screen.getByLabelText('固定比例策略 固收 权重 (%)'), { target: { value: '30' } })
    fireEvent.change(screen.getByLabelText('固定比例策略 商品 权重 (%)'), { target: { value: '10' } })
    fireEvent.change(screen.getByLabelText('权益 最高权重 (%)'), { target: { value: '70' } })
    fireEvent.change(screen.getByLabelText('研究区间开始'), { target: { value: '2024-02-01' } })
    await waitFor(() => expect(readAllocationDraft<any>('saa:universe-1:测试方案')?.strategies[0].rows[0].weight).toBe(60))
    expect(readAllocationJourney().allocationName).toBe('测试方案')
    view.unmount()
    render(<MemoryRouter initialEntries={[route]}><ClassAllocation /></MemoryRouter>)
    await screen.findByText('沪深300ETF')
    expect(await screen.findByLabelText('固定比例策略 权益 权重 (%)')).toHaveValue(60)
    expect(screen.getByLabelText('权益 最高权重 (%)')).toHaveValue(70)
    expect(screen.getByLabelText('回测开始日期')).toHaveValue('2024-02-01')
    expect(screen.getByRole('status')).toHaveTextContent('当前数据口径下重新计算')
    expect(screen.queryByTestId('chart')).not.toBeInTheDocument()
  })

  it('百分比边界保持后端小数契约，采用真实候选并完整传给 TAA', async () => {
    const fetchMock = installFetch(execution, {
      '/api/efficient-frontier': {
        asset_names: ['权益', '固收', '商品'], scatter: [], frontier: [],
        min_variance: { value: [.04, .08], weights: [.2, .7, .1] },
        max_sharpe: { value: [.08, .12], weights: [.6, .3, .1] },
        max_return: { value: [.12, .2], weights: [.8, .1, .1] }, execution,
      },
      '/api/tactical-allocation/baselines': { id: 'saa-selected' },
    })
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/pre-investment/saa/allocation-lab?alloc=测试方案&universe=universe-1']}><ClassAllocation /></MemoryRouter>)
    await screen.findByText('沪深300ETF')
    fireEvent.change(screen.getByLabelText('权益 最低权重 (%)'), { target: { value: '10' } })
    fireEvent.change(screen.getByLabelText('权益 最高权重 (%)'), { target: { value: '80' } })
    await user.click(screen.getByRole('button', { name: '+ 添加联合约束' }))
    const group = screen.getByRole('heading', { name: '多个大类合计范围' }).parentElement!
    await user.click(within(group).getByRole('checkbox', { name: '权益' }))
    await user.click(within(group).getByRole('checkbox', { name: '商品' }))
    fireEvent.change(screen.getByLabelText('联合约束 1 最高权重 (%)'), { target: { value: '90' } })
    await user.click(screen.getByRole('button', { name: '比较收益与风险候选' }))
    expect(await screen.findByRole('table', { name: '长期配置候选' })).toHaveTextContent('8.00%')
    const payload = JSON.parse(String(fetchMock.mock.calls.find(([url]) => url === '/api/efficient-frontier')?.[1]?.body))
    expect(payload.constraints.single_limits.权益).toEqual({ lo: .1, hi: .8 })
    expect(payload.constraints.group_limits[0]).toEqual({ assets: ['权益', '商品'], lo: 0, hi: .9 })
    await user.click(screen.getByRole('button', { name: '采用较高收益风险比' }))
    expect(screen.getByLabelText('较高收益风险比配置 权益 权重 (%)')).toHaveValue(60)
    await user.click(screen.getByRole('button', { name: '以此为 SAA，研究战术偏离' }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => url === '/api/tactical-allocation/baselines')).toBe(true))
    const baseline = JSON.parse(String(fetchMock.mock.calls.find(([url]) => url === '/api/tactical-allocation/baselines')?.[1]?.body))
    expect(baseline.weights).toEqual({ 权益: .6, 固收: .3, 商品: .1 })
    expect(baseline.constraints.权益).toEqual({ min_weight: .1, max_weight: .8, max_abs_tilt: .1 })
    expect(baseline.group_limits[0]).toMatchObject({ assets: ['权益', '商品'], lo: 0, hi: .9 })
  })

  it('前沿候选旁边要印出它是按哪天、哪个产品域算出来的', async () => {
    installFetch(execution, {
      '/api/efficient-frontier': {
        asset_names: ['权益', '固收', '商品'],
        max_sharpe: { value: [.08, .12], weights: [.6, .3, .1] },
        execution,
        pit: {
          alloc_name: '测试方案', as_of: '2024-02-20', run_mode: 'RESEARCH',
          series_as_of: '2023-12-01', series_variants: ['2023-12-01'],
          hindsight_series: false, rows_dropped_by_as_of: 0,
          availability_available: true, warnings: [],
          universe: {
            source: 'asset_nv', replayable: false, established_at: '2023-12-01',
            snapshot_id: 'universe-1', snapshot_established_at: '2026-01-01', clean: false,
            findings: [{
              code: 'UNIVERSE_LOOKAHEAD', label: '可投资域「2026筛出来的池子」',
              message: '可投资域「2026筛出来的池子」是用截至 2026-01-01 的数据筛出来的，却被用于 2024-02-20 的决策——该决策带入了未来信息。',
            }],
          },
        },
      },
    })
    const user = userEvent.setup()
    render(<MemoryRouter><ClassAllocation /></MemoryRouter>)

    await selectAllocation(user)
    await user.click(screen.getByRole('button', { name: '比较收益与风险候选' }))
    await screen.findByRole('table', { name: '长期配置候选' })

    // 挑权重的就是这张表，口径必须跟它在同一块，不能只在回测那边有。
    expect(screen.getByTestId('pit-decision-notice')).toHaveTextContent('产品域研究日 2026-01-01')
    expect(screen.getByTestId('pit-universe-finding')).toHaveTextContent('带入了未来信息')
  })

  it('改研究条件清掉前沿，切换自定义不会把等权重置成 100/0', async () => {
    installFetch(execution, { '/api/efficient-frontier': { asset_names: ['权益', '固收', '商品'], max_sharpe: { value: [.08, .12], weights: [.6, .3, .1] }, execution } })
    writeAllocationDraft('saa:another-universe:测试方案', { assetNames: ['权益', '固收', '商品'], strategies: [{ id: 'old', type: 'fixed', name: '其他研究', cfg: { mode: 'custom' }, rows: [{ className: '权益', weight: 90 }] }] })
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/pre-investment/saa/allocation-lab?alloc=测试方案&universe=universe-1']}><ClassAllocation /></MemoryRouter>)
    await screen.findByText('沪深300ETF')
    expect(screen.queryByDisplayValue('其他研究')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '比较收益与风险候选' }))
    await screen.findByRole('table', { name: '长期配置候选' })
    fireEvent.change(screen.getByLabelText('研究区间结束'), { target: { value: '2025-01-01' } })
    await waitFor(() => expect(screen.queryByRole('table', { name: '长期配置候选' })).not.toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: '+ 添加新的组合策略' }))
    await user.click(screen.getByRole('button', { name: '固定比例' }))
    await screen.findByDisplayValue('固定比例策略')
    await user.click(screen.getByRole('radio', { name: '自定义权重' }))
    expect(screen.getByLabelText('固定比例策略 权益 权重 (%)')).toHaveValue(33.34)
    expect(screen.getByLabelText('固定比例策略 权益 权重 (%)')).not.toBeDisabled()
  })
})
