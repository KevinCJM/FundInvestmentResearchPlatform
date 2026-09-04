import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ClassAllocation from './ClassAllocation'

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
    expect(screen.getByText('等权来源：Numba NJIT 固定签名 · 5/5')).toBeInTheDocument()
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
    await user.click(screen.getByRole('button', { name: '计算可配置空间与有效前沿' }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      '/api/efficient-frontier',
      expect.objectContaining({ method: 'POST' }),
    ))
    await waitFor(() => expect(alert).toHaveBeenCalledWith(
      '大类配置有效前沿未提供有效的固定签名 NJIT 执行证明',
    ))
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

    await waitFor(() => expect(alert).toHaveBeenCalledWith(
      '大类权重求解未提供有效的固定签名 NJIT 执行证明',
    ))
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

    await waitFor(() => expect(alert).toHaveBeenCalledWith(
      '批量调仓权重计算未提供有效的固定签名 NJIT 执行证明',
    ))
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
    await waitFor(() => expect(alert).toHaveBeenCalledWith(
      '大类配置策略回测未提供有效的固定签名 NJIT 执行证明',
    ))
  })

  it('策略回测携带精确的正式历史情景发布引用', async () => {
    const historicalRun = {
      id: 'regime-run-1', schema_version: '2.0', definition_id: 'definition-1', definition_revision: 3,
      definition_snapshot_hash: 'b'.repeat(64), name: '牛熊状态', mode: 'realtime', created_at: '2026-09-04', immutable: true, content_hash: 'a'.repeat(64),
      causality: { is_causal: true, realtime_eligible: true },
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
