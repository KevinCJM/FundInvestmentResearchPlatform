import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ProductCompare from './ProductCompare'
import { evaluateCustomIndicators, getCustomIndicatorMeta, listCustomIndicators } from '../services/customIndicators'
import type { IndicatorDefinition } from '../services/customIndicators'

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="chart" /> }))
vi.mock('../services/customIndicators', () => ({
  evaluateCustomIndicators: vi.fn(),
  getCustomIndicatorMeta: vi.fn(),
  indicatorPeriodLabel: (period: string) => period,
  indicatorsForContext: (items: IndicatorDefinition[], context: string) => items.filter((item) => (item.context_kind ?? 'single_product') === context),
  listCustomIndicators: vi.fn(),
}))

const annualIndicator: IndicatorDefinition = {
  id: 'total-return', revision: 2, source: 'custom', read_only: false,
  name: '区间累计收益', description: '真实净值计算。', expression: '\\left(\\prod\\left(\\mathbf{r}+1\\right)\\right)-1',
  periods: ['1Y'], unit: '%', display_format: 'percent', precision: 2,
  direction: 'higher_better', annual_risk_free_rate_percent: 1.5,
  created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
}

const monthlyIndicator: IndicatorDefinition = {
  ...annualIndicator, id: 'monthly-volatility', name: '月度波动率', periods: ['1M'], display_format: 'number', unit: '', precision: 4,
}

function productResponse(id: string, name: string) {
  return {
    product_id: id, name, management: '测试管理人', status: '上市', base_info: { ts_code: id, fund_type: 'ETF' },
    metrics: { issue_amount: 100, m_fee: 0.5, c_fee: 0.1, exp_return: 8 },
    timeseries: [
      { date: '2026-01-02', close: 3.0 }, { date: '2026-01-05', close: 3.1 }, { date: '2026-01-06', close: 3.2 },
    ],
  }
}

describe('ProductCompare custom indicators', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      const id = String(input).includes('510300.SH') ? '510300.SH' : '159915.SZ'
      return Promise.resolve({ ok: true, json: async () => productResponse(id, id === '510300.SH' ? '沪深300ETF' : '创业板ETF') })
    }))
    vi.mocked(listCustomIndicators).mockResolvedValue({ items: [annualIndicator, monthlyIndicator], total: 2 })
    vi.mocked(getCustomIndicatorMeta).mockResolvedValue({ periods: [{ value: '1M', label: '近 1 月', description: '运行周期' }, { value: '1Y', label: '近 1 年', description: '运行周期' }] } as any)
    vi.mocked(evaluateCustomIndicators).mockResolvedValue({
      results: [
        { indicator_id: annualIndicator.id, indicator_revision: annualIndicator.revision, indicator_name: annualIndicator.name, target: { kind: 'etf', product_id: '510300.SH', name: '沪深300ETF' }, period: '1Y', value: 0.1234, status: 'ok', warnings: [], window: { requested_as_of: null, effective_as_of: '2026-01-06', start_date: '2025-01-06', end_date: '2026-01-06', observation_count: 250, data_latest_date: '2026-01-06' } },
        { indicator_id: annualIndicator.id, indicator_revision: annualIndicator.revision, indicator_name: annualIndicator.name, target: { kind: 'etf', product_id: '159915.SZ', name: '创业板ETF' }, period: '1Y', value: null, status: 'warning', warnings: [{ code: 'INSUFFICIENT_SAMPLE', message: '样本不足' }], window: { requested_as_of: null, effective_as_of: '2026-01-06', start_date: null, end_date: '2026-01-06', observation_count: 10, data_latest_date: '2026-01-06' } },
      ], summary: { total: 2, ok: 1, warning: 1, error: 0 }, cache: { hits: 0, misses: 2 },
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('批量展示动态指标行，并保留部分失败的不可计算原因', async () => {
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/product-compare?kind=etf&ids=510300.SH,159915.SZ']}><ProductCompare /></MemoryRouter>)

    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      '/api/instruments/products/510300.SH?kind=etf',
      expect.objectContaining({ signal: expect.anything() }),
    ))
    expect(fetch).toHaveBeenCalledWith(
      '/api/instruments/products/159915.SZ?kind=etf',
      expect.objectContaining({ signal: expect.anything() }),
    )
    expect(await screen.findByText('12.34%')).toBeInTheDocument()
    expect(screen.getByText('不可计算')).toBeInTheDocument()
    expect(screen.getByText('样本不足')).toBeInTheDocument()
    await waitFor(() => expect(evaluateCustomIndicators).toHaveBeenCalledWith({
      indicator_ids: ['total-return'],
      targets: [{ kind: 'etf', product_id: '510300.SH' }, { kind: 'etf', product_id: '159915.SZ' }],
      period: '1Y',
      as_of: undefined,
    }))
  })

  it('不同指标可独立选择计算区间并按区间拆分引擎请求', async () => {
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/product-compare?kind=etf&ids=510300.SH,159915.SZ']}><ProductCompare /></MemoryRouter>)

    await waitFor(() => expect(evaluateCustomIndicators).toHaveBeenCalledWith(expect.objectContaining({
      indicator_ids: ['total-return'],
      targets: [{ kind: 'etf', product_id: '510300.SH' }, { kind: 'etf', product_id: '159915.SZ' }],
      period: '1Y',
    })))
    vi.mocked(evaluateCustomIndicators).mockClear()

    await user.click(screen.getByRole('button', { name: /选择比较指标/ }))
    await user.click(screen.getByRole('checkbox', { name: /月度波动率/ }))

    await waitFor(() => expect(evaluateCustomIndicators).toHaveBeenCalledWith(expect.objectContaining({
      indicator_ids: ['total-return', 'monthly-volatility'],
      period: '1Y',
    })))
    await user.selectOptions(await screen.findByLabelText('月度波动率计算区间'), '1M')

    await waitFor(() => {
      expect(evaluateCustomIndicators).toHaveBeenCalledWith(expect.objectContaining({
        indicator_ids: ['total-return'],
        period: '1Y',
      }))
      expect(evaluateCustomIndicators).toHaveBeenCalledWith(expect.objectContaining({
        indicator_ids: ['monthly-volatility'],
        period: '1M',
      }))
    })
    expect(screen.getByLabelText('区间累计收益计算区间')).toHaveValue('1Y')
    expect(screen.getByLabelText('月度波动率计算区间')).toHaveValue('1M')
  })

  it('支持从资产大类按每个产品自身类别加载混合对比', async () => {
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      const id = url.includes('024011.OF') ? '024011.OF' : '159393.SZ'
      return Promise.resolve({ ok: true, json: async () => productResponse(id, id) })
    }))

    render(<MemoryRouter initialEntries={['/product-compare?ids=159393.SZ,024011.OF&kinds=etf,fund']}><ProductCompare /></MemoryRouter>)

    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      '/api/instruments/products/159393.SZ?kind=etf',
      expect.objectContaining({ signal: expect.anything() }),
    ))
    expect(fetch).toHaveBeenCalledWith(
      '/api/instruments/products/024011.OF?kind=fund',
      expect.objectContaining({ signal: expect.anything() }),
    )
    await waitFor(() => expect(evaluateCustomIndicators).toHaveBeenCalledWith(expect.objectContaining({
      targets: [
        { kind: 'etf', product_id: '159393.SZ' },
        { kind: 'fund', product_id: '024011.OF' },
      ],
    })))
    expect(listCustomIndicators).toHaveBeenCalledWith({ contextKind: 'single_product' })
  })

  it('从手动构建大类进入后通过返回按钮回到原来源页', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter
        initialEntries={[
          '/manual-construction',
          {
            pathname: '/product-compare',
            search: '?kind=etf&ids=510300.SH,159915.SZ',
            state: { returnTo: '/manual-construction', returnLabel: '返回手动构建大类' },
          },
        ]}
        initialIndex={1}
      >
        <Routes>
          <Route path="/manual-construction" element={<div>手动构建大类来源页</div>} />
          <Route path="/product-compare" element={<ProductCompare />} />
        </Routes>
      </MemoryRouter>,
    )

    await act(async () => {
      await user.click(screen.getByRole('button', { name: /返回手动构建大类/ }))
    })
    expect(await screen.findByText('手动构建大类来源页')).toBeInTheDocument()
  })
})
