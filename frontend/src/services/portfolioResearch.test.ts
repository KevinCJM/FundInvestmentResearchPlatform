import { afterEach, describe, expect, it, vi } from 'vitest'
import { diagnosePortfolioRun, getPortfolioRun, listPortfolioIndicators, runPortfolio, runPortfolioScenario } from './portfolioResearch'

const fixedExecution = {
  execution_backend: 'numba_njit_fixed_signature',
  nopython: true,
  object_mode: 0,
  python_fallback: 0,
  request_time_compilation: 0,
  kernel_signatures: { portfolio_summary_kernel: ['(Array(float64, 1, C, False, aligned=True), float64, float64)'] },
}

function response(body: unknown) {
  return { ok: true, status: 200, json: vi.fn().mockResolvedValue(body) } as unknown as Response
}

describe('portfolioResearch canonical response adapters', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('uses effective_date, maps array weights by asset key, and exposes warning messages', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response({
      id: 'run-1', name: '组合', assets: [{ asset_key: '510300.SH' }, { asset_key: '110011.OF' }],
      dates: ['2024-01-02'], portfolio_nav: [1.02], drawdown: [-0.01],
      weight_path: [{ decision_date: '2024-01-01', effective_date: '2024-01-02', weights: [.6, .4] }],
      summary: { annual_return: .12 }, warnings: [{ code: 'SHORT_SAMPLE', message: '样本长度不足' }],
      execution: fixedExecution,
    })))
    const run = await getPortfolioRun('run-1')
    expect(run.weights).toEqual([{ date: '2024-01-02', weights: { '510300.SH': .6, '110011.OF': .4 } }])
    expect(run.warnings).toEqual(['样本长度不足'])
    expect(run.nav).toEqual([{ date: '2024-01-02', value: 1.02 }])
  })

  it('normalizes diagnosis and scenario warning messages without leaking objects to the UI', async () => {
    const mockedFetch = vi.fn()
      .mockResolvedValueOnce(response({
        summary: {}, components: [{ product_id: '510300.SH', name: '沪深300ETF', current_weight: .6 }],
        contribution_series: [{ date: '2024-01-02', values: { '510300.SH': .01 } }], concentration: [],
        weight_path: [{ date: '2024-01-02', weights: { '510300.SH': .6 } }], warnings: [{ code: 'NO_DATA', message: '缺少部分历史数据' }],
        execution: fixedExecution,
      }))
      .mockResolvedValueOnce(response({ name: '压力期', metrics: [{ name: '区间收益', value: -.1, unit: 'percent' }], warnings: [{ code: 'GAP', message: '区间存在缺口' }], execution: fixedExecution }))
    vi.stubGlobal('fetch', mockedFetch)
    const diagnosis = await diagnosePortfolioRun('run-1')
    const scenario = await runPortfolioScenario('run-1', { name: '压力期', start_date: '2020-01-01', end_date: '2020-03-31' })
    expect(diagnosis.warnings).toEqual(['缺少部分历史数据'])
    expect(diagnosis.weight_path).toEqual([{ date: '2024-01-02', weights: { '510300.SH': .6 } }])
    expect(scenario.warnings).toEqual(['区间存在缺口'])
  })

  it('fails closed when a portfolio calculation omits fixed-signature NJIT evidence', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response({
      id: 'run-unsafe', name: '未审计组合', dates: [], portfolio_nav: [], drawdown: [],
    })))

    await expect(getPortfolioRun('run-unsafe')).rejects.toThrow('组合研究运行未提供有效的固定签名 NJIT 执行证明')
  })

  it('only exposes saved portfolio-context indicators to holding diagnosis', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response({ items: [
      { id: 'product-return', name: '产品收益', revision: 1, context_kind: 'single_product' },
      { id: 'portfolio-volatility', name: '组合波动率', revision: 2, context_kind: 'portfolio' },
    ] })))
    await expect(listPortfolioIndicators()).resolves.toEqual([
      { id: 'portfolio-volatility', name: '组合波动率', revision: 2, context_kind: 'portfolio' },
    ])
  })

  it('运行组合时传递精确历史情景发布并保留条件表现', async () => {
    const fetchMock = vi.fn().mockResolvedValue(response({
      id: 'run-1', name: '组合', dates: ['2024-01-03'], portfolio_nav: [1.01], drawdown: [0], summary: {}, warnings: [],
      execution: fixedExecution,
      regime_conditioning: {
        binding: { run_id: 'regime-run-1', publication_id: 'regime-publication-1' },
        coverage: { periods: 1, classified_periods: 1, classified_ratio: 1 },
        conditional_performance: { 组合: [{ state_id: 'bull', state_label: '牛市', observations: 1, return_observations: 1 }] },
        period_states: [],
        execution: fixedExecution,
      },
    }))
    vi.stubGlobal('fetch', fetchMock)

    const run = await runPortfolio('target-1', {
      historical_regime: { run_id: 'regime-run-1', publication_id: 'regime-publication-1' },
    })

    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      historical_regime: { run_id: 'regime-run-1', publication_id: 'regime-publication-1' },
    })
    expect(run.regime_conditioning?.binding.publication_id).toBe('regime-publication-1')
    expect(run.regime_conditioning?.conditional_performance.组合[0].state_label).toBe('牛市')
  })
})
