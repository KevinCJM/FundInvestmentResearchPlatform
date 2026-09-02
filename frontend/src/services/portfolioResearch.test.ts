import { afterEach, describe, expect, it, vi } from 'vitest'
import { diagnosePortfolioRun, getPortfolioRun, listPortfolioIndicators, runPortfolioScenario } from './portfolioResearch'

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
      }))
      .mockResolvedValueOnce(response({ name: '压力期', metrics: [{ name: '区间收益', value: -.1, unit: 'percent' }], warnings: [{ code: 'GAP', message: '区间存在缺口' }] }))
    vi.stubGlobal('fetch', mockedFetch)
    const diagnosis = await diagnosePortfolioRun('run-1')
    const scenario = await runPortfolioScenario('run-1', { name: '压力期', start_date: '2020-01-01', end_date: '2020-03-31' })
    expect(diagnosis.warnings).toEqual(['缺少部分历史数据'])
    expect(diagnosis.weight_path).toEqual([{ date: '2024-01-02', weights: { '510300.SH': .6 } }])
    expect(scenario.warnings).toEqual(['区间存在缺口'])
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
})
