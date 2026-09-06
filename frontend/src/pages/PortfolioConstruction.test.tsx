import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import PortfolioConstruction from './PortfolioConstruction'
import { createResearchTarget, runPortfolio } from '../services/portfolioResearch'
import { evaluateNumericControls } from '../services/businessNumeric'
import { getInvestableUniverse, searchInvestableUniverseProducts } from '../services/productPools'
import { listHistoricalRegimeRuns } from '../services/historicalRegimes'

vi.mock('../services/portfolioResearch', () => ({
  createResearchTarget: vi.fn(),
  runPortfolio: vi.fn(),
}))
vi.mock('../services/businessNumeric', () => ({ evaluateNumericControls: vi.fn() }))
vi.mock('../services/productPools', async () => {
  const actual = await vi.importActual<typeof import('../services/productPools')>('../services/productPools')
  return {
    ...actual,
    getInvestableUniverse: vi.fn(),
    searchInvestableUniverseProducts: vi.fn(),
  }
})
vi.mock('../services/historicalRegimes', () => ({ listHistoricalRegimeRuns: vi.fn() }))

describe('PortfolioConstruction', () => {
  beforeEach(() => {
    sessionStorage.clear()
    sessionStorage.setItem('portfolioResearchImport', JSON.stringify({
      name: '权益类产品配置',
      method: 'equal_weight',
      universe_snapshot_id: 'universe-1',
      constituents: [
        { product_id: '510300.SH', kind: 'etf', name: '沪深300ETF', code: '510300.SH', weight: 0, risk_budget: 0, asset_class_id: 'equity', asset_class_name: '权益类' },
      ],
    }))
    vi.mocked(getInvestableUniverse).mockResolvedValue({
      id: 'universe-1', name: '测试可投资域', research_date: '2026-09-04', version_refs: [], members: [],
      summary: { pool_count: 1, member_count: 2, eligible_count: 2, restricted_count: 0, watch_count: 0 },
      content_hash: 'hash', created_at: '2026-09-04T00:00:00Z', immutable: true,
    })
    vi.mocked(searchInvestableUniverseProducts).mockResolvedValue({
      snapshot_id: 'universe-1', snapshot_name: '测试可投资域', research_date: '2026-09-04', total: 2, page: 1, page_size: 20,
      items: [
        { product_id: '510300.SH', code: '510300.SH', kind: 'etf', name: '沪深300ETF', research_status: 'approved', usage_status: 'normal', decision_reasons: ['通过'], max_weight: 0.6, valid_until: null, substitute_groups: [], evaluation_sources: [], eligible: true, eligibility_reasons: [], warnings: [] },
        { product_id: '110011.OF', code: '110011.OF', kind: 'fund', name: '易方达中小盘', research_status: 'approved', usage_status: 'normal', decision_reasons: ['通过'], max_weight: 0.5, valid_until: null, substitute_groups: [], evaluation_sources: [], eligible: true, eligibility_reasons: [], warnings: [] },
      ],
    })
    vi.mocked(evaluateNumericControls).mockImplementation(async (groups) => ({
      items: groups.map((group) => ({ key: group.key, total: 0, difference: -100, within_tolerance: false, positive: false, normalized_shares: group.values.map(() => 0) })),
      execution: { execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0, request_time_compilation: 0 },
    }))
    vi.mocked(createResearchTarget).mockResolvedValue({ id: 'target-1', name: '未命名组合研究', kind: 'portfolio', revision: 1, definition: {} })
    vi.mocked(runPortfolio).mockResolvedValue({ id: 'run-1', target_id: 'target-1', name: '组合', nav: [], drawdown: [], metrics: [{ name: '年化收益', value: .12, unit: 'percent' }], weights: [], contributions: [], warnings: ['[INSUFFICIENT_SAMPLE] 当前样本较短'] })
    vi.mocked(listHistoricalRegimeRuns).mockResolvedValue([{
      id: 'regime-run-1', schema_version: '2.0', definition_id: 'definition-1', definition_revision: 3, definition_snapshot_hash: 'b'.repeat(64), name: '牛熊状态', mode: 'realtime', created_at: '2026-09-04', immutable: true, content_hash: 'a'.repeat(64),
      states: [], series: [], segments: [], conditional_stats: [], transition: { states: [], counts: [], probabilities: [] },
      causality: { classification: 'causal', is_causal: true, uses_future_data: false, repaints: false, realtime_eligible: true, publish_eligible_usages: ['formal_backtest'], blockers: [], warnings: [] },
      stability: {}, walk_forward: {}, governance: { formal_gate_passed: true, publish_eligible_usages: ['formal_backtest'] }, diagnostics: [], publications: [{ id: 'regime-publication-1', usage: 'formal_backtest', published_at: '2026-09-04', definition_revision: 3, run_id: 'regime-run-1', run_content_hash: 'a'.repeat(64), gate: 'comprehensive_formal_gate_passed' }],
    }])
  })

  it('从真实候选池选择产品、保存研究对象并运行快照', async () => {
    render(<MemoryRouter initialEntries={['/pre-investment/product-allocation-timing/construction?universe=universe-1']}><PortfolioConstruction /></MemoryRouter>)
    expect(await screen.findByText(/可投资域：/)).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('从可投资域搜索产品'), { target: { value: '易方达' } })
    await waitFor(() => expect(searchInvestableUniverseProducts).toHaveBeenCalledWith('universe-1', expect.objectContaining({ query: '易方达', eligibleOnly: true })))
    fireEvent.click(screen.getAllByRole('button', { name: '加入' })[1])
    fireEvent.change(await screen.findByLabelText('历史情景条件化（可选）'), { target: { value: 'regime-run-1|regime-publication-1' } })
    fireEvent.click(screen.getByRole('button', { name: '运行组合研究' }))
    await waitFor(() => expect(createResearchTarget).toHaveBeenCalledWith(expect.objectContaining({ kind: 'portfolio', definition: expect.objectContaining({ method: 'equal_weight', universe_snapshot_id: 'universe-1' }) })))
    expect(createResearchTarget).toHaveBeenCalledWith(expect.objectContaining({ definition: expect.objectContaining({ constituents: expect.arrayContaining([expect.objectContaining({ asset_class_id: 'equity', asset_class_name: '权益类' })]) }) }))
    expect(createResearchTarget).toHaveBeenCalledWith(expect.objectContaining({ definition: expect.objectContaining({ rebalance: expect.objectContaining({ transaction_cost_bps: 0 }) }) }))
    expect(runPortfolio).toHaveBeenCalledWith('target-1', { historical_regime: { run_id: 'regime-run-1', publication_id: 'regime-publication-1' } })
    expect(await screen.findByText('运行结果')).toBeInTheDocument()
    expect(screen.getByText(/样本不足：当前样本较短/)).toBeInTheDocument()
    expect(screen.queryByText(/INSUFFICIENT_SAMPLE/)).not.toBeInTheDocument()
    expect(screen.getByText('当前组合研究固定交易成本为 0，仅输出未扣费的毛收益。')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '进入持仓诊断' })).toHaveAttribute('href', '/post-investment/research-diagnosis?target=target-1&run=run-1')
  })
})
