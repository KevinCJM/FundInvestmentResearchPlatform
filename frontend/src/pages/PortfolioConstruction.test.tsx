import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import PortfolioConstruction from './PortfolioConstruction'
import { createResearchTarget, runPortfolio, searchPortfolioInstruments } from '../services/portfolioResearch'

vi.mock('../services/portfolioResearch', () => ({
  searchPortfolioInstruments: vi.fn(),
  createResearchTarget: vi.fn(),
  runPortfolio: vi.fn(),
}))

describe('PortfolioConstruction', () => {
  beforeEach(() => {
    vi.mocked(searchPortfolioInstruments).mockResolvedValue({ total: 2, items: [
      { product_id: '510300.SH', kind: 'etf', name: '沪深300ETF', code: '510300.SH' },
      { product_id: '110011.OF', kind: 'fund', name: '易方达中小盘', code: '110011.OF' },
    ] })
    vi.mocked(createResearchTarget).mockResolvedValue({ id: 'target-1', name: '未命名组合研究', kind: 'portfolio', revision: 1, definition: {} })
    vi.mocked(runPortfolio).mockResolvedValue({ id: 'run-1', target_id: 'target-1', name: '组合', nav: [], drawdown: [], metrics: [{ name: '年化收益', value: .12, unit: 'percent' }], weights: [], contributions: [], warnings: ['[INSUFFICIENT_SAMPLE] 当前样本较短'] })
  })

  it('从真实候选池选择产品、保存研究对象并运行快照', async () => {
    render(<MemoryRouter><PortfolioConstruction /></MemoryRouter>)
    fireEvent.change(screen.getByLabelText('搜索 ETF 或基金'), { target: { value: '沪深' } })
    await waitFor(() => expect(searchPortfolioInstruments).toHaveBeenCalledWith('沪深', expect.any(AbortSignal)))
    fireEvent.click(screen.getAllByRole('button', { name: '加入' })[0])
    fireEvent.click(screen.getAllByRole('button', { name: '加入' })[1])
    fireEvent.click(screen.getByRole('button', { name: '运行组合研究' }))
    await waitFor(() => expect(createResearchTarget).toHaveBeenCalledWith(expect.objectContaining({ kind: 'portfolio', definition: expect.objectContaining({ method: 'equal_weight' }) })))
    expect(createResearchTarget).toHaveBeenCalledWith(expect.objectContaining({ definition: expect.objectContaining({ rebalance: expect.objectContaining({ transaction_cost_bps: 0 }) }) }))
    expect(runPortfolio).toHaveBeenCalledWith('target-1')
    expect(await screen.findByText('运行结果')).toBeInTheDocument()
    expect(screen.getByText(/样本不足：当前样本较短/)).toBeInTheDocument()
    expect(screen.queryByText(/INSUFFICIENT_SAMPLE/)).not.toBeInTheDocument()
    expect(screen.getByText('当前组合研究固定交易成本为 0，仅输出未扣费的毛收益。')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '进入持仓诊断' })).toHaveAttribute('href', '/holding-diagnosis?target=target-1&run=run-1')
  })
})
