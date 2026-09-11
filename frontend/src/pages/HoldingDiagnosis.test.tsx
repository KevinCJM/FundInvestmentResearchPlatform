import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import HoldingDiagnosis from './HoldingDiagnosis'
import { diagnosePortfolioRun, downloadPortfolioExport, getPortfolioRun, getResearchTarget, listPortfolioIndicators, runPortfolioScenario } from '../services/portfolioResearch'

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="chart" /> }))
vi.mock('../services/portfolioResearch', () => ({
  getResearchTarget: vi.fn(), getPortfolioRun: vi.fn(), diagnosePortfolioRun: vi.fn(),
  listPortfolioRuns: vi.fn(), listPortfolioIndicators: vi.fn(), downloadPortfolioExport: vi.fn(), runPortfolioScenario: vi.fn(),
}))

describe('HoldingDiagnosis', () => {
  beforeEach(() => {
    vi.mocked(getResearchTarget).mockResolvedValue({ id: 'target-1', name: '稳健组合', kind: 'portfolio', revision: 2, definition: {} })
    vi.mocked(getPortfolioRun).mockResolvedValue({ id: 'run-1', target_id: 'target-1', name: '稳健组合', nav: [{ date: '2024-01-01', value: 1 }], drawdown: [{ date: '2024-01-01', value: 0 }], metrics: [], weights: [], contributions: [], warnings: [], regime_conditioning: {
      binding: { run_id: 'regime-run-1', publication_id: 'regime-publication-1', definition_id: 'definition-1', definition_revision: 3 },
      coverage: { periods: 20, classified_periods: 18, classified_ratio: .9 },
      conditional_performance: { 稳健组合: [{ state_id: 'bull', state_label: '牛市', observations: 10, return_observations: 9, annualized_return: .12 }] },
      period_states: [],
    } })
    vi.mocked(listPortfolioIndicators).mockResolvedValue([{
      id: 'portfolio-volatility', name: '组合波动率', revision: 2, context_kind: 'portfolio',
      source: 'custom', read_only: false, created_at: '', updated_at: '',
      description: '组合收益率标准差', expression: 'std(portfolio_returns, 1)',
      result_kind: 'scalar', output_contract: 'scalar', display_format: 'percent',
      precision: 2, unit: '%', direction: 'lower_better', annual_risk_free_rate_percent: 0,
    }])
    vi.mocked(diagnosePortfolioRun)
      .mockResolvedValueOnce({ summary: [{ name: '年化收益', value: .1, unit: 'percent' }], components: [], contributions: [{ product_id: '510300.SH', name: '沪深300ETF', contribution: .05, risk_contribution: .4 }], concentration: [{ name: 'HHI', value: .5 }], covariance: null, correlation: null, weight_path: [], warnings: ['[INSUFFICIENT_SAMPLE] 样本窗口较短'] })
      .mockResolvedValueOnce({ summary: [{ name: '年化收益', value: .1, unit: 'percent' }], components: [], contributions: [], concentration: [], covariance: null, correlation: null, weight_path: [], custom_indicators: [{ name: '组合波动率', value: null, status: 'warning', warnings: ['[INSUFFICIENT_COMMON_SAMPLE] 共同样本不足'] }], warnings: [] })
    vi.mocked(runPortfolioScenario).mockResolvedValue({ name: '历史压力区间', metrics: [{ name: '区间收益', value: -.08, unit: 'percent' }], warnings: [] })
  })

  it('以不可变快照加载诊断、情景和导出能力', async () => {
    render(<MemoryRouter initialEntries={['/holding-diagnosis?target=target-1&run=run-1']}><HoldingDiagnosis /></MemoryRouter>)
    expect(await screen.findByText('持仓诊断：稳健组合')).toBeInTheDocument()
    expect(screen.getByText(/样本不足：样本窗口较短/)).toBeInTheDocument()
    expect(screen.queryByText(/INSUFFICIENT_SAMPLE/)).not.toBeInTheDocument()
    expect(diagnosePortfolioRun).toHaveBeenCalledWith('run-1', [])
    expect(screen.getByText('收益贡献')).toBeInTheDocument()
    expect(screen.getByText('历史情景条件表现')).toBeInTheDocument()
    expect(screen.getByText('90.00% · 18/20 期')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '运行情景' }))
    await waitFor(() => expect(runPortfolioScenario).toHaveBeenCalledWith('run-1', expect.objectContaining({ name: '历史压力区间' })))
    expect(await screen.findByText(/区间收益/)).toBeInTheDocument()
    fireEvent.change(screen.getByRole('combobox', { name: 'CSV 导出表' }), { target: { value: 'correlation' } })
    fireEvent.click(screen.getByRole('button', { name: '导出 CSV' }))
    await waitFor(() => expect(downloadPortfolioExport).toHaveBeenCalledWith('run-1', 'csv', {
      table: 'correlation',
      scenario_start: '2020-01-01',
      scenario_end: '2020-03-31',
    }))
    fireEvent.click(screen.getByRole('button', { name: /选择组合指标/ }))
    fireEvent.click(screen.getByRole('checkbox', { name: /组合波动率/ }))
    fireEvent.click(screen.getByRole('button', { name: '计算选中指标' }))
    await waitFor(() => expect(diagnosePortfolioRun).toHaveBeenLastCalledWith('run-1', ['portfolio-volatility']))
    expect(await screen.findByText(/共同样本不足/)).toBeInTheDocument()
    expect(screen.queryByText(/INSUFFICIENT_COMMON_SAMPLE/)).not.toBeInTheDocument()
  })
})
