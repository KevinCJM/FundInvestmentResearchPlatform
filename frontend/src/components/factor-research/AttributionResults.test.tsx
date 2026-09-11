import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import AttributionResults from './AttributionResults'
import AttributionWorkbench from './AttributionWorkbench'
import { attributionCsv, contributionText } from './attributionPresentation'
import { fixtureAttributionRun } from '../../test/factorAttributionFixtures'
import { fixtureStudy } from '../../test/factorFixtures'
import type { Action } from './shared'

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="attribution-chart" /> }))
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks() })

describe('style exposures and linked return contributions', () => {
  it('shows server-side compound reconciliation in percentage points, not summed daily returns', () => {
    render(<AttributionResults run={fixtureAttributionRun} />)
    expect(screen.getByLabelText('贡献评价区间')).toHaveValue('out_of_sample')
    expect(screen.getByLabelText('累计因子贡献')).toHaveTextContent('3.6 个百分点')
    expect(screen.getByLabelText('累计因子贡献')).toHaveTextContent('-1.2 个百分点')
    expect(screen.getByRole('status')).toHaveTextContent('完整对账')
    expect(screen.getAllByTestId('attribution-chart')).toHaveLength(3)
    expect(screen.getByLabelText('月度贡献对账')).toHaveTextContent('4.5 个百分点')
  })

  it('distinguishes empty and incomplete intervals from zero', () => {
    const run = structuredClone(fixtureAttributionRun)
    const summary = run.attribution!.products[0].summaries.out_of_sample
    summary.status = 'incomplete'; summary.valid_days = 1
    summary.contributions = [null, null, null]; summary.contribution_sum = null; summary.reconciliation_error = null
    render(<AttributionResults run={run} />)
    expect(screen.getByRole('status')).toHaveTextContent('缺口未被跳过')
    expect(screen.getByLabelText('累计因子贡献')).not.toHaveTextContent('0 个百分点')
    fireEvent.change(screen.getByLabelText('贡献评价区间'), { target: { value: 'in_sample' } })
    expect(screen.getByRole('status')).toHaveTextContent('无可评价日期')
  })

  it('keeps old runs readable without fabricating contribution data', () => {
    const { attribution: _, ...old } = fixtureAttributionRun
    render(<AttributionResults run={old} />)
    expect(screen.getByText(/旧运行未保存逐日贡献/)).toBeInTheDocument()
    expect(screen.getByLabelText('暴露与拟合摘要')).toHaveTextContent('离线测试基金')
    expect(screen.queryByLabelText('累计因子贡献')).not.toBeInTheDocument()
  })

  it('labels rolling coefficients as the last fit rather than a retroactive constant history', () => {
    const run = structuredClone(fixtureAttributionRun)
    run.attribution!.mode = 'rolling'; run.attribution!.warmup_days = 126
    render(<AttributionResults run={run} />)
    expect(screen.getByText('末次窗口 R²')).toBeInTheDocument()
    expect(screen.getByText(/末次系数回填历史/)).toBeInTheDocument()
    expect(screen.getByText(/排除起始 126 个交易日预热/)).toBeInTheDocument()
  })

  it('exports raw decimals, missing values and neutralized formula labels', () => {
    const run = structuredClone(fixtureAttributionRun)
    run.attribution!.components[0].label = '=UNTRUSTED("x")'
    const product = run.attribution!.products[0]
    product.daily[0].contributions[0] = null
    const csv = attributionCsv(run, product)
    expect(csv).toContain("\"'=UNTRUSTED(\"\"x\"\")")
    expect(csv).toContain('"-0.04"')
    expect(csv).not.toContain("'-0.04")
    expect(csv).toContain('decimal_return_contribution')
    expect(csv).not.toContain('null')
    expect(contributionText(0)).toBe('0 个百分点')
    expect(contributionText(null)).toBe('—')
  })

  it('runs the CSV download through a blob and reports export failures', () => {
    vi.useFakeTimers()
    const create = vi.fn(() => 'blob:offline-csv')
    vi.stubGlobal('URL', { createObjectURL: create, revokeObjectURL: vi.fn() })
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
    render(<AttributionResults run={fixtureAttributionRun} />)
    fireEvent.click(screen.getByRole('button', { name: '导出逐日贡献 CSV' }))
    expect(create).toHaveBeenCalledOnce(); expect(click).toHaveBeenCalledOnce()
    create.mockImplementation(() => { throw new Error('浏览器禁止导出') })
    fireEvent.click(screen.getByRole('button', { name: '导出逐日贡献 CSV' }))
    expect(screen.getByRole('alert')).toHaveTextContent('浏览器禁止导出')
    act(() => { vi.runOnlyPendingTimers() })
  })

  it('passes rolling parameters to the existing API and clears results when mode changes', async () => {
    const posted: unknown[] = []
    vi.stubGlobal('fetch', vi.fn(async (_input, init?: RequestInit) => {
      const body = init?.body ? JSON.parse(String(init.body)) : undefined
      if (body) posted.push(body)
      return { ok: true, status: 200, json: async () => body ? fixtureAttributionRun : { items: [] } }
    }))
    const action: Action = async (_label, work) => work()
    render(<MemoryRouter><AttributionWorkbench seed={fixtureStudy} action={action} busy={false} /></MemoryRouter>)
    await act(async () => { await Promise.resolve() })
    fireEvent.change(screen.getByLabelText('归因产品代码'), { target: { value: '000001.OF' } })
    fireEvent.change(screen.getByLabelText('暴露估计方式'), { target: { value: 'rolling' } })
    expect(screen.getByLabelText('滚动窗口（交易日）')).toHaveValue(126)
    fireEvent.change(screen.getByLabelText('重新估计间隔（交易日）'), { target: { value: '5' } })
    fireEvent.click(screen.getByRole('button', { name: '运行归因研究' }))
    await waitFor(() => expect(posted).toHaveLength(1))
    expect(posted[0]).toMatchObject({ exposure_mode: 'rolling', rolling_window: 126, min_observations: 60, refit_step: 5 })
    await screen.findByLabelText('累计因子贡献')
    fireEvent.change(screen.getByLabelText('暴露估计方式'), { target: { value: 'fixed' } })
    expect(screen.queryByLabelText('累计因子贡献')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('滚动窗口（交易日）')).not.toBeInTheDocument()
  })
})
