import { act, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { getRegimeFormalRun, getRegimePreviewOverview, getRegimePreviewSeries } from '../../services/regimeGraph'
import { resultFixture } from './regimeResultFixtures'
import RegimeResultView from './RegimeResultView'

vi.mock('../../services/regimeGraph', () => ({
  getRegimeFormalRun: vi.fn(), getRegimePreviewOverview: vi.fn(), getRegimePreviewSeries: vi.fn(),
}))
vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="chart" /> }))
afterEach(() => { vi.clearAllMocks() })

describe('RegimeResultView frozen run loading', () => {
  it('旧overview慢返回不能覆盖新run，草稿过期提示不改变冻结颜色或名称', async () => {
    const a = resultFixture('A', 600)
    const b = resultFixture('B', 600)
    let resolveA!: (value: unknown) => void
    const pendingA = new Promise<unknown>(resolve => { resolveA = resolve })
    vi.mocked(getRegimePreviewOverview).mockImplementation(async id => id === 'A' ? pendingA : b.overview)
    vi.mocked(getRegimePreviewSeries).mockImplementation(async (id, options = {}) => ({ run_id: id, total: 600, offset: options.offset ?? 0, limit: options.limit ?? 5000, items: id === 'A' ? a.rows : b.rows }))
    const { rerender } = render(<RegimeResultView runId="A" />)
    rerender(<RegimeResultView runId="B" stale />)
    await screen.findByText(/曲线：B 主对照走势/)
    await act(async () => { resolveA(a.overview); await pendingA })
    expect(screen.queryByText(/曲线：A 主对照走势/)).not.toBeInTheDocument()
    expect(screen.getByText(/此结果与当前配置不同/)).toBeInTheDocument()
    expect(screen.getByRole('region', { name: '完整历史情景结果' })).toHaveAttribute('data-run-id', 'B')
    const firstSignal = vi.mocked(getRegimePreviewOverview).mock.calls[0][1]
    expect(firstSignal?.aborted).toBe(true)
  })

  it('旧series返回不能覆盖新run，未完整加载前不显示全样本摘要', async () => {
    const a = resultFixture('A', 600)
    const b = resultFixture('B', 600)
    let resolveA!: (value: Awaited<ReturnType<typeof getRegimePreviewSeries>>) => void
    const pendingA = new Promise<Awaited<ReturnType<typeof getRegimePreviewSeries>>>(resolve => { resolveA = resolve })
    vi.mocked(getRegimePreviewOverview).mockImplementation(async id => id === 'A' ? a.overview : b.overview)
    vi.mocked(getRegimePreviewSeries).mockImplementation(async (id, options = {}) => id === 'A' ? pendingA : { run_id: id, total: 600, offset: options.offset ?? 0, limit: 5000, items: b.rows })
    const { rerender } = render(<RegimeResultView runId="A" />)
    await waitFor(() => expect(getRegimePreviewSeries).toHaveBeenCalled())
    expect(screen.queryByLabelText('全样本情景摘要')).not.toBeInTheDocument()
    rerender(<RegimeResultView runId="B" />)
    await screen.findByText(/曲线：B 主对照走势/)
    await act(async () => { resolveA({ run_id: 'A', total: 600, offset: 0, limit: 5000, items: a.rows }); await pendingA })
    expect(screen.getByRole('region', { name: '完整历史情景结果' })).toHaveAttribute('data-run-id', 'B')
    expect(screen.queryByLabelText('状态概率')).not.toBeInTheDocument()
  })

  it('正式运行使用现有详情全量series，不调用不存在的正式分页接口', async () => {
    const { overview, rows } = resultFixture('saved-1', 600)
    vi.mocked(getRegimeFormalRun).mockResolvedValue({ id: 'saved-1', definition_id: 'def-1', definition_revision: 2, name: '正式研究', mode: 'retrospective', created_at: '2026-09-06', overview: { ...overview, run_kind: 'saved' }, series: rows })
    render(<RegimeResultView runId="saved-1" runKind="formal" />)
    await screen.findByText(/曲线：saved-1 主对照走势/)
    expect(getRegimePreviewOverview).not.toHaveBeenCalled()
    expect(getRegimePreviewSeries).not.toHaveBeenCalled()
  })
})
