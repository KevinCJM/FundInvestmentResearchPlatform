import { MemoryRouter } from 'react-router-dom'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../../services/regimeGraph'
import { createBlankRegimeDefinition, type RegimeGraphDefinition } from '../../services/regimeGraph'
import { formatDate } from '../../i18n/runtime'
import RegimeStudyList from './RegimeStudyList'

vi.mock('../../services/regimeGraph', async original => ({
  ...await original<typeof import('../../services/regimeGraph')>(),
  listRegimeGraphDefinitions: vi.fn(), listRegimeFormalRuns: vi.fn(), listHistoricalReferences: vi.fn(), listRegimeReliability: vi.fn(),
}))

const study = (over: Partial<RegimeGraphDefinition>): RegimeGraphDefinition => ({
  ...createBlankRegimeDefinition(), id: 'reference-1', revision: 3, name: '牛熊参考', description: '月频峰谷划分',
  states: [{ id: 'bull', label: '牛' }, { id: 'bear', label: '熊' }] as RegimeGraphDefinition['states'],
  updated_at: '2026-09-20T02:00:00Z', study: { purpose: 'historical_reference', family: 'market_trend' }, ...over,
})
const reference = { run_id: 'run-1', publication_id: 'pub-1', content_hash: 'hash-1' }
const published = { ...reference, definition_id: 'reference-1', definition_revision: 3, name: '牛熊参考', frequency: 'monthly', states: [], as_of: '2026-06-30', created_at: '2026-09-20', series_summary: null }

const mount = (stage: 'historical' | 'realtime' | 'validation', newHref?: string) => render(
  <MemoryRouter initialEntries={['/settings/scenario-algorithms?center=market-state']}>
    <RegimeStudyList
      stage={stage}
      studyHref={definition => `?center=market-state&definition=${definition.id}&revision=${definition.revision}`}
      newHref={newHref}
    />
  </MemoryRouter>,
)

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(api.listRegimeFormalRuns).mockResolvedValue([])
  vi.mocked(api.listHistoricalReferences).mockResolvedValue([])
  vi.mocked(api.listRegimeReliability).mockResolvedValue([])
  vi.mocked(api.listRegimeGraphDefinitions).mockResolvedValue([])
})

describe('市场状态研究的已保存清单', () => {
  it.each(['historical', 'realtime', 'validation'] as const)('%s 的 PIT 只使用当前修订与模式的最近正式运行，缺失时不猜日期', async stage => {
    const mode = stage === 'historical' ? 'retrospective' : 'realtime'
    const purpose = stage === 'historical' ? 'historical_reference' : 'realtime_recognition'
    const model = (id: string) => study({ id, name: id, study: { purpose, family: 'custom', ...(stage !== 'historical' ? { reference } : {}) } })
    vi.mocked(api.listRegimeGraphDefinitions).mockResolvedValue(['dated', 'unset', 'legacy', 'new-revision'].map(model))
    const run = (over: Partial<api.RegimeFormalRun>): api.RegimeFormalRun => ({
      id: 'run', definition_id: 'dated', definition_revision: 3, name: '正式运行', mode,
      as_of: '2019-12-31', created_at: '2026-09-20', ...over,
    })
    vi.mocked(api.listRegimeFormalRuns).mockResolvedValue([
      run({ id: 'older', as_of: '2018-12-31', created_at: '2026-09-19' }),
      run({ id: 'other-revision', definition_revision: 2, as_of: '2024-01-01', created_at: '2026-09-23' }),
      run({ id: 'other-mode', mode: mode === 'realtime' ? 'retrospective' : 'realtime', as_of: '2025-01-01', created_at: '2026-09-22' }),
      run({ id: 'latest' }),
      run({ id: 'null-cutoff', definition_id: 'unset', as_of: null }),
      run({ id: 'missing-cutoff', definition_id: 'legacy', as_of: undefined }),
      run({ id: 'previous-revision-only', definition_id: 'new-revision', definition_revision: 2 }),
    ])
    vi.mocked(api.listHistoricalReferences).mockResolvedValue([published])
    mount(stage)
    await screen.findByRole('link', { name: 'dated' })
    expect(screen.getByRole('columnheader', { name: 'PIT 日期' })).toBeVisible()
    const row = (name: string) => within(screen.getByRole('link', { name }).closest('tr')!)
    expect(row('dated').getByText(formatDate('2019-12-31'))).toHaveAttribute('datetime', '2019-12-31')
    expect(row('dated').queryByText(formatDate(published.as_of))).not.toBeInTheDocument()
    expect(screen.queryByText(formatDate('2024-01-01'))).not.toBeInTheDocument()
    expect(screen.queryByText(formatDate('2025-01-01'))).not.toBeInTheDocument()
    expect(row('unset').getByText('未设置')).toBeVisible()
    expect(row('legacy').getByText('未记录')).toBeVisible()
    expect(row('new-revision').getByText('当前版本尚未运行')).toBeVisible()
  })

  it('运行目录失败不伪装成无 PIT，重试后恢复日期', async () => {
    vi.mocked(api.listRegimeGraphDefinitions).mockResolvedValue([study({})])
    vi.mocked(api.listRegimeFormalRuns).mockRejectedValueOnce(new Error('运行目录不可用'))
    mount('historical')
    await screen.findByRole('link', { name: '牛熊参考' })
    expect(screen.getByRole('alert')).toHaveTextContent('正式运行记录读取失败')
    expect(screen.getByText('读取失败', { exact: true })).toBeVisible()
    expect(screen.queryByText('当前版本尚未运行')).not.toBeInTheDocument()
    vi.mocked(api.listRegimeFormalRuns).mockResolvedValue([
      { id: 'run', definition_id: 'reference-1', definition_revision: 3, name: '正式运行', mode: 'retrospective', as_of: '2019-12-31', created_at: '2026-09-20' },
    ])
    await userEvent.click(screen.getByRole('button', { name: '重试读取 PIT 日期' }))
    await waitFor(() => expect(screen.getByText(formatDate('2019-12-31'))).toBeVisible())
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('按研究用途归档，历史参考确认后才是可用状态', async () => {
    vi.mocked(api.listRegimeGraphDefinitions).mockResolvedValue([
      study({}),
      study({ id: 'draft-1', revision: 1, name: '待确认参考', updated_at: '2026-09-19T02:00:00Z' }),
      study({ id: 'model-1', name: '实时模型', study: { purpose: 'realtime_recognition', family: 'custom' } }),
    ])
    vi.mocked(api.listHistoricalReferences).mockResolvedValue([published])
    mount('historical', '?center=market-state&stage=historical&new=1')

    const row = (name: string) => screen.getByRole('link', { name }).closest('tr')!
    await waitFor(() => expect(screen.getByRole('link', { name: '牛熊参考' })).toBeVisible())
    // 实时识别模型不混进历史参考清单。
    expect(screen.queryByRole('link', { name: '实时模型' })).not.toBeInTheDocument()
    expect(within(row('牛熊参考')).getByText('已确认参考')).toBeVisible()
    expect(within(row('牛熊参考')).getByText('2 个状态 · 0 个节点')).toBeVisible()
    expect(within(row('待确认参考')).getByText('尚未运行')).toBeVisible()
    expect(screen.getByRole('link', { name: '牛熊参考' })).toHaveAttribute('href', '/settings/scenario-algorithms?center=market-state&definition=reference-1&revision=3')
    expect(screen.getByRole('link', { name: '新建历史参考算法' })).toHaveAttribute('href', '/settings/scenario-algorithms?center=market-state&stage=historical&new=1')
  })

  it('实时模型显示绑定的历史参考与发布状态', async () => {
    vi.mocked(api.listRegimeGraphDefinitions).mockResolvedValue([
      study({ id: 'model-1', name: '实时模型', study: { purpose: 'realtime_recognition', family: 'custom', reference } }),
      study({ id: 'model-2', name: '未绑定模型', study: { purpose: 'realtime_recognition', family: 'custom' } }),
    ])
    vi.mocked(api.listHistoricalReferences).mockResolvedValue([published])
    vi.mocked(api.listRegimeFormalRuns).mockResolvedValue([
      { id: 'run-1', definition_id: 'model-1', definition_revision: 3, name: '正式运行', mode: 'realtime', created_at: '2026-09-21', publications: [{ id: 'p1', usage: 'taa', published_at: '2026-09-21' }] },
    ])
    mount('realtime', '?stage=realtime&new=1')

    await waitFor(() => expect(screen.getByRole('link', { name: '实时模型' })).toBeVisible())
    const row = (name: string) => screen.getByRole('link', { name }).closest('tr')!
    expect(within(row('实时模型')).getByText('已发布')).toBeVisible()
    expect(within(row('实时模型')).getByText('牛熊参考 · v3')).toBeVisible()
    expect(within(row('未绑定模型')).getByText('未绑定参考')).toBeVisible()
    expect(within(row('未绑定模型')).getByText('尚未运行')).toBeVisible()
  })

  it('验证步骤列出实时模型与验证结论，且不提供新建', async () => {
    vi.mocked(api.listRegimeGraphDefinitions).mockResolvedValue([
      study({ id: 'model-1', name: '实时模型', study: { purpose: 'realtime_recognition', family: 'custom' } }),
    ])
    vi.mocked(api.listRegimeReliability).mockResolvedValue([
      { id: 'report-1', calibration_id: 'cal-1', created_at: '2026-09-21', definition_id: 'model-1', revision: 3, reference, status: 'insufficient', calibration: null, verification: { status: 'insufficient', recognition_ready: false, verified_states: [], unverified_states: ['bull'] } } as never,
    ])
    mount('validation')

    await waitFor(() => expect(screen.getByRole('link', { name: '实时模型' })).toBeVisible())
    expect(screen.getByText('已出报告 · 未达可识别')).toBeVisible()
    expect(screen.getByRole('link', { name: '去验证' })).toBeVisible()
    expect(screen.queryByRole('link', { name: /新建/ })).not.toBeInTheDocument()
  })

  it('空清单说明下一步，读取失败可重试', async () => {
    mount('historical', '?stage=historical&new=1')
    await waitFor(() => expect(screen.getByText('还没有历史参考算法')).toBeVisible())
    expect(screen.getByText(/跑出历史区间并确认后/)).toBeVisible()

    vi.mocked(api.listRegimeGraphDefinitions).mockRejectedValueOnce(new Error('清单服务不可用'))
    const failed = mount('historical', '?stage=historical&new=1')
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('暂时无法读取数据，请重试。'))
    expect(failed.container.querySelector('img[src*="mascot-error"]')).not.toBeNull()
    expect(within(failed.container).queryByRole('table')).not.toBeInTheDocument()
    expect(within(failed.container).queryByText('还没有历史参考算法')).not.toBeInTheDocument()
    vi.mocked(api.listRegimeGraphDefinitions).mockResolvedValue([study({})])
    await userEvent.click(within(failed.container).getByRole('button', { name: '重试读取' }))
    await waitFor(() => expect(within(failed.container).getByRole('link', { name: '牛熊参考' })).toBeVisible())
  })
})
