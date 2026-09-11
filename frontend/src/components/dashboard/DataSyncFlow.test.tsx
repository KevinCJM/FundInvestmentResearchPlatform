import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest'
import DataHealthRefreshPanel from './DataHealthRefreshPanel'

const makeStatus = (job: Record<string, unknown> = {}, token = true) => ({
  source: 'tushare', enabled: true, full_refresh_enabled: true,
  available_modules: ['base', 'etf', 'fund', 'index', 'macro'],
  token_configured: token, token_configuration_enabled: true, token_editable: true,
  job: { status: 'idle', message: '尚未启动更新', ...job }, datasets: {},
})
let snapshot = makeStatus()
let unavailable = false
let rejectStart = false
type FixtureResponse = { ok: boolean; status?: number; json: () => Promise<unknown> }
let fetchMock: Mock<[RequestInfo | URL, RequestInit?], Promise<FixtureResponse>>

beforeEach(() => {
  snapshot = makeStatus(); unavailable = false; rejectStart = false
  vi.spyOn(window, 'confirm').mockReturnValue(true)
  fetchMock = vi.fn<[RequestInfo | URL, RequestInit?], Promise<FixtureResponse>>(async input => {
    const url = String(input)
    if (url.startsWith('/api/data/refresh/status')) {
      if (unavailable) return { ok: false, status: 503, json: async () => ({}) }
      return { ok: true, json: async () => structuredClone(snapshot) }
    }
    if (url === '/api/data/token') {
      snapshot = makeStatus()
      return { ok: true, json: async () => ({ token_configured: true }) }
    }
    if (url === '/api/data/analytics/rebuild') {
      snapshot = makeStatus({ status: 'succeeded', message: '分析数据已整理', finished_at: new Date().toISOString() })
      return { ok: true, json: async () => ({ status: 'succeeded' }) }
    }
    if (url === '/api/data/refresh') {
      if (rejectStart) return { ok: false, status: 400, json: async () => ({ detail: { code: 'SOURCE_DISABLED', message: '接口已停用，请先启用接口。' } }) }
      snapshot = makeStatus({ status: 'running', message: '后台任务已启动', job_id: 'new-job' })
      return { ok: true, json: async () => structuredClone(snapshot) }
    }
    throw new Error('Unexpected request: ' + url)
  })
  vi.stubGlobal('fetch', fetchMock)
})
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

async function openPage() {
  render(<MemoryRouter><DataHealthRefreshPanel /></MemoryRouter>)
  await waitFor(() => expect(screen.queryByText('正在检查数据源状态...')).not.toBeInTheDocument())
  await waitFor(() => expect(screen.getByRole('button', { name: '重新检查状态' })).toBeEnabled())
}

const starts = () => fetchMock.mock.calls.filter(([url]) => url === '/api/data/refresh')

describe('数据同步用户流程', () => {
  it('日常默认不下载指数宏观，凭据和具体选项按需展开', async () => {
    await openPage()
    expect(screen.getByLabelText('输入 Token')).not.toBeVisible()
    const summary = within(screen.getByRole('region', { name: '本次同步清单' }))
    expect(summary.getByText('ETF')).toBeVisible()
    expect(summary.queryByText('指数')).not.toBeInTheDocument()
    expect(summary.queryByText('宏观数据')).not.toBeInTheDocument()
    expect(screen.getByRole('group', { name: '指数下载内容' })).not.toBeVisible()
    expect(starts()).toHaveLength(0)
  })

  it('研究方案只改变选择，明确启动后才提交正确范围', async () => {
    await openPage()
    fireEvent.click(screen.getByRole('button', { name: '指数与宏观研究' }))
    const summary = within(screen.getByRole('region', { name: '本次同步清单' }))
    expect(summary.getByText('指数')).toBeVisible()
    expect(summary.getByText('宏观数据')).toBeVisible()
    expect(summary.queryByText('ETF')).not.toBeInTheDocument()
    expect(starts()).toHaveLength(0)
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '开始数据更新' })) })
    const body = JSON.parse(starts()[0][1]!.body as string)
    expect(body.modules).toEqual(['base', 'index', 'macro'])
    expect(body.mode).toBe('incremental')
    expect(Object.keys(body.module_scopes)).toEqual(['base', 'index', 'macro'])
  })

  it('首次使用自动打开凭据；保存后才能同步', async () => {
    snapshot = makeStatus({}, false)
    await openPage()
    expect(screen.getByLabelText('输入 Token')).toBeVisible()
    expect(screen.getByRole('button', { name: '开始数据更新' })).toBeDisabled()
    fireEvent.change(screen.getByLabelText('输入 Token'), { target: { value: 'test-credential-value' } })
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '保存 Token' })) })
    await waitFor(() => expect(screen.getByRole('button', { name: '开始数据更新' })).toBeEnabled())
    expect(screen.getByLabelText('输入 Token')).toHaveValue('')
    expect(screen.getByLabelText('输入 Token')).not.toBeVisible()
  })

  it('后处理失败优先仅重建，不推荐重新抓取', async () => {
    snapshot = makeStatus({ status: 'failed', fetch_complete: true, staging_data_dir: 'candidate', resume_available: true,
      mode: 'full', modules: ['fund'], analytics_snapshot: { status: 'failed' } })
    await openPage()
    expect(screen.getByRole('region', { name: '恢复数据处理' })).toBeVisible()
    expect(screen.queryByRole('button', { name: '按原配置继续上次更新' })).not.toBeInTheDocument()
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '重建并接入候选快照' })) })
    expect(fetchMock).toHaveBeenCalledWith('/api/data/analytics/rebuild', expect.objectContaining({ body: JSON.stringify({ candidate: true }) }))
    expect(starts()).toHaveLength(0)
  })

  it('状态失联不能按旧状态启动，重新检查成功后解锁', async () => {
    await openPage()
    unavailable = true
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '重新检查状态' })) })
    expect(screen.getByRole('status')).toHaveTextContent('无法获取 Tushare 数据更新状态')
    expect(screen.getByRole('button', { name: '开始数据更新' })).toBeDisabled()
    expect(screen.queryByText('数据更新正在后台运行，下载入口已锁定。')).not.toBeInTheDocument()
    unavailable = false
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '重新检查状态' })) })
    expect(screen.getByRole('button', { name: '开始数据更新' })).toBeEnabled()
  })

  it('结构化错误显示具体原因而非 object Object', async () => {
    await openPage(); rejectStart = true
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '开始数据更新' })) })
    expect(screen.getByRole('status')).toHaveTextContent('接口已停用，请先启用接口。')
    expect(screen.getByRole('status')).not.toHaveTextContent('[object Object]')
    expect(screen.getByRole('button', { name: '开始数据更新' })).toBeEnabled()
  })

  it('手动选中未启用模块的一项，只加入该项及必要依赖', async () => {
    await openPage()
    fireEvent.click(screen.getByRole('button', { name: '自定义范围' }))
    const index = within(screen.getByRole('group', { name: '指数下载内容' }))
    fireEvent.click(index.getByRole('checkbox', { name: /国际指数/ }))
    expect(index.getByRole('checkbox', { name: /指数目录/ })).toBeChecked()
    expect(index.getByRole('checkbox', { name: /指数目录/ })).toBeDisabled()
    expect(index.getByRole('checkbox', { name: /境内指数/ })).not.toBeChecked()
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '开始数据更新' })) })
    const body = JSON.parse(starts()[0][1]!.body as string)
    expect(body.module_scopes.index).toEqual(['catalog', 'global'])
  })

  it('取消全量确认不会启动任务', async () => {
    await openPage()
    fireEvent.click(screen.getByRole('radio', { name: /^全量更新/ }))
    vi.mocked(window.confirm).mockReturnValue(false)
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: '开始数据更新' })) })
    expect(window.confirm).toHaveBeenCalledTimes(1)
    expect(starts()).toHaveLength(0)
  })
})
