import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import SourceSyncPanel from './SourceSyncPanel'
import type { ConfigRecord, InterfaceConfig, SourceConfig } from '../../services/dataSources'
import * as api from '../../services/dataSources'

vi.mock('../../services/dataSources', () => ({ listSourceSyncJobs: vi.fn(), startSourceSync: vi.fn() }))
const source = { id: 'akshare', name: 'AKShare', transport: 'akshare' } as SourceConfig
const record = { config: { id: 'akshare.etf_daily', name: 'ETF 日行情', params: { symbol: '510300' }, start_param: 'start_date', end_param: 'end_date' }, revision: 2 } as unknown as ConfigRecord<InterfaceConfig>

beforeEach(() => {
  vi.resetAllMocks()
  vi.spyOn(window, 'confirm').mockReturnValue(true)
  vi.mocked(api.listSourceSyncJobs).mockResolvedValue([])
})
async function open() {
  render(<form><SourceSyncPanel record={record} source={source} disabled={false} /></form>)
  const summary = screen.getByText('5. 下载与更新此接口')
  const details = summary.parentElement as HTMLDetailsElement
  details.open = true
  fireEvent(details, new Event('toggle'))
  await waitFor(() => expect(api.listSourceSyncJobs).toHaveBeenCalled())
}

describe('SourceSyncPanel', () => {
  it('其他接口运行时禁止重复下载', async () => {
    vi.mocked(api.listSourceSyncJobs).mockResolvedValue([{ job_id: 'other', interface_id: 'akshare.fund_nav', source_id: 'akshare', status: 'RUNNING', mode: 'full', rows: 0, pages: 0, published: false }])
    await open()
    expect(await screen.findByText(/另一个接口正在下载/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '开始此接口下载', hidden: true })).toBeDisabled()
    expect(api.startSourceSync).not.toHaveBeenCalled()
  })

  it('状态读取失败时不能启动', async () => {
    vi.mocked(api.listSourceSyncJobs).mockRejectedValue(new Error('服务暂时不可用'))
    await open()
    expect(await screen.findByRole('alert')).toHaveTextContent('服务暂时不可用')
    expect(screen.getByRole('button', { name: '开始此接口下载', hidden: true })).toBeDisabled()
  })

  it('非法 JSON 草稿不能偷偷提交上一次有效值', async () => {
    await open()
    await waitFor(() => expect(screen.getByRole('button', { name: '开始此接口下载', hidden: true })).toBeEnabled())
    fireEvent.change(screen.getByLabelText('本次下载参数'), { target: { value: '{invalid' } })
    fireEvent.click(screen.getByRole('button', { name: '开始此接口下载', hidden: true }))
    expect(api.startSourceSync).not.toHaveBeenCalled()
    expect(screen.getByRole('alert')).toHaveTextContent('修正无效的参数')
  })

  it('取消确认不会发起网络下载任务', async () => {
    await open()
    await waitFor(() => expect(screen.getByRole('button', { name: '开始此接口下载', hidden: true })).toBeEnabled())
    vi.mocked(window.confirm).mockReturnValue(false)
    fireEvent.click(screen.getByRole('button', { name: '开始此接口下载', hidden: true }))
    expect(api.startSourceSync).not.toHaveBeenCalled()
  })
})
