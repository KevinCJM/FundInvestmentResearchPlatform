import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useState } from 'react'
import GlobalEventLibrary from './GlobalEventLibrary'
import EventLibraryEditor from './EventLibraryEditor'
import RegimeManualEventEditor from './RegimeManualEventEditor'
import * as api from '../../services/eventLibrary'
import { libraryEventFixture as fixture } from '../../test/eventLibraryFixtures'
import type { ManualHistoricalEvent } from '../../services/regimeGraph'

vi.mock('../../services/eventLibrary', async original => ({ ...await original<typeof api>(),
  listLibraryEvents: vi.fn(), listEventPacks: vi.fn(), getEventPack: vi.fn(), resolveLibraryEvents: vi.fn(),
  getLibraryEventHistory: vi.fn(), saveLibraryEvent: vi.fn(),
}))
const linked: ManualHistoricalEvent = { id: 'library_abc', label: fixture.name, start_date: '2020-01-01', end_date: '2020-01-20', color: fixture.color, description: '窗口理由', library_reference: { event_id: fixture.id, revision: 1, window_id: 'acute', content_hash: fixture.content_hash } }

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(api.listLibraryEvents).mockResolvedValue({ items: [fixture], total: 1, offset: 0, limit: 20 })
  vi.mocked(api.listEventPacks).mockResolvedValue({ items: [{ id: 'all', name: '全部历史事件', count: 1 }] })
  vi.mocked(api.getEventPack).mockResolvedValue({ selections: [{ event_id: fixture.id, revision: 1, window_id: 'acute' }], labels: { 'event-demo:1:acute': '全球供应链事件 · 急性窗口' } })
  vi.mocked(api.getLibraryEventHistory).mockResolvedValue({ items: [fixture] })
  vi.mocked(api.resolveLibraryEvents).mockResolvedValue({ events: [linked] })
})

describe('全球事件库', () => {
  it('明确区分未核验事实与可选择研究窗口，并按确切修订加入', async () => {
    const onSelect = vi.fn(); const user = userEvent.setup()
    render(<GlobalEventLibrary onSelect={onSelect} />)
    await user.click(await screen.findByRole('button', { name: /全球供应链事件/ }))
    const detail = screen.getByRole('article', { name: '事件详情' })
    expect(within(detail).getByText(/尚未核实/)).toBeInTheDocument()
    expect(within(detail).getByText('待核验，不是官方标准区间')).toBeInTheDocument()
    await user.click(within(detail).getAllByRole('button', { name: '选择此窗口' })[0])
    expect(screen.getByRole('region', { name: '已选事件' })).toHaveTextContent('1')
    await user.click(screen.getByRole('button', { name: '加入当前情景' }))
    expect(api.resolveLibraryEvents).toHaveBeenCalledWith([{ event_id: fixture.id, revision: 1, window_id: 'acute' }], expect.any(AbortSignal))
    expect(onSelect).toHaveBeenCalledWith([linked])
  })
  it('事件包与逐项选择去重，并能取消单项', async () => {
    const user = userEvent.setup(); render(<GlobalEventLibrary onSelect={vi.fn()} />)
    await user.click(await screen.findByRole('button', { name: '选择主要窗口' }))
    await user.click(screen.getByRole('button', { name: '全部历史事件 · 1' }))
    await waitFor(() => expect(screen.getByRole('region', { name: '已选事件' })).toHaveTextContent('1'))
    await user.click(screen.getByRole('button', { name: '已选择 · 移除主要窗口' }))
    expect(screen.getByRole('button', { name: '加入当前情景' })).toBeDisabled()
  })
  it('搜索请求使用当前筛选，空结果不会保留旧列表', async () => {
    render(<GlobalEventLibrary />)
    await screen.findByRole('button', { name: /全球供应链事件/ })
    vi.mocked(api.listLibraryEvents).mockResolvedValue({ items: [], total: 0, offset: 0, limit: 20 })
    fireEvent.change(screen.getByLabelText('事件库搜索'), { target: { value: '不存在的事件' } })
    await screen.findByText(/没有匹配事件/)
    expect(api.listLibraryEvents).toHaveBeenLastCalledWith(expect.objectContaining({ query: '不存在的事件' }), expect.any(AbortSignal))
    expect(screen.queryByRole('button', { name: /全球供应链事件/ })).not.toBeInTheDocument()
  })
  it('取消选择后，迟到的解析结果不得写入草稿', async () => {
    let resolve!: (value: { events: ManualHistoricalEvent[] }) => void
    vi.mocked(api.resolveLibraryEvents).mockReturnValue(new Promise(r => { resolve = r }))
    const onSelect = vi.fn(); const onCancel = vi.fn(); const user = userEvent.setup()
    render(<GlobalEventLibrary onSelect={onSelect} onCancel={onCancel} />)
    await user.click(await screen.findByRole('button', { name: '选择主要窗口' }))
    await user.click(screen.getByRole('button', { name: '加入当前情景' }))
    await user.click(screen.getByRole('button', { name: '取消选择' }))
    await act(async () => { resolve({ events: [linked] }) })
    expect(onCancel).toHaveBeenCalledOnce(); expect(onSelect).not.toHaveBeenCalled()
  })
  it('表单允许自然输入多地区，事实日期未知不会用研究窗口冒充', async () => {
    const user = userEvent.setup(); const onSaved = vi.fn()
    vi.mocked(api.saveLibraryEvent).mockResolvedValue(fixture)
    render(<EventLibraryEditor onSaved={onSaved} onCancel={vi.fn()} />)
    await user.type(screen.getByLabelText('库事件名称'), '测试贸易冲击')
    await user.type(screen.getByLabelText('地区（逗号分隔）'), '中国, 美国')
    fireEvent.change(screen.getByLabelText('窗口1开始'), { target: { value: '2020-01-01' } })
    fireEvent.change(screen.getByLabelText('窗口1结束'), { target: { value: '2020-02-01' } })
    await user.type(screen.getByLabelText('窗口1理由'), '检验市场压力阶段')
    await user.click(screen.getByRole('button', { name: '保存事件' }))
    await waitFor(() => expect(api.saveLibraryEvent).toHaveBeenCalledWith(expect.objectContaining({ regions: ['中国', '美国'], fact_start: null, fact_end: null, verification: 'unreviewed' }), undefined))
    expect(onSaved).toHaveBeenCalledWith(fixture)
  })
  it('库引用只读；转人工副本会分配新身份，不与库记录混淆', async () => {
    let current = [linked]
    function Harness() { const [events, setEvents] = useState([linked]); current = events; return <RegimeManualEventEditor value={events} onChange={setEvents} /> }
    const user = userEvent.setup(); render(<Harness />)
    expect(screen.getByLabelText('事件1名称')).toHaveAttribute('readonly')
    await user.click(screen.getByRole('button', { name: '转为人工副本后编辑' }))
    expect(screen.getByLabelText('事件1名称')).not.toHaveAttribute('readonly')
    expect(current[0].library_reference).toBeUndefined()
    expect(current[0].id).not.toEqual(linked.id)
  })
})
