import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import AgentPanel from './IndicatorAgentPanel'
import type { AgentEvent, AgentRun, PageEvidenceSnapshot } from '../../services/agent'

const props = { pageContext: { page: 'indicator-studio' as const, page_instance_id: 'test', context_revision: 0, view_state: 'inherit' as const,
  calculation: { context_kind: 'single_product', targets: [], period: '1Y' } }, draft: { expression: '' } }
const evidence = (marker: string, index = 0): PageEvidenceSnapshot => ({
  version: 1, snapshot_id: `snap-${String(index).padStart(32, '0')}`, captured_at: '2026-09-21T02:00:00.000Z',
  page: 'indicator-studio', sections: { editing: { marker } },
})
afterEach(() => { cleanup(); sessionStorage.clear(); vi.unstubAllGlobals() })
const response = (data: unknown, status = 200) => ({ ok: status < 400, status, json: async () => JSON.parse(JSON.stringify(data)) })

function server(options: { configured?: boolean; active?: boolean; draft?: boolean; defer?: boolean } = {}) {
  let revision = 0, context = props.pageContext, run: AgentRun | null = null
  let sessionId = 's', sessionCount = 0
  const events: AgentEvent[] = [], requests: Record<string, unknown>[] = []
  let release: ((value?: boolean) => void) | null = null
  const definition = { name: '价差均值', expression: 'mean(adjusted_high - adjusted_low)', context_kind: 'single_product' }
  const draft = options.draft ? { valid: true, draft_revision: 1, definition_hash: 'hash', definition } : null
  const add = (event: AgentEvent) => { const value = { ...event, seq: events.length + 1 }; events.push(value); return value }
  const finish = (text = '已检查你的指标需求。', status: AgentRun['status'] = 'completed') => {
    if (!run) return []
    const artifacts = options.draft && revision === 1 ? { draft } : {}
    run = { ...run, status, response: { session_id: sessionId, session_revision: revision, reply: { text }, draft, artifacts } }
    return [add({ type: 'assistant.message', id: `${run.run_id}-reply`, run_id: run.run_id, speaker: 'assistant', text, artifacts }), add({ type: `run.${status}`, run_id: run.run_id, data: { status } })]
  }
  const fetcher = vi.fn(async (url: string, init?: RequestInit): Promise<unknown> => {
    const path = new URL(url, 'http://localhost').pathname
    if (path.endsWith('/meta')) return response({ configured: options.configured !== false })
    if (path.endsWith('/sessions')) {
      context = JSON.parse(String(init?.body)).page_context
      sessionId = ++sessionCount === 1 ? 's' : `s${sessionCount}`
      revision = 0; run = null; events.length = 0
      return response({ session_id: sessionId, session_revision: 0 })
    }
    if (path.endsWith('/messages')) {
      const body = JSON.parse(String(init?.body)); requests.push(body)
      if (options.defer && requests.length === 1) {
        const success = await new Promise<boolean>(resolve => { release = value => resolve(!!value) })
        if (!success) return response({ detail: { message: '连接暂时失败' } }, 502)
      }
      revision += 1
      add({ type: 'user.message', speaker: 'user', id: body.message_id, text: body.text })
      run = { run_id: `r${revision}`, session_id: sessionId, message_id: body.message_id, session_revision: revision, run_revision: 1, status: 'running', phase: 'thinking' }
      add({ type: 'run.started', run_id: run.run_id, data: { status: 'running', phase: 'thinking' } })
      if (!options.active) finish()
      return response(run, 202)
    }
    if (path.endsWith('/events')) {
      const after = Number(new URL(url, 'http://localhost').searchParams.get('after_seq'))
      const items = events.filter(e => (e.seq || 0) > after).slice(0, 200)
      return response({ items, has_more: false, last_seq: items[items.length - 1]?.seq || after, next_event_seq: events.length + 1 })
    }
    if (path.endsWith('/cancel')) { finish('已停止，进度已保留。', 'cancelled'); return response(run) }
    if (path.endsWith('/invalidate-context')) { finish('口径已变化。', 'paused'); return response(run) }
    if (path.includes('/runs/')) return response(run)
    if (path.endsWith('/commit-preview')) return response({ confirmation_id: 'confirm', definition_hash: 'hash', draft_revision: 1, definition, preview_status: 'valid', impact: { action: 'create', name: definition.name, context_kind: 'single_product', target: null, name_conflict_indicator_id: 'existing-indicator' } })
    if (path.endsWith('/commit')) return response({ indicator_id: 'i', revision: 1 })
    if (path.endsWith(`/sessions/${sessionId}`)) return response({ session_id: sessionId, session_revision: revision, page_context: { calculation: { as_of: null, ...context.calculation }, view_state: context.view_state, context_revision: context.context_revision, page_instance_id: context.page_instance_id, page: context.page }, messages: events.filter(e => e.type === 'user.message' || e.type === 'assistant.message').slice(-200), older_message_cursor: null, events: events.slice(0, 200), next_event_seq: events.length + 1, active_run: run, draft })
    throw new Error(`unexpected ${url}`)
  })
  vi.stubGlobal('fetch', fetcher); vi.stubGlobal('EventSource', undefined); vi.stubGlobal('confirm', vi.fn(() => true))
  return { fetcher, requests, finish, release: (value?: boolean) => release?.(value), definition }
}
async function open() {
  fireEvent.click(screen.getByRole('button', { name: '打开 AI 助手' }))
  await waitFor(() => expect(screen.getByRole('textbox')).toBeEnabled())
}
function send(text: string) { fireEvent.change(screen.getByRole('textbox'), { target: { value: text } }); fireEvent.click(screen.getByRole('button', { name: '发送' })) }

describe('AgentPanel', () => {
  it('先展示冻结预览与同名影响，取消不写入，再次明确确认才保存', async () => {
    const api = server({ draft: true })
    const confirm = vi.fn((_message: string) => {
      expect(api.fetcher.mock.calls.some(([url]) => url.endsWith('/commit-preview'))).toBe(true)
      expect(api.fetcher.mock.calls.some(([url]) => url.endsWith('/commit'))).toBe(false)
      return false
    })
    vi.stubGlobal('confirm', confirm)
    render(<AgentPanel {...props} />); await open(); send('生成平均价差')
    const save = await screen.findByRole('button', { name: '确认保存指标' })
    await waitFor(() => expect(save).toBeEnabled())
    fireEvent.click(save)
    await waitFor(() => expect(confirm).toHaveBeenCalledOnce())
    expect(confirm.mock.calls[0][0]).toContain(api.definition.name)
    expect(confirm.mock.calls[0][0]).toContain(api.definition.expression)
    expect(confirm.mock.calls[0][0]).toContain('同名指标')
    expect(confirm.mock.calls[0][0]).toContain('不会覆盖')
    expect(api.fetcher.mock.calls.filter(([url]) => url.endsWith('/commit'))).toHaveLength(0)
    await waitFor(() => expect(save).toBeEnabled())
    confirm.mockReturnValue(true)
    fireEvent.click(save)
    await screen.findByRole('button', { name: '已保存' })
    expect(confirm).toHaveBeenCalledTimes(2)
    expect(api.fetcher.mock.calls.filter(([url]) => url.endsWith('/commit-preview'))).toHaveLength(2)
    expect(api.fetcher.mock.calls.filter(([url]) => url.endsWith('/commit'))).toHaveLength(1)
  })

  it.each([false, true])('保存成功回执直接返回或恢复后，宿主目录只刷新一次，回执丢失=%s', async lost => {
    const api = server({ draft: true }), committed = vi.fn()
    let saved = false, writes = 0
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      const path = new URL(url, 'http://localhost').pathname
      if (path.endsWith('/commit')) {
        saved = true; writes++
        if (lost) throw new TypeError('保存成功但回执丢失')
      }
      const reply = await api.fetcher(url, init) as ReturnType<typeof response>
      if (path.endsWith('/sessions/s')) return response({ ...await reply.json(),
        saved_commit: saved ? { definition_hash: 'hash', indicator_id: 'i', revision: 1 } : null })
      return reply
    }))
    const view = render(<AgentPanel {...props} onCommitted={committed} />); await open(); send('生成平均价差')
    await waitFor(() => expect(screen.getByRole('button', { name: '确认保存指标' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: '确认保存指标' }))
    if (lost) expect(await screen.findByRole('alert')).toHaveTextContent('保存成功但回执丢失')
    else await waitFor(() => expect(committed).toHaveBeenCalledOnce())
    fireEvent.click(screen.getByRole('button', { name: '关闭 AI 助手' })); await open()
    expect(await screen.findByRole('button', { name: '已保存' })).toBeDisabled()
    await waitFor(() => expect(committed).toHaveBeenCalledOnce())
    await waitFor(() => expect(within(screen.getByRole('region', { name: '本轮指标草稿' })).queryByRole('alert')).toBeNull())
    // The real host creates a fresh callback on render; that must not cause a refresh loop.
    view.rerender(<AgentPanel {...props} onCommitted={() => committed()} />)
    fireEvent.click(screen.getByRole('button', { name: '关闭 AI 助手' })); await open()
    expect(await screen.findByRole('button', { name: '已保存' })).toBeDisabled()
    expect(committed).toHaveBeenCalledOnce()
    expect(writes).toBe(1)
    view.rerender(<AgentPanel {...props} onCommitted={() => committed()}
      onApplyDraft={() => { throw new Error('编辑器拒绝填入') }} />)
    fireEvent.click(screen.getByRole('button', { name: '填入编辑器' }))
    expect(await within(screen.getByRole('region', { name: '本轮指标草稿' })).findByRole('alert')).toHaveTextContent('编辑器拒绝填入')
  })

  it.each([false, true])('页面重挂从服务端回执恢复已保存状态，回执曾丢失=%s', async lost => {
    const api = server({ draft: true }), committed = vi.fn()
    let saved = false, writes = 0
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      const path = new URL(url, 'http://localhost').pathname
      if (path.endsWith('/commit')) {
        saved = true; writes++
        if (lost) throw new TypeError('保存回执连接中断')
      }
      const reply = await api.fetcher(url, init) as ReturnType<typeof response>
      if (path.endsWith('/sessions/s')) return response({ ...await reply.json(),
        saved_commit: saved ? { definition_hash: 'hash', indicator_id: 'saved-once', revision: 1 } : null })
      return reply
    }))
    const view = render(<AgentPanel {...props} onCommitted={committed} />); await open(); send('生成平均价差')
    await waitFor(() => expect(screen.getByRole('button', { name: '确认保存指标' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: '确认保存指标' }))
    if (lost) expect(await screen.findByRole('alert')).toHaveTextContent('保存回执连接中断')
    else await screen.findByRole('button', { name: '已保存' })
    view.unmount(); committed.mockClear(); render(<AgentPanel {...props} onCommitted={committed} />); await open()
    expect(await screen.findByRole('button', { name: '已保存' })).toBeDisabled()
    await waitFor(() => expect(committed).toHaveBeenCalledOnce())
    expect(writes).toBe(1)
    expect(api.requests).toHaveLength(1)
  })

  it.each([false, true])('保存回执丢失后重试复用原确认和幂等键，关闭重开=%s', async reopen => {
    const api = server({ draft: true }), requests: any[] = [], applied = new Set<string>()
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (new URL(url, 'http://localhost').pathname.endsWith('/commit')) {
        const body = JSON.parse(String(init?.body)); requests.push(body); applied.add(body.request_id)
        if (requests.length === 1) throw new TypeError('保存回执连接中断')
        return response({ indicator_id: 'saved-once', revision: 1 })
      }
      return api.fetcher(url, init)
    }))
    render(<AgentPanel {...props} />); await open(); send('生成平均价差')
    await waitFor(() => expect(screen.getByRole('button', { name: '确认保存指标' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: '确认保存指标' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('保存回执连接中断')
    if (reopen) { fireEvent.click(screen.getByRole('button', { name: '关闭 AI 助手' })); await open() }
    await waitFor(() => expect(screen.getByRole('button', { name: '确认保存指标' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: '确认保存指标' }))
    expect(await screen.findByRole('button', { name: '已保存' })).toBeDisabled()
    expect(requests).toHaveLength(2); expect(requests[1]).toEqual(requests[0]); expect(applied.size).toBe(1)
    expect(window.confirm).toHaveBeenCalledTimes(1)
    expect(api.fetcher.mock.calls.filter(([url]) => url.endsWith('/commit-preview'))).toHaveLength(1)
    fireEvent.click(screen.getByRole('button', { name: '关闭 AI 助手' })); await open()
    expect(await screen.findByRole('button', { name: '已保存' })).toBeDisabled()
  })

  it('服务端明确拒绝过期确认后，下次人工保存重新确认', async () => {
    const api = server({ draft: true }), requests: any[] = []
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      if (new URL(url, 'http://localhost').pathname.endsWith('/commit')) {
        requests.push(JSON.parse(String(init?.body)))
        if (requests.length === 1) return response({ detail: { code: 'AGENT_CONFIRMATION_STALE', message: '确认已过期' } }, 409)
      }
      return api.fetcher(url, init)
    }))
    render(<AgentPanel {...props} />); await open(); send('生成平均价差')
    await waitFor(() => expect(screen.getByRole('button', { name: '确认保存指标' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: '确认保存指标' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('确认已过期')
    await waitFor(() => expect(screen.getByRole('button', { name: '确认保存指标' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: '确认保存指标' }))
    await screen.findByRole('button', { name: '已保存' })
    expect(requests[0].request_id).not.toBe(requests[1].request_id)
    expect(api.fetcher.mock.calls.filter(([url]) => url.endsWith('/commit-preview'))).toHaveLength(2)
  })

  it('清空后丢弃旧消息、草稿与未发送输入，新会话不恢复旧运行', async () => {
    const api = server({ active: true, draft: true })
    render(<AgentPanel {...props} />); await open(); send('旧对话的秘密口径')
    await screen.findByRole('button', { name: '停止' })
    expect(screen.getByRole('button', { name: '清空上下文' })).toBeDisabled()
    await act(async () => { api.finish('旧对话已暂停', 'paused') })
    await screen.findByText('旧对话已暂停')
    expect(screen.getByRole('region', { name: '本轮指标草稿' })).toBeVisible()
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '未发送的旧想法' } })
    await waitFor(() => expect(screen.getByRole('button', { name: '清空上下文' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: '清空上下文' }))
    expect(screen.getByRole('dialog', { name: 'AI 助手' })).toBeVisible()
    expect(screen.getByRole('textbox')).toHaveValue('')
    expect(screen.queryByText('旧对话的秘密口径')).not.toBeInTheDocument()
    expect(screen.queryByText('旧对话已暂停')).not.toBeInTheDocument()
    expect(screen.queryByRole('region', { name: '本轮指标草稿' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '继续分析' })).not.toBeInTheDocument()
    expect(sessionStorage.getItem('agent-session:indicator-studio:single_product:test')).toBeNull()
    await waitFor(() => expect(screen.getByRole('textbox')).toBeEnabled())
    send('新研究需求')
    await waitFor(() => expect(api.requests).toHaveLength(2))
    expect(api.requests[1]).toMatchObject({ expected_session_revision: 0, text: '新研究需求' })
    expect(api.requests[1]).not.toHaveProperty('resume_from_run_id')
    expect(api.fetcher.mock.calls.some(([url]) => url.includes('/sessions/s2/messages'))).toBe(true)
    expect(sessionStorage.getItem('agent-session:indicator-studio:single_product:test')).toBe('s2')
  })

  it('浏览器清除失败时保留原对话和输入，允许重试', async () => {
    server(); render(<AgentPanel {...props} />); await open(); send('保留我的对话')
    await screen.findByText('已检查你的指标需求。')
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '保留我的输入' } })
    await waitFor(() => expect(screen.getByRole('button', { name: '清空上下文' })).toBeEnabled())
    const remove = vi.spyOn(Storage.prototype, 'removeItem').mockImplementationOnce(() => { throw new Error('unavailable') })
    try { fireEvent.click(screen.getByRole('button', { name: '清空上下文' })) } finally { remove.mockRestore() }
    expect(await screen.findByRole('alert')).toHaveTextContent('原对话已保留')
    expect(screen.getByText('保留我的对话')).toBeVisible()
    expect(screen.getByRole('textbox')).toHaveValue('保留我的输入')
    fireEvent.click(screen.getByRole('button', { name: '清空上下文' }))
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.queryByText('保留我的对话')).not.toBeInTheDocument()
  })

  it('立即显示消息和思考状态，传输失败保留原消息重试', async () => {
    const api = server({ defer: true }); render(<AgentPanel {...props} />); await open(); send('滚动夏普比率，窗口可变')
    expect(within(screen.getByRole('log')).getByText('滚动夏普比率，窗口可变')).toBeVisible()
    expect(screen.getByText('正在发送…')).toBeVisible(); expect(screen.getByRole('textbox')).toHaveValue('')
    await waitFor(() => expect(api.requests).toHaveLength(1))
    await act(async () => { api.release() })
    expect(await screen.findByRole('alert')).toHaveTextContent('连接暂时失败')
    fireEvent.click(screen.getByRole('button', { name: '重试这条消息' }))
    expect(await screen.findByText('已检查你的指标需求。')).toBeVisible()
    expect(api.requests).toHaveLength(2); expect(api.requests[0]).toEqual(api.requests[1])
    expect(within(screen.getByRole('log')).getAllByText('滚动夏普比率，窗口可变')).toHaveLength(1)
  })
  it('未配置 LLM 时保留小牛入口并禁用发送', async () => {
    server({ configured: false }); const { container } = render(<AgentPanel {...props} />)
    const launcher = screen.getByRole('button', { name: '打开 AI 助手' }); expect(launcher.querySelector('img')).toHaveAttribute('src', '/homepage/images/mascot-welcome-240.webp')
    fireEvent.click(launcher)
    expect(await screen.findByRole('link', { name: '前往 LLM API 配置' })).toHaveAttribute('href', '/settings/llm-api')
    expect(screen.getByRole('button', { name: '发送' })).toBeDisabled(); expect(container).not.toContainElement(screen.getByRole('dialog'))
    expect(document.querySelectorAll('img[src*="mascot-"]')).toHaveLength(1)
  })
  it('关闭和Esc收起，打开时小牛移入标题，重开保留输入并归还焦点', async () => {
    server(); render(<AgentPanel {...props} />); await open()
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '检查公式' } })
    fireEvent.click(screen.getByRole('button', { name: '关闭 AI 助手' }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('button', { name: '打开 AI 助手' })).toHaveFocus())
    await open(); expect(screen.getByRole('textbox')).toHaveValue('检查公式')
    fireEvent.keyDown(document, { key: 'Escape' }); expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    await open(); expect(screen.queryByRole('button', { name: '打开 AI 助手' })).not.toBeInTheDocument(); expect(screen.getByRole('dialog').querySelector('header img')).toHaveClass('h-8')
  })
  it('无产品可生成、填入和人工保存草稿，恢复顺序变化不误判上下文', async () => {
    const api = server({ draft: true }); const apply = vi.fn(), committed = vi.fn()
    render(<AgentPanel {...props} onApplyDraft={apply} onCommitted={committed} />); await open(); send('生成平均价差')
    await screen.findByText(api.definition.expression)
    fireEvent.click(screen.getByRole('button', { name: '填入编辑器' })); expect(apply).toHaveBeenCalledWith(api.definition)
    expect(api.requests[0].page_context).toMatchObject({ calculation: { targets: [] } })
    expect(api.fetcher.mock.calls.some(([url]) => url.endsWith('/commit'))).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: '确认保存指标' }))
    await waitFor(() => expect(committed).toHaveBeenCalledOnce())
    const preview = api.fetcher.mock.calls.find(([url]) => url.endsWith('/commit-preview'))
    expect(JSON.parse(String(preview?.[1]?.body))).not.toHaveProperty('target')
    expect(api.fetcher.mock.calls.some(([url]) => url.endsWith('/invalidate-context'))).toBe(false)
  })
  it('关闭不取消，运行中可输入，停止后不清掉下一条输入', async () => {
    const api = server({ active: true }); render(<AgentPanel {...props} />); await open(); send('检查指标')
    await screen.findByRole('button', { name: '停止' })
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '下一条需求' } })
    fireEvent.click(screen.getByRole('button', { name: '关闭 AI 助手' })); await open()
    expect(api.fetcher.mock.calls.some(([url]) => url.endsWith('/cancel'))).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: '停止' }))
    expect(await screen.findByText('已停止，进度已保留。')).toBeVisible()
    expect(screen.getByRole('textbox')).toHaveValue('下一条需求')
    expect(api.requests).toHaveLength(1)
  })
  it.each(['', 'outdated_expression'])('时序草稿展示所有通道公式，不依赖顶层 expression=%s，填入保存保留完整定义', async expression => {
    const api = server({ draft: true }); const apply = vi.fn(), committed = vi.fn()
    const outputs = [
      { id: 'direction', label: '动量方向', expression: 'sign(mean(rolling_window(log_returns, momentum_window)))' },
      { id: 'strength', label: '动量强度', expression: 'absolute(mean(rolling_window(log_returns, momentum_window)))' },
    ]
    Object.assign(api.definition, { name: 'ETF 量价动量', result_kind: 'time_series', expression, series_outputs: outputs })
    render(<AgentPanel {...props} onApplyDraft={apply} onCommitted={committed} />); await open(); send('生成动量方向和强度')
    const card = await screen.findByRole('region', { name: '本轮指标草稿' })
    for (const output of outputs) expect(card).toHaveTextContent(`${output.label} = ${output.expression}`)
    expect(card).not.toHaveTextContent('等待公式')
    expect(card).not.toHaveTextContent('outdated_expression')
    fireEvent.click(within(card).getByRole('button', { name: '填入编辑器' }))
    expect(apply).toHaveBeenCalledWith(api.definition)
    fireEvent.click(within(card).getByRole('button', { name: '确认保存指标' }))
    await waitFor(() => expect(committed).toHaveBeenCalledOnce())
    const preview = api.fetcher.mock.calls.find(([url]) => url.endsWith('/commit-preview'))
    expect(JSON.parse(String(preview?.[1]?.body)).definition).toEqual(api.definition)
    for (const output of outputs) expect(vi.mocked(window.confirm).mock.calls[0][0]).toContain(`${output.label} = ${output.expression}`)
  })
  it('重新挂载恢复消息，不重新发送或保存研究数据到浏览器', async () => {
    const api = server(); const view = render(<AgentPanel {...props} />); await open(); send('解释指标')
    await screen.findByText('已检查你的指标需求。'); view.unmount()
    expect(sessionStorage.getItem('agent-session:indicator-studio:single_product:test')).toBe('s')
    render(<AgentPanel {...props} />); await open()
    expect(await screen.findByText('解释指标')).toBeVisible()
    expect(await screen.findByText('已检查你的指标需求。')).toBeVisible(); expect(api.requests).toHaveLength(1)
  })
  it('事件重复不会重复消息，缺口由分页补齐', async () => {
    const api = server({ active: true })
    const sources: FakeSource[] = []
    class FakeSource {
      onopen: (() => void) | null = null; onerror: (() => void) | null = null
      listener: ((event: MessageEvent) => void) | null = null
      constructor() { sources.push(this) }
      addEventListener(_name: string, fn: (event: MessageEvent) => void) { this.listener = fn }
      close() {}
      emit(event: AgentEvent) { this.listener?.({ data: JSON.stringify(event) } as MessageEvent) }
    }
    vi.stubGlobal('EventSource', FakeSource)
    render(<AgentPanel {...props} />); await open(); send('研究需求')
    await waitFor(() => expect(sources.length).toBeGreaterThan(0))
    const completed = api.finish()
    await act(async () => { sources[sources.length - 1]?.emit(completed[1]); sources[sources.length - 1]?.emit(completed[0]) })
    expect(await screen.findByText('已检查你的指标需求。')).toBeVisible()
    expect(screen.getAllByText('已检查你的指标需求。')).toHaveLength(1); expect(api.requests).toHaveLength(1)
  })
  it('停止并发送使用新消息与最新会话版本，不重放上一轮', async () => {
    const api = server({ active: true }); render(<AgentPanel {...props} />); await open(); send('第一条研究需求')
    await screen.findByRole('button', { name: '停止' })
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '补充计算口径' } })
    fireEvent.click(screen.getByRole('button', { name: '停止并发送' }))
    await waitFor(() => expect(api.requests).toHaveLength(2))
    expect(api.requests[1].expected_session_revision).toBe(1)
    expect(api.requests[1].message_id).not.toBe(api.requests[0].message_id)
    expect(api.requests[1].resume_from_run_id).toBe('r1')
    expect(within(screen.getByRole('log')).getAllByText('补充计算口径')).toHaveLength(1)
  })
  it('第二条消息尚未收到运行回执时，发送状态在新消息后而非旧回复中', async () => {
    const api = server(); render(<AgentPanel {...props} />); await open(); send('第一条')
    await screen.findByText('已检查你的指标需求。')
    const original = api.fetcher.getMockImplementation()!
    let release: (() => void) | undefined
    api.fetcher.mockImplementation(async (url, init) => {
      if (url.endsWith('/messages?response_mode=async')) await new Promise<void>(resolve => { release = resolve })
      return original(url, init)
    })
    send('第二条')
    const status = await screen.findByText('正在发送…')
    const oldReply = document.querySelector('[data-run-id="r1"]') as HTMLElement
    expect(oldReply).not.toContainElement(status)
    expect(within(oldReply).getByRole('button', { name: '复制这条消息' })).toBeVisible()
    expect(screen.getByText('第二条').compareDocumentPosition(status) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    await waitFor(() => expect(release).toBeDefined())
    await act(async () => { release!() })
    await waitFor(() => expect(api.requests).toHaveLength(2))
  })
  it('复制多通道原文、折叠公式与关闭重开不更改定义，保存回执不会被填入动作清掉', async () => {
    const api = server({ draft: true }), writeText = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { clipboard: { writeText }, language: 'zh-CN' })
    const outputs = [{ id: 'one', label: '方向', expression: 'sign(returns)' }, { id: 'two', label: '强度', expression: 'absolute(returns)' }]
    Object.assign(api.definition, { result_kind: 'time_series', expression: '', series_outputs: outputs })
    render(<AgentPanel {...props} onApplyDraft={vi.fn()} />); await open(); send('多通道')
    const card = await screen.findByRole('region', { name: '本轮指标草稿' })
    const disclosure = within(card).getByRole('button', { name: '查看完整公式（2 个输出）' })
    expect(disclosure).toHaveAttribute('aria-expanded', 'false')
    fireEvent.click(disclosure)
    fireEvent.click(within(card).getByRole('button', { name: '复制公式' }))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith('方向 = sign(returns)\n\n强度 = absolute(returns)'))
    fireEvent.click(screen.getByRole('button', { name: '关闭 AI 助手' })); await open()
    expect(disclosure).toHaveAttribute('aria-expanded', 'true')
    fireEvent.click(within(card).getByRole('button', { name: '确认保存指标' }))
    await within(card).findByText('指标已保存（v1）。')
    fireEvent.click(within(card).getByRole('button', { name: '填入编辑器' }))
    expect(within(card).getByRole('button', { name: '已保存' })).toBeDisabled()
    expect(api.fetcher.mock.calls.filter(([url]) => url.endsWith('/commit'))).toHaveLength(1)
  })
  it('切换到较低指标 revision 仍可使旧运行失效', async () => {
    const api = server({ active: true })
    const view = render(<AgentPanel {...props} pageContext={{ ...props.pageContext, context_revision: 5 }} />)
    await open(); send('第一条研究需求'); await screen.findByRole('button', { name: '停止' })
    view.rerender(<AgentPanel {...props} />)
    await screen.findByText('口径已变化。')
    const call = api.fetcher.mock.calls.find(([url]) => url.endsWith('/invalidate-context'))
    expect(JSON.parse(String(call?.[1]?.body)).page_context.context_revision).toBeGreaterThan(5)
    expect(api.requests).toHaveLength(1)
  })

  it('条件提示按实际差异显示，相同条件重排和恢复原条件不留下警告', async () => {
    const api = server()
    const view = render(<AgentPanel {...props} />)
    await open(); send('讨论研究需求'); await screen.findByText('已检查你的指标需求。')
    const notice = '页面研究条件已变化，下一条消息将使用新的上下文；历史试算保留原条件。'
    view.rerender(<AgentPanel {...props} pageContext={{ ...props.pageContext,
      calculation: { as_of: null, period: '1Y', targets: [], context_kind: 'single_product' } }} />)
    expect(screen.queryByText(notice)).not.toBeInTheDocument()
    view.rerender(<AgentPanel {...props} pageContext={{ ...props.pageContext,
      calculation: { ...props.pageContext.calculation, period: '3Y' } }} />)
    expect(screen.getByText(notice)).toBeVisible()
    view.rerender(<AgentPanel {...props} />)
    expect(screen.queryByText(notice)).not.toBeInTheDocument()
    expect(api.fetcher.mock.calls.some(([url]) => url.endsWith('/invalidate-context'))).toBe(false)
  })

  it('按需加载更早消息，保持顺序且不重放工具日志', async () => {
    sessionStorage.setItem('agent-session:indicator-studio:single_product:test', 's')
    const recent = [{ seq: 201, id: 'u2', speaker: 'user', text: '新需求' }, { seq: 203, id: 'a2', speaker: 'assistant', text: '新回复' }]
    const fetcher = vi.fn(async (url: string) => {
      if (url.endsWith('/meta')) return response({ configured: true })
      if (url.endsWith('/sessions/s')) return response({ session_id: 's', session_revision: 2, page_context: props.pageContext, messages: recent, older_message_cursor: 201, next_event_seq: 205 })
      if (url.includes('kind=messages')) return response({ items: [{ seq: 1, id: 'u1', speaker: 'user', text: '旧需求' }, { seq: 3, id: 'a1', speaker: 'assistant', text: '旧回复' }], older_cursor: null })
      if (url.includes('/events?')) return response({ items: [], has_more: false, next_event_seq: 205 })
      throw new Error('unexpected request')
    })
    vi.stubGlobal('fetch', fetcher); vi.stubGlobal('EventSource', undefined)
    render(<AgentPanel {...props} />); await open()
    fireEvent.click(await screen.findByRole('button', { name: '查看更早消息' }))
    await screen.findByText('旧需求')
    expect(within(screen.getByRole('log')).getAllByText(/旧需求|旧回复|新需求|新回复/).map(el => el.textContent)).toEqual(['旧需求', '旧回复', '新需求', '新回复'])
    expect(fetcher.mock.calls.some(([url]) => url.includes('after_seq=204'))).toBe(true)
    expect(screen.queryByRole('button', { name: '查看更早消息' })).not.toBeInTheDocument()
  })

  it('排队消息沿用排队时捕获的页面快照，新消息重新捕获', async () => {
    const api = server({ active: true })
    let marker = 'first', captured = 0
    const capture = vi.fn(() => evidence(marker, captured++))
    render(<AgentPanel {...props} capturePageSnapshot={capture} />); await open()
    send('第一条问题')
    await waitFor(() => expect(api.requests).toHaveLength(1))
    expect(api.requests[0].page_snapshot).toMatchObject({ snapshot_id: `snap-${'0'.repeat(32)}`, sections: { editing: { marker: 'first' } } })

    marker = 'queued'
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '第二条问题' } })
    fireEvent.click(screen.getByRole('button', { name: '停止并发送' }))
    // 排队之后页面又变了：这不能改写已经冻结的排队证据。
    marker = 'after-queue'
    await waitFor(() => expect(api.requests).toHaveLength(2))
    expect(capture).toHaveBeenCalledTimes(2)
    expect(api.requests[1].page_snapshot).toMatchObject({ snapshot_id: `snap-${'0'.repeat(31)}1`, sections: { editing: { marker: 'queued' } } })
  })

  it('传输失败重试沿用同一份页面快照，不重新读取页面', async () => {
    const api = server({ defer: true })
    let captured = 0
    const capture = vi.fn(() => evidence('frozen', captured++))
    render(<AgentPanel {...props} capturePageSnapshot={capture} />); await open(); send('这个指标为什么是 0')
    await waitFor(() => expect(api.requests).toHaveLength(1))
    await act(async () => { api.release() })
    fireEvent.click(await screen.findByRole('button', { name: '重试这条消息' }))
    await waitFor(() => expect(api.requests).toHaveLength(2))
    expect(api.requests[1].page_snapshot).toEqual(api.requests[0].page_snapshot)
    expect(capture).toHaveBeenCalledTimes(1)
  })

  it('修改消息在编辑时重新捕获快照，并用新快照替换旧证据', async () => {
    const api = server({ active: true })
    let marker = 'before-edit', captured = 0
    const capture = vi.fn(() => evidence(marker, captured++))
    render(<AgentPanel {...props} capturePageSnapshot={capture} />); await open(); send('原问题')
    await screen.findByRole('button', { name: '停止' })
    fireEvent.click(screen.getByRole('button', { name: '停止' }))
    await screen.findByText('已停止，进度已保留。')
    marker = 'at-edit'
    fireEvent.click(screen.getByRole('button', { name: '修改这条消息' }))
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '改写后的问题' } })
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
    await waitFor(() => expect(api.requests).toHaveLength(2))
    expect(api.requests[1].edit_of_message_id).toBe(api.requests[0].message_id)
    expect(api.requests[1].page_snapshot).toMatchObject({ sections: { editing: { marker: 'at-edit' } } })
    expect(capture).toHaveBeenCalledTimes(2)
  })

  it('冻结失败时重试仍发送同一份副本，绝不把可变页面对象当作快照', async () => {
    const api = server({ defer: true })
    const live: any = evidence('live', 0)
    const original = globalThis.structuredClone
    vi.stubGlobal('structuredClone', (value: any) => {
      if (value && typeof value === 'object' && 'snapshot_id' in value) throw new Error('cannot freeze')
      return original(value)
    })
    render(<AgentPanel {...props} capturePageSnapshot={() => live} />); await open(); send('为什么是 0')
    await waitFor(() => expect(api.requests).toHaveLength(1))
    live.sections.editing.marker = 'mutated'
    await act(async () => { api.release() })
    fireEvent.click(await screen.findByRole('button', { name: '重试这条消息' }))
    await waitFor(() => expect(api.requests).toHaveLength(2))
    expect(api.requests[0].page_snapshot).toMatchObject({ sections: { editing: { marker: 'live' } } })
    expect(api.requests[1].page_snapshot).toEqual(api.requests[0].page_snapshot)
  })

  it('无法安全序列化时显式声明冻结失败，消息文本仍正常送达', async () => {
    const api = server()
    const live: any = evidence('live', 0)
    live.sections.editing.self = live
    const original = globalThis.structuredClone
    vi.stubGlobal('structuredClone', (value: any) => {
      if (value && typeof value === 'object' && 'snapshot_id' in value) throw new Error('cannot freeze')
      return original(value)
    })
    render(<AgentPanel {...props} capturePageSnapshot={() => live} />); await open(); send('文本仍要送达')
    await waitFor(() => expect(api.requests).toHaveLength(1))
    expect(api.requests[0].text).toBe('文本仍要送达')
    expect(api.requests[0].page_snapshot).toMatchObject({ sections: { editing: { omitted: { code: 'page_evidence_freeze_failed' } } } })
    expect(api.requests[0].page_snapshot).toMatchObject({ snapshot_id: expect.stringMatching(/^snap-[0-9a-f]{32}$/) })
  })

})

it('上下文变更后丢弃迟到的试算数据，不覆盖当前页面', async () => {
  const api = server({ draft: true })
  const original = api.fetcher.getMockImplementation()!
  let resolvePreview: ((value: unknown) => void) | undefined
  const reference = { preview_id: 'p1', definition_hash: 'hash', result_kind: 'scalar', target: { kind: 'etf', product_id: '510300.SH' }, period: '1Y' }
  api.fetcher.mockImplementation(async (url, init) => {
    if (url.includes('/previews/')) return new Promise(resolve => { resolvePreview = resolve })
    const received = await original(url, init) as { json: () => Promise<Record<string, any>> }
    const data = await received.json()
    if (data.response) data.response.preview = reference
    if (url.endsWith('/sessions/s')) data.preview = reference
    return response(data)
  })
  const onPreview = vi.fn()
  const view = render(<AgentPanel {...props} onPreview={onPreview} />)
  await open(); send('用一个产品展示指标')
  await screen.findByText('正在加载试算结果…')
  await waitFor(() => expect(resolvePreview).toBeDefined())
  view.rerender(<AgentPanel {...props} pageContext={{ ...props.pageContext, calculation: { ...props.pageContext.calculation, period: '3Y' } }} onPreview={onPreview} />)
  await act(async () => { resolvePreview?.(response({ ...reference, session_id: 's', definition: api.definition, result: { results: [{ value: 123 }] } })) })
  expect(onPreview.mock.calls.every(([value]) => value === null)).toBe(true)
  expect(screen.queryByRole('button', { name: '查看试算结果' })).not.toBeInTheDocument()
})

it('旧会话只有试算元数据时明确要求重新试算', async () => {
  const api = server({ draft: true })
  const original = api.fetcher.getMockImplementation()!
  api.fetcher.mockImplementation(async (url, init) => {
    const received = await original(url, init) as { json: () => Promise<Record<string, any>> }
    const data = await received.json()
    const old = { definition_hash: 'hash', target: { kind: 'etf', product_id: '510300.SH' }, period: '1Y' }
    if (url.endsWith('/sessions/s')) data.preview = old
    return response(data)
  })
  render(<AgentPanel {...props} />); await open(); send('展示旧结果')
  expect(await screen.findByRole('alert')).toHaveTextContent('旧试算未保存完整结果')
  expect(screen.queryByRole('button', { name: '查看试算结果' })).not.toBeInTheDocument()
})


it('草稿归属原回复，讨论新逻辑不把旧草稿挂到新回复，重开仍保持归属', async () => {
  server({ draft: true }); const apply = vi.fn()
  render(<AgentPanel {...props} onApplyDraft={apply} />); await open(); send('设计第一项指标')
  await screen.findByRole('button', { name: '确认保存指标' })
  send('现在讨论另一个逻辑')
  await waitFor(() => expect(document.querySelector('[data-run-id="r2"]')).not.toBeNull())
  const first = document.querySelector('[data-run-id="r1"]') as HTMLElement
  const second = document.querySelector('[data-run-id="r2"]') as HTMLElement
  expect(within(first).getByRole('region', { name: '本轮指标草稿' })).toBeVisible()
  expect(within(second).queryByRole('region', { name: '本轮指标草稿' })).toBeNull()
  expect(screen.getAllByRole('region', { name: '本轮指标草稿' })).toHaveLength(1)
  expect(screen.queryByRole('button', { name: '确认保存指标' })).toBeNull()
  fireEvent.click(within(first).getByRole('button', { name: '填入编辑器' }))
  expect(apply).toHaveBeenCalledWith(expect.objectContaining({ name: '价差均值' }))
  fireEvent.click(screen.getByRole('button', { name: '关闭 AI 助手' })); await open()
  expect(within(document.querySelector('[data-run-id="r2"]') as HTMLElement).queryByRole('region', { name: '本轮指标草稿' })).toBeNull()
})
