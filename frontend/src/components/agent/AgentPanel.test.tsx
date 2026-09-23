import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import AgentPanel from './AgentPanel'
import { agentSessionStorageKey } from '../../services/agentContext'
import type { AgentPageContext } from '../../services/agent'

const page = (id: string): AgentPageContext => ({ page: 'product-detail', page_instance_id: id, context_revision: 0, view_state: 'inherit', calculation: { context_kind: 'single_product', targets: [], period: '1Y' } })
const response = (data: unknown) => ({ ok: true, status: 200, json: async () => data })
afterEach(() => { cleanup(); sessionStorage.clear(); vi.unstubAllGlobals() })

it('普通研究页无需指标组件即可聊天，切换实例后迟到结果不能进入新会话', async () => {
  let release: (() => void) | undefined
  const pending = new Promise<void>(resolve => { release = resolve })
  const states = new Map<string, any>(), requests: Array<{ sid: string; body: any }> = []
  vi.stubGlobal('EventSource', undefined)
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    const path = new URL(url,'http://localhost').pathname
    if (path.endsWith('/meta')) return response({ configured: true })
    if (path.endsWith('/sessions')) {
      const context = JSON.parse(String(init?.body)).page_context, sid = `session-${context.page_instance_id}`
      const state = { session_id: sid, session_revision: 0, page_context: context, messages: [], next_event_seq: 1 }
      states.set(sid,state); return response(state)
    }
    const sid = path.split('/')[4], state = states.get(sid)
    if (path.endsWith('/messages')) {
      const body = JSON.parse(String(init?.body)); requests.push({ sid,body })
      if (sid === 'session-A') await pending
      const run = { session_id: sid, run_id: `${sid}-run`, message_id: body.message_id, run_revision: 1, session_revision: 1, status: 'completed', phase: 'thinking', response: { session_id: sid, session_revision: 1, reply: { text: `${sid}的回复` } } }
      state.session_revision = 1; state.active_run = run; return response(run)
    }
    if (path.endsWith('/events')) return response({ items: [], has_more: false })
    if (path.includes('/runs/')) return response(state.active_run)
    return response(state)
  }))
  const view = render(<AgentPanel pageContext={page('A')} />)
  fireEvent.click(screen.getByRole('button',{name:'打开 AI 助手'}))
  await waitFor(() => expect(screen.getByRole('textbox')).toBeEnabled())
  expect(screen.queryByRole('button',{name:'设计指标'})).not.toBeInTheDocument()
  fireEvent.change(screen.getByRole('textbox'),{target:{value:'A 的需求'}})
  fireEvent.click(screen.getByRole('button',{name:'发送'}))
  await waitFor(() => expect(requests).toHaveLength(1))
  view.rerender(<AgentPanel pageContext={page('B')} />)
  fireEvent.click(screen.getByRole('button',{name:'打开 AI 助手'}))
  await waitFor(() => expect(screen.getByRole('textbox')).toBeEnabled())
  expect(screen.queryByText('A 的需求')).not.toBeInTheDocument()
  fireEvent.change(screen.getByRole('textbox'),{target:{value:'B 的需求'}})
  fireEvent.click(screen.getByRole('button',{name:'发送'}))
  expect(await screen.findByText('session-B的回复')).toBeVisible()
  await act(async () => release?.())
  expect(screen.queryByText('session-A的回复')).not.toBeInTheDocument()
  expect(requests[1]).toMatchObject({sid:'session-B',body:{expected_session_revision:0}})
  expect(requests[1].body).not.toHaveProperty('resume_from_run_id')
  expect(sessionStorage.getItem(agentSessionStorageKey(page('A')))).toBe('session-A')
  expect(sessionStorage.getItem(agentSessionStorageKey(page('B')))).toBe('session-B')
  expect(screen.queryByRole('button',{name:'确认保存指标'})).not.toBeInTheDocument()
})

it('所有消息可复制，停止且无回复的最后一条用户消息可修改并重新发送', async () => {
  const context = page('E'), storage = agentSessionStorageKey(context)
  sessionStorage.setItem(storage, 'session-E')
  let messages: any[] = [
    { seq: 1, id: 'q-1', run_id: 'run-1', speaker: 'user', text: '第一轮问题' },
    { seq: 2, id: 'run-1-reply', run_id: 'run-1', speaker: 'assistant', text: '第一轮回复' },
    { seq: 3, id: 'q-2', run_id: 'run-2', speaker: 'user', text: '被停止的问题' },
    { seq: 4, id: 'run-2-reply', run_id: 'run-2', speaker: 'assistant', text: '已停止自动处理，已提交的进度已保留。' },
  ]
  const stoppedRun = { run_id: 'run-2', session_id: 'session-E', message_id: 'q-2', session_revision: 2, run_revision: 2, status: 'cancelled', phase: 'thinking',
    response: { session_id: 'session-E', session_revision: 2, reply: { text: '已停止自动处理，已提交的进度已保留。' }, artifacts: {} } }
  const sent: any[] = []
  vi.stubGlobal('EventSource', undefined)
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    const path = new URL(url, 'http://localhost').pathname
    if (path.endsWith('/meta')) return response({ configured: true })
    if (path.endsWith('/messages')) {
      const body = JSON.parse(String(init?.body)); sent.push(body)
      messages = [...messages.filter(item => item.id !== body.edit_of_message_id && item.run_id !== 'run-2'),
        { seq: 5, id: body.message_id, run_id: 'run-3', speaker: 'user', text: body.text },
        { seq: 6, id: 'run-3-reply', run_id: 'run-3', speaker: 'assistant', text: '改写后的回复' }]
      return response({ run_id: 'run-3', session_id: 'session-E', message_id: body.message_id, run_revision: 1, session_revision: 3, status: 'completed', phase: 'thinking',
        response: { session_id: 'session-E', session_revision: 3, reply: { text: '改写后的回复' }, artifacts: {} } })
    }
    if (path.includes('/runs/')) return response(stoppedRun)
    if (path.endsWith('/events')) return response({ items: [], has_more: false })
    return response({ session_id: 'session-E', session_revision: 2, page_context: context, messages, next_event_seq: 5, active_run: stoppedRun })
  }))
  render(<AgentPanel pageContext={context} />)
  fireEvent.click(screen.getByRole('button', { name: '打开 AI 助手' }))
  await waitFor(() => expect(screen.getByText('被停止的问题')).toBeVisible())
  expect(screen.getAllByRole('button', { name: '复制这条消息' })).toHaveLength(4)
  const input = screen.getByRole('textbox')
  fireEvent.change(input, { target: { value: '尚未发送的草稿' } })
  fireEvent.click(screen.getByRole('button', { name: '修改这条消息' }))
  expect(screen.getAllByText('编辑消息').length).toBeGreaterThan(0)
  expect(screen.getByRole('button', { name: '取消修改' })).toBeVisible()
  expect(input).toHaveValue('被停止的问题')
  fireEvent.click(screen.getByRole('button', { name: '取消修改' }))
  expect(input).toHaveValue('尚未发送的草稿')
  expect(screen.queryByRole('button', { name: '取消修改' })).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '修改这条消息' }))
  fireEvent.change(input, { target: { value: '改写后的问题' } })
  fireEvent.click(screen.getByRole('button', { name: '发送' }))
  await waitFor(() => expect(sent).toHaveLength(1))
  expect(sent[0]).toMatchObject({ text: '改写后的问题', edit_of_message_id: 'q-2', expected_session_revision: 2 })
  expect(sent[0]).not.toHaveProperty('resume_from_run_id')
  await waitFor(() => expect(screen.queryByText('被停止的问题')).not.toBeInTheDocument())
  expect(screen.getByText('改写后的问题')).toBeVisible()
  expect(screen.queryByText('已停止自动处理，已提交的进度已保留。')).not.toBeInTheDocument()
  expect(input).toHaveValue('')
  expect(screen.queryByRole('button', { name: '取消修改' })).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '修改这条消息' })).not.toBeInTheDocument()
})

it('已获回复或仍在运行的一轮没有修改入口', async () => {
  const context = page('F'), storage = agentSessionStorageKey(context)
  sessionStorage.setItem(storage, 'session-F')
  vi.stubGlobal('EventSource', undefined)
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    const path = new URL(url, 'http://localhost').pathname
    if (path.endsWith('/meta')) return response({ configured: true })
    if (path.endsWith('/events')) return response({ items: [], has_more: false })
    return response({ session_id: 'session-F', session_revision: 1, page_context: context, next_event_seq: 3, active_run: {
      run_id: 'run-1', session_id: 'session-F', message_id: 'q-1', session_revision: 1, run_revision: 1, status: 'running', phase: 'thinking' },
      messages: [{ seq: 1, id: 'q-1', run_id: 'run-1', speaker: 'user', text: '正在处理的问题' }] })
  }))
  const view = render(<AgentPanel pageContext={context} />)
  fireEvent.click(screen.getByRole('button', { name: '打开 AI 助手' }))
  await waitFor(() => expect(screen.getByText('正在处理的问题')).toBeVisible())
  expect(screen.queryByRole('button', { name: '修改这条消息' })).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: '复制这条消息' })).toBeVisible()
  view.unmount()
  sessionStorage.setItem(storage, 'session-F')
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    const path = new URL(url, 'http://localhost').pathname
    if (path.endsWith('/meta')) return response({ configured: true })
    if (path.endsWith('/events')) return response({ items: [], has_more: false })
    return response({ session_id: 'session-F', session_revision: 2, page_context: context, next_event_seq: 3, active_run: {
      run_id: 'run-1', session_id: 'session-F', message_id: 'q-1', session_revision: 2, run_revision: 3, status: 'completed', phase: 'thinking',
      response: { session_id: 'session-F', session_revision: 2, reply: { text: '真正的回复' }, artifacts: { draft: { valid: true } } } },
      messages: [{ seq: 1, id: 'q-1', run_id: 'run-1', speaker: 'user', text: '已回答的问题' },
        { seq: 2, id: 'run-1-reply', run_id: 'run-1', speaker: 'assistant', text: '真正的回复', artifacts: { draft: { valid: true } } }] })
  }))
  render(<AgentPanel pageContext={context} />)
  fireEvent.click(screen.getByRole('button', { name: '打开 AI 助手' }))
  await waitFor(() => expect(screen.getByText('已回答的问题')).toBeVisible())
  expect(screen.queryByRole('button', { name: '修改这条消息' })).not.toBeInTheDocument()
})

it('缓存指向其他页面实例时不显示其历史，也不恢复运行', async () => {
  sessionStorage.setItem(agentSessionStorageKey(page('B')),'session-A')
  const fetcher = vi.fn(async (url: string) => response(url.endsWith('/meta') ? { configured:true } : {
    session_id:'session-A',session_revision:1,page_context:page('A'),messages:[{id:'old',speaker:'assistant',text:'其他产品的历史'}],next_event_seq:2,
  }))
  vi.stubGlobal('fetch',fetcher)
  render(<AgentPanel pageContext={page('B')} />)
  fireEvent.click(screen.getByRole('button',{name:'打开 AI 助手'}))
  await waitFor(() => expect(sessionStorage.getItem(agentSessionStorageKey(page('B')))).toBeNull())
  expect(screen.queryByText('其他产品的历史')).not.toBeInTheDocument()
  expect(fetcher.mock.calls.some(([url]) => url.includes('/runs/') || url.includes('/events'))).toBe(false)
})

function editPanel(options: { fail?: boolean; delay?: boolean; loseReply?: boolean } = {}) {
  const context = page('G'), storage = agentSessionStorageKey(context)
  sessionStorage.setItem(storage, 'session-G')
  const sent: any[] = []
  let release: (() => void) | undefined
  const gate = new Promise<void>(resolve => { release = resolve })
  const stopNotice = '已停止自动处理，已提交的进度已保留。'
  const stoppedRun = { run_id: 'run-2', session_id: 'session-G', message_id: 'q-2', session_revision: 2, run_revision: 2, status: 'cancelled', phase: 'thinking',
    response: { session_id: 'session-G', session_revision: 2, reply: { text: stopNotice }, artifacts: {} } }
  let acceptedRun: typeof stoppedRun | null = null
  vi.stubGlobal('EventSource', undefined)
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    const path = new URL(url, 'http://localhost').pathname
    if (path.endsWith('/meta')) return response({ configured: true })
    if (path.endsWith('/messages')) {
      const body = JSON.parse(String(init?.body)); sent.push(body)
      if (options.delay) await gate
      // Only the first attempt fails, so a retry can prove the accepted request and its cleanup.
      if (options.fail && sent.length === 1) return { ok: false, status: 502, json: async () => ({ detail: { message: '模型接口响应超时，请重试。' } }) }
      acceptedRun = { run_id: 'run-3', session_id: 'session-G', message_id: body.message_id, run_revision: 1, session_revision: 3, status: 'completed', phase: 'thinking',
        response: { session_id: 'session-G', session_revision: 3, reply: { text: '改写后的回复' }, artifacts: {} } }
      if (options.loseReply && sent.length === 1) throw new TypeError('response lost')
      return response(acceptedRun)
    }
    if (options.loseReply && acceptedRun && path.includes('/runs/')) return response(acceptedRun)
    if (path.includes('/runs/')) return response(stoppedRun)
    if (path.endsWith('/events')) return response({ items: [], has_more: false })
    if (options.loseReply && acceptedRun) return response({ session_id: 'session-G', session_revision: 3, page_context: context,
      next_event_seq: 6, active_run: acceptedRun,
      messages: [{ seq: 5, id: acceptedRun.message_id, run_id: acceptedRun.run_id, speaker: 'user', text: '改写后的问题', edit_of: 'q-2' }] })
    return response({ session_id: 'session-G', session_revision: 2, page_context: context, next_event_seq: 5, active_run: stoppedRun,
      messages: [{ seq: 1, id: 'q-1', run_id: 'run-1', speaker: 'user', text: '第一轮问题' },
        { seq: 2, id: 'run-1-reply', run_id: 'run-1', speaker: 'assistant', text: '第一轮回复' },
        { seq: 3, id: 'q-2', run_id: 'run-2', speaker: 'user', text: '旧问题' },
        { seq: 4, id: 'run-2-reply', run_id: 'run-2', speaker: 'assistant', text: stopNotice }] })
  }))
  return { sent, release: () => release?.() }
}

it('修改发送失败后重试沿用同一消息标识，成功后退出编辑状态', async () => {
  const api = editPanel({ fail: true })
  render(<AgentPanel pageContext={page('G')} />)
  fireEvent.click(screen.getByRole('button', { name: '打开 AI 助手' }))
  await waitFor(() => expect(screen.getByText('旧问题')).toBeVisible())
  const input = screen.getByRole('textbox')
  fireEvent.click(screen.getByRole('button', { name: '修改这条消息' }))
  fireEvent.change(input, { target: { value: '改写后的问题' } })
  fireEvent.click(screen.getByRole('button', { name: '发送' }))
  await waitFor(() => expect(api.sent).toHaveLength(1))
  expect(await screen.findByRole('alert')).toHaveTextContent('模型接口响应超时')
  expect(input).toHaveValue('改写后的问题')
  expect(screen.getByRole('button', { name: '取消修改' })).toBeVisible()
  expect(screen.getByText('旧问题')).toBeVisible()
  expect(screen.getByText('已停止自动处理，已提交的进度已保留。')).toBeVisible()
  fireEvent.click(screen.getByRole('button', { name: '重试这条消息' }))
  await waitFor(() => expect(api.sent).toHaveLength(2))
  expect(api.sent[1]).toMatchObject({ message_id: api.sent[0].message_id, text: '改写后的问题', edit_of_message_id: 'q-2' })
  await waitFor(() => expect(screen.queryByRole('button', { name: '取消修改' })).not.toBeInTheDocument())
  expect(input).toHaveValue('')
  expect(screen.getByText('改写后的问题')).toBeVisible()
  expect(screen.queryByText('旧问题')).not.toBeInTheDocument()
})

it('修改已接受但回执丢失，恢复新运行后仍能结束原编辑重试', async () => {
  const api = editPanel({ loseReply: true })
  render(<AgentPanel pageContext={page('G')} />)
  fireEvent.click(screen.getByRole('button', { name: '打开 AI 助手' }))
  await screen.findByText('旧问题')
  fireEvent.click(screen.getByRole('button', { name: '修改这条消息' }))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '改写后的问题' } })
  fireEvent.click(screen.getByRole('button', { name: '发送' }))
  await screen.findByText('response lost')
  fireEvent.click(screen.getByRole('button', { name: '关闭 AI 助手' }))
  fireEvent.click(screen.getByRole('button', { name: '打开 AI 助手' }))
  await screen.findByText('改写后的回复')
  expect(screen.queryByText('旧问题')).not.toBeInTheDocument()
  await waitFor(() => expect(screen.getByRole('button', { name: '重试这条消息' })).toBeEnabled())
  fireEvent.click(screen.getByRole('button', { name: '重试这条消息' }))
  await waitFor(() => expect(screen.queryByRole('button', { name: '取消修改' })).not.toBeInTheDocument())
  expect(api.sent).toHaveLength(2)
  expect(api.sent[1]).toMatchObject({ ...api.sent[0], expected_session_revision: 3 })
  expect(screen.getByRole('textbox')).toHaveValue('')
})

it('修改失败后取消编辑会清掉失败重试，随后普通发送不携带修改标识', async () => {
  const api = editPanel({ fail: true })
  render(<AgentPanel pageContext={page('G')} />)
  fireEvent.click(screen.getByRole('button', { name: '打开 AI 助手' }))
  await waitFor(() => expect(screen.getByText('旧问题')).toBeVisible())
  const input = screen.getByRole('textbox')
  fireEvent.change(input, { target: { value: '尚未发送的草稿' } })
  fireEvent.click(screen.getByRole('button', { name: '修改这条消息' }))
  fireEvent.change(input, { target: { value: '放弃的修改' } })
  fireEvent.click(screen.getByRole('button', { name: '发送' }))
  await waitFor(() => expect(api.sent).toHaveLength(1))
  await screen.findByRole('alert')
  fireEvent.click(screen.getByRole('button', { name: '取消修改' }))
  expect(input).toHaveValue('尚未发送的草稿')
  expect(screen.queryByRole('button', { name: '重试这条消息' })).not.toBeInTheDocument()
  fireEvent.change(input, { target: { value: '普通的下一轮' } })
  fireEvent.click(screen.getByRole('button', { name: '发送' }))
  await waitFor(() => expect(api.sent).toHaveLength(2))
  expect(api.sent[1].text).toBe('普通的下一轮')
  expect(api.sent[1]).not.toHaveProperty('edit_of_message_id')
  expect(api.sent[1].message_id).not.toBe(api.sent[0].message_id)
})

it('提交被接受前取消操作不可用，迟到的成功回执不清掉用户新草稿', async () => {
  const api = editPanel({ delay: true })
  render(<AgentPanel pageContext={page('G')} />)
  fireEvent.click(screen.getByRole('button', { name: '打开 AI 助手' }))
  await waitFor(() => expect(screen.getByText('旧问题')).toBeVisible())
  const input = screen.getByRole('textbox')
  fireEvent.click(screen.getByRole('button', { name: '修改这条消息' }))
  fireEvent.change(input, { target: { value: '改写后的问题' } })
  fireEvent.click(screen.getByRole('button', { name: '发送' }))
  await waitFor(() => expect(api.sent).toHaveLength(1))
  expect(screen.getByRole('button', { name: '取消修改' })).toBeDisabled()
  fireEvent.change(input, { target: { value: '提交期间输入的新草稿' } })
  await act(async () => api.release())
  await waitFor(() => expect(screen.queryByRole('button', { name: '取消修改' })).not.toBeInTheDocument())
  expect(input).toHaveValue('提交期间输入的新草稿')
  expect(screen.getByText('改写后的问题')).toBeVisible()
})


describe('会话恢复与重试', () => {
  const context = page('recovery')
  const response = (data: unknown, status = 200) => ({ ok: status < 400, status, json: async () => data })
  it('恢复中的旧会话禁止发送，恢复后原输入发往原会话', async () => {
    sessionStorage.setItem(agentSessionStorageKey(context), 'old-session')
    let resolveRestore!: (value: unknown) => void
    const restoring = new Promise(resolve => { resolveRestore = resolve })
    const created: string[] = [], posted: string[] = []
    const states: Record<string, any> = {
      'new-session': { session_id: 'new-session', session_revision: 0, page_context: context, messages: [], next_event_seq: 1 },
    }
    vi.stubGlobal('EventSource', undefined)
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      const path = new URL(url, 'http://localhost').pathname
      if (path.endsWith('/meta')) return response({ configured: true })
      if (path.endsWith('/sessions') && init?.method === 'POST') {
        created.push('new-session'); return response(states['new-session'])
      }
      if (path.endsWith('/sessions/old-session')) return restoring
      const sid = path.split('/')[4]
      if (path.endsWith('/messages')) {
        posted.push(sid)
        const body = JSON.parse(String(init?.body))
        const run = { run_id: 'new-run', session_id: sid, message_id: body.message_id, session_revision: 1,
          run_revision: 1, status: 'completed', phase: 'thinking',
          response: { session_id: sid, session_revision: 1, reply: { text: '新会话回复' } } }
        Object.assign(states[sid], { active_run: run, session_revision: 1 })
        return response(run)
      }
      if (path.endsWith('/events')) return response({ items: [], has_more: false })
      if (path.includes('/runs/')) return response(states[sid].active_run)
      return response(states[sid])
    }))
    render(<AgentPanel pageContext={context} />)
    fireEvent.click(screen.getByRole('button', { name: '打开 AI 助手' }))
    await waitFor(() => expect(screen.getByRole('textbox')).toBeEnabled())
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '继续刚才的研究' } })
    expect(screen.getByRole('button', { name: '发送' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter', ctrlKey: true })
    await act(async () => { await Promise.resolve() })
    states['old-session'] = { session_id: 'old-session', session_revision: 3, page_context: context,
      messages: [{ id: 'old-reply', speaker: 'assistant', text: '旧会话研究结果' }], next_event_seq: 2 }
    await act(async () => resolveRestore(response(states['old-session'])))
    await screen.findByText('旧会话研究结果')
    expect(screen.getByRole('textbox')).toHaveValue('继续刚才的研究')
    await waitFor(() => expect(screen.getByRole('button', { name: '发送' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
    await waitFor(() => expect(posted).toEqual(['old-session']))
    expect(created).toEqual([])
    expect(sessionStorage.getItem(agentSessionStorageKey(context))).toBe('old-session')
  })

  it('版本冲突后恢复新版本，原消息可再次发送', async () => {
    sessionStorage.setItem(agentSessionStorageKey(context), 'session')
    let revision = 1
    let activeRun: Record<string, unknown> | null = null
    const attempts: any[] = []
    vi.stubGlobal('EventSource', undefined)
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      const path = new URL(url, 'http://localhost').pathname
      if (path.endsWith('/meta')) return response({ configured: true })
      if (path.endsWith('/sessions/session')) return response({ session_id: 'session', session_revision: revision,
        page_context: context, next_event_seq: revision + 1, active_run: activeRun,
        messages: [{ id: `history-${revision}`, speaker: 'assistant', text: `已恢复版本${revision}` }] })
      if (path.endsWith('/events')) return response({ items: [], has_more: false })
      if (path.includes('/runs/')) return response(activeRun)
      if (path.endsWith('/messages')) {
        const body = JSON.parse(String(init?.body)); attempts.push(body)
        if (body.expected_session_revision !== revision) return response({ detail: { code: 'REVISION_CONFLICT', message: '会话已更新，请刷新后重试。' } }, 409)
        activeRun = { session_id: 'session', run_id: 'next', message_id: body.message_id, session_revision: revision,
          run_revision: 1, status: 'completed', phase: 'thinking', response: { session_revision: revision, reply: { text: '已接收' } } }
        return response(activeRun)
      }
      throw new Error(path)
    }))
    render(<AgentPanel pageContext={context} />)
    fireEvent.click(screen.getByRole('button', { name: '打开 AI 助手' }))
    await screen.findByText('已恢复版本1')
    // Another tab sharing sessionStorage can advance this durable session.
    revision = 2
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '新的研究问题' } })
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
    await screen.findByText('会话已更新，请刷新后重试。')
    fireEvent.click(screen.getByRole('button', { name: '关闭 AI 助手' }))
    fireEvent.click(screen.getByRole('button', { name: '打开 AI 助手' }))
    await screen.findByText('已恢复版本2')
    await waitFor(() => expect(screen.getByRole('button', { name: '重试这条消息' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: '重试这条消息' }))
    await waitFor(() => expect(attempts).toHaveLength(2))
    expect(attempts[1].message_id).toBe(attempts[0].message_id)
    expect(attempts[1].expected_session_revision).toBe(2)
    await screen.findByText('已接收')
  })

  it.each([['failed', 'completed'], ['failed', 'paused'], ['failed', 'running'], ['completed', 'paused']])('续接回执丢失后保留原父运行，原状态=%s，恢复状态=%s', async (before, status) => {
    sessionStorage.setItem(agentSessionStorageKey(context), 'session')
    const parent = { run_id: 'parent', session_id: 'session', message_id: 'prior', session_revision: 1,
      run_revision: 1, status: before, phase: 'thinking', response: { session_revision: 1, reply: { text: '原轮次回复' } } }
    let current: any = parent
    const attempts: any[] = []
    vi.stubGlobal('EventSource', undefined)
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      const path = new URL(url, 'http://localhost').pathname
      // Replaying an accepted request needs no newly configured model.
      if (path.endsWith('/meta')) return response({ configured: attempts.length === 0 })
      if (path.endsWith('/sessions/session')) return response({ session_id: 'session', session_revision: current.session_revision,
        page_context: context, next_event_seq: 1, messages: [], active_run: current })
      if (path.endsWith('/events')) return response({ items: [], has_more: false })
      if (path.endsWith('/cancel')) throw new Error('Retry must not cancel the accepted run')
      if (path.includes('/runs/')) return response(current)
      if (path.endsWith('/messages')) {
        const body = JSON.parse(String(init?.body)); attempts.push(body)
        if (attempts.length === 1) {
          current = { ...parent, run_id: 'accepted', message_id: body.message_id, session_revision: 2, status,
            response: { session_revision: 2, reply: { text: '已完成续接任务' } } }
          throw new TypeError('response lost')
        }
        if (body.resume_from_run_id !== attempts[0].resume_from_run_id) return response({ detail: {
          code: 'REVISION_CONFLICT', message: '同一消息标识不能提交不同内容。' } }, 409)
        return response(current)
      }
      throw new Error(path)
    }))
    render(<AgentPanel pageContext={context} />)
    fireEvent.click(screen.getByRole('button', { name: '打开 AI 助手' }))
    await screen.findByText('原轮次回复')
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '继续分析' } })
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
    await screen.findByText('response lost')
    fireEvent.click(screen.getByRole('button', { name: '关闭 AI 助手' }))
    fireEvent.click(screen.getByRole('button', { name: '打开 AI 助手' }))
    await screen.findByText('已完成续接任务')
    await waitFor(() => expect(screen.getByRole('button', { name: '重试这条消息' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: '重试这条消息' }))
    await waitFor(() => expect(attempts).toHaveLength(2))
    expect(attempts[1].message_id).toBe(attempts[0].message_id)
    expect(attempts[1].resume_from_run_id).toBe(attempts[0].resume_from_run_id)
  })
})
