import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import useAgentConversation from './useAgentConversation'
import type { AgentEvent, AgentPageContext, AgentPreviewReference, AgentRun, AgentSession } from '../../services/agent'

const context: AgentPageContext = { page: 'indicator-studio', page_instance_id: 'test', context_revision: 0, view_state: 'inherit',
  calculation: { context_kind: 'single_product', targets: [], period: '1Y' } }
const run: AgentRun = { run_id: 'current', session_id: 'session', message_id: 'question', session_revision: 1, run_revision: 1, status: 'running', phase: 'thinking' }
const response = (data: unknown) => ({ ok: true, json: async () => data })

class Events {
  static instances: Events[] = []
  closed = false
  onopen: (() => void) | null = null
  onerror: (() => void) | null = null
  listener: ((event: MessageEvent) => void) | null = null
  constructor(public url: string) { Events.instances.push(this) }
  addEventListener(_type: string, listener: (event: MessageEvent) => void) { this.listener = listener }
  close() { this.closed = true }
  emit(event: AgentEvent) { this.listener?.({ data: JSON.stringify(event) } as MessageEvent) }
}

function setup(snapshot: Partial<AgentSession>, events: AgentEvent[] = [], failHistory = false) {
  sessionStorage.setItem('agent-session:indicator-studio:single_product:test', 'session')
  const fetcher = vi.fn(async (url: string) => {
    const parsed = new URL(url, 'http://localhost')
    if (parsed.pathname.endsWith('/sessions/session')) return response({ session_id: 'session', session_revision: 1, page_context: context,
      messages: [], next_event_seq: 1, active_run: run, ...snapshot })
    if (parsed.pathname.endsWith('/runs/current')) return response(run)
    if (parsed.pathname.endsWith('/events')) {
      if (parsed.searchParams.get('kind') === 'messages') return response({ items: [{ seq: 3, id: 'earlier', type: 'user.message', speaker: 'user', text: '更早的问题' }], older_cursor: null })
      const after = Number(parsed.searchParams.get('after_seq'))
      if (failHistory && after < (snapshot.next_event_seq || 1) - 1) throw new Error('history unavailable')
      const items = events.filter(event => event.seq! > after).slice(0, 200)
      return response({ items, has_more: false, last_seq: items[items.length - 1]?.seq || after, next_event_seq: snapshot.next_event_seq || 1 })
    }
    throw new Error(`Unexpected request: ${url}`)
  })
  vi.stubGlobal('fetch', fetcher)
  vi.stubGlobal('EventSource', Events)
  return fetcher
}

afterEach(() => { cleanup(); sessionStorage.clear(); Events.instances = []; vi.unstubAllGlobals() })
const connected = async () => { await waitFor(() => expect(Events.instances.some(source => !source.closed)).toBe(true)); return Events.instances.find(source => !source.closed)! }

it.each(['SSE', 'poll'] as const)('通过%s收到终态后恢复提案和引用来源，重复通知不重复读取', async transport => {
  const proposal = { proposal_id: 'proposal', status: 'pending', summary: '以后请用中文回答。' }
  const memory = { memory_id: 'memory', text: '使用简洁中文', version: 1 }
  let terminal = false, snapshots = 0
  sessionStorage.setItem('agent-session:indicator-studio:single_product:test', 'session')
  vi.stubGlobal('EventSource', Events)
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    const path = new URL(url, 'http://localhost').pathname
    const current = terminal ? { ...run, status: 'completed', run_revision: 2 } : run
    if (path.endsWith('/sessions/session')) {
      snapshots += 1
      return response({ session_id: 'session', session_revision: 1, page_context: context, active_run: current,
        messages: [], next_event_seq: terminal ? 2 : 1, memory_proposals: terminal ? [proposal] : [], memory_sources: terminal ? [memory] : [] })
    }
    if (path.endsWith('/runs/current')) return response(current)
    return response({ items: [], has_more: false })
  }))
  const { result } = renderHook(() => useAgentConversation(context, true))
  const source = await connected(), before = snapshots
  await act(async () => {
    terminal = true
    if (transport === 'SSE') source.emit({ seq: 1, type: 'run.completed', run_id: run.run_id })
    else source.onerror?.()
  })
  await waitFor(() => expect(result.current.session?.memory_proposals).toEqual([proposal]))
  expect(result.current.session?.memory_sources).toEqual([memory])
  expect(result.current.run?.status).toBe('completed')
  expect(snapshots).toBe(before + 1)
  await act(async () => source.emit({ seq: 2, type: 'run.completed', run_id: run.run_id }))
  expect(snapshots).toBe(before + 1)
})

it('终态会话读取失败显示断线，重连恢复提案且不创建新会话', async () => {
  let terminal = false, failRestore = true
  sessionStorage.setItem('agent-session:indicator-studio:single_product:test', 'session')
  vi.stubGlobal('EventSource', Events)
  const fetcher = vi.fn(async (url: string) => {
    const path = new URL(url, 'http://localhost').pathname
    if (path.endsWith('/sessions/session')) {
      if (terminal && failRestore) throw new Error('会话读取失败')
      return response({ session_id: 'session', session_revision: 1, page_context: context, messages: [], next_event_seq: 1,
        active_run: terminal ? { ...run, status: 'paused', run_revision: 2 } : run,
        memory_proposals: terminal ? [{ proposal_id: 'proposal', status: 'pending' }] : [] })
    }
    if (path.endsWith('/runs/current')) return response(terminal ? { ...run, status: 'paused', run_revision: 2 } : run)
    return response({ items: [], has_more: false })
  })
  vi.stubGlobal('fetch', fetcher)
  const { result } = renderHook(() => useAgentConversation(context, true))
  const source = await connected()
  await act(async () => { terminal = true; source.emit({ seq: 1, type: 'run.paused', run_id: run.run_id }) })
  await waitFor(() => expect(result.current.disconnected).toBe(true))
  expect(sessionStorage.getItem('agent-session:indicator-studio:single_product:test')).toBe('session')
  await act(async () => { failRestore = false; result.current.refresh() })
  await waitFor(() => expect(result.current.session?.memory_proposals).toHaveLength(1))
  expect(result.current.disconnected).toBe(false)
  expect(fetcher.mock.calls.some(([url]) => url.endsWith('/sessions'))).toBe(false)
})

it('旧运行结束后延迟返回的会话快照不覆盖新运行', async () => {
  let terminal = false, posted = false
  let release!: (value: unknown) => void
  const delayed = new Promise(resolve => { release = resolve })
  const next = { ...run, run_id: 'next', message_id: 'next-question', session_revision: 2 }
  const snapshot = (active: AgentRun) => ({ session_id: 'session', session_revision: active.session_revision,
    page_context: context, active_run: active, messages: [], next_event_seq: 1, memory_proposals: [] })
  sessionStorage.setItem('agent-session:indicator-studio:single_product:test', 'session')
  vi.stubGlobal('EventSource', Events)
  const fetcher = vi.fn(async (url: string) => {
    const path = new URL(url, 'http://localhost').pathname
    if (path.endsWith('/messages')) { posted = true; return response(next) }
    if (path.endsWith('/sessions/session')) return terminal && !posted ? delayed : response(snapshot(posted ? next : run))
    if (path.endsWith('/runs/next')) return response(next)
    if (path.endsWith('/runs/current')) return response(terminal ? { ...run, status: 'completed', run_revision: 2 } : run)
    return response({ items: [], has_more: false })
  })
  vi.stubGlobal('fetch', fetcher)
  const { result } = renderHook(() => useAgentConversation(context, true))
  const source = await connected(), before = fetcher.mock.calls.filter(([url]) => url.endsWith('/sessions/session')).length
  await act(async () => { terminal = true; source.emit({ seq: 1, type: 'run.completed', run_id: run.run_id }) })
  await waitFor(() => expect(fetcher.mock.calls.filter(([url]) => url.endsWith('/sessions/session'))).toHaveLength(before + 1))
  await act(async () => { await result.current.send('下一轮请求') })
  await waitFor(() => expect(result.current.session?.session_revision).toBe(2))
  await act(async () => release(response({ ...snapshot(run), memory_proposals: [{ proposal_id: 'old' }] })))
  expect(result.current.run?.run_id).toBe('next')
  expect(result.current.session?.session_revision).toBe(2)
  expect(result.current.session?.memory_proposals).toEqual([])
})

function queuedSetup() {
  let current = run, terminalReads = 0
  let releaseQueue!: (value: unknown) => void, rejectQueue!: (reason: Error) => void
  let releasePost!: (value: unknown) => void
  const queueRead = new Promise((resolve, reject) => { releaseQueue = resolve; rejectQueue = reject })
  const posting = new Promise(resolve => { releasePost = resolve })
  const requests: Array<Record<string, any>> = []
  const snapshot = () => ({ session_id: 'session', session_revision: current.session_revision,
    page_context: context, active_run: current, messages: [], next_event_seq: 1 })
  sessionStorage.setItem('agent-session:indicator-studio:single_product:test', 'session')
  vi.stubGlobal('EventSource', Events)
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    const path = new URL(url, 'http://localhost').pathname
    if (path.endsWith('/cancel')) { current = { ...run, status: 'cancelled', run_revision: 2 }; return response(current) }
    if (path.endsWith('/messages')) { requests.push(JSON.parse(String(init?.body))); return posting }
    if (path.endsWith('/sessions/session')) {
      if (current.status === 'cancelled' && ++terminalReads === 1) return queueRead
      return response(snapshot())
    }
    if (path.includes('/runs/')) return response(current)
    return response({ items: [], has_more: false })
  }))
  return {
    requests, terminalReads: () => terminalReads,
    release: () => releaseQueue(response(snapshot())), reject: () => rejectQueue(new Error('排队会话读取失败')),
    complete: () => {
      current = { ...run, run_id: 'next', message_id: requests[0].message_id, session_revision: 2, status: 'completed' }
      releasePost(response(current))
    },
  }
}

describe('useAgentConversation queued handoff', () => {
  it.each([false, true])('读取排队会话到POST接管前持续阻止插队，关闭浮窗不取消：%s', async close => {
    const api = queuedSetup()
    const { result, rerender } = renderHook(({ open }) => useAgentConversation(context, open), { initialProps: { open: true } })
    await connected()
    await act(async () => { await result.current.send('排队 A') })
    await waitFor(() => { expect(api.terminalReads()).toBe(2); expect(result.current.restoring).toBe(false) })
    if (close) rerender({ open: false })
    expect(result.current.sending).toBe(true)
    await act(async () => { expect(await result.current.send('插队 B')).toBeNull() })
    expect(api.requests).toEqual([])
    expect(result.current.messages.map(message => message.text)).toEqual(['排队 A'])
    await act(async () => api.release())
    await waitFor(() => expect(api.requests.map(request => request.text)).toEqual(['排队 A']))
    expect(result.current.sending).toBe(true)
    expect(result.current.messages[0].queued).toBe(false)
    await act(async () => api.complete())
    await waitFor(() => expect(result.current.sending).toBe(false))
    expect(result.current.failed).toBeNull()
    expect(api.requests).toHaveLength(1)
  })

  it('取消排队后忽略迟到读取，不干扰随后发送的新消息', async () => {
    const api = queuedSetup()
    const { result } = renderHook(() => useAgentConversation(context, true))
    await connected()
    await act(async () => { await result.current.send('排队 A') })
    await waitFor(() => expect(result.current.restoring).toBe(false))
    act(() => result.current.cancelQueued())
    expect(result.current.messages[0]).toMatchObject({ queued: false, failed: true })
    let pending!: Promise<unknown>
    act(() => { pending = result.current.send('新消息 B') })
    await waitFor(() => expect(api.requests.map(request => request.text)).toEqual(['新消息 B']))
    await act(async () => api.release())
    expect(result.current.sending).toBe(true)
    expect(result.current.failed).toBeNull()
    await act(async () => { api.complete(); await pending })
    expect(api.requests).toHaveLength(1)
  })

  it.each(['read-error', 'disabled'] as const)('排队未能发送时保留原消息并可重试：%s', async reason => {
    const api = queuedSetup()
    const { result, rerender } = renderHook(({ enabled }) => useAgentConversation(context, true, { enabled }), { initialProps: { enabled: true } })
    await connected()
    await act(async () => { await result.current.send('排队 A') })
    await waitFor(() => expect(result.current.restoring).toBe(false))
    const id = result.current.messages[0].id
    if (reason === 'disabled') rerender({ enabled: false })
    await act(async () => { if (reason === 'read-error') api.reject(); else api.release() })
    await waitFor(() => expect(result.current.failed?.id).toBe(id))
    expect(result.current.messages[0]).toMatchObject({ queued: false, failed: true })
    expect(result.current.sending).toBe(false)
    expect(result.current.error).not.toBe('')
    expect(api.requests).toEqual([])
    rerender({ enabled: true })
    await waitFor(() => expect(result.current.restoring).toBe(false))
    let retry!: Promise<unknown>
    act(() => { retry = result.current.retry() })
    await waitFor(() => expect(api.requests).toHaveLength(1))
    expect(api.requests[0]).toMatchObject({ message_id: id, text: '排队 A' })
    await act(async () => { api.complete(); await retry })
    expect(result.current.messages).toHaveLength(1)
    expect(result.current.failed).toBeNull()
  })

  it('研究对象卸载后迟到的排队读取不能发送消息', async () => {
    const api = queuedSetup()
    const { result, unmount } = renderHook(() => useAgentConversation(context, true, { cancelOnUnmount: true }))
    await connected()
    await act(async () => { await result.current.send('排队 A') })
    await waitFor(() => expect(result.current.restoring).toBe(false))
    unmount()
    await act(async () => api.release())
    expect(api.requests).toEqual([])
  })
})

function controlSetup() {
  let current = run, pageContext = context
  let resolve!: (value: unknown) => void, reject!: (reason: Error) => void
  const pending = new Promise((done, fail) => { resolve = done; reject = fail })
  sessionStorage.setItem('agent-session:indicator-studio:single_product:test', 'session')
  vi.stubGlobal('EventSource', Events)
  const fetcher = vi.fn(async (url: string, init?: RequestInit) => {
    const path = new URL(url, 'http://localhost').pathname
    if (path.endsWith('/cancel') || path.endsWith('/invalidate-context')) return pending
    if (path.endsWith('/messages')) {
      const body = JSON.parse(String(init?.body)); pageContext = body.page_context
      current = { ...run, run_id: 'next', message_id: body.message_id, session_revision: 2, status: 'completed' }
      return response(current)
    }
    if (path.endsWith('/sessions/session')) return response({ session_id: 'session', session_revision: current.session_revision,
      page_context: pageContext, active_run: current, messages: [], next_event_seq: 1 })
    if (path.includes('/runs/')) return response(current)
    return response({ items: [], has_more: false })
  })
  vi.stubGlobal('fetch', fetcher)
  return {
    fetcher, finish: () => { current = { ...run, status: 'completed', run_revision: 2 } },
    answer: (failed: boolean) => {
      if (failed) reject(new Error('控制请求失败'))
      else { current = { ...run, status: 'cancelled', run_revision: 2 }; resolve(response(current)) }
    },
  }
}

it.each((['stop', 'invalidate'] as const).flatMap(mode => [false, true].map(next => ({ mode, next }))))('忽略$mode在已结束运行后的迟到错误，已进入下一轮=$next', async ({ mode, next }) => {
  const api = controlSetup()
  const { result, rerender } = renderHook(({ page }) => useAgentConversation(page, true), { initialProps: { page: context } })
  const source = await connected()
  let stopped: Promise<unknown> | undefined
  if (mode === 'stop') act(() => { stopped = result.current.stop() })
  else rerender({ page: { ...context, calculation: { ...context.calculation, period: '3Y' } } })
  await waitFor(() => expect(api.fetcher.mock.calls.some(([url]) => url.endsWith(mode === 'stop' ? '/cancel' : '/invalidate-context'))).toBe(true))
  await act(async () => { api.finish(); source.emit({ seq: 1, type: 'run.completed', run_id: run.run_id }) })
  await waitFor(() => { expect(result.current.run?.status).toBe('completed'); expect(result.current.restoring).toBe(false) })
  if (next) {
    await act(async () => { await result.current.send('下一轮请求') })
    await waitFor(() => { expect(result.current.run?.run_id).toBe('next'); expect(result.current.restoring).toBe(false) })
  }
  const before = api.fetcher.mock.calls.length
  await act(async () => { api.answer(true); await stopped })
  expect(result.current.error).toBe('')
  expect(result.current.run?.run_id).toBe(next ? 'next' : run.run_id)
  expect(api.fetcher.mock.calls).toHaveLength(before)
})

it.each((['stop', 'invalidate'] as const).flatMap(mode => [false, true].map(failed => ({ mode, failed }))))('关闭浮窗保留当前运行的$mode结果，失败=$failed', async ({ mode, failed }) => {
  const api = controlSetup()
  const { result, rerender } = renderHook(({ page, open }) => useAgentConversation(page, open), { initialProps: { page: context, open: true } })
  await connected()
  const page = mode === 'stop' ? context : { ...context, calculation: { ...context.calculation, period: '3Y' } }
  let stopped: Promise<unknown> | undefined
  if (mode === 'stop') act(() => { stopped = result.current.stop() })
  else rerender({ page, open: true })
  await waitFor(() => expect(api.fetcher.mock.calls.some(([url]) => url.endsWith(mode === 'stop' ? '/cancel' : '/invalidate-context'))).toBe(true))
  rerender({ page, open: false })
  await act(async () => { api.answer(failed); await stopped })
  expect(result.current.run?.status).toBe(failed ? 'running' : 'cancelled')
  expect(result.current.error).toBe(failed ? '控制请求失败' : '')
})

it.each([false, true].flatMap(edit => [false, true].map(lost => ({ edit, lost }))))('恢复先确认接收后，迟到POST不倒退运行或回滚消息：编辑=$edit，回包丢失=$lost', async ({ edit, lost }) => {
  const oldMessage = { seq: 1, id: 'old-message', run_id: 'old-run', speaker: 'user' as const, text: '旧问题' }
  const oldReply = { seq: 2, id: 'old-run-reply', run_id: 'old-run', speaker: 'assistant' as const, text: '旧停止提示' }
  let current: AgentRun = { ...run, run_id: 'old-run', message_id: oldMessage.id, status: 'cancelled', run_revision: 2 }
  let messages: AgentEvent[] = [oldMessage, oldReply], holdRestore = false
  const requests: Array<Record<string, any>> = []
  let resolve!: (value: unknown) => void, reject!: (reason: Error) => void
  const pending = new Promise((done, fail) => { resolve = done; reject = fail })
  const proposal = { proposal_id: 'new-proposal', status: 'pending', summary: '以后用中文回答。' }
  sessionStorage.setItem('agent-session:indicator-studio:single_product:test', 'session')
  vi.stubGlobal('EventSource', undefined)
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    const path = new URL(url, 'http://localhost').pathname
    if (path.endsWith('/messages')) {
      const body = JSON.parse(String(init?.body)); requests.push(body)
      current = { ...run, run_id: 'new-run', message_id: body.message_id, session_revision: 2, run_revision: 3, status: 'completed',
        response: { session_id: 'session', session_revision: 2, reply: { text: '新回复' }, artifacts: {} } }
      messages = [...(edit ? [] : [oldMessage, oldReply]),
        { seq: 3, id: body.message_id, run_id: 'new-run', speaker: 'user', text: '新问题', ...(edit ? { edit_of: oldMessage.id } : {}) },
        { seq: 4, id: 'new-run-reply', run_id: 'new-run', speaker: 'assistant', text: '新回复' }]
      return pending
    }
    if (path.endsWith('/sessions/session')) {
      if (holdRestore) return new Promise(() => undefined)
      return response({ session_id: 'session', session_revision: current.session_revision, page_context: context, active_run: current,
        messages, next_event_seq: (messages[messages.length - 1]?.seq || 0) + 1, memory_proposals: requests.length ? [proposal] : [] })
    }
    if (path.includes('/runs/')) return response(current)
    return response({ items: [], has_more: false })
  }))
  const { result, rerender } = renderHook(({ open }) => useAgentConversation(context, open), { initialProps: { open: true } })
  await waitFor(() => { expect(result.current.messages).toHaveLength(2); expect(result.current.restoring).toBe(false) })
  let posting!: Promise<AgentRun | null>
  act(() => { posting = edit ? result.current.edit({ id: oldMessage.id, text: oldMessage.text }, '新问题') : result.current.send('新问题') })
  await waitFor(() => expect(requests).toHaveLength(1))
  rerender({ open: false }); rerender({ open: true })
  await waitFor(() => { expect(result.current.run?.run_id).toBe('new-run'); expect(result.current.restoring).toBe(false) })
  const expectedText = [...(edit ? [] : ['旧问题', '旧停止提示']), '新问题', '新回复']
  expect(result.current.messages.map(message => message.text)).toEqual(expectedText)
  holdRestore = true
  await act(async () => {
    if (lost) reject(new Error('原始回包丢失'))
    else resolve(response({ ...current, status: 'queued', run_revision: 0, response: undefined }))
    expect(await posting).toMatchObject({ run_id: 'new-run', status: 'completed', run_revision: 3 })
  })
  expect(result.current.run).toMatchObject({ run_id: 'new-run', status: 'completed', run_revision: 3 })
  expect(result.current.session?.memory_proposals).toEqual([proposal])
  expect(result.current.error).toBe('')
  expect(result.current.failed).toBeNull()
  expect(result.current.messages.map(message => message.text)).toEqual(expectedText)
  holdRestore = false
  act(() => result.current.refresh())
  await waitFor(() => expect(result.current.restoring).toBe(false))
  expect(result.current.messages.map(message => message.text)).toEqual(expectedText)
  expect(requests).toHaveLength(1)
})

it.each((['send', 'edit', 'queue', 'retry'] as const).flatMap(mode => [false, true].map(postFirst => ({ mode, postFirst }))))('旧会话读取与$mode提交交错时，运行保持自己的冻结口径：POST先返回=$postFirst', async ({ mode, postFirst }) => {
  let current: AgentRun = { ...run, status: mode === 'queue' ? 'running' : 'cancelled', run_revision: 2 }
  let serverContext = context, delayRestore = false
  const requests: Array<Record<string, any>> = [], invalidations: Array<Record<string, any>> = []
  let resolvePost!: (value: unknown) => void, resolveRestore!: (value: unknown) => void
  const pendingPost = new Promise(resolve => { resolvePost = resolve })
  const pendingRestore = new Promise(resolve => { resolveRestore = resolve })
  const snapshot = () => ({ session_id: 'session', session_revision: current.session_revision, page_context: serverContext,
    active_run: current, messages: [{ seq: 1, id: run.message_id, run_id: run.run_id, speaker: 'user', text: '旧问题' }], next_event_seq: 2 })
  sessionStorage.setItem('agent-session:indicator-studio:single_product:test', 'session')
  vi.stubGlobal('EventSource', Events)
  const fetcher = vi.fn(async (url: string, init?: RequestInit) => {
    const path = new URL(url, 'http://localhost').pathname
    if (path.endsWith('/messages')) {
      requests.push(JSON.parse(String(init?.body)))
      if (mode === 'retry' && requests.length === 1) throw new Error('尚未提交')
      return pendingPost
    }
    if (path.endsWith('/cancel')) { current = { ...current, status: 'cancelled', run_revision: 3 }; return response(current) }
    if (path.endsWith('/invalidate-context')) {
      if (path.includes('/runs/new/')) invalidations.push(JSON.parse(String(init?.body)))
      return response(current)
    }
    if (path.endsWith('/sessions/session')) return delayRestore ? pendingRestore : response(snapshot())
    if (path.includes('/runs/')) return response(current)
    return response({ items: [], has_more: false })
  })
  vi.stubGlobal('fetch', fetcher)
  const { result, rerender } = renderHook(({ page, open }) => useAgentConversation(page, open), { initialProps: { page: context, open: true } })
  await waitFor(() => { expect(result.current.run?.run_id).toBe(run.run_id); expect(result.current.restoring).toBe(false) })
  const nextContext = { ...context, calculation: { ...context.calculation, period: '3Y' } }
  rerender({ page: nextContext, open: true })
  let sending: Promise<unknown>
  act(() => { sending = mode === 'edit' ? result.current.edit({ id: run.message_id, text: '旧问题' }, '三年研究') : result.current.send('三年研究') })
  if (mode === 'retry') {
    await waitFor(() => expect(result.current.failed).not.toBeNull())
    act(() => { sending = result.current.retry() })
  }
  await waitFor(() => expect(requests).toHaveLength(mode === 'retry' ? 2 : 1))
  const oldSnapshot = snapshot()
  delayRestore = true
  rerender({ page: nextContext, open: false }); rerender({ page: nextContext, open: true })
  const accept = async () => {
    delayRestore = false; serverContext = nextContext
    current = { ...run, run_id: 'new', message_id: requests[requests.length - 1].message_id, session_revision: 2, status: 'running' }
    await act(async () => { resolvePost(response(current)); await sending })
  }
  if (postFirst) { await accept(); await act(async () => resolveRestore(response(oldSnapshot))) }
  else { await act(async () => resolveRestore(response(oldSnapshot))); await accept() }
  await waitFor(() => expect(result.current.restoring).toBe(false))
  expect(result.current.run?.run_id).toBe('new')
  expect(result.current.contextChanged).toBe(false)
  expect(invalidations).toEqual([])
  expect(requests[requests.length - 1].page_context).toEqual(nextContext)
  if (mode === 'retry') expect(requests[1]).toEqual(requests[0])
  // A later genuine page change still stops this accepted run under its original 3Y context.
  rerender({ page: { ...nextContext, calculation: { ...nextContext.calculation, period: '5Y' } }, open: true })
  await waitFor(() => expect(invalidations).toHaveLength(1))
  expect(invalidations[0].page_context.calculation.period).toBe('5Y')
})

it.each(['send', 'edit', 'queue', 'retry'] as const)('%s新运行不能用上一轮试算许可忽略真实口径变化', async mode => {
  const target = { kind: 'etf', product_id: '510300.SH' }
  const original = { ...context, calculation: { ...context.calculation, targets: [target], period: '1Y', as_of: '2020-01-01' } }
  const reference: AgentPreviewReference = { preview_id: 'old-preview', run_id: mode === 'edit' ? 'earlier' : run.run_id,
    definition_hash: 'hash', target, period: '1Y', as_of: '2020-01-01', result_kind: 'scalar' }
  let current: AgentRun = { ...run, status: mode === 'queue' ? 'running' : mode === 'edit' ? 'cancelled' : 'completed' }
  let accepted = false
  let release!: (value: unknown) => void
  const pending = new Promise(resolve => { release = resolve })
  const requests: Array<Record<string, any>> = [], invalidations: string[] = []
  sessionStorage.setItem('agent-session:indicator-studio:single_product:test', 'session')
  vi.stubGlobal('EventSource', Events)
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    const path = new URL(url, 'http://localhost').pathname
    if (path.endsWith('/messages')) {
      requests.push(JSON.parse(String(init?.body)))
      if (mode === 'retry' && requests.length === 1) throw new Error('尚未提交')
      return pending
    }
    if (path.endsWith('/cancel')) { current = { ...current, status: 'cancelled', run_revision: 2 }; return response(current) }
    if (path.endsWith('/invalidate-context')) {
      if (path.includes('/runs/new/')) invalidations.push(JSON.parse(String(init?.body)).page_context.calculation.period)
      return response(current)
    }
    if (path.endsWith('/sessions/session')) return response({ session_id: 'session', session_revision: current.session_revision,
      page_context: accepted ? requests[requests.length - 1].page_context : original, active_run: current, messages: [], next_event_seq: 1,
      preview: accepted ? null : reference, draft: accepted ? null : { valid: true, draft_revision: 1, definition_hash: 'hash', definition: { name: '旧试算指标' } } })
    if (path.includes('/runs/')) return response(current)
    return response({ items: [], has_more: false })
  }))
  const options = { adoptPreviewContext: (frozen: AgentPageContext, preview: AgentPreviewReference) => ({ ...frozen,
    calculation: { ...frozen.calculation, targets: [preview.target], period: preview.period, as_of: preview.as_of || null } }) }
  const { result, rerender } = renderHook(({ page }) => useAgentConversation(page, true, options), { initialProps: { page: original } })
  await waitFor(() => { expect(result.current.run?.run_id).toBe(run.run_id); expect(result.current.restoring).toBe(false) })
  rerender({ page: { ...original, calculation: { ...original.calculation, period: '3Y' } } })
  let sending: Promise<unknown>
  act(() => { sending = mode === 'edit' ? result.current.edit({ id: run.message_id, text: '旧问题' }, '三年研究') : result.current.send('三年研究') })
  if (mode === 'retry') {
    await waitFor(() => expect(result.current.failed).not.toBeNull())
    act(() => { sending = result.current.retry() })
  }
  await waitFor(() => expect(requests).toHaveLength(mode === 'retry' ? 2 : 1))
  rerender({ page: original })
  await act(async () => {
    accepted = true
    current = { ...run, run_id: 'new', message_id: requests[requests.length - 1].message_id, session_revision: 2 }
    release(response(current)); await sending
  })
  await waitFor(() => { expect(result.current.restoring).toBe(false); expect(result.current.session?.preview).toBeNull() })
  expect(requests[requests.length - 1].page_context.calculation.period).toBe('3Y')
  expect(result.current.contextChanged).toBe(true)
  expect(invalidations).toEqual(['1Y'])
})

it('本轮试算回填可采纳，清除引用后保持该轮已采纳口径', async () => {
  const reference: AgentPreviewReference = { preview_id: 'preview', run_id: run.run_id, definition_hash: 'hash',
    target: { kind: 'etf', product_id: '510300.SH' }, period: '3Y', as_of: '2020-01-01', result_kind: 'scalar' }
  let clear = false
  const fetcher = setup({ preview: reference, draft: { valid: true, draft_revision: 1, definition_hash: 'hash', definition: {} } })
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    const value = await fetcher(url)
    return clear && url.endsWith('/sessions/session') ? response({ ...(await value.json() as Record<string, unknown>), preview: null }) : value
  }))
  const adopt = (frozen: AgentPageContext, preview: AgentPreviewReference) => ({ ...frozen,
    calculation: { ...frozen.calculation, targets: [preview.target], period: preview.period, as_of: preview.as_of || null } })
  const { result, rerender } = renderHook(({ page }) => useAgentConversation(page, true, { adoptPreviewContext: adopt }), { initialProps: { page: context } })
  await connected()
  rerender({ page: adopt(context, reference) })
  expect(result.current.contextChanged).toBe(false)
  act(() => { clear = true; result.current.refresh() })
  await waitFor(() => expect(result.current.session?.preview).toBeNull())
  expect(result.current.contextChanged).toBe(false)
  expect(fetcher.mock.calls.some(([url]) => url.endsWith('/invalidate-context'))).toBe(false)
})

it('恢复口径拥有独立冻结副本，调用者原地修改不能扩大试算采纳权限', async () => {
  const borrowed = structuredClone(context)
  const preview: AgentPreviewReference = { preview_id: 'preview', run_id: run.run_id, definition_hash: 'hash',
    target: { kind: 'etf', product_id: '510300.SH' }, period: '3Y', result_kind: 'scalar' }
  const fetcher = setup({ page_context: borrowed, preview })
  const adopt = (frozen: AgentPageContext, reference: AgentPreviewReference) => ({ ...frozen,
    calculation: { ...frozen.calculation, targets: [reference.target], period: reference.period, as_of: null } })
  const { result, rerender } = renderHook(({ page }) => useAgentConversation(page, true, { adoptPreviewContext: adopt }), { initialProps: { page: context } })
  await connected()
  borrowed.context_revision = 1
  rerender({ page: adopt(borrowed, preview) })
  await waitFor(() => expect(fetcher.mock.calls.some(([url]) => url.endsWith('/invalidate-context'))).toBe(true))
  expect(result.current.contextChanged).toBe(true)
})

it('同会话版本补齐记忆时，旧运行快照不能恢复已释放的执行屏障和旧草稿', async () => {
  const currentDraft = { valid: true, draft_revision: 2, definition_hash: 'current', definition: {} }
  const blocked: AgentRun = { ...run, status: 'cancelled', run_revision: 3, execution_blocked_by: 'worker' }
  const released: AgentRun = { ...blocked, run_revision: 4, execution_blocked_by: null,
    response: { session_id: 'session', session_revision: 1, draft: currentDraft, artifacts: {} } }
  let oldSnapshot = false
  const proposal = { proposal_id: 'pending', status: 'pending', summary: '以后用简洁中文回答。' }
  sessionStorage.setItem('agent-session:indicator-studio:single_product:test', 'session')
  vi.stubGlobal('EventSource', Events)
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    const path = new URL(url, 'http://localhost').pathname
    if (path.endsWith('/sessions/session')) return response({ session_id: 'session', session_revision: 2, page_context: context,
      active_run: blocked, execution_blocked_by: 'worker', memory_proposals: oldSnapshot ? [proposal] : [],
      draft: oldSnapshot ? { ...currentDraft, definition_hash: 'old', draft_revision: 1 } : currentDraft, messages: [], next_event_seq: 1 })
    if (path.includes('/runs/')) return oldSnapshot ? new Promise(() => undefined) : response(released)
    return response({ items: [], has_more: false })
  }))
  const { result } = renderHook(() => useAgentConversation(context, true))
  await waitFor(() => expect(result.current.run?.run_revision).toBe(4))
  act(() => { oldSnapshot = true; result.current.refresh() })
  await waitFor(() => expect(result.current.session?.memory_proposals).toEqual([proposal]))
  expect(result.current.session?.session_revision).toBe(2)
  expect(result.current.session?.active_run?.run_revision).toBe(4)
  expect(result.current.session?.execution_blocked_by).toBeNull()
  expect(result.current.draft).toEqual(currentDraft)
  expect(result.current.busy).toBe(false)
})

it.each(['events', 'runs/current'])('会话已读取成功后%s的404仅报告断线，不清除会话或误建会话', async resource => {
  const draft = { valid: true, draft_revision: 1, definition_hash: 'hash', definition: {} }
  sessionStorage.setItem('agent-session:indicator-studio:single_product:test', 'session')
  vi.stubGlobal('EventSource', Events)
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    const path = new URL(url, 'http://localhost').pathname
    if (path.endsWith('/sessions/session')) return response({ session_id: 'session', session_revision: 1,
      page_context: context, active_run: { ...run, status: 'completed' }, draft, messages: [], next_event_seq: 1 })
    if (path.endsWith('/' + resource)) return { ok: false, status: 404, json: async () => ({ detail: { message: '附属资源暂不可用' } }) }
    return response({ items: [], has_more: false })
  }))
  const { result } = renderHook(() => useAgentConversation(context, true))
  await waitFor(() => expect(result.current.disconnected).toBe(true))
  expect(result.current.session?.session_id).toBe('session')
  expect(result.current.run?.run_id).toBe(run.run_id)
  expect(result.current.draft).toEqual(draft)
  expect(sessionStorage.getItem('agent-session:indicator-studio:single_product:test')).toBe('session')
})

it('阶段事件不能把正在停止的同轮恢复为运行中，会话更新保持归属', async () => {
  setup({})
  const { result } = renderHook(() => useAgentConversation(context, true))
  const source = await connected()
  await act(async () => source.emit({ seq: 1, type: 'run.phase', run_id: run.run_id, data: { status: 'stopping', phase: 'tool' } }))
  await act(async () => source.emit({ seq: 2, type: 'run.started', run_id: run.run_id, data: { status: 'running', phase: 'thinking' } }))
  expect(result.current.run).toMatchObject({ status: 'stopping', phase: 'tool' })
  // Session-only updates (e.g. a memory decision) cannot replace the current run identity.
  act(() => result.current.acceptSession({ session_id: 'foreign', session_revision: 999 }))
  expect(result.current.session?.session_id).toBe('session')
  expect(result.current.run?.run_id).toBe(run.run_id)
})

it('实时阶段更新不重放旧运行快照中的成果', async () => {
  const first = { valid: true, draft_revision: 1, definition_hash: 'first', definition: { name: '第一项定义' } }
  const updated = { ...first, draft_revision: 2, definition_hash: 'updated', definition: { name: '已更新定义' } }
  const current = { ...run, artifacts: { draft: first } }
  const fetcher = setup({ active_run: current, draft: first })
  vi.stubGlobal('fetch', vi.fn(async (url: string) => url.endsWith('/runs/current') ? response(current) : fetcher(url)))
  const { result } = renderHook(() => useAgentConversation(context, true))
  const source = await connected()
  await act(async () => source.emit({ seq: 1, run_id: run.run_id, type: 'draft.updated', data: { draft: updated } }))
  await act(async () => {
    source.emit({ seq: 2, run_id: run.run_id, type: 'tool.started', data: { tool: 'metrics.validate' } })
    source.emit({ seq: 3, run_id: run.run_id, type: 'run.phase', data: { status: 'running', phase: 'thinking' } })
  })
  expect(result.current.draft).toEqual(updated)
  expect(result.current.messages.find(message => message.id === `${run.run_id}-reply`)?.artifacts?.draft).toEqual(updated)
  expect(result.current.session?.active_run).toBe(result.current.run)
})

it('缺少当前运行的旧会话仍保留执行屏障，收到合法运行后按其版本释放', async () => {
  let restoredRun = false
  const blocked = { ...run, status: 'cancelled' as const, run_revision: 3, execution_blocked_by: 'worker' }
  const released = { ...blocked, run_revision: 4, execution_blocked_by: null }
  sessionStorage.setItem('agent-session:indicator-studio:single_product:test', 'session')
  vi.stubGlobal('EventSource', Events)
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    const path = new URL(url, 'http://localhost').pathname
    if (path.endsWith('/sessions/session')) return response({ session_id: 'session', session_revision: 1,
      page_context: context, active_run: restoredRun ? blocked : null, execution_blocked_by: 'worker', messages: [], next_event_seq: 1 })
    if (path.includes('/runs/')) return response(released)
    return response({ items: [], has_more: false })
  }))
  const { result } = renderHook(() => useAgentConversation(context, true))
  await waitFor(() => expect(result.current.restoring).toBe(false))
  expect(result.current.run).toBeNull()
  expect(result.current.session?.execution_blocked_by).toBe('worker')
  expect(result.current.busy).toBe(true)
  act(() => { restoredRun = true; result.current.refresh() })
  await waitFor(() => expect(result.current.run?.run_revision).toBe(4))
  expect(result.current.session?.active_run).toBe(result.current.run)
  expect(result.current.session?.execution_blocked_by).toBeNull()
  expect(result.current.busy).toBe(false)
})

it.each([503, 404])('恢复请求结束前阻止直接发送，仅明确404允许新建，status=%s', async status => {
  sessionStorage.setItem('agent-session:indicator-studio:single_product:test', 'session')
  let release!: (value: unknown) => void
  const pending = new Promise(resolve => { release = resolve })
  let restored = false
  const writes: string[] = []
  vi.stubGlobal('EventSource', undefined)
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    if (init?.method === 'POST') {
      writes.push(url)
      return response(url.endsWith('/sessions') ? { session_id: 'new', session_revision: 0 } : { ...run, status: 'completed' })
    }
    if (url.endsWith('/events?after_seq=0&limit=200')) return response({ items: [], has_more: false })
    return restored ? response({ session_id: 'session', session_revision: 1, page_context: context, messages: [] }) : pending
  }))
  const { result } = renderHook(() => useAgentConversation(context, true))
  await act(async () => { expect(await result.current.send('恢复期间')).toBeNull() })
  expect(writes).toEqual([])
  expect(result.current.messages).toEqual([])
  await act(async () => release({ ok: false, status, json: async () => ({ detail: { message: '恢复读取失败' } }) }))
  if (status === 503) {
    expect(result.current.restoring).toBe(true)
    await act(async () => { expect(await result.current.send('读取失败后')).toBeNull() })
    expect(writes).toEqual([])
    expect(sessionStorage.getItem('agent-session:indicator-studio:single_product:test')).toBe('session')
    restored = true
    act(() => result.current.refresh())
    await waitFor(() => expect(result.current.restoring).toBe(false))
    expect(result.current.session?.session_id).toBe('session')
  } else {
    expect(result.current.restoring).toBe(false)
    await act(async () => { await result.current.send('不存在时创建') })
    expect(writes.filter(url => url.endsWith('/sessions'))).toHaveLength(1)
  }
})

describe('useAgentConversation public activity', () => {
  it('研究对象卸载取消已运行任务，关闭对话不取消', async () => {
    const fetcher = setup({})
    const { rerender, unmount } = renderHook(({ open }) => useAgentConversation(context, open, { cancelOnUnmount: true }), { initialProps: { open: true } })
    await connected()
    rerender({ open: false })
    await act(async () => undefined)
    expect(fetcher.mock.calls.some(([url]) => url.endsWith('/cancel'))).toBe(false)
    unmount()
    await waitFor(() => expect(fetcher.mock.calls.some(([url]) => url.endsWith('/cancel'))).toBe(true))
  })

  it('研究对象卸载后才接受的运行也被取消', async () => {
    let resolveMessage!: (value: unknown) => void
    const accepted = new Promise(resolve => { resolveMessage = resolve })
    const fetcher = vi.fn(async (url: string) => url.endsWith('/sessions')
      ? response({ session_id: 'session', session_revision: 0 }) : url.includes('/messages?') ? accepted : response({ ...run, status: 'cancelled' }))
    vi.stubGlobal('fetch', fetcher)
    const { result, unmount } = renderHook(() => useAgentConversation(context, false, { cancelOnUnmount: true }))
    let pending: Promise<unknown>
    act(() => { pending = result.current.send('分析') })
    await waitFor(() => expect(fetcher.mock.calls.some(([url]) => url.includes('/messages?'))).toBe(true))
    unmount()
    await act(async () => { resolveMessage(response(run)); await pending })
    expect(fetcher.mock.calls.some(([url]) => url.endsWith('/cancel'))).toBe(true)
  })
  it('恢复最近公开动作，按轮保留独立开始/完成事件且不泄露工具内容', async () => {
    const events: AgentEvent[] = [
      { seq: 805, type: 'tool.started', run_id: 'previous', data: { tool: 'products.search', operation_id: 'private-operation' } },
      { seq: 997, type: 'tool.started', run_id: 'current', data: { tool: 'metrics.validate', arguments: { secret: 'arguments' } } },
      { seq: 999, type: 'tool.completed', run_id: 'current', text: 'private output', summary: 'private summary', data: { tool: 'metrics.validate', status: 'ok', duration_ms: 25, reasoning_content: 'private thought', result: { secret: 'result' } } },
    ]
    const fetcher = setup({ next_event_seq: 1001, older_message_cursor: 800,
      messages: [{ seq: 800, id: 'previous-reply', run_id: 'previous', speaker: 'assistant', text: '原有回答' }] }, events)
    const { result } = renderHook(() => useAgentConversation(context, true))
    const source = await connected()
    await waitFor(() => expect(result.current.activity).toHaveLength(3))
    expect(result.current.activity).toEqual([
      { seq: 805, id: 'activity-805', type: 'tool.started', run_id: 'previous', data: { tool: 'products.search' } },
      { seq: 997, id: 'activity-997', type: 'tool.started', run_id: 'current', data: { tool: 'metrics.validate' } },
      { seq: 999, id: 'activity-999', type: 'tool.completed', run_id: 'current', data: { tool: 'metrics.validate', status: 'ok', duration_ms: 25 } },
    ])
    expect(source.url).toContain('after_seq=1000')
    expect(fetcher.mock.calls.some(([url]) => url.includes('after_seq=800&limit=200'))).toBe(true)
    expect(result.current.messages.map(message => message.text)).toEqual(['原有回答'])
    await act(async () => { await result.current.loadEarlier() })
    expect(fetcher.mock.calls.some(([url]) => url.includes('kind=messages&before_seq=800'))).toBe(true)
    expect(result.current.messages.map(message => message.text)).toEqual(['更早的问题', '原有回答'])
    expect(result.current.activity).toHaveLength(3)
  })

  it('实时事件去重，忽略私有或无归属记录，最多保留最近100条', async () => {
    setup({})
    const { result } = renderHook(() => useAgentConversation(context, true))
    const source = await connected()
    act(() => {
      for (let seq = 1; seq <= 120; seq += 1) {
        const event = { seq, run_id: seq < 60 ? 'previous' : 'current', type: seq % 2 ? 'tool.started' : 'tool.completed', data: { tool: 'metrics.validate', status: 'ok', duration_ms: 0, arguments: 'hidden' } }
        source.emit(event); source.emit(event)
      }
      source.emit({ seq: 121, run_id: 'current', type: 'model.reasoning', data: { tool: 'hidden', reasoning_content: 'hidden' } })
      source.emit({ seq: 122, type: 'tool.completed', data: { tool: 'missing run' } })
    })
    expect(result.current.activity).toHaveLength(100)
    expect(result.current.activity[0].seq).toBe(21)
    expect(result.current.activity[result.current.activity.length - 1]?.seq).toBe(120)
    expect(new Set(result.current.activity.map(event => event.seq)).size).toBe(100)
    expect(result.current.messages).toEqual([])
    expect(JSON.stringify(result.current.activity)).not.toContain('hidden')
  })

  it('辅助历史读取失败不影响消息订阅，也不无限重拉历史', async () => {
    const fetcher = setup({ next_event_seq: 501 }, [], true)
    const { result, rerender } = renderHook(({ open }) => useAgentConversation(context, open), { initialProps: { open: true } })
    const source = await connected()
    const historyRequests = () => fetcher.mock.calls.filter(([url]) => url.includes('after_seq=300&limit=200')).length
    const initialRequests = historyRequests()
    act(() => source.emit({ seq: 501, run_id: 'current', type: 'tool.completed', data: { tool: 'metrics.preview', status: 'discarded', duration_ms: -1, output: 'hidden' } }))
    expect(result.current.activity).toEqual([{ seq: 501, id: 'activity-501', run_id: 'current', type: 'tool.completed', data: { tool: 'metrics.preview', status: 'discarded' } }])
    expect(result.current.disconnected).toBe(false)
    rerender({ open: true })
    expect(historyRequests()).toBe(initialRequests)
    expect(source.url).toContain('after_seq=500')
  })

  it('重新打开去重恢复记录，快照之后的新动作仍由正常事件游标接收', async () => {
    const events: AgentEvent[] = [
      { seq: 9, type: 'tool.completed', run_id: 'current', data: { tool: 'catalog.search', status: 'ok', duration_ms: 2 } },
      { seq: 11, type: 'tool.started', run_id: 'current', data: { tool: 'catalog.get' } },
    ]
    const fetcher = setup({ next_event_seq: 11, events: events.slice(0, 1) }, events)
    const { result, rerender } = renderHook(({ open }) => useAgentConversation(context, open), { initialProps: { open: true } })
    await connected()
    await waitFor(() => expect(result.current.activity.map(event => event.seq)).toEqual([9, 11]))
    const historyRequests = fetcher.mock.calls.filter(([url]) => url.includes('after_seq=0&limit=200')).length
    rerender({ open: false }); rerender({ open: true })
    await connected()
    expect(result.current.activity.map(event => event.seq)).toEqual([9, 11])
    expect(fetcher.mock.calls.filter(([url]) => url.includes('after_seq=0&limit=200'))).toHaveLength(historyRequests)
    expect(Events.instances.find(source => !source.closed)?.url).toContain('after_seq=11')
  })
})

function editSetup(fail = false) {
  sessionStorage.setItem('agent-session:indicator-studio:single_product:test', 'session')
  const requests: Array<Record<string, any>> = []
  const stopNotice = '已停止自动处理，已提交的进度已保留。'
  const stopped: AgentRun = { run_id: 'run-2', session_id: 'session', message_id: 'q-2', session_revision: 2, run_revision: 2, status: 'cancelled', phase: 'thinking',
    response: { session_id: 'session', session_revision: 2, reply: { text: stopNotice }, artifacts: {} } }
  // The fixture session mirrors the server: an accepted replacement rewrites the transcript in place.
  const messages: Array<Record<string, any>> = [
    { seq: 1, id: 'q-1', run_id: 'run-1', speaker: 'user', text: '第一轮问题' },
    { seq: 2, id: 'run-1-reply', run_id: 'run-1', speaker: 'assistant', text: '第一轮回复' },
    { seq: 3, id: 'q-2', run_id: 'run-2', speaker: 'user', text: '旧问题' },
    { seq: 4, id: 'run-2-reply', run_id: 'run-2', speaker: 'assistant', text: stopNotice },
  ]
  vi.stubGlobal('EventSource', undefined)
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    const path = new URL(url, 'http://localhost').pathname
    if (path.endsWith('/messages')) {
      const body = JSON.parse(String(init?.body)); requests.push(body)
      if (fail) throw new Error('网络中断')
      messages.splice(messages.findIndex(item => item.id === body.edit_of_message_id), 2,
        { seq: 5, id: body.message_id, run_id: 'run-3', speaker: 'user', text: body.text, edit_of: body.edit_of_message_id },
        { seq: 6, id: 'run-3-reply', run_id: 'run-3', speaker: 'assistant', text: '新回复' })
      return response({ run_id: 'run-3', session_id: 'session', message_id: body.message_id, run_revision: 1, session_revision: 3, status: 'completed', phase: 'thinking',
        response: { session_id: 'session', session_revision: 3, reply: { text: '新回复' }, artifacts: {} } })
    }
    if (path.endsWith('/events')) return response({ items: [], has_more: false, next_event_seq: messages.length + 1 })
    if (path.includes('/runs/')) return response(stopped)
    return response({ session_id: 'session', session_revision: 2, page_context: context, next_event_seq: messages.length + 1, active_run: stopped, messages })
  }))
  return { requests, messages }
}

describe('useAgentConversation message editing', () => {
  it('替换停止后未答复的最后一条用户消息，不携带旧运行引用', async () => {
    const { requests } = editSetup()
    const { result } = renderHook(() => useAgentConversation(context, true))
    await waitFor(() => expect(result.current.messages).toHaveLength(4))
    await act(async () => { expect(await result.current.edit({ id: 'q-2', text: '旧问题' }, ' 改写后的问题 ')).not.toBeNull() })
    expect(requests[0]).toMatchObject({ text: '改写后的问题', edit_of_message_id: 'q-2', expected_session_revision: 2 })
    expect(requests[0]).not.toHaveProperty('resume_from_run_id')
    expect(result.current.messages.map(message => message.text)).toEqual(['第一轮问题', '第一轮回复', '改写后的问题', '新回复'])
    expect(result.current.error).toBe('')
  })

  it('失败时恢复原文和停止提示，并可用同一消息标识重试', async () => {
    const { requests, messages } = editSetup(true)
    const { result } = renderHook(() => useAgentConversation(context, true))
    await waitFor(() => expect(result.current.messages).toHaveLength(4))
    await act(async () => { expect(await result.current.edit({ id: 'q-2', text: '旧问题' }, '改写后的问题')).toBeNull() })
    expect(result.current.error).toBe('网络中断')
    expect(result.current.messages.map(message => message.text)).toEqual(['第一轮问题', '第一轮回复', '旧问题', '已停止自动处理，已提交的进度已保留。'])
    expect(result.current.failed?.editOf).toBe('q-2')
    // Pressing send again for the same attempt replays one identity instead of a second request.
    await act(async () => { expect(await result.current.edit({ id: 'q-2', text: '旧问题' }, '改写后的问题')).toBeNull() })
    await waitFor(() => expect(requests).toHaveLength(2))
    expect(requests[1]).toMatchObject({ message_id: requests[0].message_id, text: '改写后的问题', edit_of_message_id: 'q-2' })
    await act(async () => { expect(await result.current.retry()).toBeNull() })
    await waitFor(() => expect(requests).toHaveLength(3))
    expect(requests[2]).toMatchObject({ message_id: requests[0].message_id, text: '改写后的问题', edit_of_message_id: 'q-2' })
    expect(result.current.messages.map(message => message.text)).toEqual(['第一轮问题', '第一轮回复', '旧问题', '已停止自动处理，已提交的进度已保留。'])
    // A rejected attempt is not tombstoned: the stopped turn's later rows still merge normally.
    messages.push({ seq: 5, id: 'run-2-late', run_id: 'run-2', speaker: 'assistant', text: '停止后的补充' })
    await act(async () => { result.current.refresh() })
    await waitFor(() => expect(result.current.messages.map(message => message.text)).toContain('停止后的补充'))
  })

  it('替换事件在实时合并时移除旧消息，并抑制被替换运行的迟到回复', async () => {
    sessionStorage.setItem('agent-session:indicator-studio:single_product:test', 'session')
    const running: AgentRun = { run_id: 'run-3', session_id: 'session', message_id: 'q-3', session_revision: 3, run_revision: 1, status: 'running', phase: 'thinking' }
    const stopNotice = { seq: 4, id: 'run-2-reply', run_id: 'run-2', speaker: 'assistant' as const, text: '已停止自动处理，已提交的进度已保留。' }
    const replacement = { seq: 6, type: 'user.message', id: 'q-3', run_id: 'run-3', speaker: 'user' as const, text: '改写后的问题', edit_of: 'q-2' }
    const late = { seq: 7, type: 'assistant.message', id: 'run-2-late', run_id: 'run-2', speaker: 'assistant' as const, text: '迟到的停止提示' }
    const first = [
      { seq: 1, id: 'q-1', run_id: 'run-1', speaker: 'user' as const, text: '第一轮问题' },
      { seq: 2, id: 'run-1-reply', run_id: 'run-1', speaker: 'assistant' as const, text: '第一轮回复' },
      { seq: 3, id: 'q-2', run_id: 'run-2', speaker: 'user' as const, text: '旧问题' },
      stopNotice,
    ]
    let snapshot: Record<string, unknown> = { session_id: 'session', session_revision: 3, page_context: context, next_event_seq: 6, active_run: running, messages: first }
    vi.stubGlobal('EventSource', Events)
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      const parsed = new URL(url, 'http://localhost')
      if (parsed.pathname.endsWith('/runs/run-3')) return response(running)
      if (parsed.pathname.endsWith('/events')) return response({ items: [], has_more: false, last_seq: 5, next_event_seq: 6 })
      return response(snapshot)
    }))
    const { result } = renderHook(() => useAgentConversation(context, true))
    const source = await connected()
    await waitFor(() => expect(result.current.messages).toHaveLength(4))
    act(() => { source.emit(replacement); source.emit(late) })
    await waitFor(() => expect(result.current.messages.map(message => message.text)).toEqual(['第一轮问题', '第一轮回复', '改写后的问题']))
    // A later snapshot that still carries the stopped turn cannot resurrect it either.
    snapshot = { ...snapshot, next_event_seq: 8, messages: [...first, replacement, late] }
    await act(async () => { result.current.refresh() })
    await waitFor(() => expect(result.current.messages.map(message => message.text)).toEqual(['第一轮问题', '第一轮回复', '改写后的问题']))
    expect(result.current.messages.map(message => message.run_id)).toEqual(['run-1', 'run-1', 'run-3'])
  })

  it('运行中或非停止完成的一轮不能被修改', async () => {
    sessionStorage.setItem('agent-session:indicator-studio:single_product:test', 'session')
    const running: AgentRun = { run_id: 'run-2', session_id: 'session', message_id: 'q-2', session_revision: 2, run_revision: 2, status: 'running', phase: 'thinking' }
    vi.stubGlobal('EventSource', undefined)
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      const path = new URL(url, 'http://localhost').pathname
      if (path.endsWith('/events')) return response({ items: [], has_more: false, next_event_seq: 3 })
      if (path.includes('/runs/')) return response(running)
      return response({ session_id: 'session', session_revision: 2, page_context: context, next_event_seq: 3, active_run: running,
        messages: [{ seq: 1, id: 'q-2', run_id: 'run-2', speaker: 'user', text: '处理中的问题' }] })
    }))
    const { result } = renderHook(() => useAgentConversation(context, true))
    await waitFor(() => expect(result.current.messages).toHaveLength(1))
    await act(async () => { expect(await result.current.edit({ id: 'q-2', text: '处理中的问题' }, '改写')).toBeNull() })
    expect(result.current.messages.map(message => message.text)).toEqual(['处理中的问题'])
    expect(result.current.error).not.toBe('')
  })

  it('成功替换后发起端屏蔽被替换运行的迟到回复与旧快照', async () => {
    sessionStorage.setItem('agent-session:indicator-studio:single_product:test', 'session')
    const stopNotice = '已停止自动处理，已提交的进度已保留。'
    const stopped: AgentRun = { run_id: 'run-2', session_id: 'session', message_id: 'q-2', session_revision: 2, run_revision: 2, status: 'cancelled', phase: 'thinking',
      response: { session_id: 'session', session_revision: 2, reply: { text: stopNotice }, artifacts: {} } }
    const edited: AgentRun = { run_id: 'run-3', session_id: 'session', message_id: 'q-3', session_revision: 3, run_revision: 1, status: 'running', phase: 'thinking' }
    const earlier = [
      { seq: 1, id: 'q-1', run_id: 'run-1', speaker: 'user' as const, text: '第一轮问题' },
      { seq: 2, id: 'run-1-reply', run_id: 'run-1', speaker: 'assistant' as const, text: '第一轮回复' },
      { seq: 3, id: 'q-2', run_id: 'run-2', speaker: 'user' as const, text: '旧问题' },
      { seq: 4, id: 'run-2-reply', run_id: 'run-2', speaker: 'assistant' as const, text: stopNotice },
    ]
    const requests: Array<Record<string, any>> = []
    let replacementId = ''
    let snapshot: Record<string, unknown> = { session_id: 'session', session_revision: 2, page_context: context, next_event_seq: 5, active_run: stopped, messages: earlier }
    vi.stubGlobal('EventSource', Events)
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      const path = new URL(url, 'http://localhost').pathname
      if (path.endsWith('/messages')) {
        const body = JSON.parse(String(init?.body)); requests.push(body); replacementId = body.message_id
        snapshot = { ...snapshot, session_revision: 3, next_event_seq: 6, active_run: edited,
          messages: [...earlier.slice(0, 2), { seq: 5, id: replacementId, run_id: 'run-3', speaker: 'user', text: '改写后的问题', edit_of: 'q-2' }] }
        return response({ ...edited, message_id: body.message_id, response: { session_id: 'session', session_revision: 3, reply: { text: '改写后的回复' }, artifacts: {} } })
      }
      if (path.endsWith('/events')) return response({ items: [], has_more: false, last_seq: 4, next_event_seq: 6 })
      if (path.includes('/runs/run-3')) return response(edited)
      if (path.includes('/runs/')) return response(stopped)
      return response(snapshot)
    }))
    const { result } = renderHook(() => useAgentConversation(context, true))
    await waitFor(() => expect(result.current.messages).toHaveLength(4))
    await act(async () => { expect(await result.current.edit({ id: 'q-2', text: '旧问题' }, '改写后的问题')).not.toBeNull() })
    await waitFor(() => expect(result.current.messages.map(message => message.text)).toEqual(['第一轮问题', '第一轮回复', '改写后的问题', '改写后的回复']))
    // A late event of the replaced run cannot reappear for the initiating client.
    const source = await connected()
    act(() => source.emit({ seq: 5, type: 'assistant.message', id: 'run-2-late', run_id: 'run-2', speaker: 'assistant', text: '迟到的停止提示' }))
    // A stale snapshot that still carries the stopped turn cannot reappear either.
    snapshot = { ...snapshot, next_event_seq: 7, messages: [...earlier, { seq: 5, id: replacementId, run_id: 'run-3', speaker: 'user', text: '改写后的问题', edit_of: 'q-2' }] }
    await act(async () => { result.current.refresh() })
    await waitFor(() => expect(result.current.messages.map(message => message.text)).toEqual(['第一轮问题', '第一轮回复', '改写后的问题', '改写后的回复']))
    expect(result.current.messages.map(message => message.run_id)).toEqual(['run-1', 'run-1', 'run-3', 'run-3'])
    expect(requests).toHaveLength(1)
  })
})
