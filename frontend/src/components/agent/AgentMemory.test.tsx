import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { useState } from 'react'
import AgentMemory from './AgentMemory'
import type { AgentConversationState } from './useAgentConversation'
import type { AgentMemorySource } from '../../services/agent'

const api = vi.hoisted(() => ({ load: vi.fn(), session: vi.fn(), decide: vi.fn(), revoke: vi.fn() }))
vi.mock('../../services/agent', () => ({ fetchAgentMemory: api.load, fetchAgentSession: api.session, decideAgentMemory: api.decide, revokeAgentMemory: api.revoke }))
const source: AgentMemorySource = { memory_id: 'memory-a', version: 1, scope: 'indicator_center', object_id: 'scope', key: 'reply.language', text: '用中文', source_session_id: 'older', source_message_id: 'm1', accepted_at: '2026-09-21' }
const refresh = vi.fn()
const chat = (props: object = {}) => ({ session: { session_id: 's', session_revision: 1,
  memory_proposals: [{ proposal_id: 'p', status: 'pending', summary: '用英文', key: 'reply.language', object_id: 'scope' }], ...props }, refresh, acceptSession: vi.fn() }) as unknown as AgentConversationState
afterEach(() => { cleanup(); vi.clearAllMocks() })

it('创建回执补全为同版本会话快照时，加载本轮已引用的确认偏好', async () => {
  api.load.mockResolvedValue({ items: [source] })
  const initial = { session: { session_id: 's', session_revision: 1 }, refresh } as unknown as AgentConversationState
  const view = render(<AgentMemory chat={initial} busy={false} />)
  expect(api.load).not.toHaveBeenCalled()
  view.rerender(<AgentMemory chat={{ ...initial, session: { ...initial.session!, memory_proposals: [], memory_sources: [source] } }} busy={false} />)
  expect(await screen.findByText(source.text)).toBeInTheDocument()
  expect(api.load).toHaveBeenCalledOnce()
  expect(screen.getByText('本轮已引用的确认偏好')).toBeInTheDocument()
})

it('独立显示待确认提案、旧偏好和来源，明确替换与撤销且不触发业务保存', async () => {
  api.load.mockResolvedValue({ items: [source] }); api.decide.mockResolvedValue({}); api.revoke.mockResolvedValue({})
  api.session.mockResolvedValue(chat().session)
  render(<AgentMemory chat={chat({ memory_sources: [source] })} busy={false} />)
  fireEvent.click(screen.getByText('偏好与记忆（1 项待确认）'))
  expect(await screen.findByRole('button', { name: '替换此偏好' })).toBeEnabled()
  expect(screen.getByText('本轮已引用的确认偏好')).toBeVisible()
  expect(screen.getByText('人工确认于 2026-09-21')).toBeVisible()
  fireEvent.click(screen.getByRole('button', { name: '替换此偏好' }))
  await waitFor(() => expect(api.decide).toHaveBeenCalledWith('s', 'p', 'accept', source))
  await waitFor(() => expect(refresh).toHaveBeenCalled())
  await waitFor(() => expect(screen.getByRole('button', { name: '撤销记忆' })).toBeEnabled())
  fireEvent.click(screen.getByRole('button', { name: '撤销记忆' }))
  await waitFor(() => expect(api.revoke).toHaveBeenCalledWith('s', source, expect.any(String)))
})

it('运行中禁用决定并说明原因，失败保留候选和重试入口', async () => {
  api.load.mockResolvedValue({ items: [], legacy_unavailable: true }); api.decide.mockRejectedValue(new Error('决定保存失败'))
  const view = render(<AgentMemory chat={chat()} busy />)
  fireEvent.click(screen.getByText('偏好与记忆（1 项待确认）'))
  expect(screen.getByRole('button', { name: '接受记忆' })).toBeDisabled()
  expect(await screen.findByText('旧记忆缺少可核验来源，本轮不会引用；如需继续使用，请重新提出并确认。')).toBeVisible()
  expect(screen.getByText('请先完成或停止当前任务，再确认记忆。')).toBeVisible()
  view.rerender(<AgentMemory chat={chat()} busy={false} />)
  fireEvent.click(screen.getByRole('button', { name: '不记住' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('决定保存失败')
  expect(screen.getByText('用英文')).toBeVisible()
  expect(screen.getByRole('button', { name: '重新读取' })).toBeEnabled()
})

it.each((['accept', 'reject', 'revoke'] as const).flatMap(decision => [true, false].map(lost => ({ decision, lost }))))('恢复$decision（响应丢失=$lost），读取失败不重发或改变决定', async ({ decision, lost }) => {
  const initial = chat({ memory_proposals: decision === 'revoke' ? [] : chat().session!.memory_proposals }).session!
  let persisted = false, failSession = true, failItems = false
  api.load.mockImplementation(async () => {
    if (failItems) throw new Error('记忆列表读取失败')
    return { items: decision === 'revoke' ? (persisted ? [] : [source]) : persisted && decision === 'accept' ? [source] : [] }
  })
  api.session.mockImplementation(async () => {
    if (failSession) throw new Error('会话读取失败')
    return { ...initial, session_revision: 2, memory_proposals: decision === 'revoke' ? [] : [{ ...initial.memory_proposals![0], status: decision === 'accept' ? 'accepted' : 'rejected' }] }
  })
  const mutate = decision === 'revoke' ? api.revoke : api.decide
  mutate.mockImplementation(async () => { persisted = true; if (lost) throw new Error('操作响应丢失'); return {} })
  function Conversation() {
    const [session, acceptSession] = useState(initial)
    return <AgentMemory chat={{ session, acceptSession, refresh } as unknown as AgentConversationState} busy={false} />
  }
  render(<Conversation />)
  await waitFor(() => expect(api.load).toHaveBeenCalledOnce())
  fireEvent.click(screen.getByText(`偏好与记忆（${decision === 'revoke' ? 0 : 1} 项待确认）`))
  const action = screen.getByRole('button', { name: decision === 'accept' ? '接受记忆' : decision === 'reject' ? '不记住' : '撤销记忆' })
  await waitFor(() => expect(action).toBeEnabled())
  fireEvent.click(action)
  expect(await screen.findByRole('alert')).toHaveTextContent(lost ? '操作响应丢失' : '会话读取失败')
  expect(action).toBeDisabled()
  fireEvent.click(screen.getByRole('button', { name: '重新读取' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('会话读取失败')
  expect(api.load).toHaveBeenCalledOnce()
  expect(action).toBeDisabled()
  failSession = false; failItems = true
  fireEvent.click(screen.getByRole('button', { name: '重新读取' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('记忆列表读取失败')
  expect(screen.queryByRole('button', { name: '替换此偏好' })).not.toBeInTheDocument()
  for (const button of screen.queryAllByRole('button', { name: /接受记忆|不记住|撤销记忆/ })) expect(button).toBeDisabled()
  failItems = false
  fireEvent.click(screen.getByRole('button', { name: '重新读取' }))
  await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument())
  expect(screen.queryByRole('button', { name: /接受记忆|不记住|替换此偏好/ })).not.toBeInTheDocument()
  if (decision === 'accept') await waitFor(() => expect(screen.getByRole('button', { name: '撤销记忆' })).toBeEnabled())
  else expect(screen.queryByRole('button', { name: '撤销记忆' })).not.toBeInTheDocument()
  expect(mutate).toHaveBeenCalledOnce()
})
