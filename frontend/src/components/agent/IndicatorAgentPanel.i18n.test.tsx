import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import AgentPanel from './IndicatorAgentPanel'
import { i18n } from '../../i18n/runtime'
import type { AgentDraft, AgentEvent } from '../../services/agent'

const draft: AgentDraft = { valid: true, draft_revision: 1, definition_hash: 'hash', definition: {
  name: '用户命名', result_kind: 'time_series', expression: '',
  series_outputs: [{ id: 'result', label: '用户通道', expression: 'mean(rolling_window(returns,20))' }],
  parameter_schema: [{ id: 'window', label: '用户参数', default: 20, minimum: 1, maximum: 100, step: 1 }],
} }
const messages: AgentEvent[] = [
  { id: 'question', speaker: 'user', text: '用户原始问题' },
  { id: 'reply', speaker: 'assistant', run_id: 'run', text: '模型原始回答', artifacts: { draft } },
]
const conversation = {
  session: { session_id: 'session' }, run: { run_id: 'run', status: 'completed', phase: 'thinking' }, messages, draft,
  activity: [{ seq: 1, type: 'tool.completed', run_id: 'run', data: { tool: 'metrics.validate', status: 'ok', duration_ms: 12 } }],
  busy: false, sending: false, error: '后端原始错误', preview: null, setError: vi.fn(), send: vi.fn(),
}
vi.mock('./useAgentConversation', async importOriginal => ({ ...await importOriginal<typeof import('./useAgentConversation')>(), default: () => conversation }))
afterEach(async () => { cleanup(); vi.unstubAllGlobals(); conversation.run.status = 'completed'; conversation.send.mockClear(); await i18n.changeLanguage('zh-CN') })

describe('AgentPanel localization', () => {
  it('英文保存确认展示冻结名称、全部通道和同名影响，取消不提交', async () => {
    await i18n.changeLanguage('en-US')
    const fetcher = vi.fn(async (url: string) => ({ ok: true, json: async () => url.endsWith('/commit-preview')
      ? { confirmation_id: 'c', definition_hash: 'hash', draft_revision: 1, definition: draft.definition, preview_status: 'valid',
          impact: { action: 'create', name: '用户命名', context_kind: 'single_product', target: null, name_conflict_indicator_id: 'existing' } }
      : { configured: true } }))
    const confirm = vi.fn(() => false)
    vi.stubGlobal('fetch', fetcher); vi.stubGlobal('confirm', confirm)
    render(<AgentPanel pageContext={{ page: 'indicator-studio', page_instance_id: 'test', context_revision: 0, view_state: 'inherit', calculation: { context_kind: 'single_product', period: '1Y', targets: [] } }} draft={{}} />)
    fireEvent.click(screen.getByRole('button', { name: 'Open AI Assistant' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Save indicator' }))
    await waitFor(() => expect(confirm).toHaveBeenCalledOnce())
    const text = vi.mocked(window.confirm).mock.calls[0][0]
    expect(text).toContain('An indicator with this name already exists')
    expect(text).toContain('用户命名')
    expect(text).toContain('用户通道 = mean(rolling_window(returns,20))')
    expect(fetcher.mock.calls.some(([url]) => url.endsWith('/commit'))).toBe(false)
  })

  it('英文继续按钮保持服务端识别的恢复命令', async () => {
    await i18n.changeLanguage('en-US')
    conversation.run.status = 'paused'
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => ({ configured: true }) })))
    render(<AgentPanel pageContext={{ page: 'indicator-studio', page_instance_id: 'test', context_revision: 0, view_state: 'unknown', calculation: {} }} draft={{}} />)
    fireEvent.click(screen.getByRole('button', { name: 'Open AI Assistant' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Continue analysis' }))
    expect(conversation.send).toHaveBeenCalledWith('继续')
  })
  it('语言切换即时翻译界面及动态摘要，并保持用户、模型、公式和后端错误原文', async () => {
    await i18n.changeLanguage('en-US')
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => ({ configured: true, model: 'original-model' }) })))
    render(<AgentPanel pageContext={{ page: 'indicator-studio', page_instance_id: 'test', context_revision: 0, view_state: 'unknown', calculation: { context_kind: 'single_product', period: '1Y', targets: [] } }} draft={{ expression: '' }} onApplyDraft={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Open AI Assistant' }))
    expect(await screen.findByText('Configured model: original-model')).not.toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'Context and model' }))
    expect(screen.getByText('Configured model: original-model')).toBeVisible()
    expect(screen.getByText('Indicator definition · No product selection needed')).toBeVisible()
    expect(screen.getByRole('button', { name: 'Close AI Assistant' })).toBeVisible()
    expect(screen.getByRole('button', { name: 'Clear context' })).toBeVisible()
    expect(screen.getByText('Definition validated')).toBeVisible()
    expect(screen.getByText('Recent activity (1)')).toBeVisible()
    expect(screen.getByText('Validating the definition')).toBeInTheDocument()
    expect(screen.getByText('Completed · 12 ms')).toBeInTheDocument()
    expect(screen.getByText('Default 20; at least 1, at most 100; step 1')).toBeInTheDocument()
    for (const text of ['用户原始问题', '模型原始回答', '用户命名', '后端原始错误', '用户通道 = mean(rolling_window(returns,20))']) expect(screen.getByText(text)).toBeInTheDocument()
    await act(async () => { await i18n.changeLanguage('zh-CN') })
    expect(screen.getByRole('button', { name: '关闭 AI 助手' })).toBeVisible()
    expect(screen.getByRole('button', { name: '清空上下文' })).toBeVisible()
    expect(screen.getByRole('button', { name: '查看完整公式（1 个输出）' })).toBeVisible()
    expect(screen.getByText('查看处理记录（最近 1 条）')).toBeVisible()
    expect(screen.getByText('默认 20；不小于 1，不大于 100；步长 1')).toBeInTheDocument()
    expect(screen.getByText('配置模型：original-model')).toBeVisible()
  })
})
