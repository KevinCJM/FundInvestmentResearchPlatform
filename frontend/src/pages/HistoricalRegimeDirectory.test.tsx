import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import HistoricalRegimeDirectory from './HistoricalRegimeDirectory'

const ok = (body: unknown, status = 200) => ({ ok: true, status, json: async () => body } as Response)

const v2Definition = {
  id: 'graph-1', revision: 3, schema_version: '2.0', name: '自由牛熊识别', description: '自定义图谱',
  graph: { nodes: [{ id: 'source', type: 'source.index', parameters: {}, inputs: {} }], edges: [], outputs: {} },
  states: [{ id: 'bull', label: '牛市' }], evaluation_targets: [], validation: { walk_forward: true }, usage_intent: 'taa', updated_at: '2026-09-04T08:00:00Z',
}

const v1Definition = {
  id: 'legacy-1', revision: 2, status: 'validated', name: '旧牛熊模型', description: '', template_id: 'legacy',
  target: { kind: 'index', name: '沪深300' }, features: {}, algorithm: { family: 'causal_filter', parameters: {} },
  states: [{ id: 'bull', label: '牛市', color: '#16a34a' }], validation: {}, usage_intent: 'research_display',
}

function LocationProbe() {
  const location = useLocation()
  return <output data-testid="location-probe">{location.pathname}{location.search}</output>
}

function renderDirectory() {
  return render(<MemoryRouter initialEntries={['/settings/scenario-algorithms']}><HistoricalRegimeDirectory /><LocationProbe /></MemoryRouter>)
}

describe('HistoricalRegimeDirectory', () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

  it('默认展示真实 V2 定义、模板、运行摘要与精确版本入口', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input)
      if (path.endsWith('/v2/definitions')) return ok({ items: [v2Definition] })
      if (path.endsWith('/templates/v2')) return ok({ items: [{ id: 'builtin-bull-bear', name: '牛熊震荡模板', description: '可编辑模板', tags: ['因果'] }] })
      if (path.endsWith('/runs')) return ok({ items: [{ id: 'run-1', definition_id: 'graph-1', definition_revision: 3, name: '正式运行', mode: 'realtime', created_at: '2026-09-04T09:00:00Z', series_included: false, publications: [{ id: 'p-1', usage: 'taa', published_at: '2026-09-04T10:00:00Z' }] }] })
      if (path.endsWith('/definitions')) return ok({ items: [v1Definition] })
      throw new Error(`Unexpected request: ${path}`)
    }))
    const user = userEvent.setup()
    renderDirectory()

    expect(await screen.findByRole('heading', { name: '历史情景识别' })).toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: '自由牛熊识别' })).toBeInTheDocument()
    expect(screen.getByText('战术配置')).toBeInTheDocument()
    expect(screen.getByText('正式运行').parentElement).toHaveTextContent('1')
    expect(screen.getByRole('link', { name: '使用模板：牛熊震荡模板' })).toHaveAttribute('href', '/settings/scenario-algorithms/workbench?template=builtin-bull-bear')
    expect(screen.getByRole('link', { name: '研究数据实验室' })).toHaveAttribute('href', '/settings/research-data-lab')
    expect(screen.getByRole('link', { name: '从模板开始' })).toHaveAttribute('href', '#regime-templates')
    expect(screen.getByRole('link', { name: '继续已有研究' })).toHaveAttribute('href', '#regime-saved')
    const templateRegion = screen.getByRole('region', { name: '选择研究模板' })
    const savedRegion = screen.getByRole('region', { name: '已有识别方案' })
    expect(templateRegion.compareDocumentPosition(savedRegion) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()

    await act(async () => {
      await user.clear(screen.getByLabelText('自由牛熊识别 精确 revision'))
      await user.type(screen.getByLabelText('自由牛熊识别 精确 revision'), '2')
    })
    expect(screen.getByRole('link', { name: '打开精确版本' })).toHaveAttribute('href', '/settings/scenario-algorithms/workbench?definition=graph-1&revision=2')
    await act(async () => { await user.click(screen.getByRole('link', { name: '使用模板：牛熊震荡模板' })) })
    expect(screen.getByTestId('location-probe')).toHaveTextContent('/settings/scenario-algorithms/workbench?template=builtin-bull-bear')
  })

  it('没有模板时保留自由构建入口，不把用户困在模板选择', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ok({ items: [] })))
    const user = userEvent.setup()
    renderDirectory()

    expect(await screen.findByText(/当前没有可用模板/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '从空白画板自由构建' })).toHaveAttribute('href', '/settings/scenario-algorithms/workbench')
    await act(async () => { await user.click(screen.getByRole('link', { name: '自由构建' })) })
    expect(screen.getByTestId('location-probe')).toHaveTextContent('/settings/scenario-algorithms/workbench')
  })

  it('V1 仅在只读迁移区出现，调用真实复制接口后打开新 V2 精确版本', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input)
      if (path.includes('/definitions/legacy-1/copy-to-v2?revision=2')) return ok({ source_v1: { id: 'legacy-1', revision: 2 }, definition: { ...v2Definition, id: 'migrated-v2', revision: 1 }, inference: { valid: true, errors: [], warnings: [] } })
      if (path.endsWith('/v2/definitions')) return ok({ items: [] })
      if (path.endsWith('/templates/v2')) return ok({ items: [] })
      if (path.endsWith('/runs')) return ok({ items: [] })
      if (path.endsWith('/definitions')) return ok({ items: [v1Definition] })
      throw new Error(`Unexpected request: ${path} ${init?.method || 'GET'}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    renderDirectory()

    await user.click(await screen.findByText(/V1 只读迁移区/))
    expect(await screen.findByRole('heading', { name: '旧牛熊模型' })).toBeInTheDocument()
    expect(screen.queryByRole('textbox', { name: /旧牛熊模型/ })).not.toBeInTheDocument()
    await act(async () => { await user.click(screen.getByRole('button', { name: '复制为 V2 图谱' })) })
    await waitFor(() => expect(screen.getByTestId('location-probe')).toHaveTextContent('/settings/scenario-algorithms/workbench?definition=migrated-v2&revision=1'))
    expect(fetchMock.mock.calls.some(([path, init]) => String(path).includes('/definitions/legacy-1/copy-to-v2?revision=2') && init?.method === 'POST')).toBe(true)
  })
})


it('峰谷模板仅放入事后识别分组，并携带正确模式进入工作台', async () => {
  vi.stubGlobal('fetch', vi.fn(async input => String(input).endsWith('/templates/v2') ? ok({ items: [
    { id: 'peak-trough-ps-v2', name: '峰谷定界法', default_mode: 'retrospective', supported_modes: ['retrospective'] },
    { id: 'bull-bear-causal-v2', name: '指数牛熊震荡', default_mode: 'realtime' },
  ] }) : ok({ items: [] })))
  renderDirectory()
  await screen.findByRole('region', { name: '事后识别' })
  const offline = screen.getByRole('region', { name: '事后识别' })
  expect(within(offline).getByRole('link', { name: '使用模板：峰谷定界法' })).toHaveAttribute('href', '/settings/scenario-algorithms/workbench?template=peak-trough-ps-v2&mode=retrospective')
  expect(within(screen.getByRole('region', { name: '实时识别' })).queryByText('峰谷定界法')).not.toBeInTheDocument()
})
