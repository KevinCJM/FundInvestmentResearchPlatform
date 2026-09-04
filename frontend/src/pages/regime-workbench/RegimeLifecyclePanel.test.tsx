import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import RegimeLifecyclePanel from './RegimeLifecyclePanel'
import type { RegimeGraphDefinition } from '../../services/regimeGraph'

const fixedExecution = {
  execution_backend: 'numba_njit_fixed_signature',
  nopython: true,
  object_mode: 0,
  python_fallback: 0,
  request_time_compilation: 0,
  kernel_signatures: { regime_graph: ['float64[:]->int64[:]'] },
}

const definition: RegimeGraphDefinition = {
  id: 'REGIME-V2-1', revision: 3, schema_version: '2.0', name: '牛熊识别', description: '',
  graph: { nodes: [{ id: 'source', type: 'source.index', type_version: 1, parameters: { ts_code: '000300.SH' }, inputs: {} }, { id: 'model', type: 'model.threshold', type_version: 1, parameters: {}, inputs: { value: { node_id: 'source', port: 'value' } } }], outputs: { state: { node_id: 'model', port: 'state' } } },
  states: [{ id: 'bull', label: '牛市' }, { id: 'bear', label: '熊市' }], evaluation_targets: [], validation: { walk_forward: true }, usage_intent: 'research_display',
}

const run = {
  id: 'RUN-FORMAL-1', schema_version: '2.0', definition_id: definition.id, definition_revision: definition.revision,
  name: definition.name, mode: 'realtime', created_at: '2026-09-04T08:00:00Z', immutable: true,
  causality: { publish_eligible_usages: ['research_display', 'product_research'] },
  calculation_audits: [fixedExecution], series: [{ date: '2024-01-02', value: 3500, state_id: 'bull' }],
  artifact_manifest: { node_outputs: { checksum: 'sha256:nodes', arrays: [{ node_id: 'model', port: 'state' }] }, series: { checksum: 'sha256:series', row_count: 1 } },
  evaluation_results: {
    equity: {
      id: 'equity', name: '沪深300全收益', primary: true,
      source: { kind: 'index', ts_code: '000300.SH' },
      snapshot: { kind: 'index', fingerprint: 'sha256:equity-evaluation' },
      conditional_metrics: [
        { state_id: 'bull', state_label: '牛市', observations: 20, annualized_return: 0.18, volatility: 0.12, max_drawdown: -0.05, sharpe: 1.5, win_rate: 0.62 },
        { state_id: 'bear', state_label: '熊市', observations: 10, annualized_return: -0.08, volatility: 0.2, max_drawdown: -0.16, sharpe: -0.4, win_rate: 0.35 },
      ],
    },
    commodity: {
      id: 'commodity', name: '南华商品指数', primary: false,
      source: { kind: 'index', ts_code: 'NHCI' },
      snapshot: { kind: 'index', fingerprint: 'sha256:commodity-evaluation' },
      conditional_metrics: [
        { state_id: 'bull', state_label: '牛市', observations: 20, annualized_return: 0.06, volatility: 0.1, max_drawdown: -0.04, sharpe: 0.6, win_rate: 0.55 },
      ],
    },
  },
  publications: [],
}

const ok = (body: unknown) => ({ ok: true, status: 200, json: async () => body } as Response)

describe('RegimeLifecyclePanel', () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

  it('正式运行必须由用户先显式预热，运行请求只传已保存引用和 token', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input)
      if (path.endsWith(`/runs?definition_id=${definition.id}`)) return ok({ items: [] })
      if (path.endsWith('/prepare')) return ok({ plan_id: 'PLAN-3', compile_token: 'TOKEN-3', graph_hash: 'graph-3', runtime_audit: fixedExecution })
      if (path.endsWith('/run') && init?.method === 'POST') return ok(run)
      throw new Error(`Unexpected request: ${path} ${init?.method || 'GET'}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<RegimeLifecyclePanel definition={definition} dirty={false} valid mode="realtime" asOf="2025-12-31" onError={vi.fn()} onNotice={vi.fn()} />)

    const runButton = await screen.findByRole('button', { name: '运行已保存版本' })
    expect(runButton).toBeDisabled()
    expect(fetchMock.mock.calls.some(([path]) => String(path).endsWith('/prepare'))).toBe(false)

    await user.click(screen.getByRole('button', { name: '显式预热计划' }))
    expect(await screen.findByText(/计算计划已就绪/)).toBeInTheDocument()
    await waitFor(() => expect(runButton).toBeEnabled())
    await user.click(runButton)
    expect(await screen.findByText('sha256:series')).toBeInTheDocument()
    expect(screen.getByText('sha256:nodes')).toBeInTheDocument()
    expect(screen.getByRole('region', { name: '正式运行评价目标结果' })).toBeInTheDocument()
    expect(screen.getByText('沪深300全收益')).toBeInTheDocument()
    expect(screen.getByText('南华商品指数')).toBeInTheDocument()
    await user.click(screen.getByText('查看分状态条件表现（2 个状态）'))
    expect(screen.getByRole('table', { name: '沪深300全收益分状态条件表现' })).toBeInTheDocument()
    expect(screen.getByText('18.00%')).toBeInTheDocument()

    const runCall = fetchMock.mock.calls.find(([path]) => String(path).endsWith('/run'))
    expect(JSON.parse(String(runCall?.[1]?.body))).toEqual({ definition: { schema_version: '2.0', id: definition.id, revision: 3 }, mode: 'realtime', compile_token: 'TOKEN-3', as_of: '2025-12-31' })
    expect(fetchMock.mock.calls.filter(([path]) => String(path).endsWith('/prepare'))).toHaveLength(1)
  })

  it('比较和发布均调用正式运行 API，不在浏览器计算一致率', async () => {
    const second = { ...run, id: 'RUN-FORMAL-2', created_at: '2026-09-04T09:00:00Z' }
    const summaries = [run, second].map(({ series: _series, calculation_audits: _audits, ...item }) => ({ ...item, series_included: false, series_detail_endpoint: `/api/historical-regimes/runs/${item.id}` }))
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input)
      if (path.includes('/runs?definition_id=')) return ok({ items: summaries })
      if (path.endsWith('/compare')) return ok({ run_ids: [run.id, second.id], reference_run_id: run.id, agreement_rate: 0.91, disagreement_periods: [], pairwise: [{ left_run_id: run.id, right_run_id: second.id, agreement_rate: 0.91, boundary_distance: 2 }], execution: fixedExecution })
      if (path.endsWith(`/runs/${run.id}/publish`) && init?.method === 'POST') return ok({ run_id: run.id, publication: { id: 'PUB-1', usage: 'research_display', published_at: '2026-09-04' }, publications: [{ id: 'PUB-1', usage: 'research_display', published_at: '2026-09-04' }] })
      if (path.endsWith(`/runs/${run.id}`)) return ok({ ...run, publications: [{ id: 'PUB-1', usage: 'research_display', published_at: '2026-09-04' }] })
      if (path.endsWith(`/runs/${second.id}`)) return ok(second)
      throw new Error(`Unexpected request: ${path}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<RegimeLifecyclePanel definition={definition} dirty={false} valid mode="realtime" asOf="" onError={vi.fn()} onNotice={vi.fn()} />)
    await screen.findByText('2 条')
    expect(await screen.findByText('sha256:series')).toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([path]) => String(path).endsWith(`/runs/${run.id}`))).toBe(true)
    await user.click(screen.getByLabelText(`选择比较${run.id}`))
    await user.click(screen.getByLabelText(`选择比较${second.id}`))
    await user.click(screen.getByRole('button', { name: '比较 2 个版本' }))
    expect((await screen.findAllByText(/91.00%/)).length).toBeGreaterThan(0)

    await user.click(screen.getByRole('button', { name: '发布运行' }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => String(path).endsWith(`/runs/${run.id}/publish`))).toBe(true))
    expect(JSON.parse(String(fetchMock.mock.calls.find(([path]) => String(path).endsWith('/compare'))?.[1]?.body)).run_ids).toEqual([run.id, second.id])
  })
})
