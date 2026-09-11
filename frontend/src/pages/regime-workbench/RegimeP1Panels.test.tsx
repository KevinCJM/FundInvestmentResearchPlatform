import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { RegimeGraphDefinition, RegimeNodeSchema } from '../../services/regimeGraph'
import RegimeExperimentPanel from './RegimeExperimentPanel'
import RegimeGraphAssetsPanel from './RegimeGraphAssetsPanel'
import RegimeValidationPanel from './RegimeValidationPanel'

const fixedExecution = {
  execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0,
  request_time_compilation: 0, kernel_signatures: { regime_graph_v2_experiment: ['fixed'] },
}
const ok = (body: unknown, status = 200) => ({ ok: true, status, json: async () => body } as Response)
const definition: RegimeGraphDefinition = {
  id: 'graph-1', revision: 2, schema_version: '2.0', name: '牛熊研究', description: '',
  graph: { nodes: [
    { id: 'source-1', type: 'source.index', type_version: 1, label: '沪深300', parameters: { kind: 'index', ts_code: '000300.SH', source_api: 'index_daily', field: 'close' }, inputs: {} },
    { id: 'constant-1', type: 'source.constant', type_version: 1, label: '阈值常量', parameters: { value: 0 }, inputs: {} },
    { id: 'ema-1', type: 'transform.ema', type_version: 1, label: '趋势滤波', parameters: { window: 20 }, inputs: { series: { node_id: 'source-1', port: 'value' } } },
  ], edges: [{ source: { node_id: 'source-1', port: 'value' }, target: { node_id: 'ema-1', port: 'series' } }], outputs: {} },
  states: [{ id: 'bull', label: '牛市', color: '#16a34a', order: 0 }, { id: 'bear', label: '熊市', color: '#dc2626', order: 1 }],
  evaluation_targets: [], validation: { walk_forward: true, folds: 4 }, usage_intent: 'research_display',
}
const schemas: RegimeNodeSchema[] = [{
  id: 'transform.ema', label: '因果 EMA', category: 'transform', inputs: [{ id: 'series' }], outputs: [{ id: 'value' }],
  parameter_schema: { type: 'object', properties: { window: { type: 'integer', label: '窗口', minimum: 2 } } },
}]

function ValidationHarness() {
  const [value, setValue] = useState(definition)
  return <><RegimeValidationPanel definition={value} onChange={setValue} /><output data-testid="definition-json">{JSON.stringify(value)}</output></>
}

describe('regime V2 P1 panels', () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

  it('从图中真实数据源建立独立评估目标，并编辑走步验证规则', async () => {
    const user = userEvent.setup()
    render(<ValidationHarness />)
    expect(screen.queryByRole('button', { name: '添加 阈值常量' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '添加 沪深300' }))
    fireEvent.change(screen.getByLabelText('走步验证折数'), { target: { value: '6' } })
    const payload = JSON.parse(screen.getByTestId('definition-json').textContent || '{}')
    expect(payload.evaluation_targets[0]).toMatchObject({ name: '沪深300', primary: true, source: { kind: 'index', ts_code: '000300.SH' } })
    expect(payload.validation.folds).toBe(6)
  })

  it('保存所选子图时剔除选择外的悬空输入并调用版本化资产接口', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input)
      if (path.endsWith('/v2/graph-assets') && init?.method === 'POST') return ok({ id: 'asset-1', kind: 'subgraph', revision: 1, name: '趋势片段', content_hash: 'hash' }, 201)
      if (path.endsWith('/v2/graph-assets')) return ok({ items: [] })
      throw new Error(`Unexpected request: ${path}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    const onNotice = vi.fn()
    render(<RegimeGraphAssetsPanel definition={definition} selectedNodeIds={['ema-1']} valid onLoadDefinition={vi.fn()} onInsertGraph={vi.fn()} onError={vi.fn()} onNotice={onNotice} />)

    await user.selectOptions(screen.getByLabelText('图谱资产类型'), 'subgraph')
    await user.type(screen.getByLabelText('图谱资产名称'), '趋势片段')
    await user.click(screen.getByRole('button', { name: '另存为新子图' }))
    await waitFor(() => expect(onNotice).toHaveBeenCalledWith('用户子图已保存 · r1。'))
    const call = fetchMock.mock.calls.find(([, init]) => init?.method === 'POST')
    const body = JSON.parse(String(call?.[1]?.body))
    expect(body.kind).toBe('subgraph')
    expect(body.asset.graph.nodes).toHaveLength(1)
    expect(body.asset.graph.nodes[0]).toMatchObject({ id: 'ema-1', inputs: {} })
    expect(body.asset.graph.edges).toEqual([])
  })

  it('使用已显式预热计划运行真实批量实验，并展示基准、排名和分歧区间', async () => {
    const experiment = {
      id: 'experiment-1', schema_version: '2.0', definition_id: 'graph-1', definition_revision: 2, mode: 'realtime',
      parameter_grid: [{ node_id: 'ema-1', parameter: 'window', values: [10, 40] }], ranking_metric: 'agreement',
      baseline: { classified_ratio: 0.9, flip_rate: 0.1, state_switches: 4 }, candidate_count: 2,
      ranking: [{ rank: 1, candidate_id: 'candidate-1', rank_value: 0.94, parameter_differences: [{ node_id: 'ema-1', parameter: 'window', baseline: 20, candidate: 40 }], metrics: { agreement: 0.94, classified_ratio: 0.91, flip_rate: 0.08, mean_boundary_distance_observations: 2 }, disagreement_intervals: [{ start_date: '2024-01-02', end_date: '2024-01-05', observations: 4 }] }],
      calculation_audit: fixedExecution, immutable: true,
    }
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input)
      if (path.endsWith('/v2/experiments') && init?.method === 'POST') return ok(experiment, 201)
      if (path.includes('/v2/experiments?definition_id=')) return ok({ items: [] })
      throw new Error(`Unexpected request: ${path}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    const onNotice = vi.fn()
    render(<RegimeExperimentPanel definition={definition} schemas={schemas} dirty={false} valid mode="realtime" asOf="" preparedPlan={{ plan_id: 'plan-1', compile_token: 'token-1', graph_hash: 'hash', runtime_audit: fixedExecution }} onPrepared={vi.fn()} onError={vi.fn()} onNotice={onNotice} />)

    await user.type(screen.getByLabelText('实验维度1候选值'), '10, 40')
    await user.click(screen.getByRole('button', { name: '运行参数网格' }))
    expect(await screen.findByText('实验结果 · experiment-1')).toBeInTheDocument()
    expect(screen.getByText(/2024-01-02 → 2024-01-05/)).toBeInTheDocument()
    const call = fetchMock.mock.calls.find(([, init]) => init?.method === 'POST')
    expect(JSON.parse(String(call?.[1]?.body))).toMatchObject({ definition: { schema_version: '2.0', id: 'graph-1', revision: 2 }, compile_token: 'token-1', parameter_grid: [{ node_id: 'ema-1', parameter: 'window', values: [10, 40] }] })
    expect(onNotice).toHaveBeenCalledWith(expect.stringContaining('服务端 NJIT 链路排序'))
  })
})
