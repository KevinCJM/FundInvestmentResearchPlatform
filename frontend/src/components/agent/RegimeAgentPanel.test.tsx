import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import RegimeAgentPanel from './RegimeAgentPanel'
import { mergeRegimeAgentDraft, regimeAgentSnapshot } from '../../services/regimeAgent'
import { createBlankRegimeDefinition, type RegimeGraphDefinition } from '../../services/regimeGraph'

const fixture = vi.hoisted(() => ({ infer: vi.fn(), props: null as any, stale: false }))
vi.mock('../../services/regimeGraph', async importOriginal => ({ ...await importOriginal<object>(), inferRegimeGraph: fixture.infer }))
vi.mock('../../app/ResearchContext', () => ({ useResearchDay: () => '2019-12-31', useResearchContextIdentity: () => 'pit-2019' }))
vi.mock('./AgentPanel', () => ({ default: (props: any) => {
  fixture.props = props
  const draft = { definition: proposal, valid: true, definition_hash: 'hash', artifact_kind: 'regime_graph', stale: fixture.stale }
  return props.renderArtifacts({ event: { run_id: 'r', artifacts: { draft } }, eventKey: 'r', isCurrent: true,
    chat: { draft, session: { page_context: props.pageContext } }, busy: props.busy, close: vi.fn() })
} }))

const proposal = { schema_version: '2.0', name: '趋势识别', description: '观察趋势与阈值', graph: {
  nodes: [{ id: 'market', type: 'source.index', parameters: { ts_code: '000300.SH' } },
    { id: 'model', type: 'model.threshold', parameters: { upper: 0.1, lower: -0.1 }, inputs: { value: { node_id: 'market', port: 'value' } } }],
  outputs: { state: { node_id: 'model', port: 'state' } },
}, states: [{ id: 'bull', label: '牛市' }, { id: 'bear', label: '熊市' }] }
const current = (): RegimeGraphDefinition => ({ ...createBlankRegimeDefinition(), id: 'saved', revision: 4,
  study: { purpose: 'realtime_recognition', family: 'custom', reference: { reference_id: 'ref', reference_revision: 2 } as any, qualification_id: 'old-proof' },
  evaluation_targets: [{ id: 'benchmark', name: '沪深300', source: { type: 'source.index' }, primary: true }],
})
afterEach(() => { cleanup(); fixture.infer.mockReset(); fixture.stale = false })

it('保留研究关联和保存身份，清除旧资格；快照不携带结果', () => {
  const base = current()
  const next = mergeRegimeAgentDraft(base, { ...proposal, id: 'forged', study: { qualification_id: 'fake' } })
  expect(next.id).toBe('saved'); expect(next.revision).toBe(4)
  expect(next.study?.reference).toEqual(base.study?.reference)
  expect(next.study?.qualification_id).toBeUndefined()
  expect(next.evaluation_targets).toEqual(base.evaluation_targets)
  expect(base.study?.qualification_id).toBe('old-proof')
  const snapshot = regimeAgentSnapshot(next, 'realtime', '2019-12-31', false)
  expect(Object.keys(snapshot.sections)).toEqual(['editing'])
  expect(snapshot.sections.editing).toMatchObject({ mode: 'realtime', as_of: '2019-12-31' })
})

it('点击才校验并回填，随后提示预览和保存', async () => {
  const onApply = vi.fn()
  fixture.infer.mockResolvedValue({ valid: true, errors: [], temporal_capability: { realtime_supported: true } })
  render(<RegimeAgentPanel definition={current()} mode="realtime" onApply={onApply} />)
  expect(onApply).not.toHaveBeenCalled(); expect(fixture.infer).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: '应用到编辑器' }))
  await waitFor(() => expect(onApply).toHaveBeenCalledOnce())
  expect(fixture.infer.mock.calls[0][2]).toBe('realtime')
  expect(screen.getByText('已应用。可以撤销；试算和保存请在编辑器完成。')).toBeVisible()
  expect(screen.getByRole('button', { name: '应用到编辑器' })).toBeDisabled()
})

it('校验过程中编辑器发生变化，迟到提案不能覆盖新编辑', async () => {
  let resolve!: (value: unknown) => void
  fixture.infer.mockImplementation(() => new Promise(r => { resolve = r }))
  const onApply = vi.fn(), base = current()
  const view = render(<RegimeAgentPanel definition={base} mode="retrospective" onApply={onApply} />)
  fireEvent.click(screen.getByRole('button', { name: '应用到编辑器' }))
  view.rerender(<RegimeAgentPanel definition={{ ...base, name: '人工新编辑' }} mode="retrospective" onApply={onApply} />)
  resolve({ valid: true, errors: [] })
  await screen.findByText('页面或草稿已变化，请让 AI 按当前内容重新校验。')
  expect(onApply).not.toHaveBeenCalled()
})

it('过期提案禁用；列表提案提供打开编辑器入口', () => {
  fixture.stale = true
  render(<RegimeAgentPanel mode="retrospective" onApply={vi.fn()} />)
  expect(screen.getByRole('button', { name: '打开编辑器检查' })).toBeDisabled()
  expect(fixture.props.pageContext.calculation).toMatchObject({ context_kind: 'regime_graph', mode: 'retrospective', as_of: '2019-12-31' })
})

it('当前模式校验失败会解释原因，不覆盖编辑器', async () => {
  fixture.infer.mockResolvedValue({ valid: false, errors: [{ message: '缺少状态输出' }] })
  const onApply = vi.fn()
  render(<RegimeAgentPanel definition={current()} mode="realtime" onApply={onApply} />)
  fireEvent.click(screen.getByRole('button', { name: '应用到编辑器' }))
  await screen.findByText('缺少状态输出')
  expect(onApply).not.toHaveBeenCalled()
})
