import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { SourceCatalog } from '../../services/dataSources'
import { blankStep, planEtlDependencies, type EtlDefinition } from '../../services/etl'
import { etlGraphSchemas } from '../../test/etlGraphFixtures'
import EtlWorkflowEditor from './EtlWorkflowEditor'

vi.mock('../computation-graph/GraphCanvas', () => ({ default: () => <div>测试画布</div> }))
vi.mock('../../services/etl', async importOriginal => ({ ...await importOriginal<object>(), planEtlDependencies: vi.fn() }))
const catalog = { sources: [], interfaces: [], etl_tasks: [], graph_schemas: etlGraphSchemas, editing_enabled: true } as unknown as SourceCatalog
const definition: EtlDefinition = { name: '测试', description: '', max_runtime_seconds: 60, graph_version: 1, steps: [
  { ...blankStep('task'), id: 'first', name: '日历' },
  { ...blankStep('task'), id: 'second', name: '公司', inputs: ['first'] },
] }
const result = { ...definition, steps: definition.steps.map(s => s.id === 'second' ? { ...s, inputs: [], after: ['first'] } : s) }

describe('server-owned ETL dependency planning', () => {
  beforeEach(() => { vi.resetAllMocks() })
  it('applies the verified draft and does not start a download', async () => {
    vi.mocked(planEtlDependencies).mockResolvedValue({ definition: result })
    const onChange = vi.fn()
    render(<EtlWorkflowEditor definition={definition} catalog={catalog} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: '按数据需求重新梳理依赖' }))
    await waitFor(() => expect(onChange).toHaveBeenCalledWith(result))
    expect(planEtlDependencies).toHaveBeenCalledWith(definition)
    expect(definition.steps[1].inputs).toEqual(['first'])
  })
  it('does not overwrite edits with a stale planning response', async () => {
    let complete!: (value: { definition: EtlDefinition }) => void
    vi.mocked(planEtlDependencies).mockReturnValue(new Promise(resolve => { complete = resolve }))
    const onChange = vi.fn()
    const view = render(<EtlWorkflowEditor definition={definition} catalog={catalog} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: '按数据需求重新梳理依赖' }))
    view.rerender(<EtlWorkflowEditor definition={{ ...definition, name: '已编辑' }} catalog={catalog} onChange={onChange} />)
    complete({ definition: result })
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('未覆盖你的编辑'))
    expect(onChange).not.toHaveBeenCalled()
  })
  it('shows validation errors and disables planning for read-only runs', async () => {
    vi.mocked(planEtlDependencies).mockRejectedValue(new Error('缺少日历输入'))
    const view = render(<EtlWorkflowEditor definition={definition} catalog={catalog} onChange={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: '按数据需求重新梳理依赖' }))
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('缺少日历输入'))
    view.rerender(<EtlWorkflowEditor definition={definition} catalog={catalog} onChange={vi.fn()} readOnly />)
    expect(screen.getByRole('button', { name: '按数据需求重新梳理依赖' })).toBeDisabled()
  })
})
