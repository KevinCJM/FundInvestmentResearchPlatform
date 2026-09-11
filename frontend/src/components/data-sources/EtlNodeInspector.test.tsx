import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { SourceCatalog } from '../../services/dataSources'
import { blankStep, type EtlDefinition } from '../../services/etl'
import { etlGraphSchemas } from '../../test/etlGraphFixtures'
import EtlNodeInspector from './EtlNodeInspector'

const catalog = { sources: [], interfaces: [], etl_tasks: [], editing_enabled: true } as unknown as SourceCatalog

describe('ETL dependency selector', () => {
  it('switches data dependency and order-only atomically without changing upstream', () => {
    const original: EtlDefinition = { name: '测试', description: '', max_runtime_seconds: 60, graph_version: 1, steps: [
      { ...blankStep('task'), id: 'first', name: '交易日历' },
      { ...blankStep('task'), id: 'second', name: '基金公司', inputs: ['first'] },
    ] }
    const changed = vi.fn()
    function Harness() {
      const [definition, setDefinition] = useState(original)
      return <EtlNodeInspector definition={definition} step={definition.steps[1]} catalog={catalog} schemas={etlGraphSchemas} readOnly={false}
        onPatch={step => { changed(step); setDefinition({ ...definition, steps: [definition.steps[0], step] }) }}
        onConnect={vi.fn()} onDisconnect={vi.fn()} onRemove={vi.fn()} onClose={vi.fn()} />
    }
    render(<Harness />)
    fireEvent.change(screen.getByLabelText('与交易日历的依赖关系'), { target: { value: 'control' } })
    expect(changed).toHaveBeenLastCalledWith(expect.objectContaining({ inputs: [], after: ['first'] }))
    expect(screen.getByLabelText('与交易日历的依赖关系')).toHaveValue('control')
    fireEvent.change(screen.getByLabelText('与交易日历的依赖关系'), { target: { value: 'data' } })
    expect(changed).toHaveBeenLastCalledWith(expect.objectContaining({ inputs: ['first'], after: [] }))
    expect(original.steps[1].inputs).toEqual(['first'])
    expect(screen.getByText(/不能用此选项绕过数据完整性检查/)).toBeVisible()
  })

  it('keeps a historical read-only dependency selector disabled', () => {
    const first = { ...blankStep('task'), id: 'first', name: '交易日历' }
    const second = { ...blankStep('task'), id: 'second', after: ['first'] }
    render(<EtlNodeInspector definition={{ name: '历史', description: '', max_runtime_seconds: 60, steps: [first, second] }}
      step={second} catalog={catalog} schemas={etlGraphSchemas} readOnly onPatch={vi.fn()} onConnect={vi.fn()} onDisconnect={vi.fn()} onRemove={vi.fn()} onClose={vi.fn()} />)
    expect(screen.getByLabelText('与交易日历的依赖关系')).toBeDisabled()
  })
})
