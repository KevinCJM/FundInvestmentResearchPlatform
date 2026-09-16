import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import RegimeReferenceBinding from './RegimeReferenceBinding'
import { createBlankRegimeDefinition, listHistoricalReferences, type HistoricalReference } from '../../services/regimeGraph'
vi.mock('../../services/regimeGraph', async original => ({ ...await original<typeof import('../../services/regimeGraph')>(), listHistoricalReferences: vi.fn(), getRegimeGraphDefinition: vi.fn().mockResolvedValue({ graph: { nodes: [] } }) }))
const reference: HistoricalReference = { run_id: 'run', publication_id: 'pub', content_hash: 'hash', definition_id: 'd', definition_revision: 2, name: '自定义参考', states: [{ id: 'x', label: '扩张', color: '#000000' }], frequency: 'monthly', as_of: '2025-12-31', created_at: '2026-01-01', series_summary: null }
describe('历史参考选择', () => {
  it('空目录明确下一步，允许未绑定探索', async () => {
    vi.mocked(listHistoricalReferences).mockResolvedValue([])
    const open = vi.fn()
    render(<RegimeReferenceBinding definition={createBlankRegimeDefinition()} onChange={vi.fn()} onHistorical={open} />)
    expect(await screen.findByText(/还没有已确认的历史参考/)).toBeVisible()
    await act(async () => userEvent.click(screen.getByRole('button', { name: '前往历史状态定义' })))
    expect(open).toHaveBeenCalledOnce()
  })
  it('错误可重试；选择精确版本不改变图', async () => {
    vi.mocked(listHistoricalReferences).mockRejectedValueOnce(new Error('目录失败')).mockResolvedValue([reference])
    const definition = createBlankRegimeDefinition()
    const change = vi.fn()
    render(<RegimeReferenceBinding definition={definition} onChange={change} />)
    expect(await screen.findByRole('alert')).toHaveTextContent('目录失败')
    await act(async () => userEvent.click(screen.getByRole('button', { name: '重试读取参考' })))
    const option = await screen.findByRole('option', { name: /自定义参考/ })
    await act(async () => userEvent.selectOptions(screen.getByLabelText('历史参考版本'), option))
    expect(change.mock.calls[0][0]).toMatchObject({ study: { reference: { run_id: 'run', publication_id: 'pub', content_hash: 'hash' } } })
    expect(change.mock.calls[0][0].graph).toBe(definition.graph)
  })
})
