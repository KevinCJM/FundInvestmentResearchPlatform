import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { RiskScaleResults } from './RiskScaleResults'
import { riskPreview } from '../../test/riskScaleFixtures'
import { copyRiskEditor, restoreRiskEditor } from './editor'

vi.mock('echarts-for-react', () => ({ default: ({ option }: any) => <output data-testid="chart-options">{JSON.stringify(option)}</output> }))
describe('risk chart evidence', () => {
  it('preserves failure gaps without interpolation and leaves empty representatives unavailable', async () => {
    const preview = structuredClone(riskPreview)
    preview.result.frontier[2] = { node_id: 2, volatility: null, expected_return: null, weights: null, status: 'iteration_limit' }
    preview.result.levels[2] = { ...preview.result.levels[2], representative_node_id: null, representative_weights: null, volatility: { value: null, status: 'unavailable' }, expected_return: { value: null, status: 'unavailable' }, calibration_status: 'not_calibrated' }
    render(<RiskScaleResults preview={preview} />)
    const options = JSON.parse((await screen.findByTestId('chart-options')).textContent!)
    expect(options.series[0].data[2]).toBeNull(); expect(options.series[0].connectNulls).toBe(false); expect(options.series[0].smooth).toBe(false)
    const row = screen.getByRole('button', { name: 'C3' }).closest('tr')!
    expect(within(row).getAllByText('不可用')).toHaveLength(2)
    fireEvent.click(within(row).getByRole('button')); expect(screen.getByText('该档没有有效代表组合，收益、权重及画像不可用。')).toBeInTheDocument()
  })
  it('keeps backend diagnostics and internal reference ids out of the user-facing result', async () => {
    render(<RiskScaleResults preview={riskPreview} />)
    expect(await screen.findByRole('table', { name: 'C1–C5 风险等级' })).toBeInTheDocument()
    expect(screen.queryByText('历史参数与来源证据')).not.toBeInTheDocument()
    expect(screen.queryByText('求解、网格稳定性与逐点诊断')).not.toBeInTheDocument()
    expect(screen.queryByText(riskPreview.request_echo.definition.reference_input_ref.id)).not.toBeInTheDocument()
    expect(screen.queryByText(/101 \/ 200/)).not.toBeInTheDocument()
  })
  it('restores a saved review step to segmentation without a persisted confirmation', () => {
    const editor = { ...copyRiskEditor(riskPreview.request_echo.definition), step: 4 }
    const restored = restoreRiskEditor(JSON.parse(JSON.stringify(editor)))
    expect(restored.step).toBe(3); expect(restored.definition.reference_input_ref).toEqual(editor.definition.reference_input_ref)
  })
  it('rejects unrelated draft schemas safely', () => { expect(() => restoreRiskEditor({ something: 'else' })).toThrow('') })
})
