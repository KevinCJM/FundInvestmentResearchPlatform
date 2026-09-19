import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import RiskScaleWorkspace from './RiskScaleWorkspace'
import { riskScales, type ReferencePreview } from '../services/riskScales'
import { riskCapabilities, riskPreview } from '../test/riskScaleFixtures'
import { chooseLocale } from '../i18n/runtime'

vi.mock('../app/ResearchContext', () => ({ useResearchDay: () => null }))
vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="test-chart" /> }))
beforeEach(async () => { await chooseLocale('zh-CN') })
afterEach(() => { vi.restoreAllMocks() })

it('builds a risk scale directly from cash and adjusted product history without CMA', async () => {
  vi.spyOn(riskScales, 'capabilities').mockResolvedValue(riskCapabilities)
  vi.spyOn(riskScales, 'sources').mockResolvedValue({ items: [{ id: 'etf:fund_daily:900001.SH', code: '900001.SH', name: 'Synthetic ETF', kind: 'etf', coverage: { start_date: '2020-01-02', end_date: '2026-09-17' }, reference_capability: { available: true, supported_fields: ['adj_nav'] } }], total: 1, offset: 0, limit: 20, problems: [] })
  const check = vi.spyOn(riskScales, 'previewReference').mockImplementation(async request => ({ preview_hash: 'f'.repeat(64), definition: request as unknown as Record<string, unknown>, ordered_asset_ids: request.assets.map(a => a.id), quality: { intersection_start: '2020-01-02', intersection_end: '2026-09-17', observations: 1600 }, warnings: [], moments: { annual_returns: [.0, .08], annual_volatilities: [0, .18] }, provenance: {} } as ReferencePreview))
  const confirm = vi.spyOn(riskScales, 'confirmReference').mockImplementation(async body => ({ preview_hash: body.preview_hash, definition: body.request as unknown as Record<string, unknown>, ordered_asset_ids: body.request.assets.map(a => a.id), quality: { intersection_start: '2020-01-02', intersection_end: '2026-09-17', observations: 1600 }, warnings: [], moments: { annual_returns: [0, .08], annual_volatilities: [0, .18] }, provenance: {}, id: 'reference-fixture', content_hash: 'd'.repeat(64), created_at: '2026-09-17', artifact_type: 'reference_inputs', immutable: true }))
  const preview = vi.spyOn(riskScales, 'preview').mockResolvedValue(riskPreview)

  render(<MemoryRouter><RiskScaleWorkspace /></MemoryRouter>)
  fireEvent.change(await screen.findByLabelText('标尺名称'), { target: { value: 'Direct history test' } })
  expect(screen.getByLabelText(/参考研究日/)).toHaveAttribute('readonly')
  expect(screen.queryByLabelText('预测期限（年）')).not.toBeInTheDocument()
  expect(screen.queryByLabelText('风险口径')).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '下一步' }))

  fireEvent.click(screen.getByRole('button', { name: '添加参考大类' }))
  fireEvent.change(screen.getByLabelText('大类名称'), { target: { value: 'Cash' } })
  fireEvent.change(screen.getByLabelText('资产类型'), { target: { value: 'cash' } })
  fireEvent.click(screen.getByRole('button', { name: '添加参考大类' }))
  const names = screen.getAllByLabelText('大类名称')
  fireEvent.change(names[1], { target: { value: 'Equity' } })
  const equity = screen.getAllByTestId('risk-reference-asset')[1]
  fireEvent.click(withinRegion(equity, '添加或更换代理'))
  fireEvent.change(screen.getByLabelText('来源类型'), { target: { value: 'etf' } })
  await screen.findByText('Synthetic ETF')
  expect(screen.queryByLabelText('冻结产品池')).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: '选择' })).toBeEnabled()
  fireEvent.click(screen.getByRole('button', { name: '选择' }))

  fireEvent.click(screen.getByRole('button', { name: '检查参考数据' }))
  await screen.findByText(/共同历史区间/)
  expect(check).toHaveBeenCalled()
  const request = check.mock.calls[0][0]
  expect(request).not.toHaveProperty('start_date')
  expect(request).not.toHaveProperty('end_date')
  expect(request.return_basis).toBe('selected_index_and_adjusted_product_total_return')
  expect(request.assets[1].components[0]).toEqual({ kind: 'etf', series_id: 'etf:fund_daily:900001.SH', field: 'adj_nav', weight: 1 })
  expect(request.assets[1].rebalance).toBe('daily')
  fireEvent.click(screen.getByLabelText(/我已核对来源与警告/))
  fireEvent.click(screen.getByRole('button', { name: '确认保存参考资产与代理' }))
  await screen.findByText(/历史参数与约束/)
  fireEvent.click(screen.getByRole('button', { name: '计算前沿与五档' }))
  await screen.findByRole('table', { name: 'C1–C5 风险等级' })
  expect(confirm).toHaveBeenCalled()
  expect(preview.mock.calls[0][0].definition.reference_input_ref.id).toBe('reference-fixture')
  expect(screen.queryByText(/CMA/)).not.toBeInTheDocument()
})

function withinRegion(region: HTMLElement, name: string) {
  const button = Array.from(region.querySelectorAll('button')).find(item => item.textContent === name)
  if (!button) throw new Error(`missing button ${name}`)
  return button
}
