import { useState } from 'react'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, expect, it, vi } from 'vitest'
import * as api from '../../services/researchSeries'
import type { RegimeGraphNode, RegimeNodeSchema } from '../../services/regimeGraph'
import RegimeNodeInspector from './RegimeNodeInspector'

vi.mock('../../services/researchSeries', () => ({ listResearchSeries: vi.fn(), listUploadedResearchSeries: vi.fn(), parseResearchFile: vi.fn(), getResearchSeriesProfile: vi.fn(), searchResearchProducts: vi.fn() }))
const catalog = vi.mocked(api.listResearchSeries)
const page = (items: api.ResearchSeriesCatalogItem[]) => ({ items, total: items.length, offset: 0, limit: 100 })
const props = { frequency: { type: 'string', default: 'daily', enum: ['daily', 'monthly'], enum_labels: ['日频', '月频'] }, availability_mode: { enum: ['point_in_time', 'latest'] } }
function Harness({ type, parameters = {}, properties = {} }: { type: string; parameters?: Record<string, unknown>; properties?: Record<string, object> }) {
  const [node, setNode] = useState<RegimeGraphNode>({ id: 'source', type, parameters, inputs: {}, label: '我的数据' })
  const schema = { id: type, label: type, category: 'source', inputs: [], outputs: [{ id: 'value' }], parameter_schema: { properties: { ...props, ...properties } } } as RegimeNodeSchema
  return <><RegimeNodeInspector node={node} schema={schema} nodes={[node]} schemas={[schema]} outputs={{}} inference={null} preparedPlan={null} onPatchNode={patch => setNode(previous => ({ ...previous, ...patch }))} onConnect={vi.fn()} onSetOutput={vi.fn()} onRemove={vi.fn()} /><output data-testid="node">{JSON.stringify(node)}</output></>
}
const nodeValue = () => JSON.parse(screen.getByTestId('node').textContent || '{}') as RegimeGraphNode
beforeEach(() => {
  vi.resetAllMocks()
  catalog.mockResolvedValue(page([]))
  vi.mocked(api.listUploadedResearchSeries).mockResolvedValue(page([]))
  vi.mocked(api.searchResearchProducts).mockResolvedValue({ items: [{ ts_code: '000001.OF', name: '示例基金' }], total: 1 })
})

it('已有内联数据仅保留读取，不再提供手工编辑入口', () => {
  const parameters = { inline_rows: [{ date: '2024-01-01', value: 12 }] }
  render(<Harness type="source.inline" parameters={parameters} />)
  expect(screen.getByText(/若要更换数据/)).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '添加一行' })).not.toBeInTheDocument()
  expect(nodeValue().parameters).toEqual(parameters)
})

it.each(['选择宏观数据', '上传时序'])('美林时钟的空数据占位可通过%s替换，保留节点编号、名称和月频', async action => {
  render(<Harness type="source.inline" parameters={{ rows: [], name: '增长指标', frequency: 'monthly' }} />)
  expect(screen.getByText('尚未选择输入数据')).toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: action }))
  expect(nodeValue()).toEqual({ id: 'source', type: action === '选择宏观数据' ? 'source.macro' : 'source.upload', type_version: 1, label: '我的数据', parameters: { frequency: 'monthly' }, inputs: {} })
  expect(screen.queryByText('尚未选择输入数据')).not.toBeInTheDocument()
  if (action === '选择宏观数据') expect(await screen.findByRole('searchbox')).toHaveAttribute('placeholder', expect.stringContaining('宏观'))
  else expect(screen.getByLabelText('选择时序文件')).toBeInTheDocument()
})

it('宏观只选一次并清除上一个数据集代码、日期字段和快照', async () => {
  const item: api.ResearchSeriesCatalogItem = { id: 'macro:new', name: 'CPI', kind: 'macro', status: 'available', regime_node_type: 'source.macro', fields: [{ name: 'cpi', label: '居民消费价格' }], binding_parameters: { dataset: 'new.parquet', field: 'cpi', name: 'CPI', frequency: 'monthly' } }
  catalog.mockResolvedValue(page([item]))
  render(<Harness type="source.macro" parameters={{ dataset: 'old.parquet', code: 'old', series_id: 'old', ts_code: 'old', date_field: 'old_date', file_checksum: 'old', start_date: '2020-01-01' }} properties={{ dataset: { type: 'string' }, code: { type: 'string' }, date_field: { type: 'string' }, field: { type: 'string' } }} />)
  await screen.findByRole('option', { name: 'CPI' })
  await userEvent.selectOptions(screen.getByLabelText('节点研究数据序列'), item.id)
  expect(nodeValue().parameters).toEqual({ ...item.binding_parameters, start_date: '2020-01-01' })
  expect(screen.queryByRole('textbox', { name: '宏观数据集' })).not.toBeInTheDocument()
  expect(screen.getByRole('searchbox')).toHaveAttribute('placeholder', expect.stringContaining('宏观'))
  expect(await screen.findByRole('option', { name: '居民消费价格' })).toBeInTheDocument()
})

it('引用指标按名称与版本选择、单独搜索计算对象，不要求填写内部编号', async () => {
  const item: api.ResearchSeriesCatalogItem = { id: 'indicator:r@2', name: '年度收益率', kind: 'indicator', status: 'available', regime_node_type: 'source.indicator', product_kinds: ['fund'], periods: ['1Y', '3M'], indicator_version: { indicator_id: 'r', revision: 2 }, binding_parameters: { indicator_id: 'r', indicator_revision: 2, name: '年度收益率', period: '1Y', product_kind: '', product_id: '' } }
  catalog.mockResolvedValue(page([item, { ...item, id: 'indicator:r@1', name: '旧指标', binding_supported: false, binding_reason: '组合指标不适用' }]))
  render(<Harness type="source.indicator" properties={{ indicator_id: { type: 'string' }, indicator_revision: { type: 'integer' }, product_kind: { type: 'string' }, product_id: { type: 'string' }, period: { type: 'string' } }} />)
  expect(await screen.findByRole('option', { name: /旧指标.*组合指标不适用/ })).toBeDisabled()
  await userEvent.selectOptions(screen.getByLabelText('节点研究数据序列'), item.id)
  expect(screen.queryByRole('textbox', { name: /指标编号/ })).not.toBeInTheDocument()
  expect(screen.getByLabelText('指标计算窗口')).toHaveValue('1Y')
  expect(await screen.findByRole('option', { name: '场外基金' })).toBeInTheDocument()
  expect(screen.queryByRole('option', { name: /^ETF$/ })).not.toBeInTheDocument()
  await userEvent.selectOptions(screen.getByLabelText('指标对象类型'), 'fund')
  fireEvent.change(screen.getByLabelText('搜索计算对象'), { target: { value: '示例' } })
  await screen.findByRole('option', { name: '示例基金 · 000001.OF' })
  await userEvent.selectOptions(screen.getByLabelText('指标计算对象', { selector: 'select' }), '000001.OF')
  expect(nodeValue().parameters).toMatchObject({ indicator_id: 'r', indicator_revision: 2, product_kind: 'fund', product_id: '000001.OF', product_name: '示例基金', period: '1Y' })
})

it('上传先确认列，修改频率不丢失文件，保存成功才替换绑定', async () => {
  vi.mocked(api.parseResearchFile).mockResolvedValue({ columns: ['日期', '收盘价'], rows: [{ 日期: '2024-01-01', 收盘价: 10 }, { 日期: '2024-01-02', 收盘价: null }], sheets: ['行情'], sheet: '行情' })
  const binding = { artifact_id: 'new', checksum: 'sum', name: '行情', frequency: 'monthly' }
  vi.mocked(api.getResearchSeriesProfile).mockResolvedValue({ binding_parameters: binding, series: { id: 'uploaded', kind: 'upload', name: '行情' }, dates: [], values: {}, execution: {} })
  render(<Harness type="source.upload" properties={{ artifact_id: { type: 'string' }, checksum: { type: 'string' } }} />)
  await userEvent.upload(screen.getByLabelText('选择时序文件'), new File(['excel'], '行情.xlsx'))
  await screen.findByRole('table', { name: '文件列预览' })
  expect(nodeValue().parameters.artifact_id).toBeUndefined()
  await userEvent.selectOptions(screen.getByLabelText('数据频率'), 'monthly')
  expect(screen.getByRole('table', { name: '文件列预览' })).toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: '保存并使用' }))
  await waitFor(() => expect(nodeValue().parameters).toEqual(binding))
  expect(api.getResearchSeriesProfile).toHaveBeenCalledWith(expect.objectContaining({ inline_rows: [{ date: '2024-01-01', value: 10 }, { date: '2024-01-02', value: null }], frequency: 'monthly', register_artifact: true }), expect.any(AbortSignal))
  expect(nodeValue().label).toBe('我的数据')
  expect(screen.queryByRole('textbox', { name: /校验值/ })).not.toBeInTheDocument()
})

it('上传失败和取消晚到结果均保留原来的数据绑定', async () => {
  let resolve!: (result: api.ResearchImportFile) => void
  vi.mocked(api.parseResearchFile).mockImplementation(() => new Promise(done => { resolve = done }))
  render(<Harness type="source.upload" parameters={{ artifact_id: 'original', checksum: 'sum', name: '原数据' }} />)
  await userEvent.click(screen.getByText('上传新文件'))
  await userEvent.upload(screen.getByLabelText('选择时序文件'), new File(['csv'], 'x.csv'))
  await userEvent.click(screen.getByRole('button', { name: '取消本次上传' }))
  await act(async () => resolve({ columns: ['date', 'value'], rows: [{ date: '2024-01-01', value: 10 }], sheets: [], sheet: null }))
  expect(screen.queryByRole('table')).not.toBeInTheDocument()
  expect(nodeValue().parameters.artifact_id).toBe('original')
  expect(api.getResearchSeriesProfile).not.toHaveBeenCalled()
})
