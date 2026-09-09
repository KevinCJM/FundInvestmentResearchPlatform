import { useState } from 'react'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { listResearchSeries, type ResearchSeriesCatalogItem } from '../../services/researchSeries'
import type { RegimeGraphNode, RegimeNodeSchema } from '../../services/regimeGraph'
import RegimeNodeInspector from './RegimeNodeInspector'
import { researchSourceParameterPatch } from './RegimeResearchSeriesPicker'

vi.mock('../../services/researchSeries', () => ({ listResearchSeries: vi.fn() }))

const schema: RegimeNodeSchema = {
  id: 'source.index', label: '指数行情', category: 'source', inputs: [], outputs: [{ id: 'value' }],
  parameter_schema: { required: ['ts_code'], properties: {
    ts_code: { type: 'string' }, source_api: { type: 'string', default: 'index_daily', enum: ['index_daily', 'sw_daily'], enum_labels: ['指数日行情', '申万指数日行情'] },
    name: { type: 'string' }, field: { type: 'string', default: 'close', enum: ['close', 'open', 'amount'] },
    frequency: { type: 'string', default: 'daily' }, start_date: { type: 'string' }, end_date: { type: 'string' },
    snapshot_id: { type: 'string' }, snapshot_generation: { type: 'string' }, source_file: { type: 'string' }, file_checksum: { type: 'string' },
  } },
}
const csi300: ResearchSeriesCatalogItem = {
  id: 'index:index_daily:000300.SH', name: '沪深300指数', kind: 'index', code: '000300.SH', status: 'available', regime_node_type: 'source.index',
  fields: [{ name: 'close', label: '收盘点位' }, { name: 'open', label: '开盘点位' }],
  binding_parameters: { ts_code: '000300.SH', source_api: 'index_daily', name: '沪深300指数', field: 'close', frequency: 'daily', snapshot_id: 'new-snapshot', snapshot_generation: 'new-generation', source_file: 'index_daily.parquet', file_checksum: 'new-checksum' },
}
const other: ResearchSeriesCatalogItem = { ...csi300, id: 'index:sw_daily:801010.SI', name: '农林牧渔', code: '801010.SI', binding_parameters: { ...csi300.binding_parameters, ts_code: '801010.SI', name: '农林牧渔', source_api: 'sw_daily' } }
const blank: RegimeGraphNode = { id: 'market', type: 'source.index', parameters: {}, inputs: {} }
const catalog = vi.mocked(listResearchSeries)
const page = (items: ResearchSeriesCatalogItem[], total = items.length, offset = 0) => ({ items, total, offset, limit: 100 })

function Harness({ initial = blank, sourceSchema = schema }: { initial?: RegimeGraphNode; sourceSchema?: RegimeNodeSchema }) {
  const [node, setNode] = useState(initial)
  return <><RegimeNodeInspector node={node} schema={sourceSchema} nodes={[node]} schemas={[sourceSchema]} outputs={{}} inference={null} preparedPlan={null} onPatchNode={patch => setNode(previous => ({ ...previous, ...patch }))} onConnect={vi.fn()} onSetOutput={vi.fn()} onRemove={vi.fn()} /><output data-testid="bound-node">{JSON.stringify(node)}</output></>
}
const nodeValue = () => JSON.parse(screen.getByTestId('bound-node').textContent || '{}') as RegimeGraphNode
const search = (value: string) => fireEvent.change(screen.getByRole('searchbox', { name: '搜索研究数据' }), { target: { value } })

describe('情景研究数据选择', () => {
  beforeEach(() => { catalog.mockReset() })
  afterEach(() => { vi.restoreAllMocks() })

  it('搜索完整目录中的名称或代码，并一次绑定指数、来源和快照，保留日期及自定义节点名', async () => {
    catalog.mockImplementation(async options => options?.query ? page([csi300]) : page([other], 14503))
    render(<Harness initial={{ ...blank, label: '我的市场基准', parameters: { start_date: '2010-01-01', field: 'open' } }} />)
    await screen.findByText('找到 14503 条，已加载 1 条。')
    expect(screen.queryByRole('option', { name: /沪深300/ })).not.toBeInTheDocument()
    search('沪深300')
    const option = await screen.findByRole('option', { name: '沪深300指数 · 000300.SH' })
    expect(option).toBeEnabled()
    expect(catalog).toHaveBeenLastCalledWith(expect.objectContaining({ query: '沪深300', kind: 'index', offset: 0 }), expect.any(AbortSignal))
    search('000300.sh')
    await waitFor(() => expect(catalog).toHaveBeenLastCalledWith(expect.objectContaining({ query: '000300.sh' }), expect.any(AbortSignal)))
    await screen.findByRole('option', { name: '沪深300指数 · 000300.SH' })
    await userEvent.selectOptions(screen.getByLabelText('节点研究数据序列'), csi300.id)
    expect(nodeValue()).toMatchObject({ label: '我的市场基准', parameters: { ...csi300.binding_parameters, field: 'open', start_date: '2010-01-01' } })
    expect(screen.queryByRole('textbox', { name: /指数代码/ })).not.toBeInTheDocument()
    expect(screen.queryByRole('combobox', { name: '行情数据接口' })).not.toBeInTheDocument()
    expect(screen.queryByRole('textbox', { name: '数据快照' })).not.toBeInTheDocument()
    await userEvent.click(screen.getByText('查看数据来源'))
    expect(screen.getByText('new-snapshot')).toBeVisible()
    const field = screen.getByLabelText('数值字段')
    await waitFor(() => expect(within(field).getByRole('option', { name: '开盘点位' })).toBeInTheDocument())
    await userEvent.selectOptions(field, 'close')
    expect(nodeValue().parameters).toMatchObject({ ...csi300.binding_parameters, field: 'close' })
  })

  it('自动查找模板已绑定的指数，目录加载和搜索不重写已有绑定', async () => {
    catalog.mockResolvedValue(page([csi300]))
    const initial = { ...blank, parameters: { ts_code: '000300.SH', field: 'close', name: '沪深300' } }
    render(<Harness initial={initial} />)
    await waitFor(() => expect(screen.getByLabelText('节点研究数据序列')).toHaveValue(csi300.id))
    expect(catalog).toHaveBeenCalledWith(expect.objectContaining({ query: '000300.SH', kind: 'index' }), expect.any(AbortSignal))
    expect(nodeValue()).toEqual(initial)
    catalog.mockResolvedValue(page([other]))
    search('农林牧渔')
    await screen.findByRole('option', { name: /农林牧渔/ })
    expect(screen.getByLabelText('节点研究数据序列')).toHaveValue(csi300.id)
    expect(nodeValue()).toEqual(initial)
  })

  it('分页加载搜索结果，未下载条目可见但不可绑定', async () => {
    const missing = { ...csi300, id: 'missing', name: '尚未下载指数', status: 'not_downloaded', regime_node_type: undefined }
    catalog.mockImplementation(async options => options?.offset ? page([csi300], 3, 2) : page([other, missing], 3))
    render(<Harness />)
    expect(await screen.findByRole('option', { name: /尚未下载指数/ })).toBeDisabled()
    await userEvent.click(screen.getByRole('button', { name: '加载更多结果' }))
    expect(await screen.findByRole('option', { name: /沪深300指数/ })).toBeEnabled()
    expect(catalog).toHaveBeenLastCalledWith(expect.objectContaining({ offset: 2, query: '' }), expect.any(AbortSignal))
    expect(screen.queryByRole('button', { name: '加载更多结果' })).not.toBeInTheDocument()
    expect(nodeValue().parameters).toEqual({})
  })

  it('旧搜索晚返回也不能覆盖新搜索，且搜索输入会合并请求', async () => {
    let resolveOld!: (value: ReturnType<typeof page>) => void
    catalog.mockImplementation(async options => options?.query === 'old' ? new Promise(resolve => { resolveOld = resolve }) : page([csi300]))
    render(<Harness />)
    search('o'); search('ol'); search('old')
    await waitFor(() => expect(resolveOld).toBeTypeOf('function'))
    expect(catalog).toHaveBeenCalledTimes(1)
    const oldSignal = catalog.mock.calls[0][1]
    search('沪深300')
    await screen.findByRole('option', { name: /沪深300指数/ })
    expect(oldSignal?.aborted).toBe(true)
    await act(async () => resolveOld(page([other])))
    expect(screen.queryByRole('option', { name: /农林牧渔/ })).not.toBeInTheDocument()
  })

  it('区分加载失败和空结果，支持重试', async () => {
    catalog.mockRejectedValueOnce(new Error('offline')).mockResolvedValue(page([]))
    render(<Harness />)
    expect(await screen.findByRole('alert')).toHaveTextContent('研究数据加载失败')
    await userEvent.click(screen.getByRole('button', { name: '重试' }))
    expect(await screen.findByText('没有匹配的研究数据，请换一个名称或代码。')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('更换指数清理旧文件绑定，不支持的旧字段恢复为新序列默认字段', async () => {
    const unpinned = { ...csi300, binding_parameters: { ts_code: '000300.SH', source_api: 'index_daily', field: 'close', name: '沪深300指数' } }
    catalog.mockResolvedValue(page([unpinned]))
    render(<Harness initial={{ ...blank, parameters: { ...other.binding_parameters, file_checksum: 'old-checksum', field: 'amount', end_date: '2026-01-01' } }} />)
    await screen.findByRole('option', { name: /沪深300指数/ })
    await userEvent.selectOptions(screen.getByLabelText('节点研究数据序列'), csi300.id)
    expect(nodeValue().parameters).toEqual({ ...unpinned.binding_parameters, frequency: 'daily', end_date: '2026-01-01' })
  })
})


it.each([
  ['etf', 'ETF行情', '510300.SH', '沪深300ETF', 'fund_daily', 'close', '收盘价（不复权）'],
  ['fund', '公募基金行情', '000001.OF', '华夏成长', 'fund_nav', 'unit_nav', '单位净值'],
])('新增 %s 行情只选一次产品，字段与来源自动绑定', async (kind, label, code, name, api, field, fieldLabel) => {
  const item: ResearchSeriesCatalogItem = { ...csi300, id: `${kind}:${api}:${code}`, kind, code, name, regime_node_type: `source.${kind}`,
    fields: [{ name: field, label: fieldLabel }], binding_parameters: { ...csi300.binding_parameters, ts_code: code, name, source_api: api, field } }
  catalog.mockResolvedValue(page([item]))
  render(<Harness initial={{ id: 'market', type: `source.${kind}`, label, parameters: {}, inputs: {} }} sourceSchema={{ ...schema, id: `source.${kind}`, label }} />)
  search(name)
  await screen.findByRole('option', { name: `${name} · ${code}` })
  expect(catalog).toHaveBeenLastCalledWith(expect.objectContaining({ kind, query: name }), expect.any(AbortSignal))
  await userEvent.selectOptions(screen.getByLabelText('节点研究数据序列'), item.id)
  expect(nodeValue()).toMatchObject({ type: `source.${kind}`, label: name, parameters: item.binding_parameters })
  expect(screen.queryByRole('textbox', { name: /指数代码|产品代码/ })).not.toBeInTheDocument()
  expect(screen.queryByRole('combobox', { name: '行情数据接口' })).not.toBeInTheDocument()
  await waitFor(() => expect(within(screen.getByLabelText('数值字段')).getByRole('option', { name: fieldLabel })).toBeInTheDocument())
  expect(screen.getByLabelText('数值字段')).toHaveValue(field)
})

it('ETF复权自动绑定因子版本，换到缺少因子的ETF后清除旧版本并禁用复权字段', async () => {
  const adjusted: ResearchSeriesCatalogItem = { ...csi300, id: 'etf:fund_daily:510300.SH', kind: 'etf', code: '510300.SH', name: '沪深300ETF', regime_node_type: 'source.etf',
    fields: [{ name: 'close', label: '收盘价（不复权）' }, { name: 'close_hfq', label: '收盘价（后复权）', available: true }],
    binding_parameters: { ...csi300.binding_parameters, ts_code: '510300.SH', source_api: 'fund_daily', adjustment_checksum: 'factor-version' } }
  const missing = { ...adjusted, id: 'etf:fund_daily:510500.SH', code: '510500.SH', name: '另一只ETF',
    fields: [{ name: 'close', label: '收盘价（不复权）' }, { name: 'close_hfq', label: '收盘价（后复权）', available: false, unavailable_reason: '缺少复权因子' }],
    binding_parameters: { ...csi300.binding_parameters, ts_code: '510500.SH', source_api: 'fund_daily' } }
  catalog.mockResolvedValue(page([adjusted, missing]))
  render(<Harness initial={{ ...blank, type: 'source.etf' }} sourceSchema={{ ...schema, id: 'source.etf', parameter_schema: { ...schema.parameter_schema, properties: { ...schema.parameter_schema?.properties, adjustment_checksum: { type: 'string' } } } }} />)
  await screen.findByRole('option', { name: '沪深300ETF · 510300.SH' })
  await userEvent.selectOptions(screen.getByLabelText('节点研究数据序列'), adjusted.id)
  await screen.findByRole('option', { name: '收盘价（后复权）' })
  await userEvent.selectOptions(screen.getByLabelText('数值字段'), 'close_hfq')
  expect(nodeValue().parameters).toMatchObject({ field: 'close_hfq', adjustment_checksum: 'factor-version' })
  expect(screen.queryByRole('textbox', { name: 'adjustment_checksum' })).not.toBeInTheDocument()
  await userEvent.selectOptions(screen.getByLabelText('节点研究数据序列'), missing.id)
  expect(nodeValue().parameters.field).toBe('close')
  expect(nodeValue().parameters).not.toHaveProperty('adjustment_checksum')
  expect(await screen.findByRole('option', { name: /缺少复权因子/ })).toBeDisabled()
})

it('同一ETF可切换复权净值和市价，自动绑定各自文件并保留日期设置', async () => {
  const priceBinding = { ...csi300.binding_parameters, ts_code: '510300.SH', source_api: 'fund_daily', source_file: 'etf_daily_candle_df.parquet', file_checksum: 'price-checksum', adjustment_checksum: 'factor-checksum', field: 'close' }
  const navBinding = { ...csi300.binding_parameters, ts_code: '510300.SH', source_api: 'fund_nav', source_file: 'etf_daily_df.parquet', file_checksum: 'nav-checksum', field: 'adj_nav' }
  const item: ResearchSeriesCatalogItem = { ...csi300, id: 'etf:fund_daily:510300.SH', kind: 'etf', name: '沪深300ETF', code: '510300.SH', regime_node_type: 'source.etf', binding_parameters: priceBinding,
    fields: [{ name: 'close', label: '收盘价（不复权）', binding_parameters: priceBinding }, { name: 'adj_nav', label: '复权净值', binding_parameters: navBinding }] }
  let resolveCatalog!: (value: ReturnType<typeof page>) => void
  catalog.mockImplementation(() => new Promise(resolve => { resolveCatalog = resolve }))
  render(<Harness initial={{ ...blank, type: 'source.etf', parameters: { ...priceBinding, start_date: '2020-01-01' } }} sourceSchema={{ ...schema, id: 'source.etf', parameter_schema: { properties: { field: { enum: ['close', 'adj_nav'] } } } }} />)
  expect(screen.getByLabelText('数值字段')).toBeDisabled()
  await waitFor(() => expect(resolveCatalog).toBeTypeOf('function'))
  await act(async () => resolveCatalog(page([item])))
  expect(screen.getByLabelText('数值字段')).toBeEnabled()
  await userEvent.selectOptions(screen.getByLabelText('数值字段'), 'adj_nav')
  expect(nodeValue().parameters).toEqual({ ...navBinding, start_date: '2020-01-01' })
  expect(screen.getByLabelText('节点研究数据序列')).toHaveValue(item.id)
  expect(screen.queryByRole('textbox', { name: /产品代码|指数代码/ })).not.toBeInTheDocument()
  await userEvent.selectOptions(screen.getByLabelText('数值字段'), 'close')
  expect(nodeValue().parameters).toEqual({ ...priceBinding, start_date: '2020-01-01' })
})

it('仅改字段不会静默切换已保存研究的数据快照', () => {
  const node = { ...blank, type: 'source.etf', parameters: { snapshot_generation: 'old-generation', source_file: 'etf_daily_candle_df.parquet', field: 'close' } }
  const item: ResearchSeriesCatalogItem = { ...csi300, fields: [{ name: 'adj_nav', binding_parameters: { snapshot_generation: 'new-generation', source_file: 'etf_daily_df.parquet', field: 'adj_nav' } }] }
  expect(researchSourceParameterPatch(node, 'field', 'adj_nav', item)).toEqual({ parameters: { ...node.parameters, field: 'adj_nav' } })
})
