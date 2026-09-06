import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import DataSourceCenter from './DataSourceCenter'
import * as api from '../services/dataSources'
import type { SourceCatalog } from '../services/dataSources'
import type { DataModelTable } from '../services/dataModel'

vi.mock('../services/dataSources', () => ({
  fetchSourceCatalog: vi.fn(), saveSource: vi.fn(), saveInterface: vi.fn(), deleteSourceConfig: vi.fn(),
  saveSourceCredential: vi.fn(), validateSourceMapping: vi.fn(), previewSourceMapping: vi.fn(), sampleSourceInterface: vi.fn(),
}))

const policy = { requests_per_minute: 60, rows_per_minute: null, min_interval_seconds: 0.2, max_rows_per_request: 5000, max_concurrency: 1, connect_timeout_seconds: 5, read_timeout_seconds: 30, max_attempts: 3, backoff_seconds: 2, rate_limit_wait_seconds: 60, max_response_bytes: 8388608, max_runtime_seconds: 3600 }
const source = { id: 'tushare', name: 'Tushare', transport: 'tushare' as const, base_url: 'https://api.tushare.pro', enabled: true, auth_mode: 'none' as const, auth_header: 'X-API-Key', notes: '', policy }
const target: DataModelTable = { table_id: 'market.quote_daily', usage: 'external_import', category_id: 'market', label: '日行情', description: '日行情合同', layer: 'canonical', storage_engine: 'parquet', storage_location: 'data/canonical/market/quote_daily', delivery_phase: 'core', grain: '每个标的每日', primary_key: ['instrument_id', 'trade_date'], update_strategy: 'append', source_mappable: true, partition_by: [], sort_by: [], pit_supported: true,
  fields: [
    { name: 'instrument_id', label: '标的 ID', data_type: 'string', nullable: false, role: 'foreign_key', description: '内部 ID', unit: null, enum_values: [], reference: 'master.instrument.instrument_id', source_mappable: false },
    { name: 'close', label: '收盘价', data_type: 'float64', nullable: true, role: 'measure', description: '交易收盘价', unit: 'CNY', enum_values: [], reference: null, source_mappable: true },
  ],
}
const config = { id: 'tushare.fund_daily', source_id: 'tushare', name: 'ETF 日行情', enabled: true, api_name: 'fund_daily', method: 'POST' as const, path: '', params: {}, headers: {}, response: { format: 'json_columns' as const, records_path: 'data.items', columns_path: 'data.fields', delimiter: ',' }, source_fields: [{ name: 'close', data_type: 'number' as const, description: '收盘', unit: 'CNY' }], policy, pagination: { mode: 'none' as const, cursor_param: 'offset', limit_param: 'limit', page_size: 1000, max_pages: 20 }, start_param: 'start_date', end_param: 'end_date', incremental_field: 'trade_date', notes: '', entitlement_confirmed: false,
  mappings: [{ target_table: target.table_id, contract_version: '1.2.0', enabled: true, identities: [], fields: [{ target_field: 'close', source_field: 'close', operation: 'copy' as const, factor: 1, constant: null, enum_map: {}, timezone: 'Asia/Shanghai', date_format: null }] }],
}
let catalog: SourceCatalog
const ready = { valid: true, ready: true, errors: [], warnings: [] }

beforeEach(() => {
  vi.resetAllMocks()
  catalog = { sources: [{ config: structuredClone(source), revision: 1, builtin: true, updated_at: '', credential_configured: true }], interfaces: [{ config: structuredClone(config), revision: 1, builtin: true, updated_at: '', validation: ready, effective_policy: policy }], editing_enabled: true, boundary: '候选数据不替代研究数据。', templates: { source: { ...source, id: 'custom', name: '新数据源', transport: 'http', base_url: 'https://example.com', enabled: false }, interface: { ...config, id: 'custom.endpoint', api_name: '', source_id: 'custom', enabled: false, mappings: [], source_fields: [] } }, targets: { model_id: 'platform', schema_version: '1.2.0', status: 'defined', scope: 'external', description: '', principles: [], type_conventions: [], categories: [{ category_id: 'market', label: '行情、净值与估值', description: '行情和估值数据。', order: 30, table_count: 1, field_count: 2 }], tables: [target, { ...target, table_id: 'mart.metric', label: '系统内部指标', source_mappable: false, usage: 'system_internal' }], summary: { table_count: 1, category_count: 1, field_count: 2, mapping_target_table_count: 1, mapping_target_field_count: 1, pit_table_count: 1, by_layer: {}, by_storage_engine: {}, by_delivery_phase: {} } } }
  vi.mocked(api.fetchSourceCatalog).mockImplementation(async () => structuredClone(catalog))
  vi.mocked(api.saveInterface).mockImplementation(async value => {
    catalog.interfaces[0] = { ...catalog.interfaces[0], config: structuredClone(value), revision: 2 }
    return catalog.interfaces[0]
  })
  vi.mocked(api.saveSource).mockImplementation(async value => ({ config: value, revision: 1, builtin: false, updated_at: '' }))
  vi.mocked(api.validateSourceMapping).mockResolvedValue(ready)
  vi.spyOn(window, 'confirm').mockReturnValue(true)
})

async function openInterface() {
  render(<MemoryRouter><DataSourceCenter /></MemoryRouter>)
  await screen.findByRole('heading', { name: '数据源与接口映射' })
  fireEvent.click(screen.getByRole('button', { name: /ETF 日行情/ }))
  await screen.findByRole('button', { name: '保存接口配置' })
}

describe('DataSourceCenter', () => {
  it('默认先展示使用引导，不要求用户重新填写 Tushare', async () => {
    render(<MemoryRouter><DataSourceCenter /></MemoryRouter>)
    expect(await screen.findByRole('region', { name: '数据源使用引导' })).toBeVisible()
    expect(screen.queryByRole('button', { name: '保存数据源' })).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: '进入 Tushare 下载与更新 →' })).toHaveAttribute('href', '/settings/data-sources?source=tushare')
    expect(screen.getByRole('navigation', { name: '数据工作区' })).toBeVisible()
    expect(screen.getByText(/凭据已保存；接口权限需要通过实际请求验证/)).toBeVisible()
  })

  it('待完善筛选与接口定义状态一致', async () => {
    render(<MemoryRouter><DataSourceCenter /></MemoryRouter>)
    await screen.findByRole('region', { name: '数据源使用引导' })
    fireEvent.click(screen.getByRole('button', { name: '查看待完善映射' }))
    expect(screen.getByLabelText('接口状态')).toHaveValue('incomplete')
    expect(screen.queryByRole('button', { name: /ETF 日行情/ })).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('接口状态'), { target: { value: 'all' } })
    expect(screen.getByRole('button', { name: /ETF 日行情/ })).toBeInTheDocument()
  })

  it('自定义来源说明接口下载入口并明确未开放定时调度', async () => {
    catalog.sources = [{ config: { ...source, id: 'vendor', name: '测试来源', transport: 'http', base_url: 'https://example.com' }, revision: 1, builtin: false, updated_at: '' }]
    catalog.interfaces = []
    render(<MemoryRouter><DataSourceCenter /></MemoryRouter>)
    await screen.findByRole('region', { name: '数据源使用引导' })
    expect(screen.getByText(/进入统一下载页选择本来源支持的数据/)).toBeVisible()
    expect(screen.getByText(/定时调度尚未开放/)).toBeVisible()
    expect(screen.queryByRole('link', { name: '进入 Tushare 下载与更新 →' })).not.toBeInTheDocument()
  })

  it('非法 JSON 草稿切换前仍会提示，取消后保留原输入', async () => {
    await openInterface()
    const input = screen.getByLabelText('默认请求参数（JSON，不含凭据）')
    fireEvent.change(input, { target: { value: '{invalid' } })
    vi.mocked(window.confirm).mockReturnValue(false)
    fireEvent.click(screen.getByRole('button', { name: '来源概览' }))
    expect(window.confirm).toHaveBeenCalledWith('当前修改尚未保存，放弃修改并切换？')
    expect(input).toHaveValue('{invalid')
    expect(screen.getByRole('button', { name: '保存接口配置' })).toBeInTheDocument()
  })

  it('已保存接口可展开编辑，不因初始化来源锁定', async () => {
    await openInterface()
    expect(screen.getByLabelText('接口 API 名称')).not.toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: '1. 接口结构' }))
    expect(screen.getByLabelText('接口 API 名称')).toBeVisible()
    expect(screen.getByLabelText('接口 API 名称')).toBeEnabled()
    expect(screen.getByLabelText('请求方式')).toBeEnabled()
    expect(screen.getByLabelText('响应数据格式')).toBeEnabled()
    expect(screen.getByLabelText('分页方式')).toBeEnabled()
    expect(screen.queryByText('系统预置')).not.toBeInTheDocument()
  })

  it('展示已保存来源并只允许选择外部导入目标', async () => {
    await openInterface()
    expect(screen.getByLabelText('接口 API 名称')).toHaveValue('fund_daily')
    expect(screen.getByLabelText('接口 API 名称')).toBeEnabled()
    expect(screen.getByLabelText('close 转换方式')).toHaveValue('copy')
    expect(screen.queryByRole('option', { name: /系统内部指标/ })).not.toBeInTheDocument()
    expect(screen.getByRole('group', { name: '行情、净值与估值（1）' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '进入 Tushare 下载与更新 →' })).toHaveAttribute('href', '/settings/data-sources')
  })

  it('保存限制与映射并携带原修订号', async () => {
    await openInterface()
    fireEvent.change(screen.getByLabelText('每分钟请求次数'), { target: { value: '30' } })
    fireEvent.click(screen.getByRole('button', { name: '保存接口配置' }))
    await waitFor(() => expect(api.saveInterface).toHaveBeenCalledWith(expect.objectContaining({ policy: expect.objectContaining({ requests_per_minute: 30 }) }), 1))
    expect(await screen.findByText(/修订 2/)).toBeInTheDocument()
  })

  it('JSON 草稿非法时禁止提交旧值', async () => {
    await openInterface()
    fireEvent.change(screen.getByLabelText('默认请求参数（JSON，不含凭据）'), { target: { value: '{invalid' } })
    fireEvent.click(screen.getByRole('button', { name: '保存接口配置' }))
    expect(api.saveInterface).not.toHaveBeenCalled()
    expect(screen.getByRole('alert')).toHaveTextContent('请修正标出的输入')
  })

  it('修订冲突保留草稿并显示错误', async () => {
    await openInterface()
    vi.mocked(api.saveInterface).mockRejectedValue(new Error('配置已被修改，请重新加载后再保存。'))
    fireEvent.change(screen.getByLabelText('接口名称'), { target: { value: '我修改的接口' } })
    fireEvent.click(screen.getByRole('button', { name: '保存接口配置' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('配置已被修改')
    expect(screen.getByLabelText('接口名称')).toHaveValue('我修改的接口')
  })

  it('离线预览不触发真实采样', async () => {
    await openInterface()
    vi.mocked(api.previewSourceMapping).mockResolvedValue({ ...ready, source_rows: 1, preview_only: true, tables: [{ table_id: target.table_id, columns: ['close'], rows: [{ close: 3.5 }], accepted_rows: 1, rejected_rows: 0 }] })
    fireEvent.change(screen.getByLabelText('粘贴来源样本（按接口响应格式）'), { target: { value: '{"data":{"fields":["close"],"items":[[3.5]]}}' } })
    fireEvent.click(screen.getByRole('button', { name: '离线映射预览' }))
    expect(await screen.findByText('3.5')).toBeInTheDocument()
    expect(api.sampleSourceInterface).not.toHaveBeenCalled()
  })

  it('真实采样需要确认，未保存修改不能采样', async () => {
    await openInterface()
    vi.mocked(api.sampleSourceInterface).mockResolvedValue({ ...ready, source_rows: 1, tables: [], preview_only: true, requests: 1, download_complete: false })
    const button = screen.getByRole('button', { name: '确认并采样一次', hidden: true })
    vi.mocked(window.confirm).mockReturnValueOnce(false)
    fireEvent.click(button)
    expect(api.sampleSourceInterface).not.toHaveBeenCalled()
    fireEvent.click(button)
    await waitFor(() => expect(api.sampleSourceInterface).toHaveBeenCalledWith(config.id, 1, {}))
    await waitFor(() => expect(button).toBeEnabled())
    fireEvent.change(screen.getByLabelText('接口名称'), { target: { value: '未保存' } })
    expect(button).toBeDisabled()
  })

  it('创建自定义 HTTPS 数据源', async () => {
    render(<MemoryRouter><DataSourceCenter /></MemoryRouter>)
    await screen.findByRole('heading', { name: '数据源与接口映射' })
    fireEvent.click(screen.getByRole('button', { name: '新建数据源' }))
    fireEvent.change(screen.getByLabelText('数据源 ID'), { target: { value: 'vendor' } })
    fireEvent.change(screen.getByLabelText('数据源名称'), { target: { value: '自定义行情' } })
    fireEvent.click(screen.getByRole('button', { name: '保存数据源' }))
    await waitFor(() => expect(api.saveSource).toHaveBeenCalledWith(expect.objectContaining({ id: 'vendor', transport: 'http', name: '自定义行情' }), 0))
  })

  it('只读环境不允许保存或真实采样', async () => {
    catalog.editing_enabled = false
    await openInterface()
    expect(screen.getByRole('button', { name: '保存接口配置' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '确认并采样一次', hidden: true })).toBeDisabled()
    expect(screen.getByRole('button', { name: '校验映射定义' })).toBeEnabled()
  })
})
