import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import DataModelCatalog from './DataModelCatalog'
import type { DataModelCatalog as DataModelCatalogPayload, DataModelTable } from '../services/dataModel'

const field = (
  name: string,
  label: string,
  overrides: Partial<DataModelTable['fields'][number]> = {},
): DataModelTable['fields'][number] => ({
  name,
  label,
  data_type: 'string',
  nullable: false,
  role: 'dimension',
  description: `${label}说明`,
  unit: null,
  enum_values: [],
  reference: null,
  source_mappable: true,
  ...overrides,
})

const table = (
  tableId: string,
  categoryId: string,
  label: string,
  fields: DataModelTable['fields'],
  overrides: Partial<DataModelTable> = {},
): DataModelTable => {
  const sourceMappable = overrides.source_mappable ?? true
  return {
    table_id: tableId,
    usage: sourceMappable ? 'external_import' : 'system_internal',
    category_id: categoryId,
    label,
    description: `${label}的系统标准表。`,
    layer: 'canonical',
    storage_engine: 'parquet',
    storage_location: `data/canonical/v1/${tableId.replace('.', '/')}/`,
    delivery_phase: 'core',
    grain: `每个${label}一条记录`,
    primary_key: [fields[0].name],
    update_strategy: 'append',
    fields,
    source_mappable: sourceMappable,
    partition_by: [],
    sort_by: [fields[0].name],
    pit_supported: true,
    ...overrides,
  }
}

const principles = ['业务代码只使用平台内部 ID。']
const typeConventions = [
  { logical_type: 'business_date', physical_type: 'date32', rule: '只表达自然日。' },
]

const externalCatalog: DataModelCatalogPayload = {
  model_id: 'fund-investment-research-platform-data-model',
  schema_version: '1.1.0',
  status: 'defined',
  scope: 'external',
  description: '外部数据须映射到平台定义的导入表与字段。',
  principles,
  type_conventions: typeConventions,
  categories: [
    { category_id: 'master', label: '产品与参与方主数据', description: '统一内部 ID。', order: 20, table_count: 1, field_count: 2 },
    { category_id: 'market', label: '行情、净值与估值', description: '净值和行情。', order: 30, table_count: 1, field_count: 4 },
  ],
  tables: [
    table(
      'master.instrument',
      'master',
      '可投资标的',
      [
        field('instrument_id', '标的 ID', { role: 'primary_key', source_mappable: false }),
        field('canonical_name', '标准名称'),
      ],
      { layer: 'master', update_strategy: 'scd2' },
    ),
    table(
      'market.nav_daily',
      'market',
      '基金日净值',
      [
        field('instrument_id', '标的 ID', { role: 'foreign_key', source_mappable: false, reference: 'master.instrument.instrument_id' }),
        field('valuation_date', '估值日期', { data_type: 'date32', role: 'observation_time' }),
        field('available_at', '最早可得时间', { data_type: 'timestamp[us, UTC]', nullable: true, role: 'available_time' }),
        field('adjusted_nav', '复权净值', { data_type: 'float64', role: 'measure', unit: 'nav_index' }),
      ],
      {
        primary_key: ['instrument_id', 'valuation_date'],
        partition_by: ['valuation_date'],
        sort_by: ['instrument_id', 'valuation_date'],
      },
    ),
  ],
  summary: {
    category_count: 2,
    table_count: 2,
    field_count: 6,
    mapping_target_table_count: 2,
    mapping_target_field_count: 4,
    pit_table_count: 2,
    by_layer: { master: 1, canonical: 1 },
    by_storage_engine: { parquet: 2 },
    by_delivery_phase: { core: 2 },
  },
}

const internalCatalog: DataModelCatalogPayload = {
  model_id: externalCatalog.model_id,
  schema_version: '1.1.0',
  status: 'defined',
  scope: 'internal',
  description: '系统内部表由系统维护。',
  principles,
  type_conventions: typeConventions,
  categories: [
    { category_id: 'governance', label: '数据接入与治理', description: '数据源和映射。', order: 10, table_count: 1, field_count: 2 },
    { category_id: 'marts', label: '研究派生数据', description: '系统派生。', order: 70, table_count: 1, field_count: 2 },
  ],
  tables: [
    table(
      'governance.data_source',
      'governance',
      '数据源',
      [
        field('data_source_id', '数据源 ID', { role: 'primary_key', source_mappable: false }),
        field('name', '数据源名称', { source_mappable: false }),
      ],
      {
        usage: 'system_internal',
        layer: 'control',
        storage_engine: 'sqlite',
        storage_location: 'data/platform.db',
        source_mappable: false,
        update_strategy: 'upsert',
        pit_supported: false,
      },
    ),
    table(
      'mart.daily_return',
      'marts',
      '标准日收益率',
      [
        field('data_release_id', '数据版本 ID', { role: 'foreign_key', source_mappable: false }),
        field('return_value', '收益率', { data_type: 'float64', role: 'measure', unit: 'decimal', source_mappable: false }),
      ],
      {
        usage: 'system_internal',
        layer: 'mart',
        source_mappable: false,
        update_strategy: 'derived',
      },
    ),
  ],
  summary: {
    category_count: 2,
    table_count: 2,
    field_count: 4,
    mapping_target_table_count: 0,
    mapping_target_field_count: 0,
    pit_table_count: 1,
    by_layer: { control: 1, mart: 1 },
    by_storage_engine: { sqlite: 1, parquet: 1 },
    by_delivery_phase: { core: 2 },
  },
}

const successfulResponse = (payload: DataModelCatalogPayload) => ({
  ok: true,
  status: 200,
  json: async () => payload,
})

const installCatalogFetch = () => {
  const fetchMock = vi.fn().mockImplementation((url: string) => Promise.resolve(
    successfulResponse(url.includes('scope=internal') ? internalCatalog : externalCatalog),
  ))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

describe('DataModelCatalog', () => {
  afterEach(() => { vi.unstubAllGlobals() })

  it('默认展示外部导入表，并区分来源字段与系统维护字段', async () => {
    const fetchMock = installCatalogFetch()
    const user = userEvent.setup()

    render(<DataModelCatalog />)

    expect(await screen.findByRole('heading', { name: '系统数据模型' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '外部数据导入标准' })).toBeInTheDocument()
    expect(screen.getByText('Schema v1.1.0')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /产品与参与方主数据/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /行情、净值与估值/ })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: '产品与参与方主数据表' })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: '行情、净值与估值表' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '可投资标的' })).toBeInTheDocument()
    expect(screen.getByText('canonical_name', { selector: 'code' })).toBeInTheDocument()
    expect(screen.queryByText('instrument_id', { selector: 'code' })).not.toBeInTheDocument()

    await act(async () => {
      await user.click(screen.getByRole('button', { name: '高级：查看系统维护字段' }))
    })
    expect(screen.getByText('instrument_id', { selector: 'code' })).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith('/api/data-model/catalog', expect.objectContaining({ cache: 'no-store' }))
  })

  it('可按分类和字段搜索，并可查看完整系统内部表', async () => {
    const fetchMock = installCatalogFetch()
    const user = userEvent.setup()
    render(<DataModelCatalog />)
    await screen.findByRole('heading', { name: '外部数据导入标准' })

    await act(async () => {
      await user.click(screen.getByRole('button', { name: /行情、净值与估值/ }))
    })
    expect(await screen.findByRole('heading', { name: '基金日净值' })).toBeInTheDocument()
    expect(screen.getByText('adjusted_nav', { selector: 'code' })).toBeInTheDocument()
    expect(screen.getByText('单位：nav_index')).toBeInTheDocument()

    await act(async () => {
      await user.clear(screen.getByLabelText(/搜索业务表或字段/))
      await user.type(screen.getByLabelText(/搜索业务表或字段/), 'adjusted_nav')
    })
    expect(screen.getByLabelText('搜索命中字段')).toHaveTextContent('复权净值 · adjusted_nav')
    await act(async () => {
      await user.clear(screen.getByLabelText(/搜索业务表或字段/))
      await user.type(screen.getByLabelText(/搜索业务表或字段/), '不存在的字段')
    })
    expect(screen.getByText(/没有符合条件的表或字段/)).toBeInTheDocument()

    await act(async () => {
      await user.click(screen.getByRole('button', { name: '高级：系统内部表' }))
    })
    expect(await screen.findByRole('heading', { name: '系统内部结构' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /数据接入与治理/ })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '数据源' })).toBeInTheDocument()
    expect(screen.getByText('data_source_id', { selector: 'code' })).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/data-model/catalog?scope=internal',
      expect.objectContaining({ cache: 'no-store' }),
    )
  })

  it.each([
    ['master.organization_identifier', '机构外部标识映射'],
    ['master.person_identifier', '人员外部标识映射'],
    ['master.instrument_identifier', '标的外部代码映射'],
    ['portfolio.external_account_identifier', '外部账户标识映射'],
  ])('代码对照 %s 仅在高级内部视图可见', async (tableId, label) => {
    const crosswalk = table(tableId, 'master', label, [
      field('external_code', '外部代码', { source_mappable: false }),
    ], { source_mappable: false, layer: 'master', pit_supported: false })
    const advanced = {
      ...internalCatalog,
      categories: [...internalCatalog.categories, { category_id: 'master', label: '内部代码对照', description: '', order: 20, table_count: 1, field_count: 1 }],
      tables: [crosswalk, ...internalCatalog.tables],
      summary: { ...internalCatalog.summary, category_count: 3, table_count: 3, field_count: 5,
        by_layer: { ...internalCatalog.summary.by_layer, master: 1 },
        by_storage_engine: { sqlite: 1, parquet: 2 }, by_delivery_phase: { core: 3 } },
    }
    const fetchMock = vi.fn().mockImplementation((url: string) => Promise.resolve(
      successfulResponse(url.includes('scope=internal') ? advanced : externalCatalog),
    ))
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<DataModelCatalog />)
    await screen.findByRole('heading', { name: '外部数据导入标准' })
    expect(screen.queryByText(tableId, { selector: 'code' })).not.toBeInTheDocument()
    await act(async () => { await user.type(screen.getByLabelText(/搜索业务表或字段/), label) })
    expect(screen.getByText(/没有符合条件的表或字段/)).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(1)

    await act(async () => { await user.click(screen.getByRole('button', { name: '高级：系统内部表' })) })
    expect(await screen.findByRole('heading', { name: label })).toBeInTheDocument()
    expect(screen.getByText('系统内部表 · 不接受外部导入')).toBeInTheDocument()
    await act(async () => { await user.click(screen.getByRole('button', { name: '收起系统内部表' })) })
    await screen.findByRole('heading', { name: '外部数据导入标准' })
    expect(screen.queryByRole('heading', { name: label })).not.toBeInTheDocument()
  })

  it('接口失败时明确显示原因并允许重试', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({
        ok: false,
        status: 503,
        json: async () => ({ detail: { message: '数据模型服务不可用。' } }),
      })
      .mockResolvedValueOnce(successfulResponse(externalCatalog))
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()

    render(<DataModelCatalog />)

    expect(await screen.findByRole('alert')).toHaveTextContent('数据模型服务不可用。')
    await act(async () => {
      await user.click(screen.getByRole('button', { name: '重试' }))
    })
    expect(await screen.findByRole('heading', { name: '外部数据导入标准' })).toBeInTheDocument()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
  })
})
