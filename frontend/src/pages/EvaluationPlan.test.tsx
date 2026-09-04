import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import EvaluationPlan from './EvaluationPlan'
import {
  createEvaluationPlan,
  deleteEvaluationPlan,
  getCustomIndicatorMeta,
  listCustomIndicators,
  listEvaluationPlans,
  listInstrumentProducts,
  selectAllInstrumentProducts,
  runEvaluationPlan,
  updateEvaluationPlan,
  type EvaluationPlanDraft,
  type ProductKind,
} from '../services/customIndicators'
import { evaluateNumericControls } from '../services/businessNumeric'

vi.mock('../services/customIndicators', () => ({
  indicatorPeriodLabel: (period: string) => period,
  indicatorsForContext: (items: Array<{ context_kind?: string }>, context: string) => items.filter((item) => (item.context_kind ?? 'single_product') === context),
  listCustomIndicators: vi.fn(),
  getCustomIndicatorMeta: vi.fn(),
  listEvaluationPlans: vi.fn(),
  listInstrumentProducts: vi.fn(),
  selectAllInstrumentProducts: vi.fn(),
  createEvaluationPlan: vi.fn(),
  updateEvaluationPlan: vi.fn(),
  deleteEvaluationPlan: vi.fn(),
  runEvaluationPlan: vi.fn(),
}))
vi.mock('../services/businessNumeric', () => ({ evaluateNumericControls: vi.fn() }))

const indicator = {
  id: 'annual-return', revision: 3, source: 'custom', read_only: false,
  name: '年化收益率', description: '真实净值计算的年化收益率', expression: 'annualized_return(r)',
  periods: ['1Y'], unit: '%', display_format: 'percent', precision: 2,
  direction: 'higher_better', annual_risk_free_rate_percent: 1.5,
  context_kind: 'single_product', applicable_product_kinds: ['etf', 'fund'], catalog_status: 'current', ui_exposed: true,
  created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
} as const

const presentation = {
  indicator_id: indicator.id, revision: indicator.revision, name: indicator.name, source: indicator.source,
  category: 'return', category_label: '收益', context_kind: 'single_product', catalog_status: 'current',
  display_format: 'percent', precision: 2, unit: '%', notation: 'standard', value_scale: 100,
  output_measure: 'return_decimal', direction: 'higher_better', description: indicator.description,
  methodology: '按真实净值计算', data_basis: '复权净值', minimum_observations: 2,
  applicable_product_kinds: ['etf', 'fund'],
} as const

const productResponse = (kind: ProductKind) => ({
  items: kind === 'etf' ? [{
    code: '510300.SH', ts_code: '510300.SH', name: '沪深300ETF', management: '华泰柏瑞', custodian: '中国银行',
    found_date: '2012-05-04', list_date: '2012-05-28', instrument_type: 'etf', fund_type: '股票型', invest_type: '被动指数型', market: '上交所', status: '上市交易',
  }] : [{
    code: '000001.OF', ts_code: '000001.OF', name: '示例公募基金', management: '示例管理人', custodian: '示例托管行',
    found_date: '2001-01-01', instrument_type: 'fund', fund_type: '混合型', invest_type: '主动型', market: '场外', status: '存续',
  }],
  total: 1, page: 1, page_size: 10, kind,
  summary: { universe_total: 1, filtered_total: 1, active_count: 1 },
  available_filters: {
    fund_type: [{ value: kind === 'etf' ? '股票型' : '混合型', label: kind === 'etf' ? '股票型' : '混合型', count: 1 }],
    invest_type: [], market: [], status: [], management: [], custodian: [],
  },
  condition_fields: [
    { field: kind === 'etf' ? 'list_date' : 'found_date', label: kind === 'etf' ? '上市日期' : '成立日期', data_type: 'date', source: 'fund_basic', available: true },
    { field: 'return_1y', label: '近1年收益率', data_type: 'number', unit_label: '%', input_scale: 100, source: 'instrument_metrics_snapshot', available: true },
  ],
  condition_operators: [{ value: 'gte', label: '大于等于', symbol: '≥' }],
  snapshot: { status: 'ready', as_of: '2026-08-31' },
  sort_by: 'name', sort_dir: 'asc',
} as const)

describe('EvaluationPlan', () => {
  beforeEach(() => {
    vi.mocked(evaluateNumericControls).mockResolvedValue({
      items: [{ key: 'evaluation-indicator-weights', total: 100, difference: 0, within_tolerance: true, positive: true, normalized_shares: [1] }],
      execution: {
        execution_backend: 'numba_njit_fixed_signature',
        nopython: true,
        object_mode: 0,
        python_fallback: 0,
        request_time_compilation: 0,
        kernel_signatures: { numeric_control_kernel: ['fixed'] },
      },
    })
    vi.stubGlobal('crypto', { randomUUID: vi.fn(() => 'entry-id') })
    vi.mocked(listCustomIndicators).mockResolvedValue({ items: [indicator], total: 1 })
    vi.mocked(getCustomIndicatorMeta).mockResolvedValue({ periods: [{ value: '1Y', label: '近 1 年', description: '运行周期' }] } as any)
    vi.mocked(listEvaluationPlans).mockResolvedValue({ items: [], total: 0 })
    vi.mocked(listInstrumentProducts).mockImplementation(async ({ kind }) => productResponse(kind) as any)
    vi.mocked(selectAllInstrumentProducts).mockImplementation(async ({ kind }) => ({
      items: productResponse(kind).items,
      total: productResponse(kind).items.length,
      kind,
    }) as any)
    vi.mocked(createEvaluationPlan).mockImplementation(async (draft: EvaluationPlanDraft) => ({
      ...draft, id: 'plan-1', revision: 1, created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
    }))
    vi.mocked(runEvaluationPlan).mockResolvedValue({
      plan_id: 'plan-1', plan_revision: 1, run_at: '2026-01-02T00:00:00Z', as_of: null,
      ranked_count: 1, excluded_count: 0,
      normalization: { method: 'min_max_0_100', configured_weight_total: 100, effective_weight_total: 1, missing_policy: 'strict' },
      rows: [{
        rank: 1, target: { kind: 'etf', product_id: '510300.SH', name: '沪深300ETF' }, score: 50,
        status: 'ranked', missing_indicators: [], values: [{
          indicator_id: indicator.id, indicator_revision: 3, indicator_name: indicator.name, period: '1Y', value: 0.123,
          status: 'ok', warnings: [], window: { requested_as_of: null, effective_as_of: '2026-01-02', start_date: '2025-01-02', end_date: '2026-01-02', observation_count: 250, data_latest_date: '2026-01-02' },
          presentation, direction: 'higher_better', definition_direction: 'higher_better', direction_overridden: false,
          configured_weight: 100, effective_weight: 1, normalized_score: 50, weighted_contribution: 50,
        }],
      }],
    } as any)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('默认仅加载 ETF 方案、ETF 指标和 ETF 产品并保存同类方案', async () => {
    const user = userEvent.setup()
    await act(async () => { render(<MemoryRouter initialEntries={['/evaluation-plan']}><EvaluationPlan /></MemoryRouter>) })

    expect(await screen.findByText('沪深300ETF')).toBeInTheDocument()
    expect(screen.queryByText('示例公募基金')).not.toBeInTheDocument()
    expect(listCustomIndicators).toHaveBeenCalledWith(expect.objectContaining({ productKind: 'etf' }))
    expect(listEvaluationPlans).toHaveBeenCalledWith('etf')
    expect(listInstrumentProducts).toHaveBeenCalledWith(expect.objectContaining({ kind: 'etf' }))

    await user.type(screen.getByLabelText('方案名称'), 'ETF优选方案')
    await user.click(screen.getByLabelText('选择 沪深300ETF'))
    await user.click(screen.getByRole('button', { name: '保存并运行' }))

    await waitFor(() => expect(createEvaluationPlan).toHaveBeenCalledWith(expect.objectContaining({
      name: 'ETF优选方案', product_kind: 'etf', targets: [{ kind: 'etf', product_id: '510300.SH' }],
    })))
    expect(await screen.findByText('已排名')).toBeInTheDocument()
  })

  it('切换到场外公募基金后独立加载方案、指标、筛选和产品', async () => {
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/evaluation-plan']}><EvaluationPlan /></MemoryRouter>)
    await screen.findByText('沪深300ETF')

    await user.click(screen.getByRole('tab', { name: '场外公募基金' }))

    expect(await screen.findByText('示例公募基金')).toBeInTheDocument()
    expect(screen.queryByText('沪深300ETF')).not.toBeInTheDocument()
    expect(screen.getByRole('tab', { name: '场外公募基金' })).toHaveAttribute('aria-selected', 'true')
    await waitFor(() => {
      expect(listCustomIndicators).toHaveBeenLastCalledWith(expect.objectContaining({ productKind: 'fund' }))
      expect(listEvaluationPlans).toHaveBeenLastCalledWith('fund')
      expect(listInstrumentProducts).toHaveBeenLastCalledWith(expect.objectContaining({ kind: 'fund' }))
    })
  })

  it('支持分类筛选和日期或快照指标条件', async () => {
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/evaluation-plan']}><EvaluationPlan /></MemoryRouter>)
    await screen.findByText('沪深300ETF')

    await user.click(screen.getByRole('button', { name: '投资类型 全部' }))
    await user.click(screen.getByRole('checkbox', { name: /股票型/ }))
    await user.selectOptions(screen.getByLabelText('评价方案筛选字段'), 'return_1y')
    await user.type(screen.getByLabelText('评价方案筛选值'), '5')
    await user.click(screen.getByRole('button', { name: '添加条件' }))

    await waitFor(() => expect(listInstrumentProducts).toHaveBeenLastCalledWith(expect.objectContaining({
      kind: 'etf',
      filters: expect.objectContaining({ fund_type: ['股票型'] }),
      conditions: [{ field: 'return_1y', operator: 'gte', value: '5' }],
    })))
    expect(screen.getByText('近1年收益率 ≥ 5% ×')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /机构类型|基金类型/ })).not.toBeInTheDocument()
  })

  it('默认每页 10 个，可调整数量并全选全部筛选结果', async () => {
    const matchingItems = Array.from({ length: 12 }, (_, index) => ({
      ...productResponse('etf').items[0],
      code: `510${String(index).padStart(3, '0')}.SH`,
      ts_code: `510${String(index).padStart(3, '0')}.SH`,
      name: `筛选产品${index + 1}`,
    }))
    vi.mocked(listInstrumentProducts).mockResolvedValue({
      ...productResponse('etf'),
      items: matchingItems.slice(0, 10),
      total: 12,
      summary: { universe_total: 12, filtered_total: 12, active_count: 12 },
    } as any)
    vi.mocked(selectAllInstrumentProducts).mockResolvedValue({ items: matchingItems, total: 12, kind: 'etf' } as any)
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/evaluation-plan']}><EvaluationPlan /></MemoryRouter>)

    await screen.findByText('筛选产品1')
    expect(listInstrumentProducts).toHaveBeenCalledWith(expect.objectContaining({ pageSize: 10 }))

    await user.selectOptions(screen.getByLabelText('评价方案每页产品数量'), '20')
    await waitFor(() => expect(listInstrumentProducts).toHaveBeenLastCalledWith(expect.objectContaining({ pageSize: 20 })))

    await user.click(screen.getByRole('button', { name: '全选 12 条' }))
    expect(await screen.findByText('已选择全部符合筛选条件的产品')).toBeInTheDocument()
    expect(screen.getByText('已选 12')).toBeInTheDocument()
    expect(selectAllInstrumentProducts).toHaveBeenCalledWith(expect.objectContaining({ kind: 'etf' }))

    await user.type(screen.getByLabelText('方案名称'), '全量候选方案')
    await user.click(screen.getByRole('button', { name: '保存方案' }))
    await waitFor(() => expect(createEvaluationPlan).toHaveBeenCalledWith(expect.objectContaining({
      product_kind: 'etf',
      targets: matchingItems.map((item) => ({ kind: 'etf', product_id: item.ts_code })),
    })))
  })

  it('切换已保存方案时恢复筛选条件、全选模式和产品勾选', async () => {
    const savedPlan = {
      id: 'saved-plan',
      revision: 4,
      name: '股票 ETF 方案',
      description: '恢复测试',
      product_kind: 'etf',
      indicators: [{
        indicator_id: indicator.id,
        indicator_revision: indicator.revision,
        period: '1Y',
        weight: 100,
        direction: 'higher_better',
      }],
      targets: [{ kind: 'etf', product_id: '510300.SH' }],
      product_selection: {
        query: '沪深300',
        filters: {
          fund_type: ['股票型'],
          invest_type: [],
          qdii_type: [],
          market: [],
          status: [],
          management: [],
          custodian: [],
        },
        conditions: [{ field: 'return_1y', operator: 'gte', value: '5' }],
        selection_mode: 'all_matching',
      },
      missing_policy: 'strict',
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-02T00:00:00Z',
    } as const
    vi.mocked(listEvaluationPlans).mockResolvedValue({ items: [savedPlan], total: 1 } as any)
    vi.mocked(updateEvaluationPlan).mockImplementation(async (_id, draft) => ({
      ...savedPlan,
      ...draft,
      revision: 5,
      updated_at: '2026-01-03T00:00:00Z',
    }) as any)
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/evaluation-plan']}><EvaluationPlan /></MemoryRouter>)

    await screen.findByText('沪深300ETF')
    await user.selectOptions(screen.getByRole('combobox', { name: '已保存ETF方案' }), savedPlan.id)

    await waitFor(() => expect(listInstrumentProducts).toHaveBeenLastCalledWith(expect.objectContaining({
      kind: 'etf',
      query: '沪深300',
      filters: expect.objectContaining({ fund_type: ['股票型'] }),
      conditions: [{ field: 'return_1y', operator: 'gte', value: '5' }],
    })))
    expect(screen.getByPlaceholderText('按代码、名称或管理人搜索')).toHaveValue('沪深300')
    expect(screen.getByRole('button', { name: '投资类型 1 项' })).toBeInTheDocument()
    expect(screen.getByText('投资类型：股票型 ×')).toBeInTheDocument()
    expect(screen.getByText('近1年收益率 ≥ 5% ×')).toBeInTheDocument()
    expect(screen.getByLabelText('选择 沪深300ETF')).toBeChecked()
    expect(screen.getByRole('button', { name: '取消全选' })).toBeInTheDocument()
    expect(screen.getByText('已选择全部符合筛选条件的产品')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '保存方案' }))
    await waitFor(() => expect(updateEvaluationPlan).toHaveBeenCalledWith(
      savedPlan.id,
      expect.objectContaining({ product_selection: savedPlan.product_selection }),
      savedPlan.revision,
    ))
  })
})
