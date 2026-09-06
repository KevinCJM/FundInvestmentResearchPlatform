import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ProductPools from './ProductPools'

const mocks = vi.hoisted(() => ({
  listEvaluationPlans: vi.fn(),
  listProductPools: vi.fn(),
  createProductPool: vi.fn(),
  updateProductPool: vi.fn(),
  attachEvaluationPlan: vi.fn(),
  removeEvaluationPlan: vi.fn(),
  addManualPoolMember: vi.fn(),
  updateProductPoolMember: vi.fn(),
  batchUpdateProductPoolMembers: vi.fn(),
  getProductPoolReviewData: vi.fn(),
  publishProductPool: vi.fn(),
}))

vi.mock('../services/customIndicators', () => ({
  listEvaluationPlans: mocks.listEvaluationPlans,
}))

vi.mock('../services/productPools', () => ({
  listProductPools: mocks.listProductPools,
  createProductPool: mocks.createProductPool,
  updateProductPool: mocks.updateProductPool,
  attachEvaluationPlan: mocks.attachEvaluationPlan,
  removeEvaluationPlan: mocks.removeEvaluationPlan,
  addManualPoolMember: mocks.addManualPoolMember,
  updateProductPoolMember: mocks.updateProductPoolMember,
  batchUpdateProductPoolMembers: mocks.batchUpdateProductPoolMembers,
  getProductPoolReviewData: mocks.getProductPoolReviewData,
  publishProductPool: mocks.publishProductPool,
}))

const equityPlan = { id: 'plan-equity', revision: 3, name: '权益 ETF 评价', product_kind: 'etf' }
const bondPlan = { id: 'plan-bond', revision: 2, name: '固收基金评价', product_kind: 'fund' }

const equityBinding = {
  plan_id: 'plan-equity', plan_revision: 3, plan_name: '权益 ETF 评价', product_kind: 'etf', result_id: 'run-1',
  as_of: '2026-08-31', selection_mode: 'all_ranked', selection_value: null, ranked_count: 2, excluded_count: 0,
  imported_count: 2, attached_at: '2026-09-04T00:00:00Z',
}

const bondBinding = {
  plan_id: 'plan-bond', plan_revision: 2, plan_name: '固收基金评价', product_kind: 'fund', result_id: 'run-2',
  as_of: '2026-08-31', selection_mode: 'top_n', selection_value: 5, ranked_count: 8, excluded_count: 1,
  imported_count: 5, attached_at: '2026-09-04T00:01:00Z',
}

const factorBinding = {
  plan_id: 'plan-factor', plan_revision: 1, plan_name: '权益因子评价', product_kind: 'etf', result_id: 'run-3',
  as_of: '2026-08-31', selection_mode: 'all_ranked', selection_value: null, ranked_count: 3, excluded_count: 0,
  imported_count: 3, attached_at: '2026-09-04T00:02:00Z',
}

const basePool = {
  id: 'pool-1', revision: 1, name: '核心产品池', description: '', purpose: '长期配置', owner: 'Kevin', state: 'draft',
  current_version_id: null, published_at: null, created_at: '2026-09-01T00:00:00Z', updated_at: '2026-09-04T00:00:00Z',
  evaluation_plans: [],
  members: [],
}

const secondPool = {
  ...basePool,
  id: 'pool-2',
  revision: 2,
  name: '固收产品池',
  purpose: '固收配置',
  owner: 'Alice',
  evaluation_plans: [bondBinding],
}

const researchedPool = {
  ...basePool,
  revision: 4,
  evaluation_plans: [equityBinding],
  members: [{
    key: 'etf:510300.SH', kind: 'etf', product_id: '510300.SH', code: '510300.SH', name: '沪深300ETF',
    research_status: 'pending', usage_status: 'normal', primary_plan_id: 'plan-equity', max_weight: null, reasons: [], owner: '',
    review_due_date: null, valid_until: null, substitute_group: '', manual_exception: false,
    evidences: [{ source: 'evaluation_plan', plan_id: 'plan-equity', plan_revision: 3, plan_name: '权益 ETF 评价', result_id: 'run-1', as_of: '2026-08-31', rank: 1, score: 91.2, percentile: null, result_status: 'ranked', exclusion_reason: null }],
  }],
}

const reviewPool = {
  ...basePool,
  revision: 4,
  evaluation_plans: [equityBinding],
  members: [
    {
      key: 'etf:510300.SH', kind: 'etf', product_id: '510300.SH', code: '510300.SH', name: '沪深300ETF',
      research_status: 'pending', usage_status: 'normal', primary_plan_id: 'plan-equity', max_weight: null, reasons: [], owner: '',
      review_due_date: null, valid_until: null, substitute_group: '', manual_exception: false,
      evidences: [{ source: 'evaluation_plan', plan_id: 'plan-equity', plan_revision: 3, plan_name: '权益 ETF 评价', result_id: 'run-1', as_of: '2026-08-31', rank: 2, score: 82.4, percentile: null, result_status: 'ranked', exclusion_reason: null }],
    },
    {
      key: 'etf:510500.SH', kind: 'etf', product_id: '510500.SH', code: '510500.SH', name: '中证500ETF',
      research_status: 'pending', usage_status: 'normal', primary_plan_id: 'plan-equity', max_weight: null, reasons: [], owner: '',
      review_due_date: null, valid_until: null, substitute_group: '', manual_exception: false,
      evidences: [{ source: 'evaluation_plan', plan_id: 'plan-equity', plan_revision: 3, plan_name: '权益 ETF 评价', result_id: 'run-1', as_of: '2026-08-31', rank: 1, score: 91.2, percentile: null, result_status: 'ranked', exclusion_reason: null }],
    },
  ],
}

const reviewData = {
  pool_id: 'pool-1',
  pool_revision: 4,
  basic_fields: [
    { field: 'management', label: '管理人', source: 'basic', data_type: 'text', unit: null, available: true, applicable_product_kinds: ['etf', 'fund'] },
    { field: 'issue_amount', label: '发行规模', source: 'basic', data_type: 'number', unit: 'project_normalized_wan', available: true, applicable_product_kinds: ['etf', 'fund'] },
  ],
  snapshot_metric_fields: [
    { field: 'return_1y', label: '近1年收益率', source: 'snapshot', data_type: 'number', unit: 'ratio', display_format: 'percent', available: true, applicable_product_kinds: ['etf', 'fund'] },
    { field: 'sharpe_1y', label: '近1年夏普比率', source: 'snapshot', data_type: 'number', unit: 'ratio', display_format: 'number', available: true, applicable_product_kinds: ['etf', 'fund'] },
  ],
  selected_basic_fields: ['management'],
  selected_snapshot_metrics: ['return_1y'],
  snapshot_states: { etf: { status: 'ready' } },
  rows: [
    {
      key: 'etf:510300.SH', kind: 'etf', product_id: '510300.SH',
      basic_values: { management: '甲基金', issue_amount: 52_155 },
      snapshot_values: { return_1y: 0.08, sharpe_1y: 0.5 },
      snapshot_value_dates: { return_1y: '2026-08-31', sharpe_1y: '2026-08-31' },
      snapshot_statuses: { return_1y: 'available', sharpe_1y: 'available' },
      snapshot_warnings: { return_1y: null, sharpe_1y: null },
    },
    {
      key: 'etf:510500.SH', kind: 'etf', product_id: '510500.SH',
      basic_values: { management: '乙基金', issue_amount: 9_800 },
      snapshot_values: { return_1y: 0.12, sharpe_1y: 0.8 },
      snapshot_value_dates: { return_1y: '2026-08-31', sharpe_1y: '2026-08-31' },
      snapshot_statuses: { return_1y: 'available', sharpe_1y: 'available' },
      snapshot_warnings: { return_1y: null, sharpe_1y: null },
    },
  ],
}

beforeEach(() => {
  vi.clearAllMocks()
  mocks.listProductPools.mockResolvedValue({ items: [researchedPool], total: 1 })
  mocks.listEvaluationPlans.mockResolvedValue({ items: [equityPlan, bondPlan], total: 2 })
  mocks.getProductPoolReviewData.mockResolvedValue(reviewData)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('ProductPools', () => {
  it('uses evaluation plans directly as product groups', async () => {
    render(<ProductPools />)
    expect(await screen.findByRole('heading', { name: '产品池构建' })).toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: '权益 ETF 评价' })).toBeInTheDocument()
    expect(screen.getByText(/评价方案就是产品分组/)).toBeInTheDocument()
    expect(screen.getByText('沪深300ETF')).toBeInTheDocument()
    expect(screen.getByText(/待复核 1/)).toBeInTheDocument()
  })

  it('shows every evaluation plan as read-only evidence instead of a dropdown', async () => {
    const multiPlanPool = {
      ...researchedPool,
      evaluation_plans: [equityBinding, factorBinding],
      members: researchedPool.members.map((member) => ({
        ...member,
        evidences: [
          ...member.evidences,
          {
            source: 'evaluation_plan', plan_id: 'plan-factor', plan_revision: 1, plan_name: '权益因子评价', result_id: 'run-3',
            as_of: '2026-08-31', rank: 3, score: 76.5, percentile: null, result_status: 'ranked', exclusion_reason: null,
          },
        ],
      })),
    }
    mocks.listProductPools.mockResolvedValue({ items: [multiPlanPool], total: 1 })

    render(<ProductPools />)

    const productName = await screen.findByText('沪深300ETF')
    const row = productName.closest('tr')
    expect(row).not.toBeNull()
    expect(within(row as HTMLTableRowElement).getByText('权益 ETF 评价')).toBeInTheDocument()
    expect(within(row as HTMLTableRowElement).getByText('权益因子评价')).toBeInTheDocument()
    expect(within(row as HTMLTableRowElement).getByText('主证据')).toBeInTheDocument()
    expect(within(row as HTMLTableRowElement).queryByLabelText('沪深300ETF 所属评价方案')).not.toBeInTheDocument()
  })

  it('places product-pool navigation in the header and keeps the workspace full width', async () => {
    const user = userEvent.setup()
    mocks.listProductPools.mockResolvedValue({ items: [basePool, secondPool], total: 2 })
    render(<ProductPools />)

    const selector = await screen.findByLabelText('当前产品池')
    expect(selector).toHaveValue('pool-1')
    expect(screen.getByText('共 2 个产品池')).toBeInTheDocument()
    expect(screen.getByTestId('product-pool-workspace')).toHaveClass('w-full')
    expect(screen.queryByRole('complementary')).not.toBeInTheDocument()

    await user.selectOptions(selector, 'pool-2')
    expect(selector).toHaveValue('pool-2')
    expect(screen.getByLabelText('名称')).toHaveValue('固收产品池')
    expect(screen.getByLabelText('用途')).toHaveValue('固收配置')
  })

  it('creates a product-pool draft from the header dialog', async () => {
    const user = userEvent.setup()
    const createdPool = {
      ...basePool,
      id: 'pool-new',
      name: '战术产品池',
      purpose: '战术配置',
      owner: 'Kevin',
    }
    mocks.listProductPools.mockResolvedValue({ items: [basePool], total: 1 })
    mocks.createProductPool.mockResolvedValue(createdPool)
    render(<ProductPools />)

    await screen.findByLabelText('当前产品池')
    await user.click(screen.getByRole('button', { name: /新建产品池/ }))
    const dialog = screen.getByRole('dialog', { name: '新建产品池' })
    await user.type(within(dialog).getByLabelText('新产品池名称'), '战术产品池')
    await user.type(within(dialog).getByLabelText('新产品池用途'), '战术配置')
    await user.type(within(dialog).getByLabelText('新产品池负责人'), 'Kevin')
    await user.click(within(dialog).getByRole('button', { name: '创建草稿' }))

    await waitFor(() => expect(mocks.createProductPool).toHaveBeenCalledWith({
      name: '战术产品池',
      description: '',
      purpose: '战术配置',
      owner: 'Kevin',
    }))
    await waitFor(() => expect(screen.queryByRole('dialog', { name: '新建产品池' })).not.toBeInTheDocument())
    expect(screen.getByLabelText('当前产品池')).toHaveValue('pool-new')
    expect(screen.getByRole('status')).toHaveTextContent('产品池草稿已创建。')
  })

  it('shows a dash when all ranked products are selected', async () => {
    mocks.listProductPools.mockResolvedValue({ items: [basePool], total: 1 })
    render(<ProductPools />)

    const amount = await screen.findByLabelText('N / 百分比')
    expect(amount).toHaveValue('-')
    expect(amount).toBeDisabled()

    fireEvent.change(screen.getByLabelText('导入方式'), { target: { value: 'top_n' } })
    expect(screen.getByLabelText('N / 百分比')).toHaveValue(10)
    expect(screen.getByLabelText('N / 百分比')).not.toBeDisabled()
  })

  it('keeps the review table scrollable and supports dynamic numeric sorting', async () => {
    const user = userEvent.setup()
    mocks.listProductPools.mockResolvedValue({ items: [reviewPool], total: 1 })

    render(<ProductPools />)

    const scroll = await screen.findByTestId('candidate-review-scroll')
    expect(scroll).toHaveClass('max-h-[680px]', 'overflow-auto')
    expect(await within(scroll).findByRole('button', { name: /管理人/ })).toBeInTheDocument()
    expect(within(scroll).getByRole('button', { name: /近1年收益率/ })).toBeInTheDocument()

    let rows = within(scroll).getAllByRole('row').slice(1)
    expect(rows[0]).toHaveTextContent('中证500ETF')
    await user.click(within(scroll).getByRole('button', { name: /排名/ }))
    await waitFor(() => {
      rows = within(scroll).getAllByRole('row').slice(1)
      expect(rows[0]).toHaveTextContent('沪深300ETF')
    })

    await user.click(screen.getByText('显示字段（2）'))
    await user.click(screen.getByLabelText('发行规模'))
    await user.click(screen.getByLabelText('近1年夏普比率'))
    expect(await within(scroll).findByRole('button', { name: /发行规模/ })).toBeInTheDocument()
    expect(within(scroll).getByRole('button', { name: /近1年夏普比率/ })).toBeInTheDocument()

    await user.click(within(scroll).getByRole('button', { name: /近1年收益率/ }))
    await waitFor(() => {
      rows = within(scroll).getAllByRole('row').slice(1)
      expect(rows[0]).toHaveTextContent('中证500ETF')
      expect(rows[0]).toHaveTextContent('12%')
    })
  })

  it('applies and atomically saves batch review changes', async () => {
    const user = userEvent.setup()
    const updatedPool = {
      ...reviewPool,
      revision: 5,
      members: reviewPool.members.map((member) => ({
        ...member,
        research_status: 'approved',
        usage_status: 'limited',
        reasons: ['批量复核通过'],
      })),
    }
    mocks.listProductPools.mockResolvedValue({ items: [reviewPool], total: 1 })
    mocks.batchUpdateProductPoolMembers.mockResolvedValue(updatedPool)

    render(<ProductPools />)

    await screen.findByTestId('candidate-review-scroll')
    await user.click(screen.getByLabelText('选择全部当前候选'))
    await user.selectOptions(screen.getByLabelText('批量研究结论'), 'approved')
    await user.selectOptions(screen.getByLabelText('批量使用状态'), 'limited')
    await user.type(screen.getByLabelText('批量复核原因'), '批量复核通过')
    await user.click(screen.getByRole('button', { name: '应用并保存' }))

    await waitFor(() => expect(mocks.batchUpdateProductPoolMembers).toHaveBeenCalledTimes(1))
    const [poolId, request] = mocks.batchUpdateProductPoolMembers.mock.calls[0]
    expect(poolId).toBe('pool-1')
    expect(request.revision).toBe(4)
    expect(request.items).toHaveLength(2)
    expect(request.items).toEqual(expect.arrayContaining([
      expect.objectContaining({ product_id: '510300.SH', research_status: 'approved', usage_status: 'limited', reasons: ['批量复核通过'] }),
      expect.objectContaining({ product_id: '510500.SH', research_status: 'approved', usage_status: 'limited', reasons: ['批量复核通过'] }),
    ]))
  })

  it('批量设置会跳过已经满足目标值的产品，只保存真正发生变化的成员', async () => {
    const user = userEvent.setup()
    const mixedPool = {
      ...reviewPool,
      members: reviewPool.members.map((member, index) => index === 0
        ? { ...member, research_status: 'approved' }
        : member),
    }
    const updatedPool = {
      ...mixedPool,
      revision: 5,
      members: mixedPool.members.map((member) => ({ ...member, research_status: 'approved' })),
    }
    mocks.listProductPools.mockResolvedValue({ items: [mixedPool], total: 1 })
    mocks.batchUpdateProductPoolMembers.mockResolvedValue(updatedPool)

    render(<ProductPools />)

    await screen.findByTestId('candidate-review-scroll')
    await user.click(screen.getByLabelText('选择全部当前候选'))
    await user.selectOptions(screen.getByLabelText('批量研究结论'), 'approved')
    await user.click(screen.getByRole('button', { name: '应用并保存' }))

    await waitFor(() => expect(mocks.batchUpdateProductPoolMembers).toHaveBeenCalledTimes(1))
    const [, request] = mocks.batchUpdateProductPoolMembers.mock.calls[0]
    expect(request.items).toHaveLength(1)
    expect(request.items[0]).toEqual(expect.objectContaining({
      product_id: '510500.SH',
      research_status: 'approved',
    }))
  })

  it('批量字段全部为已有值时视为幂等操作，不发无意义保存请求', async () => {
    const user = userEvent.setup()
    mocks.listProductPools.mockResolvedValue({ items: [reviewPool], total: 1 })

    render(<ProductPools />)

    await screen.findByTestId('candidate-review-scroll')
    await user.click(screen.getByLabelText('选择全部当前候选'))
    await user.selectOptions(screen.getByLabelText('批量使用状态'), 'normal')
    await user.click(screen.getByRole('button', { name: '应用并保存' }))

    expect(mocks.batchUpdateProductPoolMembers).not.toHaveBeenCalled()
    expect(await screen.findByRole('status')).toHaveTextContent('已选 2 个产品均已满足当前批量设置，无需保存。')
  })

  it('renders each attached plan and removes one association independently', async () => {
    const user = userEvent.setup()
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true)
    const afterEquity = { ...basePool, revision: 2, evaluation_plans: [equityBinding] }
    const afterBond = { ...basePool, revision: 3, evaluation_plans: [equityBinding, bondBinding] }
    const afterRemove = { ...basePool, revision: 4, evaluation_plans: [bondBinding] }
    mocks.listProductPools.mockResolvedValue({ items: [basePool], total: 1 })
    mocks.attachEvaluationPlan
      .mockResolvedValueOnce(afterEquity)
      .mockResolvedValueOnce(afterBond)
    mocks.removeEvaluationPlan.mockResolvedValue(afterRemove)

    render(<ProductPools />)

    const planSelector = await screen.findByLabelText('待关联评价方案')
    await waitFor(() => expect(planSelector).toHaveValue('plan-equity'))
    await user.click(screen.getByRole('button', { name: '运行并关联' }))

    expect(await screen.findByRole('heading', { name: '权益 ETF 评价' })).toBeInTheDocument()
    expect(screen.queryByText('尚未关联评价方案。')).not.toBeInTheDocument()
    await waitFor(() => expect(screen.getByLabelText('待关联评价方案')).toHaveValue('plan-bond'))

    await user.click(screen.getByRole('button', { name: '运行并关联' }))
    expect(await screen.findByRole('heading', { name: '固收基金评价' })).toBeInTheDocument()
    expect(mocks.attachEvaluationPlan).toHaveBeenNthCalledWith(2, 'pool-1', expect.objectContaining({
      revision: 2,
      plan_id: 'plan-bond',
    }))

    await user.click(screen.getByRole('button', { name: '删除关联：权益 ETF 评价' }))
    expect(confirm).toHaveBeenCalled()
    await waitFor(() => expect(screen.queryByRole('heading', { name: '权益 ETF 评价' })).not.toBeInTheDocument())
    expect(screen.getByRole('heading', { name: '固收基金评价' })).toBeInTheDocument()
    expect(mocks.removeEvaluationPlan).toHaveBeenCalledWith('pool-1', 'plan-equity', 3)
  })
})
