import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import IndicatorStudio from './IndicatorStudio'
import type { IndicatorDefinition, IndicatorMeta, IndicatorOperator, IndicatorVariable } from '../services/customIndicators'

type MockGraphOption = { series: Array<{ layout: string; edgeSymbol: string[]; data: Array<{ id: unknown; y: number; name: string }>; links: Array<{ source: unknown; target: unknown; parameter?: string }> }> }

vi.mock('echarts-for-react', () => ({
  default: ({ option, 'aria-label': ariaLabel }: { option: MockGraphOption; 'aria-label'?: string }) => {
    const graph = option.series[0]
    return <div aria-label={ariaLabel} data-testid="dag-chart" data-layout={graph.layout} data-directed={String((graph.data[0]?.y ?? 0) < (graph.data[graph.data.length - 1]?.y ?? 0))} data-arrow={graph.edgeSymbol[1]} data-link-count={graph.links.length} data-node-id-type={typeof graph.data[0]?.id} data-edge-id-type={typeof graph.links[0]?.source} data-edge-parameter={graph.links[0]?.parameter || ''} />
  },
}))

const variables: IndicatorVariable[] = [
  { name: 'returns', label: '普通收益率序列', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{r}', shape: 'series', semantic: '复权净值计算得到的普通收益率。', semantic_role: 'ordinary_return', measure: 'return_decimal', price_basis: 'adjusted_nav', source: '真实复权净值', source_dataset: 'fund_nav', source_field: 'adj_nav', data_basis: '后复权', frequency: '交易日', unit: '小数', domains: ['single_product'], category_id: 'returns', category_label: '收益与变化', availability: 'available' },
  { name: 'adjusted_nav', label: '复权净值', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{p}_{\\mathrm{adj}}', shape: 'series', semantic: '产品在计算窗口内的复权净值。', semantic_role: 'adjusted_nav_level', measure: 'adjusted_nav', price_basis: 'adjusted_nav', source: 'Tushare 本地 Parquet', source_dataset: 'fund_nav', source_field: 'adj_nav', data_basis: '后复权', frequency: '交易日', unit: '净值', domains: ['single_product'], category_id: 'price', category_label: '净值与价格', availability: 'available' },
  { name: 'volume', label: '成交量', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{v}', shape: 'series', semantic_role: 'trading_volume', measure: 'volume', source: 'Tushare ETF 日线', source_dataset: 'candle', source_field: 'vol', unit: '原始单位', domains: ['single_product'], product_kinds: ['etf'], category_id: 'trading', category_label: '成交与流动性' },
  { name: 'risk_free_rate_per_observation', label: '单观察期无风险收益率', value_type: 'scalar', dtype: 'float64', latex: 'r_f', shape: 'scalar', semantic: '按当前观察频率换算的无风险收益率。', source: '指标配置', domains: ['single_product', 'portfolio'], category_id: 'configuration', category_label: '基准与配置' },
  { name: 'asset_returns', label: '多资产普通收益矩阵', value_type: 'matrix<time,asset>', dtype: 'float64', latex: '\\mathbf{R}', shape: 'matrix', domains: ['portfolio'], semantic_role: 'ordinary_return_matrix', category_id: 'portfolio', category_label: '组合上下文' },
  { name: 'asset_weights', label: '资产权重向量', value_type: 'vector<asset>', dtype: 'float64', latex: '\\mathbf{w}', shape: 'vector', domains: ['portfolio'], semantic_role: 'weight', category_id: 'portfolio', category_label: '组合上下文' },
]

const operators: IndicatorOperator[] = [
  { name: 'sequence_std', label: '全元素标准差', signature: 'std(values, ddof) → scalar', latex_template: '\\operatorname{std}', return_type: 'scalar', output_shape: 'scalar', mathematical_essence: '计算数值序列全部元素的标准差。', semantic: '纯数学离散程度归约。', category_id: 'statistics', category_label: '统计归约', version: '2.1.0', cost_estimate: 'O(T)', parameters: [{ name: 'values', label: '输入值', description: '待归约的一维数值序列。', allowed_shapes: ['series'] }, { name: 'ddof', label: '自由度修正', description: '样本标准差通常为 1。', allowed_shapes: ['scalar'], default: 1, optional: true }] },
  { name: 'identity_series', label: '逐元素绝对值', signature: 'abs(values) → same(values)', latex_template: '\\operatorname{abs}', return_type: 'series<time>', output_shape: 'series', mathematical_essence: '逐元素计算绝对值。', semantic: '保持时间轴不变。', category_id: 'transform', category_label: '逐元素变换', parameters: [{ name: 'values', label: '输入值', allowed_shapes: ['series'] }] },
  { name: 'add', label: '逐元素加法', signature: 'A + B → same(A,B)', latex_template: 'A+B', return_type: 'same(first)', output_shape: 'unknown', mathematical_essence: '两个兼容数值输入逐元素相加。', semantic: '只允许标量广播。', category_id: 'arithmetic', category_label: '基础运算', parameters: [{ name: 'lhs', label: '输入 A', allowed_shapes: ['scalar', 'series', 'vector', 'matrix'] }, { name: 'rhs', label: '输入 B', allowed_shapes: ['scalar', 'series', 'vector', 'matrix'] }] },
  { name: 'subtract', label: '逐元素减法', signature: 'A - B → same(A,B)', latex_template: 'A-B', return_type: 'same(first)', output_shape: 'unknown', mathematical_essence: '两个兼容数值输入逐元素相减。', semantic: '只允许标量广播。', category_id: 'arithmetic', category_label: '基础运算', cost_estimate: 'elementwise', parameters: [{ name: 'lhs', label: '输入 A', description: '允许类型：scalar | series<T> | vector<N> | matrix<A,B>。', allowed_shapes: ['scalar', 'series', 'vector', 'matrix'] }, { name: 'rhs', label: '输入 B', description: '允许类型：scalar | series<T> | vector<N> | matrix<A,B>。', allowed_shapes: ['scalar', 'series', 'vector', 'matrix'] }] },
  { name: 'covariance', label: '协方差', signature: 'matrix<T,N> → matrix<N,N> / series<T>, series<T> → scalar', latex_template: '\\operatorname{Cov}', return_type: 'matrix<asset,asset> | scalar', output_shape: 'unknown', domains: ['single_product', 'portfolio'], mathematical_essence: '计算协方差矩阵或两条序列的协方差。', semantic: '按输入签名推导输出。', category_id: 'statistics', category_label: '统计计算', parameters: [{ name: 'asset_returns', label: '多资产收益矩阵', allowed_shapes: ['matrix'] }], parameter_sets: [{ arity: 1, parameters: [{ name: 'asset_returns', label: '多资产收益矩阵', allowed_shapes: ['matrix'] }], output: 'matrix<asset,asset>' }, { arity: 2, parameters: [{ name: 'lhs', label: '输入 A', allowed_shapes: ['series'] }, { name: 'rhs', label: '输入 B', allowed_shapes: ['series'] }], output: 'scalar' }] },
  { name: 'diag', label: '对角构造或提取', signature: 'vector<N> → matrix<N,N> / matrix<N,N> → vector<N>', latex_template: '\\operatorname{diag}', return_type: 'matrix<asset,asset> | vector<asset>', output_shape: 'unknown', domains: ['portfolio'], mathematical_essence: '从向量构造对角矩阵，或提取矩阵对角线。', semantic: '输入可以是资产向量或方阵。', category_id: 'linear_algebra', category_label: '线性代数', parameters: [{ name: 'values', label: '输入值', allowed_shapes: ['vector'] }], parameter_sets: [{ arity: 1, parameters: [{ name: 'values', label: '输入值', allowed_shapes: ['vector'] }], output: 'matrix<asset,asset>' }, { arity: 1, parameters: [{ name: 'values', label: '输入值', allowed_shapes: ['matrix'] }], output: 'vector<asset>' }] },
  { name: 'matvec', label: '矩阵向量乘', signature: 'matrix<time,asset> × vector<asset> → series<time>', latex_template: '\\operatorname{matvec}', return_type: 'series<time>', output_shape: 'series', domains: ['portfolio'], mathematical_essence: '矩阵与资产向量相乘。', semantic: '资产轴必须对齐。', category_id: 'linear_algebra', category_label: '线性代数', parameters: [{ name: 'matrix', label: '矩阵', allowed_shapes: ['matrix'] }, { name: 'vector', label: '向量', allowed_shapes: ['vector'] }] },
]

const meta: IndicatorMeta = {
  engine_version: 'test', workspace_scope: 'shared', limits: {}, variables, operators,
  periods: [{ value: '1M', label: '近 1 月', description: '自然月窗口' }, { value: '1Y', label: '近 1 年', description: '自然年窗口' }],
  templates: [], predefined_calculations: [], indicator_types: [{ id: 'risk', label: '风险型指标' }, { id: 'return', label: '收益型指标' }, { id: 'other', label: '其他指标' }],
}

const builtIn: IndicatorDefinition = {
  id: 'builtin-volatility', revision: 1, source: 'built_in', read_only: true, created_at: '2026-01-01', updated_at: '2026-01-01',
  name: '收益波动率', description: '收益率序列的样本标准差。', expression: '\\operatorname{std}\\left(\\mathbf{r},1\\right)', periods: ['1M'], unit: '%', display_format: 'percent', precision: 2, direction: 'lower_better', annual_risk_free_rate_percent: 1.5,
  display_latex: '\\operatorname{Std}_{\\mathrm{ddof}=1}\\left(\\mathbf{r}\\right)',
  context_kind: 'single_product', dsl_version: '2.0.0', operator_registry_version: '2.0.0', output_contract: 'scalar',
  category_id: 'risk', category_label: '风险型指标',
  indicator_type: 'risk',
}

const portfolioBuiltIn: IndicatorDefinition = {
  ...builtIn,
  id: 'builtin-portfolio-volatility',
  name: '组合波动率',
  description: '锁定组合运行的波动率。',
  expression: '\\operatorname{std}\\left(\\operatorname{matvec}\\left(\\mathbf{R},\\mathbf{w}\\right),1\\right)',
  context_kind: 'portfolio',
}

const reusableBuiltIn: IndicatorDefinition = {
  ...builtIn,
  id: 'builtin-total-return-v2',
  name: '累计收益率',
  description: '窗口内普通收益率逐期复合后的累计收益。',
  expression: 'product(returns + 1) - 1',
  display_latex: '\\prod_t(1+r_t)-1',
  dsl_version: '2.1.0',
  operator_registry_version: '2.1.0',
  variable_registry_version: '2.1.0',
  data_contract_version: 'tushare-eod-v2',
  context_schema_version: 'typed-context-v2',
  indicator_type: 'return',
  category_id: 'return',
  category_label: '收益型指标',
  output_measure: 'return_decimal',
}

function json(body: unknown) {
  return { ok: true, status: 200, json: async () => body }
}

function apiError(message: string) {
  return { ok: false, status: 422, json: async () => ({ detail: { code: 'COMPOSE_FAILED', message } }) }
}

async function renderStudio(initialEntry = '/indicator-studio') {
  await act(async () => {
    render(<MemoryRouter initialEntries={[initialEntry]}><IndicatorStudio /></MemoryRouter>)
    await Promise.resolve()
    await Promise.resolve()
  })
  await screen.findByRole('button', { name: /收益波动率/ })
  await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/custom-indicators/meta', expect.anything()))
}

function setupUser() {
  const user = userEvent.setup()
  return {
    click: async (...args: Parameters<typeof user.click>) => { await act(async () => { await user.click(...args) }) },
    selectOptions: async (...args: Parameters<typeof user.selectOptions>) => { await act(async () => { await user.selectOptions(...args) }) },
    clear: async (...args: Parameters<typeof user.clear>) => { await act(async () => { await user.clear(...args) }) },
    type: async (...args: Parameters<typeof user.type>) => { await act(async () => { await user.type(...args) }) },
    keyboard: async (...args: Parameters<typeof user.keyboard>) => { await act(async () => { await user.keyboard(...args) }) },
  }
}

type TestUser = ReturnType<typeof setupUser>

async function openCatalog(user: TestUser) {
  await user.click(screen.getByRole('button', { name: '浏览公式构建资源' }))
  return screen.findByRole('dialog', { name: '变量、算子与已有指标' })
}

async function chooseCombobox(user: TestUser, name: string, query: string) {
  await user.click(screen.getByRole('combobox', { name }))
  const search = await screen.findByLabelText(`搜索${name}`)
  await user.clear(search)
  await user.type(search, query)
  await user.keyboard('{Enter}')
}

describe('IndicatorStudio', () => {
  let catalog: IndicatorDefinition[]
  let activeMeta: IndicatorMeta
  let composeFailure = false
  let availabilityFailure = false
  let inferAsSeries = false
  let validateAsSeries = false

  beforeEach(() => {
    catalog = [builtIn, reusableBuiltIn, portfolioBuiltIn]
    activeMeta = meta
    composeFailure = false
    availabilityFailure = false
    inferAsSeries = false
    validateAsSeries = false
    vi.stubGlobal('confirm', vi.fn(() => true))
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === '/api/custom-indicators/meta') return json(activeMeta)
      if (url === '/api/custom-indicators') {
        if (init?.method === 'POST') {
          const body = JSON.parse(String(init.body))
          const created = { ...builtIn, ...body, id: 'custom-volatility', source: 'custom', read_only: false, revision: 1 }
          catalog = [...catalog, created]
          return json(created)
        }
        return json({ items: catalog, total: catalog.length })
      }
      if (url === '/api/custom-indicators/validate' && validateAsSeries) return json({
        valid: false,
        diagnostics: [{ code: 'OUTPUT_CONTRACT_MISMATCH', message: '输出契约要求 scalar，实际推断为 series<time>[T]。', expected: 'scalar', actual: 'series<time>[T]' }],
        dependencies: ['returns'],
        python_expression: 'returns',
        dag: null,
      })
      if (url === '/api/custom-indicators/validate') return json({
        valid: true, diagnostics: [], dependencies: ['returns'], python_expression: 'sequence_std(returns, 1)',
        display_latex: '\\operatorname{Std}_{\\mathrm{ddof}=1}\\left(\\mathbf{r}\\right)',
        dag: {
          nodes: [
            { id: 'input', label: 'returns', kind: 'variable', value_type: 'series<time>', shape: 'series', symbolic_shape: ['T'] },
            { id: 'root', label: 'sequence_std', kind: 'call', value_type: 'scalar', shape: 'scalar', symbolic_shape: [] },
          ],
          edges: [{ source: 'input', target: 'root' }],
          roots: { result: 'root' },
        },
      })
      if (url === '/api/custom-indicators/compose') {
        if (composeFailure) return apiError('服务端拒绝展开，请检查参数。')
        const body = JSON.parse(String(init?.body))
        if (body.indicator_id === reusableBuiltIn.id) return json({ expression: reusableBuiltIn.expression, latex: reusableBuiltIn.expression, display_latex: reusableBuiltIn.display_latex, inferred_type: 'scalar', shape: 'scalar', semantic_warnings: [], dependencies: ['returns'], indicator_origin: { indicator_id: reusableBuiltIn.id, indicator_revision: reusableBuiltIn.revision, name: reusableBuiltIn.name, source: reusableBuiltIn.source } })
        if (body.operator_id === 'identity_series') return json({ expression: '\\operatorname{abs}\\left(\\mathbf{r}\\right)', latex: '\\operatorname{abs}\\left(\\mathbf{r}\\right)', display_latex: '\\left\\lvert\\mathbf{r}\\right\\rvert', inferred_type: 'series<time>', shape: 'series', semantic_warnings: [] })
        if (body.operator_id === 'sequence_std') return json({ expression: '\\operatorname{std}\\left(\\mathbf{r},1\\right)', latex: '\\operatorname{std}\\left(\\mathbf{r},1\\right)', display_latex: '\\operatorname{Std}_{\\mathrm{ddof}=1}\\left(\\mathbf{r}\\right)', inferred_type: 'scalar', shape: 'scalar', semantic_warnings: [] })
        return json({ expression: '\\left(\\mathbf{r}+1\\right)', latex: '\\left(\\mathbf{r}+1\\right)', display_latex: '\\left(\\mathbf{r}\\right)+\\left(1\\right)', inferred_type: 'series<time>', shape: 'series', semantic_warnings: [] })
      }
      if (url === '/api/custom-indicators/infer') return json(inferAsSeries
        ? { expression: '\\mathbf{r}', latex: '\\mathbf{r}', display_latex: '\\mathbf{r}', inferred_type: 'series<time>[T]', shape: 'series', semantic_warnings: [] }
        : { expression: '\\operatorname{std}\\left(\\mathbf{r},1\\right)', latex: '\\operatorname{std}\\left(\\mathbf{r},1\\right)', display_latex: '\\operatorname{Std}_{\\mathrm{ddof}=1}\\left(\\mathbf{r}\\right)', inferred_type: 'scalar', shape: 'scalar', semantic_warnings: [{ code: 'SEMANTIC_NOTE', message: '请确认样本自由度约定。' }] })
      if (url === '/api/custom-indicators/variables/availability') {
        if (availabilityFailure) return apiError('真实变量可用性服务暂不可用')
        const body = JSON.parse(String(init?.body))
        const requestedTargets = body.targets || [{ kind: body.kind, product_id: body.product_id }]
        return json({
          targets: requestedTargets.map((target: { kind: string; product_id: string }) => ({
            target: { ...target, name: target.product_id === '510300.SH' ? '沪深300ETF' : target.product_id },
          })),
          period: body.period,
          as_of: body.as_of || null,
          items: body.variable_ids.map((variableId: string) => ({
            variable_id: variableId,
            status: 'available',
            coverage: { coverage_ratio: 0.98 },
            actual_shape: variableId === 'risk_free_rate_per_observation' ? [] : [250],
            reason: null,
            target_statuses: requestedTargets.map((target: { kind: string; product_id: string }) => ({
              target: { ...target, name: target.product_id === '510300.SH' ? '沪深300ETF' : target.product_id },
              status: 'available',
              reason: null,
            })),
            window: { requested_as_of: body.as_of || null, effective_as_of: '2026-08-28', start_date: '2025-08-28', end_date: '2026-08-28', observation_count: 249, data_latest_date: '2026-08-28' },
          })),
        })
      }
      if (url === '/api/portfolio-runs') return json({ items: [{ id: 'run-001', target_name: '稳健组合', target_revision: 7, effective_as_of: '2026-08-28', window: { start_date: '2024-01-02', end_date: '2026-08-28' } }] })
      if (url.startsWith('/api/instruments/search')) {
        const query = new URL(url, 'http://localhost').searchParams.get('q') || ''
        const second = query === '512960.SH'
        const code = second ? '512960.SH' : '510300.SH'
        return json({ items: [{ code, ts_code: code, name: second ? '证券ETF' : '沪深300ETF', management: '测试', found_date: '2026-01-01', instrument_type: 'etf' }], total: 1, page: 1, page_size: 30, kind: 'etf' })
      }
      if (url === '/api/custom-indicators/evaluate') {
        const body = JSON.parse(String(init?.body))
        const targets = body.targets as Array<{ kind: 'etf' | 'fund'; product_id: string }>
        return json({
          results: targets.map((target) => ({ indicator_id: null, indicator_revision: null, indicator_name: '收益波动率', target: { ...target, name: target.product_id === '510300.SH' ? '沪深300ETF' : target.product_id }, period: body.period, value: 0.1234, status: 'ok', warnings: [], window: { requested_as_of: body.as_of || null, effective_as_of: '2026-08-28', start_date: '2025-08-28', end_date: '2026-08-28', observation_count: 250, data_latest_date: '2026-08-28' } })),
          summary: { total: targets.length, ok: targets.length, warning: 0, error: 0 },
          cache: { hits: 0, misses: targets.length },
        })
      }
      if (url === '/api/custom-indicators/evaluate-portfolio') return json({ results: [{ indicator_id: null, indicator_revision: null, indicator_name: '组合波动率', target: { kind: 'portfolio', product_id: 'run-001', name: '稳健组合' }, period: 'snapshot', value: 0.087, status: 'ok', warnings: [], window: { requested_as_of: null, effective_as_of: '2026-08-28', start_date: '2024-01-02', end_date: '2026-08-28', observation_count: 640, data_latest_date: '2026-08-28' } }], summary: { total: 1, ok: 1, warning: 0, error: 0 }, cache: { hits: 0, misses: 1 } })
      return json({})
    }))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('定义时不选择周期，保存不提交 periods，单产品预览仍显式选择运行周期', async () => {
    const user = setupUser()
    await renderStudio()

    expect(screen.queryByText('可用周期')).not.toBeInTheDocument()
    expect(screen.getByText(/指标定义默认支持全部计算周期/)).toBeInTheDocument()
    expect(screen.getByLabelText('计算周期')).toHaveTextContent('近 1 月')
    expect(screen.getByLabelText('计算周期')).toHaveTextContent('近 1 年')

    await user.click(screen.getByRole('button', { name: /收益波动率/ }))
    await user.click(screen.getByRole('button', { name: '保存到工作区' }))
    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/custom-indicators', expect.objectContaining({ method: 'POST' })))
    const createCall = vi.mocked(fetch).mock.calls.find(([url, init]) => url === '/api/custom-indicators' && init?.method === 'POST')
    const validateCall = vi.mocked(fetch).mock.calls.find(([url]) => url === '/api/custom-indicators/validate')
    expect(JSON.parse(String((createCall?.[1] as RequestInit).body))).not.toHaveProperty('periods')
    expect(JSON.parse(String((validateCall?.[1] as RequestInit).body))).not.toHaveProperty('periods')
  })

  it('校验与预览保留深链中的多个产品，并在一次请求中批量计算', async () => {
    const user = setupUser()
    await renderStudio('/indicator-studio?kind=etf&ids=510300.SH%2C512960.SH')

    expect(await screen.findByText('2 / 10')).toBeInTheDocument()
    expect(screen.queryByText(/仅保留第一个已选产品/)).not.toBeInTheDocument()
    await waitFor(() => {
      const calls = vi.mocked(fetch).mock.calls.filter(([url]) => url === '/api/custom-indicators/variables/availability')
      const body = JSON.parse(String((calls[calls.length - 1]?.[1] as RequestInit).body))
      expect(body.targets).toEqual([
        { kind: 'etf', product_id: '510300.SH' },
        { kind: 'etf', product_id: '512960.SH' },
      ])
    })
    await user.click(screen.getByRole('button', { name: /收益波动率/ }))
    await user.click(screen.getByRole('button', { name: '预览指标' }))

    await waitFor(() => {
      const call = vi.mocked(fetch).mock.calls.find(([url]) => url === '/api/custom-indicators/evaluate')
      expect(call).toBeDefined()
      expect(JSON.parse(String((call?.[1] as RequestInit).body)).targets).toEqual([
        { kind: 'etf', product_id: '510300.SH' },
        { kind: 'etf', product_id: '512960.SH' },
      ])
      expect(JSON.parse(String((call?.[1] as RequestInit).body)).include_series).toBe(false)
    })
    expect(await screen.findByText('预览完成：2 个成功，0 个需关注。')).toBeInTheDocument()
    expect(screen.getAllByText('512960.SH').length).toBeGreaterThan(0)
  })

  it('滚动曲线必须显式启用，避免默认执行 500 次窗口计算', async () => {
    const user = setupUser()
    await renderStudio('/indicator-studio?kind=etf&ids=510300.SH')
    await user.click(screen.getByRole('button', { name: /收益波动率/ }))
    await user.click(screen.getByRole('checkbox', { name: /同时计算滚动曲线/ }))
    await user.click(screen.getByRole('button', { name: '预览指标' }))

    await waitFor(() => {
      const calls = vi.mocked(fetch).mock.calls.filter(([url]) => url === '/api/custom-indicators/evaluate')
      const body = JSON.parse(String((calls[calls.length - 1]?.[1] as RequestInit).body))
      expect(body.include_series).toBe(true)
    })
  })

  it('搜索添加产品时追加选择而不是替换已有产品', async () => {
    const user = setupUser()
    await renderStudio()

    await user.click(screen.getByRole('button', { name: '搜索' }))
    await user.click(await screen.findByRole('button', { name: '添加' }))
    await user.type(screen.getByLabelText('搜索产品'), '512960.SH')
    await user.click(screen.getByRole('button', { name: '搜索' }))
    await user.click(await screen.findByRole('button', { name: '添加' }))

    expect(await screen.findByText('2 / 10')).toBeInTheDocument()
    expect(screen.getByText('沪深300ETF')).toBeInTheDocument()
    expect(screen.getAllByText('证券ETF').length).toBeGreaterThan(0)
  })

  it('深链产品超过十个时只保留前十个', async () => {
    const ids = Array.from({ length: 12 }, (_, index) => `TEST${String(index + 1).padStart(2, '0')}.SH`)
    await renderStudio(`/indicator-studio?kind=etf&ids=${encodeURIComponent(ids.join(','))}`)

    expect(await screen.findByText('10 / 10')).toBeInTheDocument()
    expect(screen.getByText('校验与预览最多选择 10 个产品，已保留前 10 个。')).toBeInTheDocument()
    expect(screen.queryByText('TEST11.SH')).not.toBeInTheDocument()
    expect(screen.queryByText('TEST12.SH')).not.toBeInTheDocument()
  })

  it('指标库支持按关键词、来源和分类筛选', async () => {
    const user = setupUser()
    await renderStudio()

    await user.type(screen.getByLabelText('搜索指标'), '不存在的指标')
    expect(screen.queryByRole('button', { name: /收益波动率/ })).not.toBeInTheDocument()
    await user.clear(screen.getByLabelText('搜索指标'))
    await user.selectOptions(screen.getByLabelText('指标分类'), 'risk')
    expect(screen.getByRole('button', { name: /收益波动率/ })).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('指标来源'), 'custom')
    expect(screen.getByText('当前筛选条件下没有指标。')).toBeInTheDocument()
  })

  it('大型目录通过资源类型、分类和条目三级选择，并支持键盘搜索', async () => {
    const extraVariables: IndicatorVariable[] = Array.from({ length: 240 }, (_, index) => ({
      name: `market_field_${index}`,
      label: `成交变量 ${index}`,
      value_type: 'series<time>',
      dtype: 'float64',
      latex: `x_{${index}}`,
      shape: 'series',
      semantic: `第 ${index} 个原始成交数据字段。`,
      source: 'Tushare 日线数据',
      category_id: index % 2 ? 'trading' : 'price',
      category_label: index % 2 ? '成交与流动性' : '净值与价格',
      domains: ['single_product'],
    }))
    activeMeta = { ...meta, variables: [...variables, ...extraVariables] }
    const user = setupUser()
    await renderStudio()
    const dialog = await openCatalog(user)

    expect(within(dialog).getByLabelText('资源类型')).toHaveValue('variables')
    await chooseCombobox(user, '资源分类', '成交与流动性')
    await chooseCombobox(user, '选择变量', '成交变量 217')
    expect(within(dialog).getByRole('heading', { name: '成交变量 217' })).toBeInTheDocument()
    expect(within(dialog).getByText(/Tushare 日线数据/)).toBeInTheDocument()
    expect(within(dialog).getAllByText('时间序列').length).toBeGreaterThan(0)
    expect(within(dialog).queryByText(/series<time>|\[T\]/)).not.toBeInTheDocument()
  })

  it('类型推导与输出契约错误只展示中文数据类型', async () => {
    inferAsSeries = true
    validateAsSeries = true
    const user = setupUser()
    await renderStudio()

    await user.click(screen.getByRole('button', { name: /收益波动率/ }))
    await user.click(screen.getByRole('button', { name: '推导类型' }))
    expect(await screen.findByText('时间序列')).toBeInTheDocument()
    expect(screen.queryByText(/series<time>|\[T\]/)).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '校验公式' }))
    expect(await screen.findByText('校验未通过')).toBeInTheDocument()
    expect(screen.getByText(/指标结果类型不符合要求/)).toBeInTheDocument()
    expect(screen.getByText(/指标最终结果必须是单个有限数值/)).toBeInTheDocument()
    expect(screen.getByText(/需要 有限标量/)).toBeInTheDocument()
    expect(screen.getByText(/当前是 时间序列/)).toBeInTheDocument()
    expect(screen.queryByText(/OUTPUT_CONTRACT_MISMATCH|series<time>|\[T\]|输出契约要求 scalar/)).not.toBeInTheDocument()
  })

  it('算子目录与参数配置只展示中文数据要求', async () => {
    const user = setupUser()
    await renderStudio()
    const catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await chooseCombobox(user, '资源分类', '基础运算')
    await chooseCombobox(user, '选择计算算子', '逐元素减法')

    expect(within(catalogDialog).getByLabelText('算子输入输出说明')).toHaveTextContent('输入 A支持有限标量、时间序列、资产向量、矩阵')
    expect(within(catalogDialog).getByText('逐元素计算')).toBeInTheDocument()
    expect(within(catalogDialog).queryByText(/same\(|scalar|series<|vector<|matrix<|elementwise/)).not.toBeInTheDocument()

    await user.click(within(catalogDialog).getByRole('button', { name: '配置算子参数' }))
    const composer = await screen.findByRole('dialog', { name: '配置 逐元素减法' })
    expect(within(composer).getByText(/输入 A支持有限标量/)).toBeInTheDocument()
    expect(within(composer).queryByText(/same\(|scalar|series<|vector<|matrix<|elementwise/)).not.toBeInTheDocument()
  })

  it('目录包含变量、计算算子和已有指标，不出现预定义计算或公式片段', async () => {
    const user = setupUser()
    await renderStudio()
    const dialog = await openCatalog(user)
    const resourceType = within(dialog).getByLabelText('资源类型')
    expect(resourceType).toHaveTextContent('变量')
    expect(resourceType).toHaveTextContent('计算算子')
    expect(resourceType).toHaveTextContent('已有指标')
    expect(resourceType).not.toHaveTextContent('预定义计算')
    expect(resourceType).not.toHaveTextContent('公式片段')
    expect(within(dialog).queryByText('公式片段')).not.toBeInTheDocument()
  })

  it('已有指标按锁定版本展开为独立公式', async () => {
    const user = setupUser()
    await renderStudio()
    const dialog = await openCatalog(user)
    await user.selectOptions(within(dialog).getByLabelText('资源类型'), 'indicators')
    await chooseCombobox(user, '资源分类', '收益型指标')
    await chooseCombobox(user, '选择已有指标', '累计收益率')

    expect(within(dialog).getByText('内置指标 · 锁定 v1')).toBeInTheDocument()
    expect(within(dialog).getByText(/原指标以后更新不会改变本草稿/)).toBeInTheDocument()
    await user.click(within(dialog).getByRole('button', { name: '展开为当前公式' }))

    await waitFor(() => {
      const call = vi.mocked(fetch).mock.calls.find(([url, init]) => {
        if (url !== '/api/custom-indicators/compose') return false
        const body = JSON.parse(String((init as RequestInit).body))
        return body.indicator_id === reusableBuiltIn.id
      })
      expect(call).toBeDefined()
      expect(JSON.parse(String((call?.[1] as RequestInit).body))).toMatchObject({
        indicator_id: reusableBuiltIn.id,
        indicator_revision: 1,
        context: 'single_product',
        dsl_version: '2.2.0',
        operator_registry_version: '2.2.0',
      })
    })
    expect(screen.getByText(/“累计收益率” v1 已按锁定版本展开为当前公式/)).toBeInTheDocument()
    expect(screen.getByText(/公式已由结构化构建器生成/)).toBeInTheDocument()
    expect(screen.queryByText(reusableBuiltIn.expression)).not.toBeInTheDocument()
    expect(screen.getByTestId('formula-preview').querySelector('.katex')).not.toBeNull()
  })

  it('公式源码与数学排版分离，预览不把下划线标识符渲染成下标', async () => {
    const payoffIndicator: IndicatorDefinition = {
      ...builtIn,
      id: 'builtin-payoff-ratio',
      name: '平均盈亏比',
      expression: 'mean_where(returns, greater_than(returns, 0)) / absolute(mean_where(returns, less_than(returns, 0)))',
      display_latex: '\\frac{\\mathbb{E}[\\mathbf{r}\\mid\\mathbf{r}>0]}{\\left\\lvert\\mathbb{E}[\\mathbf{r}\\mid\\mathbf{r}<0]\\right\\rvert}',
    }
    catalog = [builtIn, payoffIndicator, portfolioBuiltIn]
    const user = setupUser()
    await renderStudio()

    await user.click(screen.getByRole('button', { name: /平均盈亏比/ }))
    await user.click(screen.getByRole('tab', { name: '高级公式模式' }))
    expect(screen.getByLabelText('受限公式源码（DSL / LaTeX）')).toHaveValue(payoffIndicator.expression)
    const preview = screen.getByTestId('formula-preview')
    expect(preview.innerHTML).not.toContain('mean_where')
    expect(preview.innerHTML).not.toContain('greater_than')

    await user.type(screen.getByLabelText('受限公式源码（DSL / LaTeX）'), '+1')
    expect(screen.getByTestId('formula-preview')).toHaveTextContent('请先推导或校验公式以生成数学排版')
    expect(screen.getByTestId('formula-preview').innerHTML).not.toContain('mean_where')
  })

  it('非法数学排版不会以红色 LaTeX 源码泄露到预览区', async () => {
    const invalidNotation = {
      ...builtIn,
      display_latex: String.raw`\left(\mathbf{r}\right)_t_i`,
    }
    catalog = [invalidNotation, reusableBuiltIn, portfolioBuiltIn]
    const user = setupUser()
    await renderStudio()

    await user.click(screen.getByRole('button', { name: /收益波动率/ }))
    const preview = screen.getByTestId('formula-preview')
    expect(preview).toHaveTextContent('公式排版不可用')
    expect(preview.querySelector('.katex-error')).toBeNull()
  })

  it('算子参数仅提供兼容变量、有限常量和嵌套算子，并明确应用方式', async () => {
    const user = setupUser()
    await renderStudio()
    const catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await chooseCombobox(user, '资源分类', '统计归约')
    await chooseCombobox(user, '选择计算算子', '全元素标准差')
    await user.click(within(catalogDialog).getByRole('button', { name: '配置算子参数' }))

    const composer = await screen.findByRole('dialog', { name: '配置 全元素标准差' })
    const source = within(composer).getByLabelText('输入值 输入来源')
    expect(source).toHaveTextContent('兼容变量')
    expect(source).toHaveTextContent('有限数值常量')
    expect(source).toHaveTextContent('嵌套算子')
    expect(source).toHaveTextContent('已有标量指标')
    expect(source).not.toHaveTextContent('当前公式')
    expect(within(composer).getByLabelText('应用方式')).toHaveValue('replace_formula')

    await user.selectOptions(source, 'operator')
    expect(within(composer).getByLabelText('输入值 嵌套算子')).toHaveValue('identity_series')
    await user.click(within(composer).getByRole('button', { name: '展开到公式' }))

    await waitFor(() => expect(vi.mocked(fetch).mock.calls.filter(([url]) => url === '/api/custom-indicators/compose')).toHaveLength(2))
    const composeCalls = vi.mocked(fetch).mock.calls.filter(([url]) => url === '/api/custom-indicators/compose')
    const parentBody = JSON.parse(String((composeCalls[1][1] as RequestInit).body))
    expect(parentBody.arguments[0]).toMatchObject({ parameter: 'values', source: 'expression' })
    expect(await screen.findByText('类型推导')).toBeInTheDocument()
  })

  it('引导构建允许分别切换输入，并在参数组合层校验类型与语义量纲', async () => {
    const user = setupUser()
    await renderStudio()
    const catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await chooseCombobox(user, '资源分类', '基础运算')
    await chooseCombobox(user, '选择计算算子', '逐元素加法')
    await user.click(within(catalogDialog).getByRole('button', { name: '配置算子参数' }))

    const composer = await screen.findByRole('dialog', { name: '配置 逐元素加法' })
    const firstInput = within(composer).getByLabelText('输入 A 变量')
    const secondInput = within(composer).getByLabelText('输入 B 变量')
    expect(within(secondInput).getByRole('option', { name: /普通收益率序列/ })).not.toBeDisabled()
    expect(within(secondInput).getByRole('option', { name: /成交量/ })).not.toBeDisabled()

    await user.selectOptions(firstInput, 'volume')
    expect(within(composer).getByText(/输入 A 与 输入 B.*不兼容/)).toBeInTheDocument()
    expect(within(composer).getByRole('button', { name: '展开到公式' })).toBeDisabled()

    await user.selectOptions(secondInput, 'volume')
    expect(within(composer).queryByText(/输入 A 与 输入 B.*不兼容/)).not.toBeInTheDocument()

    expect(within(composer).getByRole('button', { name: '展开到公式' })).toBeEnabled()
    await user.click(within(composer).getByRole('button', { name: '展开到公式' }))
    const composeCall = vi.mocked(fetch).mock.calls.find(([url], index) => (
      url === '/api/custom-indicators/compose'
      && index >= 0
    ))
    const composeBody = JSON.parse(String((composeCall?.[1] as RequestInit).body))
    expect(composeBody.arguments).toEqual([
      { parameter: 'lhs', source: 'variable', value: 'volume' },
      { parameter: 'rhs', source: 'variable', value: 'volume' },
    ])
    expect(screen.queryByText(/缺少参数/)).not.toBeInTheDocument()
  })

  it('算子参数可以复用已有标量指标并在提交前展开', async () => {
    const user = setupUser()
    await renderStudio()
    const catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await chooseCombobox(user, '资源分类', '基础运算')
    await chooseCombobox(user, '选择计算算子', '逐元素加法')
    await user.click(within(catalogDialog).getByRole('button', { name: '配置算子参数' }))

    const composer = await screen.findByRole('dialog', { name: '配置 逐元素加法' })
    await user.selectOptions(within(composer).getByLabelText('输入 A 输入来源'), 'indicator')
    await user.selectOptions(within(composer).getByLabelText('输入 B 输入来源'), 'indicator')
    expect(within(composer).getByLabelText('输入 A 已有指标')).toHaveValue(`${reusableBuiltIn.id}@1`)
    expect(within(composer).getByRole('button', { name: '展开到公式' })).toBeEnabled()
    await user.click(within(composer).getByRole('button', { name: '展开到公式' }))

    await waitFor(() => {
      const calls = vi.mocked(fetch).mock.calls.filter(([url]) => url === '/api/custom-indicators/compose')
      expect(calls).toHaveLength(3)
      const parent = JSON.parse(String((calls[2][1] as RequestInit).body))
      expect(parent.arguments).toEqual([
        { parameter: 'lhs', source: 'expression', value: reusableBuiltIn.expression },
        { parameter: 'rhs', source: 'expression', value: reusableBuiltIn.expression },
      ])
    })
  })

  it('逐元素减法允许先选择净值，再完成同口径配对', async () => {
    const user = setupUser()
    await renderStudio()
    const catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await chooseCombobox(user, '资源分类', '基础运算')
    await chooseCombobox(user, '选择计算算子', '逐元素减法')
    await user.click(within(catalogDialog).getByRole('button', { name: '配置算子参数' }))

    const composer = await screen.findByRole('dialog', { name: '配置 逐元素减法' })
    const firstInput = within(composer).getByLabelText('输入 A 变量')
    const secondInput = within(composer).getByLabelText('输入 B 变量')
    expect(within(firstInput).getByRole('option', { name: /复权净值/ })).not.toBeDisabled()
    expect(within(secondInput).getByRole('option', { name: /复权净值/ })).not.toBeDisabled()

    await user.selectOptions(firstInput, 'adjusted_nav')
    expect(within(composer).getByRole('button', { name: '展开到公式' })).toBeDisabled()
    expect(within(composer).getByText(/输入 A 与 输入 B.*不兼容/)).toBeInTheDocument()

    await user.selectOptions(secondInput, 'adjusted_nav')
    expect(within(composer).getByRole('button', { name: '展开到公式' })).toBeEnabled()
  })

  it('重载算子按计算域选择可用签名，并合并同名同元数签名', async () => {
    const user = setupUser()
    await renderStudio()
    let catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await chooseCombobox(user, '资源分类', '统计计算')
    await chooseCombobox(user, '选择计算算子', '协方差')
    await user.click(within(catalogDialog).getByRole('button', { name: '配置算子参数' }))

    let composer = await screen.findByRole('dialog', { name: '配置 协方差' })
    expect(within(composer).getByLabelText('输入 A 变量')).toHaveValue('returns')
    expect(within(composer).getByLabelText('输入 B 变量')).toHaveValue('returns')
    expect(within(composer).queryByLabelText('多资产收益矩阵 变量')).not.toBeInTheDocument()
    await user.click(within(composer).getByRole('button', { name: '取消' }))

    await user.click(screen.getByLabelText('组合运行域'))
    catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await chooseCombobox(user, '资源分类', '统计计算')
    await chooseCombobox(user, '选择计算算子', '协方差')
    await user.click(within(catalogDialog).getByRole('button', { name: '配置算子参数' }))
    composer = await screen.findByRole('dialog', { name: '配置 协方差' })
    expect(within(composer).getByLabelText('多资产收益矩阵 变量')).toHaveValue('asset_returns')
    await user.click(within(composer).getByRole('button', { name: '取消' }))

    catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await chooseCombobox(user, '资源分类', '线性代数')
    await chooseCombobox(user, '选择计算算子', '对角构造或提取')
    await user.click(within(catalogDialog).getByRole('button', { name: '配置算子参数' }))
    composer = await screen.findByRole('dialog', { name: '配置 对角构造或提取' })
    const values = within(composer).getByLabelText('输入值 变量')
    expect(within(values).getByRole('option', { name: /多资产普通收益矩阵/ })).not.toBeDisabled()
    expect(within(values).getByRole('option', { name: /资产权重向量/ })).not.toBeDisabled()
  })

  it('公式编辑会清除旧推导、校验、DAG 与运行结果，并提示重新校验', async () => {
    const user = setupUser()
    await renderStudio()
    await user.click(screen.getByRole('button', { name: /收益波动率/ }))
    await user.click(screen.getByRole('button', { name: '推导类型' }))
    await user.click(screen.getByRole('button', { name: '校验公式' }))
    await user.click(screen.getByRole('button', { name: '搜索' }))
    await user.click(await screen.findByRole('button', { name: '添加' }))
    await user.click(screen.getByRole('button', { name: '预览指标' }))
    expect((await screen.findAllByText('12.34%')).length).toBeGreaterThan(0)
    expect(screen.getByText('类型推导')).toBeInTheDocument()
    expect(screen.getByText('校验通过')).toBeInTheDocument()
    expect(screen.getByTestId('dag-chart')).toBeInTheDocument()

    await user.click(screen.getByRole('tab', { name: '高级公式模式' }))
    await user.type(screen.getByLabelText('受限公式源码（DSL / LaTeX）'), '+1')

    expect(screen.queryByText('类型推导')).not.toBeInTheDocument()
    expect(screen.queryByText('校验通过')).not.toBeInTheDocument()
    expect(screen.queryByTestId('dag-chart')).not.toBeInTheDocument()
    expect(screen.queryByText('12.34%')).not.toBeInTheDocument()
    expect(screen.getByText('公式已更改，需要重新推导并校验。')).toBeInTheDocument()
  })

  it('DAG 始终使用单一 result 根，运行周期只更新展示上下文', async () => {
    const user = setupUser()
    await renderStudio()
    await user.click(screen.getByRole('button', { name: /收益波动率/ }))
    await user.click(screen.getByRole('button', { name: '校验公式' }))

    expect(await screen.findByText('校验通过')).toBeInTheDocument()
    expect(screen.getByText('结构化表达式已同步')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '编辑解析后的结构' }))
    const parsedComposer = await screen.findByRole('dialog', { name: '配置 全元素标准差' })
    expect(within(parsedComposer).getByLabelText('自由度修正 有限常量')).toHaveValue(1)
    await user.click(within(parsedComposer).getByRole('button', { name: '取消' }))
    const chart = screen.getByTestId('dag-chart')
    expect(chart).toHaveAttribute('data-layout', 'none')
    expect(chart).toHaveAttribute('data-directed', 'true')
    expect(chart).toHaveAttribute('data-arrow', 'arrow')
    expect(chart).toHaveAttribute('data-link-count', '1')
    expect(chart).toHaveAttribute('data-edge-parameter', '输入值')
    expect(screen.getByText('预览周期：1Y')).toBeInTheDocument()
    const table = screen.getByRole('table', { name: 'DAG 节点、输入参数和输出类型数据表' })
    expect(table).toHaveTextContent('输入值 ← 普通收益率序列')
    expect(table).toHaveTextContent('全元素标准差')
    expect(screen.getByRole('button', { name: '适配并复位图谱' })).toBeInTheDocument()
    const nodeDetails = screen.getByRole('article', { name: 'DAG 节点详情' })
    expect(nodeDetails).toHaveTextContent('全元素标准差')
    expect(nodeDetails).toHaveTextContent('理论规模：单个数值')
    expect(table).toHaveTextContent('理论规模：随计算窗口变化的时间点数量')
    await user.click(within(table).getByRole('button', { name: '普通收益率序列' }))
    expect(nodeDetails).toHaveTextContent('普通收益率序列')

    await user.selectOptions(screen.getByLabelText('计算周期'), '1M')
    expect(screen.getByText('预览周期：1M')).toBeInTheDocument()
    expect(screen.getByTestId('dag-chart')).toHaveAttribute('data-link-count', '1')
  })

  it('组合预览只使用不可变快照窗口，不显示或发送运行周期', async () => {
    const user = setupUser()
    await renderStudio()
    await user.click(screen.getByLabelText('组合运行域'))
    await user.click(await screen.findByRole('button', { name: /组合波动率/ }))
    expect(await screen.findByRole('option', { name: '稳健组合 · v7 · 截止 2026-08-28' })).toBeInTheDocument()
    expect(screen.queryByLabelText('计算周期')).not.toBeInTheDocument()
    expect(screen.getByText('快照窗口')).toBeInTheDocument()
    expect(screen.getByText('2024-01-02 至 2026-08-28')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '按快照窗口预览' }))
    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/custom-indicators/evaluate-portfolio', expect.objectContaining({ method: 'POST' })))
    const evaluationCall = vi.mocked(fetch).mock.calls.find(([url]) => url === '/api/custom-indicators/evaluate-portfolio')
    const body = JSON.parse(String((evaluationCall?.[1] as RequestInit).body))
    expect(body).not.toHaveProperty('period')
    expect(body.inline_definition).not.toHaveProperty('periods')
    expect(screen.getByText('快照窗口 · 结构视图')).toBeInTheDocument()
  })

  it('按产品、周期和历史截止日展示变量实际可用性，并在预览时传递 as-of', async () => {
    const user = setupUser()
    await renderStudio()
    await user.click(screen.getByRole('button', { name: /收益波动率/ }))
    await user.click(screen.getByRole('button', { name: '搜索' }))
    await user.click(await screen.findByRole('button', { name: '添加' }))
    await user.type(screen.getByLabelText('历史截止日'), '2026-08-28')

    await waitFor(() => {
      const calls = vi.mocked(fetch).mock.calls.filter(([url]) => url === '/api/custom-indicators/variables/availability')
      expect(calls.length).toBeGreaterThan(0)
      const body = JSON.parse(String((calls[calls.length - 1]?.[1] as RequestInit).body))
      expect(body).toMatchObject({ targets: [{ kind: 'etf', product_id: '510300.SH' }], period: '1Y', as_of: '2026-08-28' })
    })

    const dialog = await openCatalog(user)
    await chooseCombobox(user, '资源分类', '收益与变化')
    await chooseCombobox(user, '选择变量', '普通收益率序列')
    expect(within(dialog).getByText('250 个时间点 · 98.0%')).toBeInTheDocument()
    await user.click(within(dialog).getByRole('button', { name: '设为当前公式' }))
    await user.click(screen.getByRole('button', { name: '预览指标' }))
    await waitFor(() => {
      const call = vi.mocked(fetch).mock.calls.find(([url]) => url === '/api/custom-indicators/evaluate')
      expect(JSON.parse(String((call?.[1] as RequestInit).body)).as_of).toBe('2026-08-28')
    })
  })

  it('变量可用性核对失败时关闭入口，算子参数也不会继续提供未知变量', async () => {
    availabilityFailure = true
    const user = setupUser()
    await renderStudio()
    await user.click(screen.getByRole('button', { name: '搜索' }))
    await user.click(await screen.findByRole('button', { name: '添加' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('变量可用性核对失败')
    const catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await chooseCombobox(user, '资源分类', '统计归约')
    await chooseCombobox(user, '选择计算算子', '全元素标准差')
    await user.click(within(catalogDialog).getByRole('button', { name: '配置算子参数' }))

    const composer = await screen.findByRole('dialog', { name: '配置 全元素标准差' })
    expect(within(composer).getByLabelText('输入值 输入来源')).toHaveTextContent('兼容变量')
    expect(within(composer).getByText(/当前计算域没有满足该参数类型与语义约束的变量/)).toBeInTheDocument()
    expect(within(composer).getByRole('button', { name: '展开到公式' })).toBeDisabled()
  })

  it('资源分类只展示可用数量，并隐藏当前域不可用的分类', async () => {
    const user = setupUser()
    await renderStudio()
    const dialog = await openCatalog(user)
    await user.selectOptions(within(dialog).getByLabelText('资源类型'), 'operators')
    await user.click(within(dialog).getByRole('combobox', { name: '资源分类' }))

    const options = within(dialog).getAllByRole('option')
    expect(options.some((option) => option.textContent?.includes('/'))).toBe(false)
    expect(within(dialog).getByRole('option', { name: '基础运算（2）' })).toBeInTheDocument()
    expect(within(dialog).queryByRole('option', { name: /线性代数/ })).not.toBeInTheDocument()
  })

  it('meta 明确标记 unavailable 的变量在未选择产品时不展示对应分类', async () => {
    activeMeta = {
      ...meta,
      variables: variables.map((variable) => variable.name === 'volume'
        ? { ...variable, availability: 'unavailable' }
        : variable),
    }
    const user = setupUser()
    await renderStudio()
    const dialog = await openCatalog(user)
    await user.click(within(dialog).getByRole('combobox', { name: '资源分类' }))
    await user.type(await within(dialog).findByLabelText('搜索资源分类'), '成交与流动性')

    expect(within(dialog).queryByRole('option', { name: /成交与流动性/ })).not.toBeInTheDocument()
  })

  it('移动端三区 Tab 支持方向键、Home/End 与正确的 aria 关联', async () => {
    await renderStudio()
    const libraryTab = screen.getByRole('tab', { name: '指标库' })
    const editorTab = screen.getByRole('tab', { name: '编辑' })
    const previewTab = screen.getByRole('tab', { name: '预览' })
    expect(libraryTab).toHaveAttribute('aria-controls', 'indicator-panel-library')
    expect(editorTab).toHaveAttribute('aria-controls', 'indicator-panel-editor')
    expect(previewTab).toHaveAttribute('aria-controls', 'indicator-panel-preview')

    fireEvent.keyDown(previewTab, { key: 'Enter' })
    expect(previewTab).toHaveAttribute('aria-selected', 'true')
    fireEvent.keyDown(previewTab, { key: 'ArrowLeft' })
    expect(editorTab).toHaveAttribute('aria-selected', 'true')
    fireEvent.keyDown(editorTab, { key: 'Home' })
    expect(libraryTab).toHaveAttribute('aria-selected', 'true')
    fireEvent.keyDown(libraryTab, { key: 'End' })
    expect(previewTab).toHaveAttribute('aria-selected', 'true')
  })

  it('算子展开失败会在抽屉内给出明确错误并允许重试', async () => {
    composeFailure = true
    const user = setupUser()
    await renderStudio()
    const catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await chooseCombobox(user, '资源分类', '统计归约')
    await chooseCombobox(user, '选择计算算子', '全元素标准差')
    await user.click(within(catalogDialog).getByRole('button', { name: '配置算子参数' }))
    await user.click(screen.getByRole('button', { name: '展开到公式' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('服务端拒绝展开，请检查参数。')
    expect(screen.getByRole('dialog', { name: '配置 全元素标准差' })).toBeInTheDocument()
  })
})
