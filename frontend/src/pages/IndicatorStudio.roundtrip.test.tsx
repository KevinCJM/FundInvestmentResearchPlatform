import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import IndicatorStudio from './IndicatorStudio'
import type { IndicatorDag, IndicatorDefinition, IndicatorMeta, IndicatorOperator, IndicatorOperatorParameter } from '../services/customIndicators'

vi.mock('echarts-for-react', () => ({ default: () => <div /> }))
afterEach(() => { vi.unstubAllGlobals() })

const parameter = (name: string, label: string, scalar = false): IndicatorOperatorParameter => ({
  name, label, allowed_shapes: scalar ? ['scalar'] : ['series', 'scalar'],
  ...(scalar ? { source_policy: 'fixed_constant', constant_kind: 'integer' } : {}),
})
const values = parameter('values', '输入序列')
const windowParameter = parameter('window', '窗口期数', true)
const ddof = parameter('ddof', '自由度修正', true)
const minimum = parameter('min_periods', '最少有效观察数', true)
const pair = [parameter('lhs', '输入 A'), parameter('rhs', '输入 B')]

function sharpeLatex(rolling: boolean, window = 5, explicitMinimum = false) {
  const extra = explicitMinimum ? `,${window}` : ''
  const mean = rolling ? String.raw`\operatorname{rolling_mean}\left(\mathbf{r},${window}${extra}\right)` : String.raw`\operatorname{mean}\left(\mathbf{r}\right)`
  const std = rolling ? String.raw`\operatorname{rolling_std}\left(\mathbf{r},${window},1${extra}\right)` : String.raw`\operatorname{std}\left(\mathbf{r},1\right)`
  return String.raw`\left(\frac{\left(${mean}-r_f\right)}{${std}}\cdot \sqrt{p_{\mathrm{year}}}\right)`
}

function setup(rolling: boolean, explicitMinimum = false) {
  const mean = rolling ? 'rolling_mean' : 'mean'
  const std = rolling ? 'rolling_std' : 'std'
  const meanSource = rolling ? `rolling_mean(returns, 5${explicitMinimum ? ', 5' : ''})` : 'mean(returns)'
  const stdSource = rolling ? `rolling_std(returns, 5, 1${explicitMinimum ? ', 5' : ''})` : 'std(returns, 1)'
  const expression = `(${meanSource} - risk_free_rate_per_observation) / ${stdSource} * sqrt(periods_per_year)`
  const name = rolling ? '5 日滚动年化夏普比率' : '年化夏普比率'
  const definition: IndicatorDefinition = {
    id: rolling ? 'builtin-rolling-5d-annualized-sharpe-series' : 'builtin-annualized-sharpe-v2',
    name, description: '测试真实形状的公式构建协议。', expression,
    editable_latex: sharpeLatex(rolling, 5, explicitMinimum),
    revision: 1, source: 'built_in', read_only: true, created_at: '', updated_at: '',
    context_kind: 'single_product', result_kind: rolling ? 'time_series' : 'scalar',
    output_contract: rolling ? 'series_bundle' : 'scalar', output_measure: rolling ? 'series_bundle' : 'dimensionless',
    indicator_type: 'risk_adjusted', unit: '', precision: 3, display_format: 'number', direction: 'higher_better',
    annual_risk_free_rate_percent: 1.5, dsl_version: '2.3.0', operator_registry_version: '2.3.0', periods: ['1Y'],
    axis_anchor: rolling ? 'adjusted_nav' : null,
    series_outputs: rolling ? [{ id: 'value', label: name, expression, editable_latex: sharpeLatex(rolling, 5, explicitMinimum), unit: '', display_format: 'number', precision: 3, output_measure: 'auto' }] : [],
    ...(rolling ? { rolling_source: {
      kind: 'rolling_scalar', transform_version: '1.0.0', indicator_id: 'builtin-annualized-sharpe-v2',
      indicator_revision: 1, indicator_name: '年化夏普比率', definition_hash: 'a'.repeat(64), source_dsl_version: '2.3.0',
      window_observations: 5, minimum_observations: 5, detached: false,
    } } : {}),
  }
  const operator = (id: string, label: string, parameters: IndicatorOperatorParameter[]): IndicatorOperator => ({
    name: id, label, parameters, signature: id, latex_template: '', return_type: 'same(first)',
    output_shape: 'unknown', domains: ['single_product'], category_id: 'basic', category_label: '基础数学',
  })
  const operators = [
    operator(mean, rolling ? '滚动平均值' : '全元素算术平均值', rolling ? [values, windowParameter] : [values]),
    operator(std, rolling ? '滚动标准差' : '全元素标准差', rolling ? [values, windowParameter] : [values, ddof]),
    operator('subtract', '逐元素减法', pair), operator('divide', '逐元素安全除法', pair),
    operator('multiply', '逐元素乘法', pair), operator('sqrt', '逐元素平方根', [values]),
  ]
  if (rolling) {
    operators[0].parameter_sets = [
      { arity: 2, parameters: [values, windowParameter] },
      { arity: 3, parameters: [values, windowParameter, minimum] },
    ]
    operators[1].parameter_sets = [
      { arity: 2, parameters: [values, windowParameter] },
      { arity: 3, parameters: [values, windowParameter, ddof] },
      { arity: 4, parameters: [values, windowParameter, ddof, minimum] },
    ]
  }
  const dag: IndicatorDag = { nodes: [], edges: [], roots: { [rolling ? 'value' : 'result']: 'root' } }
  const node = (id: string, kind: string, label: string, shape: 'scalar' | 'series') => {
    dag.nodes.push({ id, kind, label, shape, value_type: shape === 'series' ? 'series<time>' : 'scalar', formula_fragment: label })
  }
  node('r', 'variable', 'returns', 'series')
  node('rf', 'variable', 'risk_free_rate_per_observation', 'scalar')
  node('year', 'variable', 'periods_per_year', 'scalar')
  node('window', 'constant', '5', 'scalar')
  node('ddof', 'constant', '1', 'scalar')
  const call = (id: string, operatorId: string, children: Array<[string, string]>, scalar = false) => {
    node(id, 'call', operatorId, scalar ? 'scalar' : 'series')
    children.forEach(([parameter, source], order) => dag.edges.push({ source, target: id, parameter, order }))
  }
  const meanArgs: Array<[string, string]> = [['values', 'r']]
  const stdArgs: Array<[string, string]> = [['values', 'r']]
  if (rolling) { meanArgs.push(['window', 'window']); stdArgs.push(['window', 'window']) }
  stdArgs.push(['ddof', 'ddof'])
  if (explicitMinimum) { meanArgs.push(['min_periods', 'window']); stdArgs.push(['min_periods', 'window']) }
  call('mean', mean, meanArgs, !rolling)
  call('std', std, stdArgs, !rolling)
  call('excess', 'subtract', [['lhs', 'mean'], ['rhs', 'rf']], !rolling)
  call('ratio', 'divide', [['lhs', 'excess'], ['rhs', 'std']], !rolling)
  call('scale', 'sqrt', [['values', 'year']], true)
  call('root', 'multiply', [['lhs', 'ratio'], ['rhs', 'scale']], !rolling)
  const meta: IndicatorMeta = {
    engine_version: 'test', workspace_scope: 'shared', limits: {}, operators,
    periods: [{ value: '1Y', label: '近一年', description: '' }], templates: [],
    variables: [
      { name: 'returns', label: '普通收益率', shape: 'series', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{r}', domains: ['single_product'] },
      { name: 'risk_free_rate_per_observation', label: '单观察期无风险收益率', shape: 'scalar', value_type: 'scalar', dtype: 'float64', latex: 'r_f', domains: ['single_product'] },
      { name: 'periods_per_year', label: '年化因子', shape: 'scalar', value_type: 'scalar', dtype: 'float64', latex: 'p_{\\mathrm{year}}', domains: ['single_product'] },
    ],
    indicator_types: [{ id: 'risk_adjusted', label: '风险调整指标' }],
  }
  const composeRequests: Array<{ operator_id: string; arguments: Array<{ parameter: string; value: string | number }> }> = []
  const json = (value: unknown) => new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } })
  vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
    if (url.endsWith('/meta')) return json(meta)
    if (url.includes('/snapshot-config')) return json({ revision: 1, items: [], max_items: 30 })
    if (url.includes('/custom-indicators?')) return json({ items: [definition], total: 1 })
    if (url.endsWith('/validate')) {
      const body = JSON.parse(String(init?.body))
      return json({ valid: true, diagnostics: [], dependencies: ['returns', 'risk_free_rate_per_observation', 'periods_per_year'],
        dag, python_expression: expression,
        editable_latex: sharpeLatex(rolling, 5, explicitMinimum),
        display_latex: '\\frac{\\mu_{t,5}(r)-r_f}{s_{t,5}(r)}\\sqrt{p_{year}}',
        output_channels: body.series_outputs,
        output_inferences: rolling ? { value: { id: 'value', expression: body.expression, python_expression: expression,
          editable_latex: sharpeLatex(rolling, 5, explicitMinimum),
          inferred_type: 'series<time>', shape: 'series', dependencies: ['returns'], root_id: 'root',
          display_latex: '\\frac{\\mu_{t,5}(r)-r_f}{s_{t,5}(r)}\\sqrt{p_{year}}' } } : undefined,
      })
    }
    if (url.endsWith('/compose')) {
      const body = JSON.parse(String(init?.body))
      composeRequests.push(body)
      const v = body.arguments.map((argument: { value: string | number }) => String(argument.value))
      const source = body.operator_id === 'subtract' ? `${v[0]} - ${v[1]}`
        : body.operator_id === 'divide' ? `(${v[0]}) / ${v[1]}`
          : body.operator_id === 'multiply' ? `${v[0]} * ${v[1]}` : `${body.operator_id}(${v.join(', ')})`
      // Legacy/display fields deliberately differ. Only editable_latex belongs
      // in the textarea; nested composition still consumes canonical DSL.
      return json({ expression: source, python_expression: source, latex: '\\left(r_f\\right)',
        editable_latex: sharpeLatex(rolling, source.includes('15') ? 15 : 5, explicitMinimum),
        display_latex: '\\mu_{t,5}(r)', inferred_type: 'series<time>', shape: rolling ? 'series' : 'scalar', semantic_warnings: [] })
    }
    throw new Error(`Unexpected request: ${url}`)
  }))
  return { definition, composeRequests }
}

async function click(user: ReturnType<typeof userEvent.setup>, element: HTMLElement) {
  await act(async () => { await user.click(element) })
}

async function openBuilder(user: ReturnType<typeof userEvent.setup>) {
  await click(user, screen.getByRole('button', { name: '浏览公式构建资源' }))
  return screen.findByRole('dialog', { name: /编辑.*计算逻辑/ })
}

describe('IndicatorStudio formula source roundtrip', () => {
  it.each([false, true])('reopening a scalar/rolling Sharpe preserves source and ddof (rolling=%s)', async (rolling) => {
    const { definition, composeRequests } = setup(rolling)
    const user = userEvent.setup()
    await act(async () => { render(<MemoryRouter><IndicatorStudio /></MemoryRouter>) })
    await click(user, await screen.findByRole('button', { name: new RegExp(definition.name) }))
    await click(user, screen.getByRole('button', { name: rolling ? '解析并校验全部通道' : '解析并校验公式' }))
    for (let pass = 0; pass < 2; pass++) {
      const drawer = await openBuilder(user)
      expect(within(drawer).getByLabelText('自由度修正 有限常量')).toHaveValue(1)
      await click(user, within(drawer).getByRole('button', { name: '应用逻辑修改' }))
      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
    }
    await click(user, screen.getByRole('tab', { name: '高级公式模式' }))
    const source = screen.getByRole('textbox', { name: /公式源码/ })
    expect(source).toHaveValue(definition.editable_latex)
    expect((source as HTMLTextAreaElement).value).toContain('\\frac{')
    expect((source as HTMLTextAreaElement).value).toContain('r_f')
    expect((source as HTMLTextAreaElement).value).not.toContain('risk_free_rate_per_observation')
    const stdCalls = composeRequests.filter((request) => request.operator_id === (rolling ? 'rolling_std' : 'std'))
    expect(stdCalls).toHaveLength(2)
    expect(stdCalls.every((request) => request.arguments.some((arg) => arg.parameter === 'ddof' && arg.value === 1))).toBe(true)
    if (rolling) expect(screen.getByText('来源已锁定')).toBeInTheDocument()
  })

  it('editing to fifteen observations retains ddof and explicit min_periods', async () => {
    const { definition, composeRequests } = setup(true, true)
    const user = userEvent.setup()
    await act(async () => { render(<MemoryRouter><IndicatorStudio /></MemoryRouter>) })
    await click(user, await screen.findByRole('button', { name: new RegExp(definition.name) }))
    const drawer = await openBuilder(user)
    expect(within(drawer).getByLabelText('自由度修正 有限常量')).toHaveValue(1)
    for (const label of ['窗口期数 有限常量', '最少有效观察数 有限常量']) {
      const inputs = within(drawer).getAllByLabelText(label)
      expect(inputs).toHaveLength(2)
      inputs.forEach((input) => fireEvent.change(input, { target: { value: '15' } }))
    }
    await click(user, within(drawer).getByRole('button', { name: '应用逻辑修改' }))
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
    await click(user, screen.getByRole('tab', { name: '高级公式模式' }))
    expect(screen.getByRole('textbox', { name: /公式源码/ })).toHaveValue(
      sharpeLatex(true, 15, true),
    )
    const request = composeRequests.find((item) => item.operator_id === 'rolling_std')
    expect(request?.arguments).toHaveLength(4)
    expect(screen.getByText('已脱离来源')).toBeInTheDocument()
    await click(user, screen.getByRole('button', { name: '解析并校验全部通道' }))
    expect(screen.getByTestId('formula-preview').querySelector('.katex')).not.toBeNull()
  })
})
