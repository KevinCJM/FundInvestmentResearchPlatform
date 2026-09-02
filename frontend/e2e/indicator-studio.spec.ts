import { expect, test, type Page } from '@playwright/test'

const meta = {
  engine_version: 'e2e',
  workspace_scope: 'shared',
  dsl_version: '2.1.0',
  operator_registry_version: '2.1.0',
  variable_registry_version: '2.1.0',
  context_schema_version: 'typed-context-v2',
  data_contract_version: 'tushare-eod-v2',
  limits: {},
  periods: [
    { value: '1M', label: '近 1 月', description: '自然月窗口' },
    { value: '1Y', label: '近 1 年', description: '自然年窗口' },
  ],
  templates: [],
  predefined_calculations: [],
  variables: [
    {
      name: 'returns', label: '普通收益率序列', value_type: 'series<time>', dtype: 'float64',
      latex: '\\mathbf{r}', shape: 'series', semantic: 'return_decimal', semantic_role: 'ordinary_return',
      measure: 'return_decimal', price_basis: 'adjusted_nav', description: '由真实复权净值派生。',
      source: 'Tushare 本地 Parquet', source_dataset: 'nav', source_field: 'adj_nav', unit: 'decimal',
      domains: ['single_product'], product_kinds: ['etf', 'fund'], category_id: 'returns', category_label: '收益与变化',
    },
  ],
  operators: [
    {
      name: 'mean', label: '全元素算术平均值', signature: 'numeric array → scalar',
      latex_template: '\\operatorname{mean}(values)', return_type: 'scalar', output_shape: 'scalar',
      mathematical_essence: '对全部元素计算算术平均值。', category_id: 'statistics', category_label: '统计归约',
      domains: ['single_product', 'portfolio'], parameters: [{ name: 'values', label: '输入值', allowed_shapes: ['series', 'vector', 'matrix'] }],
    },
  ],
}

const indicator = {
  id: 'builtin-mean-return-v2', revision: 1, source: 'built_in', read_only: true,
  created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
  name: '平均单期收益率', description: '普通收益率序列的算术平均值。', expression: 'mean(returns)',
  periods: ['1M', '1Y'], period_policy: 'all_supported', unit: '%', display_format: 'percent', precision: 3,
  direction: 'higher_better', annual_risk_free_rate_percent: 1.5, dsl_version: '2.1.0',
  operator_registry_version: '2.1.0', variable_registry_version: '2.1.0', context_kind: 'single_product', output_contract: 'scalar',
}

async function mockApi(page: Page) {
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url())
    if (url.pathname === '/api/custom-indicators/meta') return route.fulfill({ json: meta })
    if (url.pathname === '/api/custom-indicators') return route.fulfill({ json: { items: [indicator], total: 1 } })
    if (url.pathname === '/api/custom-indicators/validate') return route.fulfill({ json: {
      valid: true,
      diagnostics: [],
      dependencies: ['returns'],
      python_expression: 'mean(returns)',
      display_latex: '\\overline{\\mathbf{r}}',
      dag: {
        nodes: [
          { id: 'input', label: 'returns', kind: 'variable', value_type: 'series<time>', shape: 'series', symbolic_shape: ['T'] },
          { id: 'root', label: 'mean', operator_id: 'mean', kind: 'call', value_type: 'scalar', shape: 'scalar', symbolic_shape: [] },
        ],
        edges: [{ source: 'input', target: 'root', parameter: 'values' }],
        roots: { result: 'root' },
      },
    } })
    if (url.pathname === '/api/portfolio-runs') return route.fulfill({ json: { items: [] } })
    return route.fulfill({ json: { items: [] } })
  })
}

test.beforeEach(async ({ page }) => {
  await mockApi(page)
  await page.goto('/indicator-studio')
  await expect(page.getByRole('button', { name: /平均单期收益率/ })).toBeVisible()
})

test('指标定义无周期选择，资源目录可键盘搜索', async ({ page }, testInfo) => {
  if (testInfo.project.name === 'mobile-320') {
    await page.getByRole('tab', { name: '编辑' }).click()
  }
  await expect(page.getByText('指标定义默认支持全部计算周期')).toBeVisible()
  await expect(page.getByText('可用周期')).toHaveCount(0)
  await page.getByRole('button', { name: '浏览公式构建资源' }).click()
  const dialog = page.getByRole('dialog', { name: '变量、算子与已有指标' })
  await expect(dialog).toBeVisible()
  await dialog.getByRole('combobox', { name: '选择变量' }).press('Enter')
  await dialog.getByLabel('搜索选择变量').fill('普通收益')
  await dialog.getByLabel('搜索选择变量').press('Enter')
  await expect(dialog.getByRole('heading', { name: '普通收益率序列' })).toBeVisible()
  // 第一层 Esc 关闭仍获得焦点的资源下拉；第二层 Esc 关闭资源抽屉。
  await page.keyboard.press('Escape')
  await dialog.getByRole('button', { name: '关闭资源目录' }).focus()
  await page.keyboard.press('Escape')
  await expect(dialog).toBeHidden()
})

test('三档布局无关键水平溢出', async ({ page }, testInfo) => {
  if (testInfo.project.name === 'mobile-320') {
    await expect(page.getByRole('tablist', { name: '指标中心区域' })).toBeVisible()
    await expect(page.getByText('校验与预览')).toBeHidden()
    await page.getByRole('tab', { name: '预览' }).click()
    await expect(page.getByText('校验与预览')).toBeVisible()
  } else {
    await expect(page.getByText('校验与预览')).toBeVisible()
  }
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow).toBeLessThanOrEqual(1)
})

test('校验与预览保留多个深链产品且最多选择十个', async ({ page }, testInfo) => {
  const ids = Array.from({ length: 12 }, (_, index) => `TEST${String(index + 1).padStart(2, '0')}.SH`)
  await page.goto(`/indicator-studio?kind=etf&ids=${encodeURIComponent(ids.join(','))}`)
  await expect(page.getByRole('button', { name: /平均单期收益率/ })).toBeVisible()
  if (testInfo.project.name === 'mobile-320') {
    await page.getByRole('tab', { name: '预览' }).click()
  }

  await expect(page.getByText('10 / 10')).toBeVisible()
  await expect(page.getByText('校验与预览最多选择 10 个产品，已保留前 10 个。')).toBeVisible()
  await expect(page.getByText('TEST10.SH')).toBeVisible()
  await expect(page.getByText('TEST11.SH')).toHaveCount(0)
})

test('校验成功后可展示数组形式的数据规模且页面不白屏', async ({ page }, testInfo) => {
  await page.getByRole('button', { name: /平均单期收益率/ }).click()
  if (testInfo.project.name === 'mobile-320') {
    await page.getByRole('tab', { name: '编辑' }).click()
  }
  await page.getByRole('button', { name: '校验公式' }).click()
  await expect(page.getByText('校验通过', { exact: true })).toBeVisible()

  if (testInfo.project.name === 'mobile-320') {
    await page.getByRole('tab', { name: '预览' }).click()
  }
  await expect(page.getByRole('heading', { name: '层级计算 DAG' })).toBeVisible()
  await expect(page.getByRole('article', { name: 'DAG 节点详情' })).toContainText('理论规模：单个数值')
  await expect(page.getByRole('table', { name: 'DAG 节点、输入参数和输出类型数据表' })).toContainText('理论规模：随计算窗口变化的时间点数量')
})
