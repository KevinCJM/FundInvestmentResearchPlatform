import { expect, test } from '@playwright/test'
import { auditTextContrast } from './helpers/contrast'

/**
 * 产品列表页的构图声明只能在浏览器里量：容器宽度、横向溢出、sticky 行为和
 * 有数据时的对比度都不是静态扫描能判定的。contrast.spec.ts 只覆盖离线错误外壳。
 */
const execution = {
  execution_backend: 'numba_njit_fixed_signature',
  nopython: true,
  object_mode: 0,
  python_fallback: 0,
  request_time_compilation: 0,
  kernel_signatures: { product_summary_kernel: ['fixed'] },
}

const items = Array.from({ length: 10 }, (_, index) => ({
  ts_code: `51000${index}.SH`,
  name: `样例交易型开放式指数基金 ${String(index + 1).padStart(2, '0')}`,
  type: 'ETF',
  fund_type: '股票型',
  invest_type: '宽基指数',
  qdii_type: index % 2 === 0 ? '非QDII' : 'QDII',
  market: '上交所',
  status: index % 3 === 0 ? '终止上市' : '上市',
  management: '示例基金管理有限公司',
  custodian: '示例银行股份有限公司',
  issue_amount: 12345 + index * 17,
  m_fee: 0.5,
  c_fee: 0.1,
  benchmark: '沪深300指数收益率',
  list_date: '2012-05-28',
  found_date: '2012-04-01',
}))

const payload = {
  items,
  page: 1,
  page_size: 10,
  total: 42,
  summary: {
    universe_total: 600, filtered_total: 42, active_count: 40, active_rate: 0.95,
    avg_m_fee: 0.5, avg_c_fee: 0.1, total_issue_amount: 123456, median_issue_amount: 1200, unique_managements: 30,
  },
  available_filters: { fund_type: [], type: [], invest_type: [], qdii_type: [], market: [], status: [], management: [], custodian: [] },
  condition_fields: [{ field: 'list_date', label: '上市日期', data_type: 'date', unit_label: null, input_scale: 1, source: 'fund_basic', available: true }],
  condition_operators: [{ value: 'gte', label: '大于等于', symbol: '≥' }],
  snapshot: { status: 'ready', as_of: '2026-08-31' },
  sort_by: 'issue_amount',
  sort_dir: 'desc',
  execution,
}

test.beforeEach(async ({ page }) => {
  // 后注册的路由优先，兜底必须先注册，避免真实开发数据被请求到。
  await page.route('**/api/**', route => route.fulfill({ status: 503, json: { detail: 'Offline fixture' } }))
  await page.route('**/api/custom-indicators**', route => route.fulfill({ status: 200, json: { items: [], total: 0, periods: [{ value: '1Y', label: '近 1 年', description: '运行周期' }] } }))
  await page.route('**/api/instruments/products*', route => route.fulfill({ status: 200, json: payload }))
})

test('列表页没有页面级横向滚动，宽表格只在自己的容器里横向滚动', async ({ page }, testInfo) => {
  await page.goto('/product-research/products')
  await expect(page.getByRole('link', { name: '进入样例交易型开放式指数基金 01的单产品研究页面', exact: true })).toBeVisible()

  const overflow = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }))
  expect(overflow.scrollWidth, '页面整体不应出现横向滚动').toBeLessThanOrEqual(overflow.clientWidth + 1)

  // 表格自己的滚动容器是唯一允许的横向滚动区；纵向不再嵌套第二个滚动条。
  const box = page.locator('table').locator('xpath=..')
  const scroll = await box.evaluate((node) => ({
    overflowX: getComputedStyle(node).overflowX,
    verticalScroll: node.scrollHeight - node.clientHeight,
  }))
  expect(scroll.overflowX).toBe('auto')
  expect(scroll.verticalScroll, '表格容器不应有自己的纵向滚动').toBeLessThanOrEqual(1)

  // 提示按实测溢出显示：1440 下这张表已经放得下，就不该再提示可以滑动。
  const hint = page.getByText('表格可左右滑动查看更多列')
  if (testInfo.project.name === 'desktop-1440') await expect(hint).toBeHidden()
  else await expect(hint).toBeVisible()
})

test('内容区用满阶段外壳的宽度，选中后出现跟随滚动的操作条', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop-1440', '构图断言只在桌面断点有意义')
  await page.goto('/product-research/products')
  await expect(page.getByRole('link', { name: '进入样例交易型开放式指数基金 01的单产品研究页面', exact: true })).toBeVisible()

  // 页面不再自己套 max-w-6xl（1152px）再加 px-6；1440 下可用宽度由外壳决定。
  const width = await page.locator('table').locator('xpath=..').evaluate((node) => node.clientWidth)
  expect(width).toBeGreaterThan(1088)

  await page.getByRole('checkbox', { name: '选择 样例交易型开放式指数基金 01', exact: true }).check()
  const bar = page.getByRole('region', { name: '已选产品操作' })
  await expect(bar).toBeVisible()
  await expect(bar.getByRole('button', { name: '产品对比' })).toBeEnabled()
  expect(await bar.evaluate((node) => getComputedStyle(node).position)).toBe('sticky')
})

test('有数据时列表、状态徽章和排序控件的文字对比度达标', async ({ page }) => {
  await page.goto('/product-research/products')
  await expect(page.getByRole('link', { name: '进入样例交易型开放式指数基金 01的单产品研究页面', exact: true })).toBeVisible()
  await page.getByRole('checkbox', { name: '选择 样例交易型开放式指数基金 01', exact: true }).check()
  // 全选筛选结果会同时显示禁用说明（琥珀色）和取消全选，把这两种文字一并纳入。
  await page.getByRole('button', { name: '全选 42 条' }).click()
  await expect(page.getByText('全选筛选结果是逻辑选择；请取消全选后手动选择最多 10 个产品进行对比。')).toBeVisible()

  const failures = await page.evaluate(auditTextContrast)
  expect(failures, failures.map((item) => `${item.ratio}:1 «${item.text}» ${item.color} — ${item.path}`).join('\n')).toEqual([])
})

test('排序按钮同时更新方向图形和表头的 aria-sort', async ({ page }) => {
  await page.goto('/product-research/products')
  await expect(page.getByRole('link', { name: '进入样例交易型开放式指数基金 01的单产品研究页面', exact: true })).toBeVisible()

  const header = page.getByRole('columnheader', { name: /发行规模/ })
  await expect(header).toHaveAttribute('aria-sort', 'descending')
  const descendingPath = await header.locator('svg path').getAttribute('d')

  await header.getByRole('button', { name: '按发行规模排序' }).click()
  await expect(header).toHaveAttribute('aria-sort', 'ascending')
  expect(await header.locator('svg path').getAttribute('d')).not.toBe(descendingPath)
})
