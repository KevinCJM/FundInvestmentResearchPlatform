import { expect, test, type Page } from '@playwright/test'

async function mockIndexApi(page: Page) {
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url())
    if (url.pathname === '/api/data/refresh/status') {
      return route.fulfill({ json: {
        source: 'tushare', enabled: true, full_refresh_enabled: true,
        available_modules: ['base', 'etf', 'fund', 'index'],
        available_index_scopes: ['catalog', 'domestic', 'industry', 'concept', 'global', 'futures', 'valuation', 'constituents'],
        default_index_scopes: ['catalog', 'domestic', 'industry', 'global'],
        token_configured: true, token_configuration_enabled: true, token_editable: true,
        job: { status: 'idle', message: '尚未启动更新' }, datasets: {},
      } })
    }
    if (url.pathname === '/api/indices/summary') {
      return route.fulfill({ json: {
        schema_version: 1, status: 'complete', catalog_count: 2000, source_count: 8,
        covered_count: 1800, latest_date: '2026-08-31', missing_count: 150, stale_count: 50,
        datasets: [
          { key: 'index_domestic', scope: 'domestic', file: 'index_daily_df.parquet', exists: true, status: 'ready', rows: 500000, earliest_date: '2005-01-04', latest_date: '2026-08-31' },
          { key: 'index_sw', scope: 'industry', file: 'index_sw_daily_df.parquet', exists: true, status: 'ready', rows: 300000, earliest_date: '2014-01-02', latest_date: '2026-08-31' },
          { key: 'index_global', scope: 'global', file: 'index_global_daily_df.parquet', exists: true, status: 'ready', rows: 100000, earliest_date: '2010-01-04', latest_date: '2026-08-31' },
        ],
      } })
    }
    if (url.pathname === '/api/indices') {
      return route.fulfill({ json: {
        schema_version: 1, status: 'complete', page: 1, page_size: 20, total: 1,
        filters: { source_api: ['index_basic'], category: ['规模指数'], market: ['CSI'], coverage_status: ['ready'] },
        items: [{
          source_api: 'index_basic', quote_source_api: 'index_daily', ts_code: '000300.SH',
          name: '沪深300', category: '规模指数', market: 'CSI', publisher: '中证指数公司',
          list_date: '2005-04-08', exp_date: null, status: 'active', first_date: '2005-04-08',
          latest_date: '2026-08-31', rows: 5000, coverage_status: 'ready',
        }],
      } })
    }
    return route.fulfill({ status: 404, json: {} })
  })
}

test.beforeEach(async ({ page }) => {
  await mockIndexApi(page)
  await page.goto('/index-data')
  await expect(page.getByRole('heading', { name: '指数数据中心' })).toBeVisible()
})

test('三档布局只允许目录表局部横向滚动，筛选同步 URL', async ({ page }) => {
  await expect(page.getByRole('table', { name: '指数目录、来源与行情覆盖状态' })).toBeVisible()
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow).toBeLessThanOrEqual(1)

  const scrollRegion = page.getByLabel('指数目录表格滚动区域')
  await scrollRegion.focus()
  await expect(scrollRegion).toBeFocused()
  await page.getByLabel('来源').selectOption('index_basic')
  await expect(page).toHaveURL(/source=index_basic/)
  await expect(page).toHaveURL(/page=1/)
})
