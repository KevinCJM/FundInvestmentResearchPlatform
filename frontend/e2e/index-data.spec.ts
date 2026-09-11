import { expect, test, type Page } from '@playwright/test'

const fixedExecution = {
  execution_backend: 'numba_njit_fixed_signature',
  nopython: true,
  object_mode: 0,
  python_fallback: 0,
  request_time_compilation: 0,
  kernel_signatures: { coverage_ratio_kernel: ['fixed'] },
}

async function mockDataQualityApi(page: Page) {
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url())
    if (url.pathname === '/api/data/refresh/status') {
      return route.fulfill({ json: {
        source: 'tushare', enabled: true, full_refresh_enabled: true,
        available_modules: ['base', 'etf', 'fund', 'index'],
        available_index_scopes: ['catalog', 'domestic', 'industry', 'concept', 'global', 'futures', 'valuation', 'constituents'],
        default_index_scopes: ['catalog', 'domestic', 'industry', 'global'],
        token_configured: true, token_configuration_enabled: true, token_editable: true,
        job: { status: 'idle', message: '尚未启动更新' },
        datasets: {
          calendar: { file: 'trade_day_df.parquet', exists: true, status: 'ready', rows: 6000, earliest_date: '2010-01-01', latest_date: '2026-08-31', updated_at: '2026-08-31T08:00:00Z' },
          etf_nav: { file: 'etf_daily_df.parquet', exists: true, status: 'ready', rows: 10000, earliest_date: '2010-01-04', latest_date: '2026-08-31', updated_at: '2026-08-31T08:00:00Z' },
          fund_nav: { file: 'fund_nav_df.parquet', exists: false, status: 'missing', rows: 0, earliest_date: null, latest_date: null },
          index_ci: { file: 'index_ci_daily_df.parquet', exists: true, status: 'ready', rows: 0, earliest_date: null, latest_date: null },
          instrument_metrics: { file: 'instrument_metrics_snapshot.parquet', exists: true, status: 'ready', rows: 3000, earliest_date: '2024-01-01', latest_date: '2026-08-31', updated_at: '2026-08-31T08:30:00Z' },
        },
      } })
    }
    if (url.pathname === '/api/data/quality') {
      return route.fulfill({ json: {
        schema_version: 1, status: 'attention', generated_at: '2026-09-01T08:30:00Z', activated_at: '2026-09-01T08:45:00Z', as_of: '2026-08-31',
        summary: {
          checks_total: 9, checks_passed: 5, checks_warning: 4, checks_failed: 0, checks_unavailable: 0,
          total_products: 300, affected_products: 18, affected_rate: 0.06, issue_count: 2,
          critical_issue_count: 0, high_issue_count: 2, medium_issue_count: 0,
          nav_anomaly_products: 3, nav_anomaly_events: 4, stale_active_products: 8,
        },
        checks: [
          { key: 'schema', label: '结构契约', dimension: 'validity', status: 'passed', summary: '必要字段齐全', detail: '检查固定字段契约。', threshold: '必要字段必须 100% 存在' },
          { key: 'primary_key', label: '主键唯一性', dimension: 'integrity', status: 'passed', summary: '产品键无重复', detail: '检查产品和日期复合键。', threshold: '重复键 = 0' },
          { key: 'nav_discontinuity', label: '净值突变', dimension: 'continuity', status: 'warning', summary: '3 个产品 / 4 个异常点', detail: '用参考净值交叉确认跳变。', threshold: '复权变动 > 20% 且参考净值稳定；或单日变动 > 100%' },
          { key: 'series_density', label: '序列连续性', dimension: 'continuity', status: 'warning', summary: '7 个产品需补数', detail: '检查交易日覆盖率。', threshold: '交易日覆盖 ≥ 90%；连续缺失 ≤ 5 个交易日' },
        ],
        issues: [
          { id: 'nav-discontinuity', code: 'NAV_DISCONTINUITY', severity: 'high', dimension: 'continuity', scope: 'ETF 与场外基金净值', title: '发现净值突变或复权断点', description: '复权净值发生大幅跳变。', evidence: '3 个产品共发现 4 个异常点。', impact: '相关区间指标不可用。', affected_count: 3, affected_rate: 0.01, record_count: 4, action: 'inspect_source', samples: [{ kind: 'fund', ts_code: '000001.OF', name: '测试基金', latest_date: '2026-08-31', observed: '2 个异常点' }] },
          { id: 'series-gap', code: 'SERIES_INTERNAL_GAP', severity: 'high', dimension: 'continuity', scope: '近一年产品净值', title: '净值序列存在密度不足或连续缺口', description: '近一年窗口未通过连续性检查。', evidence: '7 个产品需要补数。', impact: '相关指标不会进入正式排行。', affected_count: 7, affected_rate: 0.023, record_count: 0, action: 'refresh', samples: [] },
        ],
        validation: { status: 'passed', manifest: 'tushare_active.json' },
        execution: fixedExecution,
      } })
    }
    if (url.pathname === '/api/instruments/analytics') {
      return route.fulfill({ json: {
        schema_version: 1, kind: 'all', status: 'partial', as_of: '2026-08-31',
        availability: {},
        summary: { etf: { index_coverage_rate: 0.92 }, fund: {} },
        segments: {}, available_filters: {}, metric_definitions: {},
        data_quality: {
          snapshot: { exists: true, status: 'ready', rows: 3000, as_of: '2026-08-31' },
          segments: {
            etf: { info_file_exists: true, info_rows: 100, snapshot_rows: 90, nav_coverage_rate: 0.9, warnings: [] },
            fund: { info_file_exists: true, info_rows: 200, snapshot_rows: 160, nav_coverage_rate: 0.8, warnings: [] },
          },
          warnings: [{ code: 'NAV_PARTIAL', message: '部分产品净值历史不足。', kind: 'fund' }],
        },
        execution: fixedExecution,
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
        execution: fixedExecution,
      } })
    }
    return route.fulfill({ status: 404, json: {} })
  })
}

test.beforeEach(async ({ page }) => {
  await mockDataQualityApi(page)
  await page.goto('/settings/data-quality')
  await expect(page.getByRole('heading', { name: '数据质量监控与治理' })).toBeVisible()
})

test('三档布局展示深度检查与异常证据，并只允许明细表局部横向滚动', async ({ page }) => {
  const overview = page.getByRole('region', { name: '数据质量概览' })
  for (const metricLabel of ['深度检查规则', '受影响产品', '净值突变', '待处理问题']) {
    await expect(overview.locator('article').filter({ hasText: metricLabel }).getByText('需关注')).toBeVisible()
  }
  await expect(page.getByRole('heading', { name: '深度检查矩阵' })).toBeVisible()
  await expect(page.getByText('发现净值突变或复权断点')).toBeVisible()
  await expect(page.getByText('000001.OF')).toBeVisible()
  await expect(page.getByText('交易日覆盖 ≥ 90%；连续缺失 ≤ 5 个交易日')).toBeVisible()
  const table = page.getByRole('table', { name: '核心数据集质量明细' })
  await expect(table).toBeVisible()
  await expect(page.getByText('部分指数缺少行情覆盖')).toBeVisible()
  await expect(page.getByText('指数数据中心')).toHaveCount(0)
  await expect(page.getByText(/全景驾驶舱/)).toHaveCount(0)
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow).toBeLessThanOrEqual(1)

  const scrollRegion = page.getByLabel('核心数据集质量明细滚动区域')
  await scrollRegion.focus()
  await expect(scrollRegion).toBeFocused()
  await page.getByRole('button', { name: '只看需处理' }).click()
  await expect(page.getByRole('button', { name: '只看需处理' })).toHaveAttribute('aria-pressed', 'true')
  await expect(table.getByText('ETF 净值', { exact: true })).toHaveCount(0)
  await page.getByLabel('数据域', { exact: true }).selectOption('index')
  await expect(table.getByText('中信行业行情', { exact: true })).toBeVisible()
  await expect(table.getByText('场外基金净值', { exact: true })).toHaveCount(0)
})
