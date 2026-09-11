import { expect, test, type Page } from '@playwright/test'

const fixedExecution = {
  execution_backend: 'numba_njit_fixed_signature',
  nopython: true,
  object_mode: 0,
  python_fallback: 0,
  request_time_compilation: 0,
  kernel_signatures: { dashboard_kernel: ['fixed'] },
}

const summary = {
  share_code_count: 120,
  active_count: 100,
  issuing_count: 2,
  inactive_count: 18,
  unknown_status_count: 0,
  unique_managements: 12,
  nav_covered_count: 110,
  nav_coverage_rate: 110 / 120,
  issue_amount_total: 1000,
  issue_amount_coverage_rate: 0.8,
  latest_nav_date: '2026-08-31',
  latest_candle_date: '2026-08-31',
  index_covered_count: 108,
  index_coverage_rate: 0.9,
  liquidity_covered_count: 96,
  liquidity_coverage_rate: 0.8,
  purchase_redemption_covered_count: 90,
  purchase_redemption_coverage_rate: 0.75,
}

function segment(kind: 'etf' | 'fund') {
  return {
    availability: 'ready',
    snapshot_availability: 'ready',
    summary,
    distributions: {
      fund_type: [{ name: kind === 'etf' ? '股票型ETF' : '混合型', value: 80 }],
      invest_type: [{ name: '被动指数型', value: 70 }],
      market: [{ name: kind === 'etf' ? '上交所' : '场外', value: 120 }],
      status: [{ name: kind === 'etf' ? '上市交易' : '存续', value: 100 }],
      management: [{ name: '示例基金管理人', value: 60 }],
      index_name: kind === 'etf' ? [{ name: '沪深300', value: 20 }] : [],
      m_fee: [{ name: '0.25%–0.50%', value: 80 }],
      c_fee: [{ name: '≤0.25%', value: 90 }],
    },
    event_trend: {
      date_field: kind === 'etf' ? 'list_date' : 'found_date',
      label: kind === 'etf' ? 'ETF上市趋势' : '场外公募基金成立趋势',
      points: [{ year: 2025, count: 20 }, { year: 2026, count: 30 }],
    },
    latest_products: [{
      ts_code: kind === 'etf' ? '510300.SH' : '000001.OF',
      name: kind === 'etf' ? '沪深300ETF' : '示例场外基金',
      market: kind === 'etf' ? '上交所' : '场外',
      index_name: kind === 'etf' ? '沪深300' : null,
      fund_type: '混合型',
      invest_type: '被动指数型',
      list_date: kind === 'etf' ? '2026-08-20' : null,
      found_date: kind === 'fund' ? '2026-08-18' : null,
      purc_startdate: kind === 'fund' ? '2026-08-25' : null,
      redm_startdate: kind === 'fund' ? '2026-08-26' : null,
    }],
  }
}

function analytics(kind: 'all' | 'etf' | 'fund') {
  const etf = segment('etf')
  const fund = segment('fund')
  return {
    schema_version: 1,
    kind,
    status: 'complete',
    as_of: '2026-08-31',
    availability: { etf_info: 'ready', fund_info: 'ready', analysis_snapshot: 'ready' },
    summary: kind === 'all'
      ? { all: { ...summary, share_code_count: 240 }, etf: summary, fund: summary }
      : { [kind]: summary },
    segments: kind === 'all' ? { etf, fund } : { [kind]: kind === 'etf' ? etf : fund },
    available_filters: {
      fund_type: [{ value: '混合型', label: '混合型', count: 80 }],
      invest_type: [], status: [], management: [], market: [],
    },
    data_quality: {
      snapshot: { exists: true, rows: 220, updated_at: '2026-08-31', as_of: '2026-08-31' },
      segments: {
        etf: { info_file_exists: true, info_rows: 120, snapshot_rows: 110, nav_coverage_rate: 110 / 120, warnings: [] },
        fund: { info_file_exists: true, info_rows: 120, snapshot_rows: 110, nav_coverage_rate: 110 / 120, warnings: [] },
      },
      warnings: [],
    },
    metric_definitions: { return_1y: { label: '近1年收益率', unit: 'ratio', source: 'adj_nav' } },
    units: { return_1y: 'ratio' },
    execution: fixedExecution,
  }
}

async function mockDashboardApi(page: Page) {
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url())
    if (url.pathname === '/api/instruments/analytics') {
      return route.fulfill({ json: analytics((url.searchParams.get('kind') ?? 'all') as 'all' | 'etf' | 'fund') })
    }
    if (url.pathname === '/api/instruments/analytics/trend') {
      const kind = (url.searchParams.get('kind') ?? 'all') as 'all' | 'etf' | 'fund'
      const series = kind === 'all'
        ? { etf: segment('etf').event_trend, fund: segment('fund').event_trend }
        : { [kind]: segment(kind).event_trend }
      return route.fulfill({ json: { schema_version: 1, kind, status: 'complete', series, available_values: [], data_quality: { warnings: [] }, execution: fixedExecution } })
    }
    if (url.pathname === '/api/instruments/analytics/rankings') {
      const kind = (url.searchParams.get('kind') ?? 'etf') as 'etf' | 'fund'
      return route.fulfill({ json: {
        schema_version: 1, kind, status: 'complete', metric: 'return_1y',
        metric_definition: { label: '近1年收益率', unit: 'ratio', source: 'adj_nav' },
        sort_dir: 'desc', page: 1, page_size: 10, total: 1, as_of: '2026-08-31',
        items: [{
          instrument_type: kind, ts_code: kind === 'etf' ? '510300.SH' : '000001.OF',
          name: kind === 'etf' ? '沪深300ETF' : '示例场外基金', management: '示例基金管理人',
          status: '存续', latest_date: '2026-08-31', observation_count: 250, value: 0.12,
          metrics: { annual_volatility_1y: 0.18, max_drawdown_3y: -0.15, sharpe_1y: 0.7 },
        }],
        data_quality: { warnings: [] },
        execution: fixedExecution,
      } })
    }
    return route.fulfill({ status: 404, json: {} })
  })
}

test.beforeEach(async ({ page }) => {
  await mockDashboardApi(page)
  await page.goto('/product-research/panorama?kind=all')
  await expect(page.getByRole('heading', { name: 'ETF市场镜头' })).toBeVisible()
  await expect(page.getByRole('heading', { name: '场外公募基金市场镜头' })).toBeVisible()
})

test('三档布局无页面水平溢出且 KPI 响应式排列', async ({ page }, testInfo) => {
  const layout = await page.evaluate(() => ({
    overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    offenders: Array.from(document.querySelectorAll<HTMLElement>('body *'))
      .filter((element) => element.getBoundingClientRect().right > window.innerWidth + 1)
      .slice(0, 12)
      .map((element) => ({
        tag: element.tagName,
        className: element.className,
        right: Math.round(element.getBoundingClientRect().right),
        scrollWidth: element.scrollWidth,
        clientWidth: element.clientWidth,
      })),
  }))
  expect(layout.overflow, JSON.stringify(layout.offenders)).toBeLessThanOrEqual(1)

  const first = await page.getByLabel('市场关键指标').locator(':scope > div').nth(0).boundingBox()
  const second = await page.getByLabel('市场关键指标').locator(':scope > div').nth(1).boundingBox()
  expect(first).not.toBeNull()
  expect(second).not.toBeNull()
  if (testInfo.project.name === 'mobile-320') {
    expect(second!.y).toBeGreaterThan(first!.y)
  } else if (testInfo.project.name === 'tablet-768') {
    expect(Math.abs(second!.y - first!.y)).toBeLessThanOrEqual(1)
  } else {
    const etfLens = await page.getByRole('heading', { name: 'ETF市场镜头' }).locator('..').locator('..').boundingBox()
    const fundLens = await page.getByRole('heading', { name: '场外公募基金市场镜头' }).locator('..').locator('..').boundingBox()
    expect(etfLens).not.toBeNull()
    expect(fundLens).not.toBeNull()
    expect(Math.abs(etfLens!.y - fundLens!.y)).toBeLessThanOrEqual(1)
  }
})

test('范围页签支持方向键且驾驶舱不展示数据下载模块', async ({ page }) => {
  const allTab = page.getByRole('tab', { name: '全市场' })
  await allTab.focus()
  await allTab.press('ArrowRight')
  await expect(page).toHaveURL(/kind=etf/)
  await expect(page.getByRole('tab', { name: 'ETF' })).toHaveAttribute('aria-selected', 'true')

  await expect(page.getByRole('button', { name: /数据管理：查看明细并拉取最新数据/ })).toHaveCount(0)
})
