import { expect, test } from '@playwright/test'
import { auditTextContrast } from './helpers/contrast'
import { allStages } from '../src/app/processRegistry'

/**
 * 文字对比度只能在渲染结果上量。祖先的底色常写在三元里，
 * scripts/check_frontend_design.mjs 的 same-element-contrast 静态判定不了那一类，
 * 本用例负责补上：逐个元素向上找到真正生效的底色再算。
 *
 * 覆盖边界（不要把这份用例说成完整的 WCAG AA 验收）：
 * - 只量初始渲染状态。页签切换、展开、悬停、错误态后的颜色不在内。
 * - 渐变按声明出来的色标估算最坏值，不是逐像素采样；背景图与视频底不量。
 * - 本文件将 API 固定为离线错误态，只覆盖该状态的外壳；不能代表有数据的表格。
 * - 指标工作台的正常载入和页签切换另由 indicator-studio.spec.ts 使用固定夹具检查。
 * - 下方独立的产品象限用例覆盖正常数据下的 HTML tooltip 悬停状态。
 *
 * 路由不手写。手写过一次，把 /portfolio-center/onboarding 写成了不存在的路径，
 * 页面跳回首页而用例照样通过。改成从 processRegistry 派生，并断言落地路由。
 */
const ROUTES = ['/', ...new Set(allStages.flatMap((stage) => [stage.path, ...stage.nodes.map((node) => node.path)]))]


// 所有外壳测试都使用固定离线错误态，禁止请求开发者的真实数据服务。
test.beforeEach(async ({ page }) => {
  await page.route('**/api/**', route => route.fulfill({ status: 503, json: { detail: 'Offline contrast fixture' } }))
})

for (const route of ROUTES) {
  test(`${route} 的初始可见文字对比度检查`, async ({ page }) => {
    await page.goto(route)
    await page.waitForLoadState('networkidle')
    // 路由写错时页面会跳回首页，量到的是首页而不是目标页。
    expect(new URL(page.url()).pathname, `${route} 没有落在目标路由上`).toBe(route)
    const failures = await page.evaluate(auditTextContrast)
    expect(failures, failures.map((f) => `${f.ratio}:1 «${f.text}» ${f.color} — ${f.path}`).join('\n')).toEqual([])
  })
}

test('对比度算法覆盖可见 aria-hidden、组透明度及真实显示的占位符', async ({ page }) => {
  await page.setContent(`<style>body{background:white;color:black;font:14px sans-serif}input{background:white;color:black}input::placeholder{color:black;opacity:.5}#filled::placeholder{color:#eee}</style>
    <p aria-hidden="true" style="color:#ddd">可见但不读屏</p>
    <div style="opacity:.5;background:black;color:white"><span>半透明深色组</span></div>
    <input placeholder="需要检查的占位符" />
    <input id="filled" value="已有内容" placeholder="没有显示的占位符" />
    <button disabled style="color:#eee">禁用</button>
    <p style="opacity:0">完全透明</p>`)
  const failures = await page.evaluate(auditTextContrast)
  expect(failures.map(item => item.text).sort()).toEqual(['可见但不读屏', '半透明深色组', '占位符：需要检查的占位符'].sort())
  expect(failures.find(item => item.text === '半透明深色组')!.ratio).toBeCloseTo(3.98, 1)
})

test('产品收益风险象限的悬浮提示在深色背景上保持可读', async ({ page }, testInfo) => {
  const dates = ['2026-01-02', '2026-01-05', '2026-01-06']
  const range = {
    window: { start_date: dates[0], end_date: dates[2], observation_count: 3 },
    metrics: { cumulativeReturn: 6.67, annualizedReturn: 18.5, volatility: 12.3, maxDrawdown: -4.2, returnToFee: 11.12, totalFee: 0.6, sharpeRatio: 1.5, calmarRatio: 4.4 },
    normalized_nav: dates.map((date, i) => ({ date, value: 1 + i / 30 })),
    drawdown: dates.map(date => ({ date, value: 0 })),
    rolling_volatility: dates.map((date, i) => ({ date, value: i < 2 ? null : 12.3 })),
  }
  await page.route('**/api/**', route => {
    const path = new URL(route.request().url()).pathname
    const product = path.match(/^\/api\/instruments\/products\/(510300\.SH|159915\.SZ)(\/compare-analysis)?$/)
    if (product) {
      const id = product[1]
      return route.fulfill({ json: product[2] ? {
        schema_version: 1, product_id: id, ranges: { performance: range, risk: range, efficiency: range },
        execution: {
          backend: 'numba_njit_fixed_signature', execution_backend: 'numba_njit_fixed_signature',
          engine: 'offline-fixture', kernel_version: 'offline-fixture', kernel_coverage: '31/31',
          kernel_signatures: { product_compare_analysis_kernel: ['fixed-signature'] }, kernel_fingerprint: 'offline-fixture',
          nopython: true, object_mode: 0, njit_required: true, python_fallback: 0, request_time_compilation: 0,
        },
      } : {
        product_id: id, name: id === '510300.SH' ? '沪深300ETF' : '创业板ETF', management: '离线测试管理人', status: '上市',
        base_info: { ts_code: id, fund_type: 'ETF' }, metrics: { issue_amount: 100, m_fee: 0.5, c_fee: 0.1, exp_return: 8 },
        timeseries: dates.map((date, i) => ({ date, close: 3 + i / 10 })),
      } })
    }
    if (path === '/api/custom-indicators/meta') return route.fulfill({ json: { periods: [] } })
    if (path === '/api/custom-indicators') return route.fulfill({ json: { items: [], total: 0 } })
    return route.fulfill({ status: 503, json: { detail: 'Offline contrast fixture' } })
  })
  await page.goto('/product-research/compare?kind=etf&ids=510300.SH,159915.SZ')
  const chart = page.getByRole('heading', { name: '收益风险象限图', exact: true }).locator('xpath=ancestor::div[3]').locator('.echarts-for-react')
  await expect(chart).toBeVisible()
  await chart.scrollIntoViewIfNeeded()
  const code = chart.locator('span').filter({ hasText: /^(510300\.SH|159915\.SZ)$/ })
  // 两只产品使用相同的离线指标，散点位于图形区域中心；真实鼠标悬停触发 HTML tooltip。
  await expect(async () => {
    const box = (await chart.boundingBox())!
    await page.mouse.move(box.x + (60 + box.width - 36) / 2, box.y + (64 + box.height - 56) / 2)
    await expect(code).toBeVisible()
  }).toPass()
  await expect.poll(async () => (await page.evaluate(auditTextContrast)).filter(item => /^(510300\.SH|159915\.SZ)$/.test(item.text))).toEqual([])
  await page.screenshot({ path: testInfo.outputPath('product-quadrant-tooltip.png') })
})
