import { expect, test } from '@playwright/test'
import { cppAotAudit } from '../src/test/cppAotFixture'
import { auditTextContrast } from './helpers/contrast'

// Browser acceptance of the response contract. Native math is covered by the
// installed-wheel backend integration tests, not these synthetic UI fixtures.
for (const valid of [true, false]) {
  test(`C++ AOT ${valid ? 'accepted' : 'rejected'} at the product comparison gate`, async ({ page }, info) => {
    await page.route('**/api/**', async route => {
      const url = new URL(route.request().url())
      if (url.pathname.endsWith('/compare-analysis')) {
        const range = {
          window: { start_date: '2026-01-02', end_date: '2026-01-06', observation_count: 3 },
          metrics: { cumulativeReturn: 6.67, annualizedReturn: 18.5, volatility: 12.3,
            maxDrawdown: -4.2, returnToFee: 11.12, totalFee: .6, sharpeRatio: 1.5, calmarRatio: 4.4 },
          normalized_nav: [{ date: '2026-01-02', value: 1 }, { date: '2026-01-06', value: 1.0667 }],
          drawdown: [], rolling_volatility: [],
        }
        return route.fulfill({ json: {
          schema_version: 1, product_id: url.pathname.split('/')[4],
          ranges: { performance: range, risk: range, efficiency: range },
          execution: { ...cppAotAudit, python_fallback: valid ? 0 : 1 },
        } })
      }
      if (url.pathname.startsWith('/api/instruments/products/')) {
        return route.fulfill({ json: {
          product_id: url.pathname.split('/').at(-1), name: '合成测试产品', management: '测试管理人',
          status: '上市', base_info: {}, metrics: { issue_amount: 100, m_fee: .5, c_fee: .1 },
          timeseries: [{ date: '2026-01-02', close: 3 }, { date: '2026-01-05', close: 3.1 }, { date: '2026-01-06', close: 3.2 }],
        } })
      }
      if (url.pathname.endsWith('/meta')) return route.fulfill({ json: { periods: [] } })
      return route.fulfill({ json: { items: [], total: 0 } })
    })
    await page.goto('/product-compare?kind=etf&ids=510300.SH,159915.SZ')
    if (valid) {
      const proof = page.getByText('数值引擎：C++ AOT', { exact: true })
      await expect(proof).toBeVisible()
      await proof.click()
      await expect(page.getByText('原生预编译：', { exact: true })).toBeVisible()
      await expect(page.getByText(cppAotAudit.plan_fingerprint, { exact: true })).toBeVisible()
    } else {
      await expect(page.getByText(/未通过 C\+\+ AOT 执行校验/)).toBeVisible()
      await expect(page.getByText('数值引擎：C++ AOT', { exact: true })).toHaveCount(0)
    }
    await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
    await page.screenshot({ path: info.outputPath('aot-gate.png'), fullPage: true })
  })
}
