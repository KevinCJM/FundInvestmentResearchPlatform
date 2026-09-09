import { test, expect } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../../', import.meta.url))
const python = process.env.TEST_PYTHON || '/Users/chenjunming/Desktop/myenv_312/bin/python3.12'
const catalog = JSON.parse(execFileSync(python, ['-c', 'import json,tempfile; from pathlib import Path; from backend.data_sources.store import SourceStore; from backend.data_sources.service import catalog; t=tempfile.TemporaryDirectory(); s=SourceStore(Path(t.name)); s.seed(); print(json.dumps(catalog(s),ensure_ascii=False)); t.cleanup()'], { cwd: root, encoding: 'utf8' }))
const plan = { plan_id: 'snapshot-plan', snapshot: 'tushare_snapshot_fixture', cutoff_date: '2026-09-06', lookback_trade_days: 5, ready: true, errors: [],
  steps: [{ id: 'auto_2', name: '场外公募基金净值', strategy: 'incremental', latest_date: '2026-09-03', start_date: '20260828', end_date: '20260906', message: '回查最近 5 个交易日，补充新增日期。' }] }

test('automatic incremental previews dates and submits only the confirmed frozen plan', async ({ page }, info) => {
  const errors: string[] = [], submitted: unknown[] = []
  let runs: object[] = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('dialog', dialog => dialog.accept())
  await page.route('**/api/**', route => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/data-sources/catalog') return route.fulfill({ json: { ...catalog, editing_enabled: true } })
    if (path === '/api/data-sources/etl/workflows') return route.fulfill({ json: [] })
    if (path === '/api/data-sources/etl/validate') return route.fulfill({ json: { valid: true, errors: [], steps: [], auto_plan: plan } })
    if (path === '/api/data-sources/etl/runs' && route.request().method() === 'POST') {
      const body = route.request().postDataJSON(); submitted.push(body)
      const run = { run_id: 'offline-run', name: '自动增量更新', status: 'RUNNING', created_at: '2026-09-07T05:00:00Z', updated_at: '', attempt: 1, published: false, options: body.options, auto_plan: plan, steps: [] }
      runs = [run]; return route.fulfill({ json: run })
    }
    if (path === '/api/data-sources/etl/runs') return route.fulfill({ json: runs })
    return route.fulfill({ status: 404, json: { detail: { message: 'offline fixture' } } })
  })
  await page.goto('/settings/data-sources')
  await page.getByRole('button', { name: '自动增量（无需日期）', exact: true }).click()
  await page.getByLabel('场外公募基金净值', { exact: true }).check()
  await expect(page.locator('input[type="date"]')).toHaveCount(0)
  await page.getByRole('button', { name: '分析快照并预览区间' }).click()
  await expect(page.getByRole('region', { name: '自动增量计划' })).toContainText('2026-08-28 至 2026-09-06')
  await expect(page.getByRole('button', { name: '确认计划并开始自动增量' })).toBeEnabled()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
  await page.screenshot({ path: info.outputPath('auto-plan.png'), fullPage: true })
  await page.getByRole('button', { name: '确认计划并开始自动增量' }).click()
  await expect(page.getByRole('heading', { name: '运行记录与恢复' })).toBeVisible()
  expect(submitted).toHaveLength(1)
  expect(submitted[0]).toMatchObject({ options: { mode: 'auto_incremental', parameters: {} }, auto_plan_id: 'snapshot-plan' })
  await page.getByText('查看本次自动增量区间与快照依据').click()
  await expect(page.getByRole('region', { name: '自动增量计划' })).toContainText('tushare_snapshot_fixture')
  expect(errors).toEqual([])
})
