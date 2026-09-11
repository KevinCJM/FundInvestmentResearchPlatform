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
  await page.getByText('采集基线与披露修订策略', { exact: true }).click()
  await expect(page.getByLabel('修订复核间隔（天）')).toHaveValue('7')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
  await page.getByRole('button', { name: '分析快照并预览区间' }).click()
  await expect(page.getByRole('region', { name: '自动增量计划' })).toContainText('2026-08-28 至 2026-09-06')
  await expect(page.getByRole('button', { name: '确认计划并开始自动增量' })).toBeEnabled()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
  await page.screenshot({ path: info.outputPath('auto-plan.png'), fullPage: true })
  await page.getByRole('button', { name: '确认计划并开始自动增量' }).click()
  await expect(page.getByRole('heading', { name: '下载任务' })).toBeVisible()
  expect(submitted).toHaveLength(1)
  expect(submitted[0]).toMatchObject({ options: { mode: 'auto_incremental', parameters: {} }, auto_plan_id: 'snapshot-plan' })
  await page.getByText('查看本次自动增量区间与快照依据').click()
  await expect(page.getByRole('region', { name: '自动增量计划' })).toContainText('tushare_snapshot_fixture')
  expect(errors).toEqual([])
})

test('blocked automatic download is disabled until candidate selection is revalidated', async ({ page }, info) => {
  const requests: unknown[] = []
  const candidate = 'a'.repeat(32)
  await page.route('**/api/**', route => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/data-sources/catalog') return route.fulfill({ json: { ...catalog, editing_enabled: true } })
    if (path === '/api/data-sources/etl/workflows' || path === '/api/data-sources/etl/runs') {
      if (route.request().method() === 'POST') requests.push(route.request().postDataJSON())
      return route.fulfill({ json: [] })
    }
    if (path === '/api/data-sources/etl/validate') {
      const selected = route.request().postDataJSON().options?.auto_baseline_run_id === candidate
      const errors = selected ? [] : [{ code: 'AUTO_BASELINE_NOT_ACTIVE', step_id: 'auto_2', message: '基金净值已有完成下载，尚未启用。' }]
      return route.fulfill({ json: { valid: selected, errors, steps: [], auto_plan: { ...plan, ready: selected, errors,
        supplemental_baseline: { run_id: selected ? candidate : null, files: selected ? ['fund_nav_df.parquet'] : [] },
        baseline_choices: [{ run_id: candidate, name: '历史完整下载', finished_at: '2026-09-09T00:00:00Z', files: [{ name: 'fund_nav_df.parquet', rows: 100, latest_date: '2026-09-03' }] }],
      } } })
    }
    return route.fulfill({ status: 404, json: {} })
  })
  await page.goto('/settings/data-sources?downloadView=auto')
  const start = page.getByRole('button', { name: '确认计划并开始自动增量' })
  await expect(start).toBeDisabled()
  await page.getByLabel('场外公募基金净值', { exact: true }).check()
  await page.getByRole('button', { name: '分析快照并预览区间' }).click()
  await expect(page.getByText('基金净值已有完成下载，尚未启用。')).toBeVisible()
  await expect(start).toBeDisabled()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
  await page.screenshot({ path: info.outputPath('blocked-plan.png'), fullPage: true })
  await page.getByLabel('补足缺失基线').selectOption(candidate)
  await expect(start).toBeDisabled()
  await page.getByRole('button', { name: '分析快照并预览区间' }).click()
  await expect(start).toBeEnabled()
  expect(requests).toEqual([])
})
