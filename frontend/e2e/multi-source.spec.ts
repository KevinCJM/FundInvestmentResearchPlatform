import { test, expect } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../../', import.meta.url))
const python = process.env.TEST_PYTHON || '/Users/chenjunming/Desktop/myenv_312/bin/python3.12'
const fixture = JSON.parse(execFileSync(python, ['-c', 'import json,tempfile; from pathlib import Path; from backend.data_sources.store import SourceStore; from backend.data_sources.service import catalog; from backend.data_sources.resolution_store import get_policy; t=tempfile.TemporaryDirectory(); s=SourceStore(Path(t.name)); c=catalog(s); print(json.dumps({"catalog":c,"policy":get_policy(s)},ensure_ascii=False)); t.cleanup()'], { cwd: root, encoding: 'utf8' }))

test('AKShare uses the same editable interface and bounded download workflow', async ({ page }, testInfo) => {
  const errors: string[] = []
  let download: Record<string, unknown> | undefined
  page.on('pageerror', error => errors.push(error.message))
  page.on('dialog', dialog => dialog.accept())
  let jobs: object[] = []
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/data-sources/catalog') return route.fulfill({ json: fixture.catalog })
    if (path === '/api/data-sources/sync/jobs') return route.fulfill({ json: jobs })
    if (path === '/api/data-sources/interfaces/akshare.etf_daily/sync') {
      download = route.request().postDataJSON()
      jobs = [{ job_id: 'fixture', interface_id: 'akshare.etf_daily', source_id: 'akshare', status: 'SUCCEEDED', mode: 'incremental', rows: 4, pages: 1, published: false, message: '下载完成；标准候选未发布。' }]
      return route.fulfill({ json: jobs[0] })
    }
    return route.fulfill({ status: 404, json: { detail: 'Offline fixture' } })
  })
  await page.goto('/settings/source-center?source=akshare&interface=akshare.etf_daily')
  await expect(page.getByRole('heading', { name: 'ETF 日行情', exact: true })).toBeVisible()
  await expect(page.getByLabel('代码标准化')).toHaveValue('cn_etf_code')
  await page.getByRole('button', { name: '1. 接口结构' }).click()
  await expect(page.getByLabel('接口 API 名称')).toBeEnabled()
  await expect(page.getByLabel('接口 API 名称')).toHaveValue('fund_etf_hist_em')
  await page.getByText('5. 下载与更新此接口', { exact: true }).click()
  await page.getByLabel('产品代码', { exact: true }).fill('510300')
  await page.getByLabel('下载开始日期').fill('2024-01-02')
  await page.getByLabel('下载结束日期').fill('2024-01-05')
  await page.getByRole('button', { name: '开始此接口下载' }).click()
  await expect(page.getByRole('region', { name: '接口下载结果' })).toContainText('标准候选未发布')
  expect(download).toMatchObject({ expected_revision: 1, mode: 'incremental', confirm: true, params: { symbol: '510300', start_date: '20240102', end_date: '20240105' } })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: testInfo.outputPath('akshare-download.png') })
  expect(errors).toEqual([])
})

test('source priority changes are saved before candidate arbitration', async ({ page }, testInfo) => {
  let policy = structuredClone(fixture.policy)
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/data-sources/catalog') return route.fulfill({ json: fixture.catalog })
    if (path === '/api/data-sources/resolution/config') {
      if (route.request().method() === 'PUT') {
        const body = route.request().postDataJSON()
        expect(body.expected_revision).toBe(1)
        expect(body.config.default_source_priority).toEqual(['akshare', 'tushare'])
        policy = { ...policy, config: body.config, revision: 2 }
      }
      return route.fulfill({ json: policy })
    }
    if (path === '/api/data-sources/resolution/run') {
      expect(route.request().postDataJSON().expected_revision).toBe(2)
      return route.fulfill({ json: { table_id: 'market.quote_daily', published: false, status: 'CANDIDATE_READY', summary: { selected_rows: 1, FALLBACK: 1 }, decisions: [{ key: { instrument_id: 'fixture', trade_date: '2024-01-02' }, status: 'FALLBACK', selected_source: 'tushare', skipped: [{ source_id: 'akshare', reasons: ['VALUE_OUT_OF_RANGE'] }], conflicts: [] }] } })
    }
    return route.fulfill({ status: 404, json: { detail: 'Offline fixture' } })
  })
  await page.goto('/settings/source-center')
  await page.getByRole('button', { name: '多源取值规则', exact: true }).click()
  await page.getByRole('button', { name: '全局来源优先级 akshare 上移' }).click()
  await page.getByLabel(/主来源异常时使用下一来源/).check()
  await expect(page.getByRole('button', { name: '应用规则到已下载候选' })).toBeDisabled()
  await page.getByRole('button', { name: '保存取值规则' }).click()
  await expect(page.getByText(/规则修订 2/)).toBeVisible()
  await page.getByRole('button', { name: '应用规则到已下载候选' }).click()
  await expect(page.getByRole('region', { name: '多源取值结果' })).toContainText('备用替代 1 行')
  await expect(page.getByRole('heading', { name: '取值结果 · 未发布' })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  await page.getByRole('heading', { name: '多源优先级与异常处理' }).scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('source-priority.png') })
  expect(errors).toEqual([])
})
