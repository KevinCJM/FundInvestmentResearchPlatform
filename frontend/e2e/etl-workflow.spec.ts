import { test, expect, type Page } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import type { EtlDefinition, EtlRun, EtlWorkflow } from '../src/services/etl'

const root = fileURLToPath(new URL('../../', import.meta.url))
const python = process.env.TEST_PYTHON || '/Users/chenjunming/Desktop/myenv_312/bin/python3.12'
const catalog = JSON.parse(execFileSync(python, ['-c', 'import json,tempfile; from pathlib import Path; from backend.data_sources.store import SourceStore; from backend.data_sources.service import catalog; tmp=tempfile.TemporaryDirectory(); print(json.dumps(catalog(SourceStore(Path(tmp.name))),ensure_ascii=False)); tmp.cleanup()'], { cwd: root, encoding: 'utf8' }))

async function fixture(page: Page, initialRuns: EtlRun[] = []) {
  const submitted: EtlDefinition[] = []
  let runs = initialRuns
  let workflows: EtlWorkflow[] = []
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    const method = route.request().method()
    if (path === '/api/data-sources/catalog') return route.fulfill({ json: catalog })
    if (path === '/api/data-sources/etl/workflows') return route.fulfill({ json: workflows })
    if (path.startsWith('/api/data-sources/etl/workflows/') && method === 'PUT') {
      const body = route.request().postDataJSON()
      const saved = { id: path.split('/').at(-1)!, revision: body.expected_revision + 1, definition: body.definition, updated_at: new Date().toISOString() }
      workflows = [saved]; return route.fulfill({ json: saved })
    }
    if (path === '/api/data-sources/etl/validate') {
      const body = route.request().postDataJSON()
      // Exercise the actual backend plan model in an isolated control store.
      const validation = JSON.parse(execFileSync(python, ['-c', 'import json,sys,tempfile; from pathlib import Path; from backend.data_sources.store import SourceStore; from backend.data_sources.etl_service import validate; t=tempfile.TemporaryDirectory(); s=SourceStore(Path(t.name)); s.seed(); print(json.dumps(validate(s,json.load(sys.stdin)),ensure_ascii=False)); t.cleanup()'], { cwd: root, encoding: 'utf8', input: JSON.stringify(body.definition) }))
      return route.fulfill({ json: validation })
    }
    if (path === '/api/data-sources/etl/runs' && method === 'POST') {
      const body = route.request().postDataJSON()
      submitted.push(body.definition)
      const run: EtlRun = { run_id: 'browser-run', name: body.definition.name, created_at: new Date().toISOString(), updated_at: new Date().toISOString(), status: 'SUCCEEDED', attempt: 1, published: false, steps: body.definition.steps.map((s: {id:string; name:string; kind:string}) => ({ ...s, status: 'SUCCEEDED', rows: 4 })) }
      runs = [run]; return route.fulfill({ json: run })
    }
    if (path === '/api/data-sources/etl/runs') return route.fulfill({ json: runs })
    if (path.endsWith('/resume')) {
      runs = runs.map(run => ({ ...run, status: 'SUCCEEDED', attempt: 2, error: undefined, steps: run.steps.map(s => ({ ...s, status: 'SUCCEEDED' })) }))
      return route.fulfill({ json: runs[0] })
    }
    return route.fulfill({ status: 404, json: { detail: { message: 'Offline fixture' } } })
  })
  return { submitted }
}

test('download page selects a source and executes full or incremental ETL with explicit scope', async ({ page }, info) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message))
  const { submitted } = await fixture(page)
  page.on('dialog', d => d.accept())
  await page.goto('/settings/data-sources')
  await page.getByLabel('下载数据源', { exact: true }).selectOption('akshare')
  await expect(page.getByRole('checkbox', { name: '公募基金单位净值', exact: true })).toBeVisible()
  await expect(page.getByRole('checkbox', { name: '基金公司', exact: true })).toHaveCount(0)
  await page.getByRole('checkbox', { name: '公募基金单位净值', exact: true }).check()
  await page.getByLabel('下载模式', { exact: true }).selectOption('full')
  await page.getByLabel('开始日期', { exact: true }).fill('2024-01-01')
  await page.getByLabel('结束日期', { exact: true }).fill('2024-01-10')
  await expect(page.getByLabel('产品代码（symbol）', { exact: true })).toHaveValue('000001')
  await expect(page.getByRole('checkbox', { name: '最后计算指标快照', exact: true })).toBeDisabled()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  await page.getByRole('button', { name: '确认并开始下载' }).scrollIntoViewIfNeeded()
  await page.screenshot({ path: info.outputPath('etl-quick-download.png') })
  await page.getByRole('button', { name: '确认并开始下载' }).click()
  await expect(page.getByRole('heading', { name: '运行记录与恢复' })).toBeVisible()
  expect(submitted).toHaveLength(1)
  expect(submitted[0].steps.map(s => s.kind)).toEqual(['download', 'map', 'resolve'])
  expect(submitted[0].steps[0]).toMatchObject({ source_id: 'akshare', mode: 'full', params: { symbol: '000001', start_date: '20240101', end_date: '20240110' } })
  await expect(page.getByText(/候选未发布/).first()).toBeVisible()
  expect(errors).toEqual([])
})

test('user composes cross-source steps, controls order, validates and saves a workflow', async ({ page }, info) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message))
  await fixture(page)
  page.on('dialog', d => d.accept())
  await page.goto('/settings/data-sources')
  await page.getByLabel('下载数据源', { exact: true }).selectOption('akshare')
  await page.getByRole('checkbox', { name: '公募基金单位净值', exact: true }).check()
  await page.getByRole('button', { name: '转为 ETL 流程编辑' }).click()
  await page.getByLabel('流程名称').fill('每日基金更新')
  await expect(page.getByRole('region', { name: 'ETL 流程编辑器' })).toBeVisible()
  await page.getByRole('button', { name: '步骤 2 上移' }).click()
  await expect(page.getByRole('alert').first()).toContainText('缺少有效前置结果')
  await page.getByRole('button', { name: '步骤 1 下移' }).click()
  await page.getByRole('button', { name: '＋下载原始数据', exact: true }).click()
  const last = page.getByRole('article').last()
  await last.getByLabel('数据源', { exact: true }).selectOption('tushare')
  await last.getByLabel('下载数据', { exact: true }).selectOption('tushare.trade_cal')
  await page.getByRole('button', { name: '校验流程', exact: true }).click()
  await expect(page.getByText('流程校验通过；不代表已下载或已发布。')).toBeVisible()
  await page.getByRole('button', { name: '保存流程', exact: true }).click()
  await expect(page.getByText('流程已保存，可反复执行；保存不会启动下载。')).toBeVisible()
  await expect(page.getByLabel('已保存流程')).toContainText('每日基金更新')
  await page.getByRole('button', { name: '＋指标快照计算', exact: true }).click()
  await page.getByRole('button', { name: '校验流程', exact: true }).click()
  await expect(page.getByRole('alert').last()).toContainText('产品信息和基金净值')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: info.outputPath('etl-workflow-editor.png') })
  expect(errors).toEqual([])
})

test('failed ETL resumes only unfinished steps without creating a new download', async ({ page }) => {
  const run: EtlRun = { run_id: 'failed-run', name: '基金更新', status: 'FAILED', attempt: 1, created_at: '2026-09-01T08:00:00Z', updated_at: '2026-09-01T08:00:00Z', published: false, error: '取值步骤失败', steps: [{ id:'d',name:'下载净值',kind:'download',status:'SUCCEEDED',rows:4 },{ id:'r',name:'取值',kind:'resolve',status:'FAILED' }] }
  const { submitted } = await fixture(page, [run])
  page.on('dialog', d => d.accept())
  await page.goto('/settings/data-sources')
  await page.getByRole('button', { name: '运行记录与恢复', exact: true }).click()
  await page.getByRole('button', { name: '继续未完成步骤' }).click()
  await expect(page.getByText(/运行第 2 次尝试/)).toBeVisible()
  expect(submitted).toHaveLength(0)
})
