import { test, expect } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../../', import.meta.url))
const python = process.env.TEST_PYTHON || '/Users/chenjunming/Desktop/myenv_312/bin/python3.12'
const data = JSON.parse(execFileSync(python, ['-c', 'import json,tempfile; from pathlib import Path; from backend.data_sources.store import SourceStore; from backend.data_sources.service import catalog; from backend.data_sources.etl_templates import tushare_all_data_workflow; t=tempfile.TemporaryDirectory(); s=SourceStore(Path(t.name)); s.seed(); print(json.dumps({"catalog":catalog(s),"definition":tushare_all_data_workflow(s).model_dump(mode="json")},ensure_ascii=False)); t.cleanup()'], {cwd:root,encoding:'utf8'}))

test('all-data workflow uses the generic editor without single-product inputs', async ({ page }, info) => {
  const errors: string[] = []
  const writes: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/api/**', route => {
    const path = new URL(route.request().url()).pathname
    if (route.request().method() !== 'GET') writes.push(path)
    if (path === '/api/data-sources/catalog') return route.fulfill({json:data.catalog})
    if (path === '/api/data-sources/etl/runs') return route.fulfill({json:[]})
    if (path === '/api/data-sources/etl/workflows') return route.fulfill({json:[{id:'generic-full-fixture',revision:1,definition:data.definition,updated_at:'2026-09-01'}]})
    if (path === '/api/data-sources/etl/templates') return route.fulfill({json:[{id:'generic-template',name:data.definition.name,description:data.definition.description,definition:data.definition}]})
    return route.fulfill({status:404,json:{detail:{message:'offline fixture'}}})
  })
  await page.goto('/settings/data-sources')
  await page.getByRole('button',{name:'ETL 任务编排',exact:true}).click()
  await page.getByLabel('已保存流程',{exact:true}).selectOption('generic-full-fixture')
  await expect(page.getByTestId('etl-graph-canvas')).toContainText('31 节点')
  await expect(page.getByTestId('etl-graph-mobile-list').locator('article')).toHaveCount(31)
  await expect(page.getByLabel('流程名称',{exact:true})).toHaveValue('Tushare 全数据同步')
  await expect(page.getByLabel('ETF 代码',{exact:true})).toHaveCount(0)
  await expect(page.getByLabel('场外基金代码',{exact:true})).toHaveCount(0)
  const options = page.getByRole('group',{name:'本次运行设置'})
  await expect(options.getByLabel('历史开始日期',{exact:true})).toBeVisible()
  await options.getByLabel('本次运行模式',{exact:true}).selectOption('full')
  await options.getByLabel('本次截止日期',{exact:true}).fill('2026-09-04')
  await page.getByText('执行顺序 · 31 个节点', { exact: true }).click()
  await page.getByRole('button', { name: `1. ${data.definition.steps[0].name}`, exact: true }).click()
  const first = page.getByRole('complementary', { name: 'ETL 节点检查器' })
  await expect(first.getByLabel('任务类型',{exact:true})).toBeEnabled()
  await expect(first.getByLabel('任务数据源',{exact:true})).toHaveValue('tushare')
  await first.getByLabel('数据集更新策略',{exact:true}).selectOption('full')
  await page.getByRole('button', { name: '节点库', exact: true }).click()
  await expect(page.getByRole('button',{name:'＋数据集任务',exact:true})).toBeEnabled()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true)
  await first.scrollIntoViewIfNeeded()
  await page.screenshot({path:info.outputPath('generic-full-data.png')})
  expect(writes).toEqual([])
  expect(errors).toEqual([])
})
