import { test, expect, type Page } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import type { EtlDefinition, EtlRun, EtlWorkflow, EtlRunOptions } from '../src/services/etl'

const root = fileURLToPath(new URL('../../', import.meta.url))
const python = process.env.TEST_PYTHON || '/Users/chenjunming/Desktop/myenv_312/bin/python3.12'
const catalog = JSON.parse(execFileSync(python, ['-c', 'import json,tempfile; from pathlib import Path; from backend.data_sources.store import SourceStore; from backend.data_sources.service import catalog; tmp=tempfile.TemporaryDirectory(); print(json.dumps(catalog(SourceStore(Path(tmp.name))),ensure_ascii=False)); tmp.cleanup()'], { cwd: root, encoding: 'utf8' }))

async function fixture(page: Page, initialRuns: EtlRun[] = []) {
  const submitted: EtlDefinition[] = []
  const runOptions: EtlRunOptions[] = []
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
      runOptions.push(body.options)
      const run: EtlRun = { run_id: 'browser-run', name: body.definition.name, created_at: new Date().toISOString(), updated_at: new Date().toISOString(), status: 'SUCCEEDED', attempt: 1, published: false, steps: body.definition.steps.map((s: {id:string; name:string; kind:string}) => ({ ...s, status: 'SUCCEEDED', rows: 4 })) }
      runs = [run]; return route.fulfill({ json: run })
    }
    if (path === '/api/data-sources/etl/runs') return route.fulfill({ json: runs })
    if (path.endsWith('/recovery') && method === 'POST') {
      runs = runs.map(run => ({ ...run, status: 'SUCCEEDED', attempt: 2, error: undefined, steps: run.steps.map(s => ({ ...s, status: 'SUCCEEDED' })) }))
      return route.fulfill({ status: 202, json: {id:'recovery',source_run_id:runs[0].run_id,target_run_id:runs[0].run_id,status:'SUCCEEDED',phase:'恢复完成',message:'已恢复',created_at:'',updated_at:'',logs:[]} })
    }
    return route.fulfill({ status: 404, json: { detail: { message: 'Offline fixture' } } })
  })
  return { submitted, runOptions }
}

test('download page selects a source and executes full or incremental ETL with explicit scope', async ({ page }, info) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message))
  const { submitted, runOptions } = await fixture(page)
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
  await expect(page.getByRole('heading', { name: '下载任务' })).toBeVisible()
  expect(submitted).toHaveLength(1)
  expect(submitted[0].steps.map(s => s.kind)).toEqual(['download', 'map', 'resolve'])
  expect(submitted[0].steps[0]).toMatchObject({ source_id: 'akshare', mode: 'inherit', params: { symbol: '000001', start_date: '20240101', end_date: '20240110' } })
  expect(runOptions[0]).toEqual({ mode: 'full', parameters: {} })
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
  await page.getByText('流程名称、说明与运行参数', { exact: true }).click()
  await page.getByLabel('流程名称', { exact: true }).fill('每日基金更新')
  await expect(page.getByRole('region', { name: 'ETL 流程编辑器' })).toBeVisible()
  await expect(page.getByTestId('etl-graph-canvas')).toContainText('3 节点')
  await page.getByRole('button', { name: '节点库', exact: true }).click()
  await page.getByRole('button', { name: '＋下载原始数据', exact: true }).click()
  await expect(page.getByTestId('etl-graph-canvas')).toContainText('4 节点')
  await page.getByRole('button', { name: '撤销', exact: true }).click()
  await expect(page.getByTestId('etl-graph-canvas')).toContainText('3 节点')
  await page.getByRole('button', { name: '重做', exact: true }).click()
  const last = page.getByRole('complementary', { name: 'ETL 节点检查器' })
  await last.getByLabel('数据源', { exact: true }).selectOption('tushare')
  await last.getByLabel('下载数据', { exact: true }).selectOption('tushare.trade_cal')
  const upstream = await last.getByLabel('添加仅执行顺序', { exact: true }).locator('option').evaluateAll(options => options.find(option => (option as HTMLOptionElement).text.includes('取值'))?.getAttribute('value'))
  await last.getByLabel('添加仅执行顺序', { exact: true }).selectOption(upstream!)
  await last.getByRole('button', { name: '收起检查器' }).click()
  await page.getByRole('button', { name: '校验流程', exact: true }).click()
  await expect(page.getByText('流程校验通过；不代表已下载或已发布。')).toBeVisible()
  await page.getByRole('button', { name: '保存流程', exact: true }).click()
  await expect(page.getByText('流程已保存，可反复执行；保存不会启动下载。')).toBeVisible()
  await expect(page.getByLabel('已保存流程')).toContainText('每日基金更新')
  await page.getByRole('button', { name: '节点库', exact: true }).click()
  await page.getByRole('button', { name: '＋指标快照计算', exact: true }).click()
  const snapshotInspector = page.getByRole('complementary', { name: 'ETL 节点检查器' })
  const resolved = await snapshotInspector.getByLabel('添加取值结果', { exact: true }).locator('option').evaluateAll(options => options.find(option => !(option as HTMLOptionElement).disabled && (option as HTMLOptionElement).value)?.getAttribute('value'))
  await snapshotInspector.getByLabel('添加取值结果', { exact: true }).selectOption(resolved!)
  await snapshotInspector.getByRole('button', { name: '收起检查器' }).click()
  await page.getByRole('button', { name: '校验流程', exact: true }).click()
  await expect(page.getByRole('alert').last()).toContainText('产品信息和基金净值')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: info.outputPath('etl-workflow-editor.png') })
  expect(errors).toEqual([])
})

test('saved workflow supports both run modes without resaving its definition', async ({ page }) => {
  const { submitted, runOptions } = await fixture(page)
  page.on('dialog', dialog => dialog.accept())
  await page.goto('/settings/data-sources')
  await page.getByLabel('下载数据源', { exact: true }).selectOption('akshare')
  await page.getByRole('checkbox', { name: '公募基金单位净值', exact: true }).check()
  await page.getByRole('button', { name: '转为 ETL 流程编辑' }).click()
  await page.getByRole('button', { name: '保存流程', exact: true }).click()
  await expect(page.getByText('流程已保存，可反复执行；保存不会启动下载。')).toBeVisible()
  const savedId = await page.getByLabel('已保存流程', { exact: true }).inputValue()
  for (const mode of ['full', 'incremental']) {
    await page.getByRole('button', { name: 'ETL 任务编排', exact: true }).click()
    await page.getByLabel('本次运行模式', { exact: true }).selectOption(mode)
    await page.getByRole('button', { name: '确认并运行流程', exact: true }).click()
    await expect(page.getByRole('heading', { name: '下载任务' })).toBeVisible()
  }
  expect(runOptions.map(value => value.mode)).toEqual(['full', 'incremental'])
  expect(submitted[0]).toEqual(submitted[1])
  await page.getByRole('button', { name: 'ETL 任务编排', exact: true }).click()
  await expect(page.getByLabel('已保存流程', { exact: true })).toHaveValue(savedId)
  await expect(page.getByLabel('已保存流程', { exact: true })).toContainText('v1')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
})

test('failed ETL resumes only unfinished steps without creating a new download', async ({ page }) => {
  const run: EtlRun = { run_id: 'failed-run', name: '基金更新', status: 'FAILED', attempt: 1, created_at: '2026-09-01T08:00:00Z', updated_at: '2026-09-01T08:00:00Z', published: false, error: '取值步骤失败', steps: [{ id:'d',name:'下载净值',kind:'download',status:'SUCCEEDED',rows:4 },{ id:'r',name:'取值',kind:'resolve',status:'FAILED' }] }
  const { submitted } = await fixture(page, [run])
  page.on('dialog', d => d.accept())
  await page.goto('/settings/data-sources')
  await page.getByRole('button', { name: '运行记录与恢复', exact: true }).click()
  await page.getByRole('button', { name: '恢复下载' }).click()
  await page.getByRole('button', { name: '确认继续', exact: true }).click()
  await expect(page.getByText(/已完成 2 \/ 2 个步骤/)).toBeVisible()
  await expect(page.getByTestId('etl-task-card')).toHaveCount(1)
  expect(submitted).toHaveLength(0)
})

test('long interrupted flow shows resume loading and rejection beside its button', async ({ page }, info) => {
  const run: EtlRun = { run_id:'interrupted-run',name:'长流程下载',status:'INTERRUPTED',attempt:1,created_at:'2026-09-07T00:00:00Z',updated_at:'2026-09-07T00:00:00Z',published:false,
    steps:Array.from({length:31},(_,i)=>({id:`s${i}`,name:`数据集 ${i+1}`,kind:'task',status:i<10?'SUCCEEDED':i===10?'INTERRUPTED':'PENDING'})) }
  await fixture(page,[run])
  let release!:()=>void
  let calls=0
  await page.route('**/api/data-sources/etl/runs/interrupted-run/recovery',async route=>{
    calls++
    await new Promise<void>(resolve=>{release=resolve})
    return route.fulfill({status:409,json:{detail:{code:'DATA_TASK_RUNNING',message:'后台仍有下载进程持锁，不能重复启动。'}}})
  })
  await page.goto('/settings/data-sources')
  await page.getByRole('button',{name:'运行记录与恢复',exact:true}).click()
  const actions=page.getByRole('region',{name:'长流程下载运行操作'})
  await actions.getByRole('button',{name:'恢复下载'}).click()
  await expect(actions.getByRole('group',{name:'确认恢复任务'})).toBeVisible()
  expect(calls).toBe(0)
  await actions.getByRole('button',{name:'确认继续',exact:true}).click()
  await expect(actions.getByRole('status')).toContainText('正在核验执行版本')
  await expect(actions.getByRole('button',{name:'正在恢复…'})).toBeDisabled()
  await expect.poll(()=>calls).toBe(1)
  release()
  await expect(actions.getByRole('alert')).toBeVisible()
  await expect(actions.getByRole('alert')).toContainText('不能重复启动')
  await actions.screenshot({path:info.outputPath('resume-inline-error.png')})
  run.recovery={can_resume:false,artifact_check_pending:false,blockers:[{code:'DATA_TASK_RUNNING',message:'后台仍有进程持锁。'},{code:'ETL_IMPLEMENTATION_CHANGED',message:'执行程序已更新，旧任务不能原地续跑。'}]}
  await expect(actions.getByRole('button',{name:'恢复下载'})).toBeEnabled({timeout:8000})
  expect(calls).toBe(1)
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true)
})

test('recovery remains clickable after code changes, survives refresh, reports failures and links successor', async ({ page }, info) => {
  const old: EtlRun = {run_id:'old',name:'概念行情恢复',status:'FAILED',attempt:1,created_at:'2026-09-09T00:00:00Z',updated_at:'',published:false,steps:[],
    recovery:{can_resume:false,artifact_check_pending:false,blockers:[{code:'ETL_IMPLEMENTATION_CHANGED',message:'执行版本变化，需要迁移'}]}}
  await fixture(page, [old])
  let requests = 0
  await page.route('**/api/data-sources/etl/runs/old/recovery', async route => {
    requests++
    old.recovery!.job = {id:'job'+requests,source_run_id:'old',target_run_id:'old',status:'RUNNING',phase:'校验与迁移',message:'正在核验 GB 旧文件',created_at:'',updated_at:'',logs:[]}
    await route.fulfill({status:202,json:old.recovery!.job})
  })
  const actions = page.getByRole('region',{name:'概念行情恢复运行操作'})
  await page.goto('/settings/data-sources')
  await page.getByRole('button',{name:'运行记录与恢复',exact:true}).click()
  await expect(actions.getByRole('button',{name:'恢复下载',exact:true})).toBeEnabled()
  expect(requests).toBe(0)
  await actions.getByRole('button',{name:'恢复下载',exact:true}).click()
  await actions.getByRole('button',{name:'确认继续',exact:true}).click()
  await expect(actions.getByRole('progressbar')).toBeVisible()
  await expect(actions.getByRole('button',{name:'正在恢复…'})).toBeDisabled()
  await page.reload()
  await page.getByRole('button',{name:'运行记录与恢复',exact:true}).click()
  await expect(actions.getByText(/正在核验 GB 旧文件/)).toBeVisible()
  expect(requests).toBe(1)
  old.recovery!.job!.status = 'FAILED'; old.recovery!.job!.message = '部分文件校验不通过，原数据保留'
  await expect(actions.getByRole('alert')).toContainText('恢复下载失败', {timeout:8000})
  await expect(actions.getByRole('button',{name:'恢复下载',exact:true})).toBeEnabled()
  await actions.screenshot({path:info.outputPath('recovery-failure.png')})
  old.recovery!.successor = {run_id:'new',name:'后续任务',status:'RUNNING'}
  const next: EtlRun = {...old,run_id:'new',name:'后续任务',status:'RUNNING',recovery:undefined,steps:[{id:'a',name:'下载概念行情',kind:'task',status:'RUNNING'}]}
  await page.route('**/api/data-sources/etl/runs/new', route => route.fulfill({json:next}))
  await expect(actions.getByRole('button',{name:'查看后续任务进度'})).toBeVisible({timeout:8000})
  await actions.getByRole('button',{name:'查看后续任务进度'}).click()
  await expect(page.locator('#etl-run-new')).toBeVisible()
  await expect(page.locator('#etl-run-new > summary')).toBeFocused()
  expect(requests).toBe(1)
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true)
})
