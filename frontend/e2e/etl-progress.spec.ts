import { test, expect } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../../', import.meta.url))
const python = process.env.TEST_PYTHON || '/Users/chenjunming/Desktop/myenv_312/bin/python3.12'
const catalog = JSON.parse(execFileSync(python, ['-c', 'import json,tempfile; from pathlib import Path; from backend.data_sources.store import SourceStore; from backend.data_sources.service import catalog; t=tempfile.TemporaryDirectory(); s=SourceStore(Path(t.name)); s.seed(); print(json.dumps(catalog(s),ensure_ascii=False)); t.cleanup()'], { cwd: root, encoding: 'utf8' }))

test('cross-date warning permits explicit resume and survives refresh', async ({ page }, info) => {
  let resumeCalls = 0
  const timing = { timezone: 'Asia/Shanghai', first_date: '2026-09-07', last_date: '2026-09-07', cross_date: false,
    warnings: [] as { code: string; message: string }[], boundary: '同日采集也不代表已通过 PIT 校验。', scope: '本流程采集记录，不覆盖增量基线的全部历史批次。',
    windows: [{ run_id: 'original-collection-run', step_id: 'nav', name: '公募基金净值', attempt: 1,
      first_at: '2026-09-07T01:00:00Z', last_at: '2026-09-07T02:00:00Z', basis: 'batch_receipts' }] }
  const run = { run_id: 'clock-fixture', name: '跨日续跑验收', status: 'CANCELLED', created_at: '2026-09-07T01:00:00Z', updated_at: '', attempt: 1, published: false,
    steps: [], collection_timing: timing, recovery: { can_resume: true, artifact_check_pending: true, blockers: [], warnings: [{ code: 'ETL_CROSS_DATE_RESUME', message: '跨日期续跑提醒：已有数据于 2026-09-07 下载，本次于 2026-09-08 继续，PIT 可能不一致；允许继续。' }] } }
  await page.route('**/api/**', route => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/data-sources/catalog') return route.fulfill({ json: { ...catalog, editing_enabled: true } })
    if (path === '/api/data-sources/etl/workflows') return route.fulfill({ json: [] })
    if (path === '/api/data-sources/etl/runs/clock-fixture/recovery') {
      expect(route.request().method()).toBe('POST')
      expect(route.request().postDataJSON()).toEqual({ confirm: true, request_id: expect.any(String) })
      resumeCalls += 1
      run.status = 'RUNNING'; run.attempt = 2
      timing.last_date = '2026-09-08'; timing.cross_date = true
      timing.windows.push({ run_id: 'clock-fixture', step_id: 'nav', name: '公募基金净值', attempt: 2,
        first_at: '2026-09-08T01:00:00Z', last_at: '2026-09-08T01:00:00Z', basis: 'batch_receipts' })
      timing.warnings = [{ code: 'ETL_CROSS_DATE_COLLECTION', message: '采集记录已跨日期，下载内容的时点可能不一致。' }]
      return route.fulfill({ status:202, json: {id:'clock-job',source_run_id:run.run_id,target_run_id:run.run_id,status:'SUCCEEDED',phase:'恢复完成',message:'已恢复',created_at:'',updated_at:'',logs:[]} })
    }
    if (path === '/api/data-sources/etl/runs') return route.fulfill({ json: [run] })
    return route.fulfill({ status: 404, json: {} })
  })
  await page.goto('/settings/data-sources')
  await page.getByRole('button', { name: '运行记录与恢复', exact: true }).click()
  await expect(page.getByText(/跨日期续跑提醒/)).toBeVisible()
  await page.getByRole('button', { name: '恢复下载', exact: true }).click()
  expect(resumeCalls).toBe(0)
  await expect(page.getByRole('group', { name: '确认恢复任务' })).toContainText('不会改变原下载区间')
  await page.getByRole('button', { name: '确认继续', exact: true }).click()
  await expect(page.getByText('采集记录已跨日期，下载内容的时点可能不一致。')).toBeVisible()
  expect(resumeCalls).toBe(1)
  await page.reload()
  await page.getByRole('button', { name: '运行记录与恢复', exact: true }).click()
  await expect(page.getByText('采集记录已跨日期，下载内容的时点可能不一致。')).toBeVisible()
  await page.getByText('查看采集批次时间与续跑记录').click()
  await expect(page.getByRole('table', { name: '节点采集时间范围' })).toContainText('批次接收时间')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
  await page.screenshot({ path: info.outputPath('etl-cross-date-warning.png'), fullPage: true })
})

test('running node updates progress and logs through polling, without losing terminal evidence', async ({ page }, info) => {
  const errors: string[] = []
  page.on('pageerror', e => errors.push(e.message))
  const now = new Date().toISOString()
  const progress = { phase: '下载分片', message: '公募基金持仓：正在下载公告日期分片。', completed: 120 as number | null, total: 800 as number | null, batches: 135, received_rows: 186420, activity_at: now,
    logs: [{ at: now, message: '[INFO] 日期进度 120/800，异常 0。' }, { at: now, message: '[INFO] 已接收持仓数据，凭据 [REDACTED]。' }] }
  const run = { run_id: 'progress-fixture', name: 'Tushare 全数据同步', status: 'RUNNING', created_at: now, updated_at: now, attempt: 1, published: false,
    steps: [{ id: 'manager', name: '公募基金经理履历', kind: 'task', status: 'SUCCEEDED', rows: 85189 },
      { id: 'portfolio', name: '公募基金季度股票持仓披露', kind: 'task', status: 'RUNNING', started_at: new Date(Date.now() - 123000).toISOString(), heartbeat_at: now, progress },
      { id: 'dividend', name: '公募基金分红', kind: 'task', status: 'PENDING' }] }
  await page.route('**/api/**', route => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/data-sources/catalog') return route.fulfill({ json: { ...catalog, editing_enabled: true } })
    if (path === '/api/data-sources/etl/workflows') return route.fulfill({ json: [] })
    if (path === '/api/data-sources/etl/runs') return route.fulfill({ json: [run] })
    return route.fulfill({ status: 404, json: {} })
  })
  await page.goto('/settings/data-sources')
  await page.getByRole('button', { name: '运行记录与恢复', exact: true }).click()
  await expect(page.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '15')
  await page.getByText('最近日志（2 条，已脱敏）').click()
  await expect(page.getByRole('list', { name: '最近执行日志' })).toContainText('120/800')
  await page.getByRole('progressbar').scrollIntoViewIfNeeded()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
  await page.screenshot({ path: info.outputPath('etl-progress.png'), fullPage: true })
  progress.completed = 240
  await expect(page.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '30', { timeout: 8000 })
  progress.phase = '合并与校验'; progress.message = '本地合并历史数据'; progress.completed = null; progress.total = null
  await expect(page.getByText('本地合并历史数据')).toBeVisible({ timeout: 8000 })
  await expect(page.getByRole('progressbar')).not.toHaveAttribute('aria-valuenow')
  await expect(page.getByRole('list', { name: '最近执行日志' })).toBeVisible()
  run.status = 'FAILED'; run.steps[1].status = 'FAILED'
  await expect(page.getByText('最后执行进度')).toBeVisible({ timeout: 8000 })
  await expect(page.getByRole('progressbar')).toHaveCount(0)
  expect(errors).toEqual([])
})
