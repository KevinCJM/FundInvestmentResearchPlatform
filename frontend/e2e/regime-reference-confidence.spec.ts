import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, test, type Page } from '@playwright/test'
import { auditTextContrast } from './helpers/contrast'
import { qualityDefinition, qualityPreviewFixture } from '../src/pages/regime-workbench/regimeQualityFixtures'
import { reliabilityDefinition, reliabilityPreviewFixture, reliabilityReference } from '../src/pages/regime-workbench/regimeReliabilityFixtures'

const audit = { execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0, request_time_compilation: 0, kernel_signatures: { regime_graph: ['float64[:]->int8[:]'] } }
async function fixture(page: Page, options: { empty?: boolean; error?: boolean; delayed?: boolean; modern?: boolean } = {}) {
  if (process.env.REGIME_OFFLINE_BUILD) {
    const buildRoot = path.resolve(process.env.REGIME_OFFLINE_BUILD)
    await page.route('http://regime-fixture.test/**', async route => {
      const pathname = new URL(route.request().url()).pathname
      const relative = pathname.startsWith('/assets/') ? pathname.slice(1) : 'index.html'
      const file = path.resolve(buildRoot, relative)
      if (!file.startsWith(buildRoot + path.sep)) return route.abort()
      const contentType = file.endsWith('.js') ? 'application/javascript' : file.endsWith('.css') ? 'text/css' : file.endsWith('.woff2') ? 'font/woff2' : 'text/html'
      await route.fulfill({ body: await readFile(file), contentType })
    })
  }
  let definition = structuredClone(reliabilityDefinition)
  let release: (() => void) | undefined
  const pending = new Promise<void>(resolve => { release = resolve })
  let previewStarted = false
  let qualityRuns = 0
  const qualityReports: Array<ReturnType<typeof qualityPreviewFixture> & { id: string; created_at: string; immutable: true; content_hash: string }> = []
  const reports: ReturnType<typeof reliabilityPreviewFixture>[] = []
  const saved: Array<ReturnType<typeof reliabilityPreviewFixture> & { id: string; calibration_id: string; created_at: string; immutable: boolean; content_hash: string }> = []
  const reference2 = { ...reliabilityReference, run_id: 'reference-2', publication_id: 'pub-2', content_hash: 'c'.repeat(64), name: '第二个离线参考' }
  await page.route('**/api/**', async route => {
    const req = route.request()
    const path = new URL(req.url()).pathname
    const post = req.method() === 'POST' || req.method() === 'PUT'
    const body = post ? req.postDataJSON() : undefined
    let value: unknown
    if (path.endsWith('/nodes')) value = { items: [{ id: 'source.inline', label: '离线观察数据', category: 'source', causal: true, supports_realtime: true, repaints: false, inputs: [], outputs: [{ id: 'state', label: '状态', value_type: 'state_codes<int64>' }], parameter_schema: { properties: {} } }] }
    else if (path.endsWith('/templates/v2')) value = { items: [] }
    else if (path.endsWith('/references')) {
      if (options.error) return route.fulfill({ status: 503, json: { detail: '离线参考目录暂不可用' } })
      value = { items: options.empty ? [] : [reliabilityReference, reference2] }
    } else if (path.endsWith('/v2/definitions/history')) value = qualityDefinition
    else if (path.endsWith('/v2/definitions/recognition')) {
      if (post) definition = { ...body.definition, id: 'recognition', revision: definition.revision! + 1 }
      value = definition
    } else if (path.endsWith('/v2/definitions')) value = { items: [definition] }
    else if (path.endsWith('/infer')) value = { valid: true, errors: [], warnings: [], graph_hash: 'g', inferred: { causal: true, realtime_eligible: true } }
    else if (path.endsWith('/authoring/resolve')) value = { valid: true, definition: body.definition, source: '', diagnostics: [], display_latex: {} }
    else if (path.endsWith('/prepare')) value = { plan_id: 'plan', compile_token: 'token', graph_hash: 'g', runtime_audit: audit }
    else if (path.endsWith('/reliability/preview')) {
      previewStarted = true
      if (options.delayed) await pending
      const result = reliabilityPreviewFixture()
      result.request = { definition_id: body.definition_id, revision: body.revision, reference: body.reference, policy: body.policy }
      if (options.modern) {
        result.report.stability = { status: 'causal_probes_executed', parameter_sensitivity: qualityPreviewFixture().report.stability }
        result.report.confidence_interval = { status: 'available', reason: null, method: 'paired_moving_block', scope: 'holdout', confidence_level: 0.95, block_length: body.policy.bootstrap?.block_length || 10, replicates: 200, seed: 1729, samples: 80, full_blocks: 8, complete_cycles: 4, metrics: { accuracy: { estimate: 0.75, lower: 0.65, upper: 0.85, valid_replicates: 200, reason: null, unit: 'fraction' } } }
      }
      reports.push(result); value = result
    } else if (path.endsWith('/reference-quality/preview')) {
      qualityRuns += 1
      value = { ...qualityPreviewFixture(), request: { definition_id: body.definition_id, revision: body.revision, mode: body.mode, as_of: body.as_of, policy: body.policy } }
    } else if (path.endsWith('/reference-quality/confirm')) {
      expect(Object.keys(body).sort()).toEqual(['preview_hash', 'request'])
      const item = { ...qualityPreviewFixture(), request: body.request, id: 'quality-1', created_at: '2026-09-15', immutable: true as const, content_hash: 'q'.repeat(64) }
      qualityReports.push(item); value = item
    } else if (path.endsWith('/reference-quality/catalog')) value = { items: qualityReports.map(item => ({ id: item.id, created_at: item.created_at, definition_id: item.request.definition_id, revision: item.request.revision, status: item.report.status })) }
    else if (path.endsWith('/reference-quality/reports/quality-1')) value = qualityReports[0]
    else if (path.endsWith('/reliability/confirm')) {
      expect(Object.keys(body).sort()).toEqual(['preview_hash', 'request'])
      const report = reports.find(item => item.preview_hash === body.preview_hash)!
      const item = { ...report, id: 'saved-report', calibration_id: 'saved-report', created_at: '2026-09-14', immutable: true, content_hash: 'f'.repeat(64) }
      saved.push(item); value = item
    } else if (path.endsWith('/reliability/catalog')) value = { items: saved.map(item => ({ id: item.id, calibration_id: item.calibration_id, created_at: item.created_at, definition_id: item.request.definition_id, revision: item.request.revision, reference: item.request.reference, status: item.report.status, calibration: item.report.calibration })) }
    else if (path.endsWith('/reliability/reports/saved-report')) value = saved[0]
    else if (path.endsWith('/v2/graph-assets') || path.endsWith('/v2/experiments')) value = { items: [] }
    else return route.fulfill({ status: 404, json: { detail: 'Offline fixture only' } })
    await route.fulfill({ json: value }).catch(error => { if (!req.failure()) throw error })
  })
  return { release: () => release?.(), started: () => previewStarted, reports, qualityRuns: () => qualityRuns }
}
async function openRealtime(page: Page) {
  await page.goto('/settings/scenario-algorithms?center=historical&mode=realtime&definition=recognition&revision=2')
  await expect(page.getByRole('heading', { name: '建立实时识别', exact: true })).toBeVisible()
  await expect(page.getByLabel('研究名称')).toHaveValue(reliabilityDefinition.name)
  await expect(page.getByLabel('历史参考版本')).toBeEnabled()
}
test('三类情景研究、市场状态三步流程、旧深链接、独立草稿与响应式对比度', async ({ page }, testInfo) => {
  await fixture(page)
  await openRealtime(page)
  const areas = page.getByRole('tablist', { name: '情景研究类型' })
  await expect(areas.getByRole('tab')).toHaveCount(3)
  await expect(areas.getByRole('tab', { name: '市场状态研究' })).toHaveAttribute('aria-selected', 'true')
  const steps = page.getByRole('tablist', { name: '市场状态研究步骤' })
  await expect(steps.getByRole('tab')).toHaveCount(3)
  await expect(steps.getByRole('tab', { name: /建立实时识别/ })).toHaveAttribute('aria-selected', 'true')
  await expect(page.getByRole('radiogroup', { name: 'V2 识别模式' })).toHaveCount(0)
  await expect(page.getByLabel('算法库识别方式')).toHaveCount(0)
  await page.getByLabel('研究名称').fill('实时草稿保留')
  await steps.getByRole('tab', { name: /定义历史参考/ }).click()
  await expect(page.getByRole('heading', { name: '定义历史参考', exact: true })).toBeVisible()
  await page.getByLabel('研究名称').filter({ visible: true }).fill('历史草稿保留')
  await expect(page.getByLabel('历史参考版本')).not.toBeVisible()
  const tab = steps.getByRole('tab', { name: /定义历史参考/ })
  await tab.focus(); await page.keyboard.press('ArrowRight')
  await expect(steps.getByRole('tab', { name: /建立实时识别/ })).toBeFocused()
  await expect(page.getByLabel('研究名称').filter({ visible: true })).toHaveValue('实时草稿保留')
  await page.keyboard.press('Home')
  await expect(steps.getByRole('tab', { name: /定义历史参考/ })).toBeFocused()
  await expect(page.getByLabel('研究名称').filter({ visible: true })).toHaveValue('历史草稿保留')
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: testInfo.outputPath('historical-definition-workspace.png'), fullPage: true })
})
test('无参考与目录错误都有下一步', async ({ page }) => {
  await fixture(page, { empty: true })
  await page.goto('/settings/scenario-algorithms?center=realtime')
  await expect(page.getByText('还没有已确认的历史参考。先生成历史区间，再保存为历史参考。')).toBeVisible()
  await expect(page.getByRole('button', { name: '验证识别能力', exact: true })).toBeDisabled()
  await page.getByRole('button', { name: '前往历史状态定义' }).click()
  await expect(page.getByRole('heading', { name: '定义历史参考', exact: true })).toBeVisible()
  await page.unroute('**/api/**')
  await fixture(page, { error: true })
  await page.goto('/settings/scenario-algorithms?center=realtime')
  await expect(page.getByText('离线参考目录暂不可用')).toBeVisible()
  await expect(page.getByRole('button', { name: '重试读取参考' })).toBeVisible()
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
})
test('验证报告明确不足、确认保存和精确版本重载，真实图表与详情可读', async ({ page }, testInfo) => {
  await fixture(page)
  await openRealtime(page)
  await page.getByLabel('校准截止日', { exact: true }).fill('2020-12-31')
  await page.getByRole('button', { name: '验证识别能力', exact: true }).click()
  const report = page.getByLabel('识别能力验证报告')
  await expect(report).toBeVisible()
  await expect(report.getByText('样本不足', { exact: true })).toBeVisible()
  await expect(report.getByText('不可部署', { exact: true })).toBeVisible()
  await expect(report.getByText('仅回顾性评分', { exact: true })).toBeVisible()
  await expect(report).toContainText('原始数值为 1 不代表 100% 可信')
  await report.getByText('样本分段、状态表现与混淆矩阵', { exact: true }).click()
  await expect(report.getByRole('table', { name: '混淆矩阵：参考行与预测列' })).toBeVisible()
  await report.getByText('概率评分与可靠性图', { exact: true }).click()
  await expect(report.locator('canvas')).toBeVisible()
  await expect(report.getByRole('table', { name: '校准可靠性分箱' })).toBeVisible()
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: testInfo.outputPath('reference-confidence-report.png'), fullPage: true })
  await page.getByRole('button', { name: '保存验证报告' }).click()
  await expect(page.getByText('已保存的不可变报告', { exact: true })).toBeVisible()
  await page.reload()
  await expect(page.getByLabel('研究名称')).toHaveValue(reliabilityDefinition.name)
  await page.getByText('已保存报告', { exact: true }).click()
  await page.getByLabel('已保存验证报告').selectOption('saved-report')
  await page.getByRole('button', { name: '载入报告' }).click()
  await expect(page.getByText('已保存的不可变报告', { exact: true })).toBeVisible()
  await expect(page.getByLabel('识别能力验证报告')).toContainText('仅回顾性评分')
})
test('参考切换阻断旧修订验证并丢弃迟到报告', async ({ page }) => {
  const state = await fixture(page, { delayed: true })
  await openRealtime(page)
  await page.getByLabel('校准截止日', { exact: true }).fill('2020-12-31')
  await page.getByRole('button', { name: '验证识别能力', exact: true }).click()
  await expect.poll(state.started).toBeTruthy()
  const value = await page.getByLabel('历史参考版本').locator('option').filter({ hasText: '第二个离线参考' }).getAttribute('value')
  await page.getByLabel('历史参考版本').selectOption(value!)
  state.release()
  await expect(page.getByRole('button', { name: '验证识别能力', exact: true })).toBeDisabled()
  await expect(page.getByText('请先保存当前模型，再验证这个精确修订。')).toBeVisible()
  await expect(page.getByLabel('识别能力验证报告')).toHaveCount(0)
  await expect(page.getByRole('button', { name: '保存验证报告' })).toHaveCount(0)
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
})

test('历史质量按需检查、区间统计、保存重载与草稿失效', async ({ page }, testInfo) => {
  const state = await fixture(page)
  await page.goto('/settings/scenario-algorithms?center=historical&definition=history&revision=3')
  await expect(page.getByRole('heading', { name: '定义历史参考', exact: true })).toBeVisible()
  const graph = page.getByRole('region', { name: '历史情景可编辑节点画布' })
  await expect(graph).toContainText('离线观察数据')
  expect(state.qualityRuns()).toBe(0)
  await page.getByRole('button', { name: '检查划分质量', exact: true }).click()
  const report = page.getByLabel('历史划分质量报告')
  await expect(report).toContainText('分类覆盖 80.0%')
  await expect(report).toContainText('尾端未分类 15')
  await report.getByText('状态支持、持续时间与区间收益', { exact: true }).click()
  await expect(report.getByRole('table', { name: '历史状态区间统计' })).toBeVisible()
  await report.getByText('参数、窗口与边界变化详情', { exact: true }).click()
  await expect(report.getByRole('table', { name: '稳定性变体' })).toContainText('0.1 → 0.11')
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: testInfo.outputPath('historical-quality-report.png'), fullPage: true })
  await page.getByRole('button', { name: '确认保存质量报告' }).click()
  await expect(page.getByText('质量报告已保存', { exact: true })).toBeVisible()
  await page.reload()
  await expect(page.getByRole('button', { name: '检查划分质量', exact: true })).toBeEnabled()
  await page.getByText('已保存质量报告', { exact: true }).click()
  await page.getByLabel('已保存质量报告', { exact: true }).selectOption('quality-1')
  await page.getByRole('button', { name: '载入质量报告' }).click()
  await expect(page.getByText('质量报告已保存', { exact: true })).toBeVisible()
  expect(state.qualityRuns()).toBe(1)
  await page.getByLabel('研究名称').fill('尚未保存的新参数版本')
  await expect(report).toHaveCount(0)
  await expect(page.getByRole('button', { name: '检查划分质量', exact: true })).toBeDisabled()
})
test('新报告真实区间、稳定性、可靠性图与高级观测单位', async ({ page }, testInfo) => {
  const state = await fixture(page, { modern: true })
  await openRealtime(page)
  await page.getByLabel('校准截止日', { exact: true }).fill('2020-12-31')
  await page.getByText('高级检查设置', { exact: true }).click()
  await page.getByLabel('每块观测数', { exact: true }).fill('12')
  expect(state.reports).toHaveLength(0)
  await page.getByRole('button', { name: '检查稳定性', exact: true }).click()
  const report = page.getByLabel('识别能力验证报告')
  await expect(report).toContainText('历史留出指标区间：已估计')
  expect(state.reports[0].request.policy.bootstrap?.block_length).toBe(12)
  await report.getByText('置信区间与重采样详情', { exact: true }).click()
  await expect(report.getByRole('table', { name: '历史指标区间' })).toContainText('65.0%')
  await report.getByText('参数、窗口与边界变化详情', { exact: true }).click()
  await expect(report.getByRole('table', { name: '稳定性变体' })).toBeVisible()
  await report.getByText('概率评分与可靠性图', { exact: true }).click()
  await expect(report.locator('canvas')).toBeVisible()
  await expect(report.getByRole('table', { name: '校准可靠性分箱' })).toBeVisible()
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: testInfo.outputPath('regime-completion-reliability.png'), fullPage: true })
})
