import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { expect, test, type Page } from '@playwright/test'
import { auditTextContrast } from './helpers/contrast'
import { reliabilityDefinition, reliabilityReference } from '../src/pages/regime-workbench/regimeReliabilityFixtures'
import { prospectiveAssessment, prospectiveProgress, prospectiveProtocol, prospectiveReference, prospectiveSaved } from '../src/pages/regime-workbench/regimeProspectiveFixtures'
import { referenceKey } from '../src/pages/regime-workbench/RegimeReferenceBinding'
import type { ProspectiveSourcePreview, ProspectiveSourceVersion } from '../src/services/regimeProspective'

// No listener, external network, production backend, or shared dist.
async function fixture(page: Page, rejected = false, withSource = false) {
  const root = path.resolve(process.env.REGIME_OFFLINE_BUILD || '/tmp/bettersaataa-regime-forward-build')
  await page.route('**/*', async route => {
    const url = new URL(route.request().url())
    if (url.hostname !== 'regime-fixture.test') return route.abort()
    const file = path.resolve(root, url.pathname.startsWith('/assets/') ? url.pathname.slice(1) : 'index.html')
    if (!file.startsWith(root + path.sep)) return route.abort()
    await route.fulfill({ body: await readFile(file), contentType: file.endsWith('.js') ? 'application/javascript' : file.endsWith('.css') ? 'text/css' : 'text/html' })
  })
  let registered = false
  let captures = 0
  let assessment: typeof prospectiveAssessment | null = null
  let adopted: Record<string, unknown> | null = null
  let writes = 0
  let model = structuredClone(reliabilityDefinition)
  if (withSource) { model.graph.nodes[0].type = 'source.index'; model.graph.nodes[0].parameters = { name: '离线观察数据', ts_code: '000300.SH', source_api: 'index_daily', field: 'close' } }
  const sourcePreview: ProspectiveSourcePreview = { preview_hash: 'a'.repeat(64), protocol_id: prospectiveProtocol.id, previous_id: prospectiveProtocol.id, as_of: '2027-09-14', model_binding_hash: 'binding-next', model_bindings: { source: { snapshot_id: 'next-data', snapshot_generation: 'next-data', source_file: 'index_daily_df.parquet', file_checksum: 'b'.repeat(64) } }, old_observations: 100, new_observations: 101, added_observations: 1, prefix_unchanged: true, reference_id: 'history', reference_revision: 3 }
  let sourceVersion: ProspectiveSourceVersion | null = null
  await page.route('**/api/**', async route => {
    const req = route.request(), url = new URL(req.url()).pathname
    const post = req.method() === 'POST' || req.method() === 'PUT'
    const body = post ? req.postDataJSON() : null
    let value: unknown
    if (url.endsWith('/prospective/register')) {
      expect(body).toEqual({ calibration_id: 'cal-1' }); writes++; registered = true; value = prospectiveProtocol
    } else if (url.endsWith('/prospective/protocol-1/sources/preview')) {
      expect(body).toEqual({}); value = sourcePreview
    } else if (url.endsWith('/prospective/protocol-1/sources/confirm')) {
      expect(body).toEqual({ preview_hash: sourcePreview.preview_hash }); writes++
      sourceVersion = { ...sourcePreview, id: 'source-next', kind: 'source_version', status: 'accepted', recorded_at: '2027-09-15', reference_definition: { ...prospectiveProtocol.reference_definition, revision: 4 } }; value = sourceVersion
    } else if (url.endsWith('/prospective/protocol-1/capture')) {
      expect(body).toEqual({}); writes++; captures++; value = { status: 'captured' }
    } else if (url.endsWith('/prospective/protocol-1/assess')) {
      expect(body).toEqual({ reference: { run_id: prospectiveReference.run_id, publication_id: prospectiveReference.publication_id, content_hash: prospectiveReference.content_hash } })
      writes++; assessment = { ...prospectiveAssessment, status: rejected ? 'rejected' : 'qualified', reasons: rejected ? ['insufficient_complete_class_regimes'] : [] }; value = assessment
    } else if (url.endsWith('/prospective/catalog')) value = { items: registered ? [prospectiveProtocol] : [] }
    else if (url.endsWith('/prospective/protocols/protocol-1/progress')) value = { ...prospectiveProgress, observations: captures, last_observation_date: captures ? '2027-09-14' : null, latest_assessment: assessment, current_source_version: sourceVersion, reference_definitions: sourceVersion ? [prospectiveProtocol.reference_definition, sourceVersion.reference_definition] : [prospectiveProtocol.reference_definition] }
    else if (url.endsWith('/prospective/qualifications/qualification-1')) value = assessment
    else if (url.endsWith('/nodes')) value = { items: [{ id: withSource ? 'source.index' : 'source.inline', label: '离线观察数据', category: 'source', causal: true, supports_realtime: true, repaints: false, inputs: [], outputs: [{ id: 'state', label: '状态', value_type: 'state_codes<int64>' }], parameter_schema: { properties: {} } }] }
    else if (url.endsWith('/templates/v2')) value = { items: [] }
    else if (url.endsWith('/references')) value = { items: [reliabilityReference, prospectiveReference, { ...prospectiveReference, definition_revision: 99, name: '不可选的不同修订' }] }
    else if (url.endsWith('/v2/definitions/history')) value = { ...reliabilityDefinition, id: 'history', revision: 3 }
    else if (url.endsWith('/v2/definitions/recognition')) {
      if (post) { adopted = body.definition.study; model = { ...body.definition, id: 'recognition', revision: 3 } }
      value = model
    } else if (url.endsWith('/v2/definitions')) value = { items: [model] }
    else if (url.endsWith('/infer')) value = { valid: true, errors: [], warnings: [], graph_hash: 'g', inferred: { causal: true, realtime_eligible: true } }
    else if (url.endsWith('/authoring/resolve')) value = { valid: true, definition: body.definition, source: '', diagnostics: [], display_latex: {} }
    else if (url.endsWith('/prepare')) value = { plan_id: 'plan', compile_token: 'token', graph_hash: 'g', runtime_audit: { execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0, request_time_compilation: 0, kernel_signatures: { graph: ['float64[:]'] } } }
    else if (url.endsWith('/research-versions')) value = { id: 'research-v3', name: model.name, revision: 3, run_id: 'run-v3', series_summary: { row_count: 252, first_observation_date: '2027-01-01', last_observation_date: '2027-12-31' }, available_for: ['research_display'] }
    else if (url.endsWith('/reliability/catalog')) value = { items: [{ id: 'cal-1', calibration_id: 'cal-1', created_at: prospectiveSaved.created_at, definition_id: 'recognition', revision: 2, reference: prospectiveSaved.request.reference, status: prospectiveSaved.report.status, calibration: prospectiveSaved.report.calibration }] }
    else if (url.endsWith('/reliability/reports/cal-1')) value = prospectiveSaved
    else if (url.endsWith('/v2/graph-assets') || url.endsWith('/v2/experiments')) value = { items: [] }
    else return route.fulfill({ status: 404, json: { detail: '离线夹具未提供此接口' } })
    await route.fulfill({ json: value })
  })
  return { writes: () => writes, adopted: () => adopted, model: () => model }
}
async function load(page: Page) {
  await page.goto('http://regime-fixture.test/settings/scenario-algorithms?center=realtime&definition=recognition&revision=2')
  await expect(page.getByLabel('研究名称')).toHaveValue(reliabilityDefinition.name)
  await expect(page.getByText(`已从目录载入 ${reliabilityDefinition.name} · r2。`, { exact: true })).toBeVisible()
  await page.getByText('已保存报告', { exact: true }).click()
  await expect(page.getByLabel('已保存验证报告')).toBeEnabled()
  await page.getByLabel('已保存验证报告').selectOption('cal-1')
  await page.getByRole('button', { name: '载入报告', exact: true }).click()
  await expect(page.getByLabel('前瞻验证', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: '刷新前瞻进度' })).toBeEnabled()
}
async function audit(page: Page) {
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
}
for (const width of [320, 768, 1440]) {
  test(`前瞻登记、捕获、检验、重载和显式采用 ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 1000 })
    const state = await fixture(page)
    await load(page)
    const panel = page.getByLabel('前瞻验证', { exact: true })
    await panel.getByRole('button', { name: '登记前瞻验证' }).click()
    await expect(panel).toContainText('前瞻验证待完成')
    await expect(panel).toContainText('尚无记录')
    await audit(page)
    await panel.getByRole('button', { name: '记录当前判断' }).click()
    await expect(panel).toContainText('2027-09-14')
    await expect(panel.getByLabel('前瞻检验参考').locator('option')).toHaveCount(2)
    await panel.getByLabel('前瞻检验参考').selectOption(referenceKey(prospectiveReference))
    await panel.getByRole('button', { name: '检验前瞻结果' }).click()
    await expect(panel).toContainText('前瞻检验通过')
    await audit(page)
    await page.screenshot({ path: testInfo.outputPath(`prospective-qualified-${width}.png`), fullPage: true })
    expect(state.writes()).toBe(3)
    await load(page)
    await expect(panel).toContainText('前瞻检验通过')
    expect(state.writes()).toBe(3)
    await panel.getByRole('button', { name: '采用已验证校准' }).click()
    await expect(page.getByText('已采用前瞻验证校准与资格，请保存为新修订后使用。')).toBeVisible()
    await expect(panel).toHaveCount(0)
    await page.getByRole('button', { name: '保存', exact: true }).click()
    await page.getByRole('button', { name: '保存并用于研究', exact: true }).click()
    await expect(page.getByRole('button', { name: '已保存，可用于研究', exact: true })).toBeVisible()
    await expect.poll(state.adopted).toMatchObject({ calibration_id: 'cal-1', qualification_id: 'qualification-1' })
    await audit(page)
  })
}
for (const width of [320, 768, 1440]) {
  test(`指数数据续接预览、确认与重载 ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 1000 })
    const state = await fixture(page, false, true)
    await load(page)
    const panel = page.getByLabel('前瞻验证', { exact: true })
    await panel.getByRole('button', { name: '登记前瞻验证' }).click()
    await expect(panel.getByRole('button', { name: '检查新数据版本' })).toBeEnabled()
    await panel.getByRole('button', { name: '检查新数据版本' }).click()
    await expect(panel).toContainText('历史前缀一致：100 → 101')
    expect(state.writes()).toBe(1)
    await audit(page)
    await panel.getByRole('button', { name: '确认续接数据' }).click()
    const link = panel.getByRole('link', { name: '打开后续参考修订（新页面）' })
    await expect(link).toHaveAttribute('href', /definition=history&revision=4/)
    expect(state.writes()).toBe(2)
    await expect(panel).toContainText('已确认数据版本：next-data')
    await audit(page)
    await page.screenshot({ path: testInfo.outputPath(`source-version-${width}.png`), fullPage: true })
    await load(page)
    await expect(link).toBeVisible()
    expect(state.writes()).toBe(2)
    expect(state.model().graph.nodes[0].parameters).not.toHaveProperty('snapshot_id')
  })
}

test('未通过检验不提供采用操作', async ({ page }) => {
  await fixture(page, true); await load(page)
  const panel = page.getByLabel('前瞻验证', { exact: true })
  await panel.getByRole('button', { name: '登记前瞻验证' }).click()
  await expect(panel).toContainText('前瞻验证待完成')
  await panel.getByLabel('前瞻检验参考').selectOption(referenceKey(prospectiveReference))
  await panel.getByRole('button', { name: '检验前瞻结果' }).click()
  await expect(panel).toContainText('前瞻检验未通过')
  await expect(panel.getByRole('button', { name: '采用已验证校准' })).toHaveCount(0)
  await audit(page)
})
