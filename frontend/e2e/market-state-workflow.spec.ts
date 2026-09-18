import { expect, test, type Page } from '@playwright/test'
import { auditTextContrast } from './helpers/contrast'

const api = 'http://127.0.0.1:8769'
async function connect(page: Page) {
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url())
    if (!url.pathname.startsWith('/api/historical-regimes/')) return route.fulfill({ status: 404, json: { detail: 'Outside fixture.' } })
    const response = await route.fetch({ url: api + url.pathname + url.search, timeout: 120_000 })
    await route.fulfill({ response })
  })
  return (await page.request.get(api + '/ready')).json()
}
test.afterEach(async ({ page }) => { await page.unrouteAll({ behavior: 'wait' }) })

test('连续状态算子保存与真实预览完整覆盖输入', async ({ page }, info) => {
  const identity = await connect(page)
  const draft = await (await page.request.get(`${api}/api/historical-regimes/v2/definitions/${identity.realtime_id}?revision=1`)).json()
  const reference = (node_id: string, port = 'value') => ({ node_id, port })
  draft.name = '连续状态浏览器验收'
  draft.graph.edges = []
  draft.graph.nodes.push(
    { id: 'init_check', type: 'condition.compare', parameters: { operator: 'ge', threshold: 100 }, inputs: { value: reference('source') } },
    { id: 'init_state', type: 'state.select', parameters: { true_code: 0, false_code: 2 }, inputs: { condition: reference('init_check', 'condition') } },
    { id: 'continuous', type: 'state.continuous', label: '连续状态保持与确认', parameters: { confirmation: 2 }, inputs: { candidate: draft.graph.outputs.state, initial: reference('init_state', 'state'), value: reference('source') } },
  )
  draft.graph.outputs = { state: reference('continuous', 'state'), evidence: reference('continuous', 'evidence') }
  draft.graph.channel_metadata = { evidence: { label: '连续状态判定依据' } }
  const created = await page.request.post(api + '/api/historical-regimes/v2/definitions', { data: { definition: draft } })
  expect(created.ok(), await created.text()).toBeTruthy()
  const saved = await created.json()
  const prepared = await page.request.post(api + '/api/historical-regimes/prepare', { data: { definition: saved } })
  expect(prepared.ok(), await prepared.text()).toBeTruthy()
  const { compile_token } = await prepared.json()
  const preview = await page.request.post(api + '/api/historical-regimes/preview-runs', { data: { definition: saved, compile_token, mode: 'realtime' } })
  expect(preview.ok(), await preview.text()).toBeTruthy()
  const { id } = await preview.json()
  await expect.poll(async () => (await (await page.request.get(`${api}/api/historical-regimes/preview-runs/${id}`)).json()).status).toBe('completed')
  const series = await (await page.request.get(`${api}/api/historical-regimes/preview-runs/${id}/series?limit=5000`)).json()
  expect(series.items).toHaveLength(900)
  expect(series.items.every((row: { state_code: number; state_id: string }) => row.state_code >= 0 && row.state_id !== 'unclassified')).toBeTruthy()
  await page.goto(`/settings/scenario-algorithms?center=market-state&stage=realtime&definition=${saved.id}&revision=1`)
  await expect(page.getByLabel('研究名称')).toHaveValue('连续状态浏览器验收')
  await expect(page.getByText('正在加载模板与可用节点…', { exact: true })).toBeHidden()
  await visual(page, info.outputPath('continuous-state.png'))
})
async function visual(page: Page, path: string) {
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path, fullPage: true })
}

test('新实时模型先选精确参考，空白模型继承状态，步骤切换保留绑定', async ({ page }, info) => {
  const identity = await connect(page)
  await page.goto('/settings/scenario-algorithms?center=market-state&stage=realtime')
  await expect(page.getByText(/先完成历史参考选择与状态对应/)).toBeVisible()
  await expect(page.getByLabel('研究名称')).toBeHidden()
  await expect(page.getByLabel('历史参考版本')).toBeEnabled()
  await visual(page, info.outputPath('reference-required.png'))
  await page.getByLabel('历史参考版本').selectOption(JSON.stringify([identity.reference.run_id, identity.reference.publication_id, identity.reference.content_hash]))
  await expect(page.getByText('已绑定参考 · 待验证', { exact: true })).toBeVisible()
  await expect(page.getByLabel('研究名称')).toBeVisible()
  const value = await page.getByLabel('历史参考版本').inputValue()
  await page.getByRole('tab', { name: /验证识别能力与应用/ }).click()
  await expect(page.getByLabel('历史参考版本')).toHaveValue(value)
  await expect(page.getByText(/模型原始概率、相对参考校准后的匹配概率/)).toBeVisible()
  await visual(page, info.outputPath('reference-bound-validation.png'))
})

test('目录错误可重试，空目录能返回历史步骤或显式进入探索', async ({ page }, info) => {
  await connect(page)
  let fail = true
  await page.route('**/historical-regimes/references', route => route.fulfill(fail
    ? { status: 503, json: { detail: '参考目录暂不可用' } }
    : { json: { items: [] } }))
  await page.goto('/settings/scenario-algorithms?center=market-state&stage=realtime')
  await expect(page.getByRole('alert').filter({ hasText: /参考目录/ })).toBeVisible()
  fail = false
  await page.getByRole('button', { name: '重试读取参考' }).click()
  await expect(page.getByText(/还没有已确认的历史参考/)).toBeVisible()
  await page.getByText('高级：无参考探索', { exact: true }).click()
  await page.getByRole('button', { name: '进入探索草稿' }).click()
  await expect(page.getByText('探索草稿 · 尚无验证基准')).toBeVisible()
  await expect(page.getByLabel('研究名称')).toBeVisible()
  await visual(page, info.outputPath('exploration-empty-reference.png'))
})

test('已保存历史参考读取失败可重试，截至日切换后重新核对', async ({ page }, info) => {
  const identity = await connect(page)
  let fail = true
  let empty = true
  await page.route('**/historical-regimes/references', async route => {
    if (fail) return route.fulfill({ status: 503, json: { detail: '历史参考目录暂不可用' } })
    if (empty) return route.fulfill({ json: { items: [] } })
    return route.fulfill({ response: await route.fetch({ url: api + '/api/historical-regimes/references' }) })
  })
  await page.goto(`/settings/scenario-algorithms?center=market-state&stage=historical&definition=${identity.historical_id}&revision=1`)
  await expect(page.getByRole('button', { name: '重试读取历史参考', exact: true })).toBeVisible()
  const next = page.getByRole('button', { name: '下一步：建立实时识别', exact: true })
  await expect(next).toBeDisabled()
  await visual(page, info.outputPath('historical-reference-lookup-error.png'))
  fail = false
  await page.getByRole('button', { name: '重试读取历史参考', exact: true }).click()
  await expect(page.getByText('先“保存为历史参考”确认本版本的历史区间。', { exact: true })).toBeVisible()
  await expect(next).toBeDisabled()
  await page.getByRole('tab', { name: /^校验与预览/ }).click()
  empty = false
  await page.getByLabel('V2 截至日', { exact: true }).fill(identity.cutoff)
  await expect(next).toBeEnabled()
  await visual(page, info.outputPath('historical-reference-cutoff-restored.png'))
})

test('三种滤波节点用真实准备与预览接口执行，事后节点拒绝实时预览', async ({ page }, info) => {
  const identity = await connect(page)
  const original = await (await page.request.get(`${api}/api/historical-regimes/v2/definitions/${identity.historical_id}?revision=1`)).json()
  const definition = original.definition || original
  const catalog = await (await page.request.get(api + '/api/historical-regimes/nodes')).json()
  for (const type of ['filter.butterworth_zero_phase', 'filter.savitzky_golay_centered', 'filter.ehlers_error_correcting']) {
    const item = catalog.items.find((n: { id: string }) => n.id === type)
    expect(item).toBeTruthy()
    const draft = structuredClone(definition)
    const node = draft.graph.nodes.find((n: { id: string }) => n.id === 'smooth')
    node.type = type; node.type_id = type; node.parameters = {}
    const preview_target = { node_id: 'smooth', port: 'value' }
    const prepared = await page.request.post(api + '/api/historical-regimes/prepare', { data: { definition: draft, preview_target } })
    expect(prepared.ok(), await prepared.text()).toBeTruthy()
    const { compile_token } = await prepared.json()
    const body = { definition: draft, preview_target, compile_token, mode: item.causal ? 'realtime' : 'retrospective' }
    const created = await page.request.post(api + '/api/historical-regimes/preview-runs', { data: body })
    expect(created.ok(), await created.text()).toBeTruthy()
    const { id } = await created.json()
    await expect.poll(async () => (await (await page.request.get(`${api}/api/historical-regimes/preview-runs/${id}`)).json()).status).toBe('completed')
    const series = await (await page.request.get(`${api}/api/historical-regimes/preview-runs/${id}/series`)).json()
    expect(series.items.some((row: { value: number | null }) => row.value !== null)).toBeTruthy()
    if (!item.causal) {
      const blocked = await page.request.post(api + '/api/historical-regimes/preview-runs', { data: { ...body, mode: 'realtime' } })
      expect(blocked.status()).toBe(422)
    }
  }
  await page.goto('/settings/scenario-algorithms?center=market-state&stage=historical&template=market-trend-smoothed-savgol-reference-v2')
  await expect(page.getByLabel('研究名称')).toHaveValue('平滑峰谷参考 · Savitzky–Golay 居中')
  await visual(page, info.outputPath('smoothed-reference-template.png'))
})

test('探索模型真实保存后保留草稿提示，不触发研究发布', async ({ page }, info) => {
  const identity = await connect(page)
  const payload = await (await page.request.get(`${api}/api/historical-regimes/v2/definitions/${identity.realtime_id}?revision=1`)).json()
  const draft = payload.definition || payload
  delete draft.study.reference
  draft.name = '无参考探索 · ' + info.project.name
  const response = await page.request.post(api + '/api/historical-regimes/v2/definitions', { data: { definition: draft } })
  expect(response.ok(), await response.text()).toBeTruthy()
  const saved = await response.json()
  let activations = 0
  page.on('request', request => { if (request.url().includes('/research-versions')) activations++ })
  await page.goto(`/settings/scenario-algorithms?center=market-state&stage=realtime&definition=${saved.id}&revision=1`)
  await expect(page.getByText('探索草稿 · 尚无验证基准')).toBeVisible()
  await expect(page.getByRole('button', { name: '保存', exact: true })).toBeEnabled()
  await page.getByRole('button', { name: '保存', exact: true }).click()
  const dialog = page.getByRole('dialog')
  await dialog.getByRole('textbox', { name: '情景名称', exact: true }).fill(draft.name + ' 已修改')
  await dialog.getByRole('button', { name: '保存探索草稿', exact: true }).click()
  await expect(dialog.getByText(/探索草稿已保存/)).toBeVisible()
  await expect(dialog.getByText('请先完成公式检查，再保存情景。', { exact: true })).toHaveCount(0)
  await expect(dialog.getByRole('link', { name: '前往产品研究' })).toHaveCount(0)
  expect(activations).toBe(0)
  await visual(page, info.outputPath('exploration-saved.png'))
})
