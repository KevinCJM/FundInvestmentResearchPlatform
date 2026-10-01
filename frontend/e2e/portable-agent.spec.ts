import { test, expect } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => sessionStorage.setItem('pit.view.override', JSON.stringify({ off: true })))
})

test('真实双服务：指标草稿、人工保存与刷新恢复', async ({ page }, info) => {
  const errors: string[] = []
  const oldRequests: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('request', request => { if (/\/api\/(agent|settings\/llm)(\/|$)/.test(new URL(request.url()).pathname)) oldRequests.push(request.url()) })
  await page.goto('/settings/indicators-models')
  const agent = page.locator('portable-agent')
  await expect(agent).toHaveCount(1)
  await agent.getByRole('button', { name: '打开 AI 助手', exact: true }).click()
  await expect(agent.getByRole('button', { name: '发送', exact: true })).toBeEnabled()
  await agent.getByLabel('发送消息', { exact: true }).fill(`创建累计收益指标 [${info.project.name}]`)
  await agent.getByRole('button', { name: '发送', exact: true }).click()
  const draft = agent.getByRole('region', { name: '指标草稿' })
  await expect(draft).toContainText('独立框架累计收益')
  await expect(agent.getByRole('button', { name: '发送', exact: true })).toBeEnabled()
  page.once('dialog', dialog => dialog.accept())
  await draft.getByRole('button', { name: '保存', exact: true }).click()
  await expect(draft.getByRole('button', { name: '已保存', exact: true })).toBeDisabled()
  await page.reload()
  await agent.getByRole('button', { name: '打开 AI 助手', exact: true }).click()
  await expect(draft.getByRole('button', { name: '已保存', exact: true })).toBeDisabled()
  await expect(page.locator('body')).toContainText('独立框架累计收益')
  expect(oldRequests).toEqual([])
  expect(errors).toEqual([])
  await page.screenshot({ path: info.outputPath('portable-saved.png'), fullPage: true })
})

test('真实试算只传摘要，并将完整结果交回指标工作台', async ({ page, request }, info) => {
  await page.goto('/settings/indicators-models')
  const previewTab = page.locator('#workspace-tab-preview')
  if (await previewTab.isVisible()) await previewTab.click()
  else await page.locator('#indicator-tab-preview').click()
  await page.getByLabel('计算周期', { exact: true }).selectOption('ALL')
  const agent = page.locator('portable-agent')
  await agent.getByRole('button', { name: '打开 AI 助手', exact: true }).click()
  await expect(agent.getByRole('button', { name: '发送', exact: true })).toBeEnabled()
  await agent.getByLabel('发送消息', { exact: true }).fill('创建累计收益指标并试算上证50ETF')
  await agent.getByRole('button', { name: '发送', exact: true }).click()
  await expect(agent.getByRole('button', { name: '查看试算结果', exact: true })).toBeVisible()
  await expect(agent.getByRole('button', { name: '发送', exact: true })).toBeEnabled()
  await agent.getByRole('button', { name: '查看试算结果', exact: true }).click()
  const result = page.getByRole('region', { name: 'AI 试算结果', exact: true })
  await expect(result).toBeVisible()
  await expect(result).toContainText('1/1')
  await expect(result).toContainText('50.00%')
  await expect(result).not.toContainText('不可计算')
  const captured = await (await request.get('http://127.0.0.1:18080/fixture/model-requests')).json()
  expect(captured.items.length).toBeGreaterThan(2)
  expect(JSON.stringify(captured).replace(/\s/g, '')).not.toContain('[1,1.25,1,1.25,1.5,1.2,1.2,1.5]')
  await page.screenshot({ path: info.outputPath('portable-preview.png'), fullPage: true })
})

test('中央助手经真实交接进入指标页，刷新不会重做任务', async ({ page, request }) => {
  await page.goto('/settings/ai-agent')
  let agent = page.locator('portable-agent')
  await expect(agent.getByRole('button', { name: '发送', exact: true })).toBeEnabled()
  await agent.getByLabel('发送消息', { exact: true }).fill('帮我创建累计收益指标')
  await agent.getByRole('button', { name: '发送', exact: true }).click()
  await expect(page).toHaveURL(/\/settings\/indicators-models\?portable_handoff=/)
  agent = page.locator('portable-agent')
  await expect(agent.getByRole('region', { name: '指标草稿' })).toContainText('独立框架累计收益')
  await expect(agent.getByRole('button', { name: '发送', exact: true })).toBeEnabled()
  const before = (await (await request.get('http://127.0.0.1:18080/fixture/model-requests')).json()).items.length
  await page.reload()
  await agent.getByRole('button', { name: '打开 AI 助手', exact: true }).click()
  await expect(agent.getByRole('region', { name: '指标草稿' })).toContainText('独立框架累计收益')
  expect((await (await request.get('http://127.0.0.1:18080/fixture/model-requests')).json()).items.length).toBe(before)
})

test('模型设置由外部服务管理，保存不自动切换使用项', async ({ page }) => {
  await page.goto('/settings/llm-api')
  const agent = page.locator('portable-agent')
  await agent.getByRole('button', { name: '新增 API', exact: true }).click()
  await agent.getByLabel('配置名称', { exact: true }).fill('第二组离线模型')
  await agent.getByLabel('模型名称', { exact: true }).fill('second-fixture')
  await agent.getByLabel('API 密钥', { exact: true }).fill('offline-only')
  await agent.getByRole('button', { name: '保存设置', exact: true }).click()
  await expect(agent.getByLabel('API 密钥', { exact: true })).toBeHidden()
  await expect(agent.locator('.profile-items')).toContainText('第二组离线模型')
  await expect(agent.locator('.profile-items')).toContainText('本地固定模型')
  const profiles = await page.evaluate(async () => {
    const element = document.querySelector('portable-agent') as HTMLElement & { api: (path: string) => Promise<{active_profile_id: string}> }
    return element.api('/model-profiles')
  })
  expect(profiles.active_profile_id).toBe('fixture-model')
})

test('情景成果界面：读取失败可重试，校验失败不显示已核验', async ({ page }) => {
  // UI failure fixture only; numerical scenario execution is covered by the real-service backend tests.
  let release!: () => void
  const gate = new Promise<void>(resolve => { release = resolve })
  let failing = true
  await page.route('**/api/research/scenario-artifacts/ui-fixture', async route => {
    await gate
    if (failing) {
      return route.fulfill({ status: 503, json: { error: { message: '暂不可用，请重试' } } })
    }
    return route.fulfill({ json: { definition: { name: '无效情景候选' }, valid: false, validation_scope: 'graph', workspace: 'graph', source_run_id: 'ui-run' } })
  })
  await page.goto('/settings/scenario-algorithms')
  const agent = page.locator('portable-agent').first()
  await agent.getByRole('button', { name: '打开 AI 助手', exact: true }).click()
  await expect(agent.getByRole('button', { name: '发送', exact: true })).toBeEnabled()
  await agent.evaluate(element => {
    const container = document.createElement('div'); container.slot = 'ui-fixture'; element.append(container)
    const slot = document.createElement('slot'); slot.name = 'ui-fixture'; element.shadowRoot!.querySelector('.messages')!.append(slot)
    element.dispatchEvent(new CustomEvent('agent-artifact-mount', { cancelable: true, detail: {
      artifact: { id: 'ui-fixture', tool: 'scenarios_validate', data: { type: 'research.scenario', artifact_id: 'ui-fixture' } },
      runId: 'ui-run', context: {}, bindingRevision: 0, isCurrent: true, container,
      setUpdate: () => {}, setDispose: () => {},
    } }))
  })
  const card = agent.getByRole('region', { name: '情景候选与试算' })
  await expect(card).toContainText('正在读取结果')
  await expect(card).not.toContainText('结构已核验')
  release()
  await expect(card.getByRole('alert')).toContainText('暂不可用')
  failing = false
  await card.getByRole('button', { name: '重试', exact: true }).click()
  await expect(card).toContainText('校验未通过')
  await expect(card).not.toContainText('结构已核验')
  await expect(card.getByRole('button', { name: '填入工作区' })).toBeDisabled()
})

test('真实情景服务：读取保存版本、候选回填与数值试算', async ({ page, request }) => {
  await page.goto('/settings/scenario-algorithms?center=market-state&stage=historical')
  let agent = page.locator('portable-agent')
  await expect(agent).toHaveCount(1)
  await agent.getByRole('button', { name: '打开 AI 助手', exact: true }).click()
  await expect(agent.getByRole('button', { name: '发送', exact: true })).toBeEnabled()
  await agent.getByLabel('发送消息', { exact: true }).fill('创建已保存算法的候选，并试算')
  await agent.getByRole('button', { name: '发送', exact: true }).click()
  const card = agent.getByRole('region', { name: '情景候选与试算' }).last()
  await expect(card.getByRole('button', { name: '查看试算结果' })).toBeVisible()
  await expect(card.getByRole('button', { name: '填入工作区' })).toBeEnabled()
  await expect(card).toContainText('联调已保存日频峰谷')
  await card.getByRole('button', { name: '填入工作区' }).click()
  await expect(page).toHaveURL(/new=1/)
  await expect(page.getByLabel('研究名称', { exact: true })).toHaveValue('联调已保存日频峰谷')
  agent = page.locator('portable-agent')
  await expect(agent).toHaveCount(1)
  const captured = await (await request.get('http://127.0.0.1:18080/fixture/model-requests')).json()
  const messages = JSON.stringify(captured.items)
  expect(messages).toContain('scenarios_read')
  expect(messages).toContain('source.index')
  expect(messages).not.toContain('"node_outputs"')
})

test('助手服务断开后，手工指标编辑仍可使用', async ({ page }) => {
  await page.route('**/assistant/**', route => route.abort())
  await page.goto('/settings/indicators-models')
  await expect(page.getByRole('heading', { name: /指标/ }).first()).toBeVisible()
  const name = page.getByLabel('名称', { exact: true })
  if (!await name.isVisible()) await page.getByRole('button', { name: '新建指标', exact: true }).click()
  if (!await name.isVisible()) await page.getByRole('tab', { name: '编辑', exact: true }).click()
  await expect(name).toBeEnabled()
  await name.fill('助手离线时的手工草稿')
  await expect(name).toHaveValue('助手离线时的手工草稿')
})

for (const [capability, route] of [
  ['product-research', '/product-research/products'],
  ['product-compare', '/product-research/compare'],
  ['holding-diagnosis', '/post-investment/research-diagnosis'],
] as const) {
  test(`研究页面真实组件与宿主授权：${capability}`, async ({ page, request }) => {
    const run = await (await request.get('http://127.0.0.1:18080/fixture/portfolio-run')).json()
    await page.goto(route+(capability === 'holding-diagnosis' ? `?run=${run.id}` : ''))
    const agent = page.locator('portable-agent')
    await expect(agent).toHaveCount(1)
    await agent.getByRole('button', { name: '打开 AI 助手', exact: true }).click()
    await expect(agent.getByRole('button', { name: '发送', exact: true })).toBeEnabled()
    await agent.getByLabel('发送消息', { exact: true }).fill('读取当前页面请求，保持研究条件不变')
    await agent.getByRole('button', { name: '发送', exact: true }).click()
    await expect(agent).toContainText('已完成本轮处理')
    await expect(agent.getByRole('button', { name: '发送', exact: true })).toBeEnabled()
    const captured = await (await request.get('http://127.0.0.1:18080/fixture/model-requests')).json()
    const latest = JSON.stringify(captured.items.at(-1))
    expect(latest).toContain(capability)
    expect(latest).toContain('page_read')
    if (capability === 'holding-diagnosis') {
      expect(latest).toContain(run.id)
      expect(latest).toContain('page_analyze')
      expect(latest).not.toContain('daily_weights')
    }
  })
}

test('人工历史事件仅启用当前事件工作区助手并通过真实授权', async ({ page }, info) => {
  const registrations: Array<Record<string, any>> = []
  page.on('request', request => {
    if (request.url().endsWith('/api/integrations/portable-agent/contexts')) registrations.push(request.postDataJSON())
  })
  await page.goto('/settings/scenario-algorithms?center=events&event_view=manual')
  const agent = page.getByRole('region', { name: '人工历史事件工作区', exact: true }).locator('portable-agent')
  await expect(agent).toHaveCount(1)
  await expect(page.getByRole('button', { name: '打开 AI 助手', exact: true })).toHaveCount(1)
  await agent.getByRole('button', { name: '打开 AI 助手', exact: true }).click()
  await expect(agent.getByRole('button', { name: '发送', exact: true })).toBeEnabled()
  const manual = registrations.filter(item => item.page_context.calculation.purpose === 'manual_events')
  expect(manual.length).toBeGreaterThan(0)
  expect(manual.every(item => item.page_context.page === 'global-events' && item.page_context.calculation.workspace === 'events')).toBe(true)
  await expect(agent.getByText('AGENT_SCOPE_PAGE_MISMATCH')).toHaveCount(0)
  await page.screenshot({ path: info.outputPath('manual-events-assistant.png'), fullPage: true })
})
