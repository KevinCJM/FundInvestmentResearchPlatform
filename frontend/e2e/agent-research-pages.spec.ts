import { expect, test, type Page } from '@playwright/test'
import { auditTextContrast } from './helpers/contrast'

const execution = { execution_backend: 'numba_njit_fixed_signature', backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0, request_time_compilation: 0, njit_required: true, kernel_signatures: { fixture: ['fixed'] } }
const runId = `run-${'a'.repeat(32)}`
const dates = ['2019-01-02', '2019-07-01', '2019-12-30']
const points = dates.map((date, index) => ({ date, value: 1 + index * .1 }))
const items = Array.from({ length: 12 }, (_, index) => ({ ts_code: `5100${String(index).padStart(2, '0')}.SH`, name: `样例产品${index}`, fund_type: '股票型', m_fee: .5, c_fee: .1 }))
const settings = { settings: { active_release_id: null, as_of: '2019-12-31', run_mode: 'RESEARCH' }, effective: { as_of: '2019-12-31', run_mode: 'RESEARCH', no_pit: false, label: '研究日 2019-12-31' }, release: null, release_error: null, available_releases: [], can_apply: true }
const ranges = Object.fromEntries(['performance', 'risk', 'efficiency'].map(key => [key, {
  window: { start_date: dates[0], end_date: dates[2], observation_count: 3 },
  metrics: { cumulativeReturn: 20, annualizedReturn: 20, volatility: 2, maxDrawdown: 0, totalFee: .6, returnToFee: 33, sharpeRatio: 2, calmarRatio: null },
  normalized_nav: points, drawdown: points.map(point => ({ ...point, value: 0 })), rolling_volatility: points,
}]))

async function fixture(page: Page) {
  const messages: any[] = [], decisions: any[] = [], cancels: string[] = []
  const sessions = new Map<string, any>()
  let memory: any[] = []
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    const body = route.request().method() === 'POST' ? route.request().postDataJSON() : null
    const json = (value: unknown, status = 200) => route.fulfill({ status, json: value })
    if (path === '/api/pit/settings') return json(settings)
    if (path === '/api/custom-indicators/meta') return json({ periods: [{ value: '1Y', label: '近1年' }], limits: {} })
    if (path === '/api/custom-indicators') return json({ items: [], total: 0 })
    if (path === '/api/instruments/products') return json({ items, total: 42, page: 1, page_size: 20, summary: { universe_total: 42, filtered_total: 42 }, available_filters: {}, condition_fields: [], condition_operators: [], snapshot_metric_fields: [], pit: { as_of: '2019-12-31', snapshot_is_hindsight: false }, execution })
    if (path.endsWith('/compare-analysis')) return json({ schema_version: 1, product_id: path.split('/')[4], ranges, execution })
    if (path.startsWith('/api/instruments/products/')) return json({ product_id: path.split('/').pop(), name: '真实页面样例', base_info: { ts_code: path.split('/').pop(), fund_type: '股票型' }, metrics: { m_fee: .5, c_fee: .1 }, timeseries: dates.map((date, index) => ({ date, close: 1 + index * .1, volume: 987654.321 })) })
    if (path === `/api/portfolio-runs/${runId}`) return json({ id: runId, target_id: 'target-1', target_name: '固定组合', target_revision: 3, requested_as_of: '2019-12-31', effective_as_of: dates[2], dates, portfolio_nav: points.map(point => point.value), drawdown: [0, 0, 0], assets: [{ product_id: '510000.SH' }], weight_path: [{ effective_date: dates[0], weights: { '510000.SH': 1 } }], summary: { cumulative_return: .2 }, warnings: [], execution })
    if (path === `/api/portfolio-runs/${runId}/diagnose`) return json({ summary: { cumulative_return: .2 }, components: [{ product_id: '510000.SH', name: '样例持仓', weight: 1 }], contributions: [{ product_id: '510000.SH', name: '样例持仓', contribution: .2 }], concentration: [], covariance: { labels: ['样例持仓'], values: [[.1]] }, correlation: { labels: ['样例持仓'], values: [[1]] }, weight_path: [], warnings: [], execution })
    if (path === `/api/portfolio-runs/${runId}/scenario`) return json({ name: '历史压力区间', metrics: [{ name: '情景收益', value: -.1 }], warnings: [], execution })
    if (path === '/api/agent/meta') return json({ configured: true, model: 'offline-fixture' })
    if (path === '/api/agent/sessions') {
      const id = `session-${sessions.size + 1}`
      const session = { session_id: id, session_revision: 0, page_context: body.page_context, messages: [], next_event_seq: 1, memory_sources: memory, memory_proposals: [] }
      sessions.set(id, session); return json(session)
    }
    if (path.startsWith('/api/agent/sessions/')) {
      const id = path.split('/')[4], session = sessions.get(id)
      if (!session) return json({ detail: 'missing' }, 404)
      if (path.endsWith('/messages')) {
        messages.push(body)
        const currentRun = { run_id: `agent-run-${messages.length}`, session_id: id, message_id: body.message_id, session_revision: ++session.session_revision, run_revision: 1, status: body.text.includes('持续') ? 'running' : 'completed', phase: 'thinking', response: { reply: { text: '已核对当前页面冻结参数。' } } }
        session.page_context = body.page_context; session.active_run = currentRun
        session.messages = [{ id: body.message_id, speaker: 'user', text: body.text, run_id: currentRun.run_id }, { id: `${currentRun.run_id}-reply`, speaker: 'assistant', text: currentRun.response.reply.text, run_id: currentRun.run_id }]
        if (body.text.includes('记住')) session.memory_proposals = [{ proposal_id: `proposal-${messages.length}`, status: 'pending', summary: body.text, source_message_id: body.message_id, key: 'answer.style', object_id: 'scope' }]
        return json(currentRun, 202)
      }
      if (path.endsWith('/memory/revoke')) { decisions.push(body); memory = []; session.memory_sources = []; session.session_revision++; return json({ revoked: true, session_id: id, session_revision: session.session_revision }) }
      if (path.endsWith('/memory') && body) {
        decisions.push(body)
        const proposal = session.memory_proposals.find((item: any) => item.proposal_id === body.proposal_id)
        proposal.status = body.decision === 'accept' ? 'accepted' : 'rejected'
        if (body.decision === 'accept') memory = [{ memory_id: 'memory-1', version: 1, scope: 'product_research', object_id: 'scope', key: proposal.key, text: proposal.summary, source_session_id: id, source_message_id: proposal.source_message_id, accepted_at: '2026-09-21T00:00:00Z' }]
        session.memory_sources = memory; session.session_revision++; return json({ accepted: body.decision === 'accept', session_id: id, session_revision: session.session_revision })
      }
      if (path.endsWith('/memory')) return json({ items: memory })
      if (path.endsWith('/events')) return json({ items: [], has_more: false, last_seq: 0, next_event_seq: 1 })
      if (path.endsWith('/cancel') || path.endsWith('/invalidate-context')) { cancels.push(path); session.active_run.status = 'cancelled'; return json(session.active_run) }
      if (path.includes('/runs/')) return json(session.active_run)
      return json(session)
    }
    return json({ detail: 'offline fixture unavailable' }, 503)
  })
  return { messages, decisions, cancels }
}

async function send(page: Page, text: string) {
  const input = page.getByRole('textbox', { name: '发送消息', exact: true })
  await input.fill(text)
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect(page.getByRole('log')).toContainText('已核对当前页面冻结参数')
}

for (const host of [
  { page: 'product-research', url: '/product-research/products?page_size=20', title: '样例产品0' },
  { page: 'product-compare', url: '/product-research/compare?ids=510000.SH,000001.OF&kinds=etf,fund', title: '基础信息对比' },
  { page: 'holding-diagnosis', url: `/post-investment/research-diagnosis?run=${runId}`, title: '持仓诊断：固定组合' },
]) test(`${host.page} 真实页面冻结证据和共享助手三视口交互`, async ({ page }, info) => {
  const capture = await fixture(page)
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.goto(host.url)
  await expect(page.getByText(host.title, { exact: true }).first()).toBeVisible()
  const launcher = page.getByRole('button', { name: '打开 AI 助手', exact: true })
  await expect(launcher).toHaveCount(1)
  expect(capture.messages).toHaveLength(0)
  if (host.page !== 'product-research') await expect(page.locator('canvas').first()).toBeVisible()
  await launcher.click()
  if (host.page === 'product-research') await expect(page.getByRole('dialog').getByRole('status')).toContainText('本页前 10 项，本页共 12 项')
  await send(page, '解释当前页面的研究口径')
  const body = capture.messages[0]
  expect(body.page_snapshot.page).toBe(host.page)
  expect(JSON.stringify(body)).not.toContain('987654.321')
  expect(JSON.stringify(body)).not.toMatch(/normalized_nav|portfolio_nav|contribution_series|weight_path/)
  if (host.page === 'product-compare') expect(body.page_snapshot.sections.request.targets.map((target: any) => target.kind)).toEqual(['etf', 'fund'])
  if (host.page === 'holding-diagnosis') expect(body.page_context.calculation).toEqual({ context_kind: 'portfolio', run_id: runId })
  const dialog = page.getByRole('dialog', { name: 'AI 助手', exact: true })
  const box = (await dialog.boundingBox())!
  expect(box.x).toBeGreaterThanOrEqual(0)
  expect(box.x + box.width).toBeLessThanOrEqual(page.viewportSize()!.width)
  await expect(page.getByRole('button', { name: '关闭 AI 助手', exact: true })).toBeInViewport()
  const contrast = await page.evaluate(auditTextContrast)
  expect(contrast.filter(item => /当前分析批次|已核对当前页面|发送消息|AI 助手/.test(item.text))).toEqual([])
  await info.attach('contrast-review.json', { body: JSON.stringify(contrast), contentType: 'application/json' })
  await page.screenshot({ path: `../.run/harness-upgrade-20260921/p4-native-${host.page}-${info.project.name}.png`, fullPage: true })
  await page.getByRole('button', { name: '关闭 AI 助手', exact: true }).click()
  await expect(launcher).toBeFocused()
  await launcher.press('Enter')
  await page.getByRole('button', { name: '清空上下文', exact: true }).click()
  await expect(page.getByRole('log')).not.toContainText('解释当前页面')
  await expect(page.getByRole('textbox', { name: '发送消息', exact: true })).toBeFocused()
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow).toBeLessThanOrEqual(1)
})

test('比较三段区间与指标截止日独立，改条件取消运行而关闭面板保留', async ({ page }) => {
  const capture = await fixture(page)
  await page.addInitScript(() => { (window as any).EventSource = undefined })
  await page.goto('/product-research/compare?ids=510000.SH,000001.OF&kinds=etf,fund')
  await expect(page.getByText('基础信息对比')).toBeVisible()
  await page.locator('#risk-range').selectOption('CUSTOM')
  const risk = page.locator('#risk-range').locator('..')
  await risk.locator('input[type=date]').first().fill('2019-07-01')
  await page.locator('#efficiency-range').selectOption('CUSTOM')
  const efficiency = page.locator('#efficiency-range').locator('..')
  await efficiency.locator('input[type=date]').last().fill('2019-07-01')
  await page.getByLabel('截止日', { exact: true }).fill('2020-12-31')
  await page.getByRole('button', { name: '打开 AI 助手', exact: true }).click()
  await send(page, '持续分析当前页面')
  expect(capture.messages[0].page_snapshot.sections.request).toMatchObject({ as_of: '2019-12-31', metrics_as_of: '2020-12-31', ranges: {
    performance: { start_date: dates[0], end_date: dates[2] }, risk: { start_date: dates[1], end_date: dates[2] }, efficiency: { start_date: dates[0], end_date: dates[1] },
  } })
  await page.getByRole('button', { name: '关闭 AI 助手', exact: true }).click()
  expect(capture.cancels).toHaveLength(0)
  await page.getByLabel('截止日', { exact: true }).fill('2021-12-31')
  await expect.poll(() => capture.cancels.length).toBeGreaterThan(0)
  await expect(page.locator('canvas').first()).toBeVisible()
})

test('研究页面记忆独立确认、来源、拒绝和撤销', async ({ page }) => {
  const capture = await fixture(page)
  await page.goto('/product-research/products')
  await expect(page.getByText('样例产品0', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: '打开 AI 助手', exact: true }).click()
  await send(page, '记住：回答时先说明研究日')
  await page.getByText(/偏好与记忆/).click()
  await page.getByRole('button', { name: '接受记忆', exact: true }).click()
  await expect(page.getByRole('button', { name: '撤销记忆', exact: true })).toBeVisible()
  await page.getByText('查看来源记录', { exact: true }).click()
  await expect(page.getByText(/session-1 \//)).toBeVisible()
  await page.getByRole('button', { name: '撤销记忆', exact: true }).click()
  await expect(page.getByRole('button', { name: '撤销记忆', exact: true })).toHaveCount(0)
  await send(page, '记住：使用简短说明')
  await page.getByText(/偏好与记忆/).click()
  await page.getByRole('button', { name: '不记住', exact: true }).click()
  await expect(page.getByRole('button', { name: '接受记忆', exact: true })).toHaveCount(0)
  expect(capture.decisions.map(value => value.decision || 'revoke')).toEqual(['accept', 'revoke', 'reject'])
})

test('没有组合快照时不能发送，保留构建入口', async ({ page }) => {
  const capture = await fixture(page)
  await page.goto('/post-investment/research-diagnosis')
  await expect(page.getByRole('link', { name: '去构建组合', exact: true })).toBeVisible()
  await page.getByRole('button', { name: '打开 AI 助手', exact: true }).click()
  await page.getByRole('textbox', { name: '发送消息', exact: true }).fill('分析持仓')
  await expect(page.getByRole('button', { name: '发送', exact: true })).toBeDisabled()
  expect(capture.messages).toHaveLength(0)
})

test('持仓情景首次失败和复算失败如实传给助手，成功重试恢复就绪', async ({ page }, info) => {
  const capture = await fixture(page)
  let attempt = 0
  await page.route(`**/api/portfolio-runs/${runId}/scenario`, route => {
    attempt += 1
    return attempt % 2
      ? route.fulfill({ status: 503, json: { detail: '情景服务不可用' } })
      : route.fulfill({ json: { name: '历史压力区间', metrics: [{ name: '情景收益', value: -.1 }], warnings: [], execution } })
  })
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.goto(`/post-investment/research-diagnosis?run=${runId}`)
  await expect(page.getByText('持仓诊断：固定组合')).toBeVisible()
  for (const [index, status] of ['error', 'ready', 'error', 'ready'].entries()) {
    await page.getByRole('button', { name: '运行情景', exact: true }).click()
    if (status === 'error') await expect(page.getByRole('alert')).toContainText('情景服务不可用')
    else {
      await expect(page.getByRole('alert')).toHaveCount(0)
      await expect(page.getByText('情景收益', { exact: false })).toBeVisible()
    }
    await page.getByRole('button', { name: '打开 AI 助手', exact: true }).click()
    await send(page, `解释第${index + 1}次情景结果`)
    await expect.poll(() => capture.messages.length).toBe(index + 1)
    const reference = capture.messages[index].page_snapshot.sections.results.refs.scenario_result
    expect(reference.status).toBe(status)
    if (index === 0) expect(reference.frozen_request).toBeNull()
    else expect(reference.frozen_request.run_id).toBe(runId)
    if (index === 2) {
      const contrast = await page.evaluate(auditTextContrast)
      expect(contrast.filter(item => /情景服务不可用|已核对当前页面/.test(item.text))).toEqual([])
      await info.attach('scenario-error.png', { body: await page.screenshot(), contentType: 'image/png' })
    }
    await page.getByRole('button', { name: '关闭 AI 助手', exact: true }).click()
  }
})
