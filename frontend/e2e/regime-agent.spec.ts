import { expect, test, type Page } from '@playwright/test'
import { auditTextContrast } from './helpers/contrast'

const definition = { name: 'AI 趋势识别', description: '使用已登记节点的离线验收草稿',
  graph: { nodes: [{ id: 'market', type: 'source.index', label: '指数数据', parameters: { ts_code: '000300.SH', field: 'close' } },
    { id: 'model', type: 'model.threshold', label: '阈值识别', parameters: { upper: 0.1, lower: -0.1 }, inputs: { value: { node_id: 'market', port: 'value' } } }],
    outputs: { state: { node_id: 'model', port: 'state' } }, exposed_node_ids: [] },
  states: [{ id: 'bull', label: '牛市', color: '#16a34a', role: 'positive', order: 1 }, { id: 'bear', label: '熊市', color: '#dc2626', role: 'negative', order: 2 }] }

async function fixture(page: Page) {
  let context: any, run: any = null, events: any[] = [], draft: any = null
  const sent: any[] = [], applied: any[] = []
  await page.addInitScript(() => { (window as any).EventSource = undefined })
  await page.route('**/api/**', route => {
    const req = route.request(), url = new URL(req.url()), path = url.pathname
    const body = req.method() === 'POST' ? req.postDataJSON() : undefined
    const ok = (json: unknown) => route.fulfill({ json })
    if (path === '/api/agent/meta') return ok({ configured: true })
    if (path === '/api/agent/sessions') { context = body.page_context; return ok({ session_id: 'scenario', session_revision: 0 }) }
    if (path.endsWith('/messages')) {
      sent.push(body); context = body.page_context
      draft = { definition, artifact_kind: 'regime_graph', valid: true, stale: false, definition_hash: 'graph', draft_revision: 1 }
      const text = '草稿已校验，请检查后应用到编辑器。'
      run = { run_id: 'scenario-run', session_id: 'scenario', message_id: body.message_id, status: 'completed', phase: 'completed', run_revision: 2,
        session_revision: 1, response: { session_id: 'scenario', session_revision: 1, reply: { text }, draft, artifacts: { draft } } }
      events = [{ seq: 1, type: 'user.message', id: body.message_id, speaker: 'user', text: body.text },
        { seq: 2, type: 'assistant.message', id: 'scenario-run-reply', run_id: 'scenario-run', speaker: 'assistant', text, artifacts: { draft } }]
      return ok(run)
    }
    if (path.endsWith('/events')) return ok({ items: events.filter(e => e.seq > Number(url.searchParams.get('after_seq') || 0)), has_more: false, last_seq: events.length })
    if (path.includes('/runs/')) return ok(run)
    if (path === '/api/agent/sessions/scenario') return ok({ session_id: 'scenario', session_revision: 1, page_context: context, draft, active_run: run, messages: events, events, next_event_seq: 3 })
    if (path.endsWith('/invalidate-context')) return ok(run || {})
    if (path.endsWith('/infer')) { applied.push(body); return ok({ valid: true, errors: [], warnings: [], inferred: { nodes: {} }, temporal_capability: { realtime_supported: true } }) }
    if (path.endsWith('/authoring/resolve')) return ok({ valid: true, definition: body.definition, source: '', diagnostics: [], display_latex: {} })
    if (req.method() === 'GET') return ok({ items: [] })
    return route.fulfill({ status: 400, json: { detail: 'Unexpected mutation in scenario agent acceptance' } })
  })
  return { sent, applied }
}

test('情景列表 AI 提案经人工检查进入编辑器，切换中心后没有隐藏的 AI 浮层', async ({ page }, info) => {
  const api = await fixture(page)
  await page.goto('/settings/scenario-algorithms?center=market-state&stage=historical')
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const dialog = page.getByRole('dialog', { name: 'AI 助手' })
  await expect(dialog).toBeVisible()
  await page.getByRole('button', { name: '设计情景算法', exact: true }).click()
  await expect(page.getByRole('textbox', { name: '发送消息' })).not.toBeEmpty()
  await page.getByRole('button', { name: '发送', exact: true }).click()
  const apply = page.getByRole('button', { name: '打开编辑器检查' })
  await expect(apply).toBeEnabled()
  expect(api.sent[0].page_context.calculation.mode).toBe('retrospective')
  expect(api.sent[0].page_snapshot.sections.editing.definition).toBeNull()
  await page.getByText('查看步骤与参数', { exact: true }).click()
  await expect(dialog.getByText('阈值识别', { exact: true })).toBeVisible()
  await expect(dialog).toHaveCSS('opacity', '1')
  expect(await dialog.evaluate(auditTextContrast)).toEqual([])
  const box = await dialog.boundingBox()
  expect(box!.x).toBeGreaterThanOrEqual(0)
  expect(box!.x + box!.width).toBeLessThanOrEqual(page.viewportSize()!.width + 1)
  await page.screenshot({ path: info.outputPath('scenario-ai-draft.png'), fullPage: true })
  await apply.click()
  await expect(page).toHaveURL(/new=1/)
  await expect(page.getByRole('textbox', { name: '研究名称', exact: true })).toHaveValue('AI 趋势识别')
  expect(api.applied.some(request => request.definition.name === 'AI 趋势识别' && request.mode === 'retrospective')).toBe(true)
  await expect(page.getByRole('button', { name: '打开 AI 助手' })).toHaveCount(1)
  await page.getByRole('tab', { name: /全球历史事件库/ }).click()
  await expect(page.getByRole('button', { name: '打开 AI 助手' })).toHaveCount(0)
})
