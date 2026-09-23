import { expect, test, type Page } from '@playwright/test'
import { auditTextContrast } from './helpers/contrast'
import { agentSessionStorageKey } from '../src/services/agentContext'

async function openStudio(page: Page, failed = false) {
  await page.route('**/api/**', route => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/agent/meta') return route.fulfill(failed
      ? { status: 502, json: { detail: { message: '模型服务暂时不可用，请稍后重试。'.repeat(100) } } }
      : { json: { configured: false } })
    if (path === '/api/custom-indicators/meta') return route.fulfill({ json: { variables: [], operators: [], templates: [], periods: [] } })
    return route.fulfill({ json: { items: [] } })
  })
  await page.goto('/settings/indicators-models')
}

async function expectCloseReachable(page: Page) {
  const close = page.getByRole('button', { name: '关闭 AI 助手', exact: true })
  await expect(close).toBeInViewport()
  expect(await close.evaluate(el => {
    const r = el.getBoundingClientRect()
    return el.contains(document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2))
  })).toBe(true)
  const dialog = await page.getByRole('dialog', { name: 'AI 助手' }).boundingBox()
  expect(dialog!.x).toBeGreaterThanOrEqual(0)
  expect(dialog!.y).toBeGreaterThanOrEqual(0)
  expect(dialog!.x + dialog!.width).toBeLessThanOrEqual(page.viewportSize()!.width)
  await expect(page.getByRole('textbox', { name: '发送消息' })).toBeInViewport()
  const send = page.getByRole('button', { name: '发送', exact: true })
  if (await send.isVisible()) await expect(send).toBeInViewport()
}

test('小牛入口、浮窗关闭和 Esc 在页面滚动后仍可用', async ({ page }) => {
  await openStudio(page)
  const launcher = page.getByRole('button', { name: '打开 AI 助手' })
  await expect(launcher.locator('img')).toBeVisible()
  await launcher.click()
  await expect(launcher).toHaveCount(0)
  await expect(page.getByRole('button', { name: '收起 AI 助手' })).toHaveCount(0)
  const headerCow = page.getByRole('dialog', { name: 'AI 助手' }).locator('header img')
  await expect(headerCow).toBeVisible()
  expect(await headerCow.evaluate(el => el.getBoundingClientRect().width)).toBe(32)
  await expect(page.getByRole('link', { name: '前往 LLM API 配置' })).toBeVisible()
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight))
  await expectCloseReachable(page)
  await expect(page.locator('img[src*="mascot-"]')).toHaveCount(1)
  await expect(page.getByRole('button', { name: '发送', exact: true })).toBeDisabled()
  // Sample contrast only after the 180ms enter animation settled; a mid-fade ancestor opacity would
  // otherwise composite every text color against white and report false failures.
  await expect(page.getByRole('dialog', { name: 'AI 助手' })).toHaveCSS('opacity', '1')
  const failures = await page.evaluate(auditTextContrast)
  expect(failures.filter(item => ['AI 助手', '关闭', '保存需确认', '想一起研究什么？', '前往 LLM API 配置'].includes(item.text))).toEqual([])
  await page.getByRole('button', { name: '关闭 AI 助手', exact: true }).click()
  await expect(page.getByRole('dialog', { name: 'AI 助手' })).toHaveCount(0)
  await expect(launcher).toBeFocused()
  await launcher.click()
  await page.keyboard.press('Escape')
  await expect(launcher).toBeFocused()
  await launcher.click()
  await page.getByRole('button', { name: '关闭 AI 助手' }).click()
  await expect(page.getByRole('dialog', { name: 'AI 助手' })).toHaveCount(0)
})

test('长内容与短视口不挤走关闭和输入区', async ({ page }) => {
  await page.setViewportSize({ width: page.viewportSize()!.width, height: page.viewportSize()!.width === 320 ? 360 : 480 })
  await openStudio(page, true)
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await expect(page.getByRole('alert')).toContainText('模型服务暂时不可用')
  await page.getByRole('alert').evaluate(el => el.parentElement!.scrollTo(0, el.parentElement!.scrollHeight))
  await expectCloseReachable(page)
  await page.getByRole('button', { name: '关闭 AI 助手', exact: true }).click()
  await expect(page.getByRole('dialog', { name: 'AI 助手' })).toHaveCount(0)
})

test('320px短视口展开上下文时，六行输入与发送关闭仍可点击', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 320 })
  await openStudio(page)
  await agentApi(page)
  await page.getByRole('button', { name: '打开 AI 助手', exact: true }).click()
  const dialog = page.getByRole('dialog', { name: 'AI 助手', exact: true })
  await test.info().attach('compact-layout.json', {
    body: JSON.stringify(await dialog.evaluate(el => ({
      headerHeight: el.querySelector('header')!.getBoundingClientRect().height,
      composerHeight: el.querySelector('form')!.getBoundingClientRect().height,
    }))),
    contentType: 'application/json',
  })
  const input = dialog.getByRole('textbox', { name: '发送消息', exact: true })
  await input.fill(Array.from({ length: 6 }, (_, index) => `第${index + 1}行：保留研究需求和指标计算口径。`).join('\n'))
  const info = dialog.getByRole('button', { name: '查看上下文与模型', exact: true })
  await info.click()
  await expect(dialog.getByRole('button', { name: '输入帮助', exact: true })).toHaveCount(0)
  await expect(info).toHaveAttribute('aria-expanded', 'true')
  for (const control of [input, dialog.getByRole('button', { name: '发送', exact: true }), dialog.getByRole('button', { name: '关闭 AI 助手', exact: true })]) {
    await expect(control).toBeInViewport()
    const box = (await control.boundingBox())!
    expect(box.x).toBeGreaterThanOrEqual(0)
    expect(box.y).toBeGreaterThanOrEqual(0)
    expect(box.x + box.width).toBeLessThanOrEqual(320)
    expect(box.y + box.height).toBeLessThanOrEqual(320)
    expect(await control.evaluate(el => {
      const rect = el.getBoundingClientRect()
      return el.contains(document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2))
    })).toBe(true)
  }
  await page.mouse.move(0, 0)
  await dialog.screenshot({ path: test.info().outputPath('short-viewport-expanded-context.png') })
})

test('图片不可用时仍能通过文字和键盘打开关闭', async ({ page }) => {
  await page.route('**/mascot-welcome-240.webp', route => route.abort())
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await openStudio(page)
  const launcher = page.getByRole('button', { name: '打开 AI 助手' })
  await launcher.focus()
  await page.keyboard.press('Enter')
  await expectCloseReachable(page)
  await page.keyboard.press('Escape')
  await expect(launcher).toBeFocused()
})

async function agentApi(page: Page, options: { delay?: boolean; running?: boolean; model?: string; loseCommitReply?: boolean; loseMessageReply?: boolean; memory?: boolean; loseMemoryReply?: 'accept' | 'reject' | 'revoke'; nameConflict?: boolean } = {}) {
  let revision = 0, context: Record<string, unknown> = {}, run: Record<string, any> | null = null
  let sessionId = 's', sessionCount = 0
  let draft: Record<string, unknown> | null = null
  let events: Array<Record<string, any>> = []
  const deliveredEvents: Array<Record<string, any>> = []
  let saved = false
  let commitWrites = 0
  let memoryProposal: Record<string, unknown> | null = null
  let memoryReadsUnavailable = false
  const revokedMemoryIds = new Set<string>()
  const memoryDecisions: string[] = []
  const recalledMemory = options.memory ? { memory_id: 'memory-existing', version: 1,
    scope: 'indicator_center', object_id: 'scope', key: 'reply.detail', text: '回答先给结论。',
    source_session_id: 'earlier-session', source_message_id: 'earlier-message', accepted_at: '2026-09-21' } : null
  let release: (() => void) | null = null
  const wait = new Promise<void>(resolve => { release = resolve })
  const requests: any[] = []
  const accepted = new Map<string, { identity: string; run: Record<string, any> }>()
  const definition = { name: '区间平均价差', expression: 'mean(adjusted_high - adjusted_low)', context_kind: 'single_product' }
  const add = (event: Record<string, unknown>) => events.push({ ...event, seq: events.length + 1 })
  const finish = (text: string, status = 'completed') => {
    if (!run) return
    run.status = status
    draft ??= revision > 1 ? { valid: true, draft_revision: 1, definition_hash: 'hash', definition } : null
    run.response = { session_id: sessionId, session_revision: revision, reply: { text }, draft, artifacts: { draft } }
    add({ type: 'assistant.message', id: `${run.run_id}-reply`, run_id: run.run_id, speaker: 'assistant', text, artifacts: { draft } })
    add({ type: `run.${status}`, run_id: run.run_id, data: { status } })
  }
  await page.route('**/api/agent/**', async route => {
    const url = new URL(route.request().url()), path = url.pathname
    if (path.endsWith('/meta')) return route.fulfill({ json: { configured: true, model: options.model || 'deepseek-v4.1-flash' } })
    if (path.endsWith('/sessions')) {
      context = route.request().postDataJSON().page_context
      sessionId = ++sessionCount === 1 ? 's' : `s${sessionCount}`
      revision = 0; run = null; draft = null; events = []; saved = false
      accepted.clear()
      return route.fulfill({ json: { session_id: sessionId, session_revision: 0 } })
    }
    if (path.endsWith('/messages')) {
      const body = route.request().postDataJSON(); requests.push(body)
      const { expected_session_revision, ...identity } = body
      const prior = accepted.get(body.message_id)
      if (prior) return route.fulfill(prior.identity === JSON.stringify(identity)
        ? { status: 202, json: prior.run }
        : { status: 409, json: { detail: { code: 'REVISION_CONFLICT', message: '同一消息标识不能提交不同内容。' } } })
      if (expected_session_revision !== revision) return route.fulfill({ status: 409,
        json: { detail: { code: 'REVISION_CONFLICT', message: '会话已更新，请刷新后重试。' } } })
      expect(body.page_context.calculation.targets).toEqual([])
      if (options.delay && requests.length === 1) {
        await wait
        return route.fulfill({ status: 502, json: { detail: { message: '模型接口响应超时，请重试。' } } })
      }
      if (body.edit_of_message_id) {
        // The server replaces the stopped turn: neither its text nor its stop notice stays in the transcript.
        events = events.filter(event => event.id !== body.edit_of_message_id && event.run_id !== run?.run_id)
        run = null
      }
      context = body.page_context
      revision += 1
      add({ type: 'user.message', id: body.message_id, speaker: 'user', text: body.text })
      run = { run_id: `r${revision}`, session_id: sessionId, message_id: body.message_id, session_revision: revision, run_revision: 1, status: 'running', phase: 'thinking' }
      add({ type: 'run.started', run_id: run.run_id, data: { status: 'running', phase: 'thinking' } })
      if (!options.running) finish(revision === 1 ? '请确认是交易价格还是总市值？' : '已生成公式，尚未对产品试算。')
      accepted.set(body.message_id, { identity: JSON.stringify(identity), run })
      if (options.loseMessageReply && requests.length === 2) return route.abort('failed')
      return route.fulfill({ status: 202, json: run })
    }
    if (path.endsWith('/events')) {
      // Exercise the supported polling fallback; stream disconnect must not restart a run.
      if (url.searchParams.get('stream')) return route.fulfill({ status: 503, json: {} })
      const after = Number(url.searchParams.get('after_seq'))
      const items = events.filter(e => e.seq > after)
      deliveredEvents.push(...structuredClone(items))
      return route.fulfill({ json: { items, last_seq: items.at(-1)?.seq || after, has_more: false, next_event_seq: events.length + 1 } })
    }
    if (path.endsWith('/cancel')) { finish('已停止，已提交的进度已保留。', 'cancelled'); return route.fulfill({ json: run }) }
    if (path.includes('/runs/')) return route.fulfill({ json: run })
    if (path.endsWith('/commit-preview')) {
      expect(route.request().postDataJSON()).not.toHaveProperty('target')
      revision++
      return route.fulfill({ json: { session_id: sessionId, session_revision: revision, confirmation_id: 'c', definition_hash: 'hash', draft_revision: 1, definition, preview_status: 'valid', impact: { action: 'create', name: definition.name, context_kind: 'single_product', target: null, name_conflict_indicator_id: options.nameConflict ? 'existing' : null } } })
    }
    if (path.endsWith('/commit')) {
      saved = true; commitWrites++; revision++
      if (options.loseCommitReply) return route.abort('failed')
      return route.fulfill({ json: { session_id: sessionId, session_revision: revision, indicator_id: 'i', revision: 1 } })
    }
    if (path.endsWith('/memory/revoke')) {
      memoryDecisions.push('revoke')
      revokedMemoryIds.add(route.request().postDataJSON().memory_id)
      revision++
      if (options.loseMemoryReply === 'revoke') { memoryReadsUnavailable = true; return route.abort('failed') }
      return route.fulfill({ json: { revoked: true } })
    }
    if (path.endsWith('/memory')) {
      if (route.request().method() === 'POST') {
        const decision = route.request().postDataJSON().decision
        memoryDecisions.push(decision)
        if (memoryProposal) memoryProposal.status = decision === 'accept' ? 'accepted' : 'rejected'
        revision++
        if (options.loseMemoryReply === decision) { memoryReadsUnavailable = true; return route.abort('failed') }
        return route.fulfill({ json: { decision } })
      }
      if (memoryReadsUnavailable) return route.fulfill({ status: 503, json: { detail: { message: '记忆暂时不可用，请重新读取。' } } })
      const items = memoryProposal?.status === 'accepted' ? [{ memory_id: 'memory-one', version: 1,
        scope: 'indicator_center', object_id: 'scope', key: 'reply.language', text: memoryProposal.summary,
        source_session_id: sessionId, source_message_id: run?.message_id, accepted_at: '2026-09-22' }] : recalledMemory ? [recalledMemory] : []
      return route.fulfill({ json: { items: items.filter(item => !revokedMemoryIds.has(item.memory_id)) } })
    }
    if (path.endsWith(`/sessions/${sessionId}`) && memoryReadsUnavailable) return route.fulfill({ status: 503, json: { detail: { message: '记忆暂时不可用，请重新读取。' } } })
    if (path.endsWith(`/sessions/${sessionId}`)) return route.fulfill({ json: { session_id: sessionId, session_revision: revision, page_context: context, next_event_seq: events.length + 1, events, active_run: run, draft,
      memory_proposals: memoryProposal ? [memoryProposal] : [], memory_sources: recalledMemory ? [recalledMemory] : [],
      saved_commit: saved ? { definition_hash: 'hash', indicator_id: 'i', revision: 1 } : null } })
    return route.fulfill({ json: {} })
  })
  const updateDraft = (valid?: boolean, stale = false) => {
    draft = { valid, stale, draft_revision: Number(draft?.draft_revision || 0) + 1, definition_hash: 'hash', definition,
      diagnostics: valid ? [] : [{ code: 'UNUSED_PARAMETER', message: '参数 window 没有被公式使用，请删除或重新关联。' }] }
    if (run) run.artifacts = { draft }
    add({ type: 'draft.updated', run_id: run?.run_id, data: { draft } })
  }
  const setPhase = (phase: string) => { if (run) { run.phase = phase; run.run_revision += 1; add({ type: 'run.phase', run_id: run.run_id, data: { status: 'running', phase } }) } }
  const toolEvent = (type: 'tool.started' | 'tool.completed') => {
    if (run) add({ type, run_id: run.run_id, data: { tool: 'metrics.validate', status: 'ok', ...(type === 'tool.completed' ? { duration_ms: 7 } : {}) } })
  }
  return { setPhase, toolEvent, requests, definition, deliveredEvents, runSnapshot: () => structuredClone(run), release: () => release?.(), saved: () => saved, commitWrites: () => commitWrites, updateDraft, finish,
    memoryDecisions, restoreMemoryReads: () => { memoryReadsUnavailable = false }, proposeMemory: () => {
      memoryProposal = { proposal_id: 'proposal-one', status: 'pending', summary: '以后请用简洁中文回答。', key: 'reply.language', object_id: 'scope' }
      finish('请确认是否记住。')
    },
    advanceRevision: () => { revision++ } }
}

for (const failRestore of [false, true]) test(`记忆提案结束后即时确认${failRestore ? '，读取失败可重连' : ''}`, async ({ page }) => {
  await openStudio(page)
  const api = await agentApi(page, { running: true, memory: true })
  let fail = false
  await page.route('**/api/agent/sessions/s', route => fail
    ? route.fulfill({ status: 503, json: { detail: { message: '会话暂时不可用' } } }) : route.fallback())
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const input = page.getByRole('textbox', { name: '发送消息' })
  await input.fill('以后请用简洁中文回答。')
  const connected = page.waitForResponse(response => response.url().includes('/runs/r1'))
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await connected
  await expect(page.getByRole('button', { name: '停止', exact: true })).toBeVisible()
  const initialMemory = page.getByText('偏好与记忆（0 项待确认）', { exact: true })
  await initialMemory.click()
  await expect(page.getByText('回答先给结论。', { exact: true })).toBeVisible()
  await expect(page.getByText('本轮已引用的确认偏好', { exact: true })).toBeVisible()
  await initialMemory.click()
  await input.fill('保留未发送的补充')
  fail = failRestore
  api.proposeMemory()
  await expect(page.getByText('请确认是否记住。', { exact: true })).toBeVisible()
  if (failRestore) {
    const reconnect = page.getByRole('button', { name: '重新连接', exact: true })
    await expect(reconnect).toBeVisible()
    fail = false
    await reconnect.click()
    await expect(reconnect).toHaveCount(0)
  }
  const summary = page.getByText('偏好与记忆（1 项待确认）', { exact: true })
  await expect(summary).toBeVisible()
  await summary.focus(); await page.keyboard.press('Enter')
  const accept = page.getByRole('button', { name: '接受记忆', exact: true })
  await expect(accept).toBeEnabled()
  await accept.scrollIntoViewIfNeeded()
  await expectCloseReachable(page)
  expect(api.memoryDecisions).toEqual([])
  expect(api.commitWrites()).toBe(0)
  expect((await page.evaluate(auditTextContrast)).filter(item => /偏好|记忆|简洁中文|用户原话/.test(item.text))).toEqual([])
  await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: test.info().outputPath('memory-confirmation.png') })
  await accept.focus(); await page.keyboard.press('Enter')
  await expect.poll(() => api.memoryDecisions).toEqual(['accept'])
  await expect(page.getByRole('button', { name: '撤销记忆', exact: true })).toBeEnabled()
  await expect(summary).toHaveCount(0)
  await expect(input).toHaveValue('保留未发送的补充')
  expect(api.requests).toHaveLength(1)
  expect(api.commitWrites()).toBe(0)
})

for (const decision of ['accept', 'reject', 'revoke'] as const) test(`记忆${decision}响应丢失后重新读取恢复`, async ({ page }) => {
  await openStudio(page)
  const api = await agentApi(page, { running: true, memory: true, loseMemoryReply: decision })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const input = page.getByRole('textbox', { name: '发送消息' })
  await input.fill('以后请用简洁中文回答。')
  const connected = page.waitForResponse(response => response.url().includes('/runs/r1'))
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await connected
  if (decision === 'revoke') api.finish('本轮研究已完成。')
  else api.proposeMemory()
  await expect(page.getByText(decision === 'revoke' ? '本轮研究已完成。' : '请确认是否记住。', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: '停止', exact: true })).toHaveCount(0)
  const summary = page.getByText(`偏好与记忆（${decision === 'revoke' ? 0 : 1} 项待确认）`, { exact: true })
  await expect(summary).toBeVisible()
  await summary.click()
  const memory = page.locator('details').filter({ has: page.locator('summary').filter({ hasText: /^偏好与记忆（/ }) })
  const actionName = { accept: '接受记忆', reject: '不记住', revoke: '撤销记忆' }[decision]
  const action = memory.getByRole('button', { name: actionName, exact: true })
  await expect(action).toBeEnabled()
  await input.fill('保留未发送的补充')
  await action.click()
  await expect.poll(() => api.memoryDecisions).toEqual([decision])
  const retry = memory.getByRole('button', { name: '重新读取', exact: true })
  await expect(retry).toBeVisible()
  await retry.click()
  await expect(memory.getByRole('alert')).toContainText('记忆暂时不可用')
  await expect(action).toBeDisabled()
  await expectCloseReachable(page)
  expect((await page.evaluate(auditTextContrast)).filter(item => /记忆|重新读取/.test(item.text))).toEqual([])
  await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: test.info().outputPath(`memory-${decision}-recovery-failed.png`) })
  api.restoreMemoryReads()
  await retry.focus(); await page.keyboard.press('Enter')
  await expect(memory.getByRole('alert')).toHaveCount(0)
  await expect(memory.getByRole('button', { name: '接受记忆', exact: true })).toHaveCount(0)
  await expect(memory.getByRole('button', { name: '不记住', exact: true })).toHaveCount(0)
  if (decision === 'revoke') {
    await expect(memory.getByRole('button', { name: '撤销记忆', exact: true })).toHaveCount(0)
    await expect(memory.getByText('回答先给结论。', { exact: true })).toHaveCount(0)
  } else {
    await expect(memory.getByRole('button', { name: '撤销记忆', exact: true })).toBeEnabled()
    await expect(memory.getByText(decision === 'accept' ? '以后请用简洁中文回答。' : '回答先给结论。', { exact: true })).toBeVisible()
  }
  await expect(input).toHaveValue('保留未发送的补充')
  await expectCloseReachable(page)
  await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: test.info().outputPath(`memory-${decision}-recovered.png`) })
  expect(api.memoryDecisions).toEqual([decision])
  expect(api.requests).toHaveLength(1)
  expect(api.commitWrites()).toBe(0)
})

test('会话恢复完成前保留输入并禁用发送，恢复后沿用原会话', async ({ page }) => {
  await openStudio(page)
  const api = await agentApi(page)
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const input = page.getByRole('textbox', { name: '发送消息' })
  await input.fill('原研究问题'); await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect(page.getByText('请确认是交易价格还是总市值？', { exact: true })).toBeVisible()
  let release!: () => void
  const gate = new Promise<void>(resolve => { release = resolve })
  await page.route('**/api/agent/sessions/s', async route => { await gate; await route.fallback() })
  await page.reload()
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await input.fill('恢复后继续研究')
  await expect(page.getByRole('button', { name: '发送', exact: true })).toBeDisabled()
  await input.press('Control+Enter')
  expect(api.requests).toHaveLength(1)
  await expectCloseReachable(page)
  await expect(page.getByRole('dialog', { name: 'AI 助手' })).toHaveCSS('opacity', '1')
  await expect(page.getByText('正在加载…', { exact: true })).toBeVisible()
  expect((await page.evaluate(auditTextContrast)).filter(item => item.text === '正在加载…')).toEqual([])
  await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: test.info().outputPath('session-restoring.png') })
  release()
  await expect(page.getByRole('button', { name: '发送', exact: true })).toBeEnabled()
  await expect(input).toHaveValue('恢复后继续研究')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect.poll(() => api.requests.length).toBe(2)
  expect(api.requests[1].expected_session_revision).toBe(1)
  expect(await page.evaluate(() => sessionStorage.getItem('indicator-agent-session:indicator-studio:single_product'))).toBe('s')
})

test('会话版本冲突后刷新并重试同一消息', async ({ page }) => {
  await openStudio(page)
  const api = await agentApi(page)
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const input = page.getByRole('textbox', { name: '发送消息' })
  await input.fill('原研究问题'); await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect(page.getByText('请确认是交易价格还是总市值？', { exact: true })).toBeVisible()
  api.advanceRevision()
  await input.fill('新的研究问题'); await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect(page.getByText('会话已更新，请刷新后重试。', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: '重试这条消息' }).click()
  await expect(page.getByText('已生成公式，尚未对产品试算。', { exact: true })).toBeVisible()
  expect(api.requests).toHaveLength(3)
  expect(api.requests[2]).toEqual({ ...api.requests[1], expected_session_revision: 2 })
  await expectCloseReachable(page)
})

test('会话续接响应丢失后恢复并重试，原请求身份不变', async ({ page }) => {
  await openStudio(page)
  const api = await agentApi(page, { loseMessageReply: true })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await page.getByRole('textbox', { name: '发送消息' }).fill('原研究问题')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect(page.getByText('请确认是交易价格还是总市值？', { exact: true })).toBeVisible()
  api.finish('本轮已暂停', 'paused')
  await page.getByRole('button', { name: '关闭 AI 助手' }).click()
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await page.getByRole('button', { name: '继续分析' }).click()
  await expect(page.getByRole('button', { name: '重试这条消息' })).toBeVisible()
  await page.getByRole('button', { name: '关闭 AI 助手' }).click()
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await expect(page.getByText('已生成公式，尚未对产品试算。', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: '重试这条消息' }).click()
  await expect(page.getByRole('button', { name: '重试这条消息' })).toHaveCount(0)
  expect(api.requests).toHaveLength(3)
  expect(api.requests[2]).toEqual({ ...api.requests[1], expected_session_revision: 2 })
  expect(api.requests[2].resume_from_run_id).toBe('r1')
  await expectCloseReachable(page)
})

for (const recovery of ['重开浮窗', '刷新页面']) test(`保存回执丢失后${recovery}恢复已保存并刷新指标库，不再次创建`, async ({ page }) => {
  await openStudio(page)
  const api = await agentApi(page, { loseCommitReply: true })
  let refreshed = 0
  await page.route('**/api/custom-indicators?*', route => {
    if (api.saved()) refreshed++
    return route.fulfill({ json: { items: api.saved() ? [{ ...api.definition, id: 'i', revision: 1,
      result_kind: 'scalar', source: 'custom', periods: ['1Y'] }] : [] } })
  })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const input = page.getByRole('textbox', { name: '发送消息' })
  await input.fill('生成平均价差'); await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect(page.getByText('请确认是交易价格还是总市值？', { exact: true })).toBeVisible()
  await input.fill('使用交易价格'); await page.getByRole('button', { name: '发送', exact: true }).click()
  page.once('dialog', dialog => dialog.accept())
  await page.getByRole('button', { name: '确认保存指标', exact: true }).click()
  await expect(page.getByRole('region', { name: '本轮指标草稿' }).getByRole('alert')).toBeVisible()
  if (recovery === '刷新页面') {
    await page.reload()
    await expect(page.locator('#indicator-panel-library').getByRole('button', { name: /区间平均价差/ })).toHaveCount(1)
  }
  else await page.getByRole('button', { name: '关闭 AI 助手' }).click()
  const beforeRestore = refreshed
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await expect(page.getByRole('button', { name: '已保存', exact: true })).toBeDisabled()
  await expect(page.getByRole('region', { name: '本轮指标草稿' }).getByRole('alert')).toHaveCount(0)
  await expect.poll(() => refreshed).toBe(beforeRestore + 1)
  expect(api.commitWrites()).toBe(1)
  expect(api.requests).toHaveLength(2)
  await expectCloseReachable(page)
  await expect(page.getByRole('dialog', { name: 'AI 助手' })).toHaveCSS('opacity', '1')
  const failures = await page.evaluate(auditTextContrast)
  expect(failures.filter(item => item.text.includes('已保存'))).toEqual([])
  await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: test.info().outputPath('restored-saved.png') })
  await page.getByRole('button', { name: '关闭 AI 助手' }).click()
  const library = page.getByRole('tab', { name: '指标库', exact: true })
  if (await library.isVisible()) await library.click()
  await expect(page.locator('#indicator-panel-library').getByRole('button', { name: /区间平均价差/ })).toBeVisible()
  await page.locator('#indicator-panel-library').screenshot({ path: test.info().outputPath('restored-library.png') })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await expect(page.getByRole('button', { name: '已保存', exact: true })).toBeDisabled()
  expect(refreshed).toBe(beforeRestore + 1)
})

test('保存响应丢失后关闭重开再保存只创建一次', async ({ page }) => {
  await openStudio(page)
  await agentApi(page)
  const requests: any[] = [], applied = new Set<string>()
  let previews = 0
  page.on('request', request => { if (new URL(request.url()).pathname.endsWith('/commit-preview')) previews++ })
  await page.route('**/api/agent/sessions/*/commit', async route => {
    const body = route.request().postDataJSON(); requests.push(body); applied.add(body.request_id)
    if (requests.length === 1) return route.abort('failed')
    return route.fulfill({ json: { indicator_id: 'saved-once', revision: 1 } })
  })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const input = page.getByRole('textbox', { name: '发送消息' })
  await input.fill('生成平均价差'); await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect(page.getByText('请确认是交易价格还是总市值？', { exact: true })).toBeVisible()
  await input.fill('使用交易价格'); await page.getByRole('button', { name: '发送', exact: true }).click()
  const save = page.getByRole('button', { name: '确认保存指标', exact: true })
  page.once('dialog', dialog => dialog.accept())
  await save.click()
  await expect(page.getByRole('region', { name: '本轮指标草稿' }).getByRole('alert')).toBeVisible()
  await page.getByRole('button', { name: '关闭 AI 助手', exact: true }).click()
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await save.click()
  await expect(page.getByRole('button', { name: '已保存', exact: true })).toBeDisabled()
  expect(requests).toHaveLength(2); expect(requests[1]).toEqual(requests[0])
  expect(applied.size).toBe(1); expect(previews).toBe(1)
  await expectCloseReachable(page)
  await expect(page.getByRole('dialog', { name: 'AI 助手' })).toHaveCSS('opacity', '1')
  const failures = await page.evaluate(auditTextContrast)
  expect(failures.filter(item => item.text.includes('指标已保存'))).toEqual([])
})

test('清空上下文后刷新不恢复旧消息，新消息进入独立会话', async ({ page }) => {
  await openStudio(page)
  const api = await agentApi(page, { running: true })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const input = page.getByRole('textbox', { name: '发送消息' })
  await input.fill('旧会话的研究口径')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect(page.getByRole('button', { name: '停止', exact: true })).toBeVisible()
  const clear = page.getByRole('button', { name: '清空上下文', exact: true })
  await expect(clear).toBeDisabled()
  await page.getByRole('button', { name: '停止', exact: true }).click()
  await expect(clear).toBeEnabled()
  await page.mouse.move(0, 0)
  await clear.focus()
  await expect(page.getByRole('tooltip', { name: '清空上下文，开始独立的新对话', exact: true })).toBeVisible()
  await page.keyboard.press('Enter')
  await expect(page.getByRole('dialog', { name: 'AI 助手' })).toBeVisible()
  await expect(page.getByText('旧会话的研究口径')).toHaveCount(0)
  await expect(page.getByRole('button', { name: '继续分析' })).toHaveCount(0)
  await expect(input).toHaveValue('')
  await expect(input).toBeFocused()
  await expectCloseReachable(page)
  await page.reload()
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await expect(page.getByText('旧会话的研究口径')).toHaveCount(0)
  const sent = page.waitForRequest(request => request.url().includes('/sessions/s2/messages'))
  await input.fill('全新的研究问题')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  const body = (await sent).postDataJSON()
  expect(body.expected_session_revision).toBe(0)
  expect(body).not.toHaveProperty('resume_from_run_id')
  expect(JSON.stringify(body)).not.toContain('旧会话')
  await page.getByRole('button', { name: '停止', exact: true }).click()
  await expect(clear).toBeEnabled()
  await page.reload()
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await expect(page.getByText('全新的研究问题')).toBeVisible()
  await expect(page.getByText('旧会话的研究口径')).toHaveCount(0)
  expect(api.requests).toHaveLength(2)
  await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: test.info().outputPath('new-conversation.png') })
})

test('浮窗按断点展开阅读，宽度切换与收起保留输入和同一实例', async ({ page }) => {
  await openStudio(page)
  const model = `deepseek-v4.1-flash-${'long-configured-model-'.repeat(12)}`
  const api = await agentApi(page, { model })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const dialog = page.getByRole('dialog', { name: 'AI 助手' })
  const input = page.getByRole('textbox', { name: '发送消息' })
  await input.fill('尚未发送的研究想法')
  const originalInput = await input.elementHandle()
  await expect(dialog.locator('header')).not.toContainText(model)
  const info = dialog.getByRole('button', { name: '查看上下文与模型', exact: true })
  await expect(info).toHaveAttribute('aria-expanded', 'false')
  await info.click()
  await expect(dialog.getByRole('region', { name: '上下文与模型', exact: true }).getByText(`配置模型：${model}`, { exact: true })).toBeVisible()
  await info.click()
  await expect(info).toHaveAttribute('aria-expanded', 'false')
  await expect(dialog).toHaveAttribute('aria-modal', 'false')
  const expand = page.getByRole('button', { name: '展开阅读', exact: true })
  const width = page.viewportSize()!.width
  expect((await dialog.boundingBox())!.width).toBe(width < 640 ? width - 24 : 480)
  if (width >= 1024) {
    await expand.click()
    expect((await dialog.boundingBox())!.width).toBe(720)
    await expect(input).toHaveValue('尚未发送的研究想法')
    expect(await originalInput!.evaluate(el => el.isConnected)).toBe(true)
    await dialog.screenshot({ path: test.info().outputPath('expanded-conversation.png') })
    await page.getByRole('button', { name: '收起宽度', exact: true }).click()
    expect((await dialog.boundingBox())!.width).toBe(480)
    await page.setViewportSize({ width: 1023, height: 800 })
    await expect(expand).toBeHidden()
    await expectCloseReachable(page)
    await page.setViewportSize({ width: 1024, height: 800 })
    await expect(expand).toBeVisible()
  } else await expect(expand).toBeHidden()
  await page.getByRole('button', { name: '关闭 AI 助手' }).click()
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await expect(input).toHaveValue('尚未发送的研究想法')
  expect(await originalInput!.evaluate(el => el.isConnected)).toBe(true)
  expect(api.requests).toHaveLength(0)
  await expectCloseReachable(page)
})

test('Enter 换行，中文输入法不误发，Ctrl 和 Command Enter 各发送一次', async ({ page }) => {
  await openStudio(page)
  const api = await agentApi(page)
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const input = page.getByRole('textbox', { name: '发送消息' })
  await input.fill('第一行')
  await input.press('End')
  await input.press('Enter')
  await expect(input).toHaveValue('第一行\n')
  await input.evaluate(el => el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', ctrlKey: true, isComposing: true, bubbles: true })))
  expect(api.requests).toHaveLength(0)
  await input.press('Control+Enter')
  await expect(page.getByRole('log')).toContainText('请确认是交易价格还是总市值？')
  expect(api.requests).toHaveLength(1)
  await input.fill('确认使用交易价格')
  await input.press('Meta+Enter')
  await expect(page.getByRole('log')).toContainText('已生成公式')
  expect(api.requests).toHaveLength(2)
  await expect(input).toHaveValue('')
})

test('图标控件保留触控尺寸与键盘提示，紧凑布局和减少动态效果生效', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'no-preference' })
  await openStudio(page)
  await agentApi(page)
  await page.evaluate(() => {
    document.documentElement.dataset.agentOpenAnimations = '0'
    document.addEventListener('animationstart', event => {
      if (event.target instanceof HTMLElement && event.target.classList.contains('agent-dialog')) {
        document.documentElement.dataset.agentOpenAnimations = String(Number(document.documentElement.dataset.agentOpenAnimations) + 1)
      }
    })
  })
  const launcher = page.getByRole('button', { name: '打开 AI 助手', exact: true })
  await launcher.click()
  const dialog = page.getByRole('dialog', { name: 'AI 助手', exact: true })
  const input = dialog.getByRole('textbox', { name: '发送消息', exact: true })
  const close = dialog.getByRole('button', { name: '关闭 AI 助手', exact: true })
  const info = dialog.getByRole('button', { name: '查看上下文与模型', exact: true })
  const send = dialog.getByRole('button', { name: '发送', exact: true })
  await expect.poll(() => page.evaluate(() => Number(document.documentElement.dataset.agentOpenAnimations))).toBe(1)
  expect((await dialog.locator('header').boundingBox())!.height).toBeLessThanOrEqual(64)
  expect((await dialog.locator('form').boundingBox())!.height).toBeLessThanOrEqual(110)
  for (const button of [close, info, send]) {
    await expect(button).toHaveText('')
    await expect(button.locator('svg')).toBeVisible()
    // Layout sizes avoid subpixel rounding while the panel is translating in.
    const size = await button.evaluate(el => ({ width: el.offsetWidth, height: el.offsetHeight }))
    expect(size.width).toBeGreaterThanOrEqual(40)
    expect(size.height).toBeGreaterThanOrEqual(40)
  }
  await input.fill('用自然语言讨论指标')
  await send.click()
  await expect(dialog.getByRole('log')).toContainText('请确认是交易价格还是总市值？')
  expect(await page.evaluate(() => Number(document.documentElement.dataset.agentOpenAnimations))).toBe(1)
  await page.mouse.move(0, 0)
  await info.focus()
  const tooltip = page.getByRole('tooltip')
  await expect(tooltip).toContainText('查看上下文与模型')
  await expect(tooltip).toHaveCSS('opacity', '1')
  const tooltipContrast = await page.evaluate(auditTextContrast)
  expect(tooltipContrast.filter(item => item.text === '查看上下文与模型')).toEqual([])
  await page.keyboard.press('Escape')
  await expect(tooltip).toBeHidden()
  await expect(dialog).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
  await expect(launcher).toBeFocused()
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await launcher.click()
  expect(await dialog.evaluate(el => getComputedStyle(el).animationName)).toBe('none')
  expect(await close.evaluate(el => getComputedStyle(el).transitionDuration)).toBe('0s')
  await expectCloseReachable(page)
  await dialog.screenshot({ path: test.info().outputPath('compact-icon-conversation.png') })
})

test('英文助手保留业务原文，成果操作、展开阅读与窄屏布局可用', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('fund-research.i18n.locale', 'en-US'))
  await openStudio(page)
  const api = await agentApi(page)
  const launcher = page.getByRole('button', { name: 'Open AI Assistant', exact: true })
  await launcher.click()
  const dialog = page.getByRole('dialog', { name: 'AI Assistant', exact: true })
  const close = dialog.getByRole('button', { name: 'Close AI Assistant', exact: true })
  await expect(dialog.getByRole('heading', { name: 'AI Assistant', exact: true })).toBeVisible()
  await expect(close).toHaveText('')
  await expect(close.locator('svg')).toBeVisible()
  const expand = dialog.getByRole('button', { name: 'Expand for reading', exact: true })
  if (page.viewportSize()!.width >= 1024) {
    await expand.click()
    expect((await dialog.boundingBox())!.width).toBe(720)
    await expect(dialog.getByRole('button', { name: 'Narrow', exact: true })).toBeVisible()
  } else await expect(expand).toBeHidden()
  const input = dialog.getByRole('textbox', { name: 'Message', exact: true })
  await input.fill('希望研究交易价格的变化')
  await dialog.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(dialog.getByRole('log', { name: 'Conversation' })).toContainText('请确认是交易价格还是总市值？')
  await input.fill('交易价格，确认生成公式')
  await dialog.getByRole('button', { name: 'Send', exact: true }).click()
  const card = dialog.getByRole('region', { name: 'Indicator draft for this reply', exact: true })
  await expect(card.getByRole('heading', { name: api.definition.name, exact: true })).toBeVisible()
  await expect(card.getByText('Definition validated', { exact: true })).toBeVisible()
  await card.getByRole('button', { name: 'Formula outputs (1)', exact: true }).click()
  await expect(card.locator('pre')).toHaveText(api.definition.expression)
  await expect(card.getByRole('checkbox', { name: 'Show original lines', exact: true })).toBeVisible()
  await expect(card.getByRole('button', { name: 'Fill editor', exact: true })).toBeEnabled()
  await expect(card.getByRole('button', { name: 'Save indicator', exact: true })).toBeEnabled()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true)
  const labels = ['AI Assistant', 'Close', 'Message', 'Definition validated', 'Fill editor', 'Save indicator', 'Formula outputs (1)', 'Copy formula']
  expect((await page.evaluate(auditTextContrast)).filter(item => labels.includes(item.text))).toEqual([])
  await expect(close).toBeInViewport()
  await expect(input).toBeInViewport()
  await dialog.screenshot({ path: test.info().outputPath('english-conversation.png') })
  await close.click()
  await expect(launcher).toBeFocused()
})

test('只展示校验通过的草稿，设计中、失败及失效草稿在恢复和实时更新时均隐藏', async ({ page }) => {
  await openStudio(page); const api = await agentApi(page, { running: true })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await page.getByRole('textbox', { name: '发送消息' }).fill('默认20个交易日，窗口可变')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect(page.getByRole('button', { name: '停止', exact: true })).toBeVisible()
  const card = page.getByRole('region', { name: '本轮指标草稿', exact: true })
  for (const valid of [undefined, false]) {
    api.updateDraft(valid)
    await page.reload()
    await page.getByRole('button', { name: '打开 AI 助手' }).click()
    await expect(page.getByText('思考中…', { exact: true })).toBeVisible()
    await expect(card).toHaveCount(0)
    await expect(page.getByText('参数 window 没有被公式使用，请删除或重新关联。')).toHaveCount(0)
    await expect(page.getByRole('button', { name: '确认保存指标' })).toHaveCount(0)
    await expect(page.getByRole('log')).toContainText('默认20个交易日，窗口可变')
  }
  for (const stale of [false, true]) {
    api.updateDraft(true)
    await expect(card).toBeVisible()
    await expect(page.getByText('定义校验通过', { exact: true })).toBeVisible()
    api.updateDraft(stale, stale)
    await expect(card).toHaveCount(0)
  }
  api.updateDraft(true); api.finish('指标已完成校验，请确认。')
  await expect(page.getByRole('button', { name: '填入编辑器' })).toBeEnabled()
  await expect(page.getByRole('button', { name: '确认保存指标' })).toBeEnabled()
  await expectCloseReachable(page)
  await expect.poll(async () => (await page.evaluate(auditTextContrast))
    .filter(item => ['区间平均价差', '定义校验通过', '填入编辑器', '确认保存指标'].includes(item.text))).toEqual([])
  await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: test.info().outputPath('validated-draft.png') })
  expect(api.requests).toHaveLength(1); expect(api.saved()).toBe(false)
})

test('未选产品可多轮讨论、生成公式并由人类保存', async ({ page }) => {
  await openStudio(page); const api = await agentApi(page, { nameConflict: true })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const input = page.getByRole('textbox', { name: '发送消息' })
  await input.fill('我想算每天最高与最低市场价值之差的平均值')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect(page.getByRole('log')).toContainText('请确认是交易价格还是总市值？')
  await input.fill('交易价格，按复权最高价与最低价生成公式，不试算')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await page.getByRole('button', { name: '查看完整公式（1 个输出）', exact: true }).click()
  await expect(page.getByText(api.definition.expression, { exact: true })).toBeVisible()
  expect(api.saved()).toBe(false)
  const pending = page.waitForEvent('dialog')
  const click = page.getByRole('button', { name: '确认保存指标' }).click()
  const dialog = await pending
  expect(dialog.type()).toBe('confirm')
  expect(dialog.message()).toContain(api.definition.name)
  expect(dialog.message()).toContain(api.definition.expression)
  expect(dialog.message()).toContain('同名指标')
  expect(dialog.message()).toContain('不会覆盖')
  expect(api.commitWrites()).toBe(0)
  await dialog.dismiss(); await click
  expect(api.saved()).toBe(false)
  expect(api.commitWrites()).toBe(0)
  await input.fill('取消保存后继续解释这个口径')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect.poll(() => api.requests.length).toBe(3)
  expect(api.requests[2].expected_session_revision).toBe(3)
  await expect(page.getByRole('button', { name: '确认保存指标' })).toBeEnabled()
  await expect(page.getByText('会话版本冲突', { exact: true })).toHaveCount(0)
  page.once('dialog', dialog => dialog.accept())
  await page.getByRole('button', { name: '确认保存指标' }).click()
  const savedCard = page.getByRole('region', { name: '本轮指标草稿' }).filter({ has: page.getByRole('button', { name: '已保存', exact: true }) })
  await expect(savedCard.getByRole('status')).toBeVisible()
  expect(api.saved()).toBe(true)
  await expect(savedCard.getByRole('status')).toContainText('指标已保存')
  expect(api.commitWrites()).toBe(1)
  await expectCloseReachable(page)
})

test('发送即显示消息和发送状态，失败重试保留同一消息', async ({ page }) => {
  await openStudio(page); const api = await agentApi(page, { delay: true })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const prompt = '我要滚动夏普比率，结果为时序向量，窗口可变。'
  await page.getByRole('textbox', { name: '发送消息' }).fill(prompt)
  await page.getByRole('button', { name: '发送', exact: true }).click()
  try {
    await expect(page.getByRole('log').getByText(prompt, { exact: true })).toBeVisible()
    await expect(page.getByText('正在发送…', { exact: true })).toBeVisible()
    await expect(page.getByRole('textbox', { name: '发送消息' })).toHaveValue('')
    const failures = await page.evaluate(auditTextContrast)
    expect(failures.filter(item => [prompt, '正在发送…', '你'].includes(item.text))).toEqual([])
  } finally { api.release() }
  await expect(page.getByRole('alert')).toContainText('响应超时')
  await page.getByRole('button', { name: '重试这条消息' }).click()
  await expect(page.getByRole('log')).toContainText('请确认是交易价格还是总市值？')
  expect(api.requests).toHaveLength(2); expect(api.requests[0]).toEqual(api.requests[1])
  await expect(page.getByRole('log').getByText(prompt, { exact: true })).toHaveCount(1)
})

for (const cancel of [false, true]) test(`停止并发送的排队读取延迟时${cancel ? '仍可取消待发送' : '阻止后发消息越过'}`, async ({ page }) => {
  await openStudio(page)
  const api = await agentApi(page, { running: true })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const input = page.getByRole('textbox', { name: '发送消息' })
  await input.fill('原运行')
  const connected = page.waitForResponse(response => response.url().includes('/runs/r1'))
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await connected
  await expect(page.getByRole('button', { name: '停止', exact: true })).toBeVisible()

  let reads = 0, release!: () => void, held!: () => void, settled!: () => void
  const gate = new Promise<void>(resolve => { release = resolve })
  const heldRead = new Promise<void>(resolve => { held = resolve })
  const settledRead = new Promise<void>(resolve => { settled = resolve })
  await page.route('**/api/agent/sessions/s', async route => {
    if (++reads !== 1) return route.fallback()
    held()
    await gate
    try { await route.fallback() } finally { settled() }
  })
  await input.fill('排队消息A')
  const terminalRestore = page.waitForResponse(response => new URL(response.url()).pathname === '/api/agent/sessions/s')
  await page.getByRole('button', { name: '停止并发送', exact: true }).click()
  try {
    await heldRead
    await terminalRestore
    await expect(page.getByText('正在加载…', { exact: true })).toHaveCount(0)
    await expect(page.getByRole('button', { name: '停止', exact: true })).toHaveCount(0)
    await input.fill('后发消息B')
    await expect(page.getByRole('button', { name: '发送', exact: true })).toBeDisabled()
    await input.press('Control+Enter')
    await expect(input).toHaveValue('后发消息B')
    expect(api.requests.map(request => request.text)).toEqual(['原运行'])
    const cancelQueued = page.getByRole('button', { name: /取消.*发送/ })
    await expect(cancelQueued).toBeEnabled()
    await expectCloseReachable(page)
    expect((await page.evaluate(auditTextContrast)).filter(item => /排队|待发送/.test(item.text))).toEqual([])
    await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: test.info().outputPath(cancel ? 'queued-before-cancel.png' : 'queued-before-send.png') })
    if (cancel) {
      await cancelQueued.click()
      await expect(page.getByRole('button', { name: '重试这条消息', exact: true })).toBeVisible()
      await expect(cancelQueued).toHaveCount(0)
    }
  } finally { release() }
  await settledRead
  if (cancel) {
    await expect(page.getByRole('button', { name: '发送', exact: true })).toBeEnabled()
    // Drain the released fetch continuation before asserting that cancellation prevented POST.
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))))
    expect(api.requests.map(request => request.text)).toEqual(['原运行'])
    await expect(page.getByRole('button', { name: '重试这条消息', exact: true })).toBeEnabled()
  } else {
    await expect.poll(() => api.requests.map(request => request.text)).toEqual(['原运行', '排队消息A'])
    await expect(page.getByRole('button', { name: '停止', exact: true })).toBeVisible()
    expect(api.requests.filter(request => request.text === '排队消息A')).toHaveLength(1)
  }
  await expect(input).toHaveValue('后发消息B')
  expect(api.commitWrites()).toBe(0)
  expect(api.memoryDecisions).toEqual([])
  await expectCloseReachable(page)
  await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: test.info().outputPath(cancel ? 'queued-cancelled.png' : 'queued-sent.png') })
})

test('运行资源短暂404保留已恢复会话，重新连接继续原运行', async ({ page }) => {
  await openStudio(page)
  const api = await agentApi(page, { running: true })
  let unavailable = true
  await page.route('**/api/agent/sessions/s/runs/r1', route => unavailable
    ? route.fulfill({ status: 404, json: { detail: { message: '运行资源暂不可用' } } }) : route.fallback())
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const input = page.getByRole('textbox', { name: '发送消息' })
  await input.fill('保留本次研究')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  const reconnect = page.getByRole('button', { name: '重新连接', exact: true })
  await expect(reconnect).toBeVisible()
  await input.fill('连接恢复后再补充')
  const sessionKey = agentSessionStorageKey(api.requests[0].page_context)
  expect(await page.evaluate(key => sessionStorage.getItem(key), sessionKey)).toBe('s')
  unavailable = false
  await reconnect.click()
  await expect(reconnect).toHaveCount(0)
  await expect(page.getByRole('button', { name: '停止', exact: true })).toBeVisible()
  await expect(page.getByRole('log').getByText('保留本次研究', { exact: true })).toBeVisible()
  api.finish('已恢复原运行并完成。')
  await expect(page.getByText('已恢复原运行并完成。', { exact: true })).toBeVisible()
  await expect(input).toHaveValue('连接恢复后再补充')
  expect(api.requests).toHaveLength(1)
  expect(api.commitWrites()).toBe(0)
  await expectCloseReachable(page)
  await expect(page.getByRole('dialog', { name: 'AI 助手' })).toHaveCSS('opacity', '1')
  expect((await page.evaluate(auditTextContrast)).filter(item => /保留本次|恢复原运行|恢复后再补充/.test(item.text))).toEqual([])
  await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: test.info().outputPath('run-resource-recovery.png') })
})

test('上一轮试算不会替新运行采纳用户改回的旧周期', async ({ page }) => {
  await openStudio(page)
  const target = { kind: 'etf', product_id: '510300.SH', name: '沪深300样例ETF' }
  const definition = { name: '上一轮均值', expression: 'mean(returns)', result_kind: 'scalar', context_kind: 'single_product', parameter_schema: [] }
  const draft = { valid: true, draft_revision: 1, definition_hash: 'old-hash', definition }
  const preview = { preview_id: 'old-preview', run_id: 'old-run', definition_hash: 'old-hash', target, period: '1Y', as_of: '2020-01-01', result_kind: 'scalar' }
  let context: Record<string, any> = {}, current: Record<string, any> | null = null
  const messages: Array<Record<string, any>> = [], requests: Array<Record<string, any>> = [], invalidations: Array<Record<string, any>> = []
  let release!: () => void
  const pending = new Promise<void>(resolve => { release = resolve })
  await page.route('**/api/agent/**', async route => {
    const path = new URL(route.request().url()).pathname
    if (path.endsWith('/meta')) return route.fulfill({ json: { configured: true, model: 'fixture' } })
    if (path.endsWith('/sessions')) { context = route.request().postDataJSON().page_context; return route.fulfill({ json: { session_id: 'p', session_revision: 0 } }) }
    if (path.endsWith('/messages')) {
      const body = route.request().postDataJSON(); requests.push(body)
      if (requests.length === 2) await pending
      context = body.page_context
      current = { run_id: requests.length === 1 ? 'old-run' : 'new-run', session_id: 'p', message_id: body.message_id,
        session_revision: requests.length, run_revision: 1, phase: 'thinking', status: requests.length === 1 ? 'completed' : 'running' }
      messages.push({ seq: messages.length + 1, id: body.message_id, run_id: current.run_id, speaker: 'user', text: body.text })
      if (requests.length === 1) {
        current.response = { session_id: 'p', session_revision: 1, reply: { text: '上一轮试算已完成。' }, draft, preview, artifacts: { draft, preview } }
        messages.push({ seq: 2, id: 'old-run-reply', run_id: 'old-run', speaker: 'assistant', text: '上一轮试算已完成。', artifacts: { draft, preview } })
      }
      return route.fulfill({ status: 202, json: current })
    }
    if (path.includes('/previews/')) return route.fulfill({ json: { ...preview, session_id: 'p', definition, expires_at: '2099-01-01',
      result: { results: [{ target, period: '1Y', status: 'ok', value: 0.01, parameters: {}, warnings: [],
        window: { start_date: '2019-01-02', end_date: '2020-01-01', observation_count: 245 },
        presentation: { name: definition.name, display_format: 'number', precision: 4, unit: '' } }] } } })
    if (path.endsWith('/invalidate-context')) {
      invalidations.push(route.request().postDataJSON())
      current = { ...current, status: 'cancelled', run_revision: 2,
        response: { session_id: 'p', session_revision: 2, reply: { text: '周期已变化，本轮已停止。' }, artifacts: {} } }
      return route.fulfill({ json: current })
    }
    if (path.endsWith('/events')) return route.fulfill({ json: { items: [], has_more: false } })
    if (path.includes('/runs/')) return route.fulfill({ json: current })
    if (path.endsWith('/sessions/p')) return route.fulfill({ json: { session_id: 'p', session_revision: current?.session_revision || 0,
      page_context: context, active_run: current, messages, next_event_seq: messages.length + 1,
      draft: current?.run_id === 'old-run' ? draft : null, preview: current?.run_id === 'old-run' ? preview : null } })
    return route.fulfill({ json: {} })
  })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const input = page.getByRole('textbox', { name: '发送消息' })
  await input.fill('先试算一年均值')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect(page.getByRole('region', { name: 'AI 试算结果', exact: true })).toBeVisible()
  expect(invalidations).toEqual([])
  await page.getByRole('button', { name: '关闭 AI 助手' }).click()
  const period = page.getByRole('combobox', { name: '计算周期', exact: true })
  await expect(period).toHaveValue('1Y')
  await period.selectOption('3Y')
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await input.fill('改为三年继续研究')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect.poll(() => requests.length).toBe(2)
  await input.fill('保留未发送补充')
  await page.getByRole('button', { name: '关闭 AI 助手' }).click()
  await period.selectOption('1Y')
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await expect(page.getByText('正在加载…', { exact: true })).toHaveCount(0)
  release()
  await expect.poll(() => invalidations.length).toBe(1)
  expect(requests[1].page_context.calculation.period).toBe('3Y')
  expect(invalidations[0].page_context.calculation.period).toBe('1Y')
  await expect(page.getByText('周期已变化，本轮已停止。', { exact: true })).toBeVisible()
  await expect(input).toHaveValue('保留未发送补充')
  await expectCloseReachable(page)
  await expect(page.getByRole('dialog', { name: 'AI 助手' })).toHaveCSS('opacity', '1')
  expect((await page.evaluate(auditTextContrast)).filter(item => /三年继续|周期已变化|未发送补充/.test(item.text))).toEqual([])
  await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: test.info().outputPath('preview-run-ownership.png') })
})

test('发送期间重开恢复旧口径不误停新运行，之后真实改口径仍停止', async ({ page }) => {
  await openStudio(page)
  const api = await agentApi(page, { running: true })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const input = page.getByRole('textbox', { name: '发送消息' })
  await input.fill('上一轮研究')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect.poll(() => api.requests.length).toBe(1)
  api.finish('上一轮已完成。')
  await expect(page.getByText('上一轮已完成。', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: '关闭 AI 助手' }).click()
  const period = page.getByRole('combobox', { name: '计算周期', exact: true })
  if (!await period.isVisible()) await page.getByRole('tab', { name: page.viewportSize()!.width < 1280 ? /^预览$/ : /^校验与预览/ }).click()
  await period.selectOption('3Y')
  let release!: () => void, waiting = false
  const pending = new Promise<void>(resolve => { release = resolve })
  const invalidations: Array<Record<string, any>> = []
  await page.route('**/api/agent/**', async route => {
    const path = new URL(route.request().url()).pathname
    if (path.endsWith('/messages') && route.request().postDataJSON().text === '按三年口径继续研究') {
      waiting = true; await pending
    }
    if (path.endsWith('/invalidate-context')) {
      invalidations.push(route.request().postDataJSON())
      api.finish('研究口径变化，本轮已停止。', 'cancelled')
      return route.fulfill({ json: { session_id: 's', session_revision: 2, run_id: 'r2', message_id: api.requests[1].message_id,
        run_revision: 2, status: 'cancelled', phase: 'thinking' } })
    }
    return route.fallback()
  })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await input.fill('按三年口径继续研究')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect.poll(() => waiting).toBe(true)
  await input.fill('保留下一轮未发送补充')
  await page.getByRole('button', { name: '关闭 AI 助手' }).click()
  const restored = page.waitForResponse(response => response.url().endsWith('/sessions/s'))
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  expect((await (await restored).json()).page_context.calculation.period).not.toBe('3Y')
  await expect(page.getByText('正在加载…', { exact: true })).toHaveCount(0)
  release()
  await expect.poll(() => api.requests.length).toBe(2)
  await expect(page.getByRole('button', { name: '停止', exact: true })).toBeVisible()
  await expect(input).toHaveValue('保留下一轮未发送补充')
  expect(api.requests[1].page_context.calculation.period).toBe('3Y')
  expect(invalidations).toEqual([])
  await page.getByRole('button', { name: '查看上下文与模型' }).click()
  await expect(page.getByText('周期：3Y', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: '关闭 AI 助手' }).click()
  await period.selectOption('5Y')
  await expect.poll(() => invalidations.length).toBe(1)
  expect(invalidations[0].page_context.calculation.period).toBe('5Y')
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await expect(page.getByText('研究口径变化，本轮已停止。', { exact: true })).toBeVisible()
  await expect(input).toHaveValue('保留下一轮未发送补充')
  await expectCloseReachable(page)
  await expect(page.getByRole('dialog', { name: 'AI 助手' })).toHaveCSS('opacity', '1')
  expect((await page.evaluate(auditTextContrast)).filter(item => /三年口径|研究口径变化|未发送补充/.test(item.text))).toEqual([])
  await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: test.info().outputPath('restored-message-context.png') })
})

test('编辑已从恢复快照确认后，原提交回包丢失不回滚消息', async ({ page }) => {
  await openStudio(page)
  const api = await agentApi(page, { running: true })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const input = page.getByRole('textbox', { name: '发送消息' })
  await input.fill('旧问题')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await page.getByRole('button', { name: '停止', exact: true }).click()
  await expect(page.getByRole('button', { name: '修改这条消息', exact: true })).toBeEnabled()
  let accepted: Record<string, any> | null = null
  let release!: () => void
  const pending = new Promise<void>(resolve => { release = resolve })
  await page.route('**/api/agent/**', async route => {
    const path = new URL(route.request().url()).pathname
    if (path.endsWith('/messages') && route.request().postDataJSON().edit_of_message_id) {
      const body = route.request().postDataJSON(); api.requests.push(body)
      accepted = { run_id: 'edited-run', session_id: 's', message_id: body.message_id, session_revision: 2, run_revision: 3,
        status: 'completed', phase: 'thinking', response: { session_id: 's', session_revision: 2, reply: { text: '新问题已回答。' }, artifacts: {} }, request: body }
      await pending
      return route.abort('failed')
    }
    if (accepted && path.endsWith('/sessions/s')) return route.fulfill({ json: {
      session_id: 's', session_revision: 2, page_context: accepted.request.page_context, active_run: accepted, next_event_seq: 101,
      messages: [{ seq: 99, id: accepted.message_id, run_id: accepted.run_id, speaker: 'user', text: '新问题', edit_of: accepted.request.edit_of_message_id },
        { seq: 100, id: 'edited-run-reply', run_id: accepted.run_id, speaker: 'assistant', text: '新问题已回答。' }],
      memory_proposals: [], memory_sources: [],
    } })
    if (accepted && path.includes('/runs/')) return route.fulfill({ json: accepted })
    if (accepted && path.endsWith('/events')) return route.fulfill({ json: { items: [], has_more: false } })
    return route.fallback()
  })
  await page.getByRole('button', { name: '修改这条消息', exact: true }).click()
  await input.fill('新问题')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect.poll(() => !!accepted).toBe(true)
  await page.getByRole('button', { name: '关闭 AI 助手' }).click()
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await expect(page.getByText('新问题已回答。', { exact: true })).toBeVisible()
  const failed = page.waitForEvent('requestfailed', request => request.url().includes('/messages'))
  release()
  await failed
  await expect(page.getByRole('button', { name: '取消修改', exact: true })).toHaveCount(0)
  await expect(input).toHaveValue('')
  await expect(page.getByRole('log').getByText('新问题', { exact: true })).toBeVisible()
  await expect(page.getByRole('log').getByText('旧问题', { exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: '重试这条消息' })).toHaveCount(0)
  expect(api.requests).toHaveLength(2)
  await input.fill('保留下一轮草稿')
  await expectCloseReachable(page)
  expect((await page.evaluate(auditTextContrast)).filter(item => /新问题|下一轮草稿/.test(item.text))).toEqual([])
  await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: test.info().outputPath('restored-edit-receipt.png') })
})

test('旧停止请求迟到失败不覆盖新一轮的成功状态和未发送输入', async ({ page }) => {
  await openStudio(page)
  const api = await agentApi(page, { running: true })
  let release!: () => void
  const pending = new Promise<void>(resolve => { release = resolve })
  await page.route('**/api/agent/sessions/s/runs/r1/cancel', async route => {
    await pending
    await route.fulfill({ status: 503, json: { detail: { message: '旧任务停止请求失败' } } })
  })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const input = page.getByRole('textbox', { name: '发送消息' })
  await input.fill('第一轮研究')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  const controlRequest = page.waitForRequest(request => request.url().endsWith('/runs/r1/cancel'))
  await page.getByRole('button', { name: '停止', exact: true }).click()
  await controlRequest
  api.finish('第一轮已自行完成。')
  await expect(page.getByText('第一轮已自行完成。', { exact: true })).toBeVisible()
  await input.fill('第二轮研究')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect.poll(() => api.requests.length).toBe(2)
  api.finish('第二轮已完成。')
  await expect(page.getByText('第二轮已完成。', { exact: true })).toBeVisible()
  await input.fill('保留未发送的第三轮补充')
  const controlResponse = page.waitForResponse(response => response.url().endsWith('/runs/r1/cancel'))
  release()
  await (await controlResponse).finished()
  await page.getByRole('button', { name: '关闭 AI 助手' }).click()
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await expect(page.getByRole('button', { name: '发送', exact: true })).toBeEnabled()
  await expect(page.getByRole('alert').filter({ hasText: '旧任务停止请求失败' })).toHaveCount(0)
  await expect(input).toHaveValue('保留未发送的第三轮补充')
  await expect(page.getByText('第二轮已完成。', { exact: true })).toBeVisible()
  expect(api.requests).toHaveLength(2)
  expect(api.commitWrites()).toBe(0)
  await expectCloseReachable(page)
  await expect(page.getByRole('dialog', { name: 'AI 助手' })).toHaveCSS('opacity', '1')
  expect((await page.evaluate(auditTextContrast)).filter(item => /研究|第二轮已完成|第三轮补充/.test(item.text))).toEqual([])
  await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: test.info().outputPath('late-stop-response.png') })
})

test('运行中可输入，关闭不停止，刷新恢复后仍能停止', async ({ page }) => {
  await openStudio(page); const api = await agentApi(page, { running: true })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await page.getByRole('textbox', { name: '发送消息' }).fill('滚动夏普，窗口可变')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect(page.getByRole('button', { name: '停止', exact: true })).toBeVisible()
  await page.getByRole('textbox', { name: '发送消息' }).fill('下一条补充')
  await page.getByRole('button', { name: '关闭 AI 助手' }).click()
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await expect(page.getByRole('textbox', { name: '发送消息' })).toHaveValue('下一条补充')
  await page.reload()
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await expect(page.getByRole('log')).toContainText('滚动夏普，窗口可变')
  await expect(page.getByRole('button', { name: '停止', exact: true })).toBeVisible()
  await page.getByRole('button', { name: '停止', exact: true }).click()
  await expect(page.getByRole('log')).toContainText('已停止，已提交的进度已保留。')
  expect(api.requests).toHaveLength(1)
  // Short user message plus the AI stop bubble, with the actions below each bubble.
  const shortBubble = page.getByRole('log').locator('article').filter({ hasText: '滚动夏普，窗口可变' }).locator('[data-message-bubble]')
  await expect(shortBubble).toBeVisible()
  await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: `../.run/agent-message-actions-bottom/short-user-${page.viewportSize()!.width}.png` })
  await expectCloseReachable(page)
})

test('消息可复制，停止后未获回复的最后一条消息可修改、取消并重新发送', async ({ page }) => {
  await openStudio(page)
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write'], { origin: new URL(page.url()).origin })
  const api = await agentApi(page, { running: true })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const input = page.getByRole('textbox', { name: '发送消息' })
  const log = page.getByRole('log')
  await input.fill('先停止的这一轮')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect(page.getByRole('button', { name: '停止', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: '修改这条消息' })).toHaveCount(0)
  await page.getByRole('button', { name: '停止', exact: true }).click()
  await expect(log).toContainText('已停止，已提交的进度已保留。')
  const stopped = log.locator('article').filter({ hasText: '先停止的这一轮' })
  // No visible speaker heading: the role stays in the message's accessible name only.
  await expect(stopped).toHaveAttribute('aria-label', '你')
  await expect(log.getByText('你', { exact: true })).toHaveCount(0)
  await expect(log.getByText('AI', { exact: true })).toHaveCount(0)
  await expect(log.getByRole('article', { name: 'AI' })).toHaveCount(1)
  await stopped.hover()
  const copy = stopped.getByRole('button', { name: '复制这条消息' })
  const edit = stopped.getByRole('button', { name: '修改这条消息' })
  const copyBox = (await copy.boundingBox())!
  expect(copyBox.width).toBeGreaterThanOrEqual(40)
  expect(copyBox.height).toBeGreaterThanOrEqual(40)
  // The bubble opens with its text (8px padding) and the actions sit below it, right aligned.
  const userBubble = stopped.locator('[data-message-bubble]')
  const userBubbleBox = (await userBubble.boundingBox())!
  const userTextBox = (await userBubble.getByText('先停止的这一轮', { exact: true }).boundingBox())!
  expect(userTextBox.y).toBeGreaterThanOrEqual(userBubbleBox.y)
  expect(userTextBox.y - userBubbleBox.y).toBeLessThan(20)
  const editBox = (await edit.boundingBox())!
  for (const control of [copyBox, editBox]) {
    expect(control.width).toBeGreaterThanOrEqual(40)
    expect(control.height).toBeGreaterThanOrEqual(40)
    expect(control.y).toBeGreaterThanOrEqual(userBubbleBox.y + userBubbleBox.height - 1)
    expect(control.x + control.width).toBeLessThanOrEqual(userBubbleBox.x + userBubbleBox.width + 1)
  }
  // The trailing control ends at the bubble's right edge; the optional edit control sits before it.
  expect(userBubbleBox.x + userBubbleBox.width - (copyBox.x + copyBox.width)).toBeLessThanOrEqual(14)
  expect(editBox.x + editBox.width).toBeLessThanOrEqual(copyBox.x + 1)
  await page.mouse.move(0, 0)
  await edit.focus()
  await expect(edit.locator('..')).toHaveCSS('opacity', '1')
  await copy.focus()
  await expect(copy.locator('..')).toHaveCSS('opacity', '1')
  await copy.click()
  await expect(stopped.getByRole('status')).toContainText('已复制这条消息。')
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe('先停止的这一轮')
  await input.fill('尚未发送的草稿')
  await stopped.getByRole('button', { name: '修改这条消息' }).click()
  await expect(input).toHaveValue('先停止的这一轮')
  await expect(page.getByRole('button', { name: '取消修改' })).toBeVisible()
  await page.getByRole('button', { name: '取消修改' }).click()
  await expect(input).toHaveValue('尚未发送的草稿')
  await expect(log.getByText('先停止的这一轮', { exact: true })).toBeVisible()
  await stopped.getByRole('button', { name: '修改这条消息' }).click()
  await input.fill('改写后的问题')
  const editRequest = page.waitForRequest(request => request.url().includes('/messages') && !!request.postDataJSON()?.edit_of_message_id)
  await page.getByRole('button', { name: '发送', exact: true }).click()
  const body = (await editRequest).postDataJSON()
  expect(body.text).toBe('改写后的问题')
  expect(body.edit_of_message_id).toEqual(expect.any(String))
  expect(body).not.toHaveProperty('resume_from_run_id')
  await expect(log.getByText('改写后的问题', { exact: true })).toBeVisible()
  await expect(log.getByText('先停止的这一轮', { exact: true })).toHaveCount(0)
  await expect(log.getByText('已停止，已提交的进度已保留。', { exact: true })).toHaveCount(0)
  await expect(input).toHaveValue('')
  api.finish('改写后的回复')
  await expect(log).toContainText('改写后的回复')
  await page.reload()
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await expect(log.getByText('改写后的问题', { exact: true })).toBeVisible()
  await expect(log.getByText('先停止的这一轮', { exact: true })).toHaveCount(0)
  expect(api.requests).toHaveLength(2)
  // The AI reply carries its own pale bubble, distinct from the white panel and the user's blue one.
  const aiArticle = log.locator('article').filter({ hasText: '改写后的回复' })
  const userArticle = log.locator('article').filter({ hasText: '改写后的问题' })
  const aiBubble = aiArticle.locator('[data-message-bubble]')
  const aiBackground = await aiBubble.evaluate(el => getComputedStyle(el).backgroundColor)
  const userBackground = await userArticle.locator('[data-message-bubble]').evaluate(el => getComputedStyle(el).backgroundColor)
  expect(aiBackground).toBe('rgb(248, 250, 252)')
  expect(aiBackground).not.toBe('rgba(0, 0, 0, 0)')
  expect(aiBackground).not.toBe('rgb(255, 255, 255)')
  expect(aiBackground).not.toBe(userBackground)
  expect(await aiBubble.evaluate(el => getComputedStyle(el).borderTopColor)).not.toBe('rgba(0, 0, 0, 0)')
  // Sample contrast only after the dialog's enter animation settled (see the entry test).
  await expect(page.getByRole('dialog', { name: 'AI 助手' })).toHaveCSS('opacity', '1')
  const messageFailures = await page.evaluate(auditTextContrast)
  expect(messageFailures.filter(item => ['改写后的问题', '改写后的回复'].includes(item.text))).toEqual([])
  const logBox = (await log.boundingBox())!
  const aiBox = (await aiArticle.boundingBox())!
  const aiBubbleBox = (await aiBubble.boundingBox())!
  const aiTextBox = (await aiBubble.getByText('改写后的回复', { exact: true }).boundingBox())!
  const aiCopyBox = (await aiArticle.getByRole('button', { name: '复制这条消息' }).boundingBox())!
  expect(aiBox.x).toBeGreaterThanOrEqual(logBox.x - 1)
  expect(aiBox.x + aiBox.width).toBeLessThanOrEqual(logBox.x + logBox.width + 1)
  expect(aiTextBox.y).toBeGreaterThanOrEqual(aiBubbleBox.y)
  expect(aiTextBox.y - aiBubbleBox.y).toBeLessThan(20)
  expect(aiCopyBox.width).toBeGreaterThanOrEqual(40)
  expect(aiCopyBox.height).toBeGreaterThanOrEqual(40)
  expect(aiCopyBox.y).toBeGreaterThanOrEqual(aiBubbleBox.y + aiBubbleBox.height - 1)
  expect(aiCopyBox.x + aiCopyBox.width).toBeLessThanOrEqual(aiBubbleBox.x + aiBubbleBox.width + 1)
  expect(aiBubbleBox.x + aiBubbleBox.width - (aiCopyBox.x + aiCopyBox.width)).toBeLessThanOrEqual(14)
  await aiArticle.getByRole('button', { name: '复制这条消息' }).hover()
  await expect(aiArticle.locator('.agent-reply-copy')).toHaveCSS('opacity', '1')
  await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: `../.run/agent-message-actions-bottom/bubbles-${page.viewportSize()!.width}.png` })
  await expectCloseReachable(page)
})

test('AI 历史回复渲染 Markdown 和 LaTeX，长公式不挤出浮窗', async ({ page }) => {
  await openStudio(page)
  const formula = String.raw`\text{Rolling Sharpe}_t=\frac{\operatorname{mean}(r_{t-N+1:t}-r_f)}{\operatorname{std}(r_{t-N+1:t}-r_f)}\times\sqrt{A}`
  const wideFormula = Array.from({ length: 32 }, (_, index) => `x_{${index + 1}}`).join('+')
  const markdown = String.raw`### 滚动夏普

用**最近 N 个收益观察期**计算，\(N=20\)。

\[
${formula}
\]

长公式排版示例：

\[
${wideFormula}
\]

- 窗口可变
- 校验口径

| 参数 | 含义 |
| --- | --- |
| N | 窗口长度 |

` + ['```latex', String.raw`\[\frac{1}{2}\]`, '```'].join('\n')
  let imageRequests = 0
  await page.route('https://example.com/track.png', route => { imageRequests++; return route.abort() })
  await page.evaluate(() => sessionStorage.setItem('indicator-agent-session:indicator-studio:single_product', 'markdown-test'))
  await page.route('**/api/agent/**', route => {
    const path = new URL(route.request().url()).pathname
    if (path.endsWith('/meta')) return route.fulfill({ json: { configured: true, model: 'deepseek-v4.1-flash' } })
    if (path.endsWith('/sessions/markdown-test')) return route.fulfill({ json: {
      session_id: 'markdown-test', session_revision: 1, next_event_seq: 3,
      messages: [{ seq: 1, id: 'question', speaker: 'user', text: '解释滚动夏普' }, { seq: 2, id: 'reply', speaker: 'assistant', text: markdown + '\n\n![说明图](https://example.com/track.png)' }],
    } })
    return route.fulfill({ json: { items: [], has_more: false, next_event_seq: 3 } })
  })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const log = page.getByRole('log')
  await expect(log.getByRole('heading', { name: '滚动夏普' })).toBeVisible()
  await expect(log.locator('strong')).toHaveText('最近 N 个收益观察期')
  await expect(log.locator('.katex')).toHaveCount(3)
  await expect(log.locator('.katex-display annotation').first()).toHaveText(formula)
  const math = log.getByRole('group', { name: '数学公式，可横向滚动' }).last()
  await math.scrollIntoViewIfNeeded()
  await expect(math).toBeVisible()
  expect(await math.evaluate(el => el.scrollWidth > el.clientWidth)).toBe(true)
  await math.focus(); await page.keyboard.press('ArrowRight')
  await expect.poll(() => math.evaluate(el => el.scrollLeft)).toBeGreaterThan(0)
  const panel = page.getByRole('dialog', { name: 'AI 助手' })
  expect(await panel.evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true)
  await expect(log.locator('pre code')).toHaveText(String.raw`\[\frac{1}{2}\]`)
  await expect(log.getByRole('columnheader').first()).toHaveAttribute('scope', 'col')
  expect(imageRequests).toBe(0)
  const failures = await page.evaluate(auditTextContrast)
  expect(failures.filter(item => ['滚动夏普', '最近 N 个收益观察期', '窗口可变', '参数'].includes(item.text))).toEqual([])
  await expectCloseReachable(page)
  await page.getByRole('heading', { name: '滚动夏普' }).scrollIntoViewIfNeeded()
  await page.screenshot({ path: `.run/agent-markdown-${page.viewportSize()!.width}.png` })
})

for (const mode of ['time_series', 'scalar', 'empty', 'expired'] as const) {
  test(`AI 样例试算展示并恢复真实结果：${mode}`, async ({ page }) => {
    await openStudio(page)
    if (page.viewportSize()!.width < 1280) await page.getByRole('tab', { name: '编辑', exact: true }).click()
    await page.getByLabel('名称', { exact: true }).fill('编辑器原有未保存指标')
    let context: Record<string, unknown> = {}, posts = 0, invalidations = 0
    const bodies: Record<string, any>[] = []
    const calculations: Record<string, any>[] = []
    const target = { kind: 'etf', product_id: '510300.SH', name: '沪深300样例ETF' }
    const definition = { name: '滚动夏普试算', description: '预览同步测试', unit: '', expression: 'mean(rolling_window(returns,window))', context_kind: 'single_product', result_kind: mode === 'scalar' ? 'scalar' : 'time_series', display_format: 'number', precision: 3,
      parameter_contract_version: '1.0', parameter_schema: [{ id: 'window', label: '动量窗口', type: 'integer', default: 20, minimum: 1, maximum: 5000, step: 1 }],
      series_outputs: [{ id: 'sharpe', label: '滚动夏普', expression: 'mean(rolling_window(returns,window))', unit: '', display_format: 'number', precision: 3 }] }
    const draft = { valid: true, draft_revision: 1, definition_hash: 'preview-hash', definition }
    const reference = { preview_id: 'preview-1', run_id: 'preview-run', definition_hash: 'preview-hash', target, period: '6M', as_of: '2019-12-30', result_kind: definition.result_kind }
    const window = { start_date: '2019-01-02', end_date: '2019-12-30', observation_count: 246, data_latest_date: '2019-12-30' }
    const presentation = { name: definition.name, display_format: 'number', precision: 3, unit: '' }
    const row = { target, period: '6M', status: 'warning', warnings: [{ code: 'WARMUP', message: '窗口预热保留缺失值' }], window, presentation, indicator_name: definition.name,
      value: 1.234, parameters: { window: 37 }, dates: ['2019-01-02', '2019-01-03', '2019-01-04'],
      channels: [{ id: 'sharpe', label: '滚动夏普', values: [null, 1.234, 2.345], precision: 3, display_format: 'number', unit: '' }] }
    const artifact = { ...reference, session_id: 'p', definition, expires_at: '2099-01-01', result: { results: mode === 'empty' ? [] : [row] } }
    const run = { run_id: 'preview-run', session_id: 'p', status: 'running', run_revision: 2, session_revision: 1, response: { session_id: 'p', session_revision: 1, draft, preview: reference, artifacts: { draft, preview: reference }, reply: { text: '试算完成。' } } }
    await page.route('**/api/agent/**', async route => {
      const path = new URL(route.request().url()).pathname
      if (path.endsWith('/meta')) return route.fulfill({ json: { configured: true, model: 'deepseek-v4.1-flash' } })
      if (path.endsWith('/sessions')) { context = route.request().postDataJSON().page_context; return route.fulfill({ json: { session_id: 'p', session_revision: 0 } }) }
      if (path.endsWith('/messages')) { posts++; bodies.push(route.request().postDataJSON()); return route.fulfill({ status: 202, json: run }) }
      if (path.includes('/previews/')) return route.fulfill(mode === 'expired'
        ? { status: 410, json: { detail: { code: 'AGENT_PREVIEW_EXPIRED', message: '试算结果已过期或已清理，请让助手重新试算。' } } }
        : { json: artifact })
      if (path.endsWith('/events')) return route.fulfill({ json: { items: [], has_more: false, next_event_seq: 1 } })
      if (path.endsWith('/invalidate-context')) invalidations++
      if (path.includes('/runs/')) return route.fulfill({ json: run })
      if (path.endsWith('/sessions/p')) return route.fulfill({ json: { session_id: 'p', session_revision: 1, page_context: context, messages: [], next_event_seq: 1, draft, preview: reference, active_run: run } })
      return route.fulfill({ json: {} })
    })
    await page.route('**/api/custom-indicators/validate', route => route.fulfill({ json: { valid: true, diagnostics: [], dependencies: [], compile_token: 'preview-token' } }))
    await page.route('**/api/custom-indicators/evaluate*', route => {
      calculations.push(route.request().postDataJSON())
      return route.fulfill({ json: { results: [row], summary: { ok: 0, warning: 1, error: 0 } } })
    })
    await page.route('**/api/custom-indicators/export-excel', route => {
      calculations.push(route.request().postDataJSON())
      return route.fulfill({ contentType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', body: 'fixture' })
    })
    await page.getByRole('button', { name: '打开 AI 助手' }).click()
    await page.getByRole('textbox', { name: '发送消息' }).fill('选择一个真实产品计算并展示新指标')
    await page.getByRole('button', { name: '发送', exact: true }).click()
    const result = page.getByRole('region', { name: 'AI 试算结果', exact: true })
    if (mode === 'expired') {
      await expect(page.getByRole('alert')).toContainText('试算结果已过期')
      await expect(result).toHaveCount(0)
      run.status = 'completed'
      const input = page.getByRole('textbox', { name: '发送消息' })
      await input.fill('这条补充还未发送')
      await page.getByRole('button', { name: '查看试算结果', exact: true }).click()
      await expect(page.getByRole('region', { name: '本轮试算结果' }).getByRole('alert')).toContainText('试算结果已过期')
      await expect(page.getByRole('dialog', { name: 'AI 助手' })).toBeVisible()
      await expect(input).toHaveValue('这条补充还未发送')
      return
    }
    await expect(result).toBeVisible()
    expect(invalidations).toBe(0)
    await expect(page.getByText('页面研究条件已变化，下一条消息将使用新的上下文；历史试算保留原条件。')).toHaveCount(0)
    run.status = 'completed'
    await page.getByRole('button', { name: '查看试算结果', exact: true }).click()
    await expect(page.getByRole('dialog', { name: 'AI 助手' })).toHaveCount(0)
    await expect(page.locator('#agent-preview-results-title')).toBeFocused()
    await expect(result).toBeVisible()
    await expect(result).toContainText('510300.SH')
    await expect(result).toContainText('2019-12-30')
    if (mode === 'empty') await expect(result).toContainText('没有返回可展示的数据')
    else {
      await expect(result).toContainText('1.234')
      await expect(result).toContainText('窗口预热保留缺失值')
      if (mode === 'time_series') {
        await expect(result.locator('canvas')).toBeVisible()
        await expect(result.getByRole('table')).toContainText('2.345')
        await expect(result.getByRole('table')).toContainText('—')
        await expect(result).toContainText('window=37')
      }
    }
    await expect(page.getByText('已选产品 1 / 10', { exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: '移除 沪深300样例ETF' })).toBeVisible()
    await expect(page.getByLabel('计算周期', { exact: true })).toHaveValue('6M')
    await expect(page.getByLabel('历史截止日', { exact: true })).toHaveValue('2019-12-30')
    await expect(page.getByRole('spinbutton', { name: '动量窗口', exact: true })).toHaveValue(mode === 'empty' ? '20' : '37')
    await expect(page.getByLabel('当前预览指标', { exact: true })).toContainText(definition.name)
    await expect(page.getByRole('button', { name: '预览指标', exact: true })).toBeEnabled()
    // Adopting a tool result must not invalidate the run while it is still replying.
    expect(invalidations).toBe(0)
    await expect(page.getByLabel('名称', { exact: true })).toHaveValue('编辑器原有未保存指标')
    expect(await result.evaluate(el => el.getBoundingClientRect().right <= window.innerWidth)).toBe(true)
    const contrast = await page.evaluate(auditTextContrast)
    expect(contrast.filter(item => item.text.includes('已同步本次试算') || item.text.includes('历史截止日'))).toEqual([])
    if (mode === 'scalar') {
      // A later preview from the same run can adopt a different product too.
      target.product_id = '510050.SH'
      reference.preview_id = artifact.preview_id = 'preview-2'
      await page.getByRole('button', { name: '打开 AI 助手' }).click()
      await expect(result).toContainText('510050.SH')
      await page.getByRole('button', { name: '查看试算结果', exact: true }).click()
      await expect(page.getByRole('dialog', { name: 'AI 助手' })).toHaveCount(0)
      await expect(page.locator('#agent-preview-results-title')).toBeFocused()
      expect(invalidations).toBe(0)
    }
    if (mode === 'time_series') {
      run.status = 'completed'
      await page.screenshot({ path: test.info().outputPath('agent-preview.png'), fullPage: true })
      await page.reload()
      await page.getByRole('button', { name: '打开 AI 助手' }).click()
      await page.getByRole('button', { name: '查看试算结果', exact: true }).click()
      await expect(result.getByRole('table')).toContainText('2.345')
      expect(posts).toBe(1)
      await expect(page.getByRole('spinbutton', { name: '动量窗口', exact: true })).toHaveValue('37')
      // 采纳后的 AI 试算也要随下一条消息冻结：结果来自试算快照，编辑区仍是编辑器草稿。
      await page.getByRole('button', { name: '打开 AI 助手' }).click()
      await page.getByRole('textbox', { name: '发送消息' }).fill('这份 AI 试算为什么是这样？')
      await page.getByRole('button', { name: '发送', exact: true }).click()
      await expect.poll(() => bodies.length).toBe(2)
      const snapshot = bodies[1].page_snapshot
      expect(snapshot.sections.results.displayed_source).toBe('agent_preview')
      expect(snapshot.sections.results.provenance).toMatchObject({ source: 'agent_preview', preview_id: 'preview-1' })
      expect(snapshot.sections.results.groups[0].status).toBe('warning')
      expect(snapshot.sections.results.groups[0].channels[0].sample.head).toEqual([
        { date: '2019-01-02', value: null }, { date: '2019-01-03', value: 1.234 }, { date: '2019-01-04', value: 2.345 }])
      // 编辑器定义与采纳的试算定义必须分开陈述，不得把试算参数挂到编辑器公式上。
      expect(snapshot.sections.editing.definition.source).toBe('editor')
      expect(snapshot.sections.editing.definition.expression).not.toBe(snapshot.sections.editing.active_preview.definition.expression)
      expect(snapshot.sections.editing.active_preview.source).toBe('adopted_agent_preview')
      expect(snapshot.sections.editing.active_preview.definition.expression).toBe('mean(rolling_window(returns,window))')
      expect(snapshot.sections.editing.active_preview.runtime_inputs).toMatchObject({ period: '6M', as_of: '2019-12-30' })
      // series 分区带页面上的完整数组（3 个点），中间点不再丢失。
      expect(snapshot.sections.series.groups[0].channels[0].values).toEqual([null, 1.234, 2.345])
      await page.getByRole('button', { name: '关闭 AI 助手', exact: true }).click()
    }
    if (mode === 'scalar' || mode === 'time_series') {
      await page.getByLabel('计算周期', { exact: true }).selectOption('3M')
      await expect(result).toHaveCount(0)
      if (mode === 'scalar') expect(invalidations).toBe(0)
      run.status = 'completed'
      await page.getByLabel('历史截止日', { exact: true }).fill('2019-11-29')
      await page.getByRole('spinbutton', { name: '动量窗口', exact: true }).fill('15')
      await page.getByRole('spinbutton', { name: '动量窗口', exact: true }).press('Enter')
      if (mode === 'time_series') {
        // 改周期/窗口清掉了旧试算结果，但试算定义仍在该页面生效：重新计算前先问一次，
        // 编辑器定义、活动试算定义、已显示结果必须各自陈述，不能互相冒充。
        await page.getByRole('button', { name: '打开 AI 助手' }).click()
        await page.getByRole('textbox', { name: '发送消息' }).fill('改窗口之后页面按哪个口径计算？')
        await page.getByRole('button', { name: '发送', exact: true }).click()
        await expect.poll(() => bodies.length).toBe(3)
        const stale = bodies[2].page_snapshot.sections
        expect(stale.results.displayed_source).toBeNull()
        expect(stale.results.groups).toEqual([])
        expect(stale.editing.definition.source).toBe('editor')
        expect(stale.editing.definition.expression).not.toBe(stale.editing.active_preview.definition.expression)
        expect(stale.editing.active_preview.source).toBe('adopted_agent_preview')
        expect(stale.editing.active_preview.definition.expression).toBe('mean(rolling_window(returns,window))')
        expect(stale.editing.active_preview.result_adopted).toBe(false)
        expect(stale.editing.active_preview.runtime_inputs).toMatchObject({ period: '3M', as_of: '2019-11-29', runtime_parameters: { window: 15 } })
        await page.getByRole('button', { name: '关闭 AI 助手', exact: true }).click()
      }
      await page.getByRole('button', { name: '预览指标', exact: true }).click()
      await expect.poll(() => calculations.length).toBe(1)
      const request = calculations[0]
      expect(request.period).toBe('3M')
      expect(request.as_of).toBe('2019-11-29')
      expect(mode === 'scalar' ? request.targets : [request.target]).toEqual([{ kind: 'etf', product_id: target.product_id }])
      const instance = mode === 'scalar' ? request : request.indicator_instances[0]
      expect(instance.inline_definition.name).toBe(definition.name)
      expect(instance.parameters).toEqual({ window: 15 })
      await page.getByRole('button', { name: '下载 Excel 计算逻辑', exact: true }).click()
      await expect.poll(() => calculations.length).toBe(2)
      expect(calculations[1]).toMatchObject({ inline_definition: { name: definition.name }, period: '3M', as_of: '2019-11-29', parameters: { window: 15 } })
    }
  })
}


for (const kind of ['scalar', 'time_series']) {
test(`${kind} 指标草稿展示完整公式和所属试算入口，新一轮只显示自己的内容`, async ({ page }) => {
  await openStudio(page)
  let revision = 0, context: any, run: any
  const outputs = [
    { id: 'direction', label: '动量方向', unit: '', precision: 3, display_format: 'number', expression: 'sign(mean(rolling_window(log_returns, momentum_window)))' },
    { id: 'strength', label: '动量强度', unit: '', precision: 3, display_format: 'number', expression: 'multiply(absolute(divide(mean(rolling_window(log_returns, momentum_window)), std(rolling_window(log_returns, momentum_window)))), divide(mean(rolling_window(turnover_amount, volume_short_window)), mean(rolling_window(turnover_amount, volume_long_window))))' },
  ]
  const draft = { valid: true, draft_revision: 1, definition_hash: 'first-hash', definition: { name: '第一个逻辑', result_kind: kind, expression: kind === 'scalar' ? 'mean(returns)' : '', series_outputs: kind === 'time_series' ? outputs : [] } }
  const preview = { preview_id: 'first-preview', run_id: 'r1', definition_hash: 'first-hash', target: { kind: 'etf', product_id: '510300.SH' }, period: '1Y', as_of: '2019-12-31' }
  const messages: any[] = []
  await page.route('**/api/agent/**', async route => {
    const path = new URL(route.request().url()).pathname
    if (path.endsWith('/meta')) return route.fulfill({ json: { configured: true, model: 'deepseek-v4.1-flash' } })
    if (path.endsWith('/sessions')) { context = route.request().postDataJSON().page_context; return route.fulfill({ json: { session_id: 's', session_revision: 0 } }) }
    if (path.endsWith('/messages')) {
      const body = route.request().postDataJSON(); revision++
      const artifacts = revision === 1 ? { draft, preview } : {}
      messages.push({ seq: revision * 2 - 1, id: body.message_id, speaker: 'user', text: body.text }, { seq: revision * 2, id: `r${revision}-reply`, run_id: `r${revision}`, speaker: 'assistant', text: revision === 1 ? '第一项逻辑已完成。' : '我们来讨论第二项逻辑。', artifacts })
      run = { run_id: `r${revision}`, session_id: 's', session_revision: revision, status: 'completed', run_revision: 1, response: { session_id: 's', session_revision: revision, draft, reply: { text: messages.at(-1).text }, artifacts } }
      return route.fulfill({ status: 202, json: run })
    }
    if (path.endsWith('/sessions/s')) return route.fulfill({ json: { session_id: 's', session_revision: revision, page_context: context, messages, next_event_seq: revision * 2 + 1, active_run: run, draft } })
    if (path.includes('/runs/')) return route.fulfill({ json: run })
    if (path.endsWith('/events')) return route.fulfill({ json: { items: [], has_more: false } })
    return route.fulfill({ json: {} })
  })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  const input = page.getByRole('textbox', { name: '发送消息' })
  await input.fill('先研究第一个逻辑')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  const card = page.locator('[data-run-id="r1"]').getByRole('region', { name: '本轮指标草稿' })
  await expect(card).toBeVisible()
  await expect(card.getByRole('heading', { name: '第一个逻辑', exact: true })).toBeVisible()
  await expect(card.getByText('定义校验通过', { exact: true })).toBeVisible()
  const formula = card.locator('pre')
  await expect(formula.first()).toBeHidden()
  const formulaSummary = `查看完整公式（${kind === 'scalar' ? 1 : 2} 个输出）`
  await card.getByRole('button', { name: formulaSummary, exact: true }).click()
  const formulaText = kind === 'scalar' ? ['mean(returns)'] : outputs.map(output => `${output.label} = ${output.expression}`)
  await expect(formula).toHaveText(formulaText)
  await formula.first().focus()
  await expect(formula.first()).toBeFocused()
  if (kind === 'time_series') {
    expect(await formula.last().evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true)
    await card.getByRole('checkbox', { name: '按原始行显示' }).check()
    expect(await formula.last().evaluate(el => el.scrollWidth > el.clientWidth)).toBe(true)
    await formula.last().focus()
    await page.keyboard.press('ArrowRight')
    await expect.poll(() => formula.last().evaluate(el => el.scrollLeft)).toBeGreaterThan(0)
    expect((await page.evaluate(auditTextContrast)).filter(item => item.text.includes('动量方向') || item.text.includes('动量强度'))).toEqual([])
    await card.getByRole('checkbox', { name: '按原始行显示' }).uncheck()
    await formula.first().scrollIntoViewIfNeeded()
    await expectCloseReachable(page)
    await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: test.info().outputPath('series-draft-formulas.png') })
  }
  await card.getByRole('button', { name: '填入编辑器', exact: true }).click()
  await expect(card.getByRole('status')).toContainText('尚未保存')
  await input.fill('现在讨论另一个逻辑')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect(page.locator('[data-run-id="r2"]')).toContainText('第二项逻辑')
  await expect(page.locator('[data-run-id="r2"]').getByRole('region')).toHaveCount(0)
  await expect(card.getByRole('status')).toContainText('尚未保存')
  await expect(page.locator('[data-run-id="r2"]').getByText(/尚未保存/)).toHaveCount(0)
  await expect(page.locator('[data-run-id="r1"]').getByRole('button', { name: '查看试算结果' })).toHaveCount(1)
  await expect(page.getByRole('button', { name: '确认保存指标' })).toHaveCount(0)
  await page.reload()
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await expect(page.locator('[data-run-id="r2"]')).toContainText('第二项逻辑')
  await expect(page.locator('[data-run-id="r2"]').getByRole('region')).toHaveCount(0)
  await expect(page.getByRole('region', { name: '本轮指标草稿' })).toHaveCount(1)
  await expect(card.getByText('历史草稿：填入编辑器后重新校验，再保存。', { exact: true })).toBeVisible()
  await card.getByRole('button', { name: formulaSummary, exact: true }).click()
  await expect(formula).toHaveText(formulaText)
  await expectCloseReachable(page)
  await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: test.info().outputPath('turn-artifacts.png') })
})
}


test('收到更新草稿后的阶段事件不会重放旧运行成果', async ({ page }) => {
  await openStudio(page)
  const api = await agentApi(page, { running: true })
  let holdRun = false, waiting = false
  let previousRun: Record<string, any> | null = null
  let release!: () => void
  const pending = new Promise<void>(resolve => { release = resolve })
  await page.route('**/api/agent/sessions/s/runs/r1', async route => {
    if (holdRun) {
      // An older poll can reach GET /runs before it has read the new events.
      // Let that poll finish with its old receipt; block only after event delivery,
      // so neither a deadlocked poll nor a fresh run snapshot can mask the assertion.
      const delivered = api.deliveredEvents.some(event => event.type === 'draft.updated'
        && event.data?.draft?.definition?.name === '已更新的区间价差')
      if (!delivered) return route.fulfill({ json: previousRun })
      waiting = true; await pending
    }
    await route.fallback()
  })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await page.getByRole('textbox', { name: '发送消息' }).fill('校验并完善指标')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect.poll(() => api.requests.length).toBe(1)
  api.updateDraft(true)
  const card = page.getByRole('region', { name: '本轮指标草稿', exact: true })
  await expect(card.getByRole('heading', { name: '区间平均价差', exact: true })).toBeVisible()
  previousRun = api.runSnapshot()
  holdRun = true
  api.definition.name = '已更新的区间价差'
  api.updateDraft(true)
  api.setPhase('compacting')
  try {
    await expect.poll(() => waiting).toBe(true)
    await expect(card.getByRole('heading', { name: '已更新的区间价差', exact: true })).toBeVisible()
    await expect(card.getByRole('heading', { name: '区间平均价差', exact: true })).toHaveCount(0)
    await expect(page.getByText('正在整理上下文，稍后继续…', { exact: true })).toBeVisible()
    await expectCloseReachable(page)
    expect((await page.evaluate(auditTextContrast)).filter(item => /已更新的区间价差|正在整理上下文/.test(item.text))).toEqual([])
    await page.getByRole('dialog', { name: 'AI 助手' }).screenshot({ path: test.info().outputPath('phase-preserves-latest-artifact.png') })
  } finally { release() }
})

test('整理上下文显示状态并且可以停止，保留对话', async ({ page }) => {
  await openStudio(page)
  const api = await agentApi(page, { running: true })
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await page.getByRole('textbox', { name: '发送消息' }).fill('继续研究动量指标')
  const accepted = page.waitForResponse(response => response.request().method() === 'POST' && new URL(response.url()).pathname.endsWith('/messages'))
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await accepted
  await expect(page.getByText('思考中…', { exact: true })).toBeVisible()
  api.setPhase('tool'); api.toolEvent('tool.started')
  await expect(page.getByText('正在校验指标定义…', { exact: true })).toBeVisible()
  await page.getByText('查看处理记录（最近 1 条）', { exact: true }).click()
  await expect(page.getByText('开始执行', { exact: true })).toBeVisible()
  api.toolEvent('tool.completed')
  await expect(page.getByText('已完成 · 7 毫秒', { exact: true })).toBeVisible()
  api.setPhase('compacting')
  await expect(page.getByText('正在整理上下文，稍后继续…', { exact: true })).toBeVisible()
  expect((await page.evaluate(auditTextContrast)).filter(item => item.text.includes('整理上下文'))).toEqual([])
  await page.screenshot({ path: test.info().outputPath('context-compacting.png') })
  await page.getByRole('button', { name: '停止', exact: true }).click()
  await expect(page.getByText('已停止，已提交的进度已保留。', { exact: true })).toBeVisible()
  await expect(page.getByText('继续研究动量指标', { exact: true })).toBeVisible()
  expect(api.requests).toHaveLength(1)
})

test('手动预览的真实零值随消息冻结，助手读到的是用户看得到的页面证据', async ({ page }) => {
  await openStudio(page)
  const sent: any[] = []
  let lastRun: Record<string, any> | null = null
  const builtIn = { id: 'builtin-volatility', revision: 3, source: 'built_in', read_only: true, name: '收益波动率', description: '收益率序列的样本标准差。', expression: 'std(returns, 1)', periods: ['1Y'], unit: '%', display_format: 'percent', precision: 2, direction: 'lower_better', annual_risk_free_rate_percent: 1.5, display_latex: '\\operatorname{Std}(r)', context_kind: 'single_product', dsl_version: '2.0.0', operator_registry_version: '2.0.0', output_contract: 'scalar', category_id: 'risk', category_label: '风险型指标', indicator_type: 'risk' }
  const row = { indicator_id: null, indicator_revision: null, indicator_name: '收益波动率', target: { kind: 'etf', product_id: '510300.SH', name: '沪深300ETF' }, period: '1Y', value: 0, status: 'ok', warnings: [],
    window: { requested_as_of: null, effective_as_of: '2026-09-18', start_date: '2025-09-18', end_date: '2026-09-18', observation_count: 240, data_latest_date: '2026-09-18' },
    presentation: { name: '收益波动率', display_format: 'percent', precision: 2, unit: '%' } }
  await page.route('**/api/custom-indicators?**', route => route.fulfill({ json: { items: [builtIn], total: 1 } }))
  await page.route('**/api/custom-indicators/meta', route => route.fulfill({ json: { engine_version: 'e2e', workspace_scope: 'shared', limits: {}, variables: [], operators: [], periods: [{ value: '1Y', label: '近 1 年' }], templates: [], predefined_calculations: [], indicator_types: [{ id: 'risk', label: '风险型指标' }] } }))
  await page.route('**/api/custom-indicators/snapshot-config', route => route.fulfill({ json: { schema_version: 1, revision: 1, max_items: 30, updated_at: null, snapshot: null, items: [] } }))
  await page.route('**/api/custom-indicators/validate', route => route.fulfill({ json: { valid: true, diagnostics: [], dependencies: ['returns'], display_latex: '\\operatorname{Std}(r)', editable_latex: 'std(returns, 1)', compile_token: 'e2e-token' } }))
  await page.route('**/api/custom-indicators/evaluate*', route => route.fulfill({ json: { results: [row], summary: { total: 1, ok: 1, warning: 0, error: 0 }, cache: { hits: 0, misses: 1 }, execution: { execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0, request_time_compilation: 0, kernel_signatures: { typed_indicator_plan: ['fixed'] } } } }))
  await page.route('**/api/agent/**', route => {
    const path = new URL(route.request().url()).pathname
    if (path.endsWith('/meta')) return route.fulfill({ json: { configured: true, model: 'fixture' } })
    if (path.endsWith('/sessions')) return route.fulfill({ json: { session_id: 'evidence-e2e', session_revision: 0 } })
    if (path.endsWith('/messages')) {
      const body = route.request().postDataJSON(); sent.push(body)
      lastRun = { run_id: `r${sent.length}`, session_id: 'evidence-e2e', message_id: body.message_id, session_revision: sent.length, run_revision: 1, status: 'completed', phase: 'thinking',
        response: { session_id: 'evidence-e2e', session_revision: sent.length, reply: { text: '已按页面证据核对。' }, artifacts: {} } }
      return route.fulfill({ status: 202, json: lastRun })
    }
    if (path.endsWith('/events')) return route.fulfill({ json: { items: [], has_more: false, next_event_seq: 1 } })
    if (path.includes('/runs/')) return route.fulfill({ json: lastRun || {} })
    if (path.endsWith('/sessions/evidence-e2e')) return route.fulfill({ json: { session_id: 'evidence-e2e', session_revision: sent.length, messages: [], next_event_seq: 1, active_run: lastRun } })
    return route.fulfill({ json: {} })
  })

  await page.goto('/indicator-studio?kind=etf&ids=510300.SH')
  await page.getByRole('button', { name: /收益波动率/ }).click()
  // 还没有任何结果时发送：证据必须说明“尚未预览”，不能凭空给出数值。
  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await page.getByRole('textbox', { name: '发送消息' }).fill('先看看当前页面')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect(page.getByText('已按页面证据核对。')).toBeVisible()
  await expect.poll(() => sent.length).toBe(1)
  expect(sent[0].page_snapshot).toMatchObject({ version: 1, page: 'indicator-studio' })
  expect(sent[0].page_snapshot.sections.results.displayed_source).toBeNull()
  expect(sent[0].page_snapshot.sections.results.pending.join('')).toContain('尚未预览')
  await page.getByRole('button', { name: '关闭 AI 助手', exact: true }).click()

  if (page.viewportSize()!.width < 1280) await page.getByRole('tab', { name: '预览', exact: true }).click()
  await page.getByRole('button', { name: '预览指标', exact: true }).click()
  await expect(page.getByText('预览完成：1 个成功，0 个需关注。')).toBeVisible()
  await expect(page.getByText('0.00%').first()).toBeVisible()

  await page.getByRole('button', { name: '打开 AI 助手' }).click()
  await page.getByRole('textbox', { name: '发送消息' }).fill('这个指标为什么是 0？')
  await page.getByRole('button', { name: '发送', exact: true }).click()
  await expect.poll(() => sent.length).toBe(2)
  const snapshot = sent[1].page_snapshot
  expect(snapshot.snapshot_id).not.toBe(sent[0].page_snapshot.snapshot_id)
  expect(snapshot.sections.results.displayed_source).toBe('manual_preview')
  expect(snapshot.sections.results.groups[0].value).toBe(0)
  expect(snapshot.sections.results.groups[0].window.effective_as_of).toBe('2026-09-18')
  expect(snapshot.sections.results.frozen_request.definition.expression).toBe('std(returns, 1)')
  expect(snapshot.sections.results.frozen_request.parameters).toBeNull()
  expect(snapshot.sections.results.frozen_request.parameters_submitted).toBe(false)
  expect(snapshot.sections.series.groups[0].unavailable.code).toBe('series_not_on_page')
  expect(snapshot.sections.editing.selection).toMatchObject({ indicator_id: 'builtin-volatility', indicator_revision: 3 })
  expect(snapshot.sections.editing.runtime_inputs.targets[0]).toMatchObject({ kind: 'etf', product_id: '510300.SH' })
  expect(snapshot.sections.editing.definition.expression).toBe('std(returns, 1)')
  await expect(page.getByText('已按页面证据核对。')).toHaveCount(2)
  await expect(page.getByRole('dialog', { name: 'AI 助手' })).toBeVisible()
  await expectCloseReachable(page)
})
