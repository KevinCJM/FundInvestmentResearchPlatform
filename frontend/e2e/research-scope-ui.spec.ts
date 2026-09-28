import { test, expect, type Page } from '@playwright/test'
import { auditTextContrast } from './helpers/contrast'
import { ltcmaCapabilities, ltcmaItem, ltcmaOptions } from '../src/test/ltcmaFixtures'

/**
 * Fixture-only UI acceptance for the step-02 scope library and the shared
 * batch source picker. Every /api call is intercepted; nothing is written to
 * a real backend.
 */

const day = '2026-09-12'
const pagePit = '2026-09-01'

const mandate = {
  id: 'mandate-ui', name: '长期配置目标', created_at: `${day}T00:00:00Z`, content_hash: 'm'.repeat(64),
  definition: {
    name: '长期配置目标', as_of: day, review_date: '2099-09-12', currency: 'CNY', horizon_years: 10,
    target_return: 0.03, target_excess_return: 0, min_cash_weight: 0, max_volatility: 0.15,
    min_liquid_weight: 0.2, max_illiquid_weight: 0, max_tracking_error: 0.04, risk_aversion: 5,
    rebalance_policy: 'quarterly', rebalance_note: '', note: '',
  },
}

const strategicScope = {
  mandate_id: mandate.id, mandate_hash: mandate.content_hash,
  id: 'scope-ui', name: '独立战略范围', content_hash: 'a'.repeat(64), created_at: day,
  preview_hash: 'b'.repeat(64), research_only: true, implementation_status: 'unmapped',
  implementation_gaps: ['growth'],
  definition: {
    name: '独立战略范围', as_of: day, currency: 'CNY', source: '',
    assets: [{
      id: 'growth', name: '增长资产', currency: 'CNY', role: 'growth', liquidity: 'liquid', rationale: '', source: '',
      research_proxy: {
        asset_type: 'market', cash_return: null, rebalance: 'daily',
        components: [{ kind: 'index', series_id: 'index:index_daily:000300.SH', field: 'close', weight: 1 }],
        source_labels: { 'index:index_daily:000300.SH': '沪深300指数' },
      },
    }],
  },
}

const poolVersion = {
  id: 'version-ui', pool_id: 'pool-ui', pool_name: '核心产品池', version: 3, pool_revision: 4,
  description: '', purpose: '', owner: '', effective_from: '2026-09-01', effective_to: null,
  publication_note: '',
  evaluation_plans: [{ plan_id: 'plan-ui', plan_revision: 1, plan_name: '权益评价', product_kind: 'etf', result_id: 'run-ui', as_of: '2026-09-10', selection_mode: 'all_ranked', selection_value: null, ranked_count: 1, excluded_count: 0, imported_count: 1, attached_at: day }],
  members: [], member_counts: { pending: 0, approved: 1, watch: 0, rejected: 0 },
  investable_count: 1, immutable: true, created_at: day,
}

const snapshotSummary = {
  mandate_id: mandate.id, mandate_hash: mandate.content_hash,
  id: 'universe-ui', name: '已保存产品范围', research_date: day, version_ids: ['version-ui'],
  pool_ids: ['pool-ui'], product_count: 1,
  summary: { pool_count: 1, member_count: 1, eligible_count: 1 },
  content_hash: 'c'.repeat(64), immutable: true, created_at: day,
}

const snapshot = {
  ...snapshotSummary,
  groups: [{
    evaluation_plan_id: 'plan-ui', evaluation_plan_revision: 1, evaluation_plan_name: '权益评价', product_count: 1,
    products: [{ key: 'etf:510300.SH', kind: 'etf', product_id: '510300.SH', code: '510300.SH', name: '沪深300ETF', usage_status: 'normal', max_weight: null, valid_until: null, substitute_group: '', reasons: [], source_version_ids: ['version-ui'], source_pool_ids: ['pool-ui'] }],
  }],
}

const catalog = { allocations: [], mandates: [mandate], assumptions: [], policies: [], strategic_universes: [strategicScope], implementation_maps: [] }

function sourcePage(url: URL) {
  const offset = Number(url.searchParams.get('offset') ?? 0)
  const kind = url.searchParams.get('kind') ?? 'index'
  const items = kind === 'etf'
    ? [
        { id: `etf:fund_daily:${offset}.SH`, code: `${offset}.SH`, name: offset === 0 ? 'ETF 一号' : 'ETF 二十一', kind: 'etf', coverage: { start_date: '2020-01-02', end_date: '2026-09-17' }, reference_capability: { available: true, supported_fields: ['adj_nav'] } },
        { id: `etf:fund_daily:stop-${offset}`, code: '000000.SZ', name: '停用来源', kind: 'etf', coverage: {}, reference_capability: { available: false, reason: '数据不足' } },
      ]
    : kind === 'index'
      ? [{ id: 'index:index_daily:000300.SH', code: '000300.SH', name: '沪深300指数', kind: 'index', coverage: { start_date: '2019-01-02', end_date: '2026-09-17' }, reference_capability: { available: true, supported_fields: ['close'] } }]
      : []
  return { items, total: 40, offset, limit: 20, problems: [] }
}

async function installFixtures(page: Page, writes: string[]) {
  await page.route('**/api/**', async route => {
    const request = route.request()
    const url = new URL(request.url())
    if (request.method() !== 'GET') {
      writes.push(`${request.method()} ${url.pathname}`)
      return route.fulfill({ status: 200, json: {} })
    }
    if (url.pathname === '/api/pit/settings') {
      return route.fulfill({ json: { settings: { active_release_id: null }, effective: { no_pit: false, label: '研究模式', as_of: pagePit, run_mode: 'RESEARCH' }, available_releases: [] } })
    }
    if (url.pathname === '/api/strategic-allocation/catalog') return route.fulfill({ json: catalog })
    if (url.pathname === '/api/investable-universe-snapshots') return route.fulfill({ json: { items: [snapshotSummary], total: 1 } })
    if (url.pathname === '/api/investable-universe-snapshots/universe-ui') return route.fulfill({ json: snapshot })
    if (url.pathname === '/api/strategic-allocation/universes/scope-ui') return route.fulfill({ json: strategicScope })
    if (url.pathname === '/api/product-pool-versions') return route.fulfill({ json: { items: [poolVersion], total: 1 } })
    if (url.pathname === '/api/strategic-allocation/risk-scales/capabilities') {
      return route.fulfill({ json: { ready: true, algorithms: [], limits: {}, reference: {}, templates: [], trust_mode: 'single_local_trusted_workspace' } })
    }
    if (url.pathname === '/api/strategic-allocation/reference-inputs/catalog') return route.fulfill({ json: sourcePage(url) })
    return route.fulfill({ status: 404, json: { detail: { message: `fixture missing: ${url.pathname}` } } })
  })
}

async function expectNoOverflow(page: Page) {
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1)
}

test('战略范围直接保存，错误就近显示并可重试', async ({ page }, info) => {
  await installFixtures(page, [])
  let checks = 0
  let saves = 0
  let saved: typeof strategicScope | null = null
  await page.route('**/api/strategic-allocation/universes/**', async route => {
    const url = new URL(route.request().url())
    if (url.pathname.endsWith('/preview')) {
      if (++checks === 1) return route.fulfill({ status: 422, json: { detail: [0, 1, 2].map(index => ({ type: 'extra_forbidden', loc: ['body', 'assets', index, 'research_proxy'], msg: 'Extra inputs are not permitted' })) } })
      return route.fulfill({ json: { ...strategicScope, definition: route.request().postDataJSON() } })
    }
    if (url.pathname.endsWith('/confirm')) {
      const body = route.request().postDataJSON()
      expect(body.replaces_universe_id).toBe('scope-ui')
      expect(body.preview_hash).toBe(strategicScope.preview_hash)
      if (++saves === 1) return route.fulfill({ status: 503, json: { detail: { message: '保存服务暂时不可用，请重试。' } } })
      saved = { ...strategicScope, id: 'scope-retried', definition: body.request }
      return route.fulfill({ status: 201, json: saved })
    }
    if (url.pathname.endsWith('/scope-retried') && saved) return route.fulfill({ json: saved })
    return route.fallback()
  })
  await page.goto('/pre-investment/product-pool/new?scope=strategic&strategic_universe=scope-ui&edit=1&mandate=mandate-ui')
  const save = page.getByRole('button', { name: '保存修改', exact: true })
  await expect(save).toBeEnabled()
  await expect(page.getByRole('button', { name: '预览战略范围' })).toHaveCount(0)
  await page.getByLabel('战略范围名称').fill('')
  await expect(save).toBeDisabled()
  await expect(save).toHaveAttribute('aria-describedby', 'scope-save-reason')
  await expect(page.locator('#scope-save-reason')).toHaveText('请填写战略范围名称。')
  await page.getByLabel('战略范围名称').fill('独立战略范围')
  await save.click()
  await expect(page.getByRole('alert')).toHaveText('保存失败：当前页面与服务版本不匹配，暂时无法保存指数、产品或现金收益配置。已填内容已保留，请更新或重启本项目服务后重试。')
  await expect(page.getByRole('alert')).not.toContainText(/assets|research_proxy|Extra inputs/)
  await expectNoOverflow(page)
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  await page.getByRole('alert').screenshot({ path: info.outputPath('scope-version-error.png') })
  expect(saves).toBe(0)
  await save.click()
  await expect(page.getByRole('alert')).toHaveText('保存失败：保存服务暂时不可用，请重试。')
  await expect(save).toBeEnabled()
  await expect(page.getByLabel('战略范围名称')).toHaveValue('独立战略范围')
  await expectNoOverflow(page)
  // 等待共享按钮从禁用状态恢复的透明度过渡结束，再测量稳定显示。
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  await page.screenshot({ path: info.outputPath('scope-save-error.png'), fullPage: true })
  await save.click()
  await expect(page.getByRole('table', { name: '只读战略资产摘要' })).toBeVisible()
  expect(checks).toBe(3)
  expect(saves).toBe(2)
})

test('战略大类直接配置代理与现金收益，保存恢复并交接 LTCMA', async ({ page }, info) => {
  await installFixtures(page, [])
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  let saved: typeof strategicScope | null = null
  await page.route('**/api/strategic-allocation/universes/**', async route => {
    const url = new URL(route.request().url())
    if (url.pathname.endsWith('/preview')) return route.fulfill({ json: { ...strategicScope, definition: route.request().postDataJSON() } })
    if (url.pathname.endsWith('/confirm')) {
      const definition = route.request().postDataJSON().request
      saved = { ...strategicScope, id: 'scope-proxy-ui', name: definition.name, definition }
      return route.fulfill({ status: 201, json: saved })
    }
    if (url.pathname.endsWith('/scope-proxy-ui') && saved) return route.fulfill({ json: saved })
    return route.fallback()
  })
  await page.route('**/api/strategic-allocation/cma/study-options', route => route.fulfill({ json: { ...catalog, strategic_universes: saved ? [saved] : [], regime_runs: [] } }))
  await page.route('**/api/strategic-allocation/cma/capabilities', route => route.fulfill({ json: ltcmaCapabilities }))
  await page.goto('/pre-investment/product-pool/new?scope=strategic&new=proxy-ui&mandate=mandate-ui')
  const objective = page.getByRole('region', { name: '当前投资目标与约束' })
  await expect(objective.getByRole('heading', { name: '长期配置目标' })).toBeVisible()
  await expect(objective.getByText('3.00%')).toBeVisible()
  const objectiveDetails = objective.locator('details')
  await expect(objectiveDetails).not.toHaveAttribute('open')
  const backBox = await page.getByRole('link', { name: '← 返回研究范围库' }).boundingBox()
  const objectiveBox = await objective.boundingBox()
  const editorBox = await page.getByRole('region', { name: '独立战略范围' }).boundingBox()
  expect(objectiveBox!.y).toBeGreaterThan(backBox!.y + backBox!.height)
  expect(editorBox!.y).toBeGreaterThan(objectiveBox!.y + objectiveBox!.height)
  await objective.getByText('查看目标与约束详情').focus()
  await page.keyboard.press('Enter')
  await expect(objectiveDetails).toHaveAttribute('open', '')
  await expect(objective.getByText('可变现资产最低占比')).toBeVisible()
  await expect(objective.getByText('20.00%')).toBeVisible()
  await expectNoOverflow(page)
  expect(await page.evaluate(auditTextContrast)).toEqual([])
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: info.outputPath('scope-objective-expanded.png'), fullPage: true })
  await objective.getByText('查看目标与约束详情').click()
  await page.getByLabel('战略范围名称').fill('股票与现金研究')
  const notes = page.getByText('补充说明（非必填）').locator('..')
  await expect(notes).toHaveCSS('border-top-width', '0px')
  await expect(notes).toHaveCSS('border-bottom-width', '1px')
  await page.getByRole('button', { name: '添加参考大类' }).click()
  const equity = page.getByTestId('risk-reference-asset').nth(0)
  await equity.getByLabel('大类名称').fill('中国股票')
  await expect(page.getByText(/稳定ID|标识与补充说明/)).toHaveCount(0)
  await equity.getByRole('button', { name: '添加或更换代理' }).click()
  const picker = page.getByRole('region', { name: '代理来源选择' })
  await picker.getByRole('checkbox', { name: /沪深300指数/ }).check()
  await picker.getByLabel('来源类型').selectOption('etf')
  await picker.getByRole('checkbox', { name: /ETF 一号/ }).check()
  await picker.getByRole('button', { name: '确认选择' }).click()
  await expect(page.getByRole('button', { name: '预览战略范围' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: '保存战略范围' })).toBeDisabled()
  await expect(page.locator('#scope-save-reason')).toContainText('代理权重须合计 100%')
  await equity.getByLabel('成分权重（%）').nth(0).fill('60')
  await equity.getByLabel('成分权重（%）').nth(1).fill('40')
  await equity.getByText('备注（选填）').focus()
  await page.keyboard.press('Enter')
  await expect(page.getByLabel(/配置用途|变现能力/)).toHaveCount(0)
  await equity.getByLabel('资产1备注').fill('用于长期权益研究')
  await equity.getByText('备注（选填）').click()
  await page.getByRole('button', { name: '添加参考大类' }).click()
  const cash = page.getByTestId('risk-reference-asset').nth(1)
  await cash.getByLabel('大类名称').fill('现金储备')
  await cash.getByLabel('资产类型').selectOption('cash')
  await cash.getByLabel('现金预期年化收益率（%）').fill('2')
  await expect(cash.getByRole('button', { name: '添加或更换代理' })).toHaveCount(0)
  await expect(cash.getByLabel('代理再平衡')).toHaveCount(0)
  await expectNoOverflow(page)
  expect(await page.evaluate(auditTextContrast)).toEqual([])
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: info.outputPath('strategic-proxies-editor.png'), fullPage: true })
  await page.getByRole('button', { name: '保存战略范围' }).click()
  await expect(page.getByRole('table', { name: '只读战略资产摘要' })).toContainText('沪深300指数')
  expect(saved!.definition.assets[1]).toMatchObject({ role: 'liquidity', liquidity: 'liquid', research_proxy: { cash_return: .02, asset_type: 'cash' } })
  await expect(page.getByRole('columnheader', { name: /配置用途|变现能力/ })).toHaveCount(0)
  const actions = page.getByRole('group', { name: '战略范围操作' })
  await expect(actions.getByRole('link')).toHaveText(['下一步', '返回编辑'])
  await expect(actions.getByRole('button')).toHaveCount(0)
  const dimensions = await actions.getByRole('link').evaluateAll(nodes => nodes.map(node => {
    const rect = node.getBoundingClientRect(), style = getComputedStyle(node)
    return { width: rect.width, height: rect.height, radius: style.borderRadius, textDecoration: style.textDecorationLine }
  }))
  expect(dimensions[0]).toEqual(dimensions[1])
  expect(dimensions[0].height).toBeGreaterThanOrEqual(40)
  expect(dimensions[0].textDecoration).toBe('none')
  await actions.scrollIntoViewIfNeeded()
  await expectNoOverflow(page)
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  await actions.screenshot({ path: info.outputPath('scope-bottom-buttons.png') })
  await actions.getByRole('link', { name: '返回编辑' }).focus()
  await page.keyboard.press('Enter')
  await expect(page.getByRole('button', { name: '保存修改' })).toBeVisible()
  await expect(page.getByLabel('战略范围名称')).toHaveValue('股票与现金研究')
  await page.getByRole('button', { name: '取消编辑' }).click()
  await page.reload()
  await expect(page.getByRole('table', { name: '只读战略资产摘要' })).toContainText('2.00%')
  await page.getByRole('link', { name: '下一步', exact: true }).click()
  await page.getByLabel('生成方法').selectOption('historical_statistics')
  await page.getByText('研究代理', { exact: true }).click()
  await expect(page.getByTestId('risk-reference-asset').first()).toContainText('沪深300指数')
  await expect(page.getByLabel('现金预期年化收益率（%）')).toHaveValue('2')
  await expect(page.getByLabel('成分权重（%）').nth(0)).toHaveValue('60')
  await expect(page.getByLabel('成分权重（%）').nth(1)).toHaveValue('40')
  await expectNoOverflow(page)
  expect(errors).toEqual([])
})

test('常用大类可一键补齐、隐藏与恢复建议', async ({ page }, info) => {
  await installFixtures(page, [])
  await page.goto('/pre-investment/product-pool/new?scope=strategic&new=palette-ui&mandate=mandate-ui')
  await page.getByLabel('战略范围名称').fill('复用常用大类')

  // 建议来自已保存范围里配置过的大类；点一下连研究代理一起带入，不必先加空行再逐项填。
  const palette = page.getByRole('list', { name: '常用大类' })
  await expect(palette).toContainText('沪深300指数 100%')
  await palette.getByRole('button', { name: /^增长资产/ }).click()
  const row = page.getByTestId('risk-reference-asset').first()
  await expect(row.getByLabel('大类名称')).toHaveValue('增长资产')
  await expect(row).toContainText('已配代理')
  await expect(row).toContainText('沪深300指数 100.00%')
  await expect(page.getByText('共 1 个大类 · 1 个可用 · 0 个待完成')).toBeVisible()
  await expect(palette.getByRole('button', { name: /增长资产（已添加）/ })).toBeDisabled()
  await expect(page.getByRole('button', { name: '保存战略范围', exact: true })).toBeEnabled()

  await expectNoOverflow(page)
  // 保存按钮刚从禁用恢复，等透明度过渡结束再测稳定显示。
  await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
  await page.screenshot({ path: info.outputPath('scope-category-palette.png'), fullPage: true })

  // 隐藏只影响本机建议列表，不改已配好的大类，并且可以整组恢复。
  await palette.getByRole('button', { name: '不再推荐：增长资产' }).click()
  await expect(palette).toHaveCount(0)
  await expect(row.getByLabel('大类名称')).toHaveValue('增长资产')
  await page.getByRole('button', { name: '恢复已隐藏的 1 个' }).click()
  await expect(palette.getByRole('button', { name: /^增长资产/ })).toBeVisible()
})

test('代理候选清单限高内滚，表头吸顶，不把后面的大类顶出页面', async ({ page }, info) => {
  await installFixtures(page, [])
  // 真实目录里「沪深300」能搜出几十条；夹具默认只回一条，看不出清单会不会一路顶开页面。
  await page.route('**/api/strategic-allocation/reference-inputs/catalog**', route => route.fulfill({
    json: {
      items: Array.from({ length: 20 }, (_, index) => ({ id: `index:index_daily:00030${index}.CSI`, code: `00030${index}.CSI`, name: `沪深300(全)第${index + 1}档`, kind: 'index', coverage: { start_date: '2019-01-02', end_date: '2026-09-17' }, reference_capability: { available: true, supported_fields: ['close'] } })),
      total: 40, offset: 0, limit: 20, problems: [],
    },
  }))
  await page.goto('/pre-investment/product-pool/new?scope=strategic&new=cap-ui&mandate=mandate-ui')
  await page.getByRole('button', { name: '添加参考大类' }).click()
  await page.getByRole('button', { name: '添加参考大类' }).click()
  const rows = page.getByTestId('risk-reference-asset')
  await rows.first().getByRole('button', { name: '添加或更换代理' }).click()

  const list = page.getByRole('region', { name: '代理来源选择' }).locator('div[aria-label="代理来源选择"]')
  await expect(list.getByRole('row')).toHaveCount(21)
  const box = await list.evaluate((node: HTMLElement) => ({ client: node.clientHeight, scroll: node.scrollHeight }))
  expect(box.client).toBeLessThanOrEqual(384)
  expect(box.scroll).toBeGreaterThan(box.client)

  // 滚到底后表头仍压在清单顶部，滚过去的行不会盖住它。
  await list.evaluate((node: HTMLElement) => { node.scrollTop = node.scrollHeight })
  const header = await list.getByRole('columnheader', { name: '名称与代码' }).boundingBox()
  const frame = await list.boundingBox()
  expect(header!.y - frame!.y).toBeLessThanOrEqual(2)

  // 第二个大类没有被候选清单推到几屏之外。
  const next = await rows.nth(1).boundingBox()
  expect(next!.y - (frame!.y + frame!.height)).toBeLessThan(400)
  await expectNoOverflow(page)
  await page.screenshot({ path: info.outputPath('scope-source-picker-capped.png'), fullPage: true })
})

test('范围恢复目录失败可重试，不显示空白工作区', async ({ page }) => {
  await installFixtures(page, [])
  let fail = true
  await page.route('**/api/strategic-allocation/catalog', route => fail
    ? route.fulfill({ status: 503, json: { detail: { message: '范围目录暂不可用' } } })
    : route.fulfill({ json: catalog }))
  await page.goto('/pre-investment/product-pool/new?scope=strategic&strategic_universe=scope-ui')
  await expect(page.getByRole('alert')).toContainText('范围目录暂不可用')
  await expectNoOverflow(page)
  fail = false
  await page.getByRole('button', { name: '重试读取投资目标' }).click()
  await expect(page.getByRole('combobox', { name: '投资目标与约束' })).toHaveValue(mandate.id)
  await expect(page.getByRole('combobox', { name: '投资目标与约束' })).toBeDisabled()
  await expect(page.getByRole('table', { name: '只读战略资产摘要' })).toBeVisible()
})

for (const kind of ['strategic', 'product']) {
  test(`${kind} 目标选择移入配置，列表展示目标名称且说明使用整行`, async ({ page }, info) => {
    const writes: string[] = []
    await installFixtures(page, writes)
    const strategic = kind === 'strategic'
    await page.goto(`/pre-investment/product-pool${strategic ? '?scope=strategic' : ''}`)
    await expect(page.getByRole('cell', { name: mandate.name, exact: true })).toBeVisible()
    await expect(page.getByRole('combobox', { name: '投资目标与约束' })).toHaveCount(0)
    const intro = page.locator('section[aria-labelledby="research-start-heading"]')
    const widths = await intro.evaluate(element => {
      const style = getComputedStyle(element)
      const available = element.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
      return { available, paragraphs: [...element.querySelectorAll(':scope > p')].map(p => p.getBoundingClientRect().width) }
    })
    expect(widths.paragraphs.length).toBeGreaterThanOrEqual(2)
    for (const width of widths.paragraphs) expect(Math.abs(width - widths.available)).toBeLessThanOrEqual(1)
    await expectNoOverflow(page)
    await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
    await page.screenshot({ path: info.outputPath(`${kind}-scope-library.png`), fullPage: true })
    await page.getByRole('link', { name: '新建研究范围', exact: true }).click()
    const selector = page.getByRole('combobox', { name: '投资目标与约束', exact: true })
    await expect(selector).toBeEnabled()
    await expect(selector).toHaveAttribute('required', '')
    await expect(selector).toHaveValue('')
    const container = selector.locator('xpath=ancestor::section[1]')
    await expect(container).toContainText(strategic ? '配置大类资产与研究代理' : '选择产品池版本')
    if (strategic) {
      await page.getByRole('button', { name: '添加参考大类' }).click()
      await page.getByLabel('大类名称', { exact: true }).fill('现金')
      await page.getByLabel('资产类型', { exact: true }).selectOption('cash')
    } else await page.getByRole('checkbox', { name: /核心产品池/ }).check()
    const save = page.getByRole('button', { name: strategic ? '保存战略范围' : '生成锁定快照', exact: true })
    await expect(save).toBeDisabled()
    await selector.focus()
    await selector.selectOption(mandate.id)
    await expect(save).toBeEnabled()
    await expect(page).toHaveURL(/mandate=mandate-ui/)
    await expect(page.getByRole('region', { name: '当前投资目标与约束' })).toContainText(mandate.name)
    await page.reload()
    await expect(selector).toHaveValue(mandate.id)
    await expect(save).toBeEnabled()
    await expectNoOverflow(page)
    await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
    await page.screenshot({ path: info.outputPath(`${kind}-mandate-in-config.png`), fullPage: true })
    // 初筛可能发起只读预览，但不应保存、关联或改写业务记录。
    expect(writes.filter(write => !write.endsWith('/scope-feasibility'))).toEqual([])
  })
}

for (const kind of ['strategic', 'product']) {
  test(`${kind} 范围绑定目标：从库编辑、清空书签与刷新仍自动展示`, async ({ page }, info) => {
    const writes: string[] = []
    await installFixtures(page, writes)
    const strategic = kind === 'strategic'
    await page.goto(`/pre-investment/product-pool${strategic ? '?scope=strategic' : ''}`)
    await page.getByRole('link', { name: '编辑', exact: true }).first().click()
    const panel = page.getByRole('region', { name: '当前投资目标与约束' })
    await expect(panel.getByRole('heading', { name: mandate.name })).toBeVisible()
    await expect(page.getByRole('button', { name: '保存修改' })).toBeEnabled()
    await panel.getByText('查看目标与约束详情').click()
    await expect(panel.getByText('20.00%')).toBeVisible()
    await expectNoOverflow(page)
    await expect.poll(() => page.evaluate(auditTextContrast)).toEqual([])
    await page.screenshot({ path: info.outputPath(`${kind}-bound-mandate.png`), fullPage: true })
    // 完全移除书签和 URL 目标，再直达编辑链接，服务端关联仍然生效。
    await page.evaluate(() => { localStorage.clear(); sessionStorage.clear() })
    const query = strategic ? 'scope=strategic&strategic_universe=scope-ui&edit=1' : 'universe=universe-ui&edit=universe-ui'
    await page.goto(`/pre-investment/product-pool/new?${query}`)
    await expect(panel.getByRole('heading', { name: mandate.name })).toBeVisible()
    await expect(page).toHaveURL(/mandate=mandate-ui/)
    await page.reload()
    await expect(panel.getByRole('heading', { name: mandate.name })).toBeVisible()
    await expect(page.getByRole('button', { name: '保存修改' })).toBeEnabled()
    expect(writes).toEqual([])
  })
}

test('范围库在两条路径可搜索、恢复、复制与新建，目标与 PIT 提示保持稳定', async ({ page }, info) => {
  const writes: string[] = []
  await installFixtures(page, writes)
  await page.goto('/pre-investment/product-pool?mandate=mandate-ui')

  // 产品范围库：搜索与空结果不丢工具栏。
  await expect(page.getByRole('link', { name: '已保存产品范围' })).toBeVisible()
  await page.getByLabel('搜索范围名称').fill('不存在')
  await expect(page.getByText(/没有匹配的研究范围/)).toBeVisible()
  await page.getByLabel('搜索范围名称').fill('')

  // 继续研究 → 冻结快照只读摘要，目标保持。
  await page.getByRole('link', { name: '继续研究' }).first().click()
  await expect(page.getByRole('heading', { name: '已保存的产品范围' })).toBeVisible()
  await expect(page.getByRole('combobox', { name: '投资目标与约束' })).toHaveValue('mandate-ui')
  await expect(page).toHaveURL(/mandate=mandate-ui/)

  // 复制新建 → 可编辑副本；新建 → 空编辑器。
  await page.getByRole('link', { name: '复制此范围新建' }).click()
  await expect(page.getByLabel('研究名称')).toHaveValue('已保存产品范围（副本）')
  await page.getByRole('button', { name: '新建研究范围' }).click()
  await expect(page.getByLabel('研究名称')).toHaveValue('投前研究可投资域')
  await expect(page.getByRole('checkbox', { name: /核心产品池 · V3/ })).not.toBeChecked()
  await expect(page.getByRole('heading', { name: '已保存的产品范围' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: '进入自动构建大类' })).toHaveCount(0)

  // PIT 不一致只提醒，不阻断保存。
  await expect(page.getByText(/与当前页面 PIT 日期 2026-09-01 不一致/)).toBeVisible()
  await page.getByRole('checkbox', { name: /核心产品池 · V3/ }).check()
  await expect(page.getByRole('button', { name: '生成锁定快照' })).toBeEnabled()

  // 战略范围库与只读摘要，目标仍保持。
  await page.getByRole('link', { name: '先做战略研究：独立资产范围' }).click()
  await expect(page.getByRole('link', { name: '独立战略范围' })).toBeVisible()
  await page.getByRole('link', { name: '继续研究' }).first().click()
  await expect(page.getByRole('table', { name: '只读战略资产摘要' })).toBeVisible()
  await expect(page.getByRole('combobox', { name: '投资目标与约束' })).toHaveValue('mandate-ui')
  await expect(page).toHaveURL(/mandate=mandate-ui/)

  // 同一 ID 的复制、返回与新建必须切换视图身份，不残留旧摘要或前向入口。
  await page.getByRole('link', { name: '复制新建' }).click()
  await expect(page.getByLabel('战略范围名称')).toHaveValue('独立战略范围（副本）')
  await expect(page.getByRole('table', { name: '只读战略资产摘要' })).toHaveCount(0)
  await page.getByRole('link', { name: '独立战略范围' }).click()
  await expect(page.getByRole('table', { name: '只读战略资产摘要' })).toBeVisible()
  await page.getByRole('button', { name: '新建研究范围' }).click()
  await expect(page.getByLabel('战略范围名称')).toHaveValue('独立战略范围 2026-09-01')
  await expect(page.getByRole('table', { name: '只读战略资产摘要' })).toHaveCount(0)
  await expect(page.getByRole('link', { name: '下一步', exact: true })).toHaveCount(0)
  await page.getByRole('button', { name: '新建研究范围' }).click()
  await expect(page.getByLabel('战略范围名称')).toHaveValue('独立战略范围 2026-09-01')

  expect(writes).toEqual([])
  await page.evaluate(() => window.scrollTo(0, 0))
  await expectNoOverflow(page)
  expect(await page.evaluate(auditTextContrast)).toEqual([])
  await page.screenshot({ path: info.outputPath(`scope-library-${info.project.name}.png`), fullPage: true })
})

test('范围库多行时继续研究、复制与新建把工作区带入视口并聚焦', async ({ page }, info) => {
  const writes: string[] = []
  await installFixtures(page, writes)
  const many = Array.from({ length: 24 }, (_, index) => ({
    ...snapshotSummary,
    id: `universe-${index + 1}`,
    name: `范围 ${String(index + 1).padStart(2, '0')}`,
    created_at: `2026-09-${String((index % 28) + 1).padStart(2, '0')}T00:00:00Z`,
  }))
  // 后注册的路由优先：只覆盖范围库列表与按 ID 恢复，其余夹具保持不变。
  await page.route('**/api/investable-universe-snapshots/*', route => route.fulfill({ json: snapshot }))
  await page.route('**/api/investable-universe-snapshots', route => route.fulfill({ json: { items: many, total: many.length } }))
  await page.goto('/pre-investment/product-pool?mandate=mandate-ui')

  // 点击靠后的行：工作区应回到视口（避开吸顶头部）并交出焦点。
  await page.getByRole('link', { name: '范围 24' }).scrollIntoViewIfNeeded()
  await page.getByRole('link', { name: '继续研究' }).last().click()
  const region = page.getByRole('region', { name: '产品范围工作区' })
  await expect(region).toBeFocused()
  await expect(page.getByRole('heading', { name: '已保存的产品范围' })).toBeInViewport()

  // 复制新建与新建命令同样把编辑器带入视口并聚焦。
  await page.getByRole('link', { name: '复制此范围新建' }).click()
  await expect(region).toBeFocused()
  await expect(page.getByLabel('研究名称')).toHaveValue('已保存产品范围（副本）')
  await page.getByRole('button', { name: '新建研究范围' }).click()
  await expect(region).toBeFocused()
  await expect(page.getByRole('button', { name: '生成锁定快照' })).toBeInViewport()

  expect(writes).toEqual([])
  await expectNoOverflow(page)
  await page.screenshot({ path: info.outputPath(`scope-reveal-${info.project.name}.png`) })
})

test('代理来源批量选择跨搜索、翻页与类型保留，确认与取消可控', async ({ page }, info) => {
  const writes: string[] = []
  await installFixtures(page, writes)
  await page.goto('/settings/risk-scales/new')

  await page.getByLabel('标尺名称').fill('浏览器夹具标尺')
  await page.getByRole('button', { name: '下一步' }).click()
  await page.getByRole('button', { name: '添加参考大类' }).click()
  await page.getByLabel('资产类型').selectOption('market')
  await page.getByRole('button', { name: '添加或更换代理' }).click()
  const picker = page.getByRole('region', { name: '代理来源选择' })
  await expect(picker.getByText('沪深300指数')).toBeVisible()
  await picker.getByRole('checkbox', { name: /沪深300指数/ }).check()
  await expect(picker.getByText('已选 1 项')).toBeVisible()

  await picker.getByLabel('来源类型').selectOption('etf')
  await expect(picker.getByText('ETF 一号')).toBeVisible()
  await picker.getByRole('checkbox', { name: /ETF 一号/ }).check()
  await expect(picker.getByText('已选 2 项')).toBeVisible()
  await picker.getByLabel('搜索名称或代码').fill('ETF')
  await expect(picker.getByText('已选 2 项')).toBeVisible()
  await picker.getByRole('button', { name: '下一步' }).click()
  await expect(picker.getByText('ETF 二十一')).toBeVisible()
  await picker.getByRole('checkbox', { name: /ETF 二十一/ }).check()
  await expect(picker.getByText('已选 3 项')).toBeVisible()

  await picker.getByLabel('来源类型').selectOption('index')
  await expect(picker.getByText('已选 3 项')).toBeVisible()
  await picker.getByLabel('来源类型').selectOption('etf')
  await expect(picker.getByText('已选 3 项')).toBeVisible()
  await expect(picker.getByRole('checkbox', { name: /停用来源/ })).toBeDisabled()

  // 确认前留存打开的选择器与已选托盘；截图前回到页首避免固定头部跨页。
  await page.evaluate(() => window.scrollTo(0, 0))
  await expectNoOverflow(page)
  await page.screenshot({ path: info.outputPath(`source-picker-open-${info.project.name}.png`), fullPage: true })

  await picker.getByRole('button', { name: '确认选择' }).click()
  // 行内一句话摘要，展开区逐条列出成分与权重：三个来源都原子落到同一个大类上。
  const asset = page.getByTestId('risk-reference-asset').first()
  await expect(asset.getByText('沪深300指数 100.00% + ETF 一号 100.00% + ETF 二十一 100.00%')).toBeVisible()
  for (const name of ['沪深300指数', 'ETF 一号', 'ETF 二十一']) {
    await expect(asset.getByLabel(`${name} · 成分权重（%）`)).toHaveValue('100')
  }

  // 已添加不可重复选择；取消不产生修改。
  await page.getByRole('button', { name: '添加或更换代理' }).click()
  const reopened = page.getByRole('region', { name: '代理来源选择' })
  await reopened.getByLabel('来源类型').selectOption('etf')
  await expect(reopened.getByRole('checkbox', { name: /ETF 一号/ })).toBeDisabled()
  await reopened.getByRole('button', { name: '取消' }).click()
  await expect(page.getByRole('region', { name: '代理来源选择' })).toHaveCount(0)

  expect(writes).toEqual([])
  await page.evaluate(() => window.scrollTo(0, 0))
  await expectNoOverflow(page)
  expect(await page.evaluate(auditTextContrast)).toEqual([])
  await page.screenshot({ path: info.outputPath(`source-picker-confirmed-${info.project.name}.png`), fullPage: true })
})

test('产品范围可编辑保存、同名阻止并删除', async ({ page }, info) => {
  const writes: string[] = []
  await installFixtures(page, writes)
  let listed = [snapshotSummary, { ...snapshotSummary, id: 'universe-2', name: '已保存范围乙' }]
  await page.route('**/api/investable-universe-snapshots**', async route => {
    const url = new URL(route.request().url())
    const method = route.request().method()
    if (url.pathname === '/api/investable-universe-snapshots' && method === 'GET') {
      return route.fulfill({ json: { items: listed, total: listed.length } })
    }
    if (url.pathname === '/api/investable-universe-snapshots' && method === 'POST') {
      const body = JSON.parse(route.request().postData() || '{}')
      writes.push(`POST:${body.replaces_snapshot_id ?? ''}`)
      return route.fulfill({ status: 201, json: { ...snapshot, id: 'universe-new', name: body.name } })
    }
    if (url.pathname.endsWith('/universe-ui') && method === 'DELETE') {
      writes.push('DELETE:universe-ui')
      listed = listed.filter(item => item.id !== 'universe-ui')
      return route.fulfill({ json: { deleted: true, id: 'universe-ui' } })
    }
    return route.fulfill({ json: snapshot })
  })
  await page.goto('/pre-investment/product-pool?mandate=mandate-ui&universe=universe-ui&edit=universe-ui')

  const name = page.getByRole('textbox', { name: '研究名称' })
  await expect(name).toHaveValue('已保存产品范围')
  await expect(page.locator('#scope-name-error')).toHaveCount(0)
  await name.fill('已保存范围乙')
  await expect(page.locator('#scope-name-error')).toBeVisible()
  await expect(page.getByRole('button', { name: '保存修改' })).toBeDisabled()
  await name.fill('新的研究范围')
  await page.getByRole('button', { name: '保存修改' }).click()
  await expect(page).toHaveURL(/universe=universe-new/)
  expect(writes).toContain('POST:universe-ui')

  await page.goto('/pre-investment/product-pool?mandate=mandate-ui')
  await page.getByRole('button', { name: '删除' }).first().click()
  await expect(page.getByText(/从列表移除「已保存产品范围」/)).toBeVisible()
  await page.getByRole('button', { name: '取消' }).click()
  await expect(page.getByRole('link', { name: '已保存产品范围' })).toBeVisible()
  await page.getByRole('button', { name: '删除' }).first().click()
  await page.getByRole('button', { name: '确认移除' }).click()
  await expect(page.getByRole('link', { name: '已保存产品范围' })).toHaveCount(0)
  expect(writes).toContain('DELETE:universe-ui')

  await page.evaluate(() => window.scrollTo(0, 0))
  await expectNoOverflow(page)
  await page.screenshot({ path: info.outputPath(`scope-crud-product-${info.project.name}.png`) })
})

test('战略范围可编辑取消、同名阻止、保存修改并删除', async ({ page }, info) => {
  const writes: string[] = []
  await installFixtures(page, writes)
  let universes = [strategicScope, { ...strategicScope, id: 'scope-2', name: '其他范围' }]
  await page.route('**/api/strategic-allocation/universes**', async route => {
    const url = new URL(route.request().url())
    const method = route.request().method()
    if (url.pathname.endsWith('/universes/preview')) return route.fulfill({ json: { ...strategicScope, preview_hash: 'p'.repeat(64) } })
    if (url.pathname.endsWith('/universes/confirm') && method === 'POST') {
      const body = JSON.parse(route.request().postData() || '{}')
      writes.push(`CONFIRM:${body.replaces_universe_id ?? ''}`)
      const saved = { ...strategicScope, id: 'scope-new', name: body.request?.name ?? strategicScope.name }
      universes = [saved, ...universes.filter(item => item.id !== 'scope-ui')]
      return route.fulfill({ status: 201, json: saved })
    }
    if (method === 'DELETE') {
      const id = url.pathname.split('/').pop() ?? ''
      writes.push(`DELETE:${id}`)
      universes = universes.filter(item => item.id !== id)
      return route.fulfill({ json: { deleted: true, id } })
    }
    return route.fulfill({ json: { ...strategicScope, id: url.pathname.split('/').pop() } })
  })
  await page.route('**/api/strategic-allocation/catalog', route => route.fulfill({
    json: { ...catalog, strategic_universes: universes },
  }))
  await page.goto('/pre-investment/product-pool/new?scope=strategic&mandate=mandate-ui&strategic_universe=scope-ui&edit=1')

  const name = page.getByRole('textbox', { name: '战略范围名称' })
  await expect(name).toHaveValue('独立战略范围')
  // 编辑自己的名字不算重复；与其他活动范围同名才阻止。
  await expect(page.locator('#scope-name-error')).toHaveCount(0)
  await name.fill('其他范围')
  await expect(page.locator('#scope-name-error')).toBeVisible()
  await expect(page.getByRole('button', { name: '保存修改' })).toBeDisabled()
  await name.fill('独立战略范围')
  await expect(page.locator('#scope-name-error')).toHaveCount(0)
  await page.getByRole('button', { name: '取消编辑' }).click()
  await expect(page).toHaveURL(/strategic_universe=scope-ui/)
  await expect(page.getByRole('table', { name: '只读战略资产摘要' })).toBeVisible()
  expect(writes).toEqual([])

  // 重新编辑并保存修改：同名允许，发送 replaces_universe_id 并切到新版本。
  await page.getByRole('link', { name: '返回编辑' }).click()
  await expect(page.getByRole('textbox', { name: '战略范围名称' })).toHaveValue('独立战略范围')
  await expect(page.getByRole('button', { name: '保存修改' })).toBeEnabled()
  await page.getByRole('button', { name: '保存修改' }).click()
  await expect(page).toHaveURL(/strategic_universe=scope-new/)
  expect(writes).toContain('CONFIRM:scope-ui')
  await expect(page.getByRole('table', { name: '只读战略资产摘要' })).toBeVisible()

  await page.goto('/pre-investment/product-pool?scope=strategic&mandate=mandate-ui')
  await page.getByRole('button', { name: '删除' }).first().click()
  await expect(page.getByText(/从列表移除「独立战略范围」/)).toBeVisible()
  await page.getByRole('button', { name: '取消' }).click()
  await expect(page.getByRole('link', { name: '独立战略范围' })).toBeVisible()
  await page.getByRole('button', { name: '删除' }).first().click()
  await page.getByRole('button', { name: '确认移除' }).click()
  await expect(page.getByRole('link', { name: '独立战略范围' })).toHaveCount(0)
  expect(writes).toContain('DELETE:scope-new')

  await page.evaluate(() => window.scrollTo(0, 0))
  await expectNoOverflow(page)
  await page.screenshot({ path: info.outputPath(`scope-crud-strategic-${info.project.name}.png`) })
})

// Review regressions: only fixture APIs, including the save below.
test('审核回归：复制产品范围保留原排除项', async ({ page }, info) => {
  await installFixtures(page, [])
  await page.route('**/api/investable-universe-snapshots/universe-ui', route => route.fulfill({
    json: { ...snapshot, excluded_product_keys: ['etf:excluded'] },
  }))
  let saved: Record<string, unknown> | null = null
  await page.route('**/api/investable-universe-snapshots', async route => {
    if (route.request().method() !== 'POST') return route.fallback()
    saved = route.request().postDataJSON()
    return route.fulfill({ status: 201, json: { ...snapshot, id: 'copy-ui', name: saved!.name, excluded_product_keys: saved!.excluded_product_keys } })
  })
  await page.route('**/api/investable-universe-snapshots/copy-ui', route => route.fulfill({ json: { ...snapshot, id: 'copy-ui', name: '已保存产品范围（副本）', excluded_product_keys: ['etf:excluded'] } }))
  await page.goto('/pre-investment/product-pool/new?universe=universe-ui&copy=universe-ui&mandate=mandate-ui')
  await expect(page.getByLabel('研究名称', { exact: true })).toHaveValue('已保存产品范围（副本）')
  await page.getByRole('button', { name: '生成锁定快照', exact: true }).click()
  await expect.poll(() => saved).not.toBeNull()
  expect(saved).toMatchObject({ excluded_product_keys: ['etf:excluded'], version_ids: ['version-ui'] })
  await expect(page).toHaveURL(/universe=copy-ui/)
  await expectNoOverflow(page)
  await page.screenshot({ path: info.outputPath('copied-scope.png'), fullPage: true })
})

for (const locale of ['zh-CN', 'en-US']) test(`审核回归：NIW 禁用需更新先验 ${locale}`, async ({ page }, info) => {
  await page.addInitScript(language => localStorage.setItem('fund-research.i18n.locale', language), locale)
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/pit/settings') return route.fulfill({ json: { settings: { active_release_id: null },
      effective: { no_pit: false, as_of: ltcmaItem.as_of, run_mode: 'RESEARCH', label: ltcmaItem.as_of }, available_releases: [] } })
    if (path.endsWith('/cma/capabilities')) return route.fulfill({ json: ltcmaCapabilities })
    if (path.endsWith('/cma/sample')) return route.fulfill({ json: {
      requested_start: '2020-01-01', requested_end: ltcmaItem.as_of,
      actual_start: '2020-01-02', actual_end: ltcmaItem.as_of, observations: 158,
      excluded_return_periods: 1, observation_frequency: 'daily', periods_per_year: 252,
      source_hash: 'a'.repeat(64), return_panel_hash: 'b'.repeat(64),
    } })
    if (path.endsWith('/cma/study-options')) return route.fulfill({ json: { ...ltcmaOptions, assumptions: [
      { ...ltcmaItem, name: 'Stale prior', usable: { status: 'stale', reasons: [] } },
      { ...ltcmaItem, id: 'ready-prior', name: 'Ready prior' },
    ] } })
    return route.fulfill({ status: 503, json: { detail: 'Fixture: unrelated API unavailable' } })
  })
  await page.goto('/pre-investment/ltcma/new')
  await page.getByRole('combobox', { name: locale === 'zh-CN' ? '资产范围' : 'Asset scope', exact: true }).selectOption(`allocation:${ltcmaItem.alloc_name}`)
  await page.getByRole('button', { name: locale === 'zh-CN' ? '生成方法' : 'Method', exact: true }).click()
  await page.locator('button[role="menuitemradio"][value="bayesian_niw"]').click()
  const unavailable = page.getByRole('option', { name: /^Stale prior/ })
  await expect(unavailable).toHaveJSProperty('disabled', true)
  await expect(unavailable).toContainText(locale === 'zh-CN' ? '先验或其上游已有新版本' : 'The prior or its upstream research has a newer version')
  await expect(page.getByRole('option', { name: /^Ready prior/ })).toHaveJSProperty('disabled', false)
  const priorSelect = page.getByRole('combobox', { name: locale === 'zh-CN' ? '先验 LTCMA 版本' : 'Prior LTCMA version', exact: true })
  await priorSelect.selectOption('ready-prior')
  await expect(priorSelect).toHaveValue('ready-prior')
  await page.getByRole('button', { name: locale === 'zh-CN' ? '查看样本参考' : 'Check sample', exact: true }).click()
  await expect(page.getByText(locale === 'zh-CN' ? /160 个共同净值日/ : /160 common NAV dates/)).toBeVisible()
  await expect(page.getByText(locale === 'zh-CN' ? '本次样本：158 个日收益观察' : 'Current sample: 158 daily return observations')).toBeVisible()
  await expectNoOverflow(page)
  expect(await page.evaluate(auditTextContrast)).toEqual([])
  await page.screenshot({ path: info.outputPath(`niw-priors-${locale}.png`), fullPage: true })
})
