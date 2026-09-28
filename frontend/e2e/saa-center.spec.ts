import { expect, test } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { cmaVersion, policyBaseline, policyPreview, policyFrontierFixture, strategicCatalog } from '../src/test/strategicAllocationFixtures'
import { auditTextContrast } from './helpers/contrast'

const system = JSON.parse(readFileSync(new URL('../../locales/system.json', import.meta.url), 'utf8')) as Record<string, Record<string, string>>

const plan = {
  id: policyBaseline.id, name: '三年稳健配置', as_of: '2019-12-31', created_at: '2026-09-23', mode: 'compatible_all_models',
  mandate: { id: 'mandate-1', name: '三年期绝对收益计划', definition: { target_return: .0772, max_volatility: .095, min_cash_weight: .1 } },
  scope: { research_path: 'strategy_first', id: 'scope-a', name: '基础三大类：权益、固收、现金' },
  cmas: [{ id: 'cma-a', name: '基础三大类 · 历史统计' }, { id: 'cma-b', name: '基础三大类 · 长期情景' }],
}

test('saved plans, frozen references, isolated creation, error recovery and responsive layout', async ({ page }, info) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  let state: 'ready' | 'empty' | 'error' = 'ready'
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/strategic-allocation/policies') return state === 'error'
      ? route.fulfill({ status: 503, json: { detail: { message: '方案暂时无法读取，请重试。' } } })
      : route.fulfill({ json: { items: state === 'empty' ? [] : [plan] } })
    if (path === '/api/strategic-allocation/catalog') return route.fulfill({ json: strategicCatalog })
    if (path === `/api/tactical-allocation/baselines/${policyBaseline.id}`) return route.fulfill({ json: { ...policyBaseline, name: plan.name } })
    if (path === '/api/pit/settings') return route.fulfill({ json: { settings: { active_release_id: null }, effective: { no_pit: true, as_of: null, run_mode: 'RESEARCH', label: '研究模式' }, available_releases: [], active_release: null } })
    return route.fulfill({ status: 404, json: { detail: '离线界面验收' } })
  })
  await page.goto('/pre-investment/saa')
  const table = page.getByRole('table', { name: '已保存的 SAA 方案' })
  await expect(table).toContainText('三年期绝对收益计划')
  await expect(table).toContainText('7.72%')
  await expect(table).toContainText('长期情景')
  await expect(table.getByRole('link', { name: '三年期绝对收益计划' })).toHaveAttribute('href', '/pre-investment/objectives/new?view=mandate-1')
  await expect(page.getByText('打开现有工具')).toHaveCount(0)
  // The sidebar keeps the working tools; the main content is the saved-plan list.
  for (const path of ['asset-classes', 'auto-classification', 'allocation-lab']) {
    expect(await page.locator(`a[href="/pre-investment/saa/${path}"]`).count()).toBeGreaterThan(0)
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  expect(await page.evaluate(auditTextContrast)).toEqual([])
  await page.screenshot({ path: info.outputPath('saa-list.png'), fullPage: true })
  await table.getByRole('link', { name: '查看方案' }).click()
  await expect(page.getByRole('heading', { name: plan.name })).toBeVisible()
  await page.getByRole('link', { name: '← 返回 SAA 方案列表' }).click()
  await page.getByRole('textbox', { name: '搜索方案、目标、范围或 LTCMA' }).fill('未找到的方案')
  await expect(table).toContainText('没有找到匹配的方案')
  await page.getByRole('link', { name: '新建 SAA 方案' }).click()
  await expect(page).toHaveURL(/\/saa\/policy\?new=.+/)
  const firstNew = page.url()
  await expect(page.getByLabel('投资目标版本')).toHaveValue('')
  await page.getByLabel('投资目标版本').selectOption('mandate-1')
  await page.reload()
  await expect(page.getByLabel('投资目标版本')).toHaveValue('mandate-1')
  await page.getByRole('link', { name: '← 返回 SAA 方案列表' }).click()
  await page.getByRole('link', { name: '新建 SAA 方案' }).click()
  expect(page.url()).not.toBe(firstNew)
  await expect(page.getByLabel('投资目标版本')).toHaveValue('')
  state = 'error'
  await page.getByRole('link', { name: '← 返回 SAA 方案列表' }).click()
  await expect(page.getByRole('alert')).toContainText('方案暂时无法读取')
  await expect(page.getByRole('table')).toHaveCount(0)
  await page.screenshot({ path: info.outputPath('saa-error.png'), fullPage: true })
  state = 'empty'
  await page.getByRole('button', { name: '重试', exact: true }).click()
  await expect(page.getByText('还没有保存的 SAA 方案')).toBeVisible()
  await expect(page.getByRole('link', { name: '新建 SAA 方案' })).toBeVisible()
  expect(errors).toEqual([])
})

for (const locale of ['zh-CN', 'en-US'] as const) test(`required adoption fields and disabled explanations (${locale})`, async ({ page }, info) => {
  const text = (key: string) => (system as Record<string, Record<string, string>>)[key][locale]
  const label = (key: string) => text(`preInvestment.strategicAllocationWorkspace.${key}`)
  const errors: string[] = []
  const submitted: any[] = []
  let releaseSave: (() => void) | undefined
  page.on('pageerror', error => errors.push(error.message))
  await page.addInitScript(locale => {
    localStorage.setItem('fund-research.i18n.locale', locale)
    // Pin this isolated form test's clock before asynchronous catalog loading.
    sessionStorage.setItem('pit.view.override', JSON.stringify({ off: true }))
  }, locale)
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    if (path === '/api/strategic-allocation/catalog') return route.fulfill({ json: strategicCatalog })
    if (path === '/api/strategic-allocation/cma/cma-1') return route.fulfill({ json: cmaVersion })
    if (path === '/api/strategic-allocation/policy/frontier') return route.fulfill({ json: policyFrontierFixture(route.request().postDataJSON()) })
    if (path === '/api/strategic-allocation/policy/preview') return route.fulfill({ json: { ...policyPreview, request: route.request().postDataJSON() } })
    if (path === '/api/strategic-allocation/policies') {
      submitted.push(route.request().postDataJSON())
      if (submitted.length === 1) {
        await new Promise<void>(resolve => { releaseSave = resolve })
        return route.fulfill({ status: 503, json: { detail: { message: '保存暂时失败，请重试。' } } })
      }
      return route.fulfill({ status: 201, json: policyBaseline })
    }
    if (path === '/api/pit/settings') return route.fulfill({ json: { settings: { active_release_id: null }, effective: { no_pit: true, as_of: null, run_mode: 'RESEARCH', label: '研究模式' }, available_releases: [], active_release: null } })
    return route.fulfill({ status: 404, json: { detail: 'Offline UI fixture' } })
  })
  await page.goto('/pre-investment/saa/policy?new=required-ui&alloc=股债分类&mandate=mandate-1&cma=cma-1')
  await page.getByRole('button', { name: text('preInvestment.policyCandidates.comparePolicyCandidatesAgainstObjectives') }).click()
  await page.getByRole('button', { name: text('preInvestment.policyCandidates.reviewThisCandidate') }).click()
  const region = page.getByRole('region', { name: label('confirmPolicyAdoption') })
  const name = region.getByLabel(label('policyVersionName'))
  const reason = region.getByLabel(label('adoptionRationaleAndReviewPriorities'))
  const confirm = region.getByRole('button', { name: label('confirmThisLongTermPolicy'), exact: true })
  await expect(confirm).toBeDisabled()
  await expect(confirm).toHaveAccessibleDescription(locale === 'zh-CN' ? '采纳理由至少 5 个字，还需填写 5 个字。' : 'Enter at least 5 characters for the adoption rationale; 5 more needed.')
  for (const input of [name, reason]) {
    await expect(input).toHaveAttribute('required', '')
    expect(await input.evaluate(el => getComputedStyle(el.closest('label')!.querySelector('span')!, '::after').content)).toBe('"*"')
  }
  await reason.fill('  采用理由  ')
  await expect(confirm).toBeDisabled()
  await expect(confirm).toHaveAccessibleDescription(/1/)
  await expect(region.getByText(locale === 'zh-CN' ? '已填 4 / 2000 字' : '4 / 2000 characters')).toBeVisible()
  await name.fill('')
  await expect(confirm).toHaveAccessibleDescription(new RegExp(text('saaRequired.name')))
  await reason.fill('采用理由足')
  await expect(confirm).toBeDisabled()
  await name.fill('浏览器验收方案')
  await expect(confirm).toBeEnabled()
  await reason.fill('')
  await confirm.scrollIntoViewIfNeeded()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
  expect(await page.evaluate(auditTextContrast)).toEqual([])
  await page.screenshot({ path: info.outputPath(`required-${locale}.png`), fullPage: true })
  await reason.fill('采用理由足')
  await confirm.focus()
  await page.keyboard.press('Enter')
  await expect(confirm).toBeDisabled()
  await expect(confirm).toHaveAccessibleDescription(text('saaRequired.saving'))
  await expect(reason).toBeDisabled()
  await expect.poll(() => Boolean(releaseSave)).toBe(true)
  releaseSave!()
  await expect(page.getByRole('alert')).toContainText('保存暂时失败')
  await expect(reason).toHaveValue('采用理由足')
  await expect(confirm).toBeEnabled()
  await confirm.click()
  const saved = region.getByRole('button', { name: label('longTermPolicyConfirmed'), exact: true })
  await expect(saved).toBeDisabled()
  await expect(saved).toHaveAccessibleDescription(text('saaRequired.saved'))
  await expect(region.getByRole('button', { name: label('continueToTaaAssessTacticalDeviations') })).toBeEnabled()
  expect(submitted).toHaveLength(2)
  expect(submitted[1].reason).toBe('采用理由足')
  expect(errors).toEqual([])
})

for (const locale of ['zh-CN', 'en-US'] as const) test(`saved SAA detail overview and evidence (${locale})`, async ({ page }, info) => {
  const text = (key: string) => system[key][locale]
  const names = ['权益', '固收', '商品', '现金']
  const weights = [.592, .1976, .1104, .1]
  const sources = ['历史统计', '长期情景'].map((name, i) => ({ cma_id: `source-${i}`, content_hash: `hash-${i}`, name: `基础四类配置 · ${name}`, as_of: '2019-12-31', weight: .5 }))
  const row = (i: number) => ({ cma_id: sources[i].cma_id, cma_hash: sources[i].content_hash, name: sources[i].name,
    metrics: { ...policyPreview.candidates[0].metrics, expected_return: .0807 + i * .0003, volatility: .0872 + i * .0013 },
    risk_contributions: {}, within_limits: true, violations: [], weight: .5 })
  const record = { ...policyBaseline, name: '基础四类配置 · 长期政策', as_of: '2019-12-31',
    assets: names.map((name, i) => ({ ...policyBaseline.assets[0], id: `asset-${i}`, name, base_weight: weights[i] })),
    policy: { ...policyBaseline.policy, mode: 'compatible_all_models',
      mandate: { ...policyBaseline.policy.mandate, name: '三年期绝对收益计划', max_volatility: .095, min_cash_weight: .1 },
      assumptions: cmaVersion.definition,
      selection: { ...policyPreview.candidates[0], metrics: { ...row(0).metrics, volatility: row(1).metrics.volatility }, cross_model_results: [row(0), row(1)] },
      multi_cma: { sources }, compatibility: { objective: 'minimax_regret', regret_basis: 'certified_regret', limitations: [], anchors: [],
        joint_solver: { status: 'converged', objective_gap: 4.54e-12, phase_one_lower_bound: null } },
    },
  }
  let state: 'ready' | 'legacy' | 'error' = 'ready'
  let reads = 0
  const errors: string[] = [], writes: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.addInitScript(locale => {
    localStorage.setItem('fund-research.i18n.locale', locale)
    sessionStorage.setItem('pit.view.override', JSON.stringify({ off: true }))
  }, locale)
  await page.route('**/api/**', async route => {
    if (route.request().method() !== 'GET') writes.push(route.request().url())
    const path = new URL(route.request().url()).pathname
    if (path === `/api/tactical-allocation/baselines/${policyBaseline.id}`) {
      reads++
      return state === 'error' ? route.fulfill({ status: 503, json: { detail: '读取暂时失败，请重试。' } })
        : route.fulfill({ json: state === 'legacy' ? policyBaseline : record })
    }
    if (path === '/api/pit/settings') return route.fulfill({ json: { settings: { active_release_id: null }, effective: { no_pit: true, as_of: null, run_mode: 'RESEARCH' }, available_releases: [], active_release: null } })
    return route.fulfill({ status: 404, json: { detail: 'Offline UI fixture' } })
  })
  await page.goto(`/pre-investment/saa/policy?baseline=${record.id}`)
  await expect(page.getByRole('heading', { name: record.name, exact: true })).toBeVisible()
  const weightsPanel = page.getByRole('region', { name: text('saaDetail.weights'), exact: true })
  for (const value of ['59.20%', '19.76%', '11.04%', '10.00%', '100.00%']) await expect(weightsPanel.getByText(value, { exact: true })).toBeVisible()
  await expect(weightsPanel.getByTestId('saa-weight-chart').locator('svg')).toBeVisible()
  const outcomes = page.getByRole('region', { name: text('saaDetail.outcome'), exact: true })
  await expect(outcomes.getByText('8.07%', { exact: true })).toBeVisible()
  await expect(outcomes.getByText('8.85%', { exact: true })).toBeVisible()
  await expect(outcomes.getByText(text('saaDetail.commonMetricsHint'))).toBeVisible()
  await expect(page.getByRole('link', { name: '三年期绝对收益计划', exact: true })).toHaveAttribute('href', /view=mandate-1/)
  const technical = page.locator('summary').filter({ hasText: text('saaDetail.technicalDetails') })
  await expect(technical.locator('..')).not.toHaveAttribute('open', '')
  await expect(page.getByText(/4.54e-12/)).not.toBeVisible()
  const next = page.getByRole('link', { name: text('preInvestment.strategicAllocationWorkspace.returnToTaaResearchForThisPolicy') })
  await expect(next).toHaveAttribute('href', '/pre-investment/taa?baseline=POLICY-1')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
  expect(await page.evaluate(auditTextContrast)).toEqual([])
  await page.screenshot({ path: info.outputPath(`detail-${locale}.png`), fullPage: true })
  const readsBeforeExpansion = reads
  await technical.focus()
  await page.keyboard.press('Enter')
  await expect(page.getByText(/4.54e-12/)).toBeVisible()
  await page.keyboard.press('Enter')
  const checks = page.locator('summary').filter({ hasText: text('saaDetail.modelChecks') })
  await checks.click()
  await expect(page.getByRole('table', { name: text('multiCma.crossTitle') })).toBeVisible()
  expect(await page.evaluate(auditTextContrast)).toEqual([])
  await page.screenshot({ path: info.outputPath(`detail-expanded-${locale}.png`), fullPage: true })
  expect(reads).toBe(readsBeforeExpansion)
  state = 'legacy'
  await page.reload()
  await expect(page.getByText(text('saaDetail.noMetrics'))).toBeVisible()
  await expect(page.getByText(text('saaDetail.minimumReturn'))).toHaveCount(0)
  state = 'error'
  await page.reload()
  await expect(page.getByRole('alert')).toContainText('读取暂时失败')
  await expect(page.getByRole('heading', { name: policyBaseline.name, exact: true })).toHaveCount(0)
  state = 'ready'
  await page.getByRole('button', { name: text('preInvestment.strategicAllocationWorkspace.reloadPolicy') }).click()
  await expect(page.getByRole('heading', { name: record.name, exact: true })).toBeVisible()
  expect(writes).toEqual([])
  expect(errors).toEqual([])
})
