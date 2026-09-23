import { test, expect } from '@playwright/test'
import { spawn, type ChildProcess } from 'node:child_process'
import { createServer } from 'node:net'
import path from 'node:path'
import katex from 'katex'
import { auditTextContrast } from './helpers/contrast'

// Real production routes in a disposable workspace, not mocked metadata/DAGs.
let backend: ChildProcess | undefined
let apiRoot = ''
let logs = ''

async function unusedPort(): Promise<number> {
  const socket = createServer()
  await new Promise<void>((resolve, reject) => {
    socket.once('error', reject)
    socket.listen(0, '127.0.0.1', resolve)
  })
  const address = socket.address()
  if (!address || typeof address === 'string') throw new Error('No test port available')
  await new Promise<void>((resolve) => socket.close(() => resolve()))
  return address.port
}

test.beforeAll(async ({ request }) => {
  test.setTimeout(120_000)
  const port = await unusedPort()
  apiRoot = `http://127.0.0.1:${port}`
  const python = process.env.INDICATOR_TEST_PYTHON || 'python3'
  backend = spawn(python, ['-m', 'uvicorn', 'indicator_formula_app:app', '--app-dir', 'tests', '--host', '127.0.0.1', '--port', String(port)], {
    cwd: path.resolve(process.cwd(), '../backend'),
    env: { ...process.env, ALL_PROXY: '', HTTP_PROXY: '', HTTPS_PROXY: '', NO_PROXY: '127.0.0.1,localhost' },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  backend.stdout?.on('data', (data) => { logs = (logs + String(data)).slice(-8000) })
  backend.stderr?.on('data', (data) => { logs = (logs + String(data)).slice(-8000) })
  backend.on('error', (error) => { logs += error.message })
  await expect.poll(async () => {
    if (backend?.exitCode !== null && backend?.exitCode !== undefined) throw new Error(logs)
    try { return (await request.get(`${apiRoot}/ready`, { timeout: 1000 })).ok() } catch { return false }
  }, { timeout: 100_000, message: 'Isolated formula API did not start' }).toBe(true)
})

test.afterAll(async () => {
  if (!backend || backend.exitCode !== null) return
  const stopped = new Promise<void>((resolve) => backend?.once('exit', () => resolve()))
  backend.kill('SIGTERM')
  await Promise.race([stopped, new Promise<void>((resolve) => setTimeout(resolve, 10_000))])
  if (backend.exitCode === null) backend.kill('SIGKILL')
})

test('复权价格与嵌套时序算子：通过校验的公式及节点均可排版', async ({ page, request }, testInfo) => {
  test.setTimeout(120_000)
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('dialog', dialog => dialog.accept())
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url())
    if (url.pathname.startsWith('/api/custom-indicators')) {
      await route.fulfill({ response: await route.fetch({ url: `${apiRoot}${url.pathname}${url.search}`, timeout: 90_000 }) })
    } else {
      await route.fulfill({ status: 503, json: { detail: 'Isolated fixture' } })
    }
  })
  await page.goto('/settings/indicators-models')
  await expect(page.getByRole('button', { name: /^累计收益率/ })).toBeVisible()
  await page.getByRole('button', { name: '新建指标', exact: true }).click()
  await page.getByRole('tab', { name: '高级公式模式' }).click()
  const expression = 'last(mean(rolling_window(multiply(sign(difference(multiply(0.5,add(adjusted_high,adjusted_low)),1)),multiply(divide(subtract(lag(adjusted_high,1),lag(adjusted_low,1)),multiply(0.5,add(lag(adjusted_high,1),lag(adjusted_low,1)))),divide_or_default(lag(volume,1),lag(mean(rolling_window(volume,20)),1),0))),20)))'
  await page.locator('#indicator-expression').fill(expression)
  const checked = page.waitForResponse(response => response.url().endsWith('/validate'))
  await page.getByRole('button', { name: '解析并校验公式', exact: true }).click()
  const validation = await (await checked).json()
  expect(validation.valid, JSON.stringify(validation.diagnostics)).toBe(true)
  expect(validation.math_notation_version).toBe('1.5.1')
  const assertRenderable = (result: typeof validation) => {
    for (const latex of [result.display_latex, ...result.dag.nodes.map((node: { latex_fragment: string }) => node.latex_fragment)]) {
      expect(latex).toBeTruthy()
      expect(() => katex.renderToString(latex, { throwOnError: true, displayMode: true })).not.toThrow()
    }
  }
  assertRenderable(validation)
  // The same renderer handles indexed variables and already-indexed operands.
  for (const nested of ['last(difference(adjusted_high,1))', 'last(lag(difference(lag(adjusted_low,1),1),1))']) {
    const response = await request.post(`${apiRoot}/api/custom-indicators/infer`, { data: { expression: nested, context: 'single_product' } })
    expect(response.ok(), await response.text()).toBe(true)
    assertRenderable(await response.json())
  }
  const preview = page.getByTestId('formula-preview')
  await expect(preview.locator('.katex')).toBeVisible()
  await expect(preview).not.toContainText('公式排版不可用')
  await expect(preview.locator('.katex-error')).toHaveCount(0)
  await preview.scrollIntoViewIfNeeded()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true)
  expect((await page.evaluate(auditTextContrast)).filter(item => item.text.includes('数学排版预览'))).toEqual([])
  await preview.screenshot({ path: testInfo.outputPath('adjusted-price-latex-preview.png') })
  expect(errors).toEqual([])
})

test('滚动夏普：真实接口下构建往返、十五日编辑、样本标准差和公式预览', async ({ page, request }, testInfo) => {
  test.setTimeout(120_000)
  // The built-in now has a configurable N-day window; this case edits a fixed-window draft.
  const sourceResponse = await request.get(`${apiRoot}/api/custom-indicators/builtin-annualized-sharpe-v2`)
  expect(sourceResponse.ok()).toBe(true)
  const scalar = await sourceResponse.json()
  const derived = await request.post(`${apiRoot}/api/custom-indicators/derive-rolling-series`, { data: {
    indicator_id: scalar.id, indicator_revision: scalar.revision, window_observations: 5, name: '5 日滚动年化夏普比率',
  } })
  expect(derived.ok(), await derived.text()).toBe(true)
  const saved = await request.post(`${apiRoot}/api/custom-indicators`, { data: (await derived.json()).definition })
  expect(saved.ok(), await saved.text()).toBe(true)
  const consoleErrors: string[] = []
  page.on('pageerror', (error) => consoleErrors.push(error.message))
  page.on('dialog', dialog => dialog.accept())
  await page.route('**/api/custom-indicators**', async (route) => {
    const url = new URL(route.request().url())
    const response = await route.fetch({ url: `${apiRoot}${url.pathname}${url.search}`, timeout: 90_000 })
    await route.fulfill({ response })
  })
  await page.goto('/settings/indicators-models')
  await page.getByRole('button', { name: /5 日滚动年化夏普比率/ }).click()
  await page.getByRole('tab', { name: '高级公式模式' }).click()
  const source = page.getByRole('textbox', { name: /公式源码/ })
  const original = await source.inputValue()
  expect(original).toContain(String.raw`\operatorname{rolling_apply}`)
  expect(original).toContain(String.raw`\operatorname{std}\left(\mathbf{r},1\right)`)
  expect(original).toContain(String.raw`\frac{`)
  expect(original).toContain('r_f')
  expect(original).not.toContain('risk_free_rate_per_observation')

  const validate = async () => {
    const responsePromise = page.waitForResponse((r) => r.url().endsWith('/api/custom-indicators/validate') && r.request().method() === 'POST')
    await page.getByRole('button', { name: '解析并校验全部通道' }).click()
    const response = await responsePromise
    expect(response.status()).toBe(200)
    const body = await response.json()
    expect(body.valid, JSON.stringify(body.diagnostics)).toBe(true)
    await expect(page.getByTestId('formula-preview').locator('.katex')).toBeVisible()
    return body
  }
  await validate()

  for (let pass = 0; pass < 2; pass++) {
    await page.getByRole('button', { name: '浏览公式构建资源' }).click()
    const drawer = page.getByRole('dialog', { name: /编辑.*计算逻辑/ })
    await expect(drawer.getByLabel('自由度修正 有限常量')).toHaveValue('1')
    await drawer.getByRole('button', { name: '应用逻辑修改' }).click()
    await expect(drawer).toBeHidden()
    await expect(source).toHaveValue(original)
    await expect(page.getByText('来源已锁定', { exact: true })).toBeVisible()
  }

  await page.getByRole('button', { name: '浏览公式构建资源' }).click()
  const drawer = page.getByRole('dialog', { name: /编辑.*计算逻辑/ })
  const windows = drawer.getByLabel('窗口期数 有限常量')
  await expect(windows).toHaveCount(1)
  await windows.fill('15')
  await drawer.getByRole('button', { name: '应用逻辑修改' }).click()
  await expect(drawer).toBeHidden()
  expect(await source.inputValue()).toContain(String.raw`\operatorname{rolling_apply}`)
  expect(await source.inputValue()).toContain(String.raw`\operatorname{std}\left(\mathbf{r},1\right)`)
  expect(await source.inputValue()).toContain(String.raw`\frac{`)
  const checked = await validate()
  expect(checked.output_inferences.value.display_latex).toContain('15')
  expect(checked.output_inferences.value.python_expression).toContain('rolling_apply(')
  expect(checked.output_inferences.value.python_expression).toContain('std(returns, 1)')
  expect(checked.output_inferences.value.python_expression).toMatch(/, 15(?:\.0)?, observation_dates, annual_risk_free_rate_decimal\)$/)
  // Manual LaTeX editing follows the same validation path as the builder.
  const edited = (await source.inputValue()).replaceAll(',15', ',10').replace('r_f', 'r_{f}')
  await source.fill(edited)
  const manual = await validate()
  expect(manual.output_inferences.value.python_expression).toContain('std(returns, 1)')
  expect(manual.output_inferences.value.python_expression).toMatch(/, 10(?:\.0)?, observation_dates, annual_risk_free_rate_decimal\)$/)
  expect(manual.output_inferences.value.dependencies).toContain('risk_free_rate_per_observation')
  await expect(page.getByText('已脱离来源', { exact: true })).toBeVisible()
  expect(consoleErrors).toEqual([])
  await page.getByTestId('formula-preview').screenshot({ path: testInfo.outputPath('sharpe-latex-preview.png') })
})

test('标量夏普：载入和构建使用 LaTeX，滚动派生重新校验编辑源码', async ({ page }) => {
  test.setTimeout(120_000)
  await page.route('**/api/custom-indicators**', async (route) => {
    const url = new URL(route.request().url())
    const response = await route.fetch({ url: `${apiRoot}${url.pathname}${url.search}`, timeout: 90_000 })
    await route.fulfill({ response })
  })
  await page.goto('/settings/indicators-models')
  await page.getByLabel('指标结果类型').selectOption('scalar')
  await page.getByRole('button', { name: /^年化夏普比率/ }).click()
  await page.getByRole('tab', { name: '高级公式模式' }).click()
  const source = page.getByRole('textbox', { name: /公式源码/ })
  const original = await source.inputValue()
  expect(original).toContain(String.raw`\frac{`)
  expect(original).toContain(String.raw`\sqrt{p_{\mathrm{year}}}`)
  expect(original).toContain('r_f')
  await expect(page.getByTestId('formula-preview').locator('.katex')).toBeVisible()
  await page.getByRole('button', { name: '浏览公式构建资源' }).click()
  const drawer = page.getByRole('dialog', { name: /编辑.*计算逻辑/ })
  await drawer.getByRole('button', { name: '应用逻辑修改' }).click()
  await expect(drawer).toBeHidden()
  await expect(source).toHaveValue(original)

  await page.getByRole('button', { name: '生成滚动时序指标' }).click()
  const derivedValidation = page.waitForResponse((r) => r.url().endsWith('/api/custom-indicators/validate') && r.request().method() === 'POST')
  page.once('dialog', (dialog) => dialog.accept())
  await page.getByRole('button', { name: '生成滚动公式' }).click()
  const response = await derivedValidation
  const checked = await response.json()
  expect(checked.valid, JSON.stringify(checked.diagnostics)).toBe(true)
  expect(response.request().postDataJSON().series_outputs[0].expression).toContain(String.raw`\frac{`)
  await page.getByRole('tab', { name: '高级公式模式' }).click()
  expect(await source.inputValue()).toContain(String.raw`\operatorname{rolling_apply}`)
  expect(await source.inputValue()).toContain(String.raw`\operatorname{std}\left(\mathbf{r},1\right)`)
  await expect(page.getByTestId('formula-preview').locator('.katex')).toBeVisible()
})
