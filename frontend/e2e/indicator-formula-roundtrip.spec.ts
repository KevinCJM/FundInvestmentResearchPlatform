import { test, expect } from '@playwright/test'
import { spawn, type ChildProcess } from 'node:child_process'
import { createServer } from 'node:net'
import path from 'node:path'

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

test('滚动夏普：真实接口下构建往返、十五日编辑、样本标准差和公式预览', async ({ page }, testInfo) => {
  test.setTimeout(120_000)
  const consoleErrors: string[] = []
  page.on('pageerror', (error) => consoleErrors.push(error.message))
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
