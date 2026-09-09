import { expect, test } from '@playwright/test'
import { factorAudit, fixtureCatalog, fixtureRun } from '../src/test/factorFixtures'
import { fixtureDatasetSummaries, fixtureFF3Dataset, fixtureReturnCatalog, fixtureReturnDataset, fixtureReturnPlan, fixtureReturnSource } from '../src/test/factorReturnFixtures'

for (const method of ['characteristic_spread', 'ff3_2x3']) test(`return construction and attribution: ${method}`, async ({ page }, testInfo) => {
  const errors: string[] = []
  const requests: Array<{ path: string; body: any }> = []
  page.on('pageerror', error => errors.push(error.message))
  const dataset = method === 'ff3_2x3' ? fixtureFF3Dataset : fixtureReturnDataset
  // Isolated browser fixtures: no production API calls or research writes.
  await page.context().route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    const post = route.request().method() === 'POST'
    const body = post ? route.request().postDataJSON() : undefined
    if (body) requests.push({ path, body })
    let value: unknown = { items: [] }
    if (path === '/api/factor-research/catalog') value = fixtureCatalog
    else if (path.endsWith('/return-catalog')) value = fixtureReturnCatalog
    else if (path === '/api/factor-research/runs') value = { items: [fixtureRun] }
    else if (path.endsWith('/runs/factor-run-test')) value = fixtureRun
    else if (path.endsWith('/return-sources')) value = { items: [fixtureReturnSource] }
    else if (path.endsWith('/return-plans') && post) value = { ...fixtureReturnPlan, ...body }
    else if (path.endsWith('/return-plans/factor-return-plan-test/runs')) value = dataset
    else if (path === '/api/factor-research/datasets') value = { items: fixtureDatasetSummaries }
    else if (path === `/api/factor-research/return-datasets/${dataset.id}`) value = dataset
    else if (path.endsWith('/export')) return route.fulfill({ contentType: 'text/csv', headers: { 'Content-Disposition': 'attachment; filename="factor-returns.csv"' }, body: 'date,SPREAD\n2024-07-01,0\n' })
    else if (path.endsWith('/attributions') && post) value = { id: 'factor-attribution-test', name: body.name, request: body, results: [], warnings: ['离线测试夹具'], execution: factorAudit }
    return route.fulfill({ json: value })
  })
  await page.goto('/settings/factor-research?run=factor-run-test')
  await expect(page.getByLabel('因子检验结果')).toBeVisible()
  await page.getByRole('button', { name: '构建因子收益率', exact: true }).click()
  await expect(page.getByRole('tab', { name: '因子收益率', exact: true })).toHaveAttribute('aria-selected', 'true')
  await expect(page.getByLabel('来源特征运行', { exact: true })).toHaveValue(fixtureRun.id)
  if (method === 'ff3_2x3') {
    await page.getByLabel('收益率构造算法').selectOption('ff3_2x3')
    await expect(page.getByText(/不自动取本地股票构建/)).toBeVisible()
    await page.getByLabel('FF3 原始面板', { exact: true }).selectOption(fixtureReturnSource.id)
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.screenshot({ path: testInfo.outputPath('return-construction.png'), fullPage: true })
  await page.getByRole('button', { name: '保存并生成收益率' }).click()
  await expect(page.getByLabel('因子收益率结果')).toBeVisible()
  expect(requests[0].body.method).toBe(method)
  expect(requests[1].body).toEqual({ revision: 1 })
  await expect(page.getByLabel('因子收益序列')).toContainText(method === 'ff3_2x3' ? 'SMB' : 'SPREAD')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  const downloaded = page.waitForEvent('download')
  await page.getByRole('button', { name: '导出 CSV' }).click()
  const file = await downloaded
  expect(file.suggestedFilename()).toBe(dataset.id + '.csv')
  const stream = await file.createReadStream()
  const chunks: Buffer[] = []
  if (stream) for await (const chunk of stream) chunks.push(Buffer.from(chunk))
  expect(Buffer.concat(chunks).toString()).toBe('date,SPREAD\n2024-07-01,0\n')
  await page.getByRole('button', { name: '用于收益归因' }).click()
  await expect(page.getByLabel('归因模型')).toHaveValue(method === 'ff3_2x3' ? 'ff3' : 'factor_regression')
  await page.getByLabel('归因产品代码', { exact: true }).fill('000001.OF')
  await page.getByRole('button', { name: '运行归因研究' }).click()
  await expect(page.getByLabel('收益归因结果')).toBeVisible()
  expect(requests[requests.length - 1].body.dataset_id).toBe(dataset.id)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  await page.getByRole('tab', { name: '收益率数据集', exact: true }).click()
  await expect(page.getByLabel('因子收益率结果')).toBeVisible()
  await page.getByLabel('导入因子收益 JSON', { exact: true }).setInputFiles({ name: 'broken.json', mimeType: 'application/json', buffer: Buffer.from('{') })
  await expect(page.getByRole('alert')).toContainText('不是有效的 JSON')
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy()
  expect(errors).toEqual([])
})
