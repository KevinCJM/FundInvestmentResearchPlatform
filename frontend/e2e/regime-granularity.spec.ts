import { test, expect } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { resolve } from 'node:path'

// Offline UI acceptance uses the real domain transformer and registry, not a
// second JavaScript implementation. It does not start the app or read markets.
const root = resolve(process.cwd(), '..')
const python = process.env.PYTHON || 'python3'
const domainScript = String.raw`
import copy, json, sys
from historical_regimes.v2_registry import node_catalog
from historical_regimes.v2_templates import MARKET_STATES
from historical_regimes.composite_expansion import expand_composite
from custom_indicators.errors import ValidationError
request = json.load(sys.stdin)
if request['action'] == 'fixtures':
    definition = {'schema_version':'2.0','name':'颗粒度操作验收','description':'离线固定样本',
      'graph':{'nodes':[
        {'id':'data','type':'source.inline','label':'固定研究序列','inputs':{},'parameters':{'frequency':'daily','rows':[
          {'observation_date':f'2020-01-{i+1:02d}','available_at':f'2020-01-{i+1:02d}','value':value}
          for i,value in enumerate([0.,.001,.002,-.001,-.002,.004,0.,.001])]}},
        {'id':'algorithm','type':'model.threshold','label':'原三状态算法','parameters':{'upper':.001,'lower':-.001},'inputs':{'value':{'node_id':'data','port':'value'}}}],
        'outputs':{'state':{'node_id':'algorithm','port':'state'}},'exposed_node_ids':['algorithm']},
      'states':copy.deepcopy(MARKET_STATES),'evaluation_targets':[],'validation':{},'usage_intent':'research_display'}
    print(json.dumps({'definition':definition,'catalog':node_catalog()},ensure_ascii=False))
elif request['action'] == 'expand':
    try:
        print(json.dumps({'status':200,'value':expand_composite(request['definition'],request['node_id'],request['mode'])},ensure_ascii=False))
    except ValidationError as error:
        print(json.dumps({'status':error.status_code,'value':{'detail':{'message':error.message,'code':error.code}}},ensure_ascii=False))
`
function domain(payload: unknown) {
  return JSON.parse(execFileSync(python, ['-c', domainScript], {
    cwd: root, env: { ...process.env, PYTHONPATH: resolve(root, 'backend') },
    input: JSON.stringify(payload), encoding: 'utf8', timeout: 60_000, maxBuffer: 4 * 1024 * 1024,
  }))
}

let fixtures: ReturnType<typeof domain>
test.beforeAll(() => { fixtures = domain({ action: 'fixtures' }) })

test('组合展开保持实际连线、业务状态下拉与单次撤销', async ({ page }, testInfo) => {
  test.setTimeout(120_000)
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  const expansions: Array<{ input: any; output: any }> = []
  const inferenceBodies: any[] = []
  let writes = 0
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname
    const body = route.request().method() === 'POST' ? route.request().postDataJSON() : null
    let value: unknown
    if (path.endsWith('/nodes')) value = fixtures.catalog
    else if (path.endsWith('/templates/v2')) value = { items: [{ id: 'granularity-e2e', name: '颗粒度验收模板', default_mode: 'realtime' }] }
    else if (path.endsWith('/templates/granularity-e2e/instantiate')) value = { definition: fixtures.definition }
    else if (path.endsWith('/authoring/expand')) {
      const result = domain({ action: 'expand', ...body })
      expansions.push({ input: body, output: result.value })
      return route.fulfill({ status: result.status, json: result.value })
    } else if (path.endsWith('/authoring/resolve')) value = { valid: true, source: '', diagnostics: [], compile_status: 'not_requested', display_latex: {} }
    else if (path.endsWith('/infer')) {
      inferenceBodies.push(body.definition)
      value = { valid: true, errors: [], warnings: [], inferred: { nodes: {} } }
    } else if (path.endsWith('/v2/definitions') || path.endsWith('/v2/graph-assets') || path.endsWith('/v2/experiments')) {
      if (body) writes += 1
      value = { items: [] }
    } else if (path.endsWith('/research-series/catalog')) value = { items: [], total: 0 }
    else return route.fulfill({ status: 404, json: { detail: 'Offline granularity acceptance' } })
    return route.fulfill({ json: value })
  })
  await page.goto('/settings/scenario-algorithms/workbench?template=granularity-e2e')
  await expect(page.getByLabel('研究名称')).toHaveValue('颗粒度操作验收')
  await page.getByRole('tab', { name: '构建向导', exact: true }).click()
  const guide = page.getByLabel('情景公式构建向导', { exact: true })
  await guide.getByRole('button', { name: /原三状态算法/ }).click()
  const builder = page.getByRole('dialog', { name: '公式构建向导', exact: true })
  await expect(builder.getByText('组合模板', { exact: true })).toBeVisible()
  await builder.getByRole('button', { name: '展开为计算步骤' }).click()
  await expect(builder.getByRole('combobox', { name: '条件满足时的状态', exact: true })).toBeVisible({ timeout: 60_000 })
  expect(expansions).toHaveLength(1)
  expect(expansions[0].output.inserted_node_ids.length).toBeGreaterThan(3)
  expect(expansions[0].output.definition.graph.nodes.some((node: any) => node.type === 'model.threshold')).toBe(false)
  const originalHash = JSON.stringify(fixtures.definition)
  await expect.poll(() => inferenceBodies.at(-1)?.graph.outputs.state.node_id).toBe(expansions[0].output.primary_node_id)
  const chooseState = builder.getByRole('combobox', { name: '条件满足时的状态', exact: true })
  await expect(chooseState.locator('option')).toContainText(['请选择', '未分类', '牛市', '震荡', '熊市'])
  await page.screenshot({ path: testInfo.outputPath('expanded-steps.png'), fullPage: true })
  await page.getByRole('button', { name: '关闭公式构建向导' }).click()
  await page.getByRole('button', { name: '撤销', exact: true }).click()
  await expect.poll(() => inferenceBodies.at(-1)?.graph.nodes.map((node: any) => node.id)).toEqual(['data', 'algorithm'])
  expect(JSON.stringify(fixtures.definition)).toBe(originalHash)

  await page.getByRole('tab', { name: '画布构建', exact: true }).click()
  await page.getByRole('button', { name: '添加节点', exact: true }).click()
  const library = page.getByRole('dialog', { name: '添加节点', exact: true })
  await library.getByLabel('节点类型').selectOption('granularity:composite')
  await library.getByLabel('搜索计算节点').fill('三状态阈值')
  await library.getByRole('button', { name: '添加并展开三状态阈值', exact: true }).click()
  await expect.poll(() => expansions.length, { timeout: 60_000 }).toBe(2)
  await expect(page.getByLabel('节点动态检查器')).toBeVisible()
  await expect.poll(() => inferenceBodies.at(-1)?.graph.nodes.length).toBeGreaterThan(2)
  expect(expansions[1].output.definition.graph.outputs).toEqual(fixtures.definition.graph.outputs)
  await page.keyboard.press('Escape')
  await page.getByRole('button', { name: '撤销', exact: true }).click()
  await expect.poll(() => inferenceBodies.at(-1)?.graph.nodes.length).toBe(2)
  expect(writes).toBe(0)
  expect(errors).toEqual([])
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy()
})
