import { test, expect, type Page } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
const root = fileURLToPath(new URL('../../', import.meta.url))
const python = process.env.TEST_PYTHON || '/Users/chenjunming/Desktop/myenv_312/bin/python3.12'
const catalog = JSON.parse(execFileSync(python, ['-c', 'import json,tempfile; from pathlib import Path; from backend.data_sources.store import SourceStore; from backend.data_sources.service import catalog; t=tempfile.TemporaryDirectory(); s=SourceStore(Path(t.name)); print(json.dumps(catalog(s),ensure_ascii=False)); t.cleanup()'], {cwd:root,encoding:'utf8'}))
function initial() {
  const interfaceRecord = catalog.interfaces.find((i: {config:{id:string}})=>i.config.id==='akshare.fund_nav')
  const common={mode:'inherit',params:{},target_tables:[],include_history:false,allow_empty:false,inputs:[],after:[]}
  return {name:'跨模块公共图验证',description:'offline',max_runtime_seconds:3600,graph_version:1,
    canvas:{version:1,positions:{d:{x:0,y:0},m:{x:350,y:0}}},
    steps:[{...common,id:'d',kind:'download',name:'基金净值下载',source_id:'akshare',interface_id:'akshare.fund_nav',interface_revision:interfaceRecord.revision,params:{symbol:'000001',start_date:'20240101',end_date:'20240110'}},
      {...common,id:'m',kind:'map',name:'映射基金净值'}]}
}
async function setup(page:Page,readOnly=false) {
  let saved=initial(); let revision=1; let writes=0
  const errors:string[]=[]; page.on('pageerror',error=>errors.push(error.message))
  await page.route('**/api/**',route=>{
    const path=new URL(route.request().url()).pathname
    if(path==='/api/data-sources/catalog')return route.fulfill({json:{...catalog,editing_enabled:!readOnly}})
    if(path==='/api/data-sources/etl/runs')return route.fulfill({json:[]})
    if(path==='/api/data-sources/etl/workflows')return route.fulfill({json:[{id:'canvas-fixture',revision,definition:saved,updated_at:'2026-09-06'}]})
    if(path.endsWith('/workflows/canvas-fixture') && route.request().method()==='PUT'){
      saved=route.request().postDataJSON().definition; revision++; writes++
      return route.fulfill({json:{id:'canvas-fixture',revision,definition:saved,updated_at:'2026-09-06'}})
    }
    if(path==='/api/data-sources/etl/validate'){
      const result=execFileSync(python,['-c','import json,sys,tempfile; from pathlib import Path; from backend.data_sources.store import SourceStore; from backend.data_sources.etl_service import validate; t=tempfile.TemporaryDirectory(); s=SourceStore(Path(t.name)); s.seed(); print(json.dumps(validate(s,json.load(sys.stdin)),ensure_ascii=False)); t.cleanup()'],{cwd:root,encoding:'utf8',input:JSON.stringify(route.request().postDataJSON().definition)})
      return route.fulfill({json:JSON.parse(result)})
    }
    return route.fulfill({status:404,json:{detail:{message:'offline'}}})
  })
  await open(page)
  return {saved:()=>saved,writes:()=>writes,errors}
}
async function open(page:Page){
  await page.goto('/settings/data-sources'); await page.getByRole('button',{name:'ETL 任务编排',exact:true}).click()
  await page.getByLabel('已保存流程',{exact:true}).selectOption('canvas-fixture')
  await expect(page.getByTestId('etl-graph-canvas')).toContainText('2 节点')
}
async function inspect(page:Page,name:string){
  const summary=page.getByText('执行顺序 · 2 个节点',{exact:true})
  const details=summary.locator('..')
  if(await details.getAttribute('open') === null)await summary.click()
  await details.getByRole('button',{name,exact:true}).click()
}

test('same canvas connects, deletes, undoes and persists graph layout',async({page},info)=>{
  const state=await setup(page)
  const desktop=(page.viewportSize()?.width??0)>=768
  if(desktop){
    const flow=page.getByTestId('etl-graph-desktop-flow'); await flow.scrollIntoViewIfNeeded()
    const source=flow.locator('.react-flow__node[data-id="d"] .react-flow__handle.source[data-handleid="data"]')
    const target=flow.locator('.react-flow__node[data-id="m"] .react-flow__handle.target[data-handleid="data"]')
    await expect(source).toBeVisible(); await expect(target).toBeVisible()
    await source.dragTo(target)
  }else{
    await inspect(page,'2. 映射基金净值')
    await page.getByLabel('添加原始批次',{exact:true}).selectOption('d')
    await page.getByRole('button',{name:'收起检查器'}).click()
  }
  await expect(page.getByTestId('etl-graph-canvas')).toContainText('1 连线')
  await inspect(page,'2. 映射基金净值')
  await page.getByRole('button',{name:'断开原始批次：基金净值下载'}).click()
  await expect(page.getByTestId('etl-graph-canvas')).toContainText('0 连线')
  await page.getByRole('button',{name:'撤销',exact:true}).click()
  await expect(page.getByTestId('etl-graph-canvas')).toContainText('1 连线')
  await page.getByRole('button',{name:'收起检查器'}).click()
  await inspect(page,'1. 基金净值下载')
  // Downstream cannot become a control prerequisite of the upstream.
  await expect(page.getByLabel('添加等待完成',{exact:true}).locator('option[value="m"]')).toHaveAttribute('disabled', '')
  await page.getByRole('button',{name:'收起检查器'}).click()
  if(desktop){
    const node=page.getByTestId('etl-graph-desktop-flow').locator('.react-flow__node[data-id="d"]')
    await node.scrollIntoViewIfNeeded(); const rect=await node.boundingBox(); expect(rect).not.toBeNull()
    await page.mouse.move(rect!.x+60,rect!.y+15); await page.mouse.down(); await page.mouse.move(rect!.x+120,rect!.y+95,{steps:8}); await page.mouse.up()
  }else{
    await page.getByTestId('etl-graph-mobile-list').getByRole('button',{name:'右移基金净值下载'}).click()
  }
  await page.getByRole('button',{name:'校验流程',exact:true}).click()
  await expect(page.getByText('流程校验通过；不代表已下载或已发布。')).toBeVisible()
  await page.getByRole('button',{name:'保存流程',exact:true}).click()
  await expect.poll(()=>state.writes()).toBe(1)
  expect(state.saved().steps[1].inputs).toEqual(['d'])
  expect(state.saved().canvas.positions.d).not.toEqual({x:0,y:0})
  const layout=state.saved().canvas
  await open(page)
  expect(state.saved().canvas).toEqual(layout)
  await expect(page.getByTestId('etl-graph-canvas')).toContainText('1 连线')
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1)).toBe(true)
  await page.getByTestId(desktop?'etl-graph-desktop-flow':'etl-graph-mobile-list').scrollIntoViewIfNeeded()
  await page.screenshot({path:info.outputPath('etl-shared-canvas.png')})
  expect(state.errors).toEqual([])
})

test('read-only canvas allows inspection but never editing or deleting',async({page})=>{
  const state=await setup(page,true)
  await expect(page.getByRole('button',{name:'自动布局',exact:true})).toBeDisabled()
  await inspect(page,'1. 基金净值下载')
  await expect(page.getByLabel('步骤名称',{exact:true})).toBeDisabled()
  await expect(page.getByRole('button',{name:'删除此节点'})).toBeDisabled()
  await expect(page.getByRole('button',{name:'保存流程',exact:true})).toBeDisabled()
  expect(state.writes()).toBe(0); expect(state.errors).toEqual([])
})
