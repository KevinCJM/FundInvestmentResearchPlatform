import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import DataDownloadWorkspace from './DataDownloadWorkspace'
import * as sourceApi from '../../services/dataSources'
import * as etlApi from '../../services/etl'
import type { SourceCatalog } from '../../services/dataSources'
import { etlGraphSchemas } from '../../test/etlGraphFixtures'
vi.mock('../../services/dataSources', async importOriginal => ({ ...await importOriginal<typeof import('../../services/dataSources')>(), fetchSourceCatalog: vi.fn() }))
vi.mock('../../services/etl', async importOriginal => ({ ...await importOriginal<typeof import('../../services/etl')>(), listEtlWorkflows: vi.fn(), listEtlRuns: vi.fn(), validateEtl: vi.fn(), runEtl: vi.fn(), saveEtlWorkflow: vi.fn(), recoverEtl: vi.fn(), getEtlRun: vi.fn(), cancelEtl: vi.fn() }))

const record = (source: string, api: string, name: string, table: string) => ({ revision: 1, builtin: true, updated_at:'', request_fields:[{name:'symbol',label:'产品代码（symbol）',data_type:'text'},{name:'start_date',label:'开始日期',data_type:'date'},{name:'end_date',label:'结束日期',data_type:'date'}], validation:{ready:true,valid:true}, config:{ id:`${source}.${api}`, source_id:source, name, api_name:api, enabled:true, params:{symbol:'510300'}, source_fields:[{name:'symbol',description:'代码'}], start_param:'start_date', end_param:'end_date', mappings:[{enabled:true,target_table:table}] } })
const catalog = { graph_schemas: etlGraphSchemas, editing_enabled:true, sources:[{config:{id:'tushare',name:'Tushare',enabled:true,transport:'tushare',auth_mode:'none'},credential_configured:true,credential_required:true},{config:{id:'akshare',name:'AKShare',enabled:true,transport:'akshare',auth_mode:'none'}}], interfaces:[record('tushare','fund_daily','ETF 日行情','market.quote_daily'),record('akshare','etf_daily','AK ETF 行情','market.quote_daily'),record('akshare','fund_nav','公募基金净值','market.nav_daily')], targets:{ categories:[{category_id:'market',label:'行情与净值'}], tables:[{table_id:'market.quote_daily',label:'日行情',category_id:'market',source_mappable:true},{table_id:'market.nav_daily',label:'净值',category_id:'market',source_mappable:true}] } } as unknown as SourceCatalog
const success = {run_id:'abc',name:'运行',status:'RUNNING',created_at:'2026-01-01',updated_at:'2026-01-01',steps:[],attempt:1,published:false} as etlApi.EtlRun
const animationFrames = new Set<number>()

beforeEach(() => {
  vi.clearAllMocks()
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} })
  // These workflow tests use an unzoomed canvas; jsdom has no DOMMatrix API.
  vi.stubGlobal('DOMMatrixReadOnly', class { readonly m22 = 1 })
  const schedule = window.requestAnimationFrame.bind(window)
  vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => {
    const id = schedule(time => { animationFrames.delete(id); callback(time) })
    animationFrames.add(id)
    return id
  })
  vi.mocked(sourceApi.fetchSourceCatalog).mockResolvedValue(catalog)
  vi.mocked(etlApi.listEtlWorkflows).mockResolvedValue([])
  vi.mocked(etlApi.listEtlRuns).mockResolvedValue([])
  vi.mocked(etlApi.validateEtl).mockResolvedValue({valid:true, errors:[], steps:[]})
  vi.mocked(etlApi.runEtl).mockResolvedValue(success)
  vi.spyOn(window,'confirm').mockReturnValue(true)
})
afterEach(() => {
  cleanup()
  animationFrames.forEach(id => window.cancelAnimationFrame(id))
  animationFrames.clear()
  vi.restoreAllMocks(); vi.unstubAllGlobals()
})
const load = async () => {render(<MemoryRouter><DataDownloadWorkspace /></MemoryRouter>); await screen.findByRole('heading',{name:'选择数据源与下载内容'}); await waitFor(() => expect(etlApi.listEtlRuns).toHaveBeenCalled())}

describe('多源下载与 ETL', () => {
  it('自动模式先置灰，校验失败仍置灰，通过后才能运行', async () => {
    const definition = etlApi.quickPlan([catalog.interfaces[0]], 'incremental', {}, false)
    vi.mocked(etlApi.listEtlWorkflows).mockResolvedValue([{ id: 'flow', revision: 1, definition, updated_at: '' }])
    await load()
    fireEvent.click(screen.getByRole('button', { name: 'ETL 任务编排' }))
    fireEvent.change(screen.getByLabelText('已保存流程'), { target: { value: 'flow' } })
    // Let the loaded graph complete its browser frame before checking workflow actions.
    await act(async () => { await new Promise<void>(resolve => window.requestAnimationFrame(() => resolve())) })
    fireEvent.change(screen.getByLabelText('本次运行模式'), { target: { value: 'auto_incremental' } })
    const start = screen.getByRole('button', { name: '确认并运行流程' })
    expect(start).toBeDisabled()
    vi.mocked(etlApi.validateEtl).mockResolvedValue({ valid: false, errors: [{ code: 'EMPTY', message: '没有基线' }], steps: [] })
    fireEvent.click(screen.getByRole('button', { name: '分析自动增量区间' }))
    await screen.findByText('没有基线')
    expect(start).toBeDisabled()
    vi.mocked(etlApi.validateEtl).mockResolvedValue({ valid: true, errors: [], steps: [], auto_plan: {
      plan_id: 'ok', snapshot: 'active', cutoff_date: '2026-09-09', lookback_trade_days: 5, ready: true, errors: [], steps: [] } })
    fireEvent.click(screen.getByRole('button', { name: '分析自动增量区间' }))
    await waitFor(() => expect(start).toBeEnabled())
    expect(etlApi.runEtl).not.toHaveBeenCalled()
  })
  it('校验返回期间用户修改配置，旧结果不能启用或启动新流程', async () => {
    await load(); fireEvent.click(screen.getByLabelText('ETF 日行情'))
    let finish!: (value: etlApi.EtlValidation) => void
    vi.mocked(etlApi.validateEtl).mockReturnValue(new Promise(resolve => { finish = resolve }))
    fireEvent.click(screen.getByRole('button', { name: '确认并开始下载' }))
    await waitFor(() => expect(etlApi.validateEtl).toHaveBeenCalled())
    fireEvent.change(screen.getByLabelText('下载模式'), { target: { value: 'full' } })
    finish({ valid: true, errors: [], steps: [] })
    await screen.findByText('配置已修改，旧校验结果已失效，请重新校验。')
    expect(etlApi.runEtl).not.toHaveBeenCalled()
  })
  it('来源目录失败不遮挡独立任务状态，未知编辑权限时不允许操作', async () => {
    vi.mocked(sourceApi.fetchSourceCatalog).mockRejectedValue(new Error('目录暂不可用'))
    vi.mocked(etlApi.listEtlRuns).mockResolvedValue([success])
    render(<MemoryRouter><DataDownloadWorkspace /></MemoryRouter>)
    expect(await screen.findByRole('heading', { name:'下载任务' })).toBeInTheDocument()
    expect(screen.getByText(/任务进度独立读取/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name:'取消运行' })).toBeDisabled()
  })
  it('切换来源只显示该来源数据，保留既有默认参数', async () => {
    await load()
    expect(screen.getByLabelText('ETF 日行情')).toBeInTheDocument()
    expect(screen.queryByText('AK ETF 行情')).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('下载数据源'),{target:{value:'akshare'}})
    expect(screen.getByLabelText('公募基金净值')).toBeInTheDocument()
    expect(screen.queryByText('fund_daily')).not.toBeInTheDocument()
    fireEvent.click(screen.getByLabelText('AK ETF 行情'))
    expect(screen.getByLabelText('产品代码（symbol）')).toHaveValue('510300')
  })
  it('全量与日期参数进入真实执行计划，下载和映射是独立步骤',async()=>{
    await load(); fireEvent.click(screen.getByLabelText('ETF 日行情'))
    fireEvent.change(screen.getByLabelText('下载模式'),{target:{value:'full'}})
    fireEvent.change(screen.getByLabelText('开始日期'),{target:{value:'2024-01-01'}})
    fireEvent.click(screen.getByRole('button',{name:'确认并开始下载'}))
    await waitFor(()=>expect(etlApi.runEtl).toHaveBeenCalledOnce())
    const [definition, , options] = vi.mocked(etlApi.runEtl).mock.calls[0]
    expect(definition.steps.map(s=>s.kind)).toEqual(['download','map','resolve'])
    expect(definition.steps[0]).toMatchObject({source_id:'tushare',mode:'inherit',params:{symbol:'510300',start_date:'20240101'}})
    expect(options).toEqual({mode:'full', parameters:{}})
    expect(screen.getByRole('heading',{name:'下载任务'})).toBeInTheDocument()
  })
  it('下载转为公共画布，新增节点可撤销重做并保存布局',async()=>{
    await load(); fireEvent.click(screen.getByLabelText('ETF 日行情'))
    fireEvent.click(screen.getByRole('button',{name:'转为 ETL 流程编辑'}))
    expect(screen.getByTestId('etl-graph-canvas')).toHaveTextContent('3 节点')
    expect(screen.queryByLabelText('步骤名称')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button',{name:'节点库'}))
    fireEvent.click(screen.getByRole('button',{name:'＋指标快照计算'}))
    expect(screen.getByTestId('etl-graph-canvas')).toHaveTextContent('4 节点')
    expect(screen.getAllByLabelText('步骤名称')).toHaveLength(1)
    fireEvent.click(screen.getByRole('button',{name:'撤销'}))
    expect(screen.getByTestId('etl-graph-canvas')).toHaveTextContent('3 节点')
    fireEvent.click(screen.getByRole('button',{name:'重做'}))
    expect(screen.getByTestId('etl-graph-canvas')).toHaveTextContent('4 节点')
    vi.mocked(etlApi.saveEtlWorkflow).mockImplementation(async(id,definition)=>({id,definition,revision:1,updated_at:''}))
    fireEvent.click(screen.getByRole('button',{name:'保存流程'}))
    await waitFor(()=>expect(etlApi.saveEtlWorkflow).toHaveBeenCalledOnce())
    expect(vi.mocked(etlApi.saveEtlWorkflow).mock.calls[0][1]).toMatchObject({graph_version:1,canvas:{version:1}})
  })
  it('校验失败不启动，后台状态失联也不能重复提交',async()=>{
    await load(); fireEvent.click(screen.getByLabelText('ETF 日行情'))
    vi.mocked(etlApi.validateEtl).mockResolvedValue({valid:false,errors:[{code:'BAD',message:'接口需要更新'}],steps:[]})
    fireEvent.click(screen.getByRole('button',{name:'确认并开始下载'}))
    expect(await screen.findByText('接口需要更新')).toBeInTheDocument()
    expect(etlApi.runEtl).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: '确认并开始下载' })).toBeDisabled()
    vi.mocked(etlApi.validateEtl).mockResolvedValue({ valid: true, errors: [], steps: [] })
    fireEvent.click(screen.getByRole('button', { name: '校验下载配置' }))
    await waitFor(() => expect(screen.getByRole('button', { name: '确认并开始下载' })).toBeEnabled())
  })
  it('只读环境禁止保存和运行但可查看',async()=>{
    vi.mocked(sourceApi.fetchSourceCatalog).mockResolvedValue({...catalog,editing_enabled:false})
    await load(); fireEvent.click(screen.getByLabelText('ETF 日行情'))
    expect(screen.getByRole('button',{name:'确认并开始下载'})).toBeDisabled()
  })
  it('同一个已保存流程可切换运行模式，不要求另存流程', async () => {
    const definition = etlApi.quickPlan([catalog.interfaces[0]], 'incremental', {}, false)
    const saved = { id:'same_flow', revision:3, definition, updated_at:'' }
    vi.mocked(etlApi.listEtlWorkflows).mockResolvedValue([saved])
    await load()
    fireEvent.click(screen.getByRole('button', {name:'ETL 任务编排'}))
    fireEvent.change(screen.getByLabelText('已保存流程'), {target:{value:'same_flow'}})
    expect(screen.getByTestId('etl-graph-canvas')).toHaveTextContent('3 节点')
    fireEvent.change(screen.getByLabelText('本次运行模式'), {target:{value:'full'}})
    fireEvent.click(screen.getByRole('button', {name:'确认并运行流程'}))
    await waitFor(() => expect(etlApi.runEtl).toHaveBeenCalledOnce())
    expect(etlApi.runEtl).toHaveBeenCalledWith(definition, expect.any(String), {mode:'full', parameters:{}})
    expect(etlApi.saveEtlWorkflow).not.toHaveBeenCalled()
    expect(saved.definition.steps[0].mode).toBe('inherit')
  })
  it('产品与日期作为运行参数传递，不更改已保存的接口参数', async () => {
    const definition = etlApi.quickPlan([catalog.interfaces[0]], 'incremental', {}, false)
    definition.parameters = [{id:'product',label:'本次产品代码',data_type:'text',default:'',required:true,date_format:'iso',description:'运行时选择'}]
    definition.steps[0].parameter_bindings = {symbol:'product'}
    const saved = {id:'parameter_flow',revision:1,definition,updated_at:''}
    vi.mocked(etlApi.listEtlWorkflows).mockResolvedValue([saved])
    await load()
    fireEvent.click(screen.getByRole('button', {name:'ETL 任务编排'}))
    fireEvent.change(screen.getByLabelText('已保存流程'), {target:{value:'parameter_flow'}})
    fireEvent.change(screen.getByLabelText('本次产品代码'), {target:{value:'159915'}})
    fireEvent.click(screen.getByRole('button', {name:'确认并运行流程'}))
    await waitFor(() => expect(etlApi.runEtl).toHaveBeenCalledOnce())
    expect(etlApi.runEtl).toHaveBeenCalledWith(definition, expect.any(String), {mode:'incremental',parameters:{product:'159915'}})
    expect(definition.steps[0].params.symbol).toBe('510300')
  })
  it('运行参数未填写也可以先保存流程定义', async () => {
    const definition = etlApi.quickPlan([catalog.interfaces[0]], 'incremental', {}, false)
    definition.parameters = [{id:'code',label:'待填写产品',data_type:'text',default:'',required:true,date_format:'iso',description:''}]
    vi.mocked(etlApi.listEtlWorkflows).mockResolvedValue([{id:'unfilled',revision:1,definition,updated_at:''}])
    vi.mocked(etlApi.saveEtlWorkflow).mockImplementation(async (id, value) => ({id,definition:value,revision:2,updated_at:''}))
    await load()
    fireEvent.click(screen.getByRole('button', {name:'ETL 任务编排'}))
    fireEvent.change(screen.getByLabelText('已保存流程'), {target:{value:'unfilled'}})
    fireEvent.click(screen.getByRole('button', {name:'保存流程'}))
    await waitFor(() => expect(etlApi.saveEtlWorkflow).toHaveBeenCalledWith('unfilled', expect.objectContaining({name:definition.name,parameters:definition.parameters,graph_version:1,canvas:expect.objectContaining({version:1})}), 1))
    expect(etlApi.runEtl).not.toHaveBeenCalled()
  })
  it('运行失败允许恢复下载而不创建新运行',async()=>{
    vi.mocked(etlApi.listEtlRuns).mockResolvedValue([{...success,status:'FAILED',error:'取值冲突'}])
    vi.mocked(etlApi.recoverEtl).mockResolvedValue({ id: 'job', source_run_id: 'abc', target_run_id: 'abc', status: 'QUEUED', phase: '等待校验', message: '已接收', created_at: '', updated_at: '', logs: [] })
    await load(); fireEvent.click(screen.getByRole('button',{name:'运行记录与恢复'}))
    fireEvent.click(screen.getByRole('button',{name:'恢复下载'}))
    fireEvent.click(screen.getByRole('button',{name:'确认继续'}))
    await waitFor(()=>expect(etlApi.recoverEtl).toHaveBeenCalledWith('abc', expect.any(String)))
    expect(etlApi.runEtl).not.toHaveBeenCalled()
  })
  it('旧后端拒绝恢复时，按钮附近显示错误，不只显示在长页面顶部', async () => {
    vi.mocked(etlApi.listEtlRuns).mockResolvedValue([{...success,status:'INTERRUPTED'}])
    vi.mocked(etlApi.recoverEtl).mockRejectedValue(new Error('后台仍有下载进程持锁，不能重复启动。'))
    await load(); fireEvent.click(screen.getByRole('button',{name:'运行记录与恢复'}))
    const actions = within(screen.getByRole('region',{name:'运行运行操作'}))
    fireEvent.click(actions.getByRole('button',{name:'恢复下载'}))
    expect(etlApi.recoverEtl).not.toHaveBeenCalled()
    fireEvent.click(actions.getByRole('button',{name:'确认继续'}))
    expect(await actions.findByRole('alert')).toHaveTextContent('不能重复启动')
    expect(etlApi.runEtl).not.toHaveBeenCalled()
  })
})
