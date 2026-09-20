import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import PreInvestmentImplementation from './PreInvestmentImplementation'
import CandidateFields, { initialCandidate } from '../components/implementation/CandidateFields'
import type { ImplementationReport, SourceOption } from '../services/implementation'

const source:SourceOption={kind:'saa_policy',id:'source-1',content_hash:'a'.repeat(64),name:'股债政策',as_of:'2026-09-01',expires_on:'2027-01-01',mode:'single',
  assets:[{id:'stock',products:[{kind:'etf',product_id:'510300.SH'}]}],target:{stock:1},funding:null}
const candidate=initialCandidate(source,'2026-09-19')
const item={id:'version-1',scheme_id:'package-1',revision:1,name:'股债实施研究',stage:'validation_complete',candidate,candidate_hash:'b'.repeat(64),report_id:'report-1'}
const report:ImplementationReport={id:'report-1',content_hash:'c'.repeat(64),candidate_hash:item.candidate_hash,research_ready:true,implementation_eligibility:'conditions_incomplete',
  independent_simulation:true,validation_mode:'frozen_candidate_validation',checked_at:'2026-09-19',checks:[{check_id:'path',title:'全期产品路径与结算',status:'unavailable',reason:'缺少逐产品到账证据',scope:'implementation',enforcement:'hard'}],models:[],limitations:['不代表交易授权']}
function install(options:{failed?:boolean;empty?:boolean;blocked?:boolean;stale?:boolean}={}) {
  const fetch=vi.fn(async(input:RequestInfo|URL,init?:RequestInit)=>{
    const path=String(input)
    let value:unknown
    if(options.failed) return {ok:false,json:async()=>({detail:{message:'数据磁盘暂不可用'}})} as Response
    if(path.endsWith('/catalog')) value={today:'2026-09-19',sources:options.empty?[]:[source],mappings:[]}
    else if(path.endsWith('/packages')) value=init?.method==='POST'?item:{items:options.empty?[]:[item]}
    else if(path.endsWith('/preview')) value=report
    else if(path.endsWith('/finalize')) value={...item,stage:'finalized'}
    else if(path.endsWith('/validate')) value=item
    else if(path.endsWith('/packages/package-1')) value={package:item,report:{...report,research_ready:!options.blocked},history:[item],current_eligibility:{status:options.stale?'needs_review':'current',reasons:options.stale?['来源已停止引用']:[]}}
    else throw new Error(path)
    return {ok:true,status:200,json:async()=>value} as Response
  })
  vi.stubGlobal('fetch',fetch);return fetch
}
function tree(path:string) {return <MemoryRouter initialEntries={[path]}><Routes>
  <Route path="/pre-investment/product-allocation-timing" element={<PreInvestmentImplementation/>}/>
  <Route path="/pre-investment/portfolio-synthesis" element={<PreInvestmentImplementation mode="synthesis"/>}/>
  <Route path="/pre-investment/validation" element={<PreInvestmentImplementation mode="validation"/>}/>
  <Route path="/pre-investment/approval" element={<PreInvestmentImplementation mode="approval"/>}/>
</Routes></MemoryRouter>}
beforeEach(()=>{localStorage.clear();sessionStorage.clear();vi.clearAllMocks()})
afterEach(()=>{vi.unstubAllGlobals()})

it('explains the next action when no source exists without generating research',async()=>{
  const fetch=install({empty:true});render(tree('/pre-investment/product-allocation-timing'))
  expect(await screen.findByRole('link',{name:'前往 SAA'})).toHaveAttribute('href','/pre-investment/saa/policy')
  expect(fetch.mock.calls.every(([,init])=>init?.method==='GET')).toBe(true)
})
it('loads a direct SAA source and preserves unknown fees as null',async()=>{
  const fetch=install();render(tree('/pre-investment/product-allocation-timing?source=source-1'))
  expect(await screen.findByLabelText('方案名称')).toHaveValue(source.name)
  expect(screen.getByLabelText('买入每边费率（%）')).toHaveValue(null)
  fireEvent.click(screen.getByRole('button',{name:'检查当前方案'}))
  await screen.findByText('实施条件未齐')
  const call=fetch.mock.calls.find(([url])=>String(url).endsWith('/preview'))!
  expect(JSON.parse(String(call[1]?.body)).products[0].buy_rate).toBeNull()
  fireEvent.change(screen.getByLabelText('方案名称'),{target:{value:'新输入'}})
  expect(screen.queryByText('缺少逐产品到账证据')).not.toBeInTheDocument()
})
it('shows source failure with a usable retry',async()=>{
  install({failed:true});render(tree('/pre-investment/portfolio-synthesis'))
  expect(await screen.findByText('数据磁盘暂不可用')).toBeVisible()
  expect(screen.getByRole('button',{name:'刷新数据'})).toBeEnabled()
})
it('prevents finalization while hard checks fail',async()=>{
  const fetch=install({blocked:true});render(tree('/pre-investment/approval?package=package-1'))
  expect(await screen.findByRole('button',{name:'确认研究定稿'})).toBeDisabled()
  expect(screen.getByText(/请先完成当前候选的验证/)).toBeVisible()
  expect(fetch.mock.calls.some(([url])=>String(url).endsWith('/finalize'))).toBe(false)
})
it('retains explicit limits and sends exact candidate and report hashes',async()=>{
  const fetch=install();render(tree('/pre-investment/approval?package=package-1'))
  await screen.findByLabelText('研究负责人')
  fireEvent.change(screen.getByLabelText('研究负责人'),{target:{value:'研究员甲'}})
  fireEvent.change(screen.getByLabelText('下次复核日期'),{target:{value:'2026-12-31'}})
  fireEvent.change(screen.getByLabelText('定稿理由与限制'),{target:{value:'已核对来源和限制，保存研究结论'}})
  fireEvent.click(screen.getByRole('checkbox',{name:/我已核对报告并接受/}))
  fireEvent.click(screen.getByRole('button',{name:'确认研究定稿'}))
  await waitFor(()=>expect(fetch.mock.calls.some(([url])=>String(url).endsWith('/finalize'))).toBe(true))
  const call=fetch.mock.calls.find(([url])=>String(url).endsWith('/finalize'))!
  expect(JSON.parse(String(call[1]?.body))).toMatchObject({candidate_hash:item.candidate_hash,validation_report_hash:report.content_hash,expected_revision:1,accept_research_limits:true})
})
it('retired dependencies disable new validation and keep copy/export available',async()=>{
  install({stale:true});render(tree('/pre-investment/validation?package=package-1'))
  expect(await screen.findByText('来源已停止引用')).toBeVisible()
  expect(screen.getByRole('button',{name:'锁定候选并独立验证'})).toBeDisabled()
  expect(screen.getByRole('link',{name:'复制重研'})).toHaveAttribute('href','/pre-investment/product-allocation-timing?copy=package-1')
  expect(screen.getByRole('link',{name:'导出完整研究包'})).toHaveAttribute('href','/api/pre-investment/packages/package-1/export')
})


it('shows the original nominal due amount without treating a planned amount as paid',()=>{
  const flow={occurrence_id:'d'.repeat(64),name:'每月支出',kind:'withdrawal',month:1,amount:100,nominal_amount:100.165158,due_at:'2026-09-19'}
  const withFunding:SourceOption={...source,funding:{origin:'2026-08-19',months:24,plan:{total_capital:100000,outside_reserve:0,inflation:.02,amount_basis:'real'},occurrences:[flow]}}
  const onChange=vi.fn()
  render(<CandidateFields candidate={{...candidate,state:{...candidate.state,elapsed_months:1}}} source={withFunding} catalog={{today:'2026-09-19',sources:[withFunding],mappings:[]}} onChange={onChange}/>)
  fireEvent.click(screen.getByRole('button',{name:'2. 资金与支付'}))
  expect(screen.getByText(/100.165158/)).toBeVisible()
  fireEvent.click(screen.getByRole('button',{name:'填写计划金额（仍需核对实际付款）'}))
  const row=onChange.mock.calls[0][0].state.reconciliation[0]
  expect(row).toMatchObject({paid_amount:100.165158,status:'unknown',evidence:''})
})
