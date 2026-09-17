import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, afterEach, it, expect, vi } from 'vitest'
import StrategicAllocationWorkspace from './StrategicAllocationWorkspace'
import { writeAllocationDraft } from '../app/allocationJourney'
import { cmaDefinition, cmaPreview, cmaVersion, strategicCatalog, policyPreview } from '../test/strategicAllocationFixtures'
import { taaExecution } from '../test/tacticalAllocationFixtures'
import type { BlackLittermanRequest } from '../services/cmaModelTypes'

const clock = vi.hoisted(() => ({ day: '2026-09-12' as string | undefined }))
vi.mock('../app/ResearchContext', () => ({ useResearchDay: () => clock.day }))
const model: BlackLittermanRequest = { method: 'black_litterman', asset_ids: ['equity','bond'], as_of: cmaDefinition.as_of,
  currency: 'CNY', source: '模型风险输入来源', covariance: [[.04,0],[0,.01]], risk_covariance_basis: 'input_covariance',
  market_weights: { equity: .6, bond: .4 }, market_weight_source: '显式市场权重来源', delta: 3, tau: .05, risk_free_rate: .02, views: [] }
const raw = { ...cmaDefinition, model, assets: cmaDefinition.assets.map(a => ({ ...a, annual_return: null, annual_volatility: null })), correlation: null }
const effective = { ...cmaDefinition, model, assets: cmaDefinition.assets.map((a,i) => ({ ...a, annual_return: [.092,.032][i], annual_volatility: [.2,.1][i] })) }
const result = { ...cmaPreview, definition: raw, effective_assumptions: effective, effective_returns: [.092,.032], effective_covariance: model.covariance,
  covariance: model.covariance, model_result: { asset_ids: model.asset_ids, method: model.method, definition: model,
    effective_returns: [.092,.032], effective_covariance: model.covariance, posterior_mean_covariance: [[.002,0],[0,.0005]], content_hash: 'f'.repeat(64),
    model_audit: { limitations: ['均值后验协方差不作为资产风险，也不自动转为稳健半宽。'] }, execution: taaExecution } }
const response = (x: unknown) => Promise.resolve({ ok: true, json: async () => x } as Response)
function install(overrides: Record<string,(init?: RequestInit)=>Promise<Response>> = {}) {
  const fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url=String(input)
    if (overrides[url]) return overrides[url](init)
    if(url.endsWith('/catalog')) return response(strategicCatalog)
    if(url.endsWith('/cma/cma-1')) return response({ ...cmaVersion,...result })
    if(url.endsWith('/cma/preview')) return response(result)
    if(url.endsWith('/cma')) return response({ ...cmaVersion,...result })
    if(url.endsWith('/policy/preview')) return response({ ...policyPreview, request: JSON.parse(String(init?.body)) })
    throw new Error(url)
  })
  vi.stubGlobal('fetch',fetch); return fetch
}
function tree() { return <MemoryRouter initialEntries={['/pre-investment/saa/policy?alloc=股债分类&mandate=mandate-1']}><StrategicAllocationWorkspace /></MemoryRouter> }
function draft() {
  writeAllocationDraft('strategic-policy:股债分类:mandate-1', { mandateId:'mandate-1', allocationName:'股债分类', strategicUniverseId:'', implementationMappingId:'',
    assumptions: { ...cmaDefinition, model, assets: cmaDefinition.assets.map(a=>({...a,annual_return:NaN,annual_volatility:NaN})) },
    reference: { alloc_name:'股债分类',as_of:cmaDefinition.as_of,start_date:'2026-01-01',end_date:cmaDefinition.as_of,shrinkage:.1,periods_per_year:252 },
    settings: policyPreview.request, policyName:'模型政策',reason:'' })
}
async function openEditor() {
  await waitFor(()=>expect(screen.getByRole('button',{name:'填写长期假设'})).toBeEnabled())
  fireEvent.click(screen.getByRole('button',{name:'填写长期假设'}))
}
beforeEach(()=>{ clock.day='2026-09-12'; localStorage.clear(); sessionStorage.clear() })
afterEach(()=>{ vi.unstubAllGlobals() })

it('mounts model mode without duplicate mean/risk inputs and saves actual effective results',async()=>{
  const fetch=install();draft();render(tree());await openEditor()
  expect(screen.getByLabelText('预期生成方法')).toHaveValue('black_litterman')
  expect(screen.queryByLabelText('equity预期年收益（%）')).toBeNull()
  expect(screen.queryByLabelText('equity年化波动（%）')).toBeNull()
  expect(screen.getByLabelText('equity均值不确定半宽（百分点）')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button',{name:'验证长期假设'}))
  expect(await screen.findByRole('table',{name:'有效收益与风险'})).toHaveTextContent('9.20%')
  expect(fetch.mock.calls.filter(([u])=>String(u).endsWith('/cma'))).toHaveLength(0)
  const body=JSON.parse(String(fetch.mock.calls.find(([u])=>String(u).endsWith('/cma/preview'))![1]?.body))
  expect(body.model.method).toBe('black_litterman')
  expect(body.assets[0].annual_return).toBeUndefined()
  expect(body.correlation).toBeUndefined()
  fireEvent.click(screen.getByRole('button',{name:'确认保存假设版本'}))
  await screen.findByRole('button',{name:'比较符合目标的政策候选'})
  fireEvent.click(screen.getByRole('button',{name:'2. 长期假设'}))
  expect(screen.getByLabelText('预期生成方法')).toBeDisabled()
  fireEvent.click(screen.getByRole('button',{name:'复制为新研究'}))
  expect(screen.getByLabelText('预期生成方法')).toBeEnabled()
  expect(screen.getByRole('button',{name:'确认保存假设版本'})).toBeDisabled()
})

it('discards late model preview when clock changes',async()=>{
  let resolve!:(r:Response)=>void
  install({ '/api/strategic-allocation/cma/preview':()=>new Promise(done=>{resolve=done}) });draft()
  const view=render(tree());await openEditor()
  fireEvent.click(screen.getByRole('button',{name:'验证长期假设'}))
  clock.day=undefined;view.rerender(tree())
  await act(async()=>resolve(await response(result)))
  expect(screen.queryByRole('table',{name:'有效收益与风险'})).toBeNull()
  expect(screen.getByRole('button',{name:'确认保存假设版本'})).toBeDisabled()
})

it('switching manual/scenario/BL clears stale manual certifications and starts blank',async()=>{
  install();render(tree());await openEditor()
  const user=userEvent.setup()
  await user.selectOptions(screen.getByLabelText('预期生成方法'),'scenario_mixture')
  expect(screen.queryByLabelText('equity预期年收益（%）')).toBeNull()
  expect(screen.getByText(/尚无情景，请添加/)).toBeInTheDocument()
  await user.selectOptions(screen.getByLabelText('预期生成方法'),'black_litterman')
  expect(screen.getByLabelText('equity市场权重（%）')).toHaveValue('')
  expect(screen.getByRole('button',{name:'验证长期假设'})).toBeDisabled()
  await user.selectOptions(screen.getByLabelText('预期生成方法'),'manual')
  expect(screen.getByLabelText('equity预期年收益（%）')).toHaveValue('')
  expect(screen.getByRole('button',{name:'读取历史风险参考'})).toBeInTheDocument()
})

it('loads saved model readonly and sends optional budget through actual policy API',async()=>{
  const fetch=install();render(tree());const user=userEvent.setup()
  await waitFor(()=>expect(screen.getByRole('button',{name:'填写长期假设'})).toBeEnabled())
  await user.selectOptions(screen.getByLabelText('或者使用已保存的长期假设'),'cma-1')
  await screen.findByRole('button',{name:'比较符合目标的政策候选'})
  fireEvent.click(screen.getByLabelText('增加风险预算候选'))
  expect(screen.getByRole('button',{name:'比较符合目标的政策候选'})).toBeDisabled()
  fireEvent.change(screen.getByLabelText('equity风险预算（%）'),{target:{value:'30'}})
  fireEvent.change(screen.getByLabelText('bond风险预算（%）'),{target:{value:'70'}})
  fireEvent.click(screen.getByRole('button',{name:'比较符合目标的政策候选'}))
  await screen.findByRole('table',{name:'长期政策候选比较'})
  expect(JSON.parse(String(fetch.mock.calls.find(([u])=>String(u).endsWith('/policy/preview'))![1]?.body)).risk_budget).toEqual({equity:.3,bond:.7})
})

it('saved model reference restores readonly after reload, with explicit copy before edit', async()=>{
  install(); draft()
  const key='allocation-draft:v1:strategic-policy:股债分类:mandate-1'
  localStorage.setItem(key,JSON.stringify({...JSON.parse(localStorage.getItem(key)!),savedCmaId:'cma-1'}))
  render(tree())
  await screen.findByRole('button',{name:'比较符合目标的政策候选'})
  fireEvent.click(screen.getByRole('button',{name:'2. 长期假设'}))
  expect(screen.getByLabelText('预期生成方法')).toBeDisabled()
  expect(screen.getByRole('button',{name:'复制为新研究'})).toBeEnabled()
})

it('changing the selected source drops old model state and late saved responses', async()=>{
  let resolve!:(r:Response)=>void
  install({ '/api/strategic-allocation/cma/cma-1':()=>new Promise(done=>{resolve=done}) })
  render(tree());const user=userEvent.setup()
  await waitFor(()=>expect(screen.getByRole('button',{name:'填写长期假设'})).toBeEnabled())
  await user.selectOptions(screen.getByLabelText('或者使用已保存的长期假设'),'cma-1')
  fireEvent.change(screen.getByLabelText('已保存的大类配置'),{target:{value:''}})
  await act(async()=>resolve(await response({...cmaVersion,...result})))
  expect(screen.queryByRole('button',{name:'比较符合目标的政策候选'})).toBeNull()
  fireEvent.change(screen.getByLabelText('已保存的大类配置'),{target:{value:'股债分类'}})
  await openEditor()
  expect(screen.getByLabelText('预期生成方法')).toHaveValue('manual')
})
