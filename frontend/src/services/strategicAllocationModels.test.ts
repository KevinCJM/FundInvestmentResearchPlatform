import { afterEach, expect, it, vi } from 'vitest'
import { previewPolicy } from './strategicAllocation'
import { policyPreview } from '../test/strategicAllocationFixtures'
afterEach(()=>{vi.unstubAllGlobals()})
it('separates unavailable fifth candidate without manufacturing weights or goal evidence', async()=>{
  const unavailable={id:'risk-budget',name:'风险预算匹配',available:false,weights:{},risk_contributions:{},metrics:{},risk_budget:{equity:.5,bond:.5},risk_budget_distance:null,unavailable_reason:'零方差风险贡献未定义'}
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({...policyPreview,candidates:[...policyPreview.candidates,unavailable]})}))
  const value=await previewPolicy({...policyPreview.request,risk_budget:{equity:.5,bond:.5}})
  expect(value.candidates).toEqual(policyPreview.candidates)
  expect(value.unavailable_candidates).toEqual([unavailable])
})
