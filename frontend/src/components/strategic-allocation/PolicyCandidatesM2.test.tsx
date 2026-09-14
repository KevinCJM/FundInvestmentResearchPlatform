import { fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import PolicyCandidates from './PolicyCandidates'
import { policyPreview } from '../../test/strategicAllocationFixtures'
import type { PolicyCandidate, PolicyPreview } from '../../services/strategicAllocation'

const fifth: PolicyCandidate={...policyPreview.candidates[0],id:'risk-budget',name:'风险预算匹配（有限搜索）',available:true,
  risk_budget_distance:.02,risk_contributions:{equity:1.1,bond:-.1}}
it('shows signed contributions and finite-search distance for the selectable fifth',()=>{
  const select=vi.fn()
  render(<PolicyCandidates value={policyPreview.request} assets={['equity','bond']} result={{...policyPreview,candidates:[fifth]}} busy={false} compareDisabled={false} onChange={vi.fn()} onCompare={vi.fn()} onSelect={select} />)
  fireEvent.click(screen.getByText('风险贡献与证据限制'))
  expect(screen.getByText(/bond -10.00%/)).toHaveTextContent('风险预算平方距离 0.020000')
  expect(screen.getByText(/有限搜索不保证精确匹配/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button',{name:'复核此候选'}))
  expect(select).toHaveBeenCalledWith(fifth)
})
it('unavailable risk budget has a visible reason and no fabricated selectable weights',()=>{
  const result: PolicyPreview={...policyPreview,unavailable_candidates:[{id:'risk-budget',name:'风险预算匹配',available:false,unavailable_reason:'零方差未定义',weights:{},risk_contributions:{},risk_budget:{equity:.5,bond:.5},risk_budget_distance:null,
    metrics:{expected_return:null,volatility:null,conservative_return:null,nominal_utility:null,robust_utility:null}}]}
  render(<PolicyCandidates value={policyPreview.request} assets={['equity','bond']} result={result} busy={false} compareDisabled={false} onChange={vi.fn()} onCompare={vi.fn()} onSelect={vi.fn()} />)
  expect(screen.getByText('风险预算匹配不可用：零方差未定义')).toBeInTheDocument()
  expect(screen.getAllByRole('button',{name:'复核此候选'})).toHaveLength(1)
})
