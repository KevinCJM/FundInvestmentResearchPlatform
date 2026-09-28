import { i18n } from '../../i18n/runtime'
import { act, fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import PolicyCandidates from './PolicyCandidates'
import { policyPreview } from '../../test/strategicAllocationFixtures'
import type { PolicyCandidate, PolicyPreview } from '../../services/strategicAllocation'

const fifth: PolicyCandidate={...policyPreview.candidates[0],id:'risk-budget',name:'风险预算匹配（有限搜索）',available:true,
  risk_budget_distance:.02,risk_contributions:{equity:1.1,bond:-.1}}
it('shows signed contributions and finite-search distance for the selectable fifth',()=>{
  const select=vi.fn()
  render(<PolicyCandidates view="results" value={policyPreview.request} assets={['equity','bond']} result={{...policyPreview,candidates:[fifth]}} busy={false} compareDisabled={false} onChange={vi.fn()} onCompare={vi.fn()} onSelect={select} />)
  fireEvent.click(screen.getByText('风险贡献与证据限制'))
  expect(screen.getByText(/bond -10.00%/)).toHaveTextContent('风险预算平方距离 0.020000')
  expect(screen.getByText(/有限搜索不保证精确匹配/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button',{name:'复核此候选'}))
  expect(select).toHaveBeenCalledWith(fifth)
})
it('unavailable risk budget has a visible reason and no fabricated selectable weights',()=>{
  const result: PolicyPreview={...policyPreview,unavailable_candidates:[{id:'risk-budget',name:'风险预算匹配',available:false,unavailable_reason:'零方差未定义',weights:{},risk_contributions:{},risk_budget:{equity:.5,bond:.5},risk_budget_distance:null,
    metrics:{expected_return:null,volatility:null,conservative_return:null,nominal_utility:null,robust_utility:null}}]}
  render(<PolicyCandidates view="results" value={policyPreview.request} assets={['equity','bond']} result={result} busy={false} compareDisabled={false} onChange={vi.fn()} onCompare={vi.fn()} onSelect={vi.fn()} />)
  expect(screen.getByText('风险预算匹配不可用：零方差未定义')).toBeInTheDocument()
  expect(screen.getAllByRole('button',{name:'复核此候选'})).toHaveLength(1)
})

it('localizes generated candidate names while preserving the selected frozen candidate', async () => {
  const select = vi.fn()
  render(<PolicyCandidates view="results" value={policyPreview.request} assets={['equity', 'bond']} result={{...policyPreview, candidates:[fifth]}} busy={false} compareDisabled={false} onChange={vi.fn()} onCompare={vi.fn()} onSelect={select} />)
  try {
    await act(async () => { await i18n.changeLanguage('en-US') })
    expect(screen.getByRole('rowheader', {name:/^Risk budget matching \(finite search\)/})).toBeVisible()
    fireEvent.click(screen.getByRole('button', {name:'Review this candidate'}))
    expect(select).toHaveBeenCalledWith(fifth)
    expect(fifth.name).toBe('风险预算匹配（有限搜索）')
  } finally { await act(async () => { await i18n.changeLanguage('zh-CN') }) }
})

it('每个候选方法都有问号说明；夏普写明无风险利率来源，回撤写明模拟口径', () => {
  const base = policyPreview.candidates[0]
  const sharpe: PolicyCandidate = { ...base, id: 'maximum-sharpe', name: '候选中最大夏普比率', risk_adjusted: { sharpe_ratio: .6523, mean_max_drawdown: .1234 } }
  const drawdown: PolicyCandidate = { ...base, id: 'minimum-drawdown', name: '候选中最小模拟回撤', risk_adjusted: { sharpe_ratio: null, mean_max_drawdown: .05 } }
  const result: PolicyPreview = { ...policyPreview, candidates: [{ ...base, risk_adjusted: { sharpe_ratio: .4, mean_max_drawdown: .2 } }, sharpe, drawdown],
    risk_adjusted_basis: { risk_free: { rate: .025, source: 'risk_scale_cash', asset_id: 'cash', asset_name: '货币基金', risk_scale: { id: 's', name: '系统标尺', content_hash: 'h' } },
      drawdown: { model: 'shared_monthly_lognormal_paths_mean_max_drawdown', paths: 256, months: 120, seed: 42 } } }
  const onBack = vi.fn()
  render(<PolicyCandidates view="results" value={policyPreview.request} assets={['equity', 'bond']} result={result} busy={false} compareDisabled={false} onChange={vi.fn()} onCompare={vi.fn()} onSelect={vi.fn()} onBack={onBack} />)
  expect(screen.getByRole('columnheader', { name: '夏普比率' })).toBeInTheDocument()
  expect(screen.getByRole('row', { name: /候选中最大夏普比率/ })).toHaveTextContent('0.65')
  fireEvent.mouseEnter(screen.getByRole('button', { name: '候选中最大夏普比率 的含义' }))
  expect(screen.getByRole('tooltip')).toHaveTextContent('无风险利率 2.50%，来源：范围未设现金收益，采用风险标尺“系统标尺”中现金大类“货币基金”的收益')
  fireEvent.mouseEnter(screen.getByRole('button', { name: '候选中最小模拟回撤 的含义' }))
  expect(screen.getAllByRole('tooltip').slice(-1)[0]).toHaveTextContent('模拟 256 条、120 个月')
  expect(screen.getByRole('button', { name: `${screen.getAllByRole('rowheader')[0].textContent?.replace('?', '')} 的含义` })).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '比较符合目标的政策候选' })).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '返回调整设置' }))
  expect(onBack).toHaveBeenCalled()
})
