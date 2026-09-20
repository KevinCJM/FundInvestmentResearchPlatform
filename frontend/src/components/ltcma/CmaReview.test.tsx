import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import BlackLittermanViews from '../strategic-allocation/BlackLittermanViews'
import MeanUncertaintyFields, { MeanUncertaintySummary } from '../strategic-allocation/MeanUncertaintyFields'
import LtcmaModelDiagnostics from './LtcmaModelDiagnostics'
import { cmaModelInputError, type BlackLittermanRequest } from '../../services/cmaModelTypes'
import type { PolicyRequest } from '../../services/strategicAllocation'

const model: BlackLittermanRequest = { method:'black_litterman', asset_ids:['股票','债券','黄金'], as_of:'2026-09-20', currency:'CNY', source:'明确的离线模型依据',
  covariance:[[.04,.006,0],[.006,.01,0],[0,0,.02]], risk_covariance_basis:'input_covariance', market_weights:{'股票':.5,'债券':.3,'黄金':.2},
  market_weight_source:'离线市场权重', delta:2.5,tau:.05,risk_free_rate:.02, views:[{kind:'absolute',asset_id:'股票',annual_return:.06,view_std:.02,observed_on:'2026-09-19',available_on:'2026-09-19',source:'离线观点依据'}] }
const policy: PolicyRequest = { mandate_id:'mandate-1', cma_id:'cma-1', constraints:{},group_limits:[],uncertainty_penalty:1,candidate_count:300,seed:42 }

function ViewEditor() {
  const [value,setValue]=useState(model)
  return <><BlackLittermanViews value={value} onChange={setValue}/><p role="status" aria-label="输入校验">{cmaModelInputError(value) ?? '输入有效'}</p><output data-testid="payload">{JSON.stringify(value.views[0])}</output></>
}
function UncertaintyEditor({available=true,mode='single'}:{available?:boolean;mode?:PolicyRequest['mode']}) {
  const [value,setValue]=useState<PolicyRequest>({...policy,mode})
  return <><MeanUncertaintyFields value={value} available={available} onChange={setValue}/><output data-testid="payload">{JSON.stringify(value)}</output></>
}

describe('CMA model review controls',()=>{
  it('edits a real basket without inventing coefficients or dropping shared evidence',async()=>{
    const user=userEvent.setup();render(<ViewEditor/>)
    await user.selectOptions(screen.getByRole('combobox',{name:'观点1类型'}),'basket')
    expect(screen.getByRole('status',{name:'输入校验'})).toHaveTextContent('不能全零')
    for(const [asset,value] of [['股票','1'],['债券','-0.5'],['黄金','-0.5']]) fireEvent.change(screen.getByLabelText(`观点1 · ${asset}系数`),{target:{value}})
    expect(screen.getByRole('status',{name:'输入校验'})).toHaveTextContent('输入有效')
    const payload=JSON.parse(screen.getByTestId('payload').textContent!)
    expect(payload).toMatchObject({kind:'basket',basis:'relative',source:'离线观点依据',annual_return:.06})
    expect(payload).not.toHaveProperty('asset_id')
    expect(payload.legs.map((v:{coefficient:number})=>v.coefficient)).toEqual([1,-.5,-.5])
    await user.selectOptions(screen.getByRole('combobox',{name:'观点1类型'}),'absolute')
    expect(JSON.parse(screen.getByTestId('payload').textContent!)).not.toHaveProperty('legs')
  })
  it('requires an explicit coverage choice and keeps the approximation acknowledgement separate',async()=>{
    const user=userEvent.setup();render(<UncertaintyEditor/>)
    await user.selectOptions(screen.getByRole('combobox',{name:'均值稳健方式'}),'ellipsoidal')
    expect(screen.getByRole('combobox',{name:'模型覆盖水平'})).toHaveValue('')
    expect(screen.queryByLabelText('均值不确定性惩罚倍数')).not.toBeInTheDocument()
    await user.selectOptions(screen.getByRole('combobox',{name:'模型覆盖水平'}),'90')
    expect(JSON.parse(screen.getByTestId('payload').textContent!)).toMatchObject({uncertainty_confidence:'90',uncertainty_approximation_acknowledged:false})
    await user.click(screen.getByRole('checkbox'))
    await user.selectOptions(screen.getByRole('combobox',{name:'均值稳健方式'}),'box')
    expect(JSON.parse(screen.getByTestId('payload').textContent!)).toMatchObject({uncertainty_confidence:null,uncertainty_approximation_acknowledged:false})
  })
  it.each([{available:false,mode:'single' as const},{available:true,mode:'parameter_average' as const}])('disables unavailable ellipsoids: %o',props=>{
    render(<UncertaintyEditor {...props}/>);expect(screen.getByRole('option',{name:'均值协方差椭球'})).toBeDisabled()
  })
  it('reads frozen coverage without treating it as a return percentile',()=>{
    render(<MeanUncertaintySummary value={{set:'ellipsoidal',confidence:'95',kappa:2.4477,dimension:2,calibration:'conditional_gaussian_mean_ellipsoid',warnings:['仅在模型假设成立时解释'],content_hash:'a'}}/>)
    expect(screen.getByLabelText('已冻结的均值稳健口径')).toHaveTextContent('95%')
    expect(screen.getByLabelText('已冻结的均值稳健口径')).toHaveTextContent('2.4477')
  })
  it('shows undefined transition rows and stationary probabilities as unavailable',()=>{
    render(<LtcmaModelDiagnostics assets={['股票']} names={new Map()} audit={{state_ids:['up','down'],regime:{state_labels:{up:'上涨',down:'下跌'}},
      transition_diagnostics:{matrix:[[1,0],[null,null]],markov_duration_observations:[null,null],stationary_probabilities:[null,null],stationary_status:'missing_outgoing_observations'}}}/>)
    expect(screen.getByRole('table',{name:'历史状态转移诊断'})).toBeVisible()
    expect(screen.getByText(/无法确认唯一平稳分布/)).toBeVisible()
    expect(screen.queryByText('Infinity')).not.toBeInTheDocument()
  })
  it('does not describe an incremental NIW batch as its complete sample',()=>{
    render(<LtcmaModelDiagnostics assets={[]} names={new Map()} audit={{evidence:{sample_horizon:{observation_years:.5,forecast_years:10,scope:'incremental_evidence_batch'}}}}/>)
    expect(screen.getByText(/仅显示 NIW 新增证据批次/)).toBeVisible()
    expect(screen.getByLabelText('样本窗口与预测期限')).toHaveTextContent('0.500')
  })
})
