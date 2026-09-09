import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import type { SourceCatalog } from '../../services/dataSources'
import { blankStep } from '../../services/etl'
import EtlDownloadFields from './EtlDownloadFields'
import EtlTaskFields from './EtlTaskFields'
import EtlRunOptionsEditor from './EtlRunOptionsEditor'
import EtlParameterDefinitions from './EtlParameterDefinitions'

const catalog = {
  sources:[{config:{id:'warehouse',name:'用户资料源',enabled:true}}],
  interfaces:[{config:{id:'warehouse.record',source_id:'warehouse',name:'业务记录',params:{},enabled:true,mappings:[],source_fields:[{name:'symbol'},{name:'ts_code'}]},revision:1,request_fields:[{name:'account_ref',label:'外部账户编号',data_type:'text'},{name:'cutoff',label:'统计日期',data_type:'date',date_format:'iso'}]}],
  etl_tasks:[{id:'warehouse.orders',name:'全量订单',category:'订单数据',source_ids:['warehouse'],requires_source:true,description:'从数据集读取全部订单',parameters:[{name:'cutoff',label:'统计日期',data_type:'date'}]}],
  targets:{tables:[],categories:[]},
} as unknown as SourceCatalog

describe('通用合同驱动表单', () => {
  it('未知供应商的请求字段直接渲染，不推测响应中的产品代码', () => {
    const onChange = vi.fn()
    render(<MemoryRouter><EtlDownloadFields catalog={catalog} step={{...blankStep('download'),source_id:'warehouse',interface_id:'warehouse.record'}} onChange={onChange} /></MemoryRouter>)
    expect(screen.getByLabelText('外部账户编号')).toBeVisible()
    expect(screen.getByLabelText('统计日期')).toBeVisible()
    expect(screen.queryByText(/产品代码/)).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('外部账户编号'),{target:{value:'ACCOUNT-100'}})
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({params:{account_ref:'ACCOUNT-100'}}))
  })
  it('未登记请求字段只显示通用参数编辑，不猜测日期或代码', () => {
    const value = {...catalog,interfaces:catalog.interfaces.map(item=>({...item,request_fields:[]}))}
    render(<MemoryRouter><EtlDownloadFields catalog={value} step={{...blankStep('download'),source_id:'warehouse',interface_id:'warehouse.record'}} onChange={()=>{}} /></MemoryRouter>)
    expect(screen.getByText(/尚未声明请求字段/)).toBeInTheDocument()
    expect(screen.queryByLabelText('开始日期')).not.toBeInTheDocument()
  })
  it('数据集节点由目录生成，换任务和参数不需要修改组件', () => {
    const onChange = vi.fn()
    render(<EtlTaskFields catalog={catalog} step={{...blankStep('task'),task_id:'warehouse.orders',source_id:'warehouse'}} parameters={[{id:'report_day',label:'报表日',data_type:'date',required:true,default:'',date_format:'compact',description:''}]} onChange={onChange} />)
    expect(screen.getByRole('option',{name:'全量订单'})).toBeInTheDocument()
    expect(screen.getByLabelText('统计日期')).toBeVisible()
    fireEvent.change(screen.getByLabelText('统计日期取值方式'),{target:{value:'report_day'}})
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({parameter_bindings:{cutoff:'report_day'}}))
  })
  it('运行参数来自任意流程定义，不绑定 ETF 或基金标识', () => {
    render(<EtlRunOptionsEditor definition={{name:'任意流程',description:'',max_runtime_seconds:60,steps:[],parameters:[{id:'report_day',label:'报表截止日',data_type:'date',default:'',required:true,date_format:'iso',description:''}]}} value={{mode:'full',parameters:{}}} disabled={false} onChange={()=>{}} />)
    expect(screen.getByLabelText('报表截止日')).toBeVisible()
    expect(screen.queryByLabelText('ETF 代码')).not.toBeInTheDocument()
  })
  it('用户可以通过表单增加参数，而不是只能使用特定模板', () => {
    const onChange=vi.fn()
    render(<EtlParameterDefinitions value={[]} onChange={onChange} />)
    fireEvent.click(screen.getByText('定义运行参数 · 0 项'))
    fireEvent.click(screen.getByRole('button',{name:'添加运行参数'}))
    expect(onChange).toHaveBeenCalledWith([expect.objectContaining({id:'param_1',data_type:'text'})])
  })
})
