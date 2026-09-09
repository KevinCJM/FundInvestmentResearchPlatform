import { describe, expect, it } from 'vitest'
import { blankStep, emptyDefinition, type EtlDefinition } from '../../services/etl'
import { etlGraphSchemas } from '../../test/etlGraphFixtures'
import { asGraph, connectionProblem, connectEtl, duplicateEtlNodes, etlEdges, etlPositions, removeEtlEdges, removeEtlNodes } from './etlGraphAdapter'
import { graphOrder } from '../computation-graph/graph'
import { createTimelineReducer } from '../computation-graph/history'
const plan = (): EtlDefinition => ({ ...emptyDefinition(), steps: [
  {...blankStep('download'),id:'d',name:'来源'},
  {...blankStep('map'),id:'m',name:'映射',inputs:['d']},
  {...blankStep('resolve'),id:'r',name:'取值',inputs:['m']},
] })

describe('ETL 图适配器',()=>{
  it('旧流程按原顺序呈现，虚线不伪装为数据输入',()=>{
    const old=plan(); old.steps.push({...blankStep('download'),id:'d2'})
    const graph=asGraph(old)
    expect(graph.steps[3].after).toEqual(['r'])
    expect(old.graph_version).toBeUndefined()
    expect(graphOrder(graph.steps.map(s=>s.id),etlEdges(graph))).toEqual(['d','m','r','d2'])
    expect(graph.steps[3].inputs).toEqual([])
  })
  it('阻止环、自连、错端口、重复和单输入槽冲突',()=>{
    const graph=asGraph(plan())
    const edge={source:'r',sourcePort:'done',target:'d',targetPort:'after'}
    expect(connectionProblem(graph,edge,etlGraphSchemas)).toMatch(/循环/)
    expect(connectionProblem(graph,{...edge,source:'d'},etlGraphSchemas)).toMatch(/自身/)
    expect(connectionProblem(graph,{source:'d',sourcePort:'data',target:'r',targetPort:'data'},etlGraphSchemas)).toMatch(/类型/)
    expect(connectionProblem(graph,{source:'d',sourcePort:'data',target:'m',targetPort:'data'},etlGraphSchemas)).toMatch(/已经/)
    graph.steps.push({...blankStep('download'),id:'other'})
    expect(connectionProblem(graph,{source:'other',sourcePort:'data',target:'m',targetPort:'data'},etlGraphSchemas)).toMatch(/一个/)
  })
  it('控制连线独立于数据连线，可删除',()=>{
    const graph=asGraph(plan())
    const added=connectEtl(graph,{source:'d',sourcePort:'done',target:'r',targetPort:'after'},etlGraphSchemas)
    expect(added.steps[2].inputs).toEqual(['m'])
    expect(added.steps[2].after).toEqual(['d'])
    const removed=removeEtlEdges(added,['control:d>r'])
    expect(removed.steps[2].after).toEqual([])
    expect(removed.steps[2].inputs).toEqual(['m'])
  })
  it('复制子图重写内部引用，保留外部输入，不共用对象',()=>{
    const graph=asGraph(plan()); const next=duplicateEtlNodes(graph,['m','r'])
    const [m,r]=next.steps.slice(-2)
    expect(m.inputs).toEqual(['d']); expect(r.inputs).toEqual([m.id])
    expect(new Set(next.steps.map(s=>s.id)).size).toBe(5)
    m.params.changed=1; expect(graph.steps[1].params).toEqual({})
    expect(etlPositions(next)[m.id].x).toBe(etlPositions(graph).m.x+44)
  })
  it('删除节点同时清除数据、控制边与位置',()=>{
    const graph=asGraph(plan()); graph.canvas={version:1,positions:etlPositions(graph)}
    graph.steps[2].after=['d']
    const next=removeEtlNodes(graph,['d'])
    expect(next.steps[0].inputs).toEqual([]); expect(next.steps[1].after).toEqual([])
    expect(next.canvas?.positions.d).toBeUndefined()
  })
  it('位置移动不改变运行图，三十一节点自动布局无重合',()=>{
    const graph=asGraph({...emptyDefinition(),steps:Array.from({length:31},(_,i)=>({...blankStep('task'),id:`n${i}`,inputs:i?[`n${i-1}`]:[]}))})
    const positions=etlPositions(graph)
    expect(new Set(Object.values(positions).map(p=>`${p.x}:${p.y}`)).size).toBe(31)
    expect(Math.max(...Object.values(positions).map(p=>p.x))).toBeLessThan(1600)
    const moved={...graph,canvas:{version:1 as const,positions:{...positions,n1:{x:-50,y:80}}}}
    expect(etlEdges(moved)).toEqual(etlEdges(graph))
  })
  it('公共历史记录可撤销、重做、分支编辑和重置',()=>{
    const reducer=createTimelineReducer<EtlDefinition>()
    const start={past:[] as EtlDefinition[],present:plan(),future:[] as EtlDefinition[]}
    const edited=reducer(start,{type:'edit',definition:{...start.present,name:'改名'}})
    const undo=reducer(edited,{type:'undo'})
    expect(undo.present.name).toBe(start.present.name)
    expect(reducer(undo,{type:'redo'}).present.name).toBe('改名')
    expect(reducer(undo,{type:'edit',definition:{...start.present,name:'分支'}}).future).toEqual([])
    expect(reducer(edited,{type:'reset',definition:start.present}).past).toEqual([])
  })
})
