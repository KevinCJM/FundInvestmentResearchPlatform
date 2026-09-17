import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import HistoricalRegimeWorkbench from './HistoricalRegimeWorkbench'
import type { RegimeStudy } from '../services/regimeGraph'
import { marketStateStageFromQuery, type MarketStateStage } from './regime-workbench/regimeStudy'

const steps: Array<{ id: MarketStateStage; index: string; title: string; description: string }> = [
  { id: 'historical', index: '1', title: '定义历史参考', description: '用完整历史数据明确“什么叫这个市场状态”。' },
  { id: 'realtime', index: '2', title: '建立实时识别', description: '只使用当时可得信息，识别同一组状态。' },
  { id: 'validation', index: '3', title: '验证识别能力', description: '绑定固定历史参考，校准并检查留出与前瞻证据。' },
]

export default function MarketStateResearchCenter() {
  const [params, setParams] = useSearchParams()
  const stage = marketStateStageFromQuery(params)
  const [visited, setVisited] = useState({ historical: stage === 'historical', realtime: stage !== 'historical' })
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([])
  const [reference, setReference] = useState<RegimeStudy['reference']>()

  useEffect(() => {
    setVisited(current => stage === 'historical'
      ? current.historical ? current : { ...current, historical: true }
      : current.realtime ? current : { ...current, realtime: true })
  }, [stage])

  const activate = (nextStage: MarketStateStage) => {
    setVisited(current => nextStage === 'historical' ? { ...current, historical: true } : { ...current, realtime: true })
    setParams(previous => {
      const next = new URLSearchParams(previous)
      const previousStage = marketStateStageFromQuery(previous)
      next.set('center', 'market-state')
      next.set('stage', nextStage)
      next.delete('mode')
      // Historical and realtime are different immutable definitions. Do not
      // accidentally carry one workspace's exact version into the other.
      if ((previousStage === 'historical') !== (nextStage === 'historical')) {
        for (const key of ['definition', 'revision', 'template']) next.delete(key)
      }
      return next
    })
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
    event.preventDefault()
    const nextIndex = event.key === 'Home' ? 0 : event.key === 'End' ? steps.length - 1
      : (index + (event.key === 'ArrowRight' ? 1 : -1) + steps.length) % steps.length
    activate(steps[nextIndex].id)
    tabRefs.current[nextIndex]?.focus()
  }

  return <section className="min-w-0" aria-label="市场状态研究">
    <div className="mb-5 rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
      <div className="mb-3 px-1">
        <h2 className="text-lg font-semibold text-slate-900">市场状态研究</h2>
        <p className="mt-1 text-sm text-slate-600">先用事后方法定义参考状态，再建立可实时运行的识别模型，最后验证它是否真的能识别同一组状态。</p>
      </div>
      <div role="tablist" aria-label="市场状态研究步骤" className="grid gap-2 md:grid-cols-3">
        {steps.map((item, index) => {
          const active = stage === item.id
          return <button
            key={item.id}
            ref={node => { tabRefs.current[index] = node }}
            type="button"
            role="tab"
            aria-selected={active}
            aria-controls={item.id === 'historical' ? 'market-state-stage-historical' : 'market-state-stage-realtime'}
            tabIndex={active ? 0 : -1}
            onClick={() => activate(item.id)}
            onKeyDown={event => handleKeyDown(event, index)}
            className={`min-h-16 rounded-lg border px-4 py-3 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 ${active ? 'border-accent-600 bg-accent-50 text-accent-900' : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50'}`}
          >
            <span className="flex items-center gap-2 text-sm font-semibold"><span className={`grid h-6 w-6 shrink-0 place-items-center rounded-full text-xs ${active ? 'bg-accent-600 text-white' : 'bg-slate-100 text-slate-700'}`}>{item.index}</span>{item.title}</span>
            <span className="mt-1 block text-xs leading-5 text-slate-600">{item.description}</span>
          </button>
        })}
      </div>
      <p className="mt-3 px-1 text-xs leading-5 text-slate-600">历史参考、实时模型和验证报告仍分别保存为不可变版本；这里仅把用户操作流程连在一起。</p>
    </div>

    <section id="market-state-stage-historical" role="tabpanel" hidden={stage !== 'historical'}>
      {visited.historical && <HistoricalRegimeWorkbench purpose="historical_reference" active={stage === 'historical'} onReferenceReady={setReference} onNextResearchStep={() => activate('realtime')} />}
    </section>
    <section id="market-state-stage-realtime" role="tabpanel" hidden={stage === 'historical'}>
      {visited.realtime && <HistoricalRegimeWorkbench
        purpose="realtime_recognition"
        incomingReference={reference}
        active={stage !== 'historical'}
        taskFocus={stage === 'validation' ? 'validation' : 'authoring'}
        onHistoricalReference={() => activate('historical')}
        onNextResearchStep={() => activate('validation')}
      />}
    </section>
  </section>
}
