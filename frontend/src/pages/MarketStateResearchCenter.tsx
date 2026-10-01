import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import ScenarioAssistant from '../integrations/portable-agent/ScenarioAssistant'
import { mergeRegimeAgentDraft } from '../services/regimeAgent'
import HistoricalRegimeWorkbench from './HistoricalRegimeWorkbench'
import RegimeStudyList from './regime-workbench/RegimeStudyList'
import { actionClass } from '../components/ui'
import type { RegimeGraphDefinition, RegimeStudy } from '../services/regimeGraph'
import { marketStateStageFromQuery, type MarketStateStage } from './regime-workbench/regimeStudy'

const steps: Array<{ id: MarketStateStage; index: string; title: string; description: string }> = [
  { id: 'historical', index: '1', title: '定义历史参考', description: '用完整历史数据明确“什么叫这个市场状态”。' },
  { id: 'realtime', index: '2', title: '建立实时识别', description: '先选固定历史参考，再用当时可得信息识别状态。' },
  { id: 'validation', index: '3', title: '验证识别能力与应用', description: '检查准确性、校准与前瞻证据，再判断可用范围。' },
]

export default function MarketStateResearchCenter({ active = true }: { active?: boolean }) {
  const [agentSeeds, setAgentSeeds] = useState<Partial<Record<'historical' | 'realtime', { definition: RegimeGraphDefinition; serial: number }>>>({})
  const [params, setParams] = useSearchParams()
  const stage = marketStateStageFromQuery(params)
  // 一条路由两种页面：地址栏带上研究身份才进工作台，否则停在该步骤的已保存清单。
  const editing = Boolean(params.get('definition') || params.get('template') || params.get('new'))
  const [opened, setOpened] = useState({ historical: editing && stage === 'historical', realtime: editing && stage !== 'historical' })
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([])
  const [reference, setReference] = useState<RegimeStudy['reference']>()

  useEffect(() => {
    if (!editing) return
    setOpened(current => stage === 'historical'
      ? current.historical ? current : { ...current, historical: true }
      : current.realtime ? current : { ...current, realtime: true })
  }, [stage, editing])

  /**
   * `keep` 沿用同侧的精确版本（实时↔验证是同一份草稿），`list` 回到清单，`new` 开一份空白研究。
   * 历史与实时是两份不可变定义，跨步骤一律不把一边的版本带进另一边。
   */
  const stageSearch = (nextStage: MarketStateStage, intent: 'keep' | 'list' | 'new', extra?: Record<string, string>) => {
    const next = new URLSearchParams(params)
    const crossing = (marketStateStageFromQuery(params) === 'historical') !== (nextStage === 'historical')
    next.set('center', 'market-state')
    next.set('stage', nextStage)
    next.delete('mode')
    if (intent !== 'keep' || crossing) for (const key of ['definition', 'revision', 'template', 'new']) next.delete(key)
    if (intent === 'new') next.set('new', '1')
    for (const [key, value] of Object.entries(extra ?? {})) next.set(key, value)
    return `?${next}`
  }

  const activate = (nextStage: MarketStateStage, intent: 'keep' | 'list' | 'new' = 'keep') => {
    if (intent === 'new') setOpened(current => ({ ...current, [nextStage === 'historical' ? 'historical' : 'realtime']: true }))
    setParams(new URLSearchParams(stageSearch(nextStage, intent).slice(1)))
  }

  const openStudy = (nextStage: MarketStateStage) => (definition: RegimeGraphDefinition) =>
    stageSearch(nextStage, 'list', { definition: definition.id ?? '', revision: String(definition.revision ?? 1) })

  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
    event.preventDefault()
    const nextIndex = event.key === 'Home' ? 0 : event.key === 'End' ? steps.length - 1
      : (index + (event.key === 'ArrowRight' ? 1 : -1) + steps.length) % steps.length
    activate(steps[nextIndex].id)
    tabRefs.current[nextIndex]?.focus()
  }

  const backToList = (nextStage: MarketStateStage, label: string) => <div className="mb-4">
    <Link className={actionClass()} to={{ search: stageSearch(nextStage, 'list') }}>返回{label}清单</Link>
  </div>

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
      <p className="mt-3 px-1 text-xs leading-5 text-slate-600">历史参考可独立用于复盘，也可供多个实时模型比较。参考或模型变化后，验证结果按新版本重新生成。</p>
    </div>

    <section id="market-state-stage-historical" role="tabpanel" hidden={stage !== 'historical'}>
      {!editing && stage === 'historical' && <RegimeStudyList
        stage="historical"
        studyHref={openStudy('historical')}
        newHref={stageSearch('historical', 'new')}
      />}
      {opened.historical && <div hidden={!editing}>
        {backToList('historical', '历史参考')}
        <HistoricalRegimeWorkbench
          key={`historical-${agentSeeds.historical?.serial || 0}`}
          initialDefinition={agentSeeds.historical?.definition}
          purpose="historical_reference"
          active={active && stage === 'historical' && editing}
          onReferenceReady={setReference}
          onNextResearchStep={() => activate('realtime', 'new')}
        />
      </div>}
    </section>
    <section id="market-state-stage-realtime" role="tabpanel" hidden={stage === 'historical'}>
      {!editing && stage !== 'historical' && <RegimeStudyList
        stage={stage === 'validation' ? 'validation' : 'realtime'}
        studyHref={openStudy(stage === 'validation' ? 'validation' : 'realtime')}
        newHref={stage === 'validation' ? undefined : stageSearch('realtime', 'new')}
      />}
      {opened.realtime && <div hidden={!editing}>
        {backToList(stage === 'validation' ? 'validation' : 'realtime', stage === 'validation' ? '待验证模型' : '实时识别模型')}
        <HistoricalRegimeWorkbench
          key={`realtime-${agentSeeds.realtime?.serial || 0}`}
          initialDefinition={agentSeeds.realtime?.definition}
          purpose="realtime_recognition"
          incomingReference={reference}
          active={active && stage !== 'historical' && editing}
          taskFocus={stage === 'validation' ? 'validation' : 'authoring'}
          onHistoricalReference={() => activate('historical', 'list')}
          onNextResearchStep={() => activate('validation')}
        />
      </div>}
    </section>
    <ScenarioAssistant page="historical-regimes" workspace="graph" purpose={stage === 'historical' ? 'historical_reference' : 'realtime_recognition'}
      active={active && !editing} mode={stage === 'historical' ? 'retrospective' : 'realtime'}
      onApply={proposal => {
        const definition = mergeRegimeAgentDraft(undefined, proposal)
        const side = stage === 'historical' ? 'historical' : 'realtime'
        setAgentSeeds(previous => ({ ...previous, [side]: { definition, serial: (previous[side]?.serial || 0) + 1 } }))
        activate(stage, 'new')
      }} />
  </section>
}
