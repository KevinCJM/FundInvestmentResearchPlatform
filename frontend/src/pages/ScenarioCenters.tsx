import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import MarketStateResearchCenter from './MarketStateResearchCenter'
import PublishedScenarioCenter from './PublishedScenarioCenter'
import { scenarioAreaFromQuery } from './regime-workbench/regimeStudy'
import GlobalEventCenter from './regime-workbench/GlobalEventCenter'

type CenterId = 'market-state' | 'simulation' | 'events'

const centers: Array<{ id: CenterId; label: string; description: string }> = [
  {
    id: 'market-state',
    label: '市场状态研究',
    description: '定义历史参考、建立实时识别，并验证识别有效性',
  },
  { id: 'events', label: '全球历史事件库', description: '管理历史事件事实、来源和可复用研究窗口' },
  {
    id: 'simulation',
    label: '情景模拟与压测',
    description: '研究未来冲击、资产与组合损益以及战术响应',
  },
]

export default function ScenarioCenters() {
  const [params, setParams] = useSearchParams()
  const activeCenter = scenarioAreaFromQuery(params)
  const [visited, setVisited] = useState<Record<CenterId, boolean>>({ 'market-state': activeCenter === 'market-state', simulation: activeCenter === 'simulation', events: activeCenter === 'events' })
  useEffect(() => { setVisited(previous => previous[activeCenter] ? previous : { ...previous, [activeCenter]: true }) }, [activeCenter])
  const activateCenter = (id: CenterId) => {
    if (id === activeCenter) return
    setVisited(previous => ({ ...previous, [id]: true }))
    setParams(previous => {
      const next = new URLSearchParams(previous)
      next.set('center', id)
      if (id === 'market-state') next.set('stage', 'historical')
      else next.delete('stage')
      // A deep link belongs to one workspace; switching centers keeps drafts,
      // but must not load that target into another workspace.
      for (const key of ['definition', 'revision', 'template', 'mode']) next.delete(key)
      return next
    })
  }
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([])

  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
    event.preventDefault()
    const nextIndex = event.key === 'Home'
      ? 0
      : event.key === 'End'
        ? centers.length - 1
        : (index + (event.key === 'ArrowRight' ? 1 : -1) + centers.length) % centers.length
    activateCenter(centers[nextIndex].id)
    tabRefs.current[nextIndex]?.focus()
  }

  return (
    <div className="space-y-5" data-testid="scenario-centers">
      <section className="rounded-xl border border-slate-200 bg-white p-1" aria-label="情景研究中心切换">
        <div role="tablist" aria-label="情景研究类型" className="grid gap-2 sm:grid-cols-3">
          {centers.map((center, index) => {
            const active = activeCenter === center.id
            return (
              <button
                key={center.id}
                ref={(node) => { tabRefs.current[index] = node }}
                type="button"
                role="tab"
                id={`scenario-center-tab-${center.id}`}
                aria-selected={active}
                aria-controls={`scenario-center-panel-${center.id}`}
                tabIndex={active ? 0 : -1}
                onClick={() => activateCenter(center.id)}
                onKeyDown={(event) => handleKeyDown(event, index)}
                className={`min-h-11 rounded-lg px-4 py-2 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 ${active ? 'bg-slate-950 text-white shadow-sm' : 'text-slate-700 hover:bg-slate-50'}`}
              >
                <span className="block text-sm font-bold">{center.label}</span>
                <span className="sr-only">{center.description}</span>
              </button>
            )
          })}
        </div>
      </section>

      <section role="tabpanel" hidden={activeCenter !== 'market-state'} id="scenario-center-panel-market-state" aria-labelledby="scenario-center-tab-market-state">{visited['market-state'] && <MarketStateResearchCenter />}</section>
      <section role="tabpanel" hidden={activeCenter !== 'events'} id="scenario-center-panel-events" aria-labelledby="scenario-center-tab-events">{visited.events && <GlobalEventCenter />}</section>
      <section role="tabpanel" hidden={activeCenter !== 'simulation'} id="scenario-center-panel-simulation" aria-labelledby="scenario-center-tab-simulation">{visited.simulation && <PublishedScenarioCenter />}</section>
    </div>
  )
}
