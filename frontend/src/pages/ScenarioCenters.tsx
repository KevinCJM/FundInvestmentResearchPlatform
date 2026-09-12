import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import HistoricalRegimeWorkbench from './HistoricalRegimeWorkbench'
import PublishedScenarioCenter from './PublishedScenarioCenter'
import GlobalEventLibrary from './regime-workbench/GlobalEventLibrary'

type CenterId = 'historical' | 'simulation' | 'events'

const centers: Array<{ id: CenterId; label: string; description: string }> = [
  {
    id: 'historical',
    label: '历史情景识别',
    description: '用自定义数据、指标与模型拆分历史市场区间',
  },
  {
    id: 'simulation',
    label: '情景模拟与压测',
    description: '对未来冲击、组合损益与战术权重偏移做研究',
  },
  { id: 'events', label: '全球历史事件库', description: '管理事件事实、来源和可复用研究窗口' },
]

export default function ScenarioCenters() {
  const [params, setParams] = useSearchParams()
  const requested = params.get('center')
  const activeCenter: CenterId = requested === 'simulation' || requested === 'events' ? requested : 'historical'
  const [visited, setVisited] = useState<Record<CenterId, boolean>>({ historical: activeCenter === 'historical', simulation: activeCenter === 'simulation', events: activeCenter === 'events' })
  useEffect(() => { setVisited(previous => previous[activeCenter] ? previous : { ...previous, [activeCenter]: true }) }, [activeCenter])
  const activateCenter = (id: CenterId) => {
    setVisited(previous => ({ ...previous, [id]: true }))
    setParams(previous => { const next = new URLSearchParams(previous); next.set('center', id); return next })
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
        <div role="tablist" aria-label="情景研究类型" className="grid gap-2 md:grid-cols-3">
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

      <section role="tabpanel" hidden={activeCenter !== 'historical'} id="scenario-center-panel-historical" aria-labelledby="scenario-center-tab-historical">{visited.historical && <HistoricalRegimeWorkbench />}</section>
      <section role="tabpanel" hidden={activeCenter !== 'events'} id="scenario-center-panel-events" aria-labelledby="scenario-center-tab-events">{visited.events && <GlobalEventLibrary />}</section>
      <section role="tabpanel" hidden={activeCenter !== 'simulation'} id="scenario-center-panel-simulation" aria-labelledby="scenario-center-tab-simulation">{visited.simulation && <PublishedScenarioCenter />}</section>
    </div>
  )
}
