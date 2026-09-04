import { useRef, useState, type KeyboardEvent } from 'react'
import HistoricalRegimeDirectory from './HistoricalRegimeDirectory'
import ScenarioSimulationCenter from './ScenarioAlgorithmCenter'

type CenterId = 'historical' | 'simulation'

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
]

export default function ScenarioCenters() {
  const [activeCenter, setActiveCenter] = useState<CenterId>('historical')
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([])

  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
    event.preventDefault()
    const nextIndex = event.key === 'Home'
      ? 0
      : event.key === 'End'
        ? centers.length - 1
        : (index + (event.key === 'ArrowRight' ? 1 : -1) + centers.length) % centers.length
    setActiveCenter(centers[nextIndex].id)
    tabRefs.current[nextIndex]?.focus()
  }

  return (
    <div className="space-y-5" data-testid="scenario-centers">
      <section className="rounded-2xl border border-slate-200 bg-white p-2 shadow-sm" aria-label="情景研究中心切换">
        <div role="tablist" aria-label="情景研究类型" className="grid gap-2 md:grid-cols-2">
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
                onClick={() => setActiveCenter(center.id)}
                onKeyDown={(event) => handleKeyDown(event, index)}
                className={`min-h-20 rounded-xl px-4 py-3 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${active ? 'bg-slate-950 text-white shadow-sm' : 'text-slate-700 hover:bg-slate-50'}`}
              >
                <span className="block text-sm font-bold">{center.label}</span>
                <span className={`mt-1 block text-xs leading-5 ${active ? 'text-slate-300' : 'text-slate-500'}`}>{center.description}</span>
              </button>
            )
          })}
        </div>
      </section>

      <section
        role="tabpanel"
        id={`scenario-center-panel-${activeCenter}`}
        aria-labelledby={`scenario-center-tab-${activeCenter}`}
      >
        {activeCenter === 'historical' ? <HistoricalRegimeDirectory /> : <ScenarioSimulationCenter />}
      </section>
    </div>
  )
}
