import { useEffect, useMemo, useRef, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsType } from 'echarts'
import { Button } from '../ui'
import { useI18n } from '../../i18n/runtime'

const fullRange = [{ start: 0, end: 100 }, { start: 0, end: 100 }]

export default function FrontierChart({ option, theme, height, description, testId }: {
  option: Record<string, unknown>; theme?: string; height: number; description: string; testId: string
}) {
  const { s } = useI18n()
  const chart = useRef<ReactECharts>(null), container = useRef<HTMLDivElement>(null)
  const [selecting, setSelecting] = useState(false)
  const [range, setRange] = useState(fullRange)
  const events = useMemo(() => ({ datazoom: (_event: unknown, instance: EChartsType) => {
    const zoom = instance.getOption().dataZoom as Array<{ start: number; end: number }>
    setRange(zoom.map(({ start, end }) => ({ start, end })))
    setSelecting(false)
  } }), [])
  useEffect(() => {
    chart.current?.getEchartsInstance().dispatchAction({ type: 'takeGlobalCursor', key: 'dataZoomSelect', dataZoomSelectActive: selecting })
    if (!selecting) return
    const cancel = (event: KeyboardEvent) => { if (event.key === 'Escape') setSelecting(false) }
    window.addEventListener('keydown', cancel)
    return () => window.removeEventListener('keydown', cancel)
  }, [selecting, option, range])
  useEffect(() => {
    if (!container.current || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(() => chart.current?.getEchartsInstance().resize())
    observer.observe(container.current)
    return () => observer.disconnect()
  }, [])
  return <div className="min-w-0 space-y-2">
    <div className="flex flex-wrap items-center gap-2">
      <Button aria-pressed={selecting} tone={selecting ? 'primary' : 'secondary'} onClick={() => setSelecting(value => !value)}>{s('frontierZoom.select')}</Button>
      <Button onClick={() => { setRange(fullRange); setSelecting(false) }}>{s('frontierZoom.reset')}</Button>
      <p className="text-xs leading-5 text-slate-600">{s(selecting ? 'frontierZoom.active' : 'frontierZoom.help')}</p>
    </div>
    <div ref={container} className="w-full min-w-0" role="img" aria-label={description} data-testid={testId}>
      <ReactECharts ref={chart} theme={theme} notMerge onEvents={events} style={{ height, width: '100%' }} option={{ ...option,
        // Keep the native brush controller; accessible HTML buttons replace its canvas icons.
        toolbox: { itemSize: 0, itemGap: 0, showTitle: false, feature: { dataZoom: { xAxisIndex: 0, yAxisIndex: 0, filterMode: 'none' } } },
        dataZoom: [
          { id: 'frontier-x', type: 'select', xAxisIndex: 0, filterMode: 'none', ...range[0] },
          { id: 'frontier-y', type: 'select', yAxisIndex: 0, filterMode: 'none', ...range[1] },
        ],
      }} />
    </div>
  </div>
}
