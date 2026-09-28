import { AdjustmentsHorizontalIcon, ArrowDownIcon, ArrowRightIcon, ChartPieIcon, ChevronDownIcon, ClipboardDocumentCheckIcon } from '@heroicons/react/24/outline'
import { Link } from 'react-router-dom'
import type { StageDefinition } from '../app/processRegistry'
import { actionClass, Badge, Card } from '../components/ui'
import { useI18n } from '../i18n/runtime'

// Group the registered nodes without introducing another route or progress model.
const phases = [
  { id: 'foundation', icon: AdjustmentsHorizontalIcon, nodes: ['objectives', 'product-pool', 'ltcma'] },
  { id: 'allocation', icon: ChartPieIcon, nodes: ['saa', 'taa', 'product-allocation-timing'] },
  { id: 'review', icon: ClipboardDocumentCheckIcon, nodes: ['portfolio-synthesis', 'validation', 'approval'] },
] as const

export default function PreInvestmentOverview({ stage }: { stage: StageDefinition }) {
  const { s } = useI18n()
  const text = (key: string) => s(`preInvestment.overview.${key}`)
  const nodeById = (id: string) => stage.nodes.find(node => node.id === id)!
  const numberFor = (id: string) => String(stage.nodes.indexOf(nodeById(id)) + 1).padStart(2, '0')

  return <div className="space-y-5" data-testid="pre-investment-overview">
    <div className="flex flex-col gap-4 py-1 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
      <div className="min-w-0 border-l-4 border-accent-600 pl-4">
        <h2 className="text-xl font-semibold text-slate-950">{text('title')}</h2>
        <p className="mt-2 text-sm leading-6 text-slate-600">{text('description')}</p>
      </div>
      <div className="flex flex-wrap gap-2">
        <Link to={nodeById('objectives').path} className={actionClass('primary', 'motion-reduce:transition-none')}>
          {text('start')}<ArrowRightIcon aria-hidden="true" className="h-4 w-4 shrink-0" />
        </Link>
        <Link to={nodeById('saa').path} className={actionClass('secondary', 'motion-reduce:transition-none')}>{text('saved')}</Link>
      </div>
    </div>

    <Card className="!p-0">
      <ol aria-label={text('flowLabel')} className="divide-y divide-slate-200">
        {phases.map((phase, phaseIndex) => <li key={phase.id} className="p-4 sm:p-5">
          <section aria-labelledby={`research-phase-${phase.id}`}>
            <header className="mb-3 flex items-start gap-3">
              <span aria-hidden="true" className={`grid h-10 w-10 shrink-0 place-items-center rounded-lg ${stage.accent.soft} ${stage.accent.text}`}><phase.icon className="h-5 w-5" /></span>
              <div className="min-w-0 flex-1">
                <h3 id={`research-phase-${phase.id}`} className="text-lg font-semibold text-slate-950">{text(`${phase.id}.title`)}</h3>
                <p className="mt-1 text-sm leading-5 text-slate-600">{text(`${phase.id}.purpose`)}</p>
              </div>
              <span aria-hidden="true" className="hidden pt-1 text-xs font-medium tabular-nums text-slate-600 sm:block">{numberFor(phase.nodes[0])} – {numberFor(phase.nodes[2])}</span>
            </header>
            <ol className="grid divide-y divide-slate-200 md:grid-cols-3 md:divide-x md:divide-y-0">
              {phase.nodes.map(id => {
                const node = nodeById(id)
                const number = numberFor(id)
                return <li key={id} className="min-w-0">
                  <Link to={node.path} aria-labelledby={`research-step-${id}-title`} aria-describedby={`research-step-${id}-description${id === 'taa' ? ' research-step-taa-optional' : ''}`} className="group flex h-full min-h-10 flex-col rounded-lg p-3 hover:bg-accent-50 focus:outline-none focus-visible:bg-accent-50 focus-visible:ring-2 focus-visible:ring-accent-500 sm:p-4">
                    <div className="mb-3 flex items-center gap-3">
                      <span aria-hidden="true" className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-slate-100 text-xs font-semibold tabular-nums text-slate-700 group-hover:bg-accent-100 group-hover:text-accent-800">{number}</span>
                      {id === 'taa' && <span id="research-step-taa-optional"><Badge>{text('optional')}</Badge></span>}
                    </div>
                    <h4 id={`research-step-${id}-title`} className="text-sm font-semibold leading-6 text-slate-950 group-hover:text-accent-800"><span className="sr-only">{number} </span>{node.label}</h4>
                    <div className="mt-2 flex flex-1 items-start gap-3">
                      <p id={`research-step-${id}-description`} className="flex-1 text-sm leading-6 text-slate-600">{text(`${id}.action`)}</p>
                      <ArrowRightIcon aria-hidden="true" className="mb-1 h-4 w-4 shrink-0 self-end text-accent-700" />
                    </div>
                  </Link>
                </li>
              })}
            </ol>
            {phase.id === 'foundation' && <details className="group/paths mt-3 border-t border-slate-100 pt-1">
              <summary className="flex min-h-10 cursor-pointer list-none items-center justify-between gap-3 rounded-lg px-3 text-sm font-medium text-accent-700 hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 [&::-webkit-details-marker]:hidden"><span>{text('paths.title')}</span><ChevronDownIcon aria-hidden="true" className="h-4 w-4 shrink-0 group-open/paths:rotate-180" /></summary>
              <div className="grid gap-4 px-3 pb-3 pt-2 sm:grid-cols-2">
                {(['strategy', 'product'] as const).map(path => <div key={path} className="border-l-2 border-accent-200 pl-3">
                  <h4 className="text-sm font-semibold text-slate-900">{text(`paths.${path}.title`)}</h4>
                  <p className="mt-1 text-sm leading-6 text-slate-600">{text(`paths.${path}.description`)}</p>
                </div>)}
              </div>
            </details>}
            <p className="mt-3 flex items-start gap-2 rounded-lg bg-slate-50 px-3 py-2.5 text-xs leading-5 text-slate-600">
              {phaseIndex < phases.length - 1 && <ArrowDownIcon aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" />}
              {text(`${phase.id}.handoff`)}
            </p>
          </section>
        </li>)}
      </ol>
    </Card>
    <p className="text-xs leading-5 text-slate-600">{text('navigationHint')}</p>
  </div>
}
