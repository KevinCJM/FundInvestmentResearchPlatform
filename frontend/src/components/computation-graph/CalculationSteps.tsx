import type { ReactNode } from 'react'

export interface CalculationStep { id: string; label: string; description?: string; kind?: string }

/** One active step at a time, shared by indicator and scenario construction. */
export default function CalculationSteps({ steps, selectedId, onSelect, children, compact = false }: {
  steps: CalculationStep[]; selectedId: string | null; onSelect: (id: string) => void; children: ReactNode; compact?: boolean
}) {
  return <div className={compact ? 'space-y-3' : 'grid min-w-0 gap-4 md:grid-cols-[minmax(180px,240px)_minmax(0,1fr)]'}>
    <nav aria-label="计算步骤" className="min-w-0 space-y-1">
      {compact ? <label className="block text-xs font-semibold text-slate-600">选择计算步骤<select aria-label="选择计算步骤" value={selectedId || ''} onChange={event => onSelect(event.target.value)} className="mt-1 min-h-10 w-full rounded-xl border border-slate-200 bg-white px-2"><option value="">选择一个步骤</option>{steps.map((step, i) => <option key={step.id} value={step.id}>{i + 1}. {step.label}</option>)}</select></label> : steps.map((step, i) => <button type="button" key={step.id} aria-current={step.id === selectedId ? 'step' : undefined} onClick={() => onSelect(step.id)} className={`block min-h-12 w-full rounded-xl border p-3 text-left ${step.id === selectedId ? 'border-accent-300 bg-accent-50' : 'border-slate-200 bg-white hover:border-accent-200'}`}><span className="block text-sm font-semibold text-slate-800">{i + 1}. {step.label}</span>{step.kind && <span className="mt-1 block text-xs text-slate-600">{step.kind}</span>}</button>)}
      {!steps.length && <p className="rounded-lg border border-dashed border-slate-200 p-4 text-xs text-slate-600">从变量、常量、算子或已有指标开始。</p>}
    </nav>
    <div className="min-w-0">{children}</div>
  </div>
}
