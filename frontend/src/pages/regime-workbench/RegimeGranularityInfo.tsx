import type { RegimeNodeSchema } from '../../services/regimeGraph'

export default function RegimeGranularityInfo({ schema, onExpand, disabled = false, expanding = false }: {
  schema?: RegimeNodeSchema
  onExpand?: () => void
  disabled?: boolean
  expanding?: boolean
}) {
  const metadata = schema?.granularity
  if (!metadata || metadata.kind === 'source' || metadata.kind === 'indicator') return null
  return <section aria-label="算子职责与颗粒度" className="rounded-xl border border-indigo-100 bg-indigo-50/50 p-3 text-xs leading-5">
    <p className="font-semibold text-indigo-950">{metadata.label}</p>
    <p className="mt-1 text-slate-600">{metadata.reason}</p>
    {metadata.steps?.length ? <p className="mt-2 text-indigo-800">{metadata.steps.join(' → ')}</p> : null}
    {metadata.expandable && onExpand && <button type="button" disabled={disabled || expanding} onClick={onExpand} className="mt-3 min-h-10 w-full rounded-lg bg-indigo-600 px-3 font-semibold text-white disabled:cursor-not-allowed disabled:opacity-40">{expanding ? '正在展开…' : '展开为计算步骤'}</button>}
  </section>
}
