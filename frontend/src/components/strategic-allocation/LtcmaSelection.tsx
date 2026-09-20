import { Link } from 'react-router-dom'
import { Field } from '../risk-models/ResearchUI'
import type { CmaDefinition, CmaVersion, MandateDefinition, StrategicCatalog } from '../../services/strategicAllocation'
import { control, linkClass, useLtcmaText } from '../ltcma/shared'
import LtcmaResults from '../ltcma/LtcmaResults'

export type CmaSelectionContext = { allocationName: string; strategicUniverseId?: string; implementationMappingId?: string; mandate?: MandateDefinition; cutoff: string | null | undefined }
type Choice = Pick<CmaDefinition, 'alloc_name' | 'strategic_universe_id' | 'implementation_mapping_id' | 'currency' | 'horizon_years' | 'as_of' | 'schema_version'> & { retired?: boolean }
export function cmaSelectionReason(choice: Choice, context: CmaSelectionContext): string | null {
  const mandate = context.mandate
  if (choice.retired) return 'retired'
  if (!mandate) return 'selectionMissing'
  if (context.strategicUniverseId ? choice.strategic_universe_id !== context.strategicUniverseId
    : choice.alloc_name !== context.allocationName || Boolean(choice.strategic_universe_id)) return 'incompatible'
  if (choice.schema_version !== '2.0' && (choice.implementation_mapping_id ?? '') !== (context.implementationMappingId ?? '')) return 'incompatible'
  if (choice.currency !== mandate.currency || choice.horizon_years !== mandate.horizon_years) return 'incompatible'
  if (choice.as_of < mandate.as_of || mandate.review_date && choice.as_of >= mandate.review_date) return 'incompatible'
  if ((mandate.cash_budget || mandate.funding_plan) && mandate.as_of !== choice.as_of) return 'incompatible'
  if (context.cutoff && choice.as_of > context.cutoff) return 'clockInvalid'
  return null
}

export default function LtcmaSelection({ items, selected, context, onSelect, busy, mandateId }: {
  items: StrategicCatalog['assumptions']; selected: CmaVersion | null; context: CmaSelectionContext
  onSelect: (id: string) => void; busy: boolean; mandateId: string
}) {
  const { t } = useLtcmaText()
  const query = new URLSearchParams()
  if (context.strategicUniverseId) query.set('strategic_universe', context.strategicUniverseId)
  else if (context.allocationName) query.set('alloc', context.allocationName)
  if (mandateId) query.set('mandate', mandateId)
  return <section className="min-w-0 space-y-4">
    <p className="text-sm leading-6 text-slate-600">{t('saaSelectionHint')}</p>
    <Field label={t('selectedCma')}><select className={control} value={selected?.id ?? ''} disabled={busy || !context.mandate || !context.allocationName && !context.strategicUniverseId} onChange={event => onSelect(event.target.value)}>
      <option value="">{t('choose')}</option>{items.map(item => {
        const reason = cmaSelectionReason(item, context)
        return <option key={item.id} value={item.id} disabled={Boolean(reason)}>{item.name} · {item.as_of} · {item.currency} · {t('horizonValue', { years: item.horizon_years })}{reason ? ` · ${t(reason)}` : ''}</option>
      })}
      {selected && !items.some(item => item.id === selected.id) && <option value={selected.id}>{selected.name}</option>}
    </select></Field>
    <div className="flex flex-wrap gap-3"><Link className={linkClass} to={`/pre-investment/ltcma/new?${query}`}>{t('new')}</Link><Link className={linkClass} to="/pre-investment/ltcma">{t('openCenter')}</Link>
      {selected && <><Link className={linkClass} to={`/pre-investment/ltcma/${encodeURIComponent(selected.id)}`}>{t('view')}</Link><Link className={linkClass} to={`/pre-investment/ltcma/new?copy=${encodeURIComponent(selected.id)}&mandate=${encodeURIComponent(mandateId)}`}>{t('copy')}</Link></>}
    </div>
    {!selected && <p className="text-sm text-slate-600">{t('selectionMissing')}</p>}
    {selected && <details><summary className="min-h-10 cursor-pointer text-sm font-medium">{selected.name} · {t('results')}</summary><LtcmaResults value={selected} /></details>}
  </section>
}
