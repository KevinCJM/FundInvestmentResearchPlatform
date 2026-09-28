import { Link } from 'react-router-dom'
import { Field } from '../risk-models/ResearchUI'
import type { CmaDefinition, CmaVersion, MandateDefinition, StrategicCatalog } from '../../services/strategicAllocation'
import { control, linkClass, useLtcmaText } from '../ltcma/shared'
import LtcmaResults from '../ltcma/LtcmaResults'
import { cmaScopeDifference } from '../../services/cmaCompatibility'
import type { ScopeFacts, Usability } from '../../services/ltcmaContract.generated'

export type CmaSelectionContext = { allocationName: string; strategicUniverseId?: string; scopeFacts?: ScopeFacts | null; implementationMappingId?: string; mandate?: MandateDefinition; cutoff: string | null | undefined }
type Choice = Pick<CmaDefinition, 'alloc_name' | 'strategic_universe_id' | 'implementation_mapping_id' | 'currency' | 'as_of' | 'schema_version'> & { retired?: boolean; usable?: Usability; downstream_eligible?: boolean; method?: string; model?: CmaDefinition['model']; scope_facts?: ScopeFacts | null }
export function cmaSelectionReason(choice: Choice, context: CmaSelectionContext): string | null {
  const mandate = context.mandate
  if (choice.retired) return 'retired'
  if (choice.usable && choice.usable.status !== 'ready') return 'handoffNotCurrent'
  if (choice.downstream_eligible === false || choice.method === 'conditional_scenario' || choice.model?.method === 'conditional_scenario') return 'scenarioHandoffBlocked'
  if (!mandate) return 'selectionMissing'
  const scopeIssue = cmaScopeDifference(choice, { strategic_universe_id: context.strategicUniverseId,
    alloc_name: context.allocationName || null, scope_facts: context.scopeFacts })
  if (scopeIssue) return scopeIssue
  if (choice.schema_version !== '2.0' && (choice.implementation_mapping_id ?? '') !== (context.implementationMappingId ?? '')) return 'incompatible'
  if (choice.currency !== mandate.currency) return 'handoffGoalCurrency'
  if (choice.as_of < mandate.as_of || mandate.review_date && choice.as_of >= mandate.review_date) return 'handoffGoalDate'
  if ((mandate.cash_budget || mandate.funding_plan) && mandate.as_of !== choice.as_of) return 'handoffCashDate'
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
    <Field required label={t('selectedCma')}><select required className={control} value={selected?.id ?? ''} disabled={busy || !context.mandate || !context.allocationName && !context.strategicUniverseId} onChange={event => onSelect(event.target.value)}>
      <option value="">{t('choose')}</option>{items.map(item => {
        const reason = cmaSelectionReason(item, context)
        return <option key={item.id} value={item.id} disabled={Boolean(reason)}>{item.name} · {item.as_of} · {item.currency}{reason ? ` · ${t(reason)}` : ''}</option>
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
