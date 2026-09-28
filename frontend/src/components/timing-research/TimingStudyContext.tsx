import { systemText, useI18n } from '../../i18n/runtime'
import type { TimingAdaptation, TimingBaskets, TimingDefinition } from '../../services/timingResearch'
import { timingField } from './TimingRuleEditor'

export type BasketDraft = { market: string; category: string }
export function basketCodes(text: string): string[] { return text.split(/[,，;；\s]+/).map(code => code.trim().toUpperCase()).filter(Boolean) }
export function basketsFromDraft(draft: BasketDraft): TimingBaskets { return { market: basketCodes(draft.market), category: basketCodes(draft.category) } }
export function basketValidation(definition: TimingDefinition, draft: BasketDraft): string {
  for (const group of ['market', 'category'] as const) {
    const codes = basketCodes(draft[group]), label = group === 'market' ? systemText('preInvestment.timingStudyContext.marketReferenceBasket') : systemText('preInvestment.timingStudyContext.peerReferenceBasket')
    if (codes.some(code => !/^\d{6}\.(SH|SZ)$/.test(code))) return systemText('preInvestment.timingStudyContext.codesMustUseFormatsSuchAs510300', { p0: label })
    if (new Set(codes).size !== codes.length) return systemText('preInvestment.timingStudyContext.containsDuplicateCodesRemoveDuplicates', { p0: label })
    if (codes.length > 12 || codes.length === 1) return systemText('preInvestment.timingStudyContext.requires212EtfsLeaveBlankIf', { p0: label })
    const required = definition.nodes.some(node => node.op === 'basket_source' && (node.parameters.group || 'market') === group)
    if (required && codes.length < 2) return systemText('preInvestment.timingStudyContext.thisAlgorithmRequiresExplicitlyProvideAtLeast', { p0: label })
  }
  return ''
}
export function TimingBasketEditor({ value, onChange, definition }: { value: BasketDraft; onChange: (value: BasketDraft) => void; definition: TimingDefinition }) {
  useI18n()
  const required = definition.nodes.some(node => node.op === 'basket_source')
  return <details open={required || undefined} className="border-t border-slate-100 pt-3"><summary className="cursor-pointer py-2 text-sm font-medium text-slate-700">{systemText('preInvestment.timingStudyContext.referenceEtfBaskets')}{required ? " " + systemText('preInvestment.timingStudyContext.requiredByCurrentAlgorithm') : " " + systemText('preInvestment.timingStudyContext.optional')}</summary><p className="mt-2 text-xs leading-5 text-slate-600">{systemText('preInvestment.timingStudyContext.usedOnlyForBreadthAndCrossSectional')}</p><div className="mt-3 space-y-3">{(['market', 'category'] as const).map(group => <label key={group} className="block text-xs text-slate-600">{group === 'market' ? systemText('preInvestment.timingStudyContext.marketReferenceBasketCodes') : systemText('preInvestment.timingStudyContext.peerReferenceBasketCodes')}<textarea aria-label={group === 'market' ? systemText('preInvestment.timingStudyContext.marketReferenceBasketCodes') : systemText('preInvestment.timingStudyContext.peerReferenceBasketCodes')} className={timingField} rows={3} spellCheck={false} placeholder={systemText('preInvestment.timingStudyContext.separateWithCommasOrNewlinesForExample')} value={value[group]} onChange={event => onChange({ ...value, [group]: event.target.value })} /><span className="mt-1 block text-xs text-slate-600">{systemText('preInvestment.timingStudyContext.entered') + " "}{basketCodes(value[group]).length} {" " + systemText('preInvestment.timingStudyContext.items212EtfsPerBasket')}</span></label>)}</div><p className="mt-2 text-xs leading-5 text-slate-600">{systemText('preInvestment.timingStudyContext.noAutomaticDateIntersectionOrZeroFilling')}</p></details>
}
export function TimingAdaptationNote({ adaptation }: { adaptation?: TimingAdaptation }) {
  useI18n()
  if (!adaptation) return null
  return <details className="rounded-xl border border-amber-200 bg-amber-50/50 p-3"><summary className="cursor-pointer text-sm font-medium text-amber-900">{systemText('preInvestment.timingStudyContext.etfAdaptationNotes') + " "}{adaptation.source_experiments.join(' / ')}</summary><p className="mt-2 text-xs leading-5 text-amber-900">{adaptation.version} {" " + systemText('preInvestment.timingStudyContext.isANewEtfStudyNotA')}</p><div className="mt-3 grid gap-3 sm:grid-cols-2"><div><h3 className="text-xs font-semibold text-slate-800">{systemText('preInvestment.timingStudyContext.retainedIdeas')}</h3><ul className="mt-2 list-disc space-y-1 pl-4 text-xs leading-5 text-slate-600">{adaptation.preserved.map((item, index) => <li key={index}>{item}</li>)}</ul></div><div><h3 className="text-xs font-semibold text-slate-800">{systemText('preInvestment.timingStudyContext.adaptationDifferences')}</h3><ul className="mt-2 list-disc space-y-1 pl-4 text-xs leading-5 text-slate-600">{adaptation.changed.map((item, index) => <li key={index}>{item}</li>)}</ul></div></div></details>
}
