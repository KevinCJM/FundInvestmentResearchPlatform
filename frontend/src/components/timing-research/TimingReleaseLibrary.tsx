import { systemText, useI18n } from '../../i18n/runtime'
import { useEffect, useRef, useState } from 'react'
import { timingApi, type TimingRelease } from '../../services/timingResearch'
import { timingField } from './TimingRuleEditor'

export default function TimingReleaseLibrary({ onView, disabled }: { onView: (runId: string) => void; disabled?: boolean }) {
  useI18n()
  const [releases, setReleases] = useState<TimingRelease[] | null>(null)
  const [selectedId, setSelectedId] = useState('')
  const [note, setNote] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [binding, setBinding] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const mounted = useRef(true)
  const selected = releases?.find(item => item.id === selectedId) || releases?.[0]
  useEffect(() => { mounted.current = true; return () => { mounted.current = false } }, [])
  useEffect(() => {
    const controller = new AbortController()
    setError(''); setReleases(null)
    void timingApi.releases(controller.signal).then(result => {
      if (!controller.signal.aborted) setReleases(result.items)
    }).catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : systemText('preInvestment.timingReleaseLibrary.unableToLoadResearchVersions')) })
    return () => controller.abort()
  }, [attempt])
  const bind = async () => {
    if (!selected) return
    setBinding(true); setError(''); setNotice('')
    try {
      await timingApi.bind(selected.id, 'pre_investment', note)
      if (mounted.current) setNotice(systemText('preInvestment.timingReleaseLibrary.referencedInPreInvestmentResearchPortfolioWeights', { p0: selected.name || systemText('preInvestment.timingReleaseLibrary.selectedResearchVersion') }))
    } catch (reason) { if (mounted.current) setError(reason instanceof Error ? reason.message : systemText('preInvestment.timingReleaseLibrary.unableToReferenceTheResearchVersion')) }
    finally { if (mounted.current) setBinding(false) }
  }
  return <section aria-label={systemText('preInvestment.timingReleaseLibrary.existingResearchVersions')} className="space-y-4">
    <div><h2 className="text-base font-semibold text-slate-900">{systemText('preInvestment.timingReleaseLibrary.selectATestedResearchVersion')}</h2><p className="mt-2 text-sm leading-6 text-slate-600">{systemText('preInvestment.timingReleaseLibrary.reviewApplicableProductsAndResultsThenReference')}</p></div>
    {error && <div role="alert" className="rounded-lg bg-rose-50 p-3 text-sm text-rose-700">{error}{!releases && <button type="button" className="ml-3 min-h-9 underline" onClick={() => setAttempt(value => value + 1)}>{systemText('preInvestment.timingReleaseLibrary.reload')}</button>}</div>}
    {notice && <p role="status" className="rounded-lg bg-emerald-50 p-3 text-sm text-emerald-800">{notice}</p>}
    {!releases && !error && <p role="status" className="py-6 text-sm text-slate-600">{systemText('preInvestment.timingReleaseLibrary.loadingExistingResearchVersions')}</p>}
    {releases?.length === 0 && <p className="rounded-xl border border-dashed border-slate-300 p-6 text-sm leading-6 text-slate-600">{systemText('preInvestment.timingReleaseLibrary.noSavedResearchVersionsCompleteResearchIn')}</p>}
    {!!releases?.length && <><div role="radiogroup" aria-label={systemText('preInvestment.timingReleaseLibrary.selectAnExistingResearchVersion')} className="grid gap-3 md:grid-cols-2">{releases.map(release => <label key={release.id} className={`flex cursor-pointer items-start gap-3 rounded-xl border p-4 ${selected?.id === release.id ? 'border-accent-400 bg-accent-50/40' : 'border-slate-200 bg-white'}`}><input type="radio" name="timing-release" className="mt-1 h-4 w-4 shrink-0" checked={selected?.id === release.id} disabled={binding || disabled} onChange={() => { setSelectedId(release.id); setNotice('') }} /><span className="min-w-0"><strong className="block break-words text-sm text-slate-900">{release.name || systemText('preInvestment.timingReleaseLibrary.timingResearchVersion')}</strong><span className="mt-1 block text-xs text-slate-600">{release.created_at?.replace('T', ' ').slice(0, 19) || systemText('preInvestment.timingReleaseLibrary.saved')} {" " + systemText('preInvestment.timingReleaseLibrary.researchUse')}</span>{release.products?.length ? <span className="mt-2 block break-words text-xs text-slate-600">{systemText('preInvestment.timingReleaseLibrary.applicableProducts')}{release.products.map(product => product.product_id).join('、')}</span> : null}{release.note && <span className="mt-2 block text-xs leading-5 text-slate-600">{release.note}</span>}</span></label>)}</div>
      <label className="block text-xs text-slate-600">{systemText('preInvestment.timingReleaseLibrary.referenceNotes')}<input className={timingField} maxLength={500} value={note} onChange={event => setNote(event.target.value)} placeholder={systemText('preInvestment.timingReleaseLibrary.recordPurposeAndPortfolioConstraints')} /></label>
      <div className="flex flex-wrap gap-3"><button type="button" className="min-h-10 rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm text-slate-700 disabled:opacity-40" disabled={!selected || binding || disabled} onClick={() => selected && onView(selected.run_id)}>{systemText('preInvestment.timingReleaseLibrary.viewThisVersionSResults')}</button><button type="button" className="min-h-10 rounded-lg bg-accent-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-40" disabled={!selected || binding || disabled} onClick={bind}>{binding ? systemText('preInvestment.timingReleaseLibrary.referencing') : systemText('preInvestment.timingReleaseLibrary.referenceThisVersionInPreInvestmentResearch')}</button></div>
      <p className="text-xs text-slate-600">{systemText('preInvestment.timingReleaseLibrary.noAutomaticTradesOrPortfolioWeightChanges')}</p>
    </>}
  </section>
}
