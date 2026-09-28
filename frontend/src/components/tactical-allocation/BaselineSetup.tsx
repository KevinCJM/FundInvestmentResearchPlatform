import { systemText, useI18n } from '../../i18n/runtime'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { allocationJourneyPath, readAllocationJourney } from '../../app/allocationJourney'
import { createTaaBaseline, type TaaBaseline, type TaaCatalog } from '../../services/tacticalAllocation'
import { buttonClass, Empty, Feedback, Field, inputClass, NumberInput, primaryClass, sectionClass, today } from '../risk-models/ResearchUI'
import { isUsable, UsabilityNote } from '../versioning'

export default function BaselineSetup({ catalog, selectedId, loading, onSelect, onCreated }: {
  catalog: TaaCatalog | null
  selectedId: string
  loading: boolean
  onSelect: (id: string) => void
  onCreated: (baseline: TaaBaseline) => void
}) {
  useI18n()
  const [creating, setCreating] = useState(false)
  const [allocation, setAllocation] = useState('')
  const [name, setName] = useState('')
  const [asOf, setAsOf] = useState(today)
  const [weights, setWeights] = useState<Record<string, number>>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const source = catalog?.allocations.find(item => item.alloc_name === allocation)
  const total = Object.values(weights).reduce((sum, value) => sum + value, 0)
  const valid = Boolean(source?.assets.length) && source!.assets.every(asset => Number.isFinite(weights[asset.id]) && weights[asset.id] >= 0 && weights[asset.id] <= 100) && Math.abs(total - 100) < 0.00001

  function selectAllocation(value: string) {
    const next = catalog?.allocations.find(item => item.alloc_name === value)
    setAllocation(value); setName(next ? systemText('preInvestment.baselineSetup.saaBaseline', { p0: next.alloc_name }) : '')
    setAsOf(next?.as_of || today()); setWeights(Object.fromEntries((next?.assets ?? []).map(asset => [asset.id, 0]))); setError('')
  }

  async function save() {
    if (!source || !valid || !name.trim() || !asOf) return
    setBusy(true); setError('')
    try {
      const baseline = await createTaaBaseline({ alloc_name: source.alloc_name, name: name.trim(), as_of: asOf, weights: Object.fromEntries(Object.entries(weights).map(([key, value]) => [key, value / 100])) })
      onCreated(baseline); setCreating(false)
    } catch (failure) { setError(failure instanceof Error ? failure.message : systemText('preInvestment.baselineSetup.unableToSaveTheSaaBaseline')) }
    finally { setBusy(false) }
  }

  return <section className={sectionClass} aria-label={systemText('preInvestment.baselineSetup.saaBaselineSelection')}><details open={!selectedId || creating}><summary className="cursor-pointer text-sm font-medium text-slate-800">{selectedId ? systemText('preInvestment.baselineSetup.longTermPortfolioChangeCreate', { p0: catalog?.baselines.find(item => item.id === selectedId)?.name ?? systemText('preInvestment.baselineSetup.loading') }) : systemText('preInvestment.baselineSetup.selectALongTermAllocationBaseline')}</summary><div className="mt-3">
    <div className="flex flex-wrap items-start justify-between gap-3"><div><h2 className="text-base font-semibold text-slate-950">{systemText('preInvestment.baselineSetup.whichLongTermPortfolioIsTheStarting')}</h2><p className="mt-1 text-sm text-slate-600">{systemText('preInvestment.baselineSetup.lockSaaWeightsAndAssetScopeThen')}</p></div><Link to={allocationJourneyPath('saa', { ...readAllocationJourney(), baselineId: selectedId })} className="min-h-11 py-2 text-sm font-medium text-accent-800 underline underline-offset-4">{systemText('preInvestment.baselineSetup.selectAPlanInSaa')}</Link></div>
    {loading ? <p role="status" className="mt-4 text-sm text-slate-600">{systemText('preInvestment.baselineSetup.loadingSaaBaseline')}</p> : <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-end"><div className="min-w-0 flex-1"><Field label={systemText('preInvestment.baselineSetup.saaBaselineVersion')}><select className={inputClass} value={selectedId} onChange={event => onSelect(event.target.value)}><option value="">{systemText('preInvestment.baselineSetup.selectASavedSaaBaseline')}</option>{catalog?.baselines.map(item => <option key={item.id} value={item.id} disabled={!isUsable(item) && item.id !== selectedId}>{item.name} · {item.as_of}{isUsable(item) ? '' : ` · ${systemText(item.usable?.status === 'blocked' ? 'versioning.blocked' : 'versioning.stale')}`}</option>)}</select></Field>{(item => item && !isUsable(item) && <UsabilityNote usable={item.usable} />)(catalog?.baselines.find(item => item.id === selectedId))}</div><button type="button" className={buttonClass} onClick={() => setCreating(value => !value)}>{creating ? systemText('preInvestment.baselineSetup.hideBaselineCreation') : systemText('preInvestment.baselineSetup.newResearchBaseline')}</button></div>}
    {!loading && !catalog?.baselines.length && !creating && <div className="mt-4"><Empty title={systemText('preInvestment.baselineSetup.defineTheLongTermAllocationBeforeDeviations')}><p>{systemText('preInvestment.baselineSetup.selectAndCarryOverAPortfolioFrom')}</p></Empty></div>}
    {creating && <div className="mt-5 space-y-4 border-t border-slate-200 pt-5"><Feedback error={error} /><p className="text-sm text-slate-600">{systemText('preInvestment.baselineSetup.setLongTermWeightsManuallySavingFreezes')}</p>
      {!catalog?.allocations.length ? <Empty title={systemText('preInvestment.baselineSetup.noAvailableAssetClassifications')}><p>{systemText('preInvestment.baselineSetup.buildAndSaveAssetClassesInSaa')}</p></Empty> : <fieldset disabled={busy} className="min-w-0 space-y-4"><div className="grid gap-4 sm:grid-cols-3"><Field label={systemText('preInvestment.baselineSetup.assetClassificationScheme')}><select className={inputClass} value={allocation} onChange={event => selectAllocation(event.target.value)}><option value="">{systemText('preInvestment.baselineSetup.selectAnExistingClassificationScheme')}</option>{catalog.allocations.map(item => <option key={item.alloc_name} value={item.alloc_name}>{item.alloc_name}</option>)}</select></Field><Field label={systemText('preInvestment.baselineSetup.baselineName')}><input className={inputClass} value={name} onChange={event => setName(event.target.value)} /></Field><Field label={systemText('preInvestment.baselineSetup.saaPolicyDate')}><input type="date" max={today()} className={inputClass} value={asOf} onChange={event => setAsOf(event.target.value)} /></Field></div>
        {source && <><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{source.assets.map(asset => <Field key={asset.id} label={systemText('preInvestment.baselineSetup.longTermWeight', { p0: asset.name })}><NumberInput className={inputClass} value={weights[asset.id]} onValueChange={value => setWeights(previous => ({ ...previous, [asset.id]: value }))} min={0} max={100} /></Field>)}</div><div className="flex flex-wrap items-center justify-between gap-3"><p role="status" className={`text-sm ${valid ? 'text-accent-800' : 'text-amber-800'}`}>{systemText('preInvestment.baselineSetup.totalWeight')}{Number.isFinite(total) ? `${total.toFixed(2)}%` : systemText('preInvestment.baselineSetup.completeAllFields')}{systemText('preInvestment.baselineSetup.mustTotal100')}</p><button type="button" className={buttonClass} onClick={() => setWeights(Object.fromEntries(source.assets.map(asset => [asset.id, 100 / source.assets.length])))}>{systemText('preInvestment.baselineSetup.startWithEqualWeights')}</button></div></>}
        <button type="button" className={primaryClass} disabled={!valid || !name.trim() || !asOf || busy} onClick={() => void save()}>{busy ? systemText('preInvestment.baselineSetup.freezingBaselineAndData') : systemText('preInvestment.baselineSetup.saveAndUseThisBaseline')}</button>
      </fieldset>}
    </div>}
  </div></details>{selectedId && <Link to={allocationJourneyPath('saa', { ...readAllocationJourney(), baselineId: selectedId })} className="mt-2 inline-block text-xs text-accent-800 underline">{systemText('preInvestment.baselineSetup.returnToThisSaaPolicy')}</Link>}</section>
}
