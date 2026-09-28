import { systemText, useI18n } from '../../i18n/runtime'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { updateAllocationJourney, useAllocationDraft } from '../../app/allocationJourney'
import { useResearchDay } from '../../app/ResearchContext'
import { Button } from '../ui'
import { Field, Feedback, inputClass, today } from '../risk-models/ResearchUI'
import type { StrategicCatalog } from '../../services/strategicAllocation'
import { previewImplementationMap, confirmImplementationMap, getImplementationMap, type MappingDefinition, type MappingPreview, type MappingVersion, type UniverseVersion } from '../../services/strategicScope'
import { listInvestableUniverseSnapshots } from '../../services/productPools'
import { useScopeOperation } from './useScopeOperation'

export default function ImplementationMappingEditor({ universe, catalog, domainId, requestedMapping, onSaved, heading = true }: {
  universe: UniverseVersion; catalog: StrategicCatalog; domainId: string; requestedMapping: string; onSaved: (value: MappingVersion) => void
  /** 外层已提供“匹配真实代理产品（可稍后完成）”标题时不再重复。 */
  heading?: boolean
}) {
  useI18n()
  const clock = useResearchDay()
  const [draft, setDraft] = useAllocationDraft<MappingDefinition>(`implementation-map:${universe.id}:${domainId}`, () => ({
    name: systemText('preInvestment.implementationMappingEditor.implementationMapping', { p0: universe.name }), strategic_universe_id: universe.id, universe_snapshot_id: domainId,
    alloc_name: '', as_of: clock ?? today(), valid_until: '', assignments: [],
  }))
  const [preview, setPreview] = useState<MappingPreview | null>(null)
  const [saved, setSaved] = useState<MappingVersion | null>(null)
  const [snapshotLabels, setSnapshotLabels] = useState<Record<string, string>>({})
  // 可读的产品域名称；历史引用缺失时回退到 ID，不伪造名称。
  useEffect(() => {
    let active = true
    listInvestableUniverseSnapshots()
      .then(value => { if (active) setSnapshotLabels(Object.fromEntries(value.items.map(item => [item.id, `${item.name} · ${item.research_date}`]))) })
      .catch(() => { if (active) setSnapshotLabels({}) })
    return () => { active = false }
  }, [])
  const operation = useScopeOperation()
  const historyRead = useScopeOperation()
  useEffect(() => { operation.invalidate(); if (!saved) setPreview(null) }, [clock])
  function readMapping() {
    operation.invalidate(); setSaved(null); setPreview(null)
    if (!requestedMapping) return
    void historyRead.run(signal => getImplementationMap(requestedMapping, signal), result => {
      if (result.id !== requestedMapping) throw new Error(systemText('preInvestment.implementationMappingEditor.theLoadedMappingIdentityDoesNotMatch'))
      if (result.definition.strategic_universe_id !== universe.id) throw new Error(systemText('preInvestment.implementationMappingEditor.theMappingDoesNotMatchTheSelected'))
      setDraft(result.definition); setSaved(result); setPreview(result)
      updateAllocationJourney({ strategicUniverseId: universe.id, universeId: result.definition.universe_snapshot_id, implementationMappingId: result.id })
    })
  }
  useEffect(() => { readMapping(); return () => historyRead.invalidate() }, [requestedMapping, universe.id])
  const change = (patch: Partial<MappingDefinition>) => { historyRead.invalidate(); operation.invalidate(); setPreview(null); setSaved(null); setDraft(current => ({ ...current, ...patch })) }
  const allocations = catalog.allocations.filter(a => a.universe_snapshot_id === draft.universe_snapshot_id)
  const allocation = allocations.find(a => a.alloc_name === draft.alloc_name)
  const issue = clock === undefined ? systemText('preInvestment.implementationMappingEditor.theKnowledgeCutoffIsUnconfirmedPreviewAnd') : !draft.name.trim() || !draft.universe_snapshot_id || !allocation ? systemText('preInvestment.implementationMappingEditor.selectALockedProductUniverseAndIts') : !draft.as_of || draft.as_of > (clock ?? today()) || !draft.valid_until || draft.valid_until <= draft.as_of ? systemText('preInvestment.implementationMappingEditor.theMappingResearchDateMustBeOn') : ''
  return <section className="space-y-4 border-t border-slate-200 pt-5" aria-label={systemText('preInvestment.implementationMappingEditor.strategicImplementationMapping')}>
    {heading && <h2 className="text-lg font-semibold">{systemText('preInvestment.implementationMappingEditor.matchActualProxyProductsCanBeCompleted')}</h2>}
    <p className="text-sm leading-6 text-slate-600">{systemText('preInvestment.implementationMappingEditor.gapsRemainExplicitInStrategicResearchNo')}</p>
    <Feedback error={operation.error || historyRead.error} />
    {historyRead.busy && <p role="status" className="text-sm text-slate-600">{systemText('preInvestment.implementationMappingEditor.loadingImmutableImplementationMapping')}</p>}
    {historyRead.error && <Button onClick={readMapping}>{systemText('preInvestment.implementationMappingEditor.retryLoadingImplementationMapping')}</Button>}
    {saved && <p className="text-sm text-slate-700">{systemText('preInvestment.implementationMappingEditor.readOnlyMapping')}{saved.name}。<Button onClick={() => { historyRead.invalidate(); operation.invalidate(); setSaved(null); setPreview(null) }}>{systemText('preInvestment.implementationMappingEditor.copyMappingAsNewResearch')}</Button></p>}
    <fieldset disabled={Boolean(saved) || historyRead.busy} className="min-w-0 space-y-4">
      <div className="grid gap-4 sm:grid-cols-2"><Field label={systemText('preInvestment.implementationMappingEditor.mappingName')} required><input required className={inputClass} value={draft.name} onChange={e => change({ name: e.target.value })} /></Field>
        <Field label={systemText('preInvestment.implementationMappingEditor.lockedProductUniverse')} required><select required className={inputClass} value={draft.universe_snapshot_id} onChange={e => change({ universe_snapshot_id: e.target.value, alloc_name: '', assignments: [] })}><option value="">{systemText('preInvestment.implementationMappingEditor.selectAnActualProductUniverse')}</option>{Array.from(new Set([domainId, ...catalog.allocations.map(a => a.universe_snapshot_id)].filter((id): id is string => Boolean(id)))).map(id => <option key={id} value={id}>{snapshotLabels[id] ?? id}</option>)}</select></Field>
        <Field label={systemText('preInvestment.implementationMappingEditor.actualProxyAssetClassScheme')} required><select required className={inputClass} value={draft.alloc_name} onChange={e => change({ alloc_name: e.target.value, assignments: [] })}><option value="">{systemText('preInvestment.implementationMappingEditor.selectASchemeFromTheSameProduct')}</option>{allocations.map(a => <option key={a.alloc_name} value={a.alloc_name}>{a.alloc_name}</option>)}</select></Field>
        <Field label={systemText('preInvestment.implementationMappingEditor.mappingResearchDate')} required><input required type="date" className={inputClass} value={draft.as_of} onChange={e => change({ as_of: e.target.value })} /></Field><Field label={systemText('preInvestment.implementationMappingEditor.mappingReviewDate')} required><input required type="date" className={inputClass} value={draft.valid_until} onChange={e => change({ valid_until: e.target.value })} /></Field></div>
      <div className="divide-y divide-slate-200">{universe.definition.assets.map(asset => {
        const row = draft.assignments.find(a => a.strategic_asset_id === asset.id)
        return <div key={asset.id} className="grid gap-3 py-4 sm:grid-cols-2"><Field label={systemText('preInvestment.implementationMappingEditor.actualProxyFor', { p0: asset.name })} optional><select className={inputClass} value={row?.proxy_asset_id ?? ''} onChange={e => change({ assignments: [...draft.assignments.filter(a => a.strategic_asset_id !== asset.id), ...(e.target.value ? [{ strategic_asset_id: asset.id, proxy_asset_id: e.target.value, rationale: row?.rationale ?? '' }] : [])] })}><option value="">{systemText('preInvestment.implementationMappingEditor.keepProductGap')}</option>{allocation?.assets.map(a => <option key={a.id} value={a.id}>{a.name}</option>)}</select></Field>{row ? <Field label={systemText('preInvestment.implementationMappingEditor.mappingRationaleFor', { p0: asset.name })} optional><input className={inputClass} value={row.rationale} onChange={e => change({ assignments: draft.assignments.map(a => a.strategic_asset_id === asset.id ? { ...a, rationale: e.target.value } : a) })} /></Field> : <p className="self-center text-sm text-amber-800">{systemText('preInvestment.implementationMappingEditor.productsAreMissingForwardLookingResearchCan')}</p>}</div>
      })}</div>
    </fieldset>
    {issue && <p role="status" className="text-sm text-amber-800">{issue}</p>}
    {!allocations.length && <Link className="inline-flex min-h-10 items-center text-sm text-accent-800 underline" to="/pre-investment/product-pool">{systemText('preInvestment.implementationMappingEditor.selectProductPoolsAndBuildActualAsset')}</Link>}
    <div className="flex flex-wrap gap-3"><Button disabled={Boolean(issue) || operation.busy || historyRead.busy || Boolean(saved)} onClick={() => void operation.run(signal => previewImplementationMap(draft, signal), setPreview)}>{systemText('preInvestment.implementationMappingEditor.checkMappingCoverage')}</Button><Button tone="primary" disabled={Boolean(issue) || operation.busy || historyRead.busy || !preview || Boolean(saved)} onClick={() => preview && void operation.run(signal => confirmImplementationMap(draft, preview.preview_hash, signal), result => { setSaved(result); setPreview(result); updateAllocationJourney({ strategicUniverseId: universe.id, universeId: result.definition.universe_snapshot_id, implementationMappingId: result.id }); onSaved(result) })}>{systemText('preInvestment.implementationMappingEditor.confirmAndSaveImplementationMapping')}</Button></div>
    {!preview && <p className="text-xs text-slate-600">{systemText('preInvestment.implementationMappingEditor.checkCoverageAndSourcesBeforeConfirmingAnd')}</p>}
    {operation.busy && <p role="status" className="text-sm text-slate-600">{systemText('preInvestment.implementationMappingEditor.checkingActualProductUniverseAndProxySources')}</p>}
    {preview && <div aria-live="polite" className="space-y-2 text-sm"><p>{preview.implementation_status === 'complete' ? systemText('preInvestment.implementationMappingEditor.allStrategicAssetsAreCoveredCurrentEligibility') : systemText('preInvestment.implementationMappingEditor.actualTradingProductsAreIncompleteAssetClass')}</p>{preview.coverage.map(row => <p key={row.strategic_asset_id}>{row.strategic_asset_id}：{row.proxy_asset_id ?? systemText('preInvestment.implementationMappingEditor.missingProducts')}</p>)}</div>}
  </section>
}
