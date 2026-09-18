import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { useResearchDay } from '../app/ResearchContext'
import { Badge, Button, Card } from '../components/ui'
import { Field } from '../components/risk-models/ResearchUI'
import { BoundsEditor, FrozenReferenceDetails } from '../components/risk-scales/AssumptionsEditor'
import { ReferenceEditor } from '../components/risk-scales/ReferenceEditor'
import { RiskScaleResults } from '../components/risk-scales/RiskScaleResults'
import { SegmentationEditor } from '../components/risk-scales/SegmentationEditor'
import { basicsIssue, copyRiskEditor, definitionIssue, editRiskEditor, newRiskEditor, referenceIssue, restoreReferenceDefinition, restoreRiskEditor, uniqueKey, type RiskEditor } from '../components/risk-scales/editor'
import { controlClass, ErrorNotice, Loading, Problems, problemMessage, TaskHeader, useRiskTask, useRiskText } from '../components/risk-scales/shared'
import { metadata, textValue, riskScales, registeredAlgorithms, RiskScaleError, type Capabilities, type DraftView, type PreviewResponse, type ReferenceInputRequest, type ReferencePreview, type ReferenceVersion, type RiskScaleDefinition } from '../services/riskScales'

const today = () => new Date().toISOString().slice(0, 10)
const sourceCode = (seriesId: string) => { const parts = seriesId.split(':'); return parts[parts.length - 1] || seriesId }

async function restoreSourceLabels(reference: ReferenceInputRequest, existing: Record<string, string>, frozen: ReferenceVersion | null, signal: AbortSignal) {
  const labels = { ...existing }
  const provenance = metadata(frozen?.provenance)
  const frozenAssets = Array.isArray(provenance.assets) ? provenance.assets : []
  for (const rawAsset of frozenAssets) {
    const asset = metadata(rawAsset)
    const sources = Array.isArray(asset.sources) ? asset.sources : []
    for (const rawSource of sources) {
      const source = metadata(rawSource), id = textValue(source.series_id), name = textValue(source.name)
      if (id && name) labels[id] = name
    }
  }
  const missing = reference.assets.flatMap(asset => asset.components).filter(component => !labels[component.series_id])
  const unique = [...new Map(missing.map(component => [component.series_id, component])).values()]
  await Promise.all(unique.map(async component => {
    const code = sourceCode(component.series_id)
    try {
      const catalog = await riskScales.sources(component.kind, code, 0, signal)
      const item = catalog.items.find(candidate => textValue(candidate.id) === component.series_id)
      labels[component.series_id] = textValue(item?.name) || code
    } catch (error) {
      if (signal.aborted) throw error
      labels[component.series_id] = code
    }
  }))
  return labels
}

export default function RiskScaleWorkspace() {
  const { t } = useRiskText(), navigate = useNavigate(), { draftId } = useParams(), [search] = useSearchParams()
  const platformDay = useResearchDay(), researchDayReady = platformDay !== undefined, researchDay = platformDay ?? today()
  const [editor, setEditor] = useState<RiskEditor>(() => newRiskEditor(researchDay)), [draft, setDraft] = useState<DraftView | null>(null)
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null), [loaded, setLoaded] = useState(false)
  const [preview, setPreview] = useState<PreviewResponse | null>(null), [fresh, setFresh] = useState(false)
  const [referencePreview, setReferencePreview] = useState<ReferencePreview | null>(null)
  const [confirmed, setConfirmed] = useState(false), [acknowledged, setAcknowledged] = useState<string[]>([]), [upstreamConfirmed, setUpstreamConfirmed] = useState(false)
  const [methodComparisons, setMethodComparisons] = useState<PreviewResponse[]>([]), [saved, setSaved] = useState(false)
  const task = useRiskTask(), bootstrap = useRiskTask(), idempotency = useRef(uniqueKey()), referenceKey = useRef(uniqueKey())
  const retryAction = useRef<((conflict: boolean) => void) | null>(null), segmentationTimer = useRef<number | null>(null)
  const value = editor.definition, step = editor.step
  const previewRequest = (definition: RiskScaleDefinition) => ({ definition, ...(draft ? { draft_id: draft.id, draft_revision: draft.revision } : {}) })

  const invalidateResults = () => { task.invalidate(); setPreview(null); setFresh(false); setConfirmed(false); setAcknowledged([]); setUpstreamConfirmed(false); setSaved(false); idempotency.current = uniqueKey(); setMethodComparisons([]) }
  const applyResearchDay = (current: RiskEditor, day: string): RiskEditor => current.definition.research_as_of === day ? current : { ...current, definition: { ...current.definition, research_as_of: day, reference_input_ref: { id: '', content_hash: '' } }, reference: { ...current.reference, as_of: day }, referenceVersion: null }

  const loadFrom = (resumeId?: string) => {
    task.invalidate(); setLoaded(false); setDraft(null); setPreview(null); setFresh(false); setReferencePreview(null)
    setConfirmed(false); setAcknowledged([]); setUpstreamConfirmed(false); setMethodComparisons([]); setSaved(false); retryAction.current = null
    idempotency.current = uniqueKey(); referenceKey.current = uniqueKey()
    void bootstrap.run(async signal => {
      const editFrom = search.get('editFrom'), copyFrom = search.get('from')
      const [caps, source] = await Promise.all([riskScales.capabilities(signal), resumeId ? riskScales.draft(resumeId, signal) : editFrom || copyFrom ? riskScales.version((editFrom || copyFrom)!, signal) : Promise.resolve(null)])
      let restored = source && 'editable_definition' in source ? restoreRiskEditor(source.editable_definition, researchDay)
        : source && 'preview' in source ? (editFrom ? editRiskEditor(source.preview.request_echo.definition, researchDay) : copyRiskEditor(source.preview.request_echo.definition, researchDay))
        : newRiskEditor(researchDay)
      let labelVersion: ReferenceVersion | null = null
      if (source && 'preview' in source) {
        const ref = source.preview.request_echo.definition.reference_input_ref
        const version = await riskScales.reference(ref.id, signal)
        if (version.id !== ref.id || version.content_hash !== ref.content_hash) throw new RiskScaleError('REFERENCE_HASH_MISMATCH', '所选冻结参考资产版本身份或内容校验不一致，请重新选择。', 'reference_input_ref', 409)
        labelVersion = version
        restored = { ...restored, reference: { ...restoreReferenceDefinition(version.definition), as_of: researchDay }, referenceVersion: null, definition: { ...restored.definition, reference_input_ref: { id: '', content_hash: '' } } }
      } else if (restored.definition.reference_input_ref.id) {
        const ref = restored.definition.reference_input_ref
        const version = await riskScales.reference(ref.id, signal)
        if (version.id !== ref.id || version.content_hash !== ref.content_hash) throw new RiskScaleError('REFERENCE_HASH_MISMATCH', '所选冻结参考资产版本身份或内容校验不一致，请重新选择。', 'reference_input_ref', 409)
        labelVersion = version
        restored = { ...restored, referenceVersion: version, reference: { ...restoreReferenceDefinition(version.definition), as_of: researchDay } }
      }
      restored = { ...restored, sourceLabels: await restoreSourceLabels(restored.reference, restored.sourceLabels, labelVersion, signal) }
      return { caps, source, restored: applyResearchDay(restored, researchDay) }
    }, ({ caps, source, restored }) => { setCapabilities(caps); if (source && 'editable_definition' in source) setDraft(source); setEditor(restored); setLoaded(true) })
  }
  const load = () => loadFrom(draftId)
  useEffect(() => { load(); return () => { if (segmentationTimer.current != null) window.clearTimeout(segmentationTimer.current); bootstrap.invalidate(); task.invalidate() } }, [draftId, search.get('from'), search.get('editFrom')])
  useEffect(() => { if (!loaded || value.research_as_of === researchDay) return; invalidateResults(); referenceKey.current = uniqueKey(); setReferencePreview(null); setEditor(current => applyResearchDay(current, researchDay)) }, [researchDay, loaded])

  const edit = (next: RiskEditor) => { invalidateResults(); setEditor(next) }
  const editDefinition = (next: RiskScaleDefinition) => edit({ ...editor, definition: { ...next, research_as_of: researchDay, risk_basis_id: 'annualized-periodic-volatility-v1' } })
  const editSegmentation = (next: RiskScaleDefinition) => {
    if (segmentationTimer.current != null) window.clearTimeout(segmentationTimer.current)
    task.invalidate(); setFresh(false); setConfirmed(false); setAcknowledged([]); setMethodComparisons([]); setSaved(false); idempotency.current = uniqueKey()
    const normalized = { ...next, research_as_of: researchDay, risk_basis_id: 'annualized-periodic-volatility-v1' as const }
    setEditor(current => ({ ...current, definition: normalized }))
    if (definitionIssue(normalized) || !normalized.reference_input_ref.id || !capabilities?.ready) return
    segmentationTimer.current = window.setTimeout(() => {
      retryAction.current = () => editSegmentation(normalized)
      void task.run(signal => riskScales.preview(previewRequest(normalized), signal), result => { setPreview(result); setFresh(true) })
    }, 180)
  }
  const changeReference = (reference: RiskEditor['reference'], sourceLabels = editor.sourceLabels) => { referenceKey.current = uniqueKey(); setReferencePreview(null); edit({ ...editor, reference: { ...reference, name: value.name || reference.name, as_of: researchDay, return_basis: 'selected_index_and_adjusted_product_total_return' }, sourceLabels, referenceVersion: null, definition: { ...value, research_as_of: researchDay, reference_input_ref: { id: '', content_hash: '' } } }) }
  const move = (next: number) => { setEditor(current => ({ ...current, step: next })); setConfirmed(false); window.setTimeout(() => document.getElementById('risk-step-title')?.focus(), 0) }
  const save = () => { if (segmentationTimer.current != null) window.clearTimeout(segmentationTimer.current); invalidateResults(); retryAction.current = conflict => { if (conflict && draft) loadFrom(draft.id); else save() }; const body = { name: value.name, scheme_id: value.scheme_id, editable_definition: { ...editor } }; void task.run(signal => draft ? riskScales.updateDraft(draft.id, { ...body, expected_revision: draft.revision }, signal) : riskScales.saveDraft(body, signal), result => { setDraft(result); setSaved(true); setEditor(current => ({ ...current, step: Math.min(current.step, 2) })) }) }
  const compute = () => { retryAction.current = compute; setFresh(false); setConfirmed(false); setAcknowledged([]); void task.run(signal => riskScales.preview(previewRequest({ ...value, research_as_of: researchDay }), signal), result => { setPreview(result); setFresh(true); move(3) }) }
  const referenceRequest = () => ({ ...editor.reference, name: value.name || t('referenceAssetsFallbackName'), as_of: researchDay, return_basis: 'selected_index_and_adjusted_product_total_return' as const })
  const checkReference = () => { retryAction.current = checkReference; setReferencePreview(null); setUpstreamConfirmed(false); void task.run(signal => riskScales.previewReference(referenceRequest(), signal), result => { setReferencePreview(result); setUpstreamConfirmed(false) }) }
  const confirmReference = () => { retryAction.current = conflict => conflict ? checkReference() : confirmReference(); void task.run(signal => riskScales.confirmReference({ request: referenceRequest(), preview_hash: referencePreview!.preview_hash, confirm: true, idempotency_key: referenceKey.current, acknowledged_warnings: referencePreview!.warnings.map(item => item.code) }, signal), result => { setEditor(current => ({ ...current, referenceVersion: result, reference: restoreReferenceDefinition(result.definition), definition: { ...current.definition, research_as_of: researchDay, reference_input_ref: { id: result.id, content_hash: result.content_hash } }, step: 2 })); setUpstreamConfirmed(false) }) }
  const publish = () => { retryAction.current = conflict => conflict ? compute() : publish(); void task.run(signal => riskScales.confirm({ request: preview!.request_echo, preview_hash: preview!.preview_hash, confirm: true, idempotency_key: idempotency.current, acknowledged_warnings: acknowledged }, signal), version => navigate(`/settings/risk-scales/versions/${encodeURIComponent(version.id)}`)) }
  const compareMethods = () => { retryAction.current = compareMethods; void task.run(async signal => { const results: PreviewResponse[] = []; for (const method of registeredAlgorithms(capabilities!).filter(item => item.available !== false && item.id !== 'manual_volatility_bands_v1')) results.push(await riskScales.preview(previewRequest({ ...value, segmentation: { algorithm_id: method.id } }), signal)); return results }, setMethodComparisons) }

  const issue = step === 0 ? basicsIssue(value) : step === 1 && !editor.referenceVersion ? referenceIssue(referenceRequest()) : step === 2 ? definitionIssue(value) : ''
  let action = () => move(step + 1), actionLabel = t('next'), actionReason = !researchDayReady ? 'researchDayPending' : issue
  if (step === 1 && !editor.referenceVersion) { action = referencePreview ? confirmReference : checkReference; actionLabel = t(referencePreview ? 'confirmReference' : 'checkReference'); if (referencePreview && !upstreamConfirmed) actionReason = 'confirmUpstreamRequired' }
  if (step === 2) { action = compute; actionLabel = t('compute') }
  if (step === 3) { action = () => move(4); actionLabel = t('review'); if (!fresh) actionReason = definitionIssue(value) || (task.busy ? 'previewUpdating' : 'previewPending'); else if (!preview?.publication_eligibility.eligible) actionReason = 'notPublishable' }
  if (step === 4) { action = publish; actionLabel = t('publish'); if (!fresh || !preview?.publication_eligibility.eligible) actionReason = 'stale'; else if (!confirmed || preview.warnings.some(item => !acknowledged.includes(item.code))) actionReason = 'confirmPublishRequired' }
  if (!capabilities?.ready && step >= 2) actionReason = 'notReady'
  if ((task.error as { status?: number } | null)?.status === 409) actionReason = 'conflict'
  const assetIds = editor.referenceVersion?.ordered_asset_ids ?? editor.reference.assets.map(asset => asset.id)
  const assetNames = new Map(editor.reference.assets.map(asset => [asset.id, asset.name] as const))
  const assets = assetIds.map(id => ({ id, name: assetNames.get(id) || id }))

  return <div className="min-w-0 space-y-3 text-slate-900">
    <TaskHeader title={value.name || t('newTitle')}><Badge>{t(saved ? 'saved' : 'editing')}</Badge><Button disabled={task.busy || !value.name.trim() || !loaded || !researchDayReady} onClick={save}>{t('saveDraft')}</Button></TaskHeader>
    <ErrorNotice error={bootstrap.error} retry={load} />{!loaded ? <Loading /> : <>
      <p className="text-xs text-slate-600">{researchDayReady ? t(platformDay ? 'researchDayFromPit' : 'researchDayFromToday', { date: researchDay }) : t('researchDayPending')}</p>{!value.name.trim() && <p className="text-xs text-slate-600">{t('nameForDraft')}</p>}
      <nav aria-label={t('steps')} className="grid gap-1 border-b border-slate-200 pb-2 sm:grid-cols-5">{['basics', 'assets', 'historicalParameters', 'frontierStep', 'reviewStep'].map((key, index) => <Button className="!whitespace-normal text-left" key={key} aria-current={index === step ? 'step' : undefined} disabled={index > step} onClick={() => move(index)}>{index + 1}. {t(key)}</Button>)}</nav>
      <ErrorNotice error={task.error} retry={() => (task.error as { code?: string } | null)?.code === 'REVISION_CONFLICT' && draft ? loadFrom(draft.id) : retryAction.current?.((task.error as { status?: number } | null)?.status === 409)} retryLabel={(task.error as { code?: string } | null)?.code === 'REVISION_CONFLICT' ? t('reloadDraft') : undefined} />
      {saved && <p role="status" className="text-sm text-emerald-800">{t('draftSaved', { revision: draft?.revision ?? 0 })}</p>}
      <Card className="min-w-0 !p-4 space-y-3"><h2 id="risk-step-title" tabIndex={-1} className="text-base font-semibold">{t(['basics', 'assets', 'historicalParameters', 'frontierStep', 'reviewStep'][step])}</h2>
      {step === 0 && <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3"><Field label={t('name')} required><input required className={controlClass} value={value.name} onChange={event => editDefinition({ ...value, name: event.target.value })} /></Field><Field label={t('researchDate')} hint={researchDayReady ? t(platformDay ? 'researchDatePitHint' : 'researchDateTodayHint') : t('researchDayPending')}><input className={controlClass} value={researchDayReady ? researchDay : ''} placeholder={researchDayReady ? undefined : t('loading')} readOnly aria-readonly="true" /></Field><Field label={t('reviewDate')} hint={t('reviewDateOptionalHint')}><input className={controlClass} type="date" value={value.review_due_at ?? ''} min={researchDayReady ? researchDay : undefined} disabled={!researchDayReady} onChange={event => editDefinition({ ...value, review_due_at: event.target.value || null })} /></Field><Field label={t('purpose')} hint={t('purposeHint')}><textarea className={controlClass} value={value.purpose ?? ''} onChange={event => editDefinition({ ...value, purpose: event.target.value })} /></Field><Field label={t('description')}><textarea className={controlClass} value={value.description ?? ''} onChange={event => editDefinition({ ...value, description: event.target.value })} /></Field></div>}
      {step === 1 && <><ReferenceEditor sourceLabels={editor.sourceLabels} value={editor.reference} onChange={changeReference} />{referencePreview && !editor.referenceVersion && <section className="space-y-2 border-t border-slate-200 pt-3"><h3 className="font-semibold">{t('referencePreviewTitle')}</h3><p className="text-sm font-medium">{t('intersectionRange', { start: String(referencePreview.quality.intersection_start ?? t('unavailable')), end: String(referencePreview.quality.intersection_end ?? t('unavailable')) })}</p><p className="text-xs text-slate-600">{t('intersectionMeaning')}</p><Problems items={referencePreview.warnings} /><label className="flex min-h-10 items-start gap-2 text-sm"><input type="checkbox" className="mt-1" checked={upstreamConfirmed} onChange={event => setUpstreamConfirmed(event.target.checked)} />{t('confirmReferenceText')}</label></section>}</>}
      {step === 2 && editor.referenceVersion && <><FrozenReferenceDetails version={editor.referenceVersion} /><BoundsEditor value={value} assets={assets} onChange={editDefinition} /></>}
      {step === 3 && capabilities && <><SegmentationEditor value={value} capabilities={capabilities} appliedCaps={fresh ? preview?.result.applied_boundaries : undefined} busy={task.busy} onChange={editSegmentation} /><p className="text-xs text-slate-600">{t('reuseFrontier')}</p>{preview && <>{!fresh && <p role="status" className="text-sm text-amber-900">{t(task.busy ? 'previewUpdating' : 'previewPending')}</p>}<RiskScaleResults preview={preview} quality={editor.referenceVersion?.quality} /><details><summary className="min-h-10 cursor-pointer py-2 text-sm font-medium">{t('compareMethods')}</summary><Button disabled={task.busy || !fresh} onClick={compareMethods}>{t('loadMethodComparison')}</Button><p className="my-2 text-xs text-slate-600">{t('comparisonHint')}</p>{methodComparisons.map(result => <div className="border-b border-slate-200 py-4" key={result.result.algorithm_id}><RiskScaleResults preview={result} compact /></div>)}</details></>}</>}
      {step === 4 && preview && <><p className="text-sm">{value.name} · {t('researchDate')}: {preview.request_echo.definition.research_as_of}{value.review_due_at ? ` · ${t('reviewDate')}: ${value.review_due_at}` : ''}</p>{value.purpose && <p className="text-sm text-slate-600">{value.purpose}</p>}<RiskScaleResults preview={preview} compact />{value.segmentation?.rationale && <p className="text-sm">{t('manualRationale')}: {value.segmentation.rationale}</p>}{preview.warnings.map(warning => <label key={warning.code} className="flex min-h-10 items-start gap-2 text-sm"><input type="checkbox" className="mt-1" checked={acknowledged.includes(warning.code)} onChange={event => setAcknowledged(current => event.target.checked ? [...current, warning.code] : current.filter(code => code !== warning.code))} />{problemMessage(warning)}</label>)}<label className="flex min-h-10 items-start gap-2 text-sm"><input type="checkbox" className="mt-1" checked={confirmed} onChange={event => setConfirmed(event.target.checked)} />{t('confirmPublishText')}</label></>}
      {task.busy && <Loading />}{actionReason && <p role="status" className="text-sm text-amber-900">{t(actionReason)}</p>}<div className="flex flex-col gap-2 border-t border-slate-200 pt-3 sm:flex-row sm:justify-between"><Button disabled={step === 0 || task.busy} onClick={() => move(step - 1)}>{t('previous')}</Button><Button tone="primary" className="!whitespace-normal" disabled={task.busy || Boolean(actionReason)} onClick={action}>{actionLabel}</Button></div>
      </Card>
    </>}
  </div>
}
