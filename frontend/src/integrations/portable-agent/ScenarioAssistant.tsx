import { useEffect, useState } from 'react'
import PortableAgentMount from './PortableAgentMount'
import { researchRequest } from './client'
import type { ArtifactBinding } from './contract'
import type { ResearchPageContext } from '../../services/researchContracts'
import { usePageContextRevision } from '../../app/useResearchBinding'
import { newPageSnapshotId } from '../../services/researchPageEvidence'
import { Button } from '../../components/ui'
import { useI18n } from '../../i18n/runtime'

type Props = {
  page: 'scenario-algorithms' | 'historical-regimes' | 'published-scenarios' | 'global-events'
  workspace: 'graph' | 'events' | 'published' | 'stress'; purpose?: string; active?: boolean; busy?: boolean
  mode?: 'realtime' | 'retrospective'; asOf?: string; definition?: Record<string, unknown> | null
  onApply?: (definition: Record<string, unknown>) => void
  onView?: (artifact: ScenarioArtifact) => void
}
export type ScenarioArtifact = { definition: Record<string, unknown>; valid: boolean; validation_scope: string; workspace: string;
  result?: Record<string, unknown>; preview_id?: string; run_id?: string; source_run_id: string }

function ScenarioResult({ binding, apply, view }: { binding: ArtifactBinding; apply?: Props['onApply']; view?: Props['onView'] }) {
  const { s } = useI18n()
  const [value, setValue] = useState<ScenarioArtifact | null>(null)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  useEffect(() => {
    let active = true
    setValue(null); setError('')
    void researchRequest<ScenarioArtifact>(`/api/research/scenario-artifacts/${encodeURIComponent(String(binding.artifact.data.artifact_id))}`)
      .then(result => { if (active) setValue(result) }).catch(reason => { if (active) setError(reason.message) })
    return () => { active = false }
  }, [binding.artifact.id, retry])
  return <section className="rounded-xl border border-slate-200 bg-white p-3 text-left text-sm text-slate-900" aria-label={s('agent.scenarioRegion')}>
    <h3 className="font-semibold">{String(value?.definition.name || s('agent.scenarioDraft'))}</h3>
    <p className="mt-1 text-xs text-slate-600">{value ? value.valid
      ? (value.validation_scope === 'graph' ? s('agent.scenarioGraphValidated') : s('agent.scenarioParametersValidated'))
      : s('agent.scenarioValidationFailed') : !error ? s('agent.scenarioLoading') : null}
      {s('agent.scenarioPublicationHelp')}</p>
    {error && <><p role="alert" className="mt-2 text-rose-700">{error}</p><Button onClick={() => setRetry(value => value + 1)}>{s('common.retry')}</Button></>}
    {value && <div className="mt-2 flex flex-wrap gap-2">
      <Button disabled={!value.valid || !apply || binding.busy || !binding.isCurrent} onClick={() => apply?.(structuredClone(value.definition))}>{s('agent.scenarioApply')}</Button>
      {value.result && <Button disabled={!view || binding.busy || !binding.isCurrent} onClick={() => view?.(value)}>{s('agent.viewPreview')}</Button>}
    </div>}
  </section>
}

export default function ScenarioAssistant(props: Props) {
  const revision = usePageContextRevision(JSON.stringify([props.definition, props.mode, props.asOf, props.purpose]))
  const pageContext: ResearchPageContext = { page: props.page, page_instance_id: `${props.page}:${props.workspace}:${props.purpose || 'research'}`,
    context_revision: revision, view_state: 'inherit', calculation: { context_kind: 'scenario', workspace: props.workspace,
      purpose: props.purpose || 'research', mode: props.mode || 'realtime', as_of: props.asOf || null } }
  return <PortableAgentMount pageContext={pageContext} active={props.active} busy={props.busy}
    capturePageSnapshot={() => ({ version: 1, page: props.page, snapshot_id: newPageSnapshotId(),
      sections: { editing: { definition: props.definition || null, mode: props.mode || 'realtime', as_of: props.asOf || null } } })}
    renderArtifact={binding => <ScenarioResult binding={binding} apply={props.onApply} view={props.onView} />} />
}
