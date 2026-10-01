import { useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import PortableAgentMount from './PortableAgentMount'
import ResearchDraftArtifact, { type DraftActions, type PublicationAttempt } from '../../components/indicators/ResearchDraftArtifact'
import type { PageEvidenceSnapshot, ResearchPageContext } from '../../services/researchContracts'

type Props = Omit<DraftActions, 'capture'> & {
  pageContext: ResearchPageContext; ready: boolean; capturePageSnapshot: () => PageEvidenceSnapshot
  onRestoredContext: (context: ResearchPageContext) => void
}

export default function IndicatorAssistant(props: Props) {
  const [search] = useSearchParams()
  // These are business publication attempts, not conversation or execution state.
  const attempts = useRef(new Map<string, PublicationAttempt>())
  const notified = useRef(new Set<string>())
  return <PortableAgentMount pageContext={props.pageContext} busy={!props.ready}
    capturePageSnapshot={props.capturePageSnapshot} handoffId={search.get('portable_handoff') || undefined}
    onRestoredContext={props.onRestoredContext}
    prepareHandoff={(intent, baseline) => {
      if (intent.capability_id !== 'indicator-studio') throw new Error('该任务不属于指标中心。')
      const parameters = intent.parameters as Record<string, unknown>
      const calculation = { ...baseline.page_context.calculation }
      for (const key of ['period', 'as_of']) if (Object.prototype.hasOwnProperty.call(parameters, key)) calculation[key] = parameters[key]
      if (Object.prototype.hasOwnProperty.call(parameters, 'targets')) calculation.targets = parameters.targets
      else if (Object.prototype.hasOwnProperty.call(parameters, 'target')) calculation.targets = [parameters.target]
      return { ...baseline, page_context: { ...baseline.page_context, calculation }, capability_id: 'indicator-studio', intent_parameters: parameters }
    }}
    renderArtifact={(binding, agent) => binding.artifact.data.type === 'research.indicator'
      ? <ResearchDraftArtifact binding={binding} agent={agent} attempts={attempts.current} notified={notified.current}
          actions={{ ...props, capture: () => ({ page_context: structuredClone(props.pageContext), page_snapshot: props.capturePageSnapshot() }) }} /> : null}
  />
}
