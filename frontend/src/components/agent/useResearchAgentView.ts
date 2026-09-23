import { useResearchContextIdentity, useResearchDay } from '../../app/ResearchContext'
import { getPitOverride } from '../../services/pitOverride'
import type { AgentPageContext } from '../../services/agent'

export default function useResearchAgentView(explicitDate = '') {
  const day = useResearchDay()
  const identity = useResearchContextIdentity()
  const override = getPitOverride()
  const viewState: AgentPageContext['view_state'] = explicitDate ? 'explicit'
    : day === undefined ? 'unknown' : override?.off ? 'off' : override ? 'explicit' : 'inherit'
  return { asOf: explicitDate || day || null, viewState, identity }
}
