import PortableAgentMount from '../integrations/portable-agent/PortableAgentMount'
export default function LlmSettings() {
  return <PortableAgentMount displayMode="settings" pageContext={{ page: 'platform-agent', page_instance_id: 'model-settings',
    context_revision: 0, view_state: 'inherit', calculation: { context_kind: 'platform' } }} />
}
