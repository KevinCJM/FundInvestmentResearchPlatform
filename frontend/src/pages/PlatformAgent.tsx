import { useNavigate } from 'react-router-dom'
import PortableAgentMount from '../integrations/portable-agent/PortableAgentMount'
import { researchRequest } from '../integrations/portable-agent/client'
import type { HostAction } from '../integrations/portable-agent/contract'
import { useI18n } from '../i18n/runtime'

export default function PlatformAgent() {
  const { s } = useI18n()
  const navigate = useNavigate()
  const open = (action: HostAction) => {
    if (action.name !== 'navigate') { action.reject(new Error('该页面动作未注册。')); return }
    if (!['navigate', 'execute'].includes(String(action.arguments.mode))) { action.reject(new Error('页面动作模式无效。')); return }
    void researchRequest<{ items: Array<{ id: string; path: string; handoff: boolean; actions: string[] }> }>('/api/integrations/portable-agent/capabilities').then(({ items }) => {
      if (action.signal?.aborted) return
      const capability = items.find(item => item.id === action.arguments.capability_id)
      if (!capability || !capability.path.startsWith('/') || capability.path.startsWith('//')) throw new Error('目标页面未注册。')
      const execute = action.arguments.mode === 'execute'
      if (execute && (!capability.handoff || !capability.actions.includes('execute'))) throw new Error('该页面尚未接入任务交接。')
      if (execute && !/^[0-9a-f]{32}$/.test(String(action.arguments.handoff_id))) throw new Error('交接标识无效。')
      const target = new URL(capability.path, window.location.origin)
      if (execute) target.searchParams.set('portable_handoff', String(action.arguments.handoff_id))
      navigate(target.pathname + target.search); action.resolve({ navigated: true })
    }).catch(error => action.reject(error instanceof Error ? error : new Error(String(error))))
  }
  return <div className="min-w-0 space-y-4">
    <div><h1 className="text-2xl font-bold text-slate-900 sm:text-3xl">{s('agent.platform.title')}</h1>
      <p className="mt-2 text-sm text-slate-600">{s('agent.platform.description')}</p></div>
    <PortableAgentMount displayMode="embedded" onAction={open} pageContext={{ page: 'platform-agent', page_instance_id: 'platform-agent',
      context_revision: 0, view_state: 'inherit', calculation: { context_kind: 'platform' } }} />
  </div>
}
