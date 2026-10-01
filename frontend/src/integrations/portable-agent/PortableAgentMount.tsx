import { useEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { useLocation } from 'react-router-dom'
import { useI18n } from '../../i18n/runtime'
import type { PageEvidenceSnapshot, ResearchPageContext } from '../../services/researchContracts'
import { bootstrap, captureIdentity, registerContext, researchRequest, type ContextCapture } from './client'
import type { ArtifactBinding, HostAction, PortableAgentElement } from './contract'

type Props = {
  pageContext: ResearchPageContext; capturePageSnapshot?: () => PageEvidenceSnapshot | null
  active?: boolean; busy?: boolean; displayMode?: 'floating' | 'embedded' | 'settings'; handoffId?: string
  renderArtifact?: (binding: ArtifactBinding, agent: PortableAgentElement) => ReactNode
  renderStatus?: () => ReactNode
  onAction?: (action: HostAction) => void
  onRestoredContext?: (context: ResearchPageContext) => void
  prepareHandoff?: (intent: Record<string, unknown>, baseline: ContextCapture) => ContextCapture
}

const modules = new Map<string, Promise<unknown>>()
async function loadModule(path: string) {
  const url = new URL(path, window.location.origin)
  if (url.origin !== window.location.origin || url.pathname !== '/assistant/widget/widget.js') throw new Error('助手组件地址不符合接入配置。')
  let task = modules.get(url.href)
  if (!task) {
    task = import(/* @vite-ignore */ url.href).catch(error => { modules.delete(url.href); throw error })
    modules.set(url.href, task)
  }
  await task
  await customElements.whenDefined('portable-agent')
}

/** Router facts only. The external SDK owns handoff state, persistence, cancellation and recovery. */
export function PortableAgentNavigation() {
  const location = useLocation()
  useEffect(() => {
    window.dispatchEvent(new CustomEvent('portable-agent-navigation', { detail: { handoffId: new URLSearchParams(location.search).get('portable_handoff') } }))
  }, [location.pathname, location.search])
  return null
}

/** Host wiring only: no messages, run polling, SSE cursor, cancellation or conversation storage. */
export default function PortableAgentMount(props: Props) {
  const host = useRef<HTMLDivElement>(null)
  const element = useRef<PortableAgentElement | null>(null)
  const latest = useRef(props); latest.current = props
  const { locale } = useI18n()
  const latestLocale = useRef(locale); latestLocale.current = locale
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  const [slots, setSlots] = useState<Record<string, ArtifactBinding>>({})
  const [statusSlot, setStatusSlot] = useState<HTMLElement | null>(null)
  const [infoSlot, setInfoSlot] = useState<HTMLElement | null>(null)
  const scope = `${props.pageContext.page}:${props.pageContext.page_instance_id}`
  const active = props.active !== false
  const contextRevision = JSON.stringify(props.pageContext)
  const capture = (): ContextCapture => ({ page_context: structuredClone(latest.current.pageContext), page_snapshot: latest.current.capturePageSnapshot?.() })
  const initialCapture = useRef<{ scope: string; value: ContextCapture }>()
  if (!initialCapture.current || initialCapture.current.scope !== scope) initialCapture.current = { scope, value: capture() }

  useEffect(() => {
    if (!active) return
    let disposed = false
    const mounts = new Set<string>()
    setError('')
    const start = async () => {
      try {
        const initial = await registerContext(capture())
        if (disposed) return
        const config = await bootstrap(initial)
        if (disposed) return
        await loadModule(config.module_url)
        if (disposed || !host.current) return
        const agent = document.createElement('portable-agent') as PortableAgentElement
        const status = document.createElement('div'); status.slot = 'context-status'; agent.append(status); setStatusSlot(status)
        const info = document.createElement('div'); info.slot = 'context-info'; agent.append(info); setInfoSlot(info)
        const artifact = (event: Event) => {
          if (!latest.current.renderArtifact) return
          const detail = (event as CustomEvent<ArtifactBinding>).detail
          if (!['research.indicator', 'research.scenario'].includes(String(detail.artifact.data.type))) return
          event.preventDefault()
          const key = detail.artifact.id; mounts.add(key)
          detail.setUpdate(update => { if (!disposed) setSlots(previous => previous[key] ? { ...previous, [key]: { ...previous[key], ...update } } : previous) })
          detail.setDispose(() => { if (!disposed) setSlots(previous => { const next = { ...previous }; delete next[key]; return next }) })
          setSlots(previous => ({ ...previous, [key]: detail }))
        }
        const action = (event: Event) => {
          if (!latest.current.onAction) return
          event.preventDefault(); latest.current.onAction((event as CustomEvent<HostAction>).detail)
        }
        agent.addEventListener('agent-artifact-mount', artifact)
        agent.addEventListener('agent-action', action)
        agent.configure({ app: config.app, endpoint: config.endpoint, protocolMajor: config.protocol_major,
          requiredCapabilities: config.required_capabilities, expectedRelease: config.expected_release,
          context: config.context, scopeKey: config.context.scope_key, displayMode: props.displayMode || 'floating',
          initialCapture: initialCapture.current!.value, principalHint: config.principal_id,
          active: latest.current.active !== false, locale: latestLocale.current, iconUrl: '/homepage/images/mascot-welcome-240.webp',
          handoffId: props.handoffId, captureContext: capture, captureIdentity,
          prepareHandoff: async (intent, baseline) => registerContext(latest.current.prepareHandoff ? latest.current.prepareHandoff(intent, baseline) : baseline),
          authorizeContext: registerContext, tokenProvider: async ({ context }) => (await bootstrap(context)).token,
          validateAdoption: request => researchRequest('/api/integrations/portable-agent/adoptions', { method: 'POST', body: JSON.stringify({
            operation_id: request.receiptRef, run_id: request.runId, session_id: request.sessionId, context_ref: request.contextRef,
          }) }),
          resolveContext: context => researchRequest(`/api/integrations/portable-agent/contexts/${encodeURIComponent(context.ref)}`),
          ...(props.onRestoredContext ? { applyContext: (value: unknown) => latest.current.onRestoredContext?.(value as ResearchPageContext) } : {}),
        })
        element.current = agent; host.current.replaceChildren(agent)
        agent.updateBinding({ active: latest.current.active !== false, busy: latest.current.busy, contextRevision: JSON.stringify(latest.current.pageContext), locale: latestLocale.current })
        setError('')
      } catch (reason) { if (!disposed) setError(reason instanceof Error ? reason.message : '助手连接失败。') }
    }
    void start()
    return () => {
      disposed = true; element.current?.remove(); element.current = null
      setStatusSlot(null)
      setInfoSlot(null)
      setSlots(previous => Object.fromEntries(Object.entries(previous).filter(([key]) => !mounts.has(key))))
    }
  }, [scope, retry, props.displayMode, active])

  useEffect(() => {
    element.current?.updateBinding({ active: props.active !== false, busy: props.busy, contextRevision, locale })
  }, [props.active, props.busy, contextRevision, locale])

  if (!active) return null
  return <>
    <div ref={host} />
    {statusSlot && createPortal(props.renderStatus?.(), statusSlot)}
    {infoSlot && createPortal(<dl className="space-y-1 text-xs text-slate-600">
      <dt className="font-semibold">{locale.startsWith('en') ? 'Current page conditions' : '当前页面条件'}</dt>
      {props.pageContext.calculation.period != null && <dd>{locale.startsWith('en') ? 'Period' : '周期'}：{String(props.pageContext.calculation.period)}</dd>}
      {props.pageContext.calculation.as_of != null && <dd>{locale.startsWith('en') ? 'As of' : '截止日'}：{String(props.pageContext.calculation.as_of)}</dd>}
    </dl>, infoSlot)}
    {error && <div role="alert" className="my-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
      <p>{error}</p><button type="button" className="mt-1 min-h-10 rounded-lg px-2 font-semibold focus-visible:ring-2 focus-visible:ring-accent-500" onClick={() => setRetry(value => value + 1)}>重新连接助手</button>
    </div>}
    {element.current && Object.entries(slots).map(([key, binding]) => createPortal(props.renderArtifact?.(binding, element.current!), binding.container, key))}
  </>
}
