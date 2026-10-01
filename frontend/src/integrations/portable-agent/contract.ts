import type { AgentRelease, ContextCapture, ContextGrant } from './client'

export type ArtifactBinding = {
  runId: string; context: ContextGrant; bindingRevision: string | number; isCurrent: boolean; busy?: boolean
  artifact: { id: string; tool: string; data: Record<string, unknown> }
  container: HTMLElement
  setDispose: (callback: () => void) => void
  setUpdate: (callback: (binding: Partial<ArtifactBinding>) => void) => void
}

export type HostAction = {
  name: string; arguments: Record<string, unknown>; context: ContextGrant; requestId: string
  resolve: (result: unknown) => void; reject: (error: Error) => void
  signal?: AbortSignal
}

export interface PortableAgentElement extends HTMLElement {
  configure(options: {
    app: string; endpoint: string; protocolMajor: number; context: ContextGrant; scopeKey: string
    displayMode: 'floating' | 'embedded' | 'settings'; active: boolean; locale: string; iconUrl: string
    requiredCapabilities?: string[]; handoffId?: string
    expectedRelease?: AgentRelease | null
    principalHint?: string; initialCapture?: ContextCapture
    prepareHandoff?: (intent: Record<string, unknown>, baseline: ContextCapture) => Promise<ContextGrant>
    captureContext: () => ContextCapture
    authorizeContext: (capture: ContextCapture) => Promise<ContextGrant>
    captureIdentity: (capture: ContextCapture) => unknown
    tokenProvider: (request: { context: ContextGrant }) => Promise<string>
    validateAdoption?: (request: Record<string, unknown>) => Promise<{ context: ContextGrant; capture_identity: unknown }>
    resolveContext?: (context: ContextGrant) => Promise<{ context: ContextGrant; identity: unknown; value: unknown }>
    applyContext?: (value: unknown) => void
  }): void
  updateBinding(binding: { active?: boolean; busy?: boolean; contextRevision?: string; locale?: string }): void
  adoptContext(value: { receiptRef: string; runId: string; expectedContextRevision: string | number }): Promise<boolean>
  beginHostAction(): () => void
  proposeMemory(sourceRef: string): Promise<void>
  close(): void
}
