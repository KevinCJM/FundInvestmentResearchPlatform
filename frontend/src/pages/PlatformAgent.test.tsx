import { afterEach, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import PlatformAgent from './PlatformAgent'
import type { HostAction } from '../integrations/portable-agent/contract'
const captured = vi.hoisted(() => ({ onAction: null as null | ((action: HostAction) => void), request: vi.fn() }))
vi.mock('../integrations/portable-agent/PortableAgentMount', () => ({ default: (props: { onAction: (action: HostAction) => void }) => { captured.onAction = props.onAction; return <div>External assistant</div> } }))
vi.mock('../integrations/portable-agent/client', () => ({ researchRequest: captured.request }))
function Location() { const location = useLocation(); return <output data-testid="location">{location.pathname}{location.search}</output> }
afterEach(() => { cleanup(); vi.clearAllMocks() })
function start() {
  render(<MemoryRouter initialEntries={['/settings/ai-agent']}><PlatformAgent /><Location /></MemoryRouter>)
}
function action(mode: string, signal?: AbortSignal): HostAction {
  return { name: 'navigate', arguments: { capability_id: 'indicator-studio', mode, handoff_id: 'a'.repeat(32) },
    context: {} as HostAction['context'], requestId: 'a'.repeat(32), signal, resolve: vi.fn(), reject: vi.fn() }
}
it('navigation only opens the declared route without an execution handoff', async () => {
  captured.request.mockResolvedValue({ items: [{ id: 'indicator-studio', path: '/settings/indicators-models', actions: ['navigate','execute'], handoff: true }] })
  start(); captured.onAction!(action('navigate'))
  await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/settings/indicators-models'))
  expect(screen.getByTestId('location').textContent).not.toContain('portable_handoff')
})
it('execution forwards only a validated opaque handoff identity', async () => {
  captured.request.mockResolvedValue({ items: [{ id: 'indicator-studio', path: '/settings/indicators-models', actions: ['navigate','execute'], handoff: true }] })
  start(); captured.onAction!(action('execute'))
  await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('portable_handoff='+'a'.repeat(32)))
})
it('late capability lookup cannot navigate after the SDK aborts the host action', async () => {
  let release!: (value: unknown) => void
  captured.request.mockReturnValue(new Promise(resolve => { release = resolve }))
  start(); const controller = new AbortController(); captured.onAction!(action('execute', controller.signal)); controller.abort()
  release({ items: [{ id: 'indicator-studio', path: '/settings/indicators-models', actions: ['execute'], handoff: true }] })
  await new Promise(resolve => setTimeout(resolve, 0))
  expect(screen.getByTestId('location')).toHaveTextContent('/settings/ai-agent')
})
it('an unregistered execution capability is rejected', async () => {
  captured.request.mockResolvedValue({ items: [{ id: 'indicator-studio', path: '/settings/indicators-models', actions: ['navigate'], handoff: false }] })
  start(); const value = action('execute'); captured.onAction!(value)
  await waitFor(() => expect(value.reject).toHaveBeenCalled())
  expect(screen.getByTestId('location')).toHaveTextContent('/settings/ai-agent')
})
