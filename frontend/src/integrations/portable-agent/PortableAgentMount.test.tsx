import { act, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import PortableAgentMount from './PortableAgentMount'
import type { ResearchPageContext } from '../../services/researchContracts'

const requests = vi.hoisted(() => ({ register: vi.fn(), bootstrap: vi.fn() }))
vi.mock('./client', () => ({ registerContext: requests.register, bootstrap: requests.bootstrap,
  captureIdentity: vi.fn(), researchRequest: vi.fn() }))

it.each(['registration', 'bootstrap'])('inactive panes skip startup and ignore late %s', async phase => {
  requests.register.mockReset(); requests.bootstrap.mockReset()
  let release!: () => void
  const pending = new Promise<void>(resolve => { release = resolve })
  const moduleUrl = vi.fn(() => '/must-not-load.js')
  requests.register.mockImplementation(async () => { if (phase === 'registration') await pending; return { ref: 'context' } })
  requests.bootstrap.mockImplementation(async () => { await pending; return { get module_url() { return moduleUrl() } } })
  const pageContext: ResearchPageContext = { page: 'historical-regimes', page_instance_id: 'retained-pane',
    context_revision: 0, view_state: 'inherit', calculation: {} }
  const mounted = render(<PortableAgentMount pageContext={pageContext} active={false} />)
  await act(async () => {})
  expect(requests.register).not.toHaveBeenCalled()
  await act(async () => mounted.rerender(<PortableAgentMount pageContext={pageContext} active />))
  expect(requests.register).toHaveBeenCalledTimes(1)
  await act(async () => mounted.rerender(<PortableAgentMount pageContext={pageContext} active={false} />))
  await act(async () => release())
  expect(requests.bootstrap).toHaveBeenCalledTimes(phase === 'registration' ? 0 : 1)
  expect(moduleUrl).not.toHaveBeenCalled()
  expect(mounted.container.querySelector('portable-agent')).toBeNull()
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  mounted.unmount()
})
