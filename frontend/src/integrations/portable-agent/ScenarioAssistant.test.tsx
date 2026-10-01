import { act, fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import ScenarioAssistant from './ScenarioAssistant'
import type { ArtifactBinding } from './contract'

const state = vi.hoisted(() => ({ current: true, request: vi.fn() }))
vi.mock('../../app/useResearchBinding', () => ({ usePageContextRevision: () => 0 }))
vi.mock('./client', () => ({ researchRequest: (...args: unknown[]) => state.request(...args) }))
vi.mock('./PortableAgentMount', () => ({ default: ({ renderArtifact }: { renderArtifact: (binding: ArtifactBinding) => React.ReactNode }) =>
  renderArtifact({ artifact: { id: 'artifact', tool: 'scenarios_preview', data: { artifact_id: 'result' } }, isCurrent: state.current,
    runId: 'run', context: { ref: 'ctx', hash: 'hash', scope_key: 'scope', workspace: 'lab' }, bindingRevision: 0,
    container: document.createElement('div'), setDispose: () => {}, setUpdate: () => {} }) }))

it('读取或校验失败不宣称通过；重试后仍阻止过期成果覆盖工作区', async () => {
  let reject!: (reason: Error) => void
  state.request.mockReturnValueOnce(new Promise((_, fail) => { reject = fail }))
  const apply = vi.fn(), view = vi.fn()
  const props = { page: 'scenario-algorithms' as const, workspace: 'graph' as const, onApply: apply, onView: view }
  const rendered = render(<ScenarioAssistant {...props} />)
  expect(screen.getByText(/正在读取结果/)).toBeInTheDocument()
  expect(screen.queryByText(/结构已核验/)).not.toBeInTheDocument()
  await act(async () => reject(new Error('暂不可用')))
  expect(screen.getByRole('alert')).toHaveTextContent('暂不可用')
  state.request.mockResolvedValueOnce({ definition: { name: '候选' }, valid: false, validation_scope: 'graph', result: {} })
  fireEvent.click(screen.getByRole('button', { name: '重试' }))
  await screen.findByText(/校验未通过/)
  expect(screen.getByRole('button', { name: '填入工作区' })).toBeDisabled()
  state.current = false
  rendered.rerender(<ScenarioAssistant {...props} />)
  expect(screen.getByRole('button', { name: '查看试算结果' })).toBeDisabled()
  expect(apply).not.toHaveBeenCalled(); expect(view).not.toHaveBeenCalled()
})
