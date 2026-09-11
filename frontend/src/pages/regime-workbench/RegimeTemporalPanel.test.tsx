import { fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import RegimeTemporalPanel from './RegimeTemporalPanel'
import { temporalFixture } from '../../test/eventLibraryFixtures'

it('条件通过不冒充已经验证，审计动作明确且可禁用', () => {
  const audit = vi.fn()
  const { rerender } = render(<RegimeTemporalPanel report={temporalFixture} onAudit={audit} />)
  expect(screen.queryByText('本次时点审计通过')).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '因果性审计' }))
  expect(audit).toHaveBeenCalledOnce()
  rerender(<RegimeTemporalPanel report={temporalFixture} onAudit={audit} busy />)
  expect(screen.getByRole('button', { name: '计算与审计中…' })).toBeDisabled()
})
it('人工事后标签不是重绘错误，原因能定位到节点', () => {
  const onNode = vi.fn()
  render(<RegimeTemporalPanel report={{ ...temporalFixture, status: 'retrospective_required', label: '仅事后研究', realtime_supported: false, semantic_hindsight: true,
    reasons: [{ node_id: 'events', code: 'MANUAL_HINDSIGHT', message: '结束日期由事后认定', path: ['events.state'] }] }} onNode={onNode} />)
  expect(screen.getByText(/不是算法错误/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '定位节点' }))
  expect(onNode).toHaveBeenCalledWith('events')
})
it('修改配置后不展示旧版本的通过结论', () => {
  render(<RegimeTemporalPanel report={{ ...temporalFixture, status: 'realtime_verified', label: '本次时点审计通过', verified: true }} stale />)
  expect(screen.getByText('配置已变更，旧审计不再适用')).toBeInTheDocument()
  expect(screen.queryByRole('heading', { name: '本次时点审计通过' })).not.toBeInTheDocument()
})
