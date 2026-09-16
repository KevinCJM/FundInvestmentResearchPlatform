import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, expect, it, vi } from 'vitest'
import * as api from '../../services/regimeGraph'
import type { RegimeQualityPreview } from '../../services/regimeDiagnostics'
import RegimeQualityPanel from './RegimeQualityPanel'
import { qualityDefinition, qualityPreviewFixture } from './regimeQualityFixtures'
vi.mock('../../services/regimeGraph', async original => ({ ...await original<typeof import('../../services/regimeGraph')>(), prepareRegimeGraph: vi.fn(), previewRegimeQuality: vi.fn(), confirmRegimeQuality: vi.fn(), listRegimeQuality: vi.fn(), getRegimeQuality: vi.fn() }))
const props = { definition: qualityDefinition, dirty: false, valid: true, asOf: '', contextKey: 'PIT-1', active: true, onSave: vi.fn() }
const click = (name: string) => fireEvent.click(screen.getByRole('button', { name }))
beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(api.listRegimeQuality).mockResolvedValue([])
  vi.mocked(api.prepareRegimeGraph).mockResolvedValue({ plan_id: 'p', compile_token: 'token', graph_hash: 'g', runtime_audit: {} })
  vi.mocked(api.previewRegimeQuality).mockResolvedValue(qualityPreviewFixture())
})
it('挂载不计算；未保存草稿不能检查旧版本', async () => {
  render(<RegimeQualityPanel {...props} dirty />)
  expect(screen.getByRole('button', { name: '检查划分质量' })).toBeDisabled()
  expect(screen.getByText(/请先保存当前历史状态定义/)).toBeVisible()
  click('先保存历史状态定义')
  expect(props.onSave).toHaveBeenCalled()
  expect(api.prepareRegimeGraph).not.toHaveBeenCalled()
})
it('预览、确认只传请求和 hash、重载不可变报告', async () => {
  const preview = qualityPreviewFixture(), saved = { ...preview, id: 'quality-1', created_at: '2026-09-15', immutable: true as const, content_hash: 'stored-hash' }
  vi.mocked(api.confirmRegimeQuality).mockResolvedValue(saved)
  vi.mocked(api.getRegimeQuality).mockResolvedValue(saved)
  vi.mocked(api.listRegimeQuality).mockResolvedValue([{ id: saved.id, created_at: saved.created_at, definition_id: 'history', revision: 3, status: 'diagnostic_only' }])
  render(<RegimeQualityPanel {...props} />)
  expect(api.previewRegimeQuality).not.toHaveBeenCalled()
  click('检查划分质量')
  expect(await screen.findByLabelText('历史划分质量报告')).toHaveTextContent('分类覆盖 80.0%')
  expect(screen.getByText(/首端未分类 5/)).toHaveTextContent('尾端未分类 15')
  expect(screen.getByText(/LTCMA研究样本：全部状态达到区间数量门槛/)).toBeVisible()
  expect(screen.getByLabelText('市场状态实测时间尺度')).toHaveTextContent('年化状态转换 0.60 次')
  expect(screen.getByLabelText('市场状态实测时间尺度')).toHaveTextContent('此旧报告按首末样本点跨度统计日历天数')
  expect(screen.getByRole('table', { name: '市场状态持续期统计' })).toHaveTextContent('450.00 / 600.00 / 750.00')
  expect(screen.getByRole('table', { name: '历史状态区间统计' })).toHaveTextContent('独立完整区间')
  expect(screen.getByRole('table', { name: '历史状态区间统计' })).toHaveTextContent('数量达标')
  expect(api.previewRegimeQuality).toHaveBeenCalledWith(expect.objectContaining({ definition_id: 'history', revision: 3, mode: 'retrospective', as_of: null, policy: expect.objectContaining({ minimum_state_episodes_for_estimation: 3 }) }), 'token', expect.any(AbortSignal))
  click('确认保存质量报告')
  expect(await screen.findByText('质量报告已保存')).toBeVisible()
  expect(api.confirmRegimeQuality).toHaveBeenCalledWith({ request: preview.request, preview_hash: preview.preview_hash }, expect.any(AbortSignal))
  fireEvent.click(screen.getByText('已保存质量报告', { selector: 'summary' }))
  fireEvent.change(screen.getByLabelText('已保存质量报告'), { target: { value: saved.id } })
  click('载入质量报告')
  await waitFor(() => expect(api.getRegimeQuality).toHaveBeenCalledWith(saved.id, expect.any(AbortSignal)))
  expect(await screen.findByText('质量报告已保存')).toBeVisible()
})
it('新报告按已声明的下一状态边界解释持续期', async () => {
  const preview = qualityPreviewFixture()
  preview.report.horizon_profile!.calendar_boundary = 'observation_inclusive_next_observation_exclusive'
  vi.mocked(api.previewRegimeQuality).mockResolvedValue(preview)
  render(<RegimeQualityPanel {...props} />)
  click('检查划分质量')
  expect(await screen.findByLabelText('市场状态实测时间尺度')).toHaveTextContent('持续到下一状态的首个观测日（不含）')
  expect(screen.queryByText(/此旧报告按首末样本点跨度/)).not.toBeInTheDocument()
})
it.each(['model', 'policy', 'PIT', 'asOf', 'source', 'inactive'] as const)('%s 变化拒绝迟到报告', async change => {
  let resolve!: (value: RegimeQualityPreview) => void
  vi.mocked(api.previewRegimeQuality).mockImplementation(() => new Promise(done => { resolve = done }))
  const { rerender } = render(<RegimeQualityPanel {...props} />)
  click('检查划分质量'); await waitFor(() => expect(api.previewRegimeQuality).toHaveBeenCalledOnce())
  if (change === 'policy') fireEvent.click(screen.getByLabelText('参数扰动'))
  else rerender(<RegimeQualityPanel {...props} {...(change === 'PIT' ? { contextKey: 'PIT-2' } : change === 'asOf' ? { asOf: '2024-01-01' } : change === 'inactive' ? { active: false } : { definition: { ...qualityDefinition, ...(change === 'model' ? { revision: 4 } : { graph: { ...qualityDefinition.graph, nodes: [] } }) } })} />)
  await act(async () => resolve(qualityPreviewFixture()))
  expect(screen.queryByLabelText('历史划分质量报告')).not.toBeInTheDocument()
})
it('错误后可重试；拒绝其他修订报告', async () => {
  vi.mocked(api.previewRegimeQuality).mockRejectedValueOnce(new Error('数据暂不可用，请重试。')).mockResolvedValueOnce({ ...qualityPreviewFixture(), request: { ...qualityPreviewFixture().request, revision: 999 } })
  render(<RegimeQualityPanel {...props} />); click('检查划分质量')
  expect(await screen.findByRole('alert')).toHaveTextContent('数据暂不可用')
  click('检查划分质量')
  expect(await screen.findByRole('alert')).toHaveTextContent('报告的定义修订或截至日不同')
  expect(screen.queryByRole('button', { name: '确认保存质量报告' })).not.toBeInTheDocument()
})
