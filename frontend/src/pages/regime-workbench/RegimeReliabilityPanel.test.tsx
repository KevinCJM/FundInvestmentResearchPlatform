import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../../services/regimeGraph'
import RegimeReliabilityPanel from './RegimeReliabilityPanel'
import { reliabilityDefinition, reliabilityPreviewFixture } from './regimeReliabilityFixtures'
vi.mock('echarts-for-react', () => ({ default: () => <div aria-label="可靠性图" /> }))
vi.mock('../../services/regimeGraph', async original => ({ ...await original<typeof import('../../services/regimeGraph')>(), prepareRegimeGraph: vi.fn(), previewRegimeReliability: vi.fn(), confirmRegimeReliability: vi.fn(), listRegimeReliability: vi.fn(), getRegimeReliability: vi.fn() }))
const props = { definition: reliabilityDefinition, dirty: false, valid: true, contextKey: 'PIT-1', onSave: vi.fn() }
const setDate = () => fireEvent.change(screen.getByLabelText('校准截止日'), { target: { value: '2020-12-31' } })
const click = async (name: string) => act(async () => userEvent.click(screen.getByRole('button', { name })))
beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(api.listRegimeReliability).mockResolvedValue([])
  vi.mocked(api.prepareRegimeGraph).mockResolvedValue({ plan_id: 'p', compile_token: 'token', graph_hash: 'h', runtime_audit: {} })
  vi.mocked(api.previewRegimeReliability).mockResolvedValue(reliabilityPreviewFixture())
})
describe('验证识别能力', () => {
  it('未保存草稿不能验证旧修订；无参考不能声称准确率', async () => {
    const { rerender } = render(<RegimeReliabilityPanel {...props} dirty />)
    setDate()
    expect(screen.getByRole('button', { name: '验证识别能力' })).toBeDisabled()
    expect(screen.getByText(/请先保存当前模型/)).toBeVisible()
    rerender(<RegimeReliabilityPanel {...props} definition={{ ...reliabilityDefinition, study: { purpose: 'realtime_recognition', family: 'custom' } }} />)
    expect(screen.getByText('选择历史参考后才能验证。')).toBeVisible()
    expect(api.previewRegimeReliability).not.toHaveBeenCalled()
  })
  it('精确请求、确认仅发送 canonical request/hash；显示不充分与非部署状态', async () => {
    const preview = reliabilityPreviewFixture()
    vi.mocked(api.confirmRegimeReliability).mockResolvedValue({ ...preview, id: 'report-1', calibration_id: 'cal-1', created_at: '2026-09-14', immutable: true, content_hash: 'saved' })
    render(<RegimeReliabilityPanel {...props} />); setDate()
    await click('验证识别能力')
    expect(api.previewRegimeReliability).toHaveBeenCalledWith(expect.objectContaining({ definition_id: 'recognition', revision: 2, reference: reliabilityDefinition.study!.reference }), 'token', expect.any(AbortSignal))
    expect(await screen.findByText('仅回顾性评分')).toBeVisible()
    expect(screen.getByText('样本不足')).toBeVisible()
    expect(screen.getByText('不可部署')).toBeVisible()
    expect(screen.getByText(/原始数值为 1 不代表 100% 可信/)).toBeVisible()
    expect(screen.getByText(/2024-12-31 · 识别 扩张 · 参考 参考未分类/)).toBeVisible()
    await click('保存验证报告')
    expect(api.confirmRegimeReliability).toHaveBeenCalledWith(preview, expect.any(AbortSignal))
    expect(await screen.findByText('已保存的不可变报告')).toBeVisible()
    expect(screen.getByRole('button', { name: '采用此校准结果' })).toBeDisabled()
  })
  it.each(['reference', 'graph', 'PIT', 'policy', 'cancel'] as const)('%s 变化后迟到结果不覆盖当前上下文', async change => {
    let resolve!: (value: api.RegimeReliabilityPreview) => void
    vi.mocked(api.previewRegimeReliability).mockImplementation(() => new Promise(done => { resolve = done }))
    const { rerender } = render(<RegimeReliabilityPanel {...props} />); setDate()
    await click('验证识别能力')
    await waitFor(() => expect(api.previewRegimeReliability).toHaveBeenCalledOnce())
    if (change === 'policy') fireEvent.change(screen.getByLabelText('校准截止日'), { target: { value: '2021-12-31' } })
    else if (change === 'cancel') await click('取消验证请求')
    else if (change === 'PIT') rerender(<RegimeReliabilityPanel {...props} contextKey="PIT-2" />)
    else rerender(<RegimeReliabilityPanel {...props} definition={change === 'reference' ? { ...reliabilityDefinition, study: { ...reliabilityDefinition.study!, reference: { ...reliabilityDefinition.study!.reference!, run_id: 'other' } } } : { ...reliabilityDefinition, graph: { ...reliabilityDefinition.graph, nodes: [] } }} />)
    await act(async () => resolve(reliabilityPreviewFixture()))
    expect(screen.queryByLabelText('识别能力验证报告')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '保存验证报告' })).not.toBeInTheDocument()
  })
  it('错误可直接重试，已保存报告可按精确版本重新载入', async () => {
    vi.mocked(api.previewRegimeReliability).mockRejectedValueOnce(new Error('样本日期轴不一致'))
    const preview = reliabilityPreviewFixture()
    const saved = { ...preview, id: 'saved-1', calibration_id: null, created_at: '2026-09-14', immutable: true as const, content_hash: 'saved' }
    vi.mocked(api.listRegimeReliability).mockResolvedValue([{ id: saved.id, calibration_id: null, created_at: saved.created_at, definition_id: 'recognition', revision: 2, reference: preview.request.reference, status: preview.report.status, calibration: preview.report.calibration }])
    vi.mocked(api.getRegimeReliability).mockResolvedValue(saved)
    render(<RegimeReliabilityPanel {...props} />); setDate(); await click('验证识别能力')
    expect(await screen.findByRole('alert')).toHaveTextContent('样本日期轴不一致')
    await click('验证识别能力')
    expect(await screen.findByLabelText('识别能力验证报告')).toBeVisible()
    fireEvent.click(screen.getByText('已保存报告'))
    await act(async () => userEvent.selectOptions(screen.getByLabelText('已保存验证报告'), saved.id))
    await click('载入报告')
    expect(api.getRegimeReliability).toHaveBeenCalledWith(saved.id, expect.any(AbortSignal))
    expect(await screen.findByText('已保存的不可变报告')).toBeVisible()
  })
  it('新修订可精确读取自己绑定的旧修订校准报告，但不会放宽到其他旧报告', async () => {
    const preview = reliabilityPreviewFixture()
    const bound = { ...preview, id: 'bound-report', calibration_id: 'bound-report', created_at: '2026-09-15', immutable: true as const, content_hash: 'bound' }
    const other = { id: 'other-old-report', calibration_id: 'other-old-report', created_at: '2026-09-14', definition_id: 'recognition', revision: 2, reference: preview.request.reference, status: preview.report.status, calibration: preview.report.calibration }
    vi.mocked(api.listRegimeReliability).mockResolvedValue([
      { id: bound.id, calibration_id: bound.calibration_id, created_at: bound.created_at, definition_id: 'recognition', revision: 2, reference: preview.request.reference, status: preview.report.status, calibration: preview.report.calibration },
      other,
    ])
    vi.mocked(api.getRegimeReliability).mockResolvedValue(bound)
    const definition = { ...reliabilityDefinition, revision: 3, study: { ...reliabilityDefinition.study!, calibration_id: bound.id } }
    render(<RegimeReliabilityPanel {...props} definition={definition} />)
    fireEvent.click(screen.getByText('已保存报告'))
    const select = await screen.findByLabelText('已保存验证报告')
    await waitFor(() => expect(select).toHaveValue(bound.id))
    expect(screen.queryByRole('option', { name: /other-old-report/ })).not.toBeInTheDocument()
    expect(Array.from((select as HTMLSelectElement).options).map(option => option.value)).toEqual(['', bound.id])
    await click('载入报告')
    expect(api.getRegimeReliability).toHaveBeenCalledWith(bound.id, expect.any(AbortSignal))
    expect(await screen.findByText('已保存的不可变报告')).toBeVisible()
  })
})

it('高级设置只修改策略，检查按钮才执行；重采样次数同步约束有效次数', async () => {
  render(<RegimeReliabilityPanel {...props} />); setDate()
  fireEvent.click(screen.getByText('高级检查设置'))
  fireEvent.change(screen.getByLabelText('每块观测数'), { target: { value: '12' } })
  fireEvent.change(screen.getByLabelText('重采样次数'), { target: { value: '40' } })
  expect(screen.getByLabelText('最少有效重采样')).toHaveValue(40)
  expect(api.previewRegimeReliability).not.toHaveBeenCalled()
  await click('检查稳定性')
  expect(api.previewRegimeReliability).toHaveBeenCalledWith(expect.objectContaining({ policy: expect.objectContaining({ stability: expect.objectContaining({ enabled: true }), bootstrap: expect.objectContaining({ block_length: 12, replicates: 40, minimum_valid_replicates: 40 }) }) }), 'token', expect.any(AbortSignal))
})

it('changing calibrator invalidates the adopted authority', async () => {
  const invalidate = vi.fn()
  render(<RegimeReliabilityPanel {...props} definition={{ ...reliabilityDefinition, study: { ...reliabilityDefinition.study!, calibration_id: 'cal', qualification_id: 'qual' } }} onInvalidateCalibration={invalidate} />)
  fireEvent.click(screen.getByText('样本门槛与校准设置'))
  fireEvent.change(screen.getByLabelText('校准方法'), { target: { value: 'temperature' } })
  expect(invalidate).toHaveBeenCalledOnce()
})
