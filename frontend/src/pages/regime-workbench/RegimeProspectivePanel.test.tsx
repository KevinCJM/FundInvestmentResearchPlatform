import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import * as api from '../../services/regimeProspective'
import * as graph from '../../services/regimeGraph'
import RegimeProspectivePanel, { prospectiveMessage, prospectiveReferences } from './RegimeProspectivePanel'
import { prospectiveAssessment, prospectiveProgress, prospectiveProtocol, prospectiveReference, prospectiveSaved } from './regimeProspectiveFixtures'
import { reliabilityDefinition, reliabilityReference } from './regimeReliabilityFixtures'
import { referenceKey } from './RegimeReferenceBinding'
vi.mock('../../services/regimeProspective', async original => ({ ...await original<typeof api>(), registerRegimeProspective: vi.fn(), captureRegimeProspective: vi.fn(), assessRegimeProspective: vi.fn(), listRegimeProspective: vi.fn(), getRegimeProspectiveProgress: vi.fn(), getRegimeProspectiveQualification: vi.fn(), previewRegimeSourceVersion: vi.fn(), confirmRegimeSourceVersion: vi.fn() }))
vi.mock('../../services/regimeGraph', async original => ({ ...await original<typeof graph>(), listHistoricalReferences: vi.fn() }))
const props = { definition: reliabilityDefinition, saved: prospectiveSaved, contextKey: 'PIT-1', dirty: false, valid: true, onApplyQualification: vi.fn() }
const click = (name: string) => fireEvent.click(screen.getByRole('button', { name }))
async function ready() { await waitFor(() => expect(screen.getByRole('button', { name: '刷新前瞻进度' })).toBeEnabled()) }
function existing(assessment: api.ProspectiveAssessment | null = null) {
  vi.mocked(api.listRegimeProspective).mockResolvedValue([prospectiveProtocol])
  vi.mocked(api.getRegimeProspectiveProgress).mockResolvedValue({ ...prospectiveProgress, latest_assessment: assessment })
}
beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(api.listRegimeProspective).mockResolvedValue([])
  vi.mocked(graph.listHistoricalReferences).mockResolvedValue([reliabilityReference, prospectiveReference])
  vi.mocked(api.registerRegimeProspective).mockResolvedValue(prospectiveProtocol)
  vi.mocked(api.getRegimeProspectiveProgress).mockResolvedValue(prospectiveProgress)
  vi.mocked(api.captureRegimeProspective).mockResolvedValue({ status: 'captured' })
  vi.mocked(api.assessRegimeProspective).mockResolvedValue(prospectiveAssessment)
  vi.mocked(api.getRegimeProspectiveQualification).mockResolvedValue(prospectiveAssessment)
})
describe('前瞻验证', () => {
  it.each(['unsaved', 'unfitted', 'dirty', 'invalid', 'inactive', 'mismatch'])('%s cannot register', kind => {
    render(<RegimeProspectivePanel {...props} saved={kind === 'unsaved' ? null : kind === 'unfitted' ? { ...prospectiveSaved, report: { ...prospectiveSaved.report, calibration: null } } : prospectiveSaved} dirty={kind === 'dirty'} valid={kind !== 'invalid'} active={kind !== 'inactive'} definition={kind === 'mismatch' ? { ...reliabilityDefinition, revision: 9 } : reliabilityDefinition} />)
    expect(screen.getByRole('button', { name: '登记前瞻验证' })).toBeDisabled()
    expect(api.registerRegimeProspective).not.toHaveBeenCalled()
  })
  it('register pending → capture → qualified, adoption explicitly rechecks and returns both ids once', async () => {
    render(<RegimeProspectivePanel {...props} />); await ready()
    click('登记前瞻验证'); click('登记前瞻验证')
    await screen.findByText('前瞻验证待完成'); await ready()
    expect(api.registerRegimeProspective).toHaveBeenCalledTimes(1)
    expect(api.registerRegimeProspective).toHaveBeenCalledWith({ calibration_id: 'cal-1' }, expect.any(AbortSignal))
    expect(screen.getByText('尚无记录')).toBeVisible()
    click('记录当前判断'); await ready()
    expect(api.captureRegimeProspective).toHaveBeenCalledWith('protocol-1', {}, expect.any(AbortSignal))
    fireEvent.change(screen.getByLabelText('前瞻检验参考'), { target: { value: referenceKey(prospectiveReference) } })
    vi.mocked(api.getRegimeProspectiveProgress).mockResolvedValue({ ...prospectiveProgress, observations: 252, last_observation_date: '2027-09-14', latest_assessment: prospectiveAssessment })
    click('检验前瞻结果'); await screen.findByText('前瞻检验通过'); await ready()
    expect(props.onApplyQualification).not.toHaveBeenCalled()
    expect(api.assessRegimeProspective).toHaveBeenCalledWith('protocol-1', { reference: { run_id: prospectiveReference.run_id, publication_id: prospectiveReference.publication_id, content_hash: prospectiveReference.content_hash } }, expect.any(AbortSignal))
    click('采用已验证校准'); click('采用已验证校准')
    await waitFor(() => expect(props.onApplyQualification).toHaveBeenCalledWith('cal-1', 'qualification-1'))
    expect(props.onApplyQualification).toHaveBeenCalledTimes(1)
    expect(prospectiveSaved.report.calibration?.deployment_eligible).toBe(false)
  })
  it('reload reads progress assessment instead of frozen initial pending status; no writes', async () => {
    existing(prospectiveAssessment)
    const { unmount } = render(<RegimeProspectivePanel {...props} />)
    await screen.findByText('前瞻检验通过'); unmount()
    render(<RegimeProspectivePanel {...props} />); await screen.findByText('前瞻检验通过')
    expect(api.getRegimeProspectiveProgress).toHaveBeenCalledTimes(2)
    expect(api.registerRegimeProspective).not.toHaveBeenCalled()
    expect(api.captureRegimeProspective).not.toHaveBeenCalled()
  })
  it('partial qualification shows qualified and fallback states and remains adoptable', async () => {
    existing({ ...prospectiveAssessment, outcome: 'partially_qualified', qualified_states: ['expansion'], fallback_states: ['contraction'], state_evidence: [
      { state_id: 'expansion', status: 'qualified', paired_reference_observations: 20, accepted_predictions: 12, complete_regimes: 3, precision: 0.83, recall: 0.5, reasons: [] },
      { state_id: 'contraction', status: 'insufficient_evidence', paired_reference_observations: 4, accepted_predictions: 2, complete_regimes: 1, precision: 1, recall: 0.5, reasons: ['insufficient_complete_state_regimes'] },
    ] })
    render(<RegimeProspectivePanel {...props} />); await ready()
    expect(screen.getByText('部分状态前瞻通过')).toBeVisible()
    expect(screen.getByRole('table', { name: '前瞻逐状态资格' })).toHaveTextContent('扩张')
    expect(screen.getByRole('table', { name: '前瞻逐状态资格' })).toHaveTextContent('证据不足')
    expect(screen.getByText(/其余状态不能授权 Regime 信号/)).toBeVisible()
    expect(screen.getByRole('button', { name: '采用已验证校准' })).toBeEnabled()
  })
  it.each(['rejected', 'pending'] as const)('%s has no adoption', async status => {
    existing({ ...prospectiveAssessment, status, reasons: ['no_forward_captures'], metrics: null })
    render(<RegimeProspectivePanel {...props} />); await ready()
    expect(screen.queryByRole('button', { name: '采用已验证校准' })).not.toBeInTheDocument()
    expect(screen.getByText(/尚无实际前瞻记录/)).toBeVisible()
    expect(screen.getByText(/无法估计/)).toBeVisible()
  })
  it('filters original, older, different definition and revision; keeps exact newer triple', () => {
    expect(prospectiveReferences([reliabilityReference, prospectiveReference, { ...prospectiveReference, definition_id: 'other' }, { ...prospectiveReference, definition_revision: 4 }, { ...prospectiveReference, created_at: '2020-01-01' }], prospectiveProtocol)).toEqual([prospectiveReference])
  })
  it.each(['no_post_registration_day', 'no_post_registration_observation', 'observation_too_old', 'unknown_state_or_invalid_probability', 'reference_label_already_published'])('capture explains %s in Chinese with next action', async reason => {
    existing(); vi.mocked(api.captureRegimeProspective).mockResolvedValue({ status: 'unavailable', reason })
    render(<RegimeProspectivePanel {...props} />); await ready(); click('记录当前判断')
    expect(await screen.findByText(prospectiveMessage(reason))).toBeVisible()
  })
  it.each(['register', 'capture', 'assess', 'progress', 'catalog', 'adopt'])('%s late response after PIT change cannot overwrite or adopt', async method => {
    let finish!: (value: never) => void
    const pending = new Promise<never>(resolve => { finish = resolve })
    const target = method === 'register' ? api.registerRegimeProspective : method === 'capture' ? api.captureRegimeProspective : method === 'assess' ? api.assessRegimeProspective : method === 'catalog' ? api.listRegimeProspective : method === 'progress' ? api.getRegimeProspectiveProgress : api.getRegimeProspectiveQualification
    if (method !== 'register' && method !== 'catalog') existing(method === 'adopt' ? prospectiveAssessment : null)
    vi.mocked(target).mockReturnValueOnce(pending)
    const { rerender } = render(<RegimeProspectivePanel {...props} />)
    if (!['catalog', 'progress'].includes(method)) {
      await ready()
      if (method === 'assess') fireEvent.change(screen.getByLabelText('前瞻检验参考'), { target: { value: referenceKey(prospectiveReference) } })
      click(({ register: '登记前瞻验证', capture: '记录当前判断', assess: '检验前瞻结果', adopt: '采用已验证校准' } as Record<string, string>)[method])
    }
    await waitFor(() => expect(target).toHaveBeenCalled())
    const calls = vi.mocked(target).mock.calls
    const last = calls[calls.length - 1]
    const signal = last[last.length - 1] as AbortSignal
    rerender(<RegimeProspectivePanel {...props} contextKey="PIT-2" active={false} />)
    expect(signal.aborted).toBe(true)
    await act(async () => finish((method === 'catalog' ? [prospectiveProtocol] : method === 'register' ? prospectiveProtocol : method === 'progress' ? { ...prospectiveProgress, latest_assessment: prospectiveAssessment } : prospectiveAssessment) as never))
    expect(props.onApplyQualification).not.toHaveBeenCalled()
    expect(screen.queryByText('前瞻检验通过')).not.toBeInTheDocument()
  })
  it.each(['graph', 'reference', 'calibration', 'report'])('%s identity drops old progress', async change => {
    existing(prospectiveAssessment)
    const { rerender } = render(<RegimeProspectivePanel {...props} />); await screen.findByText('前瞻检验通过')
    vi.mocked(api.listRegimeProspective).mockResolvedValue([])
    rerender(<RegimeProspectivePanel {...props} definition={change === 'graph' ? { ...reliabilityDefinition, graph: { ...reliabilityDefinition.graph, nodes: [] } } : change === 'reference' ? { ...reliabilityDefinition, study: { ...reliabilityDefinition.study!, reference: prospectiveReference } } : reliabilityDefinition} saved={change === 'calibration' ? { ...prospectiveSaved, calibration_id: 'other' } : change === 'report' ? { ...prospectiveSaved, id: 'other' } : prospectiveSaved} />)
    expect(screen.queryByRole('button', { name: '采用已验证校准' })).not.toBeInTheDocument()
  })
  it.each([null, new Error('token=secret traceback /private/hidden'), { detail: null }])('errors sanitized with retry, null response does not grant authority', async error => {
    vi.mocked(api.listRegimeProspective).mockRejectedValueOnce(error)
    render(<RegimeProspectivePanel {...props} />)
    expect(await screen.findByRole('alert')).toHaveTextContent('请检查当前模型')
    expect(screen.queryByText(/secret/)).not.toBeInTheDocument()
    click('刷新前瞻进度'); await ready()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
  it('null progress errors safely; null evidence remains unknown; expired qualification cannot be adopted', async () => {
    existing(); vi.mocked(api.getRegimeProspectiveProgress).mockResolvedValueOnce(null as never)
    render(<RegimeProspectivePanel {...props} />)
    await screen.findByRole('alert')
    vi.mocked(api.getRegimeProspectiveProgress).mockResolvedValue({ ...prospectiveProgress, observations: null, latest_assessment: { ...prospectiveAssessment, reasons: null, metrics: { coverage: null, classification: null }, expires_at: '2000-01-01' } })
    click('刷新前瞻进度'); await screen.findByText('资格已过期')
    expect(screen.queryByRole('button', { name: '采用已验证校准' })).not.toBeInTheDocument()
    expect(screen.getByText(/有效配对覆盖：无法估计/)).toBeVisible()
  })
  it('checks then explicitly confirms successor and adoption carries bindings', async () => {
    existing()
    const preview: api.ProspectiveSourcePreview = { preview_hash: 'a'.repeat(64), protocol_id: prospectiveProtocol.id, previous_id: prospectiveProtocol.id, as_of: '2027-09-14', model_binding_hash: 'binding-next', model_bindings: { source: { snapshot_id: 'next-data', snapshot_generation: 'next-data', source_file: 'index_daily_df.parquet', file_checksum: 'b'.repeat(64) } }, old_observations: 100, new_observations: 101, added_observations: 1, prefix_unchanged: true, reference_id: 'history', reference_revision: 3 }
    const source: api.ProspectiveSourceVersion = { ...preview, id: 'source-next', kind: 'source_version', status: 'accepted', recorded_at: '2027-09-15', reference_definition: { ...prospectiveProtocol.reference_definition, revision: 4 } }
    vi.mocked(api.previewRegimeSourceVersion).mockResolvedValue(preview)
    vi.mocked(api.confirmRegimeSourceVersion).mockResolvedValue(source)
    render(<MemoryRouter><RegimeProspectivePanel {...props} /></MemoryRouter>); await ready()
    click('检查新数据版本'); await ready()
    expect(screen.getByText(/历史前缀一致：100 → 101/)).toBeVisible()
    expect(api.confirmRegimeSourceVersion).not.toHaveBeenCalled()
    vi.mocked(api.getRegimeProspectiveProgress).mockResolvedValue({ ...prospectiveProgress, current_source_version: source, reference_definitions: [prospectiveProtocol.reference_definition, source.reference_definition], latest_assessment: prospectiveAssessment })
    click('确认续接数据'); await ready()
    expect(api.confirmRegimeSourceVersion).toHaveBeenCalledWith(prospectiveProtocol.id, preview.preview_hash, expect.any(AbortSignal))
    expect(screen.getByRole('link', { name: '打开后续参考修订（新页面）' })).toHaveAttribute('href', expect.stringContaining('revision=4'))
    click('采用已验证校准'); await ready()
    expect(props.onApplyQualification).toHaveBeenCalledWith('cal-1', 'qualification-1', source.model_bindings)
    expect(prospectiveReferences([{ ...prospectiveReference, definition_revision: 4 }], prospectiveProtocol, [source.reference_definition])).toHaveLength(1)
  })
  it('late successor check is cancelled when PIT changes', async () => {
    existing()
    let resolve!: (value: api.ProspectiveSourcePreview) => void
    vi.mocked(api.previewRegimeSourceVersion).mockReturnValue(new Promise(r => { resolve = r }))
    const { rerender } = render(<RegimeProspectivePanel {...props} />); await ready(); click('检查新数据版本')
    await waitFor(() => expect(api.previewRegimeSourceVersion).toHaveBeenCalled())
    const signal = vi.mocked(api.previewRegimeSourceVersion).mock.calls[0][1]!
    rerender(<RegimeProspectivePanel {...props} contextKey="changed" active={false} />)
    expect(signal.aborted).toBe(true)
    await act(async () => resolve({} as api.ProspectiveSourcePreview))
    expect(screen.queryByRole('button', { name: '确认续接数据' })).not.toBeInTheDocument()
    expect(api.confirmRegimeSourceVersion).not.toHaveBeenCalled()
  })
  it('source errors expose only allowed messages', async () => {
    expect(prospectiveMessage('数据库失败 token=secret')).not.toContain('secret')
    existing()
    vi.mocked(api.previewRegimeSourceVersion).mockRejectedValue(new Error('历史输入发生修订；该协议不得继续累积证据。'))
    render(<RegimeProspectivePanel {...props} />); await ready(); click('检查新数据版本')
    expect(await screen.findByRole('alert')).toHaveTextContent('历史输入发生修订')
    expect(screen.queryByRole('button', { name: '确认续接数据' })).not.toBeInTheDocument()
  })
})
