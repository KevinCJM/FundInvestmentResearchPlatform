import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import RiskScaleCenter from './RiskScaleCenter'
import RiskScaleWorkspace from './RiskScaleWorkspace'
import RiskScaleVersionView from './RiskScaleVersionView'
import RiskScaleCompare from './RiskScaleCompare'
import { riskScales, RiskScaleError } from '../services/riskScales'
import { riskCapabilities, riskDefinition, riskPreview, riskReference, riskVersion } from '../test/riskScaleFixtures'
import { copyRiskEditor } from '../components/risk-scales/editor'
import { chooseLocale } from '../i18n/runtime'

const researchClock = vi.hoisted(() => ({ day: null as string | null | undefined }))
vi.mock('../app/ResearchContext', () => ({ useResearchDay: () => researchClock.day }))
vi.mock('echarts-for-react', () => ({ default: ({ option, onEvents }: any) => <button data-testid="chart" data-option={JSON.stringify(option)} onClick={() => onEvents.click({ data: { level: 'C3' } })}>chart</button> }))
// A saved draft carries a reference frozen on its own research day. Pinning that to a fixed
// date makes the draft expire the next morning, so it follows the clock like a real one.
const researchDay = () => new Date().toISOString().slice(0, 10)
const draftReference = () => ({ ...riskReference, definition: { ...riskReference.definition, as_of: researchDay() } })
const draftEditor = (step = 0) => ({ ...copyRiskEditor(riskDefinition, researchDay()), step,
  reference: draftReference().definition, referenceVersion: draftReference(), definition: riskDefinition })
const draft = (step = 0) => ({ id: 'draft-fixture', name: riskDefinition.name, scheme_id: riskDefinition.scheme_id, editable_definition: { ...draftEditor(step) }, revision: 3, created_at: '2026-01-01', updated_at: '2026-01-01' })
function mount(path: string) { return render(<MemoryRouter initialEntries={[path]}><Routes><Route path="/settings/risk-scales" element={<RiskScaleCenter />} /><Route path="/settings/risk-scales/new" element={<RiskScaleWorkspace />} /><Route path="/settings/risk-scales/drafts/:draftId" element={<RiskScaleWorkspace />} /><Route path="/settings/risk-scales/versions/:versionId" element={<RiskScaleVersionView />} /><Route path="/settings/risk-scales/compare" element={<RiskScaleCompare />} /></Routes></MemoryRouter>) }
beforeEach(async () => {
  researchClock.day = null
  await chooseLocale('zh-CN')
  vi.spyOn(riskScales, 'capabilities').mockResolvedValue(riskCapabilities)
  vi.spyOn(riskScales, 'catalog').mockResolvedValue({ items: [{ ...riskVersion, artifact_type: 'risk_scale', base_currency: 'CNY', risk_basis_id: riskDefinition.risk_basis_id }], drafts: [draft()], total: 1, next_offset: null })
  vi.spyOn(riskScales, 'defaults').mockResolvedValue({ items: [] })
  vi.spyOn(riskScales, 'reference').mockResolvedValue(riskReference)
  vi.spyOn(riskScales, 'sources').mockResolvedValue({ items: [{ id: 'index:index_daily:000300.SH', kind: 'index', name: '沪深300', code: '000300.SH', status: 'available', coverage: { start_date: '2020-01-02', end_date: '2026-09-17' }, reference_capability: { available: true, supported_fields: ['close'] } }], total: 1, offset: 0, limit: 20, problems: [] })
  vi.spyOn(riskScales, 'draft').mockResolvedValue(draft(2))
  vi.spyOn(riskScales, 'version').mockResolvedValue(riskVersion)
  vi.spyOn(riskScales, 'preview').mockImplementation(async request => ({ ...riskPreview, request_echo: request, result: { ...riskPreview.result, algorithm_id: request.definition.segmentation!.algorithm_id! } }))
  vi.spyOn(riskScales, 'confirm').mockResolvedValue(riskVersion)
  vi.spyOn(riskScales, 'activate').mockResolvedValue({ key: 'CNY:annualized-periodic-volatility-v1', revision: 1, version_id: riskVersion.id })
  vi.spyOn(riskScales, 'retire').mockResolvedValue({ key: 'CNY:annualized-periodic-volatility-v1', revision: 5, version_id: null })
  vi.spyOn(riskScales, 'deleteDraft').mockResolvedValue({ deleted: true, id: 'draft-fixture' })
  vi.spyOn(riskScales, 'saveDraft').mockResolvedValue(draft())
})
afterEach(() => { vi.restoreAllMocks() })

describe('Risk Scale Center routes and exact versions', () => {
  it('shows one simple configuration list without currency or risk-basis selectors', async () => {
    mount('/settings/risk-scales')
    expect(await screen.findByRole('table', { name: '风险等级配置列表' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '新建标尺' })).toHaveAttribute('href', '/settings/risk-scales/new')
    expect(screen.queryByLabelText('本位币')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('风险口径')).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: '查看详情' })).toHaveAttribute('href', '/settings/risk-scales/versions/risk-fixture')
    expect(screen.getAllByRole('link', { name: '编辑' })).toHaveLength(2)
    expect(screen.queryByLabelText('标尺名称')).not.toBeInTheDocument()
  })
  it('shows review reminders in the configuration list without disabling the scale', async () => {
    vi.mocked(riskScales.catalog).mockResolvedValue({ items: [{ ...riskVersion, artifact_type: 'risk_scale', base_currency: 'CNY', risk_basis_id: riskDefinition.risk_basis_id, review_due_at: '2026-09-27', review_status: 'upcoming' }], drafts: [], total: 1, next_offset: null })
    mount('/settings/risk-scales')
    expect(await screen.findByText('临近复核')).toBeInTheDocument()
    expect(screen.getByText('2026-09-27')).toBeInTheDocument()
  })
  it('marks the active system default directly in the configuration list', async () => {
    vi.mocked(riskScales.defaults).mockResolvedValue({ items: [{ key: 'CNY:annualized-periodic-volatility-v1', revision: 2, version_id: 'risk-fixture' }] })
    mount('/settings/risk-scales')
    expect(await screen.findByText('系统默认')).toBeInTheDocument()
  })
  it('shows English labels from shared catalogs', async () => {
    await chooseLocale('en-US'); mount('/settings/risk-scales'); expect(await screen.findByRole('table', { name: 'Risk scale configurations' })).toBeInTheDocument(); expect(screen.getByRole('link', { name: 'New Risk Scale' })).toBeInTheDocument()
  })
  it('deletes a published configuration by retiring it while preserving audit history', async () => {
    vi.mocked(riskScales.defaults).mockResolvedValue({ items: [{ key: 'CNY:annualized-periodic-volatility-v1', revision: 4, version_id: 'risk-fixture' }] })
    mount('/settings/risk-scales')
    const table = await screen.findByRole('table', { name: '风险等级配置列表' })
    const publishedRow = within(table).getAllByRole('row').find(row => within(row).queryByText('已发布'))!
    fireEvent.click(within(publishedRow).getByRole('button', { name: '删除' }))
    expect(screen.getByRole('button', { name: '确认删除' })).toBeDisabled()
    expect(riskScales.retire).not.toHaveBeenCalled()
    fireEvent.click(screen.getByLabelText(/我明确确认同时清空当前默认/))
    fireEvent.click(screen.getByRole('button', { name: '确认删除' }))
    await waitFor(() => expect(riskScales.retire).toHaveBeenCalledWith('risk-fixture', expect.objectContaining({ expected_revision: 4, clear_default: true }), expect.any(AbortSignal)))
  })
  it('requires new consent if the selected version becomes default after the dialog opens', async () => {
    mount('/settings/risk-scales')
    const table = await screen.findByRole('table', { name: '风险等级配置列表' })
    const row = within(table).getAllByRole('row').find(row => within(row).queryByText('已发布'))!
    fireEvent.click(within(row).getByRole('button', { name: '删除' }))
    vi.mocked(riskScales.defaults).mockResolvedValue({ items: [{ key: 'CNY:annualized-periodic-volatility-v1', revision: 4, version_id: 'risk-fixture' }] })
    fireEvent.click(screen.getByRole('button', { name: '确认删除' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('明确确认同时清空')
    expect(riskScales.retire).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: '确认删除' })).toBeDisabled()
  })
  it('loads later pages and compares selected published versions across pages', async () => {
    const item = { ...riskVersion, artifact_type: 'risk_scale', base_currency: 'CNY', risk_basis_id: riskDefinition.risk_basis_id }
    vi.mocked(riskScales.catalog).mockResolvedValueOnce({ items: [item], drafts: [], total: 101, next_offset: 100 })
      .mockResolvedValueOnce({ items: [{ ...item, id: 'risk-right', name: '第二页标尺' }], drafts: [], total: 101, next_offset: null })
    vi.spyOn(riskScales, 'compare').mockResolvedValue({ compatible: true, left: riskVersion,
      right: { ...riskVersion, id: 'risk-right' }, boundary_differences: [0, 0, 0, 0, 0], differences: [] })
    mount('/settings/risk-scales')
    fireEvent.click(await screen.findByRole('checkbox', { name: /进行比较/ }))
    expect(screen.getByRole('button', { name: '比较选中版本' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '加载更多版本' }))
    fireEvent.click(await screen.findByRole('checkbox', { name: /第二页标尺/ }))
    expect(riskScales.catalog).toHaveBeenLastCalledWith('limit=100&offset=100', expect.any(AbortSignal))
    const link = screen.getByRole('link', { name: '比较选中版本' })
    expect(link).toHaveAttribute('href', '/settings/risk-scales/compare?left=risk-fixture&right=risk-right')
    expect(screen.queryByRole('button', { name: '加载更多版本' })).not.toBeInTheDocument()
    fireEvent.click(link)
    await waitFor(() => expect(riskScales.compare).toHaveBeenCalledWith('risk-fixture', 'risk-right', expect.any(AbortSignal)))
  })
  it('deletes a draft directly', async () => {
    mount('/settings/risk-scales')
    const table = await screen.findByRole('table', { name: '风险等级配置列表' })
    const draftRow = within(table).getAllByRole('row').find(row => within(row).queryByText('草稿'))!
    fireEvent.click(within(draftRow).getByRole('button', { name: '删除' }))
    fireEvent.click(screen.getByRole('button', { name: '确认删除' }))
    await waitFor(() => expect(riskScales.deleteDraft).toHaveBeenCalledWith('draft-fixture', 3, expect.any(AbortSignal)))
  })
  it('discovers retired versions through the history filter with read-only actions', async () => {
    vi.mocked(riskScales.catalog).mockResolvedValueOnce({ items: [], drafts: [], total: 0, next_offset: null })
      .mockResolvedValueOnce({ items: [{ ...riskVersion, retired: true, artifact_type: 'risk_scale' }], drafts: [], total: 1, next_offset: null })
    mount('/settings/risk-scales')
    await screen.findByText('尚未配置风险等级')
    fireEvent.click(screen.getByLabelText('显示已停用的历史版本'))
    expect(await screen.findByRole('link', { name: '查看详情' })).toHaveAttribute('href', '/settings/risk-scales/versions/risk-fixture')
    expect(riskScales.catalog).toHaveBeenLastCalledWith('limit=100&include_retired=true', expect.any(AbortSignal))
    expect(screen.queryByRole('button', { name: '删除' })).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: '编辑' })).not.toBeInTheDocument()
  })
  it('reads the requested immutable version without preview or recomputation', async () => {
    mount('/settings/risk-scales/versions/risk-fixture'); await screen.findByText('只读版本'); await screen.findByText(/本页读取发布时冻结/)
    expect(riskScales.version).toHaveBeenCalledWith('risk-fixture', expect.any(AbortSignal)); expect(riskScales.preview).not.toHaveBeenCalled(); expect(screen.queryByText(new RegExp(riskVersion.content_hash))).not.toBeInTheDocument(); expect(screen.getByRole('link', { name: '基于此版本新建研究' })).toHaveAttribute('href', '/settings/risk-scales/new?from=risk-fixture')
  })
  it('explains recalculated default blockers on older immutable versions', async () => {
    vi.mocked(riskScales.version).mockResolvedValue({ ...riskVersion,
      current_default_eligibility: { eligible: false, blockers: [{ code: 'UNSTABLE_CALIBRATION', message: 'Unstable' }] } })
    mount('/settings/risk-scales/versions/risk-fixture')
    expect(await screen.findByRole('button', { name: '设为系统默认' })).toBeDisabled()
    expect(screen.getByText(/101\/200 点复核未稳定/)).toBeInTheDocument()
    expect(riskScales.activate).not.toHaveBeenCalled()
  })
  it('requires independent explicit default confirmation with revision', async () => {
    mount('/settings/risk-scales/versions/risk-fixture'); fireEvent.click(await screen.findByRole('button', { name: '设为系统默认' })); expect(riskScales.activate).not.toHaveBeenCalled(); expect(screen.getByRole('button', { name: '确认执行' })).toBeDisabled(); fireEvent.click(screen.getByLabelText('我已核对版本及本次操作的影响。')); fireEvent.click(screen.getByRole('button', { name: '确认执行' })); await waitFor(() => expect(riskScales.activate).toHaveBeenCalledWith('risk-fixture', { confirm: true, expected_revision: 0 }, expect.any(AbortSignal)))
  })
  it('requires explicit clearing when retiring the current default', async () => {
    vi.mocked(riskScales.defaults).mockResolvedValue({ items: [{ key: 'CNY:annualized-periodic-volatility-v1', revision: 4, version_id: 'risk-fixture' }] })
    mount('/settings/risk-scales/versions/risk-fixture'); fireEvent.click(await screen.findByRole('button', { name: '停止新引用' })); fireEvent.change(screen.getByLabelText('停用依据（至少 5 个字符）'), { target: { value: 'Replaced after review' } }); fireEvent.click(screen.getByLabelText('我已核对版本及本次操作的影响。')); expect(screen.getByRole('button', { name: '确认执行' })).toBeDisabled(); fireEvent.click(screen.getByLabelText(/我明确确认同时清空当前默认/)); fireEvent.click(screen.getByLabelText('我已核对版本及本次操作的影响。')); fireEvent.click(screen.getByRole('button', { name: '确认执行' })); await waitFor(() => expect(riskScales.retire).toHaveBeenCalledWith('risk-fixture', expect.objectContaining({ clear_default: true, expected_revision: 4 }), expect.any(AbortSignal)))
  })
  it('compares exactly two IDs through the backend and suppresses incompatible deltas', async () => {
    vi.spyOn(riskScales, 'compare').mockResolvedValue({ compatible: false, left: riskVersion, right: { ...riskVersion, id: 'risk-right' }, boundary_differences: null, differences: ['currency'] })
    mount('/settings/risk-scales/compare?left=risk-fixture&right=risk-right'); expect(await screen.findByText(/币种或风险语义不兼容/)).toBeInTheDocument(); expect(riskScales.compare).toHaveBeenCalledWith('risk-fixture', 'risk-right', expect.any(AbortSignal)); expect(riskScales.preview).not.toHaveBeenCalled()
  })
  it('does not compare duplicate IDs', () => { const compare = vi.spyOn(riskScales, 'compare'); mount('/settings/risk-scales/compare?left=same&right=same'); expect(screen.getByText('请选择恰好两个不同的版本。')).toBeInTheDocument(); expect(compare).not.toHaveBeenCalled() })
})

describe('guided risk scale editing', () => {
  it('blocks editing actions until the PIT context is resolved', async () => {
    researchClock.day = undefined
    mount('/settings/risk-scales/new')
    fireEvent.change(await screen.findByLabelText('标尺名称'), { target: { value: '等待 PIT' } })
    expect(screen.getByRole('button', { name: '下一步' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '保存草稿' })).toBeDisabled()
    expect(screen.getAllByText(/正在读取当前 PIT 口径/).length).toBeGreaterThan(0)
  })
  it('marks required basics and keeps explanatory text optional', async () => {
    mount('/settings/risk-scales/new')
    const name = await screen.findByLabelText('标尺名称')
    expect(name.closest('label')).toHaveAttribute('data-required', 'true')
    expect(screen.getByLabelText(/适用范围与用途/).closest('label')).not.toHaveAttribute('data-required')
    expect(screen.getByRole('button', { name: '下一步' })).toBeDisabled()
    expect(screen.getByText('请填写标尺名称。')).toBeInTheDocument()
    fireEvent.change(name, { target: { value: '只填名称也可继续' } })
    expect(screen.getByRole('button', { name: '下一步' })).toBeEnabled()
  })
  it('copies inputs while clearing publication identity and results', async () => { mount('/settings/risk-scales/new?from=risk-fixture'); expect(await screen.findByLabelText('标尺名称')).toHaveValue(riskDefinition.name); expect(riskScales.preview).not.toHaveBeenCalled(); expect(screen.queryByRole('table', { name: 'C1–C5 风险等级' })).not.toBeInTheDocument() })
  it('edits an existing configuration in the same version chain', async () => {
    mount('/settings/risk-scales/new?editFrom=risk-fixture')
    expect(await screen.findByLabelText('标尺名称')).toHaveValue(riskDefinition.name)
    fireEvent.click(screen.getByRole('button', { name: '保存草稿' }))
    await waitFor(() => expect(riskScales.saveDraft).toHaveBeenCalledWith(expect.objectContaining({ scheme_id: riskDefinition.scheme_id }), expect.any(AbortSignal)))
  })
  it('restores the real proxy name and code when editing an older frozen version', async () => {
    mount('/settings/risk-scales/new?editFrom=risk-fixture')
    await screen.findByLabelText('标尺名称')
    fireEvent.click(screen.getByRole('button', { name: '下一步' }))
    expect(await screen.findByText('沪深300')).toBeInTheDocument()
    expect(screen.getByText('000300.SH · 指数')).toBeInTheDocument()
    expect(riskScales.sources).toHaveBeenCalledWith('index', '000300.SH', 0, expect.any(AbortSignal))
  })
  it('resumes a saved draft and recalculates immediately when the segmentation method changes', async () => {
    mount('/settings/risk-scales/drafts/draft-fixture'); fireEvent.click(await screen.findByRole('button', { name: '计算前沿与五档' })); await screen.findByLabelText('分档方式'); expect(screen.getByRole('option', { name: '收益等分' })).toBeInTheDocument(); const before = vi.mocked(riskScales.preview).mock.calls.length; fireEvent.change(screen.getByLabelText('分档方式'), { target: { value: 'equal_return_v1' } }); expect(screen.queryByRole('button', { name: '更新前沿与分档预览' })).not.toBeInTheDocument(); await waitFor(() => expect(vi.mocked(riskScales.preview).mock.calls.length).toBeGreaterThan(before)); expect(riskScales.preview).toHaveBeenLastCalledWith(expect.objectContaining({ definition: expect.objectContaining({ segmentation: { algorithm_id: 'equal_return_v1' } }) }), expect.any(AbortSignal)); expect(riskScales.reference).toHaveBeenCalledWith('reference-fixture', expect.any(AbortSignal))
  })
  it('fine-tunes an automatic boundary and recalculates without a manual update button', async () => {
    mount('/settings/risk-scales/drafts/draft-fixture'); fireEvent.click(await screen.findByRole('button', { name: '计算前沿与五档' })); await screen.findByText('分档边界微调'); const before = vi.mocked(riskScales.preview).mock.calls.length; fireEvent.change(screen.getByLabelText('C1 上限（%）'), { target: { value: '2.50' } }); await waitFor(() => expect(vi.mocked(riskScales.preview).mock.calls.length).toBeGreaterThan(before)); expect(riskScales.preview).toHaveBeenLastCalledWith(expect.objectContaining({ definition: expect.objectContaining({ segmentation: expect.objectContaining({ adjusted_caps: [0.025, 0.04, 0.06, 0.08, 0.1] }) }) }), expect.any(AbortSignal)); expect(screen.queryByRole('button', { name: '更新前沿与分档预览' })).not.toBeInTheDocument()
  })
  it('shows human asset names and keeps optional asset bounds collapsed by default', async () => {
    mount('/settings/risk-scales/drafts/draft-fixture')
    const table = await screen.findByRole('table', { name: '历史参数摘要' })
    expect(within(table).getByText('现金')).toBeInTheDocument(); expect(within(table).getByText('权益')).toBeInTheDocument()
    const summary = screen.getByText(/单大类约束.*默认无约束/)
    expect(summary.closest('details')).not.toHaveAttribute('open')
  })
  it('shows table and chart selection in both directions', async () => {
    mount('/settings/risk-scales/drafts/draft-fixture'); fireEvent.click(await screen.findByRole('button', { name: '计算前沿与五档' })); const table = await screen.findByRole('table', { name: 'C1–C5 风险等级' }); fireEvent.click(within(table).getByRole('button', { name: 'C2' })); expect(screen.getByText('C2 代表权重与风险画像')).toBeInTheDocument(); fireEvent.click(screen.getByTestId('chart')); expect(within(table).getByRole('button', { name: 'C3' })).toHaveAttribute('aria-pressed', 'true')
  })
  it('publishes only the exact preview hash after warnings and explicit review', async () => {
    mount('/settings/risk-scales/drafts/draft-fixture'); fireEvent.click(await screen.findByRole('button', { name: '计算前沿与五档' })); fireEvent.click(await screen.findByRole('button', { name: '核对发布内容' })); expect(screen.getByRole('button', { name: '发布新版本' })).toBeDisabled(); fireEvent.click(screen.getByLabelText('Synthetic test warning')); fireEvent.click(screen.getByLabelText(/我已核对当前内容，确认发布该版本/)); fireEvent.click(screen.getByRole('button', { name: '发布新版本' })); await waitFor(() => expect(riskScales.confirm).toHaveBeenCalledWith(expect.objectContaining({ preview_hash: riskPreview.preview_hash, confirm: true, acknowledged_warnings: ['TEST_RESEARCH_WARNING'] }), expect.any(AbortSignal))); expect(riskScales.activate).not.toHaveBeenCalled()
  })
  it('binds loaded draft identity to previews and confirmation and invalidates after saving', async () => {
    vi.spyOn(riskScales, 'updateDraft').mockResolvedValue({ ...draft(2), revision: 4 })
    mount('/settings/risk-scales/drafts/draft-fixture')
    fireEvent.click(await screen.findByRole('button', { name: '计算前沿与五档' }))
    await screen.findByLabelText('分档方式')
    expect(riskScales.preview).toHaveBeenLastCalledWith(expect.objectContaining({ draft_id: 'draft-fixture', draft_revision: 3 }), expect.any(AbortSignal))
    fireEvent.click(screen.getByRole('button', { name: '保存草稿' }))
    fireEvent.click(await screen.findByRole('button', { name: '计算前沿与五档' }))
    await screen.findByLabelText('分档方式')
    expect(riskScales.preview).toHaveBeenLastCalledWith(expect.objectContaining({ draft_id: 'draft-fixture', draft_revision: 4 }), expect.any(AbortSignal))
    expect(riskScales.updateDraft).toHaveBeenCalledWith('draft-fixture', expect.objectContaining({ editable_definition: expect.objectContaining({ step: 2 }), expected_revision: 3 }), expect.any(AbortSignal))
    fireEvent.click(screen.getByRole('button', { name: '核对发布内容' }))
    fireEvent.click(screen.getByLabelText('Synthetic test warning'))
    fireEvent.click(screen.getByLabelText(/我已核对当前内容，确认发布该版本/))
    fireEvent.click(screen.getByRole('button', { name: '发布新版本' }))
    await waitFor(() => expect(riskScales.confirm).toHaveBeenCalledWith(expect.objectContaining({ request: expect.objectContaining({ draft_id: 'draft-fixture', draft_revision: 4 }) }), expect.any(AbortSignal)))
  })
  it('keeps a publish conflict reviewable without claiming success', async () => {
    vi.mocked(riskScales.confirm).mockRejectedValue(new RiskScaleError('PREVIEW_STALE', 'Changed reference', 'reference_input_ref', 409))
    mount('/settings/risk-scales/drafts/draft-fixture'); fireEvent.click(await screen.findByRole('button', { name: '计算前沿与五档' })); fireEvent.click(await screen.findByRole('button', { name: '核对发布内容' })); fireEvent.click(screen.getByLabelText('Synthetic test warning')); fireEvent.click(screen.getByLabelText(/我已核对当前内容，确认发布该版本/)); fireEvent.click(screen.getByRole('button', { name: '发布新版本' })); expect(await screen.findByRole('alert')).toHaveTextContent('Changed reference'); expect(screen.queryByText('只读版本')).not.toBeInTheDocument()
  })
  it('invalidates a late preview after an edit', async () => {
    let resolve!: (value: typeof riskPreview) => void
    vi.mocked(riskScales.preview).mockImplementation(() => new Promise(done => { resolve = done }))
    mount('/settings/risk-scales/drafts/draft-fixture'); fireEvent.click(await screen.findByRole('button', { name: '计算前沿与五档' })); fireEvent.click(screen.getByText(/单大类约束/)); fireEvent.change(screen.getAllByLabelText('权重下限（%）')[0], { target: { value: '10' } }); await act(async () => resolve(riskPreview)); expect(screen.queryByRole('table', { name: 'C1–C5 风险等级' })).not.toBeInTheDocument()
  })
})
