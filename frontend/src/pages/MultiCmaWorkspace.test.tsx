import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import StrategicAllocationWorkspace from './StrategicAllocationWorkspace'
import { writeAllocationDraft } from '../app/allocationJourney'
import { cmaVersion, policyBaseline, policyPreview, strategicCatalog } from '../test/strategicAllocationFixtures'
import { previewPolicy, type MultiCmaEvidence, type PolicyRequest, type CompatibilityEvidence, type PolicyPreview } from '../services/strategicAllocation'

const clock = vi.hoisted(() => ({ day: '2026-09-12' as string | undefined }))
vi.mock('../app/ResearchContext', () => ({ useResearchDay: () => clock.day }))
vi.mock('echarts-for-react', () => ({ default: () => <div /> }))
const second = { ...cmaVersion, id: 'cma-2', name: '第二份冻结假设', content_hash: 'd'.repeat(64) }
const versions = [cmaVersion, second]
const root = '/api/strategic-allocation'
const response = (value: unknown, status = 200) => Promise.resolve({ ok: status < 400, json: async () => value } as Response)
function evidence(body: PolicyRequest): MultiCmaEvidence {
  return { mode: 'parameter_average', aggregation_semantics: 'parameter_average', refs: body.cma_refs!,
    sources: body.cma_refs!.map(ref => ({ ...ref, name: versions.find(version => version.id === ref.cma_id)!.name, as_of: '2026-09-12' })),
    effective_returns: [.06, .03], effective_covariance: cmaVersion.covariance, effective_mean_uncertainty: [.01, .01],
    model_disagreement: [[0, 0], [0, 0]], uncertainty_status: 'not_jointly_calibrated', content_hash: 'f'.repeat(64) }
}
function preview(body: PolicyRequest) {
  return { ...policyPreview, request: body, multi_cma: evidence(body), candidates: policyPreview.candidates.map(candidate => ({ ...candidate,
    cross_model_results: body.cma_refs!.map((ref, index) => ({ cma_id: ref.cma_id, cma_hash: ref.content_hash, name: versions[index].name, weight: ref.weight,
      metrics: candidate.metrics, risk_contributions: candidate.risk_contributions, within_limits: index === 0, violations: index ? ['原模型波动超过上限'] : [] })) })) }
}
function install(overrides: Record<string, (init?: RequestInit) => Promise<Response>> = {}) {
  const fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    if (overrides[url]) return overrides[url](init)
    if (url === `${root}/catalog`) return response({ ...strategicCatalog, assumptions: versions.map(version => ({ ...strategicCatalog.assumptions[0], id: version.id, name: version.name })) })
    const version = versions.find(version => url === `${root}/cma/${version.id}`)
    if (version) return response(version)
    if (url === `${root}/policy/preview`) return response(preview(JSON.parse(String(init?.body))))
    if (url === `${root}/policies`) return response({ ...policyBaseline, policy: { ...policyBaseline.policy, cma_id: null, multi_cma: evidence(JSON.parse(String(init?.body)).request) } })
    throw new Error(url)
  })
  vi.stubGlobal('fetch', fetch)
  return fetch
}
const tree = (path = '/pre-investment/saa/policy?alloc=股债分类&mandate=mandate-1') => <MemoryRouter initialEntries={[path]}><StrategicAllocationWorkspace /></MemoryRouter>
async function selectMultiple(user: ReturnType<typeof userEvent.setup>) {
  await screen.findByLabelText('CMA 使用方式')
  await user.selectOptions(screen.getByLabelText('CMA 使用方式'), 'parameter_average')
  await user.selectOptions(screen.getByLabelText('添加已确认 CMA'), 'cma-1')
  await waitFor(() => expect(screen.getByLabelText('添加已确认 CMA')).toBeEnabled())
  await user.selectOptions(screen.getByLabelText('添加已确认 CMA'), 'cma-2')
  await waitFor(() => expect(screen.getByRole('button', { name: '确认使用等权' })).toBeEnabled())
  await user.click(screen.getByRole('button', { name: '确认使用等权' }))
}
beforeEach(() => { clock.day = '2026-09-12'; localStorage.clear(); sessionStorage.clear() })
afterEach(() => { vi.unstubAllGlobals() })

function commonPreview(body: PolicyRequest, passes = true): PolicyPreview {
  const result = preview(body)
  const solver = { status: 'converged' as const, objective_value: .02, lower_bound: .02, objective_gap: 0,
    phase_one_lower_bound: null, iterations: 2, support_cuts: 2 }
  const compatibility: CompatibilityEvidence = { objective: body.compatibility_objective ?? 'minimax_regret',
    regret_basis: 'bounded_continuous_anchor_optima', gate: 'all_frozen_models', joint_solver: solver,
    funding_search_domain: 'one_joint_candidate_with_anchor_cross_diagnostics', anchors: [], limitations: [] }
  return { ...result, compatibility, multi_cma: { ...result.multi_cma, mode: 'compatible_all_models',
    aggregation_semantics: 'all_models_required', primary_evaluation_spec: 'each_frozen_source_model', effective_moments_role: 'display_reference_only' },
    candidates: [{ ...result.candidates[0], id: 'compatible', available: passes, all_models_pass: passes,
      unavailable_reason: passes ? null : '原模型资金检查未通过', solver,
      cross_model_results: result.candidates[0].cross_model_results.map(row => ({ ...row, within_limits: passes, violations: passes ? [] : ['原模型资金检查未通过'] })) }] }
}

it('common mode removes weights and risk-budget inputs, restores references, and adopts all-model evidence', async () => {
  const fetch = install({ [`${root}/policy/preview`]: init => response(commonPreview(JSON.parse(String(init?.body)))),
    [`${root}/policies`]: init => {
      const result = commonPreview(JSON.parse(String(init?.body)).request)
      return response({ ...policyBaseline, policy: { ...policyBaseline.policy, mode: 'compatible_all_models', cma_id: null,
        multi_cma: result.multi_cma, compatibility: result.compatibility } })
    } })
  const user = userEvent.setup(); render(tree()); await selectMultiple(user)
  await user.selectOptions(screen.getByLabelText('CMA 使用方式'), 'compatible_all_models')
  expect(screen.queryByRole('button', { name: '确认使用等权' })).not.toBeInTheDocument()
  await user.click(screen.getByRole('button', { name: '核对模型并进入共同配置' }))
  expect(screen.queryByLabelText('随机候选数')).not.toBeInTheDocument()
  await user.selectOptions(screen.getByLabelText('共同配置的选择目标'), 'maximin_return')
  await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
  await screen.findByRole('region', { name: '共同配置求解结果' })
  await user.click(screen.getByRole('button', { name: '复核此候选' }))
  await user.type(screen.getByLabelText(/采纳理由与复核关注点/), '已检查所有模型的风险与收益边界')
  await user.click(screen.getByRole('button', { name: '确认采用此长期政策' }))
  await screen.findByRole('button', { name: '长期政策已确认' })
  const body = JSON.parse(String(fetch.mock.calls.find(([url]) => String(url) === `${root}/policies`)![1]?.body)).request
  expect(body.compatibility_objective).toBe('maximin_return')
  expect(body.cma_refs.every((ref: { weight: unknown }) => ref.weight === null)).toBe(true)
})

it('shows failed common candidates but does not offer adoption', async () => {
  install({ [`${root}/policy/preview`]: init => response(commonPreview(JSON.parse(String(init?.body)), false)) })
  const user = userEvent.setup(); render(tree()); await selectMultiple(user)
  await user.selectOptions(screen.getByLabelText('CMA 使用方式'), 'compatible_all_models')
  await user.click(screen.getByRole('button', { name: '核对模型并进入共同配置' }))
  await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
  await screen.findByRole('region', { name: '共同配置求解结果' })
  expect(screen.getAllByText(/原模型资金检查未通过/).length).toBeGreaterThan(0)
  expect(screen.queryByRole('button', { name: '复核此候选' })).not.toBeInTheDocument()
})

it('switching out of common mode invalidates results and requires explicit average weights', async () => {
  install({ [`${root}/policy/preview`]: init => response(commonPreview(JSON.parse(String(init?.body)))) })
  const user = userEvent.setup(); render(tree()); await selectMultiple(user)
  await user.selectOptions(screen.getByLabelText('CMA 使用方式'), 'compatible_all_models')
  await user.click(screen.getByRole('button', { name: '核对模型并进入共同配置' }))
  await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
  await screen.findByRole('region', { name: '共同配置求解结果' })
  await user.click(screen.getByRole('button', { name: '2. 选择 LTCMA' }))
  await user.selectOptions(screen.getByLabelText('CMA 使用方式'), 'parameter_average')
  expect(screen.getByRole('button', { name: '核对权重并进入政策比较' })).toBeDisabled()
  expect(screen.queryByRole('region', { name: '共同配置求解结果' })).not.toBeInTheDocument()
})

it.each(['missing', 'downgraded', 'failed-source'])('rejects incomplete common evidence: %s', async defect => {
  const body: PolicyRequest = { ...policyPreview.request, mode: 'compatible_all_models', cma_id: null,
    cma_refs: versions.map(version => ({ cma_id: version.id, content_hash: version.content_hash, weight: null })) }
  const result = commonPreview(body)
  if (defect === 'missing') result.compatibility = undefined
  if (defect === 'downgraded') result.multi_cma!.mode = 'parameter_average'
  if (defect === 'failed-source') result.candidates[0].cross_model_results![0] = { ...result.candidates[0].cross_model_results![0], within_limits: false, violations: ['失败'] }
  install({ [`${root}/policy/preview`]: () => response(result) })
  await expect(previewPolicy(body)).rejects.toThrow()
})

it('keeps explicit source weights and permits aggregate adoption while showing original-model failures', async () => {
  const fetch = install(), user = userEvent.setup(); render(tree()); await selectMultiple(user)
  expect(screen.getByLabelText(`${second.name} · 研究权重（%）`)).toHaveValue('50')
  await user.click(screen.getByRole('button', { name: '核对权重并进入政策比较' }))
  await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
  const table = await screen.findByRole('table', { name: '原 CMA 交叉评估' })
  expect(table).toHaveTextContent(second.name)
  expect(screen.getByText('原模型波动超过上限')).toBeVisible()
  await user.click(screen.getByRole('button', { name: '复核此候选' }))
  await user.type(screen.getByLabelText(/采纳理由与复核关注点/), '理解原模型失败项并采纳融合研究')
  await user.click(screen.getByRole('button', { name: '确认采用此长期政策' }))
  await screen.findByRole('button', { name: '长期政策已确认' })
  const body = JSON.parse(String(fetch.mock.calls.find(([url]) => String(url) === `${root}/policies`)![1]?.body)).request
  expect(body).toMatchObject({ mode: 'parameter_average', cma_id: null, cma_refs: versions.map(version => ({ cma_id: version.id, content_hash: version.content_hash, weight: .5 })) })
})

it('invalid weights block comparison and editing weights discards a late preview', async () => {
  let resolve!: (value: Response) => void
  const fetch = install({ [`${root}/policy/preview`]: () => new Promise(done => { resolve = done }) })
  const user = userEvent.setup(); render(tree()); await selectMultiple(user)
  fireEvent.change(screen.getByLabelText(`${second.name} · 研究权重（%）`), { target: { value: '30' } })
  expect(screen.getByRole('button', { name: '核对权重并进入政策比较' })).toBeDisabled()
  await user.click(screen.getByRole('button', { name: '确认使用等权' }))
  await user.click(screen.getByRole('button', { name: '核对权重并进入政策比较' }))
  await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
  const body = JSON.parse(String(fetch.mock.calls.find(([url]) => String(url) === `${root}/policy/preview`)![1]?.body))
  await user.click(screen.getByRole('button', { name: '2. 选择 LTCMA' }))
  fireEvent.change(screen.getByLabelText(`${second.name} · 研究权重（%）`), { target: { value: '25' } })
  await act(async () => resolve(await response(preview(body))))
  expect(screen.queryByRole('table', { name: '长期政策候选比较' })).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: '4. 确认与交接' })).toBeDisabled()
  expect(JSON.parse(localStorage.getItem('allocation-draft:v1:strategic-policy:股债分类:mandate-1')!).cmaRefs[1].weight).toBe(.25)
})

it('restores only references, requires all frozen reads, and refuses a mismatched hash', async () => {
  install({ [`${root}/cma/cma-2`]: () => response({ ...second, content_hash: 'e'.repeat(64) }) })
  writeAllocationDraft('strategic-policy:股债分类:mandate-1', { mandateId: 'mandate-1', allocationName: '股债分类', strategicUniverseId: '', implementationMappingId: '', settings: policyPreview.request,
    policyName: '保留名称', reason: '保留理由', mode: 'parameter_average', cmaRefs: versions.map(version => ({ cma_id: version.id, content_hash: version.content_hash, weight: .5 })) })
  render(tree())
  expect(await screen.findByRole('alert')).toHaveTextContent('版本或哈希不一致')
  expect(screen.getByRole('button', { name: '核对权重并进入政策比较' })).toBeDisabled()
  expect(screen.getByRole('button', { name: '重新读取所选版本' })).toBeEnabled()
  expect(screen.queryByRole('table', { name: '长期政策候选比较' })).not.toBeInTheDocument()
})

it('preserves old single-CMA drafts and clears confirmation when switching modes', async () => {
  install()
  writeAllocationDraft('strategic-policy:股债分类:mandate-1', { mandateId: 'mandate-1', allocationName: '股债分类', strategicUniverseId: '', implementationMappingId: '', settings: policyPreview.request,
    policyName: '旧研究名称', reason: '旧研究理由', savedCmaId: 'cma-1' })
  render(tree()); const user = userEvent.setup()
  await screen.findByRole('button', { name: '比较符合目标的政策候选' })
  await user.click(screen.getByRole('button', { name: '2. 选择 LTCMA' }))
  await user.selectOptions(screen.getByLabelText('CMA 使用方式'), 'parameter_average')
  expect(screen.getByLabelText(`${cmaVersion.name} · 研究权重（%）`)).toHaveValue('100')
  expect(JSON.parse(localStorage.getItem('allocation-draft:v1:strategic-policy:股债分类:mandate-1')!).policyName).toBe('旧研究名称')
})

it('keeps historical sources readable and does not recompute them', async () => {
  const body: PolicyRequest = { ...policyPreview.request, mode: 'parameter_average', cma_id: null, cma_refs: versions.map(version => ({ cma_id: version.id, content_hash: version.content_hash, weight: .5 })) }
  const fetch = install({ '/api/tactical-allocation/baselines/POLICY-1': () => response({ ...policyBaseline, policy: { ...policyBaseline.policy, cma_id: null, multi_cma: evidence(body) } }) })
  render(tree('/pre-investment/saa/policy?baseline=POLICY-1'))
  const saved = await screen.findByRole('region', { name: '冻结的融合来源与权重' })
  expect(within(saved).getByText(second.name)).toBeVisible()
  expect(fetch.mock.calls).toHaveLength(1)
})

it('fails closed if the backend omits an original-model evaluation', async () => {
  const body: PolicyRequest = { ...policyPreview.request, mode: 'parameter_average', cma_id: null, cma_refs: versions.map(version => ({ cma_id: version.id, content_hash: version.content_hash, weight: .5 })) }
  const result = preview(body); result.candidates[0].cross_model_results.pop()
  install({ [`${root}/policy/preview`]: () => response(result) })
  await expect(previewPolicy(body)).rejects.toThrow('原 CMA 交叉评估不完整')
})

it('switching mode discards an in-flight member read instead of restoring a stale selection', async () => {
  let resolve!: (value: Response) => void
  install({ [`${root}/cma/cma-2`]: () => new Promise(done => { resolve = done }) })
  const user = userEvent.setup(); render(tree())
  await screen.findByLabelText('CMA 使用方式')
  await user.selectOptions(screen.getByLabelText('CMA 使用方式'), 'parameter_average')
  await user.selectOptions(screen.getByLabelText('添加已确认 CMA'), 'cma-2')
  await user.selectOptions(screen.getByLabelText('CMA 使用方式'), 'single')
  await act(async () => resolve(await response(second)))
  expect(screen.getByLabelText('选择已确认 LTCMA')).toHaveValue('')
  expect(screen.getByRole('button', { name: '3. 政策比较' })).toBeDisabled()
  expect(JSON.parse(localStorage.getItem('allocation-draft:v1:strategic-policy:股债分类:mandate-1')!).cmaRefs ?? []).toEqual([])
})

it('changing the knowledge cutoff invalidates every member and the pending policy result', async () => {
  let resolve!: (value: Response) => void
  const fetch = install({ [`${root}/policy/preview`]: () => new Promise(done => { resolve = done }) })
  const user = userEvent.setup(), view = render(tree()); await selectMultiple(user)
  await user.click(screen.getByRole('button', { name: '核对权重并进入政策比较' }))
  await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
  const body = JSON.parse(String(fetch.mock.calls.find(([url]) => String(url) === `${root}/policy/preview`)![1]?.body))
  clock.day = '2026-09-11'; view.rerender(tree())
  await act(async () => resolve(await response(preview(body))))
  expect(screen.getByRole('alert')).toHaveTextContent('晚于平台知识截止')
  expect(screen.getByRole('button', { name: '比较符合目标的政策候选' })).toBeDisabled()
  expect(screen.queryByRole('table', { name: '原 CMA 交叉评估' })).not.toBeInTheDocument()
})
