import { afterEach, describe, expect, it, vi } from 'vitest'
import { assessment, fundingStudy } from '../test/mandateFixtures'
import { policyPreview } from '../test/strategicAllocationFixtures'
import { taaExecution } from '../test/tacticalAllocationFixtures'
import { getMandate, previewMandate, previewPolicy, type MandateAssessment } from './strategicAllocation'
import { confirmUniverse, previewUniverse, type UniverseDefinition } from './strategicScope'

const studied = { ...fundingStudy, cma_id: 'cma-1' }
const serve = (value: unknown) => vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => value })))
afterEach(() => { vi.unstubAllGlobals() })

describe('战略范围保存错误对用户可读', () => {
  const definition: UniverseDefinition = {
    name: '长期配置', as_of: '2026-09-12', currency: 'CNY', source: '',
    assets: ['权益', '固收', '现金'].map((name, index) => ({
      id: `asset-${index}`, name, currency: 'CNY', role: index === 2 ? 'liquidity' : 'growth',
      liquidity: 'liquid', rationale: '', source: '',
    })),
  }
  it.each(['preview', 'confirm'])('%s 把多个未知研究代理字段合并为一条服务版本提示', async endpoint => {
    const detail = [0, 1, 2].map(index => ({
      type: 'extra_forbidden', loc: ['body', ...(endpoint === 'confirm' ? ['request'] : []), 'assets', index, 'research_proxy'],
      msg: 'Extra inputs are not permitted', input: { asset_type: 'market' },
    }))
    const fetch = vi.fn(async () => ({ ok: false, status: 422, json: async () => ({ detail }) }))
    vi.stubGlobal('fetch', fetch)
    const action = endpoint === 'preview' ? previewUniverse(definition) : confirmUniverse(definition, 'a'.repeat(64))
    await expect(action).rejects.toThrow('当前页面与服务版本不匹配，暂时无法保存指数、产品或现金收益配置。已填内容已保留，请更新或重启本项目服务后重试。')
    expect(fetch).toHaveBeenCalledTimes(1)
  })
  it.each([
    [{ type: 'float_type', loc: ['body', 'assets', 2, 'research_proxy', 'cash_return'], msg: 'Input should be a valid number' }, '「现金」：现金预期年化收益率请填写 -50% 至 100% 的数字。'],
    [{ type: 'less_than_equal', loc: ['body', 'request', 'assets', 0, 'research_proxy', 'components', 0, 'weight'], msg: 'Input should be less than or equal to 1' }, '「权益」：每个代理的权重请填写 0% 至 100% 的数字，合计须为 100%。'],
    [{ type: 'missing', loc: ['body', 'name'], msg: 'Field required' }, '请填写战略范围名称。'],
    [{ type: 'string_too_long', loc: ['body', 'assets', 1, 'name'], msg: 'String should have at most 120 characters', ctx: { max_length: 120 } }, '「固收」：大类名称最多填写 120 个字符。'],
    [{ type: 'date_from_datetime_parsing', loc: ['body', 'as_of'], msg: 'Input should be a valid date' }, '请选择有效的战略研究日。'],
    [{ type: 'literal_error', loc: ['body', 'assets', 0, 'research_proxy', 'rebalance'], msg: 'Input should be daily or monthly' }, '「权益」：请重新选择代理再平衡。'],
    [{ type: 'value_error', loc: ['body', 'assets', 0, 'research_proxy'], msg: 'Value error, 代理成分不能重复。' }, '「权益」：代理成分不能重复。'],
    [{ type: 'string_pattern_mismatch', loc: ['body', 'preview_hash'], msg: 'String should match pattern' }, '范围信息未能同步，请刷新页面后重试；已填内容已保留。'],
    [{ type: 'unknown', loc: ['body', 'assets', 1, 'internal_field'], msg: 'Unknown Python error' }, '「固收」：这项配置暂时无法保存，请检查后重试；若仍失败，请联系维护人员。'],
    [null, '暂时无法保存研究范围，已填内容已保留，请稍后重试。'],
  ])('结构化校验使用界面名称和中文修正建议 %#', async (issue, expected) => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 422, json: async () => ({ detail: [issue] }) })))
    await expect(previewUniverse(definition)).rejects.toThrow(expected)
  })
  it('保留后端已给出的业务原因，不把名称冲突说成服务故障', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 409, json: async () => ({ detail: { message: '研究范围名称已存在，请修改名称。' } }) })))
    await expect(previewUniverse(definition)).rejects.toThrow('研究范围名称已存在，请修改名称。')
  })
})

describe('investment mandate response evidence', () => {
  it('accepts complete diagnostics and preserves the separate capital estimates', async () => {
    const value = assessment(studied)
    serve(value)
    const result = await previewMandate(studied)
    expect(result.candidates[0].goal_check?.central.required_initial_capital).toBe(950000)
    expect(result.candidates[0].goal_check?.central.gate_required_initial_capital).toBe(975000)
    expect(result.candidates[0].goal_check?.within_limits).toBe(false)
  })

  it.each([
    ['missing goal evidence', (value: MandateAssessment) => { delete value.candidates[0].goal_check }],
    ['missing funding execution', (value: MandateAssessment) => { delete value.funding_execution }],
    ['wrong pass flag', (value: MandateAssessment) => { value.candidates[0].goal_check!.within_limits = true }],
    ['nonfinite probability', (value: MandateAssessment) => { value.candidates[0].goal_check!.central.probability_lower = NaN }],
    ['inverted interval', (value: MandateAssessment) => { value.candidates[0].goal_check!.central.probability_upper = .6 }],
    ['mismatched threshold', (value: MandateAssessment) => { value.candidates[0].goal_check!.threshold = .7 }],
    ['missing funds', (value: MandateAssessment) => { value.funding = null }],
    ['wrong CMA reference', (value: MandateAssessment) => { value.cma!.id = 'other' }],
    ['no CMA but diagnosed', (value: MandateAssessment) => { value.cma = null; value.status = 'diagnosed' }],
    ['no passing candidate but diagnosed', (value: MandateAssessment) => { value.status = 'diagnosed' }],
    ['capital estimate cannot satisfy gate', (value: MandateAssessment) => { value.candidates[0].goal_check!.central.gate_probability_lower = .7 }],
  ] as const)('refuses %s rather than drawing a passing result', async (_name, mutate) => {
    const value = structuredClone(assessment(studied))
    mutate(value)
    serve(value)
    await expect(previewMandate(studied)).rejects.toThrow('诊断缺失或口径不一致')
  })

  it('keeps historical inputs-only versions readable without inventing a diagnostic audit', async () => {
    serve({ id: 'old', name: '历史输入', definition: fundingStudy.definition, assessment: { status: 'inputs_only', candidates: [] } })
    expect((await getMandate('old')).assessment?.status).toBe('inputs_only')
  })

  it('accepts old complete diagnostics without newly-added capital gate outputs', async () => {
    const value = structuredClone(assessment(studied))
    value.funding_model!.version = 'mandate-funding-monthly-lognormal/1.0.0'
    delete value.candidates[0].goal_check!.central.capital_gate_status
    serve(value)
    await expect(previewMandate(studied)).resolves.toHaveProperty('status', 'needs_revision')
  })

  it('requires funding evidence on the SAA preview too', async () => {
    serve({ ...policyPreview, mandate: fundingStudy.definition, funding_execution: taaExecution })
    await expect(previewPolicy(policyPreview.request)).rejects.toThrow('诊断缺失或口径不一致')
  })
})
