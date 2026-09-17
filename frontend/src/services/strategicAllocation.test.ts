import { afterEach, describe, expect, it, vi } from 'vitest'
import { assessment, fundingStudy } from '../test/mandateFixtures'
import { policyPreview } from '../test/strategicAllocationFixtures'
import { taaExecution } from '../test/tacticalAllocationFixtures'
import { getMandate, previewMandate, previewPolicy, type MandateAssessment } from './strategicAllocation'

const studied = { ...fundingStudy, cma_id: 'cma-1' }
const serve = (value: unknown) => vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => value })))
afterEach(() => { vi.unstubAllGlobals() })

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
