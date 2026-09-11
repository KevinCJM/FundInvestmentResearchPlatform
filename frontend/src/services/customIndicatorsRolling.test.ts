import { beforeEach, describe, expect, it, vi } from 'vitest'

import { buildRollingScalarDraft } from './customIndicators'


describe('buildRollingScalarDraft', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('requests a locked scalar revision and fixed observation window', async () => {
    const payload = {
      definition: {
        name: '5 日滚动年化夏普比率',
        description: '',
        expression: 'rolling_mean(returns, 5, 5)',
        unit: '',
        display_format: 'number' as const,
        precision: 4,
        direction: 'higher_better' as const,
        indicator_type: 'risk_adjusted' as const,
        annual_risk_free_rate_percent: 1.5,
        dsl_version: '2.3.0',
        operator_registry_version: '2.3.0',
        numeric_kernel_version: '2.2.0',
        variable_registry_version: '2.1.0',
        context_schema_version: 'typed-context-v2',
        data_contract_version: 'tushare-eod-v2',
        period_policy: 'all_supported' as const,
        context_kind: 'single_product' as const,
        result_kind: 'time_series' as const,
        output_contract: 'series_bundle',
        output_measure: 'series_bundle',
      },
      validation: { valid: true },
      source: {
        indicator_id: 'builtin-annualized-sharpe-v2',
        indicator_revision: 1,
        name: '年化夏普比率',
      },
    }
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(payload), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )

    const result = await buildRollingScalarDraft({
      indicator_id: 'builtin-annualized-sharpe-v2',
      indicator_revision: 1,
      window_observations: 5,
      min_periods: 5,
    })

    expect(result.source.indicator_id).toBe('builtin-annualized-sharpe-v2')
    expect(fetchMock).toHaveBeenCalledOnce()
    const [path, request] = fetchMock.mock.calls[0]
    expect(path).toBe('/api/custom-indicators/rolling-scalar-draft')
    expect(request?.method).toBe('POST')
    expect(JSON.parse(String(request?.body))).toEqual({
      indicator_id: 'builtin-annualized-sharpe-v2',
      indicator_revision: 1,
      window_observations: 5,
      min_periods: 5,
    })
  })
})
