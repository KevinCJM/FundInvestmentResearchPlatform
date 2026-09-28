import { useState } from 'react'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import LtcmaScenarioFields, { scenarioSelectionIssue } from './LtcmaScenarioFields'
import LtcmaInputFields from './LtcmaInputFields'
import LtcmaScenarioResults from './LtcmaScenarioResults'
import LtcmaResults from './LtcmaResults'
import { changeMethod } from './model'
import { cmaDraftFromDefinition, type CmaDraft, type CmaPreview } from '../../services/strategicAllocation'
import { cmaModelInputError } from '../../services/cmaModelTypes'
import { cmaHandoffIssue } from '../../services/ltcmaHandoff'
import { ltcmaSaaIssue, type LtcmaOptions } from '../../services/ltcma'
import { ltcmaCapabilities, ltcmaDefinition, ltcmaItem, ltcmaOptions, ltcmaVersion } from '../../test/ltcmaFixtures'

const reference = { run_id: 'historical-1', publication_id: 'published-1', content_hash: 'a'.repeat(64) }
const historical = { id: 'historical-1', name: '股票牛熊情景', content_hash: 'a'.repeat(64), frequency: 'daily', reference,
  available: true, reasons: [], states: [{ id: 'up', label: '上涨' }, { id: 'down', label: '下跌' }] }
const realtime = { id: 'realtime-1', name: '当前牛熊概率', content_hash: 'b'.repeat(64), as_of: ltcmaDefinition.as_of,
  reference, available: true, reasons: [] }
const options: LtcmaOptions = { ...ltcmaOptions, scenario_options: { historical_references: [historical], realtime_runs: [realtime], default_historical_id: historical.id } }
function chooseHistory(value: string) {
  fireEvent.click(screen.getByLabelText('历史情景研究'))
  fireEvent.click(within(screen.getByRole('menu', { name: '历史情景研究' })).getAllByRole('menuitemradio').find(item => (item as HTMLButtonElement).value === value)!)
}
const draft = (method: 'long_term_scenario' | 'conditional_scenario') => changeMethod(cmaDraftFromDefinition(ltcmaDefinition), method)

describe('scenario input workflow', () => {
  it('requires an explicit historical choice before matching calibrated real-time evidence', async () => {
    let latest = draft('conditional_scenario')
    function Editor() {
      const [value, setValue] = useState(latest)
      return <LtcmaScenarioFields value={value} options={options} onChange={next => { latest = next; setValue(next) }} sourceLabels={{}} onLabels={() => {}} />
    }
    render(<MemoryRouter><Editor /></MemoryRouter>)
    expect(latest.model).toMatchObject({ run_ref: { id: '' }, realtime_ref: { id: '' } })
    expect(screen.getByLabelText('历史情景研究')).toHaveValue('')
    expect(scenarioSelectionIssue(latest, options, key => key)).toBe('scenarioChooseHistory')
    chooseHistory('historical-1:published-1')
    await waitFor(() => expect(latest.model).toMatchObject({ run_ref: { id: historical.id, content_hash: historical.content_hash }, historical_reference: reference,
      realtime_ref: { id: realtime.id, content_hash: realtime.content_hash } }))
    expect(latest.model).not.toHaveProperty('probabilities')
    expect(latest.model).not.toHaveProperty('shrinkage')
    expect(screen.getByLabelText('历史情景研究')).toHaveValue('historical-1:published-1')
    expect(screen.getAllByRole('combobox')).toHaveLength(1)
    expect(screen.getByRole('option', { name: '约 6 个月' })).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('预测区间'), { target: { value: '252' } })
    expect(latest.model).toMatchObject({ horizon_days: 252 })
    expect(cmaModelInputError(latest.model!)).toBeNull()
    chooseHistory('')
    expect(latest.model).toMatchObject({ run_ref: { id: '', content_hash: '' }, realtime_ref: { id: '', content_hash: '' } })
    expect(screen.getByLabelText('历史情景研究')).toHaveValue('')
  })

  it('keeps a lone eligible historical study unselected after options load or refresh', () => {
    const onChange = vi.fn()
    const value = draft('long_term_scenario')
    const editor = (ready: boolean) => <MemoryRouter><LtcmaScenarioFields value={value} options={options} optionsReady={ready}
      onChange={onChange} sourceLabels={{}} onLabels={() => {}} /></MemoryRouter>
    const { rerender } = render(editor(false))
    rerender(editor(true))
    expect(screen.getByLabelText('历史情景研究')).toHaveValue('')
    expect(onChange).not.toHaveBeenCalled()
    chooseHistory('historical-1:published-1')
    expect(onChange).toHaveBeenCalledOnce()
    expect(onChange.mock.calls[0][0].model).toMatchObject({ run_ref: { id: historical.id, content_hash: historical.content_hash }, historical_reference: reference })
  })

  it('restores a saved historical binding as an editable dropdown without changing the draft', () => {
    const value = draft('long_term_scenario')
    if (value.model?.method !== 'long_term_scenario') throw new Error('long-term draft')
    value.model = { ...value.model, run_ref: { id: historical.id, content_hash: historical.content_hash }, historical_reference: reference }
    const onChange = vi.fn()
    render(<MemoryRouter><LtcmaScenarioFields value={value} options={options} onChange={onChange} sourceLabels={{}} onLabels={() => {}} /></MemoryRouter>)
    expect(screen.getByRole('button', { name: '历史情景研究' })).toHaveValue('historical-1:published-1')
    expect(onChange).not.toHaveBeenCalled()
  })

  it.each(['empty', 'filtered', 'selected'] as const)('opens retrospective identification from the %s dropdown without changing the binding', state => {
    const value = draft('long_term_scenario')
    if (value.model?.method !== 'long_term_scenario') throw new Error('long-term draft')
    if (state === 'selected') value.model = { ...value.model, run_ref: { id: historical.id, content_hash: historical.content_hash }, historical_reference: reference }
    const candidates = state === 'empty' ? [] : [{ ...historical, available: state === 'selected' }]
    const onChange = vi.fn()
    function Location() {
      const location = useLocation()
      return <output aria-label="destination">{location.pathname}{location.search}</output>
    }
    render(<MemoryRouter><Location /><LtcmaScenarioFields value={value}
      options={{ ...options, scenario_options: { ...options.scenario_options!, historical_references: candidates } }}
      onChange={onChange} sourceLabels={{}} onLabels={() => {}} /></MemoryRouter>)
    fireEvent.click(screen.getByLabelText('历史情景研究'))
    const action = screen.getByRole('menuitem', { name: '打开情景算法中心配置事后情景' })
    expect(action).toBeEnabled()
    fireEvent.click(action)
    expect(screen.getByLabelText('destination')).toHaveTextContent('/settings/scenario-algorithms?center=market-state&stage=historical')
    expect(onChange).not.toHaveBeenCalled()
  })

  it('requires a choice when different historical studies exist and does not bind by matching names', () => {
    const onChange = vi.fn()
    const alternatives = { ...options, scenario_options: { ...options.scenario_options!, historical_references: [historical, { ...historical, id: 'historical-2', name: '增长通胀情景', content_hash: 'c'.repeat(64) }] } }
    render(<MemoryRouter><LtcmaScenarioFields value={draft('long_term_scenario')} options={alternatives} onChange={onChange} sourceLabels={{}} onLabels={() => {}} /></MemoryRouter>)
    expect(screen.getByLabelText('历史情景研究')).toHaveValue('')
    expect(onChange).not.toHaveBeenCalled()
    chooseHistory('historical-1:published-1')
    expect(onChange.mock.calls[0][0].model.historical_reference).toEqual(reference)
  })

  it('explains excluded studies and keeps their controls unavailable', () => {
    const filtered = { ...options, scenario_options: { ...options.scenario_options!, historical_references: [{ ...historical, available: false,
      reasons: [{ code: 'AFTER_PIT', message: '这项情景研究在当前研究日之后才可获得。' }] }] } }
    render(<MemoryRouter><LtcmaScenarioFields value={draft('long_term_scenario')} options={filtered} onChange={vi.fn()} sourceLabels={{}} onLabels={() => {}} /></MemoryRouter>)
    expect(screen.getByText('已有情景研究，但均不符合当前研究日或数据要求。')).toBeVisible()
    expect(screen.getByText('这项情景研究在当前研究日之后才可获得。')).toBeVisible()
    fireEvent.click(screen.getByLabelText('历史情景研究'))
    const unavailable = screen.getByRole('menuitemradio', { name: /股票牛熊情景/ })
    expect(unavailable).toHaveAttribute('aria-disabled', 'true')
    fireEvent.mouseEnter(unavailable)
    expect(screen.getByRole('tooltip')).toHaveTextContent('这项情景研究在当前研究日之后才可获得。')
    expect(screen.getByRole('link', { name: '打开情景研究中心' })).toHaveAttribute('href', '/settings/scenario-algorithms')
  })

  it('does not use stale options while the PIT date is being refreshed', () => {
    const onChange = vi.fn()
    render(<MemoryRouter><LtcmaScenarioFields value={draft('long_term_scenario')} options={options} optionsReady={false} onChange={onChange} sourceLabels={{}} onLabels={() => {}} /></MemoryRouter>)
    expect(screen.getByRole('status')).toHaveTextContent('正在按研究日匹配情景研究')
    expect(onChange).not.toHaveBeenCalled()
    expect(scenarioSelectionIssue(draft('long_term_scenario'), ltcmaOptions, key => key)).toBe('scenarioOptionsUnavailable')
  })

  it('retains one advanced section and readable historical window names', () => {
    const value = draft('long_term_scenario')
    render(<MemoryRouter><LtcmaInputFields value={value} options={options} capabilities={ltcmaCapabilities} cutoff={value.as_of} platformDay={value.as_of}
      defaultName="情景研究" onChange={vi.fn()} sourceLabels={{}} onLabels={() => {}} /></MemoryRouter>)
    expect(screen.getAllByText('高级设置与口径')).toHaveLength(1)
    expect(screen.getByRole('option', { name: '近 5 年', hidden: true })).toBeInTheDocument()
    expect(screen.queryByText('名称待补充')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('对角收缩强度（%）')).not.toBeInTheDocument()
  })

  it.each(['long_term_scenario', 'conditional_scenario'] as const)('keeps completed product-asset classifications in advanced settings for %s', method => {
    const value = draft(method)
    render(<MemoryRouter><LtcmaInputFields value={value} options={options} capabilities={ltcmaCapabilities} cutoff={value.as_of} platformDay={value.as_of}
      defaultName="情景研究" onChange={vi.fn()} sourceLabels={{}} onLabels={() => {}} /></MemoryRouter>)
    expect(screen.getByRole('heading', { name: '收益与风险参数', hidden: true })).not.toBeVisible()
    fireEvent.click(screen.getByText('高级设置与口径'))
    expect(screen.getByRole('heading', { name: '收益与风险参数' })).toBeVisible()
  })

  it.each(['role', 'liquidity'] as const)('keeps a missing %s classification visible instead of hiding a required input', missing => {
    const initial = draft('conditional_scenario')
    const value: CmaDraft = { ...initial, assets: initial.assets.map((asset, index) => index === 0 ? { ...asset, [missing]: '' } : asset) }
    render(<MemoryRouter><LtcmaInputFields value={value} options={options} capabilities={ltcmaCapabilities} cutoff={value.as_of} platformDay={value.as_of}
      defaultName="情景研究" onChange={vi.fn()} sourceLabels={{}} onLabels={() => {}} /></MemoryRouter>)
    expect(screen.getByRole('heading', { name: '收益与风险参数' })).toBeVisible()
    expect(screen.getByLabelText(`equity · ${missing === 'role' ? '经济用途' : '流动性'}`)).toBeVisible()
    expect(screen.getByLabelText(`equity · ${missing === 'role' ? '经济用途' : '流动性'}`)).toHaveValue('')
  })

  it('rejects a real-time run attached to a different publication even if its name matches', () => {
    const value = draft('conditional_scenario')
    if (value.model?.method !== 'conditional_scenario') throw new Error('conditional draft')
    value.model = { ...value.model, run_ref: { id: historical.id, content_hash: historical.content_hash }, historical_reference: reference,
      realtime_ref: { id: realtime.id, content_hash: realtime.content_hash } }
    const mismatched = { ...options, scenario_options: { ...options.scenario_options!, realtime_runs: [{ ...realtime, reference: { ...reference, publication_id: 'other-publication' } }] } }
    expect(scenarioSelectionIssue(value, mismatched, key => key)).toBe('scenarioRealtimeMissing')
  })
})

describe('scenario result semantics', () => {
  const conditional: CmaPreview = { ...ltcmaVersion, definition: { ...ltcmaDefinition, model: draft('conditional_scenario').model },
    model_result: { asset_ids: ['equity', 'bond'], method: 'conditional_scenario', definition: draft('conditional_scenario').model!,
      effective_returns: [.08, .03], effective_covariance: [[.01, 0], [0, .005]], posterior_mean_covariance: null,
      content_hash: 'd'.repeat(64), execution: ltcmaVersion.execution,
      model_audit: { limitations: [], model_validation: { status: 'research_only', downstream_eligible: false, reason: '未来预测尚未独立验证。' },
        scenario_probabilities: { state_ids: ['up', 'down'], state_labels: ['上涨', '下跌'], historical: [.5, .5], applied: [.6, .4], current: [.8, .2], endpoint: [.55, .45], average: [.6, .4] },
        horizon_distribution: { horizon_days: 126, expected_returns: [.04, .015], quantiles: [[-.1, -.02], [.03, .012], [.15, .05]], loss_probabilities: [.2, .1] } } } }
  it('separates current, endpoint and average probabilities from cumulative returns', () => {
    render(<LtcmaScenarioResults value={conditional} />)
    const probabilities = screen.getByRole('table', { name: '情景占比与概率' })
    for (const label of ['当前概率', '期末概率', '期间平均占比']) expect(within(probabilities).getByRole('columnheader', { name: label })).toBeVisible()
    expect(within(probabilities).getByText('上涨')).toBeVisible()
    expect(within(probabilities).getByText('80.00%')).toBeVisible()
    expect(screen.getByRole('table', { name: '预测区间内的累计收益' })).toHaveTextContent('4.00%')
    expect(screen.getByText('未来预测尚未独立验证。')).toBeVisible()
  })
  it('blocks all conditional SAA handoffs even if a boolean is incorrectly marked eligible', () => {
    expect(ltcmaSaaIssue(conditional)).toBe('未来预测尚未独立验证。')
    expect(cmaHandoffIssue([{ ...ltcmaItem, method: 'conditional_scenario', downstream_eligible: true }], '2099-01-01')).toBe('scenarioHandoffBlocked')
    expect(ltcmaSaaIssue({ ...conditional, model_result: { ...conditional.model_result!, model_audit: { limitations: [], model_validation: { downstream_eligible: true } } } })).toContain('尚未完成独立验证')
  })
  it('keeps the research boundary visible while grouping data and calculation diagnostics in one closed section', () => {
    const value = { ...conditional, warnings: ['样本边界说明。'], model_result: { ...conditional.model_result!,
      model_audit: { ...conditional.model_result!.model_audit, evidence: { actual_start: '2020-01-02', actual_end: '2025-12-31', observations: 1500,
        sample_window: { observation_years: 6 } } } } }
    const { container } = render(<LtcmaResults value={value} />)
    expect(container.querySelectorAll('details')).toHaveLength(1)
    expect(within(screen.getByRole('table', { name: '收益与风险假设' })).queryByRole('columnheader', { name: /均值不确定半宽/ })).not.toBeInTheDocument()
    expect(screen.getByText('均值不确定半宽（百分点）：未估计／未施加惩罚')).not.toBeVisible()
    expect(screen.getByText('未来预测尚未独立验证。')).toBeVisible()
    expect(screen.getByText('样本边界说明。')).not.toBeVisible()
    expect(screen.getByText('2020-01-02')).not.toBeVisible()
    expect(screen.getByRole('heading', { name: '历史样本覆盖', hidden: true })).not.toBeVisible()
    expect(screen.getByRole('table', { name: '相关矩阵', hidden: true })).not.toBeVisible()
    fireEvent.click(screen.getByText('计算依据与冻结输入'))
    expect(screen.getByText('样本边界说明。')).toBeVisible()
    expect(screen.getByText('2020-01-02')).toBeVisible()
    expect(screen.getByText('均值不确定半宽（百分点）：未估计／未施加惩罚')).toBeVisible()
    expect(screen.getByRole('heading', { name: '历史样本覆盖' })).toBeVisible()
    expect(screen.getByRole('table', { name: '相关矩阵' })).toBeVisible()
  })
  it('shows a computed zero cash uncertainty as zero for the long-term scenario model', () => {
    const definition = { ...ltcmaDefinition, model: draft('long_term_scenario').model,
      assets: ltcmaDefinition.assets.map(asset => asset.id === 'bond' ? { ...asset, role: 'liquidity' as const, mean_uncertainty: 0, annual_volatility: 0 } : asset) }
    const value: CmaPreview = { ...conditional, definition, effective_assumptions: definition,
      source_snapshot: { ...conditional.source_snapshot, assets: conditional.source_snapshot.assets.map(asset => asset.id === 'bond' ? { ...asset, name: '现金' } : asset) },
      model_result: { ...conditional.model_result!, method: 'long_term_scenario', definition: definition.model!,
        model_audit: { limitations: [], uncertainty_status: 'estimated_fixed_definition_block_bootstrap',
          model_validation: { status: 'historical_research', downstream_eligible: true } } } }
    render(<LtcmaResults value={value} />)
    const table = screen.getByRole('table', { name: '收益与风险假设' })
    expect(within(table).getByRole('columnheader', { name: /均值不确定半宽/ })).toBeVisible()
    const row = within(table).getByRole('row', { name: /现金/ })
    expect(within(row).getAllByRole('cell')[2]).toHaveTextContent('0.00%')
    expect(within(table).queryByText('未估计／未施加惩罚')).not.toBeInTheDocument()
  })
})
