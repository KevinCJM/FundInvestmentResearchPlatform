import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { expect, it } from 'vitest'
import LtcmaStatisticsFields from './LtcmaStatisticsFields'
import { cmaDraftFromDefinition, cmaRequest, completeCma } from '../../services/strategicAllocation'
import { ltcmaDefinition, ltcmaItem, ltcmaOptions } from '../../test/ltcmaFixtures'
import { scopeFacts } from '../../services/cmaCompatibility'
import type { UniverseVersion } from '../../services/strategicScope'

it('切换后验续更清除隐藏的先验强度，切回重设时要求重新填写', () => {
  const initial = cmaDraftFromDefinition(ltcmaDefinition)
  initial.model = { method: 'bayesian_niw', asset_ids: initial.assets.map(a => a.id),
    as_of: initial.as_of, currency: initial.currency, source: '冻结先验证据',
    prior_ref: { id: ltcmaItem.id, content_hash: ltcmaItem.content_hash }, prior_mode: 'recenter',
    mean_prior_observations: 20, covariance_prior_observations: 30, data_reuse_acknowledged: true }
  let latest = initial
  function Editor() {
    const [value, setValue] = useState(initial)
    return <LtcmaStatisticsFields value={value} options={{ ...ltcmaOptions,
      assumptions: [{ ...ltcmaItem, method: 'bayesian_niw' }] }}
      onChange={next => { latest = next; setValue(next) }} sourceLabels={{}} onLabels={() => {}} />
  }
  render(<MemoryRouter><Editor /></MemoryRouter>)
  const mode = screen.getAllByRole('combobox').find(input => (input as HTMLSelectElement).value === 'recenter')!
  fireEvent.change(mode, { target: { value: 'continue' } })
  if (!completeCma(latest)) throw new Error('Continuation must be ready to submit')
  const submitted = cmaRequest(latest)
  expect(submitted.model?.method).toBe('bayesian_niw')
  if (submitted.model?.method !== 'bayesian_niw') throw new Error('Expected NIW')
  expect(submitted.model.mean_prior_observations ?? null).toBeNull()
  expect(submitted.model.covariance_prior_observations ?? null).toBeNull()
  expect(submitted.model.data_reuse_acknowledged).toBe(false)
  expect(screen.queryByLabelText('均值先验等效日观察数')).not.toBeInTheDocument()
  fireEvent.change(mode, { target: { value: 'recenter' } })
  expect(completeCma(latest)).toBe(false)
  expect(screen.getByLabelText('均值先验等效日观察数')).toHaveValue('')
  expect(screen.getByLabelText('风险先验等效日观察数')).toHaveValue('')
})

it('相同配置的旧记录可选择，并区分历史窗口和真实不适用原因', () => {
  const universe: UniverseVersion = { id: 'new-scope', name: '范围', created_at: '', content_hash: 'a'.repeat(64), preview_hash: 'b'.repeat(64),
    implementation_status: 'unmapped', implementation_gaps: [], research_only: true,
    definition: { name: '范围', as_of: ltcmaDefinition.as_of, currency: 'CNY', source: '',
      assets: ltcmaDefinition.assets.map(a => ({ ...a, currency: 'CNY', name: a.id, source: '' })) } }
  const initial = cmaDraftFromDefinition({ ...ltcmaDefinition, alloc_name: null, strategic_universe_id: universe.id })
  initial.model = { method: 'bayesian_niw', asset_ids: initial.assets.map(a => a.id), as_of: initial.as_of,
    currency: initial.currency, source: '', prior_ref: { id: '', content_hash: '' } }
  const items = ['2Y', '5Y'].map((kind, i) => ({ ...ltcmaItem, id: `prior-${i}`, alloc_name: null, strategic_universe_id: 'old-scope',
    method: 'historical_statistics', scope_facts: scopeFacts(universe.definition),
    history: { window: { kind: kind as '2Y' | '5Y' }, start_date: null, end_date: null, observations: null, source_names: [] } }))
  function Editor() {
    const [value, onChange] = useState(initial)
    return <LtcmaStatisticsFields value={value} onChange={onChange} sourceLabels={{}} onLabels={() => {}}
      options={{ ...ltcmaOptions, strategic_universes: [universe], assumptions: [...items, { ...items[0], id: 'future', as_of: '2099-01-01' }] }} />
  }
  render(<MemoryRouter><Editor /></MemoryRouter>)
  expect(screen.getByRole('option', { name: /历史统计 · 近 2 年.*2026/ })).toBeEnabled()
  expect(screen.getByRole('option', { name: /历史统计 · 近 5 年/ })).toBeEnabled()
  expect(screen.getByRole('option', { name: /先验研究日晚于当前研究日/ })).toBeDisabled()
  fireEvent.change(screen.getByLabelText('先验 LTCMA 版本'), { target: { value: items[0].id } })
  expect(screen.getByLabelText('先验 LTCMA 版本')).toHaveValue(items[0].id)
  expect(screen.queryByText(/尚无已确认的 LTCMA/)).not.toBeInTheDocument()
})

it('历史情景允许月频区间，并解释不能使用的未来版本', () => {
  const initial = cmaDraftFromDefinition(ltcmaDefinition)
  initial.model = { method: 'historical_regime_occupancy', asset_ids: initial.assets.map(a => a.id),
    as_of: initial.as_of, currency: initial.currency, source: '', run_ref: { id: '', content_hash: '' } }
  const monthly = { id: 'monthly', name: '月频历史区间', content_hash: 'a'.repeat(64), frequency: 'monthly',
    as_of: initial.as_of, states: [{ id: 'growth', label: '增长' }], available: true, reasons: [] }
  let latest = initial
  function Editor() {
    const [value, setValue] = useState(initial)
    return <LtcmaStatisticsFields value={value} options={{ ...ltcmaOptions, regime_runs: [monthly,
      { ...monthly, id: 'future', name: '未来情景', available: false,
        reasons: [{ code: 'after_research_day', message: '该情景使用了研究日之后的数据。' }] }] }}
      onChange={next => { latest = next; setValue(next) }} sourceLabels={{}} onLabels={() => {}} />
  }
  render(<MemoryRouter><Editor /></MemoryRouter>)
  fireEvent.click(screen.getByLabelText('已保存的事后状态研究'))
  const future = screen.getByRole('menuitemradio', { name: /未来情景/ })
  fireEvent.mouseEnter(future)
  expect(screen.getByRole('tooltip')).toHaveTextContent('该情景使用了研究日之后的数据。')
  fireEvent.click(future)
  expect(latest.model).toMatchObject({ run_ref: { id: '' } })
  fireEvent.click(screen.getByRole('menuitemradio', { name: /月频历史区间/ }))
  expect(latest.model).toMatchObject({ run_ref: { id: monthly.id, content_hash: monthly.content_hash } })
  expect(screen.getByLabelText('已保存的事后状态研究')).toHaveValue(monthly.id)
})

it('NIW 已选先验需更新时解释原因并禁用选项', () => {
  const initial = cmaDraftFromDefinition(ltcmaDefinition)
  initial.model = { method: 'bayesian_niw', asset_ids: initial.assets.map(a => a.id), as_of: initial.as_of,
    currency: initial.currency, source: '', prior_ref: { id: ltcmaItem.id, content_hash: ltcmaItem.content_hash } }
  render(<MemoryRouter><LtcmaStatisticsFields value={initial} onChange={() => {}} sourceLabels={{}} onLabels={() => {}}
    options={{ ...ltcmaOptions, assumptions: [{ ...ltcmaItem, usable: { status: 'stale', reasons: [] } }] }} /></MemoryRouter>)
  expect(screen.getByRole('option', { name: /先验或其上游已有新版本/ })).toBeDisabled()
  expect(screen.getByRole('status')).toHaveTextContent('先验或其上游已有新版本')
  expect(screen.getByLabelText('先验 LTCMA 版本')).toHaveValue(ltcmaItem.id)
})
