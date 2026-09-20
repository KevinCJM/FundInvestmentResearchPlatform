import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { expect, it } from 'vitest'
import LtcmaStatisticsFields from './LtcmaStatisticsFields'
import { cmaDraftFromDefinition, cmaRequest, completeCma } from '../../services/strategicAllocation'
import { ltcmaDefinition, ltcmaItem, ltcmaOptions } from '../../test/ltcmaFixtures'

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
