import { useState } from 'react'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import LtcmaNiwStrengths from './LtcmaNiwStrengths'
import { ltcma } from '../../services/ltcma'
import { cmaDraftFromDefinition, completeCma } from '../../services/strategicAllocation'
import { ltcmaDefinition, ltcmaItem } from '../../test/ltcmaFixtures'
import type { CmaSampleSummary } from '../../services/ltcmaContract.generated'

const sample: CmaSampleSummary = { requested_start: '2020-01-01', requested_end: '2024-01-01',
  actual_start: '2020-01-02', actual_end: '2023-12-29', excluded_return_periods: 0, observations: 1000,
  observation_frequency: 'daily', periods_per_year: 252, source_hash: 'a'.repeat(64), return_panel_hash: 'b'.repeat(64) }
function initial() {
  const value = cmaDraftFromDefinition(ltcmaDefinition)
  value.model = { method: 'bayesian_niw', asset_ids: value.assets.map(a => a.id), as_of: value.as_of,
    currency: value.currency, source: '', prior_ref: { id: ltcmaItem.id, content_hash: ltcmaItem.content_hash },
    prior_mode: 'recenter', mean_prior_observations: null, covariance_prior_observations: null }
  return value
}
afterEach(() => { vi.restoreAllMocks() })

it('无需先验强度即可查看真实样本，输入后显示均值权重和风险倍数，并拦截上限', async () => {
  const load = vi.spyOn(ltcma, 'sample').mockResolvedValue(sample)
  let latest = initial()
  function Editor() {
    const [value, setValue] = useState(latest)
    return <LtcmaNiwStrengths value={value} onChange={next => { latest = next; setValue(next) }} />
  }
  render(<Editor />)
  fireEvent.click(screen.getByRole('button', { name: '查看样本参考' }))
  expect(await screen.findByText('本次样本：1,000 个日收益观察')).toBeVisible()
  expect(screen.getByText(/1,001 个共同净值日/)).toBeVisible()
  expect(load.mock.calls[0][0].model).not.toHaveProperty('mean_prior_observations')
  expect(load.mock.calls[0][0].model).not.toHaveProperty('prior_ref')
  const mean = screen.getByLabelText('均值先验等效日观察数'), risk = screen.getByLabelText('风险先验等效日观察数')
  expect(mean).toHaveValue('')
  fireEvent.change(mean, { target: { value: '1000' } })
  fireEvent.change(risk, { target: { value: '250' } })
  expect(screen.getByText(/旧判断约占 50%，本次样本约占 50%/)).toBeVisible()
  expect(screen.getByText(/相当于本次样本的 0.25 倍。这不是波动率/)).toBeVisible()
  expect(load).toHaveBeenCalledTimes(1)
  expect(completeCma(latest)).toBe(true)
  fireEvent.change(mean, { target: { value: '900000' } })
  expect(screen.getByRole('alert')).toHaveTextContent('不超过 100,000')
  expect(completeCma(latest)).toBe(false)
  fireEvent.change(mean, { target: { value: '1' } })
  expect(screen.getByText(/相当于本次样本的 0.001 倍；更新均值时，旧判断约占 0.1%/)).toBeVisible()
  expect(completeCma(latest)).toBe(true)
  fireEvent.change(risk, { target: { value: '100001' } })
  expect(completeCma(latest)).toBe(false)
  fireEvent.change(risk, { target: { value: '100000' } })
  expect(completeCma(latest)).toBe(true)
})

it('窗口变化立即清除参考，旧请求即使晚到也不能覆盖当前样本', async () => {
  let resolveOld!: (value: CmaSampleSummary) => void
  const load = vi.spyOn(ltcma, 'sample').mockResolvedValueOnce(sample)
    .mockImplementationOnce(() => new Promise(resolve => { resolveOld = resolve }))
    .mockResolvedValueOnce({ ...sample, observations: 500 })
  const value = initial(), onChange = vi.fn()
  const view = render(<LtcmaNiwStrengths value={value} onChange={onChange} />)
  fireEvent.click(screen.getByRole('button', { name: '查看样本参考' }))
  await screen.findByText('本次样本：1,000 个日收益观察')
  fireEvent.click(screen.getByRole('button', { name: '刷新样本参考' }))
  if (value.model?.method !== 'bayesian_niw') throw new Error('NIW required')
  view.rerender(<LtcmaNiwStrengths value={{ ...value, model: { ...value.model, window: { kind: '2Y' } } }} onChange={onChange} />)
  expect(screen.queryByText(/本次样本：/)).not.toBeInTheDocument()
  expect(load.mock.calls[1][1]?.aborted).toBe(true)
  fireEvent.click(screen.getByRole('button', { name: '查看样本参考' }))
  await screen.findByText('本次样本：500 个日收益观察')
  await act(async () => resolveOld(sample))
  expect(screen.getByText('本次样本：500 个日收益观察')).toBeVisible()
  expect(screen.queryByText('本次样本：1,000 个日收益观察')).not.toBeInTheDocument()
})

it('样本读取失败可重试，卸载会取消读取', async () => {
  const load = vi.spyOn(ltcma, 'sample').mockRejectedValueOnce(new Error('缺少共同交易日'))
    .mockImplementationOnce(() => new Promise(() => {}))
  const view = render(<LtcmaNiwStrengths value={initial()} onChange={() => {}} />)
  fireEvent.click(screen.getByRole('button', { name: '查看样本参考' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('缺少共同交易日')
  expect(screen.queryByText(/本次样本：/)).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '查看样本参考' }))
  await waitFor(() => expect(load).toHaveBeenCalledTimes(2))
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  view.unmount()
  expect(load.mock.calls[1][1]?.aborted).toBe(true)
})
