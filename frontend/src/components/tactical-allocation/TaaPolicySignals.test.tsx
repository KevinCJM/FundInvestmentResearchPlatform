import { useState } from 'react'
import { act, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { TaaPolicySignals, policySignalIssue, scheduledPolicy } from './TaaPolicySignals'
import { taaPreview } from '../../test/tacticalAllocationFixtures'
import type { TaaPreviewRequest } from '../../services/tacticalAllocation'

function Harness({ composite = false }: { composite?: boolean }) {
  const [request, setRequest] = useState<TaaPreviewRequest>({ ...taaPreview.request, decision_policy: { ...scheduledPolicy }, signal_mode: composite ? 'composite' : 'momentum' })
  return <><TaaPolicySignals request={request} assets={['equity', 'bond']} update={patch => setRequest(old => ({ ...old, ...patch }))} /><output data-testid="request">{JSON.stringify(request)}</output><p role="status">{policySignalIssue(request, ['equity', 'bond'])}</p></>
}

describe('TAA explicit clock and signal controls', () => {
  it('freezes clocks independently and permits explicit original daily target mode', async () => {
    const raw = userEvent.setup(); const user = new Proxy(raw, { get(target, key: keyof typeof raw) { const method = target[key]; return typeof method === 'function' ? (...args: unknown[]) => act(async () => { await (method as (...values: unknown[]) => Promise<unknown>)(...args) }) : method } }); render(<Harness />)
    await user.selectOptions(screen.getByLabelText('决策频率'), 'quarterly')
    await user.selectOptions(screen.getByLabelText('执行机会'), 'weekly')
    await user.click(screen.getByText('执行滞后、阈值与实际持仓时点'))
    fireEvent.change(screen.getByLabelText('单资产偏离阈值（百分点）'), { target: { value: '3' } })
    const request = JSON.parse(screen.getByTestId('request').textContent!)
    expect(request.decision_policy).toMatchObject({ decision_frequency: 'quarterly', execution_frequency: 'weekly', deviation_threshold: .03 })
    await user.selectOptions(screen.getByLabelText('调仓口径'), 'daily_target')
    expect(JSON.parse(screen.getByTestId('request').textContent!).decision_policy).toBeNull()
    expect(screen.queryByLabelText('决策频率')).not.toBeInTheDocument()
  })
  it('invalid external JSON clears executable inputs and complete dated zero stays available', async () => {
    const raw = userEvent.setup(); const user = new Proxy(raw, { get(target, key: keyof typeof raw) { const method = target[key]; return typeof method === 'function' ? (...args: unknown[]) => act(async () => { await (method as (...values: unknown[]) => Promise<unknown>)(...args) }) : method } }); render(<Harness composite />)
    await user.click(screen.getByRole('button', { name: '添加信号分量' }))
    await user.selectOptions(screen.getByLabelText('信号 1 来源类型'), 'value')
    await user.type(screen.getByLabelText('信号 1 研究来源'), '固定研究夹具')
    await user.type(screen.getByLabelText('信号 1 标准化方法与依据'), '外部研究标准化结果；中性值为零')
    expect(screen.getByText('外部信号需要日期化的研究标准值，不能用净值替代。')).toBeInTheDocument()
    const input = screen.getByLabelText('日期化标准值（JSON 数组）')
    fireEvent.change(input, { target: { value: '[broken' } })
    expect(screen.getByRole('alert')).toHaveTextContent('JSON 尚不完整')
    expect(input).toHaveValue('[broken')
    expect(JSON.parse(screen.getByTestId('request').textContent!).signal_components[0].observations).toEqual([])
    const observations = [{ observed_on: '2025-01-01', available_on: '2025-01-02', expires_on: '2025-01-31', values: { equity: 0, bond: 0 } }]
    fireEvent.change(input, { target: { value: JSON.stringify(observations) } })
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    const request = JSON.parse(screen.getByTestId('request').textContent!)
    expect(request.signal_components[0].observations).toEqual(observations)
    expect(policySignalIssue(request, ['equity', 'bond'])).toBe('')
    await user.selectOptions(screen.getByLabelText('信号 1 来源类型'), 'carry')
    expect(screen.getByLabelText('日期化标准值（JSON 数组）')).toHaveValue('')
    expect(JSON.parse(screen.getByTestId('request').textContent!).signal_components[0]).toMatchObject({ kind: 'carry', observations: [], source: '', methodology: '' })
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    // An incomplete draft stays editable until a deliberate kind change resets it.
    fireEvent.change(screen.getByLabelText('日期化标准值（JSON 数组）'), { target: { value: '[{"observed_on":' } })
    expect(screen.getByLabelText('日期化标准值（JSON 数组）')).toHaveValue('[{"observed_on":')
    await user.selectOptions(screen.getByLabelText('信号 1 来源类型'), 'macro')
    expect(screen.getByLabelText('日期化标准值（JSON 数组）')).toHaveValue('')
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '添加信号分量' }))
    expect(JSON.parse(screen.getByTestId('request').textContent!).signal_components.map((c: { weight: number }) => c.weight)).toEqual([1, 0])
  })
})
