import { useState } from 'react'
import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { Field, NumberInput } from './ResearchUI'

function Editor({ initial = 0 }: { initial?: number }) {
  const [value, setValue] = useState(initial)
  return <><NumberInput aria-label="压力冲击" value={value} onValueChange={setValue} /><output>{Number.isFinite(value) ? value : '未填写完整'}</output><button onClick={() => setValue(5)}>外部重置</button></>
}
describe('压力数值输入', () => {
  it.each(['-10', '-0.25', '.5', '+12', '-1e-3'])('逐字输入 %s 保留符号和小数', async text => {
    const user = userEvent.setup()
    render(<Editor />)
    const input = screen.getByRole('spinbutton', { name: '压力冲击' })
    await act(async () => { await user.clear(input); await user.type(input, text) })
    expect(input).toHaveValue(text)
    expect(screen.getByRole('status')).toHaveTextContent(String(Number(text)))
    expect(input).toHaveAttribute('aria-invalid', 'false')
  })
  it('空值和未完成负号不转成零，外部切换仍可重置', async () => {
    const user = userEvent.setup()
    render(<Editor initial={-10} />)
    const input = screen.getByRole('spinbutton', { name: '压力冲击' })
    await act(async () => { await user.clear(input); await user.type(input, '-') })
    expect(screen.getByRole('status')).toHaveTextContent('未填写完整')
    expect(input).toHaveAttribute('aria-invalid', 'true')
    await act(async () => { await user.click(screen.getByRole('button', { name: '外部重置' })) })
    expect(input).toHaveValue('5')
  })
})

it('字段标签明确区分必填与非必填', () => {
  render(<><Field label="必填字段" required><input required /></Field><Field label="可选字段" optional><input /></Field></>)
  expect(screen.getByLabelText('必填字段')).toBeRequired()
  expect(screen.getByLabelText('必填字段').closest('label')).toHaveAttribute('data-required', 'true')
  expect(screen.getByLabelText('可选字段').closest('label')).toHaveAttribute('data-optional', 'true')
  expect(screen.getByText('可选字段')).toHaveAttribute('data-optional-label', '（非必填）')
})
