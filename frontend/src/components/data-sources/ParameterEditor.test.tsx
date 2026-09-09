import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import ParameterEditor from './ParameterEditor'

function Harness({ initial = {}, saved = () => undefined, draft = () => undefined }: {
  initial?: Record<string, unknown>; saved?: (value: Record<string, unknown>) => void; draft?: () => void
}) {
  const [value, setValue] = useState(initial)
  return <form aria-label="参数表单" onSubmit={event => event.preventDefault()}><ParameterEditor label="参数" value={value} onDraftChange={draft} onChange={next => { setValue(next); saved(next) }} /><output data-testid="value">{JSON.stringify(value)}</output></form>
}

describe('ParameterEditor', () => {
  it('保留文本前导零、数字、布尔、空值和嵌套对象', () => {
    const initial = { code: '000001', count: 5, enabled: false, empty: null, nested: { keys: [1, 2] } }
    const saved = vi.fn()
    render(<Harness initial={initial} saved={saved} />)
    fireEvent.change(screen.getByLabelText('参数 2 值'), { target: { value: '12.5' } })
    expect(saved).toHaveBeenLastCalledWith({ ...initial, count: 12.5 })
    fireEvent.change(screen.getByLabelText('参数 3 值'), { target: { value: 'true' } })
    expect(saved).toHaveBeenLastCalledWith({ ...initial, count: 12.5, enabled: true })
    expect(screen.getByLabelText('参数 1 值')).toHaveValue('000001')
  })

  it('新增空行标记草稿但不会输出旧数据；空名称和重复名称禁止提交', () => {
    const saved = vi.fn()
    const draft = vi.fn()
    render(<Harness initial={{ code: '001' }} saved={saved} draft={draft} />)
    fireEvent.click(screen.getByRole('button', { name: '添加参数' }))
    expect(draft).toHaveBeenCalled()
    expect(saved).not.toHaveBeenCalled()
    const name = screen.getByLabelText('参数 2 名称')
    expect(name).toBeInvalid()
    fireEvent.change(name, { target: { value: 'code' } })
    expect(name).toBeInvalid()
    expect(saved).not.toHaveBeenCalled()
    fireEvent.change(name, { target: { value: 'market' } })
    fireEvent.change(screen.getByLabelText('参数 2 值'), { target: { value: 'SSE' } })
    expect(saved).toHaveBeenLastCalledWith({ code: '001', market: 'SSE' })
    expect(name).toBeValid()
  })

  it('空数字和非法嵌套 JSON 保持输入，不静默转换为零或旧值', () => {
    const saved = vi.fn()
    render(<Harness initial={{ count: 5, nested: {} }} saved={saved} />)
    fireEvent.change(screen.getByLabelText('参数 1 值'), { target: { value: '' } })
    expect(screen.getByLabelText('参数 1 值')).toBeInvalid()
    expect(saved).not.toHaveBeenCalled()
    const nested = screen.getByLabelText('参数 2 值')
    fireEvent.change(nested, { target: { value: '{unfinished' } })
    expect(nested).toBeInvalid()
    expect(nested).toHaveValue('{unfinished')
    expect(saved).not.toHaveBeenCalled()
  })

  it('删除行后正确更新对象，并能同步外部 JSON 修改', () => {
    const saved = vi.fn()
    const { rerender } = render(<ParameterEditor label="参数" value={{ code: '001', count: 3 }} onChange={saved} />)
    fireEvent.click(screen.getByRole('button', { name: '删除参数 2' }))
    expect(saved).toHaveBeenLastCalledWith({ code: '001' })
    rerender(<ParameterEditor label="参数" value={{ source: 'vendor', enabled: true }} onChange={saved} />)
    expect(screen.getByLabelText('参数 1 名称')).toHaveValue('source')
    expect(screen.getByLabelText('参数 2 类型')).toHaveValue('boolean')
  })
})
