import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import FactorResearchCenter from './FactorResearchCenter'
import { factorApi } from '../services/factorResearch'
import { fixtureCatalog, fixtureFactors, fixtureRun, fixtureStudy } from '../test/factorFixtures'

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="factor-chart" /> }))
let bodies: Array<{ path: string; body: any }>
let failRun: boolean
let corruptAudit: boolean

beforeEach(() => {
  bodies = []; failRun = false; corruptAudit = false
  vi.stubGlobal('fetch', vi.fn(async (input: string, init?: RequestInit) => {
    const path = new URL(String(input), 'http://localhost').pathname
    const body = init?.body ? JSON.parse(String(init.body)) : undefined
    if (body) bodies.push({ path, body })
    let value: unknown = { items: [] }
    let status = 200
    if (path.endsWith('/catalog')) value = fixtureCatalog
    else if (path.endsWith('/factors') && body) value = { ...fixtureFactors[0], ...body, id: 'factor-custom-test' }
    else if (path.endsWith('/studies') && body) value = { ...fixtureStudy, ...body }
    else if (path === '/api/factor-research/studies/factor-study-test/runs') {
      if (failRun) { status = 422; value = { detail: { message: '有效截面不足，请补齐数据。' } } }
      else { value = { ...fixtureRun, ...(corruptAudit ? { execution: undefined } : {}) } }
    } else if (path === '/api/factor-research/runs/factor-run-test') value = fixtureRun
    else if (path.endsWith('/runs')) value = { items: [] }
    return { ok: status < 400, status, json: async () => value }
  }))
})
afterEach(() => { vi.unstubAllGlobals() })

const open = () => render(<MemoryRouter><FactorResearchCenter /></MemoryRouter>)

describe('factor research workflow', () => {
  it('keeps benchmark, model and dataset distinct and saves locked factor versions before running', async () => {
    open()
    const user = userEvent.setup()
    await screen.findByLabelText('研究方案名称')
    expect(screen.getByLabelText('比较基准类型')).toHaveValue('etf')
    expect(screen.getByLabelText('因子模型')).toHaveValue('characteristic_composite')
    expect(screen.getByLabelText('因子输入数据集')).toHaveValue('active_adjusted_nav')
    await user.click(screen.getByRole('button', { name: '保存并运行检验' }))
    await screen.findByLabelText('因子检验结果')
    expect(bodies[0].body.factors[0]).toEqual({ factor_id: fixtureFactors[0].id, revision: 1, weight: .5 })
    expect(bodies[1].body).toEqual({ revision: 1 })
    expect(screen.getByLabelText('最新因子得分')).toHaveTextContent('测试ETF0')
    expect(screen.getByRole('button', { name: '样本外' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('shows a recoverable real-data error without manufacturing results', async () => {
    failRun = true
    open()
    await screen.findByLabelText('研究方案名称')
    fireEvent.click(screen.getByRole('button', { name: '保存并运行检验' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('有效截面不足')
    expect(screen.queryByLabelText('因子检验结果')).not.toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('button', { name: '保存并运行检验' })).toBeEnabled())
  })

  it('copies a template into an editable definition without altering the builtin', async () => {
    open()
    const user = userEvent.setup()
    await screen.findByLabelText('研究方案名称')
    const library = screen.getByRole('tab', { name: '因子库' })
    // The editor can render before the catalogue-loading action releases its
    // busy state. Wait for an actionable tab rather than clicking a disabled one.
    await waitFor(() => expect(library).toBeEnabled())
    await user.click(library)
    await user.click((await screen.findAllByRole('button', { name: '复制构建' }))[0])
    const name = screen.getByLabelText('因子名称')
    await user.clear(name); await user.type(name, '我的ETF动量')
    await user.click(screen.getByRole('button', { name: '保存因子' }))
    await waitFor(() => expect(bodies.some(item => item.path.endsWith('/factors'))).toBe(true))
    const saved = bodies.find(item => item.path.endsWith('/factors'))!.body
    expect(saved.name).toBe('我的ETF动量')
    expect(saved.id).toBeUndefined()
    expect(saved.operator).toBe('momentum')
  })

  it('rejects numerical responses that omit the execution proof', async () => {
    corruptAudit = true
    await expect(factorApi.run(fixtureStudy)).rejects.toThrow('执行证明')
  })
})
