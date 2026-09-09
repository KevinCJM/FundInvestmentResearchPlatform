import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation, useNavigate } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import FactorResearchCenter from './FactorResearchCenter'
import { factorApi } from '../services/factorResearch'
import { factorAudit, fixtureCatalog, fixtureRun, fixtureStudy } from '../test/factorFixtures'
import { fixtureDatasetSummaries, fixtureFF3Dataset, fixtureReturnCatalog, fixtureReturnDataset, fixtureReturnPlan, fixtureReturnSource } from '../test/factorReturnFixtures'

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="factor-chart" /> }))
let calls: Array<{ path: string; body: any }>
let planMethod: string
let omitProof: boolean
beforeEach(() => {
  calls = []; planMethod = 'characteristic_spread'; omitProof = false
  vi.stubGlobal('fetch', vi.fn(async (input: string, init?: RequestInit) => {
    const path = new URL(String(input), 'http://localhost').pathname
    const body = init?.body ? JSON.parse(String(init.body)) : undefined
    if (body) calls.push({ path, body })
    let value: any = { items: [] }
    if (path === '/api/factor-research/catalog') value = fixtureCatalog
    else if (path.endsWith('/return-catalog')) value = fixtureReturnCatalog
    else if (path === '/api/factor-research/runs') value = { items: [fixtureRun] }
    else if (path.endsWith('/runs/factor-run-test')) value = fixtureRun
    else if (path.endsWith('/return-sources')) value = body ? fixtureReturnSource : { items: [fixtureReturnSource] }
    else if (path.endsWith('/return-plans') && body) { planMethod = body.method; value = { ...fixtureReturnPlan, ...body } }
    else if (path.endsWith('/return-plans/factor-return-plan-test/runs')) value = { ...(planMethod === 'ff3_2x3' ? fixtureFF3Dataset : fixtureReturnDataset), ...(omitProof ? { execution: undefined } : {}) }
    else if (path === '/api/factor-research/datasets') value = { items: fixtureDatasetSummaries }
    else if (path.endsWith('/return-datasets/factor-dataset-spread-test')) value = fixtureReturnDataset
    else if (path.endsWith('/attributions') && body) value = { id: 'factor-attribution-test', name: body.name, request: body, results: [], warnings: [], execution: factorAudit }
    return { ok: true, status: body ? 201 : 200, json: async () => value }
  }))
})
afterEach(() => { vi.unstubAllGlobals() })
function NavigationProbe() { const location = useLocation(); const navigate = useNavigate(); return <><output data-testid="route">{location.search}</output><button onClick={() => navigate(-1)}>测试返回</button></> }
function open(search = '') { return render(<MemoryRouter initialEntries={['/settings/factor-research' + search]}><FactorResearchCenter /><NavigationProbe /></MemoryRouter>) }
async function click(name: string, role: 'button' | 'tab' = 'button') { const button = await screen.findByRole(role, { name }); await waitFor(() => expect(button).toBeEnabled()); fireEvent.click(button) }

describe('factor return module', () => {
  it('separates two modules, preserves legacy run links and supports browser back', async () => {
    open('?run=factor-run-test')
    await screen.findByLabelText('因子检验结果')
    expect(screen.getByRole('tab', { name: '产品特征因子' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.queryByRole('tab', { name: '收益归因' })).not.toBeInTheDocument()
    await click('因子收益率', 'tab')
    await screen.findByLabelText('收益率构造算法')
    expect(screen.getByRole('tab', { name: '收益归因' })).toBeInTheDocument()
    await click('测试返回')
    await screen.findByLabelText('因子检验结果')
    expect(screen.getByTestId('route')).toHaveTextContent('?run=factor-run-test')
  })

  it('builds from a frozen run and carries the generated dataset into generic attribution', async () => {
    open('?module=returns&tab=construct&source=factor-run-test')
    const submit = await screen.findByRole('button', { name: '保存并生成收益率' })
    await waitFor(() => expect(submit).toBeEnabled())
    expect(screen.getByLabelText('来源特征运行')).toHaveValue(fixtureRun.id)
    fireEvent.click(submit)
    await screen.findByLabelText('因子收益率结果')
    expect(calls[0].body.source_run_id).toBe(fixtureRun.id)
    expect(calls[1].body).toEqual({ revision: 1 })
    expect(screen.getByRole('button', { name: '导出 CSV' })).toBeEnabled()
    await click('用于收益归因')
    await waitFor(() => expect(screen.getByLabelText('归因模型')).toHaveValue('factor_regression'))
    expect(screen.getByLabelText('回归因子收益数据集')).toHaveValue(fixtureReturnDataset.id)
    fireEvent.change(screen.getByLabelText('归因产品代码'), { target: { value: '000001.OF' } })
    await click('运行归因研究')
    await screen.findByLabelText('收益归因结果')
    expect(calls[calls.length - 1]?.body).toMatchObject({ model: 'factor_regression', dataset_id: fixtureReturnDataset.id, targets: ['000001.OF'] })
  })

  it('constructs FF3 only from an explicit source panel', async () => {
    open('?module=returns')
    const algorithm = await screen.findByLabelText('收益率构造算法')
    await waitFor(() => expect(algorithm).toBeEnabled())
    fireEvent.change(algorithm, { target: { value: 'ff3_2x3' } })
    expect(screen.getByText(/不自动取本地股票构建/)).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('FF3 原始面板'), { target: { value: fixtureReturnSource.id } })
    await click('保存并生成收益率')
    await screen.findByLabelText('因子收益率结果')
    expect(calls[0].body).toMatchObject({ method: 'ff3_2x3', source_panel_id: fixtureReturnSource.id, source_run_id: null, cost_bps: 0 })
    expect(screen.getByRole('heading', { name: 'FF3 分组证据' })).toBeInTheDocument()
  })

  it('reports bad upload JSON and rejects incompatible FF3 factor columns', async () => {
    open('?module=returns&tab=datasets')
    const input = await screen.findByLabelText('导入因子收益 JSON')
    await waitFor(() => expect(input).toBeEnabled())
    const file = new File(['{'], 'broken.json', { type: 'application/json' })
    Object.defineProperty(file, 'text', { value: async () => '{' })
    fireEvent.change(input, { target: { files: [file] } })
    expect(await screen.findByRole('alert')).toHaveTextContent('不是有效的 JSON')
    expect(calls).toEqual([])
    await click('收益归因', 'tab')
    const model = await screen.findByLabelText('归因模型')
    await waitFor(() => expect(model).toBeEnabled())
    fireEvent.change(model, { target: { value: 'ff3' } })
    expect(screen.getByRole('option', { name: /测试特征收益/ })).toBeDisabled()
    expect(screen.getByRole('option', { name: /测试 FF3 数据/ })).toBeEnabled()
  })

  it('loads persisted datasets and rejects construction without numerical proof', async () => {
    open('?module=returns&tab=datasets&dataset=factor-dataset-spread-test')
    await screen.findByLabelText('因子收益序列')
    expect(screen.getByLabelText('因子收益序列')).toHaveTextContent('—')
    omitProof = true
    await expect(factorApi.runReturnPlan(fixtureReturnPlan)).rejects.toThrow('执行证明')
  })

  it('refuses to download API errors or non-CSV data as a CSV file', async () => {
    vi.mocked(fetch).mockResolvedValueOnce({ ok: false, status: 422, json: async () => ({ detail: { message: '数据集无效' } }) } as Response)
    await expect(factorApi.exportReturnCsv('wrong')).rejects.toThrow('数据集无效')
    vi.mocked(fetch).mockResolvedValueOnce({ ok: true, headers: new Headers({ 'content-type': 'application/json' }) } as Response)
    await expect(factorApi.exportReturnCsv('wrong')).rejects.toThrow('不是 CSV')
  })

  it('includes rolling settings with backwards-compatible monthly defaults', async () => {
    const { studyDraft } = await import('../services/factorResearch')
    expect(studyDraft(fixtureStudy)).toMatchObject({ signal_frequency: 'monthly', ic_window: 12, ic_min_periods: 6 })
    open()
    expect(await screen.findByLabelText('信号与调仓频率')).toHaveValue('monthly')
    expect(screen.getByLabelText('滚动 IC 窗口（期）')).toHaveValue(12)
  })
})
