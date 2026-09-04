import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ResearchDataLab from './ResearchDataLab'

vi.mock('echarts-for-react', () => ({ default: ({ 'aria-label': label }: { 'aria-label'?: string }) => <div data-testid="research-chart">{label}</div> }))

const fixedExecution = {
  execution_backend: 'numba_njit_fixed_signature',
  nopython: true,
  object_mode: 0,
  python_fallback: 0,
  request_time_compilation: 0,
  kernel_signatures: { research_profile: ['float64[:]->profile'] },
}

const series = {
  id: 'index:index_daily:000300.SH',
  kind: 'index',
  name: '沪深300指数',
  code: '000300.SH',
  status: 'available',
  source_api: 'index_daily',
  dataset: 'index_daily.parquet',
  default_field: 'close',
  fields: ['close', 'pct_chg'],
  frequency: 'daily',
  coverage: { first_date: '2020-01-02', last_date: '2024-12-31', observations: 1200 },
  missing: { count: 0, ratio: 0 },
  pit: { supported: true, policy: 'point_in_time' },
  vintage: { supported: false },
  profile_operations: ['raw', 'normalized', 'return'],
  regime_node_type: 'source.series',
  binding_parameters: { series_id: 'index:index_daily:000300.SH', field: 'close' },
}

const ok = (body: unknown, status = 200) => ({ ok: true, status, json: async () => body } as Response)
const settle = (milliseconds: number) => act(async () => { await new Promise((resolve) => window.setTimeout(resolve, milliseconds)) })

describe('ResearchDataLab', () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

  it('浏览真实目录、计算后端数据画像并绑定到计算图', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input)
      if (path.includes('/catalog?')) return ok({ schema_version: '1.0', snapshot: { generation: 'GEN-1', status: 'active' }, items: [series], total: 1, offset: 0, limit: 200, capabilities: {}, execution: fixedExecution })
      if (path.endsWith('/profile') && init?.method === 'POST') return ok({ series, coverage: series.coverage, missing: series.missing, sampling: { source_observations: 1200, returned_observations: 3, sampled: true, method: 'uniform' }, dates: ['2024-01-02', '2024-01-03', '2024-01-04'], values: { raw: [3500, 3510, 3490], return: [null, 0.0028, -0.0057] }, distribution: { quantiles: { p50: 3500 } }, pit: series.pit, vintage: series.vintage, execution: fixedExecution })
      throw new Error(`Unexpected request: ${path}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    const onBind = vi.fn()
    const user = userEvent.setup()
    render(<ResearchDataLab onBindSeries={onBind} />)
    await settle(220)

    expect(await screen.findByRole('heading', { name: '研究数据实验室' })).toBeInTheDocument()
    expect((await screen.findAllByText('沪深300指数')).length).toBeGreaterThan(0)
    expect(screen.getByText('GEN-1')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '加入计算图' }))
    expect(onBind).toHaveBeenCalledWith(expect.objectContaining({ id: series.id, regime_node_type: 'source.series' }))

    await user.click(screen.getByRole('button', { name: '计算数据画像' }))
    await settle(0)
    expect(await screen.findByTestId('research-chart')).toHaveTextContent('沪深300指数时序图')
    expect(screen.getByRole('table', { name: '研究序列抽样数据' })).toHaveTextContent('3,510')
    expect(screen.getByTestId('research-series-mobile-list')).toHaveTextContent('2024-01-04')
    expect(screen.getByText(/固定签名 NJIT 执行证明已通过/)).toBeInTheDocument()

    const profileCall = fetchMock.mock.calls.find(([path]) => String(path).endsWith('/profile'))
    const body = JSON.parse(String(profileCall?.[1]?.body))
    expect(body).toMatchObject({ series_id: series.id, field: 'close', rolling_window: 20, sample_limit: 500 })
  })

  it('类型与下载状态筛选使用后端目录参数，不为未下载数据生成走势', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input)
      return ok({ items: [], total: 0, offset: 0, limit: 200, execution: fixedExecution, request_path: path })
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<ResearchDataLab />)
    await settle(220)
    await screen.findByText('没有匹配的数据资源。')

    await user.click(screen.getByRole('button', { name: '宏观' }))
    await settle(220)
    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => String(path).includes('kind=macro'))).toBe(true))
    expect(screen.queryByTestId('research-chart')).not.toBeInTheDocument()
  })

  it('解析并校验上传文件，以 inline_rows 画像后只绑定后端返回的不可变数据源', async () => {
    const uploadItem = {
      id: 'upload:time_series', kind: 'upload', name: '用户上传时间序列', status: 'available', default_field: 'value', fields: ['value'], frequency: 'user_defined',
      coverage: { observations: 0 }, missing: {}, pit: { supported: false }, vintage: { supported: false },
      regime_node_type: 'source.inline', binding_parameters: { rows: [] }, capability: { accepted_formats: ['csv', 'json'], max_rows: 20000 },
    }
    const uploadProfile = {
      series: { ...uploadItem, name: '自定义市场状态信号', frequency: 'daily' }, coverage: { observations: 2 }, missing: { count: 0, rate: 0 },
      sampling: { computed_observations: 2, displayed_observations: 2, method: 'deterministic_even_spacing' },
      dates: ['2024-01-02', '2024-01-03'], values: { raw: [1.2, 1.3], normalized: [-1, 1] }, distribution: { raw: { valid_count: 2, mean: 1.25 } },
      pit: { supported: true, available_at: ['2024-01-03', '2024-01-04'] }, vintage: { supported: true, values: ['v1', 'v1'], revisions: [1, 2] },
      regime_node_type: 'source.upload', binding_parameters: { artifact_id: 'upload-artifact-1', checksum: 'sha256:abc', field: 'value' },
      binding: { node_type: 'source.upload', parameters: { artifact_id: 'upload-artifact-1', checksum: 'sha256:abc', field: 'value' }, fingerprint: 'sha256:abc' },
      snapshot: { source: 'immutable_upload_artifact', persisted: true, fingerprint: 'sha256:abc' }, execution: fixedExecution,
    }
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input)
      if (path.includes('/catalog?')) return ok({ items: [uploadItem], total: 1, offset: 0, limit: 200, execution: fixedExecution })
      if (path.endsWith('/profile') && init?.method === 'POST') return ok(uploadProfile)
      throw new Error(`Unexpected request: ${path}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    const onBind = vi.fn()
    const user = userEvent.setup()
    render(<ResearchDataLab onBindSeries={onBind} />)
    await settle(220)
    expect((await screen.findAllByText('用户上传时间序列')).length).toBeGreaterThan(0)

    const file = new File(['date,value,available_at,vintage,revision\n2024-01-02,1.2,2024-01-03,v1,1\n2024-01-03,1.3,2024-01-04,v1,2'], 'state-signal.csv', { type: 'text/csv' })
    await user.upload(screen.getByLabelText('选择研究时序文件'), file)
    expect(await screen.findByText(/已解析 2 行/)).toBeInTheDocument()
    await user.clear(screen.getByLabelText('上传序列名称'))
    await user.type(screen.getByLabelText('上传序列名称'), '自定义市场状态信号')
    await user.click(screen.getByRole('button', { name: '计算数据画像' }))
    expect(await screen.findByTestId('research-chart')).toBeInTheDocument()

    const profileCall = fetchMock.mock.calls.find(([path]) => String(path).endsWith('/profile'))
    const body = JSON.parse(String(profileCall?.[1]?.body))
    expect(body).toMatchObject({ series_id: 'upload:time_series', name: '自定义市场状态信号', frequency: 'daily', availability_mode: 'point_in_time', register_artifact: true })
    expect(body.inline_rows).toEqual([
      { date: '2024-01-02', value: 1.2, available_at: '2024-01-03', vintage: 'v1', revision: 1 },
      { date: '2024-01-03', value: 1.3, available_at: '2024-01-04', vintage: 'v1', revision: 2 },
    ])

    await user.click(screen.getByRole('button', { name: '加入计算图' }))
    expect(onBind).toHaveBeenCalledWith(expect.objectContaining({ regime_node_type: 'source.upload', binding_parameters: expect.objectContaining({ artifact_id: 'upload-artifact-1', checksum: 'sha256:abc' }) }))
  })

  it('选择 2–4 条真实序列，并展示后端 NJIT 返回的共同区间、相关矩阵和散点', async () => {
    const second = { ...series, id: 'index:index_daily:000905.SH', name: '中证500指数', code: '000905.SH', binding_parameters: { series_id: 'index:index_daily:000905.SH', field: 'close' } }
    const comparison = {
      schema_version: 'research-series-compare-v1',
      alignment: { method: 'strict_date_intersection', intersected_observations: 3, source_observations: [3, 3], computed_before_sampling: true },
      sampling: { method: 'deterministic_even_spacing', computed_observations: 3, displayed_observations: 3, sample_limit: 500, computed_before_sampling: true },
      dates: ['2024-01-02', '2024-01-03', '2024-01-04'],
      series: [
        { id: 'source-1', label: '沪深300指数 · close', series, values: [3500, 3510, 3490], standardized: [0, 1, -1], profile_observations: 3 },
        { id: 'source-2', label: '中证500指数 · close', series: second, values: [5000, 5020, 4980], standardized: [0, 1, -1], profile_observations: 3 },
      ],
      correlation: { method: 'pearson_pairwise_finite_after_strict_date_intersection', source_ids: ['source-1', 'source-2'], matrix: [[1, 0.88], [0.88, 1]], observation_counts: [[3, 3], [3, 3]] },
      scatter_pairs: [{ left_id: 'source-1', right_id: 'source-2', observation_count: 3, displayed_observations: 3, dates: ['2024-01-02', '2024-01-03', '2024-01-04'], x: [3500, 3510, 3490], y: [5000, 5020, 4980], standardized_x: [0, 1, -1], standardized_y: [0, 1, -1], correlation: 0.88 }],
      common_valid: { definition: 'all selected series finite on a strictly intersected date', observation_count: 3, start_date: '2024-01-02', end_date: '2024-01-04' },
      execution: fixedExecution,
    }
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input)
      if (path.includes('/catalog?')) return ok({ items: [series, second], total: 2, offset: 0, limit: 200, execution: fixedExecution })
      if (path.endsWith('/compare') && init?.method === 'POST') return ok(comparison)
      throw new Error(`Unexpected request: ${path}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<ResearchDataLab />)
    await settle(220)

    await user.click(screen.getByRole('button', { name: '加入多序列对比' }))
    await user.click(screen.getByRole('button', { name: /中证500指数/ }))
    await user.click(screen.getByRole('button', { name: '加入多序列对比' }))
    await user.click(screen.getByRole('button', { name: '比较 2 条序列' }))

    expect(await screen.findByText('共同有效起点')).toBeInTheDocument()
    expect(screen.getByText('2024-01-04')).toBeInTheDocument()
    expect(screen.getByRole('table', { name: '多序列相关性矩阵' })).toHaveTextContent('0.88')
    expect(screen.getByLabelText('散点序列组合')).toBeInTheDocument()
    expect(screen.getByText(/浏览器仅编排展示/)).toBeInTheDocument()
    const body = JSON.parse(String(fetchMock.mock.calls.find(([path]) => String(path).endsWith('/compare'))?.[1]?.body))
    expect(body.sources).toHaveLength(2)
    expect(body.sources[0]).toMatchObject({ series_id: series.id, field: 'close' })
    expect(body.sample_limit).toBe(500)
  })
})
