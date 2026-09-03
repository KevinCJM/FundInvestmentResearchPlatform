import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { EvaluationResult, IndicatorDefinition, MetricPresentation } from '../../services/customIndicators'
import { formatMetricValue, MetricDefinitionDrawer, MetricMatrix, MetricResultCard, MetricSelector, MetricStatus, MetricUnavailableReason, MetricValue } from './MetricDisplay'

const presentation = (overrides: Partial<MetricPresentation> = {}): MetricPresentation => ({
  indicator_id: 'metric-1', revision: 1, name: '测试指标', source: 'built_in',
  category: 'return', category_label: '收益', context_kind: 'single_product', catalog_status: 'current',
  display_format: 'number', precision: 2, unit: '', notation: 'standard', value_scale: 1,
  output_measure: 'dimensionless', direction: 'higher_better', description: '测试', methodology: '测试',
  data_basis: '真实数据', minimum_observations: 2, applicable_product_kinds: ['etf', 'fund'],
  ...overrides,
})

describe('统一指标展示协议', () => {
  it('公共指标选择器按业务指标类型和来源筛选，供研究、详情、对比和评价共用', () => {
    const definition = (id: string, name: string, indicatorType: 'return' | 'risk', source: 'built_in' | 'custom' = 'built_in'): IndicatorDefinition => ({
      id, revision: 1, source, read_only: source === 'built_in', created_at: '', updated_at: '',
      name, description: '', expression: 'mean(returns)', unit: '', display_format: 'number', precision: 2,
      direction: indicatorType === 'risk' ? 'lower_better' : 'higher_better', annual_risk_free_rate_percent: 0,
      context_kind: 'single_product', indicator_type: indicatorType, category_id: indicatorType,
      category_label: indicatorType === 'risk' ? '风险型指标' : '收益型指标',
    })
    render(<MetricSelector indicators={[definition('return', '年化收益率', 'return'), definition('risk', '年化波动率', 'risk'), definition('workspace-risk', '工作区风险', 'risk', 'custom')]} selectedIds={[]} onChange={() => undefined} />)

    fireEvent.click(screen.getByRole('button', { name: /选择指标/ }))
    expect(screen.getByRole('option', { name: '全部' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: '内置指标' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: '工作区指标' })).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('按指标类型筛选'), { target: { value: 'risk' } })
    fireEvent.change(screen.getByLabelText('按指标来源筛选'), { target: { value: 'custom' } })

    expect(screen.getByText(/工作区风险/)).toBeInTheDocument()
    expect(screen.queryByText(/年化波动率/)).not.toBeInTheDocument()
    expect(screen.queryByText(/年化收益率/)).not.toBeInTheDocument()
  })

  it('指标下拉面板按视口边缘修正位置，避免移出网页', () => {
    render(<MetricSelector indicators={[]} selectedIds={[]} onChange={() => undefined} />)
    const trigger = screen.getByRole('button', { name: /选择指标/ })
    const originalWidth = window.innerWidth
    const originalHeight = window.innerHeight
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 320 })
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: 640 })
    Object.defineProperty(trigger, 'getBoundingClientRect', {
      configurable: true,
      value: () => ({ left: -120, right: 80, top: 100, bottom: 144, width: 200, height: 44, x: -120, y: 100, toJSON: () => ({}) }),
    })

    fireEvent.click(trigger)

    const panel = screen.getByRole('dialog', { name: '选择指标面板' })
    expect(panel).toHaveStyle({ left: '16px', width: '288px', top: '152px' })
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: originalWidth })
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: originalHeight })
  })

  it('按展示协议处理百分比缩放、价格精度和紧凑数量单位', () => {
    expect(formatMetricValue(0.0183, presentation({ display_format: 'percent', value_scale: 100, unit: '%' }))).toBe('1.83%')
    expect(formatMetricValue(12.345, presentation({ unit: '元' }))).toBe('12.35 元')
    expect(formatMetricValue(1_234_567, presentation({ precision: 0, notation: 'compact', unit: '股' }))).toContain('万 股')
  })

  it('空值显示不可计算，并独立展示样本不足原因', () => {
    render(<div><MetricValue value={null} presentation={presentation()} /><MetricStatus status="warning" warnings={[{ code: 'INSUFFICIENT_SAMPLE', message: '至少需要 20 个观察值' }]} showReason /></div>)
    expect(screen.getByText('不可计算')).toBeInTheDocument()
    expect(screen.getByText('样本不足')).toBeInTheDocument()
    expect(screen.getByText('至少需要 20 个观察值')).toBeInTheDocument()
  })

  it('单产品结果卡不在移除按钮旁重复展示计算状态', () => {
    const metric = {
      id: 'metric-1', revision: 1, source: 'built_in', read_only: true, name: '测试指标',
      description: '测试', expression: 'mean(returns)', unit: '', display_format: 'number', precision: 2,
      direction: 'higher_better', annual_risk_free_rate_percent: 1.5, context_kind: 'single_product',
      created_at: '', updated_at: '', presentation: presentation(),
    } as IndicatorDefinition
    const result = {
      indicator_id: metric.id, indicator_revision: metric.revision, indicator_name: metric.name,
      target: { kind: 'etf', product_id: '510300.SH', name: '沪深300ETF' },
      period: '1Y', value: null, status: 'warning',
      warnings: [{ code: 'INSUFFICIENT_SAMPLE', message: '至少需要 20 个观察值' }],
      window: { requested_as_of: null, effective_as_of: '2026-01-06', start_date: null, end_date: null, observation_count: 0, data_latest_date: '2026-01-06' },
      presentation: metric.presentation!,
    } as EvaluationResult

    render(<MetricResultCard result={result} indicator={metric} onRemove={() => undefined} />)

    expect(screen.getByRole('button', { name: '移除指标 测试指标' })).toBeInTheDocument()
    expect(screen.queryByText('正常')).not.toBeInTheDocument()
    expect(screen.queryByText('样本不足')).not.toBeInTheDocument()
    expect(screen.getByText('不可计算')).toBeInTheDocument()
    expect(screen.getByText('至少需要 20 个观察值')).toBeInTheDocument()
  })

  it('不可计算原因默认展示中文字段，技术代码只在详情中出现', () => {
    render(<MetricUnavailableReason result={{
      value: null,
      status: 'unavailable',
      warnings: [{ code: 'INDICATOR_NOT_APPLICABLE', message: '指标缺少必需输入。' }],
      input_requirements: {
        status: 'blocked', required_count: 2, available_count: 0,
        blocking_inputs: [
          { variable_id: 'market_high', label: '最高价', status: 'source_unavailable', reason_code: 'SOURCE_UNAVAILABLE_FOR_PRODUCT', reason: '该产品没有日线最高价数据', source_dataset: 'ETF 日线行情', source_field: 'high' },
          { variable_id: 'market_low', label: '最低价', status: 'source_unavailable', reason_code: 'SOURCE_UNAVAILABLE_FOR_PRODUCT', reason: '该产品没有日线最低价数据', source_dataset: 'ETF 日线行情', source_field: 'low' },
        ],
      },
      target_data: { available_datasets: ['基金复权净值'], available_variables: ['adjusted_nav', 'returns'], data_latest_date: '2026-06-12' },
    }} />)

    expect(screen.getByText('该指标需要 2 个输入字段，当前产品缺少：最高价、最低价。')).toBeInTheDocument()
    expect(screen.getAllByRole('listitem').some((item) => item.textContent?.includes('最高价：该产品没有日线最高价数据'))).toBe(true)
    expect(screen.getAllByText(/SOURCE_UNAVAILABLE_FOR_PRODUCT/)[0]).not.toBeVisible()
    fireEvent.click(screen.getByText('查看技术详情'))
    expect(screen.getAllByText(/SOURCE_UNAVAILABLE_FOR_PRODUCT/)[0]).toBeVisible()
  })

  it('定义抽屉默认展示数学排版，公式源码只放在高级信息中', () => {
    const metric = {
      id: 'volatility', revision: 1, source: 'built_in', read_only: true, name: '年化波动率',
      description: '样本标准差乘以年化因子平方根。', expression: 'std(returns, 1) * sqrt(periods_per_year)',
      display_latex: '\\operatorname{Std}_{\\mathrm{ddof}=1}\\left(\\mathbf{r}\\right)\\sqrt{p_{\\mathrm{year}}}',
      unit: '%', display_format: 'percent', precision: 2, direction: 'lower_better', annual_risk_free_rate_percent: 1.5,
      context_kind: 'single_product', created_at: '', updated_at: '', presentation: presentation({ name: '年化波动率' }),
    } as const

    render(<MetricDefinitionDrawer indicator={metric} onClose={() => undefined} />)

    expect(screen.getByTestId('metric-formula-latex').querySelector('.katex')).not.toBeNull()
    fireEvent.click(screen.getByText('高级信息：查看公式源码'))
    expect(screen.getByText('std(returns, 1) * sqrt(periods_per_year)')).toBeInTheDocument()
  })

  it('比较矩阵按方向标出最佳与最弱结果且不只依赖颜色', () => {
    const metric = {
      id: 'risk', revision: 1, source: 'built_in', read_only: true, name: '风险指标', description: '风险越低越好', expression: 'std(returns, 1)', periods: ['1Y'], unit: '%', display_format: 'percent', precision: 2, direction: 'lower_better', annual_risk_free_rate_percent: 1.5, context_kind: 'single_product', created_at: '', updated_at: '',
      presentation: presentation({ indicator_id: 'risk', name: '风险指标', direction: 'lower_better', display_format: 'percent', value_scale: 100, unit: '%' }),
    } as const
    const window = { requested_as_of: null, effective_as_of: '2026-01-01', start_date: '2025-01-01', end_date: '2026-01-01', observation_count: 250, data_latest_date: '2026-01-01' }
    render(<MetricMatrix indicators={[metric]} targets={[{ kind: 'etf', product_id: 'A', name: '产品 A' }, { kind: 'etf', product_id: 'B', name: '产品 B' }]} results={[
      { indicator_id: 'risk', indicator_revision: 1, indicator_name: '风险指标', target: { kind: 'etf', product_id: 'A', name: '产品 A' }, period: '1Y', value: 0.1, status: 'ok', warnings: [], window, presentation: metric.presentation },
      { indicator_id: 'risk', indicator_revision: 1, indicator_name: '风险指标', target: { kind: 'etf', product_id: 'B', name: '产品 B' }, period: '1Y', value: 0.2, status: 'ok', warnings: [], window, presentation: metric.presentation },
    ]} />)
    expect(screen.getByText('最佳')).toBeInTheDocument()
    expect(screen.getByText('最弱')).toBeInTheDocument()
    expect(screen.getByText(/低值优先/)).toBeInTheDocument()
  })
})
