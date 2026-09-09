import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { EvaluationResult } from '../../services/customIndicators'
import { bundleResult, scalarBundle } from '../../test/scalarOutputFixtures'
import { MetricDefinitionDrawer, MetricMatrix, MetricResultCard } from './MetricDisplay'


describe('多标量跨页面展示', () => {
  it('定义抽屉展示每个结果的公式，不显示兼容指标错误提示', () => {
    render(<MetricDefinitionDrawer indicator={scalarBundle} onClose={() => undefined} />)
    expect(screen.getByLabelText('各结果的公式与口径')).toBeInTheDocument()
    expect(screen.getByText('平均收益')).toBeInTheDocument()
    expect(screen.getByText('Beta')).toBeInTheDocument()
    expect(screen.getAllByTestId('scalar-output-formula')).toHaveLength(2)
    expect(screen.queryByText('该兼容指标暂未提供数学符号排版。')).not.toBeInTheDocument()
  })
  it('组卡片使用锁定的各结果格式，不把组或失败结果当成零', () => {
    render(<MetricResultCard result={bundleResult} indicator={{ ...scalarBundle, precision: 8 }} />)
    expect(screen.getByText('3.20%')).toBeInTheDocument()
    expect(screen.getByText('1.100')).toBeInTheDocument()
    expect(screen.getByText('不可计算结果')).toBeInTheDocument()
    expect(screen.queryByText('0.00')).not.toBeInTheDocument()
  })

  it('结果窗口不一致时明确说明，不把第一项窗口冒充整组', () => {
    render(<MetricResultCard result={{ ...bundleResult, window_scope: 'per_output' }} indicator={scalarBundle} />)
    expect(screen.getByText('各结果使用的数据窗口不同，实际区间见结果明细。')).toBeInTheDocument()
  })

  it('比较矩阵展开各结果，Beta 默认不标记最佳或最弱', () => {
    const target = { kind: 'etf' as const, product_id: '510300.SH', name: '测试产品二' }
    const second: EvaluationResult = { ...bundleResult, target, outputs: bundleResult.outputs!.map(item => ({ ...item, target, value: item.value === null ? null : item.value * 2 })) }
    render(<MetricMatrix indicators={[scalarBundle]} targets={[{ ...bundleResult.target, kind: 'etf' }, target]} results={[bundleResult, second]} />)
    const row = screen.getByRole('row', { name: /研究摘要 · Beta/ })
    expect(within(row).getByText('1.100')).toBeInTheDocument()
    expect(within(row).getByText('2.200')).toBeInTheDocument()
    expect(within(row).queryByText('最佳')).not.toBeInTheDocument()
    expect(within(row).queryByText('最弱')).not.toBeInTheDocument()
    expect(screen.getAllByText('最佳')).toHaveLength(1)
  })
})
