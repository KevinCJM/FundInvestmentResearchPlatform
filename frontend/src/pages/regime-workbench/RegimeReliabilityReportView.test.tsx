import { fireEvent, render, screen, within } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import RegimeReliabilityReportView from './RegimeReliabilityReportView'
import { reliabilityPreviewFixture } from './regimeReliabilityFixtures'
vi.mock('echarts-for-react', () => ({ default: () => <div aria-label="可靠性图" /> }))
it('未知参考、拒识与自定义多状态保持区分，显示最终状态的校准值', () => {
  const report = reliabilityPreviewFixture().report
  report.states = [...report.states, { id: 'neutral', label: '中性', color: report.states[0].color }, { id: 'stress', label: '压力', color: report.states[1].color }]
  report.points[0].calibrated_probabilities = { expansion: 0.2, contraction: 0.6, neutral: 0.1, stress: 0.1 }
  report.points[0].calibrated_confidence = 0.2
  report.points[0].decision_status = 'below_floor'
  render(<RegimeReliabilityReportView report={report} />)
  expect(screen.getByText(/校准后的匹配概率/)).toHaveTextContent('20.0%')
  expect(screen.getByText(/参考在此日未分类/)).toBeVisible()
  expect(screen.getByText(/低于接受门槛/)).toBeVisible()
  expect(screen.getByText(/校准后的匹配概率/)).not.toHaveTextContent('60.0%')
})
it('缺失指标显示无法估计；原始覆盖与校准门槛覆盖分开展示', () => {
  const report = reliabilityPreviewFixture().report
  report.classification.accuracy = null
  report.points = []
  report.selective_classification = { ...report.classification, accepted_coverage: 0.5, accepted_error: null }
  report.probability.blocks.holdout.calibrated = { samples: 0, brier: null, logloss: null, ece: null, reason: 'no_valid_probability_samples', bins: [] }
  render(<RegimeReliabilityReportView report={report} />)
  expect(screen.getByText(/应用校准门槛后/)).toHaveTextContent('接受覆盖 50.0% · 接受后错误率 无法估计')
  fireEvent.click(screen.getByText('概率评分与可靠性图'))
  expect(screen.getByText(/没有可用的校准概率样本/)).toBeVisible()
  const row = within(screen.getByRole('table', { name: '概率评分' })).getByRole('row', { name: /校准结果/ })
  expect(within(row).getAllByRole('cell').map(cell => cell.textContent)).toEqual(['0', '无法估计', '无法估计', '无法估计'])
})

it('旧报告按保存状态展示，不补写为已执行；未知原因不暴露原始 token', () => {
  const report = reliabilityPreviewFixture().report
  report.confidence_interval = undefined
  render(<RegimeReliabilityReportView report={report} />)
  expect(screen.getByText(/旧报告未记录区间估计/)).toBeVisible()
  expect(screen.getByText(/旧报告没有参数稳定性证据/)).toBeVisible()
  expect(screen.queryByText('not run')).not.toBeInTheDocument()
})
it.each(['available', 'unavailable'] as const)('区分 %s 区间、完整时间块不足与今日证据', status => {
  const report = reliabilityPreviewFixture().report
  report.confidence_interval = { status, reason: status === 'unavailable' ? 'insufficient_full_blocks' : null, method: 'paired_moving_block', scope: 'holdout', confidence_level: 0.95, block_length: 10, replicates: 200, seed: 1729, full_blocks: status === 'available' ? 8 : 1, complete_cycles: 4, metrics: { accuracy: { estimate: 0.75, lower: status === 'available' ? 0.65 : null, upper: status === 'available' ? 0.85 : null, valid_replicates: status === 'available' ? 200 : 0, reason: status === 'unavailable' ? 'insufficient_full_blocks' : null, unit: 'fraction' } } }
  report.points[0].probability_evidence = { selected_probability: 0.6, top_probability: 0.6, second_probability: 0.4, margin: 0.2, entropy: 0.67, entropy_unit: 'nats' }
  render(<RegimeReliabilityReportView report={report} />)
  expect(screen.getByText(/不是今天状态的概率范围/)).toBeVisible()
  expect(screen.getByText(/前两名间距 20.0%/)).toHaveTextContent('概率熵 0.670')
  fireEvent.click(screen.getByText('置信区间与重采样详情'))
  const table = screen.getByRole('table', { name: '历史指标区间' })
  expect(table).toHaveTextContent(status === 'available' ? '65.0%' : '无法估计')
  expect(table).toHaveTextContent(status === 'available' ? '85.0%' : '完整时间块不足')
  expect(screen.getByText(/每块 10 个观测/)).toHaveTextContent('时间块不等于状态区间')
})
it('真实变体可见，种子不适用与执行失败分开，诊断报告不显示可用资格', () => {
  const report = reliabilityPreviewFixture().report
  report.status = 'eligible' // Older server may have used this loosely; actual deployment flag still controls.
  report.stability = { status: 'causal_probes_executed', parameter_sensitivity: { status: 'partial', seed_status: 'not_applicable', variants: [
    { kind: 'window', status: 'completed', changes: [{ node_id: 'window', parameter: 'length', before: 20, after: 22 }], agreement: 0.9, comparable_observations: 10, classification_coverage: 0.8, boundary_distance: 2 },
    { kind: 'parameter', status: 'failed', changes: [], agreement: null, reason: 'input_not_supported' },
  ] } }
  render(<RegimeReliabilityReportView report={report} />)
  expect(screen.getByText('不可部署')).toBeVisible()
  expect(screen.getByText(/随机种子：不适用/)).toBeVisible()
  fireEvent.click(screen.getByText('参数、窗口与边界变化详情'))
  const table = screen.getByRole('table', { name: '稳定性变体' })
  expect(table).toHaveTextContent('20 → 22')
  expect(table).toHaveTextContent('执行失败')
  expect(table).toHaveTextContent('90.0%')
  expect(table).not.toHaveTextContent('input_not_supported')
})


it('稀有状态显示证据不足而不是失败，并明确未验证状态不能授权实时信号', () => {
  const report = reliabilityPreviewFixture().report
  render(<RegimeReliabilityReportView report={report} />)
  expect(screen.getByLabelText('状态级验证')).toHaveTextContent('部分验证通过')
  expect(screen.getByLabelText('状态级验证')).toHaveTextContent('识别验证可用')
  const table = screen.getByRole('table', { name: '逐状态验证结果' })
  expect(within(table).getByRole('row', { name: /扩张/ })).toHaveTextContent('已验证')
  expect(within(table).getByRole('row', { name: /收缩/ })).toHaveTextContent('证据不足')
  expect(screen.getByText(/未验证状态不能授权下游实时 Regime 决策/)).toBeVisible()
})
