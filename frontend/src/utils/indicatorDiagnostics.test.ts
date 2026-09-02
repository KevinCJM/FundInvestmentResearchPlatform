import { describe, expect, it } from 'vitest'
import {
  formatIndicatorDiagnostic,
  humanizeIndicatorMessage,
  humanizeIndicatorTechnicalText,
  indicatorDiagnosticDetail,
} from './indicatorDiagnostics'

describe('指标用户文案转换', () => {
  it('把内部数据类型和返回规则转换为中文', () => {
    expect(humanizeIndicatorTechnicalText('scalar | series<T> | vector<N> | matrix<A,B>')).toBe('有限标量 | 时间序列 | 资产向量 | 矩阵')
    expect(humanizeIndicatorTechnicalText('same(scalar | series<T>)')).toBe('与输入相同的数据类型')
    expect(humanizeIndicatorTechnicalText('elementwise')).toBe('逐元素计算')
  })

  it('诊断只返回用户可理解的标题和处理建议', () => {
    const message = formatIndicatorDiagnostic(
      'OUTPUT_CONTRACT_MISMATCH',
      '[OUTPUT_CONTRACT_MISMATCH] 输出契约要求 scalar，实际为 series<time>[T]。',
    )
    expect(message).toContain('指标结果类型不符合要求')
    expect(message).toContain('必须是单个有限数值')
    expect(message).not.toMatch(/OUTPUT_CONTRACT_MISMATCH|scalar|series<time>|\[T\]/)
  })

  it('公共指标组件也不暴露诊断代码', () => {
    expect(indicatorDiagnosticDetail('INSUFFICIENT_SAMPLE', '[INSUFFICIENT_SAMPLE] 至少需要 20 个观察值')).toBe('至少需要 20 个观察值')
  })

  it('自动识别字符串中的错误代码并转换为文字解释', () => {
    const message = humanizeIndicatorMessage('[TYPE_MISMATCH] expected scalar, actual series<time>[T]')
    expect(message).toContain('输入数据类型不匹配')
    expect(message).toContain('有限标量')
    expect(message).toContain('时间序列')
    expect(message).not.toMatch(/TYPE_MISMATCH|scalar|series<time>|\[T\]/)
  })
})
