import { describe, expect, it } from 'vitest'
import { bundleResult, scalarBundle } from '../test/scalarOutputFixtures'
import { scalarOutputOptions, scalarSelectionKey, scalarSelectionRef, selectedOutputView } from './scalarOutputReferences'


describe('标量结果引用', () => {
  it('选择标识可逆，网络引用仍使用结构化字段', () => {
    const key = scalarSelectionKey('test-bundle', 'beta')
    expect(key).toBe('test-bundle::beta')
    expect(scalarSelectionRef(key)).toEqual({ indicator_id: 'test-bundle', output_id: 'beta' })
    expect(scalarSelectionRef('old-scalar')).toEqual({ indicator_id: 'old-scalar' })
  })
  it('每个结果保持独立名称、单位、方向、版本和公式', () => {
    const options = scalarOutputOptions([scalarBundle])
    expect(options.map(item => item.id)).toEqual(['test-bundle::mean', 'test-bundle::beta', 'test-bundle::missing'])
    expect(options[0]).toMatchObject({ name: '研究摘要 · 平均收益', display_format: 'percent', revision: 1, display_latex: '\\mu(r)' })
    expect(options[1].direction).toBe('neutral')
    expect(scalarBundle.result_kind).toBe('scalar_bundle')
    const result = selectedOutputView(bundleResult.outputs![0])
    expect(result.indicator_id).toBe(options[0].id)
    expect(result.value).toBe(0.032)
    expect(result.presentation).toEqual(bundleResult.outputs![0].presentation)
  })
})
