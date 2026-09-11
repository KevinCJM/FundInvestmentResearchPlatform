import { describe, expect, it } from 'vitest'
import { matrixFixture } from '../../test/localizationFixtures'
import { parseTranslationTsv, pastedChanges, translationTsv, validateTranslationCell } from './matrixEditing'

describe('bounded translation clipboard', () => {
  it('round-trips quoted tabs, newlines, quotes, and empty cells', () => {
    const cells = [['资产', 'Asset'], ['a\tb', 'multi\nline'], ['"quoted"', '']]
    expect(parseTranslationTsv(translationTsv(cells) + '\r\n')).toEqual(cells)
  })
  it.each(['a\tb\nc', '"unclosed', '"closed"tail', 'a\tb\tc\n1\t2', 'a'.repeat(1024 * 1024 + 1)])('rejects malformed or oversized input', text => {
    expect(() => parseTranslationTsv(text)).toThrow()
  })
  it('maps a rectangle to stable full keys and the displayed locale order', () => {
    const matrix = matrixFixture()
    expect(pastedChanges(matrix.items, ['en-US', 'zh-CN'], 0, 0, 'My asset\t我的资产\nMy time\t我的时间')).toEqual([
      { key: 'axes.asset', locale: 'en-US', value: 'My asset' }, { key: 'axes.asset', locale: 'zh-CN', value: '我的资产' },
      { key: 'axes.time', locale: 'en-US', value: 'My time' }, { key: 'axes.time', locale: 'zh-CN', value: '我的时间' },
    ])
    expect(pastedChanges(matrix.items, ['zh-CN', 'en-US'], 0, 0, '资产\tAsset').every(item => item.value === null)).toBe(true)
  })
  it('rejects key-column writes, out-of-bounds rectangles and system rows atomically', () => {
    const matrix = matrixFixture(), locales = ['zh-CN', 'en-US']
    expect(() => pastedChanges(matrix.items, locales, 0, -1, 'x')).toThrow()
    expect(() => pastedChanges(matrix.items, locales, 1, 1, 'a\tb')).toThrow()
    expect(() => pastedChanges(matrixFixture('system', ['common.save']).items, locales, 0, 0, 'x')).toThrow()
    expect(() => pastedChanges(matrix.items, locales, 0, 0, 'valid\t<script>bad</script>')).toThrow()
    expect(matrix.items[0].cells['zh-CN'].value).toBe('资产')
  })
  it('enforces placeholders and lengths but never treats formula-like plain text as executable', () => {
    const row = { ...matrixFixture().items[0], placeholders: ['count'] }
    expect(() => validateTranslationCell(row, 'Total {{count}}')).not.toThrow()
    expect(() => validateTranslationCell(row, 'Total {{other}}')).toThrow()
    expect(() => validateTranslationCell(row, '{{count}} {{')).toThrow()
    expect(() => validateTranslationCell(row, '')).not.toThrow()
    expect(() => validateTranslationCell({ ...row, placeholders: [] }, '=SUM(A1:A2)')).not.toThrow()
    expect(() => validateTranslationCell(row, 'x'.repeat(121))).toThrow()
  })
})
