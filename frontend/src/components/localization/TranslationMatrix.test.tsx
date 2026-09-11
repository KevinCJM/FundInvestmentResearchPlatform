import { useState } from 'react'
import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import TranslationMatrix from './TranslationMatrix'
import { fixtureLanguages, matrixFixture, translationState } from '../../test/localizationFixtures'
import { DEFAULT_LANGUAGES } from '../../i18n/catalogs'
import { i18n, installLanguageRegistry } from '../../i18n/runtime'
import type { TranslationChange, TranslationMatrix as Matrix } from '../../services/localization'

beforeEach(async () => { installLanguageRegistry(DEFAULT_LANGUAGES, false); await i18n.changeLanguage('zh-CN') })
function Harness({ matrix = matrixFixture(), module = '', onChanges = vi.fn(), onError = vi.fn(), onSort = vi.fn() }: { matrix?: Matrix; module?: string; onChanges?: (changes: TranslationChange[]) => void; onError?: (message: string) => void; onSort?: (field: string) => void }) {
  const [pending, setPending] = useState<Record<string, TranslationChange>>({})
  return <TranslationMatrix matrix={matrix} module={module} pending={pending} disabled={false} sortBy={matrix.sort_by} sortDir={matrix.sort_dir} onSort={onSort} onAddLanguage={vi.fn()} onError={onError} onChanges={changes => { onChanges(changes); setPending(current => ({ ...current, ...Object.fromEntries(changes.map(change => [`${change.locale}:${change.key}`, change])) })) }} />
}
const cell = (key = 'axes.asset', locale = 'zh-CN') => screen.getByRole('gridcell', { name: `${key} ${locale}` })
describe('translation matrix interactions', () => {
  it('sortable headers expose state and request the selected column', async () => {
    const user = userEvent.setup(), onSort = vi.fn(); render(<Harness onSort={onSort} />)
    const codeHeader = screen.getByRole('columnheader', { name: /系统代码/ })
    expect(codeHeader).toHaveAttribute('aria-sort', 'ascending')
    await user.click(within(codeHeader).getByRole('button', { name: /按系统代码.*降序排序/ }))
    expect(onSort).toHaveBeenCalledWith('code')
    const english = screen.getByRole('columnheader', { name: /English/ })
    expect(english).toHaveAttribute('aria-sort', 'none')
    await user.click(within(english).getByRole('button', { name: /按English升序排序/ }))
    expect(onSort).toHaveBeenCalledWith('en-US')
  })
  it('shows short codes only in their namespace, full keys remain the cell identity', () => {
    render(<Harness module="axes" />)
    const rowHeaders = screen.getAllByRole('rowheader')
    expect(rowHeaders[0]).toHaveTextContent(/^asset$/)
    expect(rowHeaders[1]).toHaveTextContent(/^time$/)
    expect(cell()).toHaveTextContent('资产')
    expect(cell('axes.asset', 'en-US')).toHaveTextContent('Asset')
  })
  it('commits with Enter, cancels with Escape, and traverses columns with Tab', async () => {
    const user = userEvent.setup(), onChanges = vi.fn(); render(<Harness onChanges={onChanges} />)
    await user.dblClick(cell())
    let input = screen.getByRole('textbox', { name: '编辑译文 axes.asset zh-CN' })
    await user.clear(input); await user.type(input, '我的资产'); await user.keyboard('{Tab}')
    expect(cell('axes.asset', 'en-US')).toHaveFocus()
    expect(onChanges).toHaveBeenLastCalledWith([{ key: 'axes.asset', locale: 'zh-CN', value: '我的资产' }])
    await user.keyboard('{Enter}')
    input = screen.getByRole('textbox', { name: '编辑译文 axes.asset en-US' })
    await user.clear(input); await user.type(input, 'discard'); await user.keyboard('{Escape}')
    expect(cell('axes.asset', 'en-US')).toHaveTextContent('Asset')
    expect(onChanges).toHaveBeenCalledTimes(1)
  })
  it('copies a selected region and pastes a rectangle in one draft transaction', async () => {
    const user = userEvent.setup(), onChanges = vi.fn(); render(<Harness onChanges={onChanges} />)
    await user.click(cell()); await user.keyboard('{Shift>}{ArrowRight}{ArrowDown}{/Shift}')
    const setData = vi.fn(); fireEvent.copy(cell('axes.time', 'en-US'), { clipboardData: { setData } })
    expect(setData).toHaveBeenCalledWith('text/plain', '资产\tAsset\r\n时间\tTime')
    await user.click(cell())
    fireEvent.paste(cell(), { clipboardData: { getData: () => '我的资产\tMy asset\n我的时间\tMy time' } })
    expect(onChanges).toHaveBeenCalledTimes(1)
    expect(onChanges.mock.calls[0][0]).toHaveLength(4)
    expect(cell('axes.time', 'en-US')).toHaveTextContent('My time')
  })
  it('invalid clipboard content and immutable system cells cannot be changed', async () => {
    const user = userEvent.setup(), onChanges = vi.fn(), onError = vi.fn()
    const view = render(<Harness onChanges={onChanges} onError={onError} />)
    await user.click(cell()); fireEvent.paste(cell(), { clipboardData: { getData: () => 'valid\t<img src=x>' } })
    expect(onError).toHaveBeenCalled(); expect(onChanges).not.toHaveBeenCalled()
    view.unmount()
    render(<Harness matrix={matrixFixture('system', ['common.save'])} onChanges={onChanges} />)
    await user.dblClick(cell('common.save')); fireEvent.paste(cell('common.save'), { clipboardData: { getData: () => 'bad' } })
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument(); expect(onChanges).not.toHaveBeenCalled()
  })
  it('resetting a new-language override previews fallback without claiming a translation', async () => {
    const user = userEvent.setup()
    const state = { ...translationState(), locales: fixtureLanguages, overrides: { 'ja-JP': { 'axes.asset': '資産' } } }
    render(<Harness matrix={matrixFixture('business', ['axes.asset'], state)} />)
    await user.click(cell('axes.asset', 'ja-JP'))
    await user.click(within(screen.getByRole('region', { name: '当前单元格' })).getByRole('button', { name: '恢复默认' }))
    expect(cell('axes.asset', 'ja-JP')).toHaveTextContent('未翻译')
    expect(cell('axes.asset', 'ja-JP')).toHaveTextContent('Asset')
    expect(cell('axes.asset', 'ja-JP')).not.toHaveTextContent('資産')
  })
})
