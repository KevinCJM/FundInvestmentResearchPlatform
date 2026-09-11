import { act, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import LocalizationProvider, { useLocalizationStatus } from './LocalizationProvider'
import { builtinBundle, DEFAULT_LANGUAGES } from './catalogs'
import { chooseLocale, i18n, installBusinessBundle, useI18n } from './runtime'
import { getTranslationBundle, getTranslationState, type TranslationBundle, type TranslationState } from '../services/localization'

vi.mock('../services/localization', () => ({ getTranslationState: vi.fn(), getTranslationBundle: vi.fn() }))
const state: TranslationState = { revision: 0, preferences_revision: 0, default_locale: 'zh-CN', overrides: {}, locales: DEFAULT_LANGUAGES, catalog_version: 'test' }
const bundle = (locale: string, label?: string): TranslationBundle => ({ locale, revision: 0, default_locale: 'zh-CN', catalog_version: 'test', fallback_keys: { system: [], business: [] }, resources: { ...builtinBundle(locale), business: { ...builtinBundle(locale).business, ...(label ? { 'variables.periods_per_year.label': label } : {}) } } })
function View() {
  const { b, s, locale } = useI18n()
  const { offline } = useLocalizationStatus()
  return <div><p>{locale}</p><p>{s('common.save')}</p><p>{b('variables.periods_per_year.label')}</p><p>{offline ? 'offline' : 'online'}</p><input aria-label="draft" defaultValue="unchanged" /></div>
}
beforeEach(async () => {
  vi.clearAllMocks(); localStorage.clear()
  installBusinessBundle('zh-CN', {}); installBusinessBundle('en-US', {})
  await i18n.changeLanguage('zh-CN')
  vi.mocked(getTranslationState).mockResolvedValue(state)
  vi.mocked(getTranslationBundle).mockImplementation(async locale => bundle(locale))
})
afterEach(async () => { localStorage.clear(); await i18n.changeLanguage('zh-CN') })

describe('localization provider', () => {
  it('uses workspace default only without an explicit browser preference', async () => {
    vi.mocked(getTranslationState).mockResolvedValue({ ...state, default_locale: 'en-US' })
    render(<LocalizationProvider><View /></LocalizationProvider>)
    await screen.findByText('Save')
    expect(screen.getByText('en-US')).toBeInTheDocument()
  })
  it('does not trust remote system-translation overrides', async () => {
    vi.mocked(getTranslationBundle).mockResolvedValue({ ...bundle('zh-CN'), resources: { system: { 'common.save': '恶意系统覆盖' }, business: { 'variables.periods_per_year.label': '工作区名称' } } })
    render(<LocalizationProvider><View /></LocalizationProvider>)
    expect(await screen.findByText('工作区名称')).toBeInTheDocument()
    expect(screen.getByText('保存')).toBeInTheDocument()
    expect(screen.queryByText('恶意系统覆盖')).not.toBeInTheDocument()
  })
  it('ignores a late response from the previous locale and preserves the editor element', async () => {
    let resolveOld!: (result: TranslationBundle) => void
    vi.mocked(getTranslationBundle).mockImplementation(locale => locale === 'zh-CN' ? new Promise(resolve => { resolveOld = resolve }) : Promise.resolve(bundle('en-US', 'Current English term')))
    render(<LocalizationProvider><View /></LocalizationProvider>)
    await waitFor(() => expect(getTranslationBundle).toHaveBeenCalledWith('zh-CN', expect.any(AbortSignal)))
    const input = screen.getByRole('textbox', { name: 'draft' })
    await act(async () => { await chooseLocale('en-US') })
    await screen.findByText('Current English term')
    await act(async () => resolveOld(bundle('zh-CN', '迟到名称')))
    expect(screen.getByText('en-US')).toBeInTheDocument()
    expect(screen.queryByText('迟到名称')).not.toBeInTheDocument()
    expect(screen.getByRole('textbox', { name: 'draft' })).toBe(input)
  })
  it('shows offline state without breaking built-in translation', async () => {
    vi.mocked(getTranslationState).mockRejectedValue(new Error('offline'))
    render(<LocalizationProvider><View /></LocalizationProvider>)
    await screen.findByText('offline')
    expect(screen.getByText('保存')).toBeInTheDocument()
    expect(screen.getByText('年化因子')).toBeInTheDocument()
  })
})
