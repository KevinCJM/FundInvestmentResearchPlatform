import { builtinCatalogs, DEFAULT_LANGUAGES, type LanguageDefinition, type TranslationScope } from '../i18n/catalogs'
import type { TranslationMatrix, TranslationState } from '../services/localization'

export const fixtureLanguages: LanguageDefinition[] = [...DEFAULT_LANGUAGES.map(item => ({ ...item })), { id: 'ja-JP', label: '日本語', fallback_locale: 'en-US', enabled: true, builtin: false, system_pack: false }]
export const translationState = (): TranslationState => ({ revision: 0, preferences_revision: 0, default_locale: 'zh-CN', catalog_version: 'test', overrides: {}, locales: DEFAULT_LANGUAGES.map(item => ({ ...item })) })
export function matrixFixture(scope: TranslationScope = 'business', keys = ['axes.asset', 'axes.time'], state = translationState()): TranslationMatrix {
  const locales = state.locales.map(item => ({ fallback_locale: 'zh-CN', enabled: true, ...item }))
  const items = keys.map(key => {
    const defaults = builtinCatalogs[scope][key]
    const cells = Object.fromEntries(locales.map(language => {
      const custom = scope === 'business' ? state.overrides[language.id]?.[key] ?? null : null
      const base = defaults[language.id] ?? null
      const fallback = defaults[language.fallback_locale] ?? defaults['zh-CN']
      return [language.id, { value: custom ?? base, default_value: base, override_value: custom, source: custom !== null ? 'custom' as const : base !== null ? 'builtin' as const : 'missing' as const, fallback_value: base === null ? fallback : null, fallback_locale: base === null ? language.fallback_locale : null }]
    }))
    const [module, ...code] = key.split('.')
    return { key, code: code.join('.'), scope, module, cells, customizable: scope === 'business', max_length: key.endsWith('.description') ? 1600 : 120, placeholders: [], usage: [module] }
  })
  return { scope, revision: state.revision, preferences_revision: state.preferences_revision, catalog_version: 'test', locales, items, modules: [...new Set(items.map(item => item.module))], total: items.length, page: 1, page_size: 50, coverage: Object.fromEntries(locales.map(item => [item.id, { total: items.length, missing: items.filter(row => row.cells[item.id].source === 'missing').length }])), sort_by: 'code', sort_dir: 'asc' }
}
