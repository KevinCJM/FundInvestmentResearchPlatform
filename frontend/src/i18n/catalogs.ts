import system from '../../../locales/system.json'
import navigation from '../../../locales/navigation.json'
import business from '../../../locales/business.json'

export const SUPPORTED_LOCALES = ['zh-CN', 'en-US'] as const
/** Bundled locales are fixed; workspace language columns are registered at runtime. */
export type BuiltinLocale = typeof SUPPORTED_LOCALES[number]
export type Locale = string
export interface LanguageDefinition { id: Locale; label: string; fallback_locale: Locale; enabled: boolean; builtin?: boolean; system_pack?: boolean }
export const DEFAULT_LANGUAGES: LanguageDefinition[] = [
  { id: 'zh-CN', label: '简体中文', fallback_locale: 'zh-CN', enabled: true, builtin: true, system_pack: true },
  { id: 'en-US', label: 'English', fallback_locale: 'zh-CN', enabled: true, builtin: true, system_pack: true },
]
export function normalizeLocale(value: unknown): Locale | null {
  if (typeof value !== 'string') return null
  const match = value.trim().match(/^([A-Za-z]{2,3})(?:-([A-Za-z]{4}))?(?:-([A-Za-z]{2}|[0-9]{3}))?$/)
  if (!match) return null
  return [match[1].toLowerCase(), match[2] ? match[2][0].toUpperCase() + match[2].slice(1).toLowerCase() : '', match[3]?.toUpperCase()].filter(Boolean).join('-')
}
export type TranslationScope = 'system' | 'business'
export type CatalogEntry = Partial<Record<Locale, string>> & { 'zh-CN': string }
export const DEFAULT_LOCALE = 'zh-CN' as const
export const isLocale = (value: unknown): value is Locale => normalizeLocale(value) !== null
export const builtinCatalogs: Record<TranslationScope, Record<string, CatalogEntry>> = {
  system: { ...system, ...Object.fromEntries(Object.entries(navigation).map(([key, entry]) => [`navigation.routes.${key}`, entry])) },
  business,
}

/** Fill missing languages from BUILT-IN defaults, never another locale's overrides. */
export const builtinBundle = (locale: Locale, fallback: Locale = DEFAULT_LOCALE): Record<TranslationScope, Record<string, string>> => ({
  system: Object.fromEntries(Object.entries(builtinCatalogs.system).map(([key, entry]) => [key, entry[locale] ?? entry[fallback] ?? entry[DEFAULT_LOCALE]])),
  business: Object.fromEntries(Object.entries(builtinCatalogs.business).map(([key, entry]) => [key, entry[locale] ?? entry[fallback] ?? entry[DEFAULT_LOCALE]])),
})
export const routeTranslationKey = (path: string) => `navigation.routes.${path.replace(/^\/+|\/+$/g, '').split('/').join('.') || 'home'}`
