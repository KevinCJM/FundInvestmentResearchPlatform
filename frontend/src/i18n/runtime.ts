import { createInstance } from 'i18next'
import { initReactI18next, useTranslation } from 'react-i18next'
import { useSyncExternalStore } from 'react'
import { builtinBundle, DEFAULT_LANGUAGES, DEFAULT_LOCALE, isLocale, normalizeLocale, SUPPORTED_LOCALES, type LanguageDefinition, type Locale, type TranslationScope } from './catalogs'

export const LANGUAGE_KEY = 'fund-research.i18n.locale'
export const REFRESH_KEY = 'fund-research.i18n.revision'
export const REFRESH_EVENT = 'fund-research:i18n-refresh'
export type TextParameters = Record<string, string | number>
export const LANGUAGES_KEY = 'fund-research.i18n.languages'
let registeredLanguages = DEFAULT_LANGUAGES.map(item => ({ ...item }))
const languageFallback = (locale: Locale) => registeredLanguages.find(item => item.id === locale)?.fallback_locale ?? DEFAULT_LOCALE

export function storedLocale(): Locale | null {
  try { const value = localStorage.getItem(LANGUAGE_KEY); return isLocale(value) ? value : null } catch { return null }
}
export const i18n = createInstance()
void i18n.use(initReactI18next).init({
  lng: storedLocale() ?? DEFAULT_LOCALE,
  // Selection is restricted by the workspace registry, not a fixed i18next list.
  supportedLngs: false, load: 'currentOnly', fallbackLng: DEFAULT_LOCALE,
  ns: ['system', 'business'], defaultNS: 'system', fallbackNS: false,
  resources: Object.fromEntries(SUPPORTED_LOCALES.map(locale => [locale, builtinBundle(locale)])),
  keySeparator: false, nsSeparator: false, initAsync: false,
  interpolation: { escapeValue: false, skipOnVariables: true },
  returnEmptyString: false, returnNull: false,
  react: { useSuspense: false, bindI18n: 'languageChanged loaded', bindI18nStore: 'added removed' },
})

let revision = 0
const subscribers = new Set<() => void>()
const changed = () => { revision += 1; subscribers.forEach(notify => notify()) }
i18n.on('languageChanged', changed)
i18n.store.on('added', changed)
i18n.store.on('removed', changed)
const subscribe = (notify: () => void) => { subscribers.add(notify); return () => { subscribers.delete(notify) } }
const snapshot = () => revision

function translate(scope: TranslationScope, key: string, fallback?: string, parameters: TextParameters = {}): string {
  if (!/^[A-Za-z][A-Za-z0-9_.-]*$/.test(key)) return fallback || (i18n.language === 'en-US' ? 'Text unavailable' : '名称待补充')
  return String(i18n.t(key, { ns: scope, defaultValue: fallback || (i18n.language === 'en-US' ? 'Text unavailable' : '名称待补充'), replace: parameters }))
}
export const systemText = (key: string, parameters: TextParameters = {}, fallback?: string) => translate('system', key, fallback, parameters)
export const businessText = (key: string, fallback?: string, parameters: TextParameters = {}) => translate('business', key, fallback, parameters)
export const hasBusinessText = (key: string) => i18n.exists(key, { ns: 'business' })

export function useI18n() {
  useTranslation(['system', 'business'], { i18n, useSuspense: false })
  const version = useSyncExternalStore(subscribe, snapshot, snapshot)
  return { s: systemText, b: businessText, locale: (isLocale(i18n.language) ? i18n.language : DEFAULT_LOCALE) as Locale, version }
}

/** No lookup is allowed to redirect one namespace to another. */
export function installBusinessBundle(locale: Locale, entries: Record<string, string>) {
  const safe: Record<string, string> = { ...builtinBundle(locale, languageFallback(locale)).business }
  for (const [key, value] of Object.entries(entries)) {
    if (/^[A-Za-z][A-Za-z0-9_.-]*$/.test(key) && !key.split('.').some(part => ['constructor', 'prototype', '__proto__'].includes(part)) && typeof value === 'string' && value.trim() && value.length <= 1600) safe[key] = value
  }
  i18n.removeResourceBundle(locale, 'business')
  i18n.addResourceBundle(locale, 'business', safe, false, true)
}

export function announceTranslationChange() {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new Event(REFRESH_EVENT))
  try { localStorage.setItem(REFRESH_KEY, String(Date.now())) } catch { /* In-memory updates still work when storage is disabled. */ }
}

export function installLanguageRegistry(items: Array<Pick<LanguageDefinition, 'id' | 'label'> & Partial<LanguageDefinition>>, persist = true) {
  if (items.length < 2 || items.length > 24) throw new Error('Invalid language registry')
  const next = items.map(item => {
    const id = normalizeLocale(item.id)
    const builtin = DEFAULT_LANGUAGES.find(language => language.id === id)
    const fallback = item.fallback_locale ?? DEFAULT_LOCALE
    if (!id || typeof item.label !== 'string' || !item.label.trim() || item.label.length > 60 || /[<>\u0000-\u001f]/.test(item.label) || !SUPPORTED_LOCALES.includes(fallback as 'zh-CN' | 'en-US')) throw new Error('Invalid language registry')
    return builtin ? { ...builtin } : { id, label: item.label, fallback_locale: fallback, enabled: item.enabled !== false, builtin: false, system_pack: false }
  })
  if (new Set(next.map(item => item.id)).size !== next.length || DEFAULT_LANGUAGES.some(item => !next.some(language => language.id === item.id))) throw new Error('Invalid language registry')
  const previous = registeredLanguages
  registeredLanguages = next
  for (const item of next) {
    if (!i18n.hasResourceBundle(item.id, 'system') || previous.find(old => old.id === item.id)?.fallback_locale !== item.fallback_locale) {
      // System text only comes from bundled resources, never workspace payloads.
      i18n.addResourceBundle(item.id, 'system', builtinBundle(item.id, item.fallback_locale).system, false, true)
      installBusinessBundle(item.id, {})
    }
  }
  if (persist) { try { localStorage.setItem(LANGUAGES_KEY, JSON.stringify(next)) } catch { /* Optional offline cache. */ } }
  changed()
}

export function availableLanguages(): LanguageDefinition[] { return registeredLanguages.map(item => ({ ...item })) }
export function resolveActiveLocale(preference: Locale | null, workspaceDefault: Locale): Locale {
  return [preference, workspaceDefault, DEFAULT_LOCALE].find(id => registeredLanguages.some(item => item.id === id && item.enabled)) ?? DEFAULT_LOCALE
}

// Restore only language metadata; translations still come from reviewed bundles.
try { const cached = localStorage.getItem(LANGUAGES_KEY); if (cached) installLanguageRegistry(JSON.parse(cached), false) } catch { /* Invalid cache never prevents startup. */ }

export async function chooseLocale(locale: Locale | null, workspaceDefault: Locale = DEFAULT_LOCALE) {
  const active = resolveActiveLocale(locale ? normalizeLocale(locale) : null, workspaceDefault)
  try { if (locale) localStorage.setItem(LANGUAGE_KEY, active); else localStorage.removeItem(LANGUAGE_KEY) } catch { /* Private-mode storage may be unavailable. */ }
  await i18n.changeLanguage(active)
  if (typeof document !== 'undefined') document.documentElement.lang = active
  announceTranslationChange()
}

export function formatNumber(value: number | null | undefined, options: Intl.NumberFormatOptions = {}, locale: Locale = isLocale(i18n.language) ? i18n.language : DEFAULT_LOCALE): string {
  if (value == null || !Number.isFinite(value)) return '—'
  return new Intl.NumberFormat(locale, options).format(value)
}
export function formatDate(value: string | null | undefined, locale: Locale = isLocale(i18n.language) ? i18n.language : DEFAULT_LOCALE): string {
  if (!value) return '—'
  const date = new Date(value)
  if (!Number.isFinite(date.getTime())) return '—'
  // Date-only API values must not shift to the previous day in western zones.
  return new Intl.DateTimeFormat(locale, { year: 'numeric', month: '2-digit', day: '2-digit', timeZone: 'UTC' }).format(date)
}
