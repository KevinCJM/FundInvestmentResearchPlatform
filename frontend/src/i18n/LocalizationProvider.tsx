import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import { I18nextProvider } from 'react-i18next'
import { getTranslationBundle, getTranslationState, type TranslationState } from '../services/localization'
import { DEFAULT_LOCALE, isLocale } from './catalogs'
import { i18n, installBusinessBundle, installLanguageRegistry, resolveActiveLocale, LANGUAGE_KEY, REFRESH_EVENT, REFRESH_KEY, storedLocale } from './runtime'

interface LocalizationContextValue { state: TranslationState | null; offline: boolean; refresh: () => void }
const Context = createContext<LocalizationContextValue>({ state: null, offline: false, refresh: () => undefined })
export const useLocalizationStatus = () => useContext(Context)

/** Language/resource updates never remount children or alter an in-progress definition. */
export default function LocalizationProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<TranslationState | null>(null)
  const [offline, setOffline] = useState(false)
  const refreshRef = useRef<() => void>(() => undefined)
  useEffect(() => {
    let sequence = 0
    let controller: AbortController | null = null
    let alive = true
    const refresh = () => {
      const current = ++sequence
      controller?.abort()
      controller = new AbortController()
      const signal = controller.signal
      const timeout = window.setTimeout(() => { if (current === sequence) controller?.abort() }, 10_000)
      void (async () => {
        try {
          const preference = storedLocale()
          if (preference && i18n.language !== preference) await i18n.changeLanguage(preference)
          const settings = await getTranslationState(signal)
          if (!alive || current !== sequence) return
          installLanguageRegistry(settings.locales)
          const locale = resolveActiveLocale(storedLocale(), isLocale(settings.default_locale) ? settings.default_locale : DEFAULT_LOCALE)
          const bundle = await getTranslationBundle(locale, signal)
          if (!alive || current !== sequence) return
          // A language selection made while this request was running wins.
          if (resolveActiveLocale(storedLocale(), settings.default_locale) !== locale || bundle.locale !== locale) { refresh(); return }
          installBusinessBundle(locale, bundle.resources.business)
          await i18n.changeLanguage(locale)
          if (!alive || current !== sequence) return
          document.documentElement.lang = locale
          setState(settings)
          setOffline(false)
        } catch {
          if (alive && current === sequence) setOffline(true)
        } finally { window.clearTimeout(timeout) }
      })()
    }
    refreshRef.current = refresh
    const onStorage = (event: StorageEvent) => { if (event.key === LANGUAGE_KEY || event.key === REFRESH_KEY || event.key === null) refresh() }
    const onFocus = () => { if (document.visibilityState !== 'hidden') refresh() }
    window.addEventListener(REFRESH_EVENT, refresh)
    window.addEventListener('storage', onStorage)
    window.addEventListener('focus', onFocus)
    const interval = window.setInterval(onFocus, 60_000)
    refresh()
    return () => {
      alive = false; sequence += 1; controller?.abort(); window.clearInterval(interval)
      window.removeEventListener(REFRESH_EVENT, refresh); window.removeEventListener('storage', onStorage); window.removeEventListener('focus', onFocus)
    }
  }, [])
  return <I18nextProvider i18n={i18n}><Context.Provider value={{ state, offline, refresh: () => refreshRef.current() }}>{children}</Context.Provider></I18nextProvider>
}
