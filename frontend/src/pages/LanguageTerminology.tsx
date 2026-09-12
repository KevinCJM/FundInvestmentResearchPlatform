import { useEffect, useRef, useState } from 'react'
import { DEFAULT_LANGUAGES, DEFAULT_LOCALE, isLocale, type LanguageDefinition, type Locale, type TranslationScope } from '../i18n/catalogs'
import TranslationMatrix from '../components/localization/TranslationMatrix'
import LanguageColumnsEditor from '../components/localization/LanguageColumnsEditor'
import { announceTranslationChange, chooseLocale, formatDate, installLanguageRegistry, resolveActiveLocale, storedLocale, useI18n } from '../i18n/runtime'
import { useLocalizationStatus } from '../i18n/LocalizationProvider'
import {
  applyTranslationImport, exportBusinessTranslations, getTranslationMatrix, getTranslationHistory, getTranslationState, updateLanguages,
  LocalizationApiError, restoreTranslations, updateBusinessTranslations, updateLanguagePreferences, validateTranslationImport,
  type ImportPreview, type TranslationMatrix as Matrix, type TranslationChange, type TranslationHistory, type TranslationPackage, type TranslationState,
} from '../services/localization'

const input = 'min-h-11 w-full min-w-0 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent-500'
const button = 'min-h-10 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40'

export default function LanguageTerminology() {
  const { s, locale } = useI18n()
  const runtime = useLocalizationStatus()
  const [tab, setTab] = useState<TranslationScope | 'history'>('system')
  const [state, setState] = useState<TranslationState | null>(null)
  const [catalog, setCatalog] = useState<Matrix | null>(null)
  const [languageManager, setLanguageManager] = useState(false)
  const [cellEditing, setCellEditing] = useState(false)
  const languageRevision = useRef(0)
  const [history, setHistory] = useState<TranslationHistory | null>(null)
  const [query, setQuery] = useState('')
  const [module, setModule] = useState('')
  const [status, setStatus] = useState('all')
  const [sortBy, setSortBy] = useState('code')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc')
  const [page, setPage] = useState(1)
  const [refresh, setRefresh] = useState(0)
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [pending, setPending] = useState<Record<string, TranslationChange>>({})
  const expectedRevision = useRef<number | null>(null)
  const loadedPreferencesRevision = useRef<number | null>(null)
  const [reason, setReason] = useState('')
  const [defaultLocale, setDefaultLocale] = useState<Locale>(DEFAULT_LOCALE)
  const [importReview, setImportReview] = useState<{ pack: TranslationPackage; preview: ImportPreview } | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const changes = Object.values(pending)
  const languages: LanguageDefinition[] = (catalog?.locales ?? state?.locales ?? DEFAULT_LANGUAGES).map(item => ({ ...item, fallback_locale: item.fallback_locale ?? DEFAULT_LOCALE, enabled: item.enabled !== false, builtin: item.builtin ?? DEFAULT_LANGUAGES.some(defaultItem => defaultItem.id === item.id) }))
  const errorMessage = (failure: unknown) => failure instanceof LocalizationApiError
    ? s(`errors.${failure.code}`, {}, failure.status === 409 ? s('i18n.conflict') : s('i18n.validation'))
    : s('common.requestFailed')

  useEffect(() => {
    const controller = new AbortController()
    let alive = true
    setLoading(true)
    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          const settings = await getTranslationState(controller.signal)
          const result = tab === 'history'
            ? await getTranslationHistory(controller.signal)
            : await getTranslationMatrix({ scope: tab, q: query, module, status, sortBy, sortDir, page }, controller.signal)
          if (!alive) return
          setState(settings)
          installLanguageRegistry(settings.locales)
          if (loadedPreferencesRevision.current !== settings.preferences_revision) {
            setDefaultLocale(settings.default_locale)
            loadedPreferencesRevision.current = settings.preferences_revision
          }
          if (tab === 'history') setHistory(result as TranslationHistory)
          else setCatalog(result as Matrix)
          setError('')
        } catch (failure) { if (alive && !controller.signal.aborted) setError(errorMessage(failure)) }
        finally { if (alive) setLoading(false) }
      })()
    }, query ? 200 : 0)
    return () => { alive = false; controller.abort(); window.clearTimeout(timer) }
  }, [tab, query, module, status, sortBy, sortDir, page, refresh])

  useEffect(() => {
    if (!changes.length) expectedRevision.current = null
    if (!changes.length && !cellEditing && !languageManager) return
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = '' }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [changes.length, cellEditing, languageManager])

  const edit = (edits: TranslationChange[]) => {
    if (tab !== 'business' || !catalog || busy || languageManager) return
    const next = { ...pending }
    for (const change of edits) {
      const row = catalog.items.find(item => item.key === change.key)
      if (!row?.customizable || !row.cells[change.locale]) return
      const id = `${change.locale}:${change.key}`
      if (row.cells[change.locale].override_value === change.value) delete next[id]
      else next[id] = change
    }
    if (Object.keys(next).length > 1000) { setError(s('i18n.matrix.batchLimit')); return }
    if (expectedRevision.current === null) expectedRevision.current = catalog.revision
    setPending(next)
  }
  const openLanguageManager = () => {
    if (changes.length || cellEditing) { setError(s('i18n.matrix.saveFirst')); return }
    languageRevision.current = catalog?.preferences_revision ?? state?.preferences_revision ?? 0
    setLanguageManager(true)
  }
  const saveLanguages = async (items: LanguageDefinition[]) => {
    setBusy(true); setError('')
    try {
      await updateLanguages(languageRevision.current, items)
      installLanguageRegistry(items)
      announceTranslationChange()
      setCatalog(null); setRefresh(value => value + 1)
      setMessage(s('i18n.matrix.languagesSaved'))
      return true
    } catch (failure) { setError(errorMessage(failure)); return false }
    finally { setBusy(false) }
  }
  const saved = (revision: number) => {
    setPending({}); expectedRevision.current = null; setImportReview(null); setReason('')
    setMessage(s('i18n.saved', { revision })); setError(''); setRefresh(value => value + 1)
    announceTranslationChange()
  }
  const save = async () => {
    if (!changes.length || expectedRevision.current === null) return
    setBusy(true); setError('')
    try { saved((await updateBusinessTranslations(expectedRevision.current, changes, reason)).revision) }
    catch (failure) { setError(errorMessage(failure)) }
    finally { setBusy(false) }
  }
  const reload = async () => {
    setBusy(true)
    try {
      const latest = await getTranslationState()
      setState(latest)
      // Explicit refresh rebases the preview, never automatically publishes it.
      if (changes.length) {
        expectedRevision.current = latest.revision
        setMessage(s('i18n.refreshReview', {}, '已刷新当前译文，未保存修改已保留。请检查修改预览后再保存。'))
      }
      if (languageManager) languageRevision.current = latest.preferences_revision
      setRefresh(value => value + 1); setError('')
    } catch (failure) { setError(errorMessage(failure)) }
    finally { setBusy(false) }
  }
  const saveDefault = async () => {
    if (!state) return
    setBusy(true)
    try {
      await updateLanguagePreferences(state.preferences_revision, defaultLocale)
      setMessage(s('i18n.defaultSaved')); setRefresh(value => value + 1); announceTranslationChange()
    } catch (failure) { setError(errorMessage(failure)) }
    finally { setBusy(false) }
  }
  const exportFile = async () => {
    setBusy(true)
    try {
      const pack = await exportBusinessTranslations()
      if (pack.scope !== 'business' || !Array.isArray(pack.entries)) throw new Error('Invalid translation export')
      const url = URL.createObjectURL(new Blob([JSON.stringify(pack, null, 2)], { type: 'application/json' }))
      const link = document.createElement('a'); link.href = url; link.download = 'business-translations.json'
      document.body.appendChild(link); link.click(); link.remove(); window.setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (failure) { setError(errorMessage(failure)) }
    finally { setBusy(false) }
  }
  const importFile = async (file?: File) => {
    if (!file || !state) return
    setBusy(true); setError(''); setImportReview(null)
    try {
      if (file.size > 1024 * 1024) throw new Error('Invalid file')
      const pack: TranslationPackage = JSON.parse(await file.text())
      if (pack?.scope !== 'business' || pack.format_version !== 1 || !Array.isArray(pack.entries)) throw new Error('Invalid file')
      const preview = await validateTranslationImport(state.revision, pack)
      setImportReview({ pack, preview })
    } catch (failure) { setError(failure instanceof LocalizationApiError ? errorMessage(failure) : s('i18n.invalidFile')) }
    finally { setBusy(false); if (fileRef.current) fileRef.current.value = '' }
  }
  const applyImport = async () => {
    if (!importReview) return
    setBusy(true)
    try { saved((await applyTranslationImport(importReview.preview.revision, importReview.pack, importReview.preview.confirmation_digest)).revision) }
    catch (failure) { setError(errorMessage(failure)) }
    finally { setBusy(false) }
  }
  const restore = async (revision: number) => {
    if (!history || !window.confirm(s('i18n.restoreConfirm', { revision }))) return
    setBusy(true)
    try { saved((await restoreTranslations(history.revision, revision)).revision) }
    catch (failure) { setError(errorMessage(failure)) }
    finally { setBusy(false) }
  }
  const sort = (field: string) => {
    if (cellEditing || busy || languageManager) return
    setPage(1)
    if (sortBy === field) setSortDir(current => current === 'asc' ? 'desc' : 'asc')
    else { setSortBy(field); setSortDir('asc') }
  }
  const clearFilters = () => {
    if (cellEditing || busy || languageManager) return
    setQuery(''); setModule(''); setStatus('all'); setPage(1)
  }
  const switchTab = (next: typeof tab) => { if (busy || languageManager || cellEditing) return; setTab(next); setPage(1); setModule(''); setStatus('all'); setQuery(''); setSortBy('code'); setSortDir('asc'); setCatalog(null); setImportReview(null) }
  const displayed = !loading && catalog?.scope === tab ? catalog : null

  return <div className="min-w-0 space-y-5" data-testid="language-terminology">
    <header className="rounded-xl border border-slate-200 bg-white p-5"><h1 className="text-2xl font-bold text-slate-900">{s('i18n.title')}</h1><p className="mt-2 text-sm leading-6 text-slate-600">{s('i18n.description')}</p>
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <label className="text-sm font-semibold">{s('i18n.language')}<select aria-label={s('i18n.language')} className={`${input} mt-1`} disabled={cellEditing || busy || languageManager} value={storedLocale() ? resolveActiveLocale(storedLocale(), state?.default_locale ?? DEFAULT_LOCALE) : ''} onChange={event => void chooseLocale(isLocale(event.target.value) ? event.target.value : null, state?.default_locale ?? DEFAULT_LOCALE)}><option value="">{s('i18n.followWorkspace')}</option>{languages.filter(item => item.enabled).map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select><span className="mt-2 block text-xs font-normal leading-5 text-slate-600">{s('i18n.browserHint')}</span></label>
        <details className="rounded-xl border border-slate-200 p-3"><summary className="cursor-pointer text-sm font-semibold">{s('i18n.defaultLanguage')}</summary><p className="my-2 text-xs leading-5 text-slate-600">{s('i18n.defaultHint')}</p><div className="flex gap-2"><select aria-label={s('i18n.defaultLanguage')} className={input} value={defaultLocale} disabled={busy || !state || cellEditing || languageManager} onChange={event => setDefaultLocale(event.target.value as Locale)}>{languages.filter(item => item.enabled).map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select><button type="button" className={button} disabled={busy || !state || cellEditing || languageManager} onClick={() => void saveDefault()}>{s('common.save')}</button></div></details>
      </div><p className="mt-3 text-xs leading-5 text-slate-600">{s('i18n.scopeWarning')}</p>
    </header>
    {languageManager && <LanguageColumnsEditor languages={languages} busy={busy} onSave={saveLanguages} onCancel={() => setLanguageManager(false)} />}
    {(runtime.offline || (!state && error)) && <p role="status" className="rounded-xl bg-amber-50 p-3 text-sm text-amber-900">{s('i18n.offline')}</p>}
    {error && <div role="alert" className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-900"><span>{error}</span><button type="button" className={button} disabled={busy || cellEditing} onClick={() => void reload()}>{s('common.refresh')}</button></div>}
    {message && <p role="status" className="rounded-xl bg-accent-50 p-3 text-sm text-accent-900">{message}</p>}
    <div className="flex flex-wrap gap-2" role="tablist" aria-label={s('i18n.tabs')}>{(['system', 'business', 'history'] as const).map(value => <button key={value} id={`i18n-tab-${value}`} aria-controls="i18n-panel" type="button" role="tab" aria-selected={tab === value} disabled={cellEditing || busy || languageManager} className={`${button} ${tab === value ? '!border-accent-500 !bg-accent-50 !text-accent-800' : ''}`} onClick={() => switchTab(value)}>{s(`i18n.${value}`)}</button>)}</div>
    <section id="i18n-panel" role="tabpanel" aria-labelledby={`i18n-tab-${tab}`} className="min-w-0 rounded-xl border border-slate-200 bg-white p-4 sm:p-5">
      <p className="mb-4 text-sm leading-6 text-slate-600">{s(tab === 'history' ? 'i18n.historyHint' : `i18n.${tab}Hint`)}</p>
      {tab !== 'history' && <>
        <div className="grid gap-3 sm:grid-cols-3"><label className="text-xs font-semibold">{s('i18n.search')}<input type="search" disabled={cellEditing || busy || languageManager} aria-label={s('i18n.search')} className={`${input} mt-1`} placeholder={s('i18n.matrix.searchHint')} value={query} onChange={event => { setQuery(event.target.value); setPage(1) }} /></label><label className="text-xs font-semibold">{s('i18n.module')}<select aria-label={s('i18n.module')} disabled={cellEditing || busy || languageManager} className={`${input} mt-1`} value={module} onChange={event => { setModule(event.target.value); setPage(1) }}><option value="">{s('common.all')}</option>{catalog?.modules.map(value => <option key={value} value={value}>{s(`i18n.modules.${value}`, {}, value)}</option>)}</select></label><label className="text-xs font-semibold">{s('i18n.status')}<select aria-label={s('i18n.status')} disabled={cellEditing || busy || languageManager} className={`${input} mt-1`} value={status} onChange={event => { setStatus(event.target.value); setPage(1) }}><option value="all">{s('common.all')}</option>{tab === 'business' && <option value="customized">{s('common.customized')}</option>}<option value="missing">{s('i18n.missing')}</option></select></label></div>
        <div className="mt-3 flex flex-wrap justify-end gap-2">{(query || module || status !== 'all') && <button type="button" className={button} disabled={busy || loading || languageManager || cellEditing} onClick={clearFilters}>{s('i18n.matrix.clearFilters')}</button>}<button type="button" className={button} disabled={busy || loading || !state || languageManager || cellEditing} onClick={openLanguageManager}>{s('i18n.matrix.manageLanguages')}</button></div>
        {displayed && <TranslationMatrix matrix={displayed} module={module} pending={pending} disabled={busy || languageManager} sortBy={sortBy} sortDir={sortDir} onSort={sort} onChanges={edit} onError={setError} onAddLanguage={openLanguageManager} onEditingChange={setCellEditing} />}
        {loading ? <p role="status" className="p-5 text-sm text-slate-600">{s('common.loading')}</p> : displayed?.items.length === 0 ? <p className="p-5 text-sm text-slate-600">{s('common.noResults')}</p> : null}
        <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600"><span>{s('common.page', { page, pages: Math.max(1, Math.ceil((catalog?.total ?? 0) / 50)), total: catalog?.total ?? 0 })}</span><div className="flex gap-2"><button type="button" className={button} disabled={loading || cellEditing || busy || languageManager || page <= 1} onClick={() => setPage(value => value - 1)}>{s('common.previous')}</button><button type="button" className={button} disabled={loading || cellEditing || busy || languageManager || page * 50 >= (catalog?.total ?? 0)} onClick={() => setPage(value => value + 1)}>{s('common.next')}</button></div></div>
        {catalog && <div className="mt-3 flex flex-wrap gap-3 text-xs text-slate-600">{Object.entries(catalog.coverage).map(([id, coverage]) => <p key={id}><code>{id}</code> · {s('i18n.coverage', coverage)}</p>)}</div>}
      </>}
      {tab === 'history' && <div className="overflow-auto"><table className="w-full min-w-[500px] text-left text-sm"><thead><tr><th scope="col" className="p-2">{s('i18n.revision')}</th><th scope="col" className="p-2">{s('i18n.time')}</th><th scope="col" className="p-2">{s('i18n.reason')}</th><th scope="col" className="p-2">{s('i18n.changeCount')}</th><th scope="col" /></tr></thead><tbody>{history?.items.map(item => <tr key={item.revision} className="border-t border-slate-100"><td className="p-2">{item.revision}</td><td className="p-2">{formatDate(item.at)}</td><td className="p-2">{s(`i18n.historyActions.${item.action}`, {}, item.action)}{item.reason ? ` · ${item.reason}` : ''}</td><td className="p-2">{item.change_count}</td><td className="p-2"><button type="button" className={button} disabled={busy || changes.length > 0 || item.revision === history.revision} onClick={() => void restore(item.revision)}>{s('i18n.restore')}</button></td></tr>)}</tbody></table></div>}
    </section>
    {tab === 'business' && <section className="space-y-3 rounded-xl border border-accent-200 bg-white p-4" aria-label={s('i18n.preview')}>
      {cellEditing && <p role="status" className="text-sm text-accent-800">{s('i18n.matrix.finishCell')}</p>}
      {changes.length > 0 && <><h2 className="font-semibold">{s('i18n.preview')} · {s('i18n.pending', { count: changes.length })}</h2><div className="max-h-56 overflow-auto">{changes.map(item => <div key={`${item.locale}:${item.key}`} className="border-b border-slate-100 py-2 text-sm"><code className="break-all text-xs text-slate-600">{item.locale} · {item.key}</code><p className="break-words"><span className="text-slate-600">{state?.overrides[item.locale]?.[item.key] ?? s('common.builtIn')}</span> → <strong>{item.value ?? s('common.builtIn')}</strong></p></div>)}</div><label className="block text-xs font-semibold">{s('i18n.reason')}<input className={`${input} mt-1`} value={reason} maxLength={200} onChange={event => setReason(event.target.value)} disabled={busy} /></label></>}
      <div className="flex flex-wrap gap-2"><button type="button" className={`${button} !border-accent-600 !bg-accent-600 !text-white`} disabled={busy || !changes.length || loading || cellEditing || languageManager} onClick={() => void save()}>{busy ? s('common.saving') : s('i18n.saveChanges')}</button><button type="button" className={button} disabled={busy || !changes.length || cellEditing} onClick={() => { setPending({}); expectedRevision.current = null }}>{s('common.cancel')}</button><button type="button" className={button} disabled={busy || cellEditing || languageManager || changes.length > 0 || !state} onClick={() => fileRef.current?.click()}>{s('i18n.import')}</button><button type="button" className={button} disabled={busy || cellEditing || languageManager || !state} onClick={() => void exportFile()}>{s('i18n.export')}</button><button type="button" className={button} disabled={busy || cellEditing} onClick={() => void reload()}>{s('common.refresh')}</button><input ref={fileRef} type="file" accept="application/json,.json" className="hidden" aria-label={s('i18n.import')} onChange={event => void importFile(event.target.files?.[0])} /></div>
    </section>}
    {changes.length > 0 && tab !== 'business' && <p role="status" className="rounded-lg bg-amber-50 p-3 text-sm text-amber-900">{s('i18n.pending', { count: changes.length })}</p>}
    {importReview && <section aria-label={s('i18n.importTitle')} className="rounded-xl border border-amber-200 bg-amber-50 p-4"><h2 className="font-semibold">{s('i18n.importTitle')}</h2><p className="mt-2 text-sm">{s('i18n.importHint')}</p><div className="my-3 max-h-64 overflow-auto">{importReview.preview.changes.map(item => <div className="border-b border-amber-100 py-2 text-sm" key={`${item.locale}:${item.key}`}><code className="break-all text-xs">{item.locale} · {item.key}</code><p className="break-words">{item.before ?? s('common.builtIn')} → <strong>{item.after ?? s('common.builtIn')}</strong></p></div>)}</div><div className="flex gap-2"><button type="button" className={button} disabled={busy || !importReview.preview.changes.length} onClick={() => void applyImport()}>{s('i18n.applyImport')}</button><button type="button" className={button} disabled={busy} onClick={() => setImportReview(null)}>{s('common.cancel')}</button></div></section>}
  </div>
}
