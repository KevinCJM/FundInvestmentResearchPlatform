import { useEffect, useRef, useState } from 'react'
import { DEFAULT_LANGUAGES, normalizeLocale, type LanguageDefinition } from '../../i18n/catalogs'
import { useI18n } from '../../i18n/runtime'

interface Props { languages: LanguageDefinition[]; busy: boolean; onSave: (languages: LanguageDefinition[]) => Promise<boolean>; onCancel: () => void }
const input = 'min-h-10 w-full min-w-0 rounded-xl border border-slate-300 bg-white px-2 py-2 text-sm disabled:bg-slate-100'
const button = 'min-h-10 rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm font-semibold disabled:opacity-40'

export default function LanguageColumnsEditor({ languages, busy, onSave, onCancel }: Props) {
  const { s } = useI18n()
  const [items, setItems] = useState(() => languages.map(item => ({ ...item })))
  const [code, setCode] = useState('')
  const [name, setName] = useState('')
  const [fallback, setFallback] = useState('zh-CN')
  const [error, setError] = useState('')
  const codeRef = useRef<HTMLInputElement>(null)
  useEffect(() => { codeRef.current?.focus() }, [])
  const move = (index: number, delta: number) => setItems(current => {
    const next = [...current], target = index + delta
    if (target >= 0 && target < next.length) [next[index], next[target]] = [next[target], next[index]]
    return next
  })
  const save = async () => {
    const id = normalizeLocale(code)
    if ((code || name) && (!id || !name.trim() || /[<>\u0000-\u001f]/.test(name) || items.some(item => item.id === id) || items.length >= 24)) { setError(s('i18n.matrix.languageInvalid')); return }
    const next = code || name ? [...items, { id: id!, label: name.trim(), fallback_locale: fallback, enabled: true }] : items
    if (await onSave(next)) onCancel()
  }
  return <section aria-label={s('i18n.matrix.manageLanguages')} className="min-w-0 space-y-3 rounded-xl border border-accent-200 bg-white p-4">
    <h2 className="font-semibold text-slate-900">{s('i18n.matrix.manageLanguages')}</h2><p className="text-xs leading-6 text-slate-600">{s('i18n.matrix.addHint')}</p>
    {error && <p role="alert" className="text-sm text-rose-700">{error}</p>}
    <div className="max-h-72 space-y-2 overflow-auto">{items.map((item, index) => <div key={item.id} className="grid min-w-0 gap-2 rounded-lg border border-slate-200 p-2 sm:grid-cols-[100px_1fr_140px_auto]">
      <code className="self-center break-all text-xs">{item.id}</code>
      <input aria-label={`${s('i18n.matrix.languageName')} ${item.id}`} className={input} disabled={busy || item.builtin} maxLength={60} value={item.label} onChange={event => setItems(current => current.map(row => row.id === item.id ? { ...row, label: event.target.value } : row))} />
      <select aria-label={`${s('i18n.matrix.fallbackLanguage')} ${item.id}`} className={input} disabled={busy || item.builtin} value={item.fallback_locale} onChange={event => setItems(current => current.map(row => row.id === item.id ? { ...row, fallback_locale: event.target.value } : row))}>{DEFAULT_LANGUAGES.map(language => <option key={language.id} value={language.id}>{language.label}</option>)}</select>
      <div className="flex flex-wrap items-center gap-2"><label className="flex items-center gap-1 text-xs"><input type="checkbox" disabled={busy || item.builtin} checked={item.enabled} onChange={event => setItems(current => current.map(row => row.id === item.id ? { ...row, enabled: event.target.checked } : row))} />{s('i18n.matrix.enabled')}</label><button type="button" aria-label={`${s('i18n.matrix.moveLeft')} ${item.id}`} className={button} disabled={busy || index === 0} onClick={() => move(index, -1)}>←</button><button type="button" aria-label={`${s('i18n.matrix.moveRight')} ${item.id}`} className={button} disabled={busy || index === items.length - 1} onClick={() => move(index, 1)}>→</button></div>
    </div>)}</div>
    <fieldset disabled={busy || items.length >= 24} className="grid min-w-0 gap-3 border-t border-slate-200 pt-3 sm:grid-cols-3"><label className="text-xs font-semibold">{s('i18n.matrix.languageCode')}<input ref={codeRef} aria-label={s('i18n.matrix.languageCode')} className={`${input} mt-1`} value={code} maxLength={35} placeholder="ja-JP" onChange={event => setCode(event.target.value)} /><span className="mt-1 block font-normal text-slate-600">{s('i18n.matrix.languageExample')}</span></label><label className="text-xs font-semibold">{s('i18n.matrix.languageName')}<input aria-label={s('i18n.matrix.languageName')} className={`${input} mt-1`} value={name} maxLength={60} placeholder="日本語" onChange={event => setName(event.target.value)} /></label><label className="text-xs font-semibold">{s('i18n.matrix.fallbackLanguage')}<select aria-label={s('i18n.matrix.fallbackLanguage')} className={`${input} mt-1`} value={fallback} onChange={event => setFallback(event.target.value)}>{DEFAULT_LANGUAGES.map(language => <option key={language.id} value={language.id}>{language.label}</option>)}</select></label></fieldset>
    <div className="flex gap-2"><button type="button" className={`${button} !bg-accent-600 !text-white`} disabled={busy} onClick={() => void save()}>{s('common.save')}</button><button type="button" className={button} disabled={busy} onClick={onCancel}>{s('common.cancel')}</button></div>
  </section>
}
