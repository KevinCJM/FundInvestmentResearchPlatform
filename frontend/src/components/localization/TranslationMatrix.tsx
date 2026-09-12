import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { useI18n } from '../../i18n/runtime'
import type { TranslationChange, TranslationMatrix as Matrix, TranslationMatrixRow } from '../../services/localization'
import { pastedChanges, translationTsv, validateTranslationCell } from './matrixEditing'

type Position = { row: number; column: number }
interface Props {
  matrix: Matrix; module: string; pending: Record<string, TranslationChange>; disabled: boolean
  sortBy: string; sortDir: 'asc' | 'desc'; onSort: (field: string) => void
  onChanges: (changes: TranslationChange[]) => void; onError: (message: string) => void; onAddLanguage: () => void
  onEditingChange?: (editing: boolean) => void
}
const button = 'min-h-9 rounded-xl border border-slate-300 bg-white px-3 py-1.5 text-sm font-semibold text-slate-700 disabled:opacity-40'

export default function TranslationMatrix({ matrix, module, pending, disabled, sortBy, sortDir, onSort, onChanges, onError, onAddLanguage, onEditingChange }: Props) {
  const { s } = useI18n()
  const [active, setActive] = useState<Position>({ row: 0, column: 1 })
  const [anchor, setAnchor] = useState<Position | null>(null)
  const [editing, setEditing] = useState<{ position: Position; value: string } | null>(null)
  const tableRef = useRef<HTMLTableElement>(null)
  const editRef = useRef<HTMLTextAreaElement>(null)
  const editSession = useRef(false)
  const rowKeys = matrix.items.map(row => row.key).join('|')
  const languageKeys = matrix.locales.map(language => language.id).join('|')
  useEffect(() => { setActive({ row: 0, column: 1 }); setAnchor(null); setEditing(null); editSession.current = false }, [rowKeys, languageKeys])
  useEffect(() => { if (editing) { editRef.current?.focus(); editRef.current?.select() } }, [editing?.position.row, editing?.position.column])
  useEffect(() => { onEditingChange?.(editing !== null) }, [editing !== null, onEditingChange])
  useEffect(() => () => { onEditingChange?.(false) }, [onEditingChange])
  const cellValue = (row: TranslationMatrixRow, locale: string) => {
    const change = pending[`${locale}:${row.key}`]
    return change ? change.value ?? row.cells[locale].default_value : row.cells[locale].value
  }
  const focus = (position: Position, extend = false) => {
    if (editSession.current && !commit()) return
    const next = { row: Math.max(0, Math.min(matrix.items.length - 1, position.row)), column: Math.max(0, Math.min(matrix.locales.length, position.column)) }
    setActive(next)
    setAnchor(extend ? anchor ?? active : null)
    tableRef.current?.querySelector<HTMLElement>(`[data-matrix-row="${next.row}"][data-matrix-column="${next.column}"]`)?.focus()
  }
  const beginEdit = (position: Position = active) => {
    const row = matrix.items[position.row], language = matrix.locales[position.column - 1]
    if (disabled || !row?.customizable || !language || (editing && !commit())) return
    setActive(position); setAnchor(null); editSession.current = true
    setEditing({ position, value: cellValue(row, language.id) ?? '' })
  }
  const commit = (next?: Position): boolean => {
    if (!editing || !editSession.current) return true
    const row = matrix.items[editing.position.row], locale = matrix.locales[editing.position.column - 1].id
    try {
      validateTranslationCell(row, editing.value)
      onChanges([{ key: row.key, locale, value: editing.value === '' || editing.value === row.cells[locale].default_value ? null : editing.value }])
      editSession.current = false; setEditing(null)
      if (next) focus(next)
      return true
    } catch { onError(s('i18n.matrix.invalidCell')); editRef.current?.focus(); return false }
  }
  const navigate = (event: KeyboardEvent, position: Position) => {
    if (disabled) return
    const { row, column } = position
    if (event.key === 'Enter' || event.key === 'F2') { event.preventDefault(); beginEdit(position); return }
    const target = event.key === 'ArrowUp' ? { row: row - 1, column } : event.key === 'ArrowDown' ? { row: row + 1, column }
      : event.key === 'ArrowLeft' ? { row, column: column - 1 } : event.key === 'ArrowRight' ? { row, column: column + 1 }
        : event.key === 'Home' ? { row: event.ctrlKey || event.metaKey ? 0 : row, column: 0 }
          : event.key === 'End' ? { row: event.ctrlKey || event.metaKey ? matrix.items.length - 1 : row, column: matrix.locales.length } : null
    if (target) { event.preventDefault(); focus(target, event.shiftKey) }
    else if (event.key === 'Tab') {
      const index = row * (matrix.locales.length + 1) + column + (event.shiftKey ? -1 : 1)
      if (index >= 0 && index < matrix.items.length * (matrix.locales.length + 1)) {
        event.preventDefault(); focus({ row: Math.floor(index / (matrix.locales.length + 1)), column: index % (matrix.locales.length + 1) })
      }
    }
  }
  const selected = (row: number, column: number) => row >= Math.min(active.row, anchor?.row ?? active.row) && row <= Math.max(active.row, anchor?.row ?? active.row) && column >= Math.min(active.column, anchor?.column ?? active.column) && column <= Math.max(active.column, anchor?.column ?? active.column)
  const ariaSort = (field: string): 'ascending' | 'descending' | 'none' => sortBy === field ? (sortDir === 'asc' ? 'ascending' : 'descending') : 'none'
  const sortMark = (field: string) => sortBy === field ? (sortDir === 'asc' ? '↑' : '↓') : '↕'
  const sortLabel = (field: string, label: string) => s(sortBy === field && sortDir === 'asc' ? 'i18n.matrix.sortDescending' : 'i18n.matrix.sortAscending', { column: label })
  const row = matrix.items[active.row]
  const language = matrix.locales[active.column - 1]
  const cell = row && language ? row.cells[language.id] : null
  return <>
    <p className="my-3 text-xs leading-6 text-slate-600">{s(matrix.scope === 'system' ? 'i18n.matrix.readOnlyHint' : 'i18n.matrix.hint')}</p>
    <div className="max-h-[65vh] max-w-full overflow-auto rounded-xl border border-slate-300 [--i18n-key-width:96px] sm:[--i18n-key-width:220px]" data-testid="translation-matrix-scroll">
      <table ref={tableRef} role="grid" aria-label={s(`i18n.${matrix.scope}`)} aria-readonly={matrix.scope === 'system'} aria-rowcount={matrix.items.length + 1} aria-colcount={matrix.locales.length + 2} className="w-full table-fixed border-separate border-spacing-0 text-left text-sm" style={{ minWidth: `calc(var(--i18n-key-width) + ${matrix.locales.length * 220 + 150}px)` }}
        onCopy={event => {
          if ((event.target as HTMLElement).closest('textarea')) return
          const rows: string[][] = []
          for (let r = Math.min(active.row, anchor?.row ?? active.row); r <= Math.max(active.row, anchor?.row ?? active.row); r++) {
            if (!matrix.items[r]) continue
            const values: string[] = []
            for (let c = Math.min(active.column, anchor?.column ?? active.column); c <= Math.max(active.column, anchor?.column ?? active.column); c++) values.push(c === 0 ? matrix.items[r].key : cellValue(matrix.items[r], matrix.locales[c - 1].id) ?? '')
            rows.push(values)
          }
          event.clipboardData.setData('text/plain', translationTsv(rows)); event.preventDefault()
        }}
        onPaste={event => {
          if ((event.target as HTMLElement).closest('textarea')) return
          event.preventDefault()
          if (disabled || matrix.scope !== 'business' || (editing && !commit())) return
          try { onChanges(pastedChanges(matrix.items, matrix.locales.map(item => item.id), active.row, active.column - 1, event.clipboardData.getData('text/plain'))); setAnchor(null) }
          catch (failure) { onError(s(failure instanceof Error && failure.message === 'invalidCell' ? 'i18n.matrix.invalidCell' : 'i18n.matrix.pasteInvalid')) }
        }}>
        <colgroup><col style={{ width: 'var(--i18n-key-width)' }} />{matrix.locales.map(item => <col key={item.id} style={{ width: 220 }} />)}<col style={{ width: 150 }} /></colgroup>
        <thead><tr role="row"><th scope="col" role="columnheader" aria-sort={ariaSort('code')} className="sticky left-0 top-0 z-30 break-words border-b border-r border-slate-300 bg-slate-100 p-2 sm:p-3"><button type="button" className="flex min-h-9 w-full items-center justify-between gap-2 text-left font-semibold text-slate-800 disabled:cursor-not-allowed" disabled={disabled || Boolean(editing)} aria-label={sortLabel('code', s('i18n.matrix.code'))} onClick={() => onSort('code')}><span>{s('i18n.matrix.code')}</span><span aria-hidden="true" className="text-slate-600">{sortMark('code')}</span></button></th>
          {matrix.locales.map(item => <th scope="col" key={item.id} role="columnheader" aria-sort={ariaSort(item.id)} lang={item.id} className="sticky top-0 z-20 min-w-[220px] border-b border-r border-slate-300 bg-slate-50 p-3"><button type="button" className="flex min-h-9 w-full items-start justify-between gap-2 text-left disabled:cursor-not-allowed" disabled={disabled || Boolean(editing)} aria-label={sortLabel(item.id, item.label)} onClick={() => onSort(item.id)}><span><span className="block font-semibold text-slate-800">{item.label}</span><code className="text-xs font-normal text-slate-600">{item.id}</code></span><span aria-hidden="true" className="pt-0.5 text-slate-600">{sortMark(item.id)}</span></button>{!item.enabled && <span className="ml-2 text-xs text-amber-700">{s('i18n.matrix.disabled')}</span>}{!item.system_pack && <p className="mt-1 text-xs font-normal text-amber-700">{s('i18n.matrix.noSystemPack')}</p>}</th>)}
          <th scope="col" className="sticky top-0 z-20 min-w-[150px] border-b border-slate-300 bg-white p-3"><button type="button" className={button} disabled={disabled} onClick={() => { if (commit()) onAddLanguage() }}>{s('i18n.matrix.addLanguage')}</button></th>
        </tr></thead>
        <tbody>{matrix.items.map((entry, r) => <tr key={entry.key} role="row" data-entry-key={entry.key}>
          <th scope="row" role="rowheader" aria-readonly="true" aria-selected={selected(r, 0)} title={entry.key} tabIndex={active.row === r && active.column === 0 ? 0 : -1} data-matrix-row={r} data-matrix-column={0} onClick={event => focus({ row: r, column: 0 }, event.shiftKey)} onKeyDown={event => navigate(event, { row: r, column: 0 })} className={`sticky left-0 z-10 border-b border-r border-slate-300 p-2 sm:p-3 align-top font-normal outline-none focus:ring-2 focus:ring-inset focus:ring-accent-500 ${selected(r, 0) ? 'bg-accent-50' : 'bg-slate-50'}`}><code className="block break-all text-xs font-semibold text-slate-800">{module === entry.module ? entry.code : entry.key}</code></th>
          {matrix.locales.map((item, index) => {
            const c = index + 1, value = cellValue(entry, item.id)
            const changed = Object.prototype.hasOwnProperty.call(pending, `${item.id}:${entry.key}`)
            const own = entry.cells[item.id]
            const source = changed ? pending[`${item.id}:${entry.key}`].value === null ? own.default_value === null ? 'missing' : 'builtin' : 'custom' : own.source
            const isEditing = editing?.position.row === r && editing.position.column === c
            return <td key={item.id} role="gridcell" aria-label={`${entry.key} ${item.id}`} aria-readonly={!entry.customizable} aria-selected={selected(r, c)} lang={item.id} dir="auto" tabIndex={active.row === r && active.column === c ? 0 : -1} data-matrix-row={r} data-matrix-column={c} data-cell-state={changed ? 'modified' : source} onClick={event => { if (!isEditing) focus({ row: r, column: c }, event.shiftKey) }} onDoubleClick={() => beginEdit({ row: r, column: c })} onKeyDown={event => { if (!isEditing) navigate(event, { row: r, column: c }) }} className={`min-w-[220px] max-w-[340px] border-b border-r border-slate-200 p-3 align-top outline-none focus:ring-2 focus:ring-inset focus:ring-accent-500 ${selected(r, c) ? 'bg-accent-50' : changed ? 'bg-amber-50' : 'bg-white'}`}>
              {isEditing ? <textarea ref={editRef} aria-label={`${s('i18n.matrix.edit')} ${entry.key} ${item.id}`} value={editing.value} rows={entry.key.endsWith('.description') ? 4 : 2} className="min-h-12 w-full resize-y rounded-lg border border-accent-400 bg-white p-2 text-sm outline-none" onChange={event => setEditing({ ...editing, value: event.target.value })} onBlur={() => commit()} onKeyDown={event => {
                event.stopPropagation()
                if (event.nativeEvent.isComposing) return
                if (event.key === 'Escape') { event.preventDefault(); editSession.current = false; setEditing(null); focus({ row: r, column: c }) }
                else if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); commit({ row: r + 1, column: c }) }
                else if (event.key === 'Tab') { event.preventDefault(); const next = c + (event.shiftKey ? -1 : 1); commit(next < 1 ? { row: r - 1, column: matrix.locales.length } : next > matrix.locales.length ? { row: r + 1, column: 1 } : { row: r, column: next }) }
              }} /> : <span className={`block whitespace-pre-wrap break-words ${value === null ? 'text-slate-600' : 'text-slate-800'}`}>{value ?? s('i18n.matrix.empty')}</span>}
              <span className={`mt-2 block text-xs ${changed ? 'text-amber-800' : 'text-slate-600'}`}>{changed ? s('i18n.matrix.modified') : source === 'custom' ? s('common.customized') : source === 'builtin' ? s('common.builtIn') : ''}</span>
              {value === null && own.fallback_value !== null && <p className="mt-1 text-xs leading-5 text-slate-600">{s('i18n.matrix.fallback', { locale: own.fallback_locale ?? '', value: own.fallback_value })}</p>}
            </td>
          })}<td className="border-b border-slate-200" />
        </tr>)}</tbody>
      </table>
    </div>
    {row && <div role="region" className="mt-3 min-w-0 rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs leading-6" aria-label={s('i18n.matrix.selection')}><p><strong>{s('i18n.matrix.selection')}：</strong><code className="break-all">{row.key}{language ? ` · ${language.id}` : ''}</code></p>{cell && <p className="whitespace-pre-wrap break-words">{s('i18n.default')}：{cell.default_value ?? s('i18n.matrix.empty')}</p>}{row.placeholders.length > 0 && <p>{s('i18n.placeholderHint', { names: row.placeholders.join(', ') })}</p>}{row.customizable && language && <div className="mt-2 flex flex-wrap gap-2"><button type="button" className={button} disabled={disabled} onClick={() => beginEdit()}>{s('i18n.matrix.edit')}</button><button type="button" className={button} disabled={disabled || Boolean(editing) || !(pending[`${language.id}:${row.key}`]?.value ?? cell?.override_value)} onClick={() => onChanges([{ key: row.key, locale: language.id, value: null }])}>{s('common.reset')}</button><span className="self-center text-slate-600">{s('i18n.matrix.editHint')}</span></div>}</div>}
  </>
}
