import type { TranslationChange, TranslationMatrixRow } from '../../services/localization'

/** Excel-compatible quoted TSV. Reject malformed/oversized rectangles before editing. */
export function parseTranslationTsv(text: string): string[][] {
  if (!text || text.length > 1024 * 1024) throw new Error('pasteInvalid')
  const rows: string[][] = []
  let row: string[] = [], cell = '', quoted = false, closed = false
  const endCell = () => { row.push(cell); cell = ''; closed = false }
  const endRow = () => { endCell(); rows.push(row); row = [] }
  for (let index = 0; index < text.length; index++) {
    const char = text[index]
    if (quoted) {
      if (char === '"') {
        if (text[index + 1] === '"') { cell += '"'; index++ }
        else { quoted = false; closed = true }
      } else cell += char
    } else if (char === '\t') endCell()
    else if (char === '\r' || char === '\n') { if (char === '\r' && text[index + 1] === '\n') index++; endRow() }
    else if (closed) throw new Error('pasteInvalid')
    else if (char === '"' && cell === '') quoted = true
    else cell += char
    if (rows.length > 200 || row.length > 24) throw new Error('pasteInvalid')
  }
  if (quoted) throw new Error('pasteInvalid')
  if (cell || row.length || !/[\r\n]$/.test(text)) endRow()
  if (!rows.length || rows.some(item => item.length !== rows[0].length) || rows.length * rows[0].length > 1000) throw new Error('pasteInvalid')
  return rows
}

export function translationTsv(rows: string[][]): string {
  return rows.map(row => row.map(cell => /[\t\r\n"]/.test(cell) ? `"${cell.replace(/"/g, '""')}"` : cell).join('\t')).join('\r\n')
}

export function validateTranslationCell(row: TranslationMatrixRow, value: string): void {
  if (!value) return // Clearing means delete the override, never blank the system default.
  if (!value.trim() || value.length > row.max_length || /<\s*\/?\s*[A-Za-z!][^>]*>|\$t\(/.test(value) || /[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(value)) throw new Error('invalidCell')
  const placeholders = [...new Set([...value.matchAll(/\{\{\s*([A-Za-z][A-Za-z0-9_]*)\s*\}\}/g)].map(match => match[1]))].sort()
  const remainder = value.replace(/\{\{\s*([A-Za-z][A-Za-z0-9_]*)\s*\}\}/g, '')
  if (JSON.stringify(placeholders) !== JSON.stringify([...row.placeholders].sort()) || remainder.includes('{{') || remainder.includes('}}')) throw new Error('invalidCell')
}

export function pastedChanges(rows: TranslationMatrixRow[], locales: string[], startRow: number, startColumn: number, text: string): TranslationChange[] {
  const rectangle = parseTranslationTsv(text)
  if (startRow < 0 || startColumn < 0 || startRow + rectangle.length > rows.length || startColumn + rectangle[0].length > locales.length) throw new Error('pasteInvalid')
  return rectangle.flatMap((values, rowOffset) => {
    const row = rows[startRow + rowOffset]
    if (!row.customizable) throw new Error('pasteInvalid')
    return values.map((value, columnOffset) => {
      validateTranslationCell(row, value)
      const locale = locales[startColumn + columnOffset]
      return { key: row.key, locale, value: value === '' || value === row.cells[locale].default_value ? null : value }
    })
  })
}
