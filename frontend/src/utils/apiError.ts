/** Turn any backend error body into text a human can act on.
 *
 * FastAPI answers a schema violation with `detail` as a list of objects, not a
 * string; interpolating that straight into an Error yields "[object Object]"
 * and the operator learns nothing. Every fetch that reads `detail` goes through
 * here.
 */
export function apiErrorMessage(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null | undefined)?.detail
  const text = describe(detail)
  return text || fallback
}

function describe(detail: unknown): string {
  if (detail == null) return ''
  if (typeof detail === 'string') return detail.trim()
  if (Array.isArray(detail)) {
    return detail.map(describe).filter(Boolean).join('；')
  }
  if (typeof detail === 'object') {
    const item = detail as { loc?: unknown[]; msg?: unknown; message?: unknown }
    const message = item.msg ?? item.message
    if (typeof message === 'string') {
      // Pydantic's loc names the offending field; without it "Input should be
      // 'a' or 'b'" does not say which input.
      const where = Array.isArray(item.loc)
        ? item.loc.filter((part) => part !== 'body').join('.')
        : ''
      return where ? `${where}: ${message}` : message
    }
    try {
      return JSON.stringify(detail)
    } catch {
      return ''
    }
  }
  return String(detail)
}
