import { builtinCatalogs } from './catalogs'
import { systemText } from './runtime'

const exact = new Map<string, string>()
const templates: Array<{ key: string; pattern: RegExp; parameters: string[] }> = []
const escape = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
for (const [key, translations] of Object.entries(builtinCatalogs.system)) {
  if (!key.startsWith('preInvestment.')) continue
  for (const text of Object.values(translations)) {
    if (!text) continue
    const parameters = [...text.matchAll(/\{\{(\w+)\}\}/g)].map(match => match[1])
    if (!parameters.length) exact.set(text, key)
    // Only complete diagnostic templates, never a generic number/name fragment.
    else if ((key.startsWith('preInvestment.messages.') || text.replace(/\{\{\w+\}\}/g, '').length >= 12) && new Set(parameters).size === parameters.length) {
      const pattern = text.split(/\{\{\w+\}\}/g).map(escape).join('([\\s\\S]+?)')
      templates.push({ key, parameters, pattern: new RegExp(`^${pattern}$`) })
    }
  }
}

/** Compatibility for saved system diagnostics lacking message keys. Never use on researcher names, notes or inputs. */
export function researchMessage(text: string): string {
  const key = exact.get(text)
  if (key) return systemText(key)
  // ponytail: bounded built-in template scan; replace with server message keys when that contract exists.
  for (const template of templates) {
    const match = text.match(template.pattern)
    if (match) return systemText(template.key, Object.fromEntries(template.parameters.map((name, i) => [name, match[i + 1]])))
  }
  return text
}
