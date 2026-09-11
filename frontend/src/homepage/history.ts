import { homeModules } from './catalog'

export const RECENT_VISITS_KEY = 'fund-research.home.recent.v1'
const LIMIT = 8

export function readRecentVisits(): string[] {
  try {
    const raw = localStorage.getItem(RECENT_VISITS_KEY)
    if (!raw || raw.length > 4096) return []
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return [...new Set(parsed.filter((id): id is string => typeof id === 'string' && homeModules.some(item => item.id === id)))].slice(0, LIMIT)
  } catch { return [] }
}

/** Store an allowlisted module ID only: never product IDs, query strings, or user input. */
export function recordRecentVisit(pathname: string): void {
  const module = homeModules.filter(item => pathname === item.path || pathname.startsWith(`${item.path}/`))
    .sort((a, b) => b.path.length - a.path.length)[0]
  if (!module) return
  try { localStorage.setItem(RECENT_VISITS_KEY, JSON.stringify([module.id, ...readRecentVisits().filter(id => id !== module.id)].slice(0, LIMIT))) } catch { /* Navigation also works with storage disabled. */ }
}

export function clearRecentVisits(): boolean {
  try { localStorage.removeItem(RECENT_VISITS_KEY); return true } catch { return false }
}
