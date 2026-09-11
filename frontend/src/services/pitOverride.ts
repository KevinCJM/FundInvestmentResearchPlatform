import type { RunMode } from './pit'

/**
 * The口径 this browser tab asked to *view* under, layered over the system one.
 *
 * The system-level PIT setting is the default, not a cage: a reader has to be
 * able to look at another vintage — or at the raw disk — without changing what
 * anyone else sees. So this one lives per tab (`sessionStorage`, gone when the
 * tab closes) precisely where the system setting must not, and it travels on
 * request headers so every endpoint that reads data honours it without growing
 * three parameters each.
 */
export interface PitViewOverride {
  /** Look at every row on disk, ignoring the applied release. */
  off: boolean
  /** Release to view instead of the applied one; ignored when `off`. */
  releaseId: string | null
  /** Day to stand on for this tab only. Valid on its own, with no release. */
  asOf?: string | null
  /** Null inherits the viewed version's mode, then the system mode. */
  runMode: RunMode | null
}

const STORAGE_KEY = 'pit.view.override'

function read(): PitViewOverride | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as PitViewOverride
    return normalize(parsed)
  } catch {
    // A private window or blocked site data just means "follow the system".
    return null
  }
}

function normalize(value: PitViewOverride | null): PitViewOverride | null {
  if (!value) return null
  if (value.off) return { off: true, releaseId: null, asOf: null, runMode: 'RESEARCH' }
  // A research day alone is a complete override: standing on another day is the
  // commonest thing a reader wants, and it needs no sealed vintage to be valid.
  if (!value.releaseId && !value.asOf) return null
  return {
    off: false,
    releaseId: value.releaseId ?? null,
    asOf: value.asOf ?? null,
    // Preserved as-is: a version-only override leaves this null so the version
    // answers, while a bare day inherits the system mode server-side.
    runMode: value.runMode === 'STRICT_PIT' ? 'STRICT_PIT' : value.runMode === 'RESEARCH' ? 'RESEARCH' : null,
  }
}

let current: PitViewOverride | null = read()

export function getPitOverride(): PitViewOverride | null {
  return current
}

export function setPitOverride(next: PitViewOverride | null): PitViewOverride | null {
  current = normalize(next)
  try {
    if (current) sessionStorage.setItem(STORAGE_KEY, JSON.stringify(current))
    else sessionStorage.removeItem(STORAGE_KEY)
  } catch {
    // Not persisting is survivable; the in-memory value still applies.
  }
  return current
}

/**
 * Applying a new viewing口径 reloads the page: every number on screen was
 * computed under the old one, and two vintages on one screen is the exact
 * confusion this feature exists to prevent. Held on an object because jsdom
 * refuses to navigate, so a test replaces `run` rather than the location.
 */
export const pageReload = { run: () => window.location.reload() }

export function pitOverrideHeaders(): Record<string, string> {
  if (!current) return {}
  if (current.off) return { 'X-Pit-Off': '1' }
  const headers: Record<string, string> = {}
  if (current.runMode) headers['X-Pit-Run-Mode'] = current.runMode
  if (current.releaseId) headers['X-Pit-Release'] = current.releaseId
  if (current.asOf) headers['X-Pit-As-Of'] = current.asOf
  return headers
}

function isOverridableApi(url: string): boolean {
  let path = url
  try {
    path = new URL(url, window.location.origin).pathname
  } catch {
    // Keep the raw string; the prefix test below is all that matters.
  }
  // `/api/pit/*` is deliberately exempt: it is the surface that reads and clears
  // the override itself, and a stale header must never be able to lock a tab out
  // of the control that would fix it.
  return path.startsWith('/api/') && !path.startsWith('/api/pit/')
}

/**
 * Inject the viewing口径 into every data request, once, at the transport layer.
 *
 * The alternative — threading three fields through every service call — is a
 * larger diff that is wrong the first time someone adds an endpoint and forgets.
 */
export function installPitOverrideFetch(): void {
  const native = window.fetch.bind(window)
  window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const headers = pitOverrideHeaders()
    const url = typeof input === 'string' ? input : input instanceof Request ? input.url : String(input)
    if (!Object.keys(headers).length || !isOverridableApi(url)) return native(input, init)
    const merged = new Headers(init?.headers ?? (input instanceof Request ? input.headers : undefined))
    for (const [key, value] of Object.entries(headers)) merged.set(key, value)
    return native(input, { ...init, headers: merged })
  }
}
