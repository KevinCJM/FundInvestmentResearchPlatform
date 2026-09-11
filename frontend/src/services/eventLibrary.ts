import type { ManualHistoricalEvent } from './regimeGraph'

export const eventCategories: Record<string, string> = {
  financial: '金融市场', geopolitical: '地缘与战争', political: '政治与政策', trade: '贸易与制裁',
  health: '公共卫生', social: '社会事件', disaster: '自然灾害', supply_chain: '能源与供应链', technology: '科技冲击',
}
export interface EventWindow { id: string; label: string; start_date: string; end_date: string; rationale: string }
export interface EventSource { title: string; url: string; published_at: string | null }
export interface EventDraft {
  name: string; name_en: string; description: string; categories: string[]; regions: string[]
  fact_start: string | null; fact_end: string | null; date_precision: 'day' | 'month' | 'year' | 'unknown'
  status: 'ongoing' | 'closed' | 'unknown'; verification: 'unreviewed' | 'verified'; known_at: string | null
  sources: EventSource[]; windows: EventWindow[]; color: string; archived: boolean
}
export interface LibraryEvent extends EventDraft {
  id: string; revision: number; content_hash: string; created_at: string; updated_at: string
  provenance: Record<string, unknown>
}
export interface EventSelection { event_id: string; revision: number; window_id: string }
export interface EventPage { items: LibraryEvent[]; total: number; offset: number; limit: number }
export interface EventPack { id: string; name: string; count: number }
const prefix = '/api/historical-regimes/event-library'
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(prefix + path, { ...init, headers: { 'Content-Type': 'application/json', ...init?.headers } })
  const body = await response.json()
  if (!response.ok) throw new Error(body?.detail?.message || (typeof body?.detail === 'string' ? body.detail : '事件库操作失败，请检查内容后重试。'))
  return body
}
export function listLibraryEvents(filters: Record<string, string | number | boolean>, signal?: AbortSignal) {
  const params = new URLSearchParams(Object.entries(filters).map(([key, value]) => [key, String(value)]))
  return request<EventPage>('/events?' + params, { signal })
}
export const getLibraryEvent = (id: string, revision?: number, signal?: AbortSignal) => request<LibraryEvent>('/events/' + encodeURIComponent(id) + (revision ? '?revision=' + revision : ''), { signal })
export const getLibraryEventHistory = (id: string, signal?: AbortSignal) => request<{ items: LibraryEvent[] }>('/events/' + encodeURIComponent(id) + '/history', { signal })
export const listEventPacks = (signal?: AbortSignal) => request<{ items: EventPack[] }>('/packs', { signal })
export const getEventPack = (id: string, signal?: AbortSignal) => request<{ selections: EventSelection[]; labels?: Record<string, string> }>('/packs/' + encodeURIComponent(id), { signal })
export const resolveLibraryEvents = (selections: EventSelection[], signal?: AbortSignal) => request<{ events: ManualHistoricalEvent[] }>('/resolve', { method: 'POST', body: JSON.stringify({ selections }), signal })
export const saveLibraryEvent = (event: EventDraft, current?: LibraryEvent) => request<LibraryEvent>('/events' + (current ? '/' + encodeURIComponent(current.id) : ''), { method: current ? 'PUT' : 'POST', body: JSON.stringify({ event, ...(current ? { revision: current.revision } : {}) }) })
export const importLibraryDefinition = (definitionId: string, revision: number) => request<{ imported: number; skipped: number }>('/import-definition', { method: 'POST', body: JSON.stringify({ definition_id: definitionId, revision }) })
export function eventDraft(event?: LibraryEvent): EventDraft {
  if (event) {
    const { id: _id, revision: _rev, content_hash: _hash, created_at: _created, updated_at: _updated, provenance: _origin, ...draft } = event
    return JSON.parse(JSON.stringify(draft))
  }
  return { name: '', name_en: '', description: '', categories: [], regions: [], fact_start: null, fact_end: null,
    date_precision: 'unknown', status: 'unknown', verification: 'unreviewed', known_at: null, sources: [],
    windows: [{ id: 'research', label: '主要研究窗口', start_date: '', end_date: '', rationale: '' }], color: '#7c3aed', archived: false }
}
