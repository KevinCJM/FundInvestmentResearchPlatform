import type { ActivateRequest, Capabilities, CatalogResponse, CompareResponse, ConfirmReferenceInput, ConfirmRequest, DefaultBinding, DefaultsResponse, DraftUpdate, DraftView, DraftWrite, PreviewRequest, PreviewResponse, ReferenceInputRequest, ReferencePreview, ReferenceVersion, RetireRequest, SourceCatalog, StudyOptionsResponse, VersionView } from './riskScaleContract.generated'
import { isRiskScaleAlgorithmId } from './riskScaleContract.generated'
export type * from './riskScaleContract.generated'

const base = '/api/strategic-allocation'
export const metadata = (value: unknown): Record<string, unknown> => value != null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
export const textValue = (value: unknown): string => typeof value === 'string' ? value : ''
export const numericValues = (value: unknown): Array<number | null> => Array.isArray(value) ? value.map(item => typeof item === 'number' && Number.isFinite(item) ? item : null) : []
export const registeredAlgorithms = (value: Capabilities) => value.algorithms.flatMap(item => isRiskScaleAlgorithmId(item.id) ? [{ id: item.id, available: item.available !== false, reason: textValue(item.reason), default: item.default === true }] : [])
export class RiskScaleError extends Error {
  constructor(public code: string, message: string, public field: string | null = null, public status = 0, public suggestedAction = '') { super(message); this.name = 'RiskScaleError' }
}
const safeText = (value: unknown, limit = 600) => typeof value === 'string' ? value.replace(/[\u0000-\u001f]/g, ' ').slice(0, limit) : ''
async function request<T>(path: string, signal?: AbortSignal, body?: unknown, method = 'GET'): Promise<T> {
  let response: Response
  try { response = await fetch(`${base}${path}`, { method, signal, headers: body === undefined ? undefined : { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) }) }
  catch (error) { if (signal?.aborted || (error instanceof Error && error.name === 'AbortError')) throw error; throw new RiskScaleError('NETWORK_ERROR', '') }
  let data: any
  try { data = await response.json() } catch { throw new RiskScaleError('INVALID_RESPONSE', '', null, response.status) }
  if (!response.ok) {
    const problem = data?.detail ?? data?.error ?? data
    if (Array.isArray(problem)) {
      const first = problem[0]
      throw new RiskScaleError('VALIDATION_ERROR', safeText(first?.msg), Array.isArray(first?.loc) ? first.loc.filter((part: unknown) => part !== 'body').join('.') : null, response.status)
    }
    throw new RiskScaleError(safeText(problem?.code) || 'REQUEST_FAILED', safeText(problem?.message), safeText(problem?.field ?? problem?.field_path) || null, response.status, safeText(problem?.suggested_action))
  }
  return data as T
}
const idPath = (id: string) => encodeURIComponent(id)
export const riskScales = {
  capabilities: (signal?: AbortSignal) => request<Capabilities>('/risk-scales/capabilities', signal),
  catalog: (query = '', signal?: AbortSignal) => request<CatalogResponse>(`/risk-scales${query ? `?${query}` : ''}`, signal),
  defaults: (signal?: AbortSignal) => request<DefaultsResponse>('/risk-scales/defaults', signal),
  studyOptions: (asOf: string, signal?: AbortSignal) => request<StudyOptionsResponse>(`/risk-scales/study-options?${new URLSearchParams({ as_of: asOf })}`, signal),
  draft: (id: string, signal?: AbortSignal) => request<DraftView>(`/risk-scales/drafts/${idPath(id)}`, signal),
  saveDraft: (value: DraftWrite, signal?: AbortSignal) => request<DraftView>('/risk-scales/drafts', signal, value, 'POST'),
  updateDraft: (id: string, value: DraftUpdate, signal?: AbortSignal) => request<DraftView>(`/risk-scales/drafts/${idPath(id)}`, signal, value, 'PATCH'),
  deleteDraft: (id: string, expected_revision: number, signal?: AbortSignal) => request<{ deleted: true; id: string }>(`/risk-scales/drafts/${idPath(id)}`, signal, { expected_revision }, 'DELETE'),
  preview: (value: PreviewRequest, signal?: AbortSignal) => request<PreviewResponse>('/risk-scales/preview', signal, value, 'POST'),
  confirm: (value: ConfirmRequest, signal?: AbortSignal) => request<VersionView>('/risk-scales/confirm', signal, value, 'POST'),
  version: (id: string, signal?: AbortSignal) => request<VersionView>(`/risk-scales/${idPath(id)}`, signal),
  activate: (id: string, value: ActivateRequest, signal?: AbortSignal) => request<DefaultBinding>(`/risk-scales/${idPath(id)}/activate`, signal, value, 'POST'),
  retire: (id: string, value: RetireRequest, signal?: AbortSignal) => request<DefaultBinding>(`/risk-scales/${idPath(id)}/retire`, signal, value, 'POST'),
  compare: (left_id: string, right_id: string, signal?: AbortSignal) => request<CompareResponse>('/risk-scales/compare', signal, { left_id, right_id }, 'POST'),
  sources: (kind: string, q: string, offset = 0, signal?: AbortSignal) => request<SourceCatalog>(`/reference-inputs/catalog?${new URLSearchParams({ kind, q, offset: String(offset), limit: '20' })}`, signal),
  reference: (id: string, signal?: AbortSignal) => request<ReferenceVersion>(`/reference-inputs/${idPath(id)}`, signal),
  previewReference: (value: ReferenceInputRequest, signal?: AbortSignal) => request<ReferencePreview>('/reference-inputs/preview', signal, value, 'POST'),
  confirmReference: (value: ConfirmReferenceInput, signal?: AbortSignal) => request<ReferenceVersion>('/reference-inputs/confirm', signal, value, 'POST'),
}
