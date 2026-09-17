import { regimeRequest, type RegimeStudy } from './regimeGraph'

export type ProspectiveReference = NonNullable<RegimeStudy['reference']>
export interface ProspectiveProtocol {
  id: string; calibration_id: string; definition_id: string; revision: number
  model_binding_hash: string; reference: ProspectiveReference
  reference_definition: { definition_id: string; revision: number; [key: string]: unknown }
  recorded_at: string; registration_day?: string; status: 'pending'
  policy?: { observation_window?: number; [key: string]: unknown } | null
  [key: string]: unknown
}
export interface ProspectiveAssessment {
  id: string; protocol_id: string; calibration_id: string; model_binding_hash: string
  status: 'pending' | 'rejected' | 'qualified'; outcome?: 'pending' | 'qualified' | 'partially_qualified' | 'insufficient_evidence' | 'failed'; reasons?: string[] | null
  qualified_states?: string[]; fallback_states?: string[]
  state_evidence?: Array<{ state_id: string; status: 'qualified' | 'insufficient_evidence' | 'failed'; paired_reference_observations: number; accepted_predictions: number; complete_regimes: number; precision: number | null; recall: number | null; reasons: string[] }>
  available_from?: string | null; available_from_date?: string | null; expires_at?: string | null
  reference?: ProspectiveReference | null
  metrics?: { coverage?: number | null; classification?: { accuracy?: number | null } | null; [key: string]: unknown } | null
  [key: string]: unknown
}
export type IndexSnapshotBindings = Record<string, { snapshot_id: string; snapshot_generation: string; source_file: string; file_checksum: string }>
export interface ProspectiveSourcePreview {
  preview_hash: string; protocol_id: string; previous_id: string; as_of: string
  model_bindings: IndexSnapshotBindings; model_binding_hash: string
  old_observations: number; new_observations: number; added_observations: number; prefix_unchanged: boolean
  reference_id: string; reference_revision: number
}
export interface ProspectiveSourceVersion extends ProspectiveSourcePreview {
  id: string; kind: 'source_version'; status: 'accepted'; recorded_at: string
  reference_definition: ProspectiveProtocol['reference_definition']
}
export interface ProspectiveProgress {
  protocol: ProspectiveProtocol; observations: number | null; last_observation_date: string | null
  latest_assessment: ProspectiveAssessment | null
  current_source_version?: ProspectiveSourceVersion | null
  reference_definitions?: ProspectiveProtocol['reference_definition'][]
}
export interface ProspectiveCapture {
  status: string; reason?: string | null
  observation?: { observation_date?: string | null; [key: string]: unknown } | null
  [key: string]: unknown
}
const root = '/api/historical-regimes/prospective'
// Only the published write contract is accepted, including callers outside TypeScript.
function exact(value: unknown, keys: string[]) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).some(key => !keys.includes(key))) throw new Error('前瞻请求字段不合法；时间、概率和资格由服务端确定。')
}
export function registerRegimeProspective(body: { calibration_id: string }, signal?: AbortSignal) {
  exact(body, ['calibration_id'])
  return regimeRequest<ProspectiveProtocol>(`${root}/register`, { method: 'POST', body: JSON.stringify({ calibration_id: body.calibration_id }), signal })
}
export function captureRegimeProspective(id: string, body: Record<string, never> = {}, signal?: AbortSignal) {
  exact(body, [])
  return regimeRequest<ProspectiveCapture>(`${root}/${encodeURIComponent(id)}/capture`, { method: 'POST', body: '{}', signal })
}
export function assessRegimeProspective(id: string, body: { reference: ProspectiveReference }, signal?: AbortSignal) {
  exact(body, ['reference']); exact(body.reference, ['run_id', 'publication_id', 'content_hash'])
  return regimeRequest<ProspectiveAssessment>(`${root}/${encodeURIComponent(id)}/assess`, { method: 'POST', body: JSON.stringify({ reference: body.reference }), signal })
}
export async function listRegimeProspective(signal?: AbortSignal) {
  const result = await regimeRequest<{ items?: ProspectiveProtocol[] | null } | null>(`${root}/catalog`, { signal })
  return result?.items || []
}
export function getRegimeProspectiveProgress(id: string, signal?: AbortSignal) {
  return regimeRequest<ProspectiveProgress>(`${root}/protocols/${encodeURIComponent(id)}/progress`, { signal })
}
export function previewRegimeSourceVersion(id: string, signal?: AbortSignal) {
  return regimeRequest<ProspectiveSourcePreview>(`${root}/${encodeURIComponent(id)}/sources/preview`, { method: 'POST', body: '{}', signal })
}
export function confirmRegimeSourceVersion(id: string, previewHash: string, signal?: AbortSignal) {
  return regimeRequest<ProspectiveSourceVersion>(`${root}/${encodeURIComponent(id)}/sources/confirm`, { method: 'POST', body: JSON.stringify({ preview_hash: previewHash }), signal })
}
export function getRegimeProspectiveQualification(id: string, signal?: AbortSignal) {
  return regimeRequest<ProspectiveAssessment>(`${root}/qualifications/${encodeURIComponent(id)}`, { signal })
}
