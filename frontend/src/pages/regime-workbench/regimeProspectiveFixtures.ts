// Offline tests only; product code must never import this module.
import type { ProspectiveAssessment, ProspectiveProgress, ProspectiveProtocol } from '../../services/regimeProspective'
import { reliabilityPreviewFixture, reliabilityReference } from './regimeReliabilityFixtures'
export const prospectiveSaved = { ...reliabilityPreviewFixture(), id: 'cal-1', calibration_id: 'cal-1', created_at: '2026-09-15', immutable: true as const, content_hash: 'saved-hash' }
export const prospectiveProtocol: ProspectiveProtocol = {
  id: 'protocol-1', calibration_id: 'cal-1', definition_id: 'recognition', revision: 2,
  model_binding_hash: 'binding-1', reference: prospectiveSaved.request.reference,
  reference_definition: { definition_id: 'history', revision: 3 }, recorded_at: '2026-09-15T00:00:00Z',
  status: 'pending', policy: { observation_window: 252 },
}
export const prospectiveReference = { ...reliabilityReference, run_id: 'later-run', publication_id: 'later-publication', content_hash: 'later-hash', created_at: '2027-09-15T00:00:00Z', name: '后续离线参考' }
export const prospectiveAssessment: ProspectiveAssessment = {
  id: 'qualification-1', protocol_id: prospectiveProtocol.id, calibration_id: 'cal-1', model_binding_hash: 'binding-1', status: 'qualified', reasons: [],
  reference: prospectiveReference, available_from_date: '2027-09-16', expires_at: '2099-12-31T00:00:00Z',
  metrics: { coverage: 0.99, classification: { accuracy: 0.8 } },
}
export const prospectiveProgress: ProspectiveProgress = { protocol: prospectiveProtocol, observations: 0, last_observation_date: null, latest_assessment: null }
