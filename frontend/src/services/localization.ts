import type { LanguageDefinition, Locale, TranslationScope } from '../i18n/catalogs'

export interface TranslationChange { key: string; locale: Locale; value: string | null }
export interface TranslationState {
  revision: number; preferences_revision: number; default_locale: Locale; catalog_version: string
  overrides: Partial<Record<Locale, Record<string, string>>>; locales: Array<Pick<LanguageDefinition, 'id' | 'label'> & Partial<LanguageDefinition>>
}
export interface TranslationEntry {
  key: string; scope: TranslationScope; module: string; label: string
  translations: Partial<Record<Locale, string>>; default_value: string; override_value: string | null; effective_value: string
  customizable: boolean; missing: boolean; max_length: number; placeholders: string[]; usage: string[]
}
export interface TranslationCatalog {
  scope: TranslationScope; locale: Locale; revision: number; catalog_version: string
  items: TranslationEntry[]; total: number; page: number; page_size: number; modules: string[]
  coverage: { total: number; missing: number }
}
export interface TranslationBundle {
  locale: Locale; revision: number; default_locale: Locale; catalog_version: string
  resources: Record<TranslationScope, Record<string, string>>; fallback_keys: Record<TranslationScope, string[]>
}
export interface TranslationCell {
  value: string | null; default_value: string | null; override_value: string | null
  source: 'custom' | 'builtin' | 'missing'; fallback_value: string | null; fallback_locale: Locale | null
}
export interface TranslationMatrixRow {
  key: string; code: string; scope: TranslationScope; module: string; cells: Record<Locale, TranslationCell>
  customizable: boolean; max_length: number; placeholders: string[]; usage: string[]
}
export interface TranslationMatrix {
  scope: TranslationScope; revision: number; preferences_revision: number; catalog_version: string
  locales: LanguageDefinition[]; items: TranslationMatrixRow[]; modules: string[]
  total: number; page: number; page_size: number; coverage: Record<Locale, { total: number; missing: number }>
  sort_by: string; sort_dir: 'asc' | 'desc'
}
export interface TranslationDiff { key: string; locale: Locale; before: string | null; after: string | null }
export interface TranslationUpdate { revision: number; changes: TranslationDiff[]; overrides: TranslationState['overrides'] }
export interface TranslationPackage { format_version: 1; scope: 'business'; catalog_version?: string | null; entries: TranslationChange[] }
export interface ImportPreview { valid: boolean; revision: number; changes: TranslationDiff[]; confirmation_digest: string; catalog_changed: boolean }
export interface TranslationHistory { revision: number; items: Array<{ revision: number; at: string | null; action: string; reason: string; change_count: number }> }

export class LocalizationApiError extends Error {
  constructor(readonly status: number, readonly code: string, message: string, readonly field?: string | null) { super(message); this.name = 'LocalizationApiError' }
}
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers)
  if (init?.body) headers.set('Content-Type', 'application/json')
  const response = await fetch(`/api/i18n${path}`, { ...init, headers, cache: 'no-store' })
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    const detail = payload?.detail
    throw new LocalizationApiError(response.status, typeof detail?.code === 'string' ? detail.code : 'I18N_REQUEST_FAILED', typeof detail?.message === 'string' ? detail.message : 'Translation request failed.', detail?.field)
  }
  return response.json() as Promise<T>
}
export const getTranslationState = (signal?: AbortSignal) => request<TranslationState>('/settings', { signal })
export const getTranslationBundle = (locale: Locale, signal?: AbortSignal) => request<TranslationBundle>(`/bundle?locale=${encodeURIComponent(locale)}`, { signal })
export const getTranslationCatalog = (options: { scope: TranslationScope; locale: Locale; q?: string; module?: string; status?: string; page?: number }, signal?: AbortSignal) => {
  const params = new URLSearchParams({ scope: options.scope, locale: options.locale, q: options.q || '', module: options.module || '', status: options.status || 'all', page: String(options.page || 1), page_size: '50' })
  return request<TranslationCatalog>(`/catalog?${params}`, { signal })
}
export const getTranslationMatrix = (options: { scope: TranslationScope; q?: string; module?: string; status?: string; sortBy?: string; sortDir?: 'asc' | 'desc'; page?: number }, signal?: AbortSignal) => {
  const params = new URLSearchParams({ scope: options.scope, q: options.q || '', module: options.module || '', status: options.status || 'all', sort_by: options.sortBy || 'code', sort_dir: options.sortDir || 'asc', page: String(options.page || 1), page_size: '50' })
  return request<TranslationMatrix>(`/matrix?${params}`, { signal })
}
export const updateLanguages = (expected_revision: number, locales: LanguageDefinition[]) => request<Pick<TranslationState, 'preferences_revision' | 'default_locale' | 'locales'>>('/languages', {
  method: 'PUT', body: JSON.stringify({ expected_revision, locales: locales.map(({ id, label, fallback_locale, enabled }) => ({ id, label, fallback_locale, enabled })) }),
})
export const updateBusinessTranslations = (expected_revision: number, changes: TranslationChange[], reason = '') => request<TranslationUpdate>('/business', { method: 'PATCH', body: JSON.stringify({ scope: 'business', expected_revision, changes, reason }) })
export const updateLanguagePreferences = (expected_revision: number, default_locale: Locale) => request<Pick<TranslationState, 'preferences_revision' | 'default_locale'>>('/preferences', { method: 'PUT', body: JSON.stringify({ expected_revision, default_locale }) })
export const validateTranslationImport = (expected_revision: number, pack: TranslationPackage) => request<ImportPreview>('/business/import/validate', { method: 'POST', body: JSON.stringify({ expected_revision, package: pack }) })
export const applyTranslationImport = (expected_revision: number, pack: TranslationPackage, confirmation_digest: string) => request<TranslationUpdate>('/business/import/apply', { method: 'POST', body: JSON.stringify({ expected_revision, package: pack, confirmation_digest }) })
export const getTranslationHistory = (signal?: AbortSignal) => request<TranslationHistory>('/business/history', { signal })
export const restoreTranslations = (expected_revision: number, target_revision: number) => request<TranslationUpdate>('/business/restore', { method: 'POST', body: JSON.stringify({ expected_revision, target_revision }) })
export const exportBusinessTranslations = () => request<TranslationPackage>('/business/export')
