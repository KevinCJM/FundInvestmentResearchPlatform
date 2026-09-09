export interface StoragePlan {
  id: string; target: string; mount: string; phase: string; message: string
  files: number; bytes: number; inventory?: { files: number; bytes: number }
}
export interface StorageStatus {
  revision?: number; logical_path: string; actual_path?: string; online: boolean; error?: string | null
  editing_enabled: boolean; free_bytes: number | null; total_bytes: number | null
  active?: { id: string; target: string; backup: string; backup_removed: boolean } | null
  pending?: StoragePlan | null
  volumes: { name: string; path: string; free_bytes: number }[]
}
export interface StorageProbe {
  target: string; mount: string; free_bytes: number; total_bytes: number
  reserve_bytes: number; same_device: boolean; message: string
}
async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api/data-storage${path}`, { ...init, cache: 'no-store', headers: { 'Content-Type': 'application/json' } })
  const value = await response.json().catch(() => null)
  if (!response.ok) throw new Error(value?.detail?.message || `存储操作失败（HTTP ${response.status}）`)
  return value as T
}
export const getDataStorage = (signal?: AbortSignal) => request<StorageStatus>('', { signal })
export const probeDataStorage = (path: string) => request<StorageProbe>('/probe', { method: 'POST', body: JSON.stringify({ path }) })
export const planDataStorage = (path: string, expected_revision: number) => request<StorageStatus>('/plan', { method: 'PUT', body: JSON.stringify({ path, expected_revision, confirm: true }) })
export const cancelStoragePlan = (expected_revision: number) => request<StorageStatus>('/plan', { method: 'DELETE', body: JSON.stringify({ expected_revision }) })
