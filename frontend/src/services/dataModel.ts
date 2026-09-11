export type DataModelLayer = 'control' | 'master' | 'canonical' | 'mart'
export type DataModelStorageEngine = 'sqlite' | 'parquet'
export type DataModelDeliveryPhase = 'core' | 'next'
export type DataModelScope = 'external' | 'internal'
export type DataModelUsage = 'external_import' | 'system_internal'

export interface DataModelField {
  name: string
  label: string
  data_type: string
  nullable: boolean
  role: string
  description: string
  unit: string | null
  enum_values: string[]
  reference: string | null
  source_mappable: boolean
}

export interface DataModelTable {
  table_id: string
  usage: DataModelUsage
  category_id: string
  label: string
  description: string
  layer: DataModelLayer
  storage_engine: DataModelStorageEngine
  storage_location: string
  delivery_phase: DataModelDeliveryPhase
  grain: string
  primary_key: string[]
  update_strategy: string
  fields: DataModelField[]
  source_mappable: boolean
  partition_by: string[]
  sort_by: string[]
  pit_supported: boolean
}

export interface DataModelCategory {
  category_id: string
  label: string
  description: string
  order: number
  table_count: number
  field_count: number
}

export interface DataModelTypeConvention {
  logical_type: string
  physical_type: string
  rule: string
}

export interface DataModelSummary {
  category_count: number
  table_count: number
  field_count: number
  mapping_target_table_count: number
  mapping_target_field_count: number
  pit_table_count: number
  by_layer: Partial<Record<DataModelLayer, number>>
  by_storage_engine: Partial<Record<DataModelStorageEngine, number>>
  by_delivery_phase: Partial<Record<DataModelDeliveryPhase, number>>
}

export interface DataModelCatalog {
  model_id: string
  schema_version: string
  status: string
  scope: DataModelScope | 'all'
  description: string
  principles: string[]
  type_conventions: DataModelTypeConvention[]
  categories: DataModelCategory[]
  tables: DataModelTable[]
  summary: DataModelSummary
}

const errorMessage = async (response: Response) => {
  try {
    const payload = await response.json() as { detail?: string | { message?: string } }
    if (typeof payload.detail === 'string') return payload.detail
    if (payload.detail?.message) return payload.detail.message
  } catch {
    // Keep the stable fallback below when the response is not JSON.
  }
  return `无法读取系统数据模型（HTTP ${response.status}）`
}

export async function fetchDataModelCatalog(
  signal?: AbortSignal,
  scope: DataModelScope = 'external',
): Promise<DataModelCatalog> {
  const url = scope === 'external' ? '/api/data-model/catalog' : '/api/data-model/catalog?scope=internal'
  const response = await fetch(url, {
    cache: 'no-store',
    signal,
  })
  if (!response.ok) throw new Error(await errorMessage(response))
  return response.json() as Promise<DataModelCatalog>
}
