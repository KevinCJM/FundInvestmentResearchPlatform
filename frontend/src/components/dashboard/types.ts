export type DashboardKind = 'all' | 'etf' | 'fund';
export type SegmentKind = Exclude<DashboardKind, 'all'>;
export type DashboardStatus = 'complete' | 'partial' | 'unavailable';

export interface DistributionPoint {
  name: string;
  value: number;
}

export interface TrendPoint {
  year: number;
  count: number;
  total_issue_amount?: number | null;
}

export interface DashboardTrendSeries {
  date_field: 'list_date' | 'found_date' | string;
  label: string;
  points: TrendPoint[];
}

export interface DashboardLatestProduct {
  ts_code: string;
  name?: string | null;
  management?: string | null;
  fund_type?: string | null;
  invest_type?: string | null;
  market?: string | null;
  index_code?: string | null;
  index_name?: string | null;
  list_date?: string | null;
  found_date?: string | null;
  due_date?: string | null;
  delist_date?: string | null;
  min_amount?: number | string | null;
  purc_startdate?: string | null;
  redm_startdate?: string | null;
}

export interface DashboardSummary {
  share_code_count: number;
  active_count: number;
  issuing_count: number;
  inactive_count: number;
  unknown_status_count: number;
  unique_managements: number;
  nav_covered_count: number;
  nav_coverage_rate?: number | null;
  issue_amount_total?: number | null;
  issue_amount_coverage_rate?: number | null;
  latest_nav_date?: string | null;
  latest_candle_date?: string | null;
  index_covered_count?: number | null;
  index_coverage_rate?: number | null;
  liquidity_covered_count?: number | null;
  liquidity_coverage_rate?: number | null;
  purchase_redemption_covered_count?: number | null;
  purchase_redemption_coverage_rate?: number | null;
}

export interface DashboardSegment {
  availability: 'ready' | 'missing';
  snapshot_availability?: 'ready' | 'missing' | 'stale';
  summary: DashboardSummary;
  distributions: Record<string, DistributionPoint[]> & {
    fund_type?: DistributionPoint[];
    invest_type?: DistributionPoint[];
    market?: DistributionPoint[];
    status?: DistributionPoint[];
    management?: DistributionPoint[];
  };
  event_trend: DashboardTrendSeries;
  latest_products?: DashboardLatestProduct[];
}

export interface DashboardFilterOption {
  value: string;
  label: string;
  count?: number;
}

export interface DataQualityWarning {
  code?: string;
  message: string;
  kind?: SegmentKind | string | null;
}

export interface SegmentDataQuality {
  info_file_exists: boolean;
  info_rows: number;
  snapshot_rows: number;
  nav_coverage_rate?: number | null;
  warnings: DataQualityWarning[];
}

export interface DashboardDataQuality {
  snapshot?: {
    exists: boolean;
    status?: 'ready' | 'missing' | 'stale' | 'error';
    rows: number;
    updated_at?: string | null;
    as_of?: string | null;
  };
  segments?: Partial<Record<SegmentKind, SegmentDataQuality>>;
  warnings: DataQualityWarning[];
}

export interface MetricDefinition {
  label: string;
  unit: string;
  source?: string | null;
}

export interface InstrumentAnalyticsResponse {
  schema_version: number;
  kind: DashboardKind;
  status: DashboardStatus;
  as_of?: string | null;
  availability?: Record<string, 'ready' | 'missing' | 'stale'>;
  summary: Partial<Record<DashboardKind, DashboardSummary>>;
  segments: Partial<Record<SegmentKind, DashboardSegment>>;
  available_filters: Record<string, DashboardFilterOption[]> & {
    fund_type?: DashboardFilterOption[];
    invest_type?: DashboardFilterOption[];
    status?: DashboardFilterOption[];
    management?: DashboardFilterOption[];
    market?: DashboardFilterOption[];
  };
  data_quality: DashboardDataQuality;
  metric_definitions: Record<string, MetricDefinition>;
  units?: Record<string, string>;
}

export interface InstrumentTrendResponse {
  schema_version: number;
  kind: DashboardKind;
  status: DashboardStatus;
  series: Partial<Record<SegmentKind, DashboardTrendSeries>>;
  available_values: DashboardFilterOption[];
  data_quality: Pick<DashboardDataQuality, 'warnings'>;
}

export interface RankingItem {
  instrument_type: SegmentKind;
  ts_code: string;
  name?: string | null;
  management?: string | null;
  fund_type?: string | null;
  invest_type?: string | null;
  status?: string | null;
  latest_date?: string | null;
  observation_count?: number | null;
  value?: number | null;
  metrics?: Partial<Record<
    'return_3m' | 'return_1y' | 'return_3y' | 'annual_volatility_1y' | 'max_drawdown_3y' | 'sharpe_1y',
    number | null
  >>;
}

export interface InstrumentRankingsResponse {
  schema_version: number;
  kind: SegmentKind;
  status: 'complete' | 'unavailable';
  metric: string;
  metric_definition: MetricDefinition;
  sort_dir: 'asc' | 'desc';
  page: number;
  page_size: number;
  total: number;
  as_of?: string | null;
  items: RankingItem[];
  data_quality: Pick<DashboardDataQuality, 'warnings'>;
}

export type DashboardFilterKey = 'fund_type' | 'invest_type' | 'status' | 'management' | 'market';
export type DashboardFilters = Record<DashboardFilterKey, string[]>;

export type RefreshJobStatus = 'idle' | 'running' | 'succeeded' | 'failed';
export type RefreshModule = 'base' | 'etf' | 'fund' | 'index';
export type RefreshMode = 'incremental' | 'full';
export type BaseRefreshScope = 'calendar' | 'stock_basic' | 'fund_company';
export type EtfRefreshScope = 'info' | 'nav' | 'share' | 'candle';
export type FundRefreshScope = 'info' | 'nav';
export type IndexScope = 'catalog' | 'domestic' | 'industry' | 'concept' | 'global' | 'futures' | 'valuation' | 'constituents';
export type RefreshScope = BaseRefreshScope | EtfRefreshScope | FundRefreshScope | IndexScope;
export type RefreshModuleScopes = Partial<Record<RefreshModule, RefreshScope[]>>;

export interface DataRefreshStatus {
  source: 'tushare';
  execution_mode?: 'background';
  refresh_locked?: boolean;
  enabled: boolean;
  full_refresh_enabled: boolean;
  available_modules: RefreshModule[];
  available_module_scopes?: RefreshModuleScopes;
  default_module_scopes?: RefreshModuleScopes;
  available_index_scopes?: IndexScope[];
  default_index_scopes?: IndexScope[];
  token_configured: boolean;
  token_source?: 'frontend_local' | 'none';
  token_configuration_enabled: boolean;
  token_editable: boolean;
  data_dir?: string;
  job: {
    job_id?: string | null;
    status: RefreshJobStatus;
    started_at?: string | null;
    finished_at?: string | null;
    modules?: RefreshModule[];
    module_scopes?: RefreshModuleScopes;
    index_scopes?: IndexScope[];
    mode?: RefreshMode | null;
    message: string;
    log_tail?: string;
    staging_data_dir?: string | null;
    fetch_complete?: boolean;
    resumed?: boolean;
    analytics_snapshot?: {
      status: 'succeeded' | 'failed';
      rebuilt_at?: string;
      failed_at?: string;
      rows?: number;
      by_kind?: Partial<Record<SegmentKind, number>>;
      as_of?: string | null;
      message?: string;
    } | null;
    warnings?: DataQualityWarning[];
  };
  datasets: Record<string, {
    file: string;
    exists: boolean;
    status?: 'ready' | 'missing' | 'error';
    rows?: number | null;
    earliest_date?: string | null;
    latest_date?: string | null;
    updated_at?: string;
    error?: string;
  }>;
}

export interface Loadable<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}
