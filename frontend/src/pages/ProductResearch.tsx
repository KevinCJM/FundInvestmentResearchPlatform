import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Link, useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import FilterDropdown, {
  FilterOption,
  SnapshotMetricSelector,
  type SnapshotMetricOption,
} from '../components/FilterDropdown';
import {
  evaluateCustomIndicators,
  getCustomIndicatorMeta,
  listCustomIndicators,
  type EvaluationResult,
  type IndicatorDefinition,
} from '../services/customIndicators';
import {
  MetricDefinitionDrawer,
  MetricMatrix,
  MetricSelector,
} from '../components/metrics/MetricDisplay';
import {
  groupIndicatorsByPeriod,
  normalizeMetricPeriods,
  useMetricDisplayPreference,
  withSelectedIndicators,
} from '../components/metrics/useMetricDisplayPreference';
import {
  assertFixedNjitExecution,
  type FixedNjitExecutionAudit,
} from '../utils/fixedNjitExecution';

interface ProductItem {
  ts_code?: string | null;
  code?: string | null;
  name?: string | null;
  management?: string | null;
  custodian?: string | null;
  fund_type?: string | null;
  type?: string | null;
  invest_type?: string | null;
  qdii_type?: 'QDII' | '非QDII' | string | null;
  market?: string | null;
  status?: string | null;
  benchmark?: string | null;
  issue_amount?: number | null;
  m_fee?: number | null;
  c_fee?: number | null;
  list_date?: string | null;
  found_date?: string | null;
  issue_date?: string | null;
  instrument_type?: 'etf' | 'fund' | string | null;
  condition_values?: Record<string, number | string | null>;
  snapshot_values?: Record<string, number | string | null>;
  snapshot_value_dates?: Record<string, string | null>;
  snapshot_statuses?: Record<string, string | null>;
  snapshot_warnings?: Record<string, string | null>;
}

type ProductConditionOperator = 'gte' | 'lte' | 'gt' | 'lt' | 'eq';

interface ProductCondition {
  field: string;
  operator: ProductConditionOperator;
  value: string;
}

interface ProductConditionField {
  field: string;
  label: string;
  data_type: 'date' | 'number';
  unit_label?: string | null;
  input_scale?: number;
  source: 'fund_basic' | 'instrument_metrics_snapshot' | string;
  available: boolean;
}

interface SnapshotMetricField extends ProductConditionField, SnapshotMetricOption {
  unit: string;
  description: string;
}

interface ProductConditionOperatorOption {
  value: ProductConditionOperator;
  label: string;
  symbol: string;
}

interface ProductsSummary {
  universe_total: number;
  filtered_total: number;
  active_count?: number | null;
  active_rate?: number | null;
  recent_listings_12m?: number | null;
  avg_m_fee?: number | null;
  avg_c_fee?: number | null;
  total_issue_amount?: number | null;
  median_issue_amount?: number | null;
  unique_managements?: number | null;
}

interface ProductsResponse {
  items: ProductItem[];
  page: number;
  page_size: number;
  total: number;
  summary: ProductsSummary;
  available_filters: Record<string, FilterOption[]>;
  condition_fields?: ProductConditionField[];
  condition_operators?: ProductConditionOperatorOption[];
  snapshot_metric_fields?: SnapshotMetricField[];
  selected_snapshot_metrics?: string[];
  snapshot?: { status?: string | null; as_of?: string | null };
  /** 概览 is allowed to screen on hindsight snapshot numbers — but must say so. */
  pit?: { as_of?: string | null; snapshot_is_hindsight?: boolean; warnings?: string[] };
  sort_by: string;
  sort_dir: 'asc' | 'desc' | string;
  execution: FixedNjitExecutionAudit;
}

const integerFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 0 });
const decimalFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 2 });

const formatPercent = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  return `${decimalFormatter.format(value)}%`;
};

const formatIssueAmount = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  if (Math.abs(value) >= 10000) {
    return `${decimalFormatter.format(value / 10000)} 亿`;
  }
  return `${decimalFormatter.format(value)} 万`;
};

const formatSnapshotValue = (value: number | string | null | undefined, unit?: string) => {
  if (value === null || value === undefined || value === '' || Number.isNaN(Number(value))) {
    return '--';
  }
  const numeric = Number(value);
  if (unit === 'ratio') {
    return `${decimalFormatter.format(numeric * 100)}%`;
  }
  if (unit === 'project_normalized_wan') {
    return formatIssueAmount(numeric);
  }
  return decimalFormatter.format(numeric);
};

const formatDate = (value?: string | null) => {
  if (!value) {
    return '--';
  }
  return value;
};

const formatText = (value?: string | null) => {
  if (!value) {
    return '未知';
  }
  return value;
};

const statusTone = (status?: string | null) => {
  if (!status) {
    return 'bg-slate-100 text-slate-600';
  }
  const clean = status.toLowerCase();
  if (clean.includes('终止') || clean.includes('退市') || clean.includes('清盘') || clean.includes('暂停')) {
    return 'bg-rose-100 text-rose-700';
  }
  if (clean.includes('存续') || clean.includes('上市')) {
    return 'bg-emerald-100 text-emerald-700';
  }
  return 'bg-slate-100 text-slate-600';
};

const filterLabels: Record<string, string> = {
  fund_type: '投资类型',
  type: '基金类型',
  invest_type: '投资风格',
  qdii_type: 'QDII 属性',
  market: '交易市场',
  status: '产品状态',
  management: '管理人',
  custodian: '托管人',
};

type FilterState = {
  fund_type: string[];
  type: string[];
  invest_type: string[];
  qdii_type: string[];
  market: string[];
  status: string[];
  management: string[];
  custodian: string[];
};

const initialFilterState: FilterState = {
  fund_type: [],
  type: [],
  invest_type: [],
  qdii_type: [],
  market: [],
  status: [],
  management: [],
  custodian: [],
};

const PAGE_SIZE_OPTIONS = [10, 20, 50, 100];
const CORE_RESEARCH_INDICATOR_IDS = [
  'builtin-total-return-v2',
  'builtin-annualized-return-v2',
  'builtin-annualized-volatility-v2',
  'builtin-maximum-drawdown-v2',
  'builtin-annualized-sharpe-v2',
];
const FILTER_KEYS = Object.keys(initialFilterState) as (keyof FilterState)[];
const CONDITION_OPERATORS = new Set<ProductConditionOperator>(['gte', 'lte', 'gt', 'lt', 'eq']);

const readProductKind = (value: string | null): 'etf' | 'fund' => value === 'fund' ? 'fund' : 'etf';

const readPositiveInteger = (value: string | null, fallback: number) => {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
};

const filtersFromSearchParams = (params: URLSearchParams): FilterState => {
  const next = { ...initialFilterState };
  FILTER_KEYS.forEach((key) => { next[key] = params.getAll(key).filter(Boolean); });
  return next;
};

const conditionsFromSearchParams = (params: URLSearchParams): ProductCondition[] => params
  .getAll('condition')
  .flatMap((entry) => {
    const [field, operator, value] = entry.split('|', 3);
    if (!field || !value || !CONDITION_OPERATORS.has(operator as ProductConditionOperator)) {
      return [];
    }
    return [{ field, operator: operator as ProductConditionOperator, value }];
  });

const fallbackConditionFields = (kind: 'etf' | 'fund'): ProductConditionField[] => [{
  field: kind === 'etf' ? 'list_date' : 'found_date',
  label: kind === 'etf' ? '上市日期' : '成立日期',
  data_type: 'date',
  unit_label: null,
  input_scale: 1,
  source: 'fund_basic',
  available: true,
}];

export default function ProductResearch() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [productKind, setProductKind] = useState<'etf' | 'fund'>(() => readProductKind(searchParams.get('kind')));
  const [response, setResponse] = useState<ProductsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(() => readPositiveInteger(searchParams.get('page'), 1));
  const [pageSize, setPageSize] = useState(() => {
    const requested = readPositiveInteger(searchParams.get('page_size'), PAGE_SIZE_OPTIONS[0]);
    return PAGE_SIZE_OPTIONS.includes(requested) ? requested : PAGE_SIZE_OPTIONS[0];
  });
  const [filters, setFilters] = useState<FilterState>(() => filtersFromSearchParams(searchParams));
  const [conditions, setConditions] = useState<ProductCondition[]>(() => conditionsFromSearchParams(searchParams));
  const [snapshotMetrics, setSnapshotMetrics] = useState<string[]>(() => (
    Array.from(new Set(searchParams.getAll('snapshot_metric').filter(Boolean))).slice(0, 8)
  ));
  const [sortKey, setSortKey] = useState(() => searchParams.get('sort_by') || 'issue_amount');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>(() => searchParams.get('sort_dir') === 'asc' ? 'asc' : 'desc');
  const [searchInput, setSearchInput] = useState(() => searchParams.get('q') ?? '');
  const [searchKeyword, setSearchKeyword] = useState(() => searchParams.get('q')?.trim() ?? '');
  const [selectedProducts, setSelectedProducts] = useState<Record<string, { id: string; name: string; code: string }>>({});
  const [allMatchingSelected, setAllMatchingSelected] = useState(false);
  const [excludedProductIds, setExcludedProductIds] = useState<Set<string>>(() => new Set());
  const [viewMode, setViewMode] = useState<'basic' | 'metrics'>('basic');
  const [researchIndicators, setResearchIndicators] = useState<IndicatorDefinition[]>([]);
  const [researchPeriods, setResearchPeriods] = useState<string[]>(['1Y']);
  const [researchResults, setResearchResults] = useState<EvaluationResult[]>([]);
  const [researchLoading, setResearchLoading] = useState(false);
  const [researchError, setResearchError] = useState<string | null>(null);
  const [researchAsOf, setResearchAsOf] = useState('');
  const [definitionIndicator, setDefinitionIndicator] = useState<IndicatorDefinition | null>(null);
  const [researchPreference, setResearchPreference] = useMetricDisplayPreference(
    'product-research',
    'single_product',
    CORE_RESEARCH_INDICATOR_IDS,
    '1Y',
    researchIndicators.map((indicator) => indicator.id),
  );
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const handler = window.setTimeout(() => {
      const nextKeyword = searchInput.trim();
      if (nextKeyword !== searchKeyword) {
        setSelectedProducts({});
        setAllMatchingSelected(false);
        setExcludedProductIds(new Set());
      }
      setSearchKeyword(nextKeyword);
      setPage(1);
    }, 400);
    return () => window.clearTimeout(handler);
  }, [searchInput, searchKeyword]);

  useEffect(() => {
    if (!['/research', '/product-research/products'].includes(location.pathname)) {
      return;
    }
    const next = new URLSearchParams();
    next.set('kind', productKind);
    if (searchKeyword) {
      next.set('q', searchKeyword);
    }
    FILTER_KEYS.forEach((key) => filters[key].forEach((value) => next.append(key, value)));
    conditions.forEach((condition) => next.append(
      'condition',
      `${condition.field}|${condition.operator}|${condition.value}`,
    ));
    snapshotMetrics.forEach((metric) => next.append('snapshot_metric', metric));
    next.set('sort_by', sortKey);
    next.set('sort_dir', sortDir);
    if (page > 1) {
      next.set('page', String(page));
    }
    if (pageSize !== PAGE_SIZE_OPTIONS[0]) {
      next.set('page_size', String(pageSize));
    }
    if (next.toString() !== searchParams.toString()) {
      setSearchParams(next, { replace: true });
    }
  }, [conditions, filters, location.pathname, page, pageSize, productKind, searchKeyword, searchParams, setSearchParams, snapshotMetrics, sortDir, sortKey]);

  useEffect(() => {
    const controller = new AbortController();
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        const params = new URLSearchParams();
        params.set('page', page.toString());
        params.set('page_size', pageSize.toString());
        params.set('sort_by', sortKey);
        params.set('sort_dir', sortDir);
        if (searchKeyword) {
          params.set('q', searchKeyword);
        }
        (Object.keys(filters) as (keyof FilterState)[]).forEach((key) => {
          filters[key].forEach((value) => {
            params.append(key, value);
          });
        });
        conditions.forEach((condition) => params.append(
          'condition',
          `${condition.field}|${condition.operator}|${condition.value}`,
        ));
        snapshotMetrics.forEach((metric) => params.append('snapshot_metric', metric));
        params.set('kind', productKind);
        const resp = await fetch(`/api/instruments/products?${params.toString()}`, { signal: controller.signal });
        if (!resp.ok) {
          if (resp.status === 404) {
            setResponse(null);
            setError(`未找到${productKind === 'etf' ? 'ETF' : '场外公募基金'}产品信息，请先在主页更新对应数据模块。`);
            return;
          }
          const payload = await resp.json().catch(() => null) as { detail?: string } | null;
          throw new Error(payload?.detail || '加载产品列表失败');
        }
        const data = (await resp.json()) as ProductsResponse;
        assertFixedNjitExecution(data.execution, '产品研究统计');
        setResponse(data);
      } catch (err) {
        if ((err as DOMException).name === 'AbortError') {
          return;
        }
        console.error('Failed to load products', err);
        setError(err instanceof Error ? err.message : '产品数据加载失败，请稍后重试。');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
    return () => controller.abort();
  }, [conditions, page, pageSize, sortKey, sortDir, searchKeyword, filters, productKind, snapshotMetrics]);

  useEffect(() => {
    let active = true;
    Promise.all([
      listCustomIndicators({ contextKind: 'single_product', productKind }),
      getCustomIndicatorMeta(),
    ]).then(([catalog, metadata]) => {
      if (!active) return;
      setResearchIndicators(catalog.items);
      const periods = metadata.periods.map((item) => item.value);
      setResearchPeriods(periods);
      setResearchPreference((current) => normalizeMetricPeriods(current, periods, '1Y'));
    }).catch(() => { if (active) setResearchError('指标目录暂时不可用。'); });
    return () => { active = false; };
  }, [productKind]);

  const currentPageTargets = useMemo(() => (response?.items ?? []).flatMap((item) => {
    const productId = item.ts_code ?? item.code;
    return productId ? [{ kind: productKind, product_id: productId, name: item.name ?? productId }] : [];
  }), [productKind, response?.items]);
  const selectedResearchIndicators = useMemo(() => researchPreference.indicatorIds
    .map((id) => researchIndicators.find((indicator) => indicator.id === id))
    .filter((indicator): indicator is IndicatorDefinition => Boolean(indicator)), [researchIndicators, researchPreference.indicatorIds]);

  useEffect(() => {
    if (viewMode !== 'metrics' || currentPageTargets.length === 0 || selectedResearchIndicators.length === 0) {
      setResearchResults([]);
      return;
    }
    let active = true;
    setResearchLoading(true); setResearchError(null);
    const selectedPreference = {
      ...researchPreference,
      indicatorIds: selectedResearchIndicators.map((indicator) => indicator.id),
    };
    Promise.all(groupIndicatorsByPeriod(selectedPreference, '1Y').map(({ indicatorIds, period }) => (
      evaluateCustomIndicators({
        indicator_ids: indicatorIds,
        targets: currentPageTargets.map(({ kind, product_id }) => ({ kind, product_id })),
        period,
        as_of: researchAsOf || undefined,
      })
    ))).then((responses) => { if (active) setResearchResults(responses.flatMap(({ results }) => results)); })
      .catch(() => { if (active) { setResearchResults([]); setResearchError('当前页指标计算失败，请检查真实数据与样本窗口。'); } })
      .finally(() => { if (active) setResearchLoading(false); });
    return () => { active = false; };
  }, [currentPageTargets, researchAsOf, researchPreference.periodsByIndicator, selectedResearchIndicators, viewMode]);

  const switchProductKind = (kind: 'etf' | 'fund') => {
    setProductKind(kind);
    setPage(1);
    setFilters(initialFilterState);
    setConditions([]);
    setSnapshotMetrics([]);
    setSelectedProducts({});
    setAllMatchingSelected(false);
    setExcludedProductIds(new Set());
    if (sortKey === 'list_date' || sortKey === 'found_date') {
      setSortKey(kind === 'etf' ? 'list_date' : 'found_date');
      setSortDir('desc');
    }
  };

  const summary = response?.summary;
  const totalPages = useMemo(() => {
    if (!response) {
      return 0;
    }
    return Math.max(1, Math.ceil(response.total / pageSize));
  }, [response, pageSize]);

  const handleFilterChange = (key: keyof FilterState) => (values: string[]) => {
    setFilters((prev) => ({ ...prev, [key]: values }));
    setSelectedProducts({});
    setAllMatchingSelected(false);
    setExcludedProductIds(new Set());
    setPage(1);
  };

  const clearAllFilters = () => {
    setFilters(initialFilterState);
    setConditions([]);
    setSelectedProducts({});
    setAllMatchingSelected(false);
    setExcludedProductIds(new Set());
    setPage(1);
  };

  const addCondition = (condition: ProductCondition) => {
    setConditions((current) => [...current, condition]);
    setSelectedProducts({});
    setAllMatchingSelected(false);
    setExcludedProductIds(new Set());
    setPage(1);
  };

  const removeCondition = (index: number) => {
    setConditions((current) => current.filter((_, itemIndex) => itemIndex !== index));
    setSelectedProducts({});
    setAllMatchingSelected(false);
    setExcludedProductIds(new Set());
    setPage(1);
  };

  const activeFilterChips = useMemo(() => {
    const chips: { key: keyof FilterState; value: string; label: string }[] = [];
    const optionLookup: Record<string, Record<string, string>> = {};
    if (response?.available_filters) {
      Object.entries(response.available_filters).forEach(([filterKey, options]) => {
        optionLookup[filterKey] = options.reduce<Record<string, string>>((acc, option) => {
          acc[option.value] = option.label;
          return acc;
        }, {});
      });
    }
    (Object.keys(filters) as (keyof FilterState)[]).forEach((key) => {
      filters[key].forEach((value) => {
        const label = optionLookup[key]?.[value] ?? value;
        chips.push({ key, value, label });
      });
    });
    return chips;
  }, [filters, response?.available_filters]);

  const removeChip = (chipKey: keyof FilterState, value: string) => {
    setFilters((prev) => ({
      ...prev,
      [chipKey]: prev[chipKey].filter((item) => item !== value),
    }));
    setSelectedProducts({});
    setAllMatchingSelected(false);
    setExcludedProductIds(new Set());
    setPage(1);
  };

  const toggleSort = (key: string) => {
    if (sortKey === key) {
      setSortDir((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir(key === 'list_date' || key === 'found_date' ? 'desc' : 'asc');
    }
  };

  const appliedFiltersCount = activeFilterChips.length + conditions.length;
  const selectedList = useMemo(() => Object.values(selectedProducts), [selectedProducts]);
  const selectedCount = allMatchingSelected
    ? Math.max(0, (response?.total ?? 0) - excludedProductIds.size)
    : selectedList.length;
  const conditionFields = response?.condition_fields ?? fallbackConditionFields(productKind);
  const snapshotMetricFields = response?.snapshot_metric_fields ?? [];
  const selectedSnapshotMetricFields = snapshotMetrics.flatMap((field) => {
    const definition = snapshotMetricFields.find((item) => item.field === field);
    return definition ? [definition] : [];
  });
  const conditionOperators = response?.condition_operators ?? [];
  const selectHeaderRef = useRef<HTMLTableCellElement | null>(null);
  const [selectColOffset, setSelectColOffset] = useState<number>(0);

  useLayoutEffect(() => {
    const updateOffset = () => {
      if (selectHeaderRef.current) {
        setSelectColOffset(selectHeaderRef.current.offsetWidth);
      }
    };
    updateOffset();
    window.addEventListener('resize', updateOffset);
    return () => window.removeEventListener('resize', updateOffset);
  }, [response, pageSize, sortKey, sortDir]);

  const productLeft = selectColOffset || selectHeaderRef.current?.offsetWidth || 0;

  const currentPageSelectableProducts = (response?.items ?? []).flatMap((item) => {
    const id = item.ts_code ?? item.code;
    return id ? [{ id, name: item.name ?? id, code: item.ts_code ?? item.code ?? id }] : [];
  });
  const isProductSelected = (productId: string) => (
    allMatchingSelected ? !excludedProductIds.has(productId) : Boolean(selectedProducts[productId])
  );
  const currentPageAllSelected = currentPageSelectableProducts.length > 0
    && currentPageSelectableProducts.every((item) => isProductSelected(item.id));

  const toggleCurrentPageSelection = () => {
    const ids = currentPageSelectableProducts.map((item) => item.id);
    if (allMatchingSelected) {
      setExcludedProductIds((current) => {
        const next = new Set(current);
        ids.forEach((id) => {
          if (currentPageAllSelected) next.add(id);
          else next.delete(id);
        });
        return next;
      });
      return;
    }
    setSelectedProducts((current) => {
      const next = { ...current };
      currentPageSelectableProducts.forEach((item) => {
        if (currentPageAllSelected) delete next[item.id];
        else next[item.id] = item;
      });
      return next;
    });
  };

  const toggleAllMatchingSelection = () => {
    if (allMatchingSelected) {
      setAllMatchingSelected(false);
      setExcludedProductIds(new Set());
      setSelectedProducts({});
      return;
    }
    setAllMatchingSelected(true);
    setExcludedProductIds(new Set());
    setSelectedProducts({});
  };

  const toggleProductSelection = (productId: string | null | undefined, productName?: string | null, productCode?: string | null) => {
    if (!productId) {
      return;
    }
    if (allMatchingSelected) {
      setExcludedProductIds((current) => {
        const next = new Set(current);
        if (next.has(productId)) next.delete(productId);
        else next.add(productId);
        return next;
      });
      return;
    }
    setSelectedProducts((prev) => {
      if (prev[productId]) {
        const { [productId]: _removed, ...rest } = prev;
        return rest;
      }
      return {
        ...prev,
        [productId]: {
          id: productId,
          name: productName ?? productId,
          code: productCode ?? productId,
        },
      };
    });
  };

  const goToComparison = () => {
    if (allMatchingSelected || selectedCount === 0 || selectedCount > 10) {
      return;
    }
    const ids = selectedList.map((item) => encodeURIComponent(item.id)).join(',');
    navigate(`/product-research/compare?ids=${ids}&kind=${productKind}`);
  };

  const canCompareSelection = !allMatchingSelected && selectedCount > 0 && selectedCount <= 10;
  const selectionActionHint = allMatchingSelected
    ? '全选筛选结果是逻辑选择；请取消全选后手动选择最多 10 个产品进行对比。'
    : selectedCount > 10
      ? '产品对比最多支持 10 个产品。'
      : undefined;

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <div className="mb-8 space-y-4">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">产品研究</h1>
            <p className="mt-2 text-base text-slate-600">
              分别研究 ETF 与场外公募基金的规模、费率、投资风格和管理人，为统一资产配置提供候选池。
            </p>
          </div>
          <div className="inline-flex self-start rounded-xl bg-slate-100 p-1">
            {(['etf', 'fund'] as const).map((kind) => (
              <button
                key={kind}
                type="button"
                onClick={() => switchProductKind(kind)}
                className={`rounded-lg px-5 py-2 text-sm font-semibold transition ${
                  productKind === kind ? 'bg-white text-emerald-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                }`}
              >
                {kind === 'etf' ? 'ETF' : '场外公募基金'}
              </button>
            ))}
          </div>
        </div>
        <div className="inline-flex self-start rounded-xl border border-slate-200 bg-white p-1" aria-label="产品研究视图">
          <button type="button" onClick={() => setViewMode('basic')} className={`min-h-11 rounded-lg px-5 text-sm font-semibold ${viewMode === 'basic' ? 'bg-slate-900 text-white' : 'text-slate-600'}`}>基础资料</button>
          <button type="button" aria-label="切换到研究指标视图" onClick={() => setViewMode('metrics')} className={`min-h-11 rounded-lg px-5 text-sm font-semibold ${viewMode === 'metrics' ? 'bg-violet-600 text-white' : 'text-slate-600'}`}><span aria-hidden="true">指标分析</span></button>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard
            title="当前筛选产品"
            value={summary ? `${integerFormatter.format(summary.filtered_total)} / ${integerFormatter.format(summary.universe_total)}` : '--'}
            description={`筛选结果 / 全部${productKind === 'etf' ? 'ETF' : '场外公募基金'}产品代码`}
          />
          <MetricCard
            title="有效存续产品"
            value={summary?.active_count !== undefined && summary?.active_count !== null ? integerFormatter.format(summary.active_count) : '--'}
            description={summary?.active_rate !== undefined && summary?.active_rate !== null
              ? `占比 ${formatSnapshotValue(summary.active_rate, 'ratio')}`
              : '存续状态不可用'}
          />
          <MetricCard
            title="筛选合计发行规模（非AUM）"
            value={formatIssueAmount(summary?.total_issue_amount)}
            description="单位：按万转亿换算"
          />
          <MetricCard
            title="平均管理费 / 托管费"
            value={`${formatPercent(summary?.avg_m_fee)} · ${formatPercent(summary?.avg_c_fee)}`}
            description="费用率均值按当前筛选样本统计"
          />
        </div>
      </div>

      <section className="mb-8 space-y-4 rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-100">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex w-full max-w-xl items-center rounded-xl border border-slate-200 bg-slate-50 px-4 py-2">
            <svg className="h-5 w-5 text-slate-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="7" />
              <line x1="20" y1="20" x2="16.65" y2="16.65" />
            </svg>
            <input
              value={searchInput}
              onChange={(event) => setSearchInput(event.target.value)}
              placeholder="按代码、名称或管理人搜索"
              className="ml-3 w-full bg-transparent text-sm text-slate-700 outline-none placeholder:text-slate-400"
            />
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-600">
              已选条件 {appliedFiltersCount}
            </span>
            {appliedFiltersCount > 0 && (
              <button
                type="button"
                onClick={clearAllFilters}
                className="text-sm font-medium text-emerald-600 hover:text-emerald-500"
              >
                重置全部筛选
              </button>
            )}
          </div>
        </div>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          <FilterDropdown
            label="投资类型"
            options={response?.available_filters?.fund_type ?? []}
            selected={filters.fund_type}
            onChange={handleFilterChange('fund_type')}
          />
          <FilterDropdown
            label="基金类型"
            options={response?.available_filters?.type ?? []}
            selected={filters.type}
            onChange={handleFilterChange('type')}
          />
          <FilterDropdown
            label="投资风格"
            options={response?.available_filters?.invest_type ?? []}
            selected={filters.invest_type}
            onChange={handleFilterChange('invest_type')}
          />
          <FilterDropdown
            label="QDII 属性"
            options={response?.available_filters?.qdii_type ?? []}
            selected={filters.qdii_type}
            onChange={handleFilterChange('qdii_type')}
          />
          <FilterDropdown
            label="交易市场"
            options={response?.available_filters?.market ?? []}
            selected={filters.market}
            onChange={handleFilterChange('market')}
          />
          <FilterDropdown
            label="产品状态"
            options={response?.available_filters?.status ?? []}
            selected={filters.status}
            onChange={handleFilterChange('status')}
          />
          <FilterDropdown
            label="管理人"
            options={response?.available_filters?.management ?? []}
            selected={filters.management}
            onChange={handleFilterChange('management')}
          />
          <FilterDropdown
            label="托管人"
            options={response?.available_filters?.custodian ?? []}
            selected={filters.custodian}
            onChange={handleFilterChange('custodian')}
          />
        </div>
        <ProductConditionBuilder
          fields={conditionFields}
          operators={conditionOperators}
          conditions={conditions}
          snapshotStatus={response?.snapshot?.status}
          onAdd={addCondition}
          onRemove={removeCondition}
        />
        {/* The研究 surfaces recompute under the研究日; this screening table reads
            the全历史 snapshot, so the difference has to be visible here. */}
        {response?.pit?.snapshot_is_hindsight && (
          <p
            className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-900"
            data-testid="product-research-snapshot-hindsight"
          >
            {response.pit.warnings?.[0]}
          </p>
        )}
        {activeFilterChips.length > 0 && (
          <div className="flex flex-wrap gap-2 border-t border-slate-100 pt-4">
            {activeFilterChips.map((chip) => (
              <button
                key={`${chip.key}-${chip.value}`}
                type="button"
                onClick={() => removeChip(chip.key, chip.value)}
                className="inline-flex items-center gap-2 rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-600 hover:bg-emerald-100"
              >
                <span className="rounded bg-white px-2 py-0.5 text-[10px] font-semibold text-emerald-500">
                  {filterLabels[chip.key] ?? chip.key}
                </span>
                {chip.label}
                <svg className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                  <path
                    fillRule="evenodd"
                    d="M10 8.586l3.182-3.182a1 1 0 011.414 1.414L11.414 10l3.182 3.182a1 1 0 01-1.414 1.414L10 11.414l-3.182 3.182a1 1 0 01-1.414-1.414L8.586 10l-3.182-3.182a1 1 0 011.414-1.414L10 8.586z"
                    clipRule="evenodd"
                  />
                </svg>
              </button>
            ))}
          </div>
        )}
      </section>

      <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-100">
        <div className="flex flex-col gap-3 border-b border-slate-100 px-6 py-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">{viewMode === 'basic' ? '产品列表' : '当前页指标矩阵'}</h2>
            <div className="mt-1 text-xs text-slate-500">
              {viewMode === 'basic' ? (
                <>
                  <span className="font-medium text-emerald-600">提示：点击产品名称可进入单产品研究页面</span>
                  <span aria-hidden="true" className="mx-1 text-slate-300">·</span>
                  <span>支持本页全选和当前筛选结果全选；产品对比最多使用 10 个产品</span>
                </>
              ) : '只批量计算当前分页产品；每个指标可独立选择计算区间'}
            </div>
            {selectionActionHint && selectedCount > 0 && (
              <div className="mt-2 max-w-xl text-xs font-medium text-amber-700" role="status">{selectionActionHint}</div>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-3 text-sm text-slate-500">
            {viewMode === 'basic' && (
              <SnapshotMetricSelector
                options={snapshotMetricFields}
                selected={snapshotMetrics}
                onChange={(metrics) => {
                  setSnapshotMetrics(metrics.slice(0, 8));
                  setPage(1);
                }}
                maxSelected={8}
              />
            )}
            {viewMode === 'metrics' && <>
              <MetricSelector indicators={researchIndicators} selectedIds={researchPreference.indicatorIds} onChange={(indicatorIds) => setResearchPreference((current) => withSelectedIndicators(current, indicatorIds, '1Y'))} maxSelected={5} label="选择展示指标" />
              <label className="text-xs font-medium text-slate-600">截止日<input type="date" value={researchAsOf} onChange={(event) => setResearchAsOf(event.target.value)} className="ml-2 min-h-11 rounded-lg border border-slate-200 px-3 text-sm" /></label>
            </>}
            <span>每页</span>
            <select
              aria-label="产品研究每页产品数量"
              value={pageSize}
              onChange={(event) => {
                setPageSize(Number(event.target.value));
                setPage(1);
              }}
              className="rounded-lg border border-slate-200 px-3 py-1 text-sm text-slate-600 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            >
              {PAGE_SIZE_OPTIONS.map((size) => (
                <option key={size} value={size}>
                  {size}
                </option>
              ))}
            </select>
            <span className="text-slate-400">共 {response?.total ?? 0} 条</span>
            <button
              type="button"
              onClick={toggleCurrentPageSelection}
              disabled={currentPageSelectableProducts.length === 0}
              className="rounded-lg border border-slate-200 px-3 py-1 text-sm font-semibold text-slate-600 hover:border-emerald-400 hover:text-emerald-600 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {currentPageAllSelected ? '取消本页全选' : '本页全选'}
            </button>
            <button
              type="button"
              onClick={toggleAllMatchingSelection}
              disabled={!response || response.total === 0}
              className={`rounded-lg border px-3 py-1 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-50 ${allMatchingSelected ? 'border-emerald-500 bg-emerald-50 text-emerald-700' : 'border-slate-200 text-slate-600 hover:border-emerald-400 hover:text-emerald-600'}`}
            >
              {allMatchingSelected ? '取消全选' : `全选 ${integerFormatter.format(response?.total ?? 0)} 条`}
            </button>
            <div className="flex items-center gap-2 rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-600">
              已选 {integerFormatter.format(selectedCount)}
            </div>
            <button
              type="button"
              onClick={goToComparison}
              disabled={!canCompareSelection}
              title={selectionActionHint}
              className="inline-flex items-center gap-2 rounded-lg bg-emerald-500 px-3 py-1 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-emerald-600 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-500"
            >
              产品对比
              {selectedCount > 0 && <span className="rounded-full bg-white px-2 py-0.5 text-xs font-bold text-emerald-600">{selectedCount}</span>}
            </button>
          </div>
        </div>

        {viewMode === 'metrics' ? (
          <div className="p-4">
            {loading || researchLoading ? <div className="py-20 text-center text-slate-500">正在计算当前页指标…</div> : error || researchError ? <div className="py-20 text-center text-rose-600">{error ?? researchError}</div> : currentPageTargets.length === 0 ? <div className="py-20 text-center text-slate-500">当前页没有可计算产品。</div> : <MetricMatrix indicators={selectedResearchIndicators} targets={currentPageTargets} results={researchResults} periodsByIndicator={researchPreference.periodsByIndicator} periodOptions={researchPeriods} onPeriodChange={(indicatorId, period) => setResearchPreference((current) => ({ ...current, periodsByIndicator: { ...current.periodsByIndicator, [indicatorId]: period } }))} onDefinition={setDefinitionIndicator} />}
          </div>
        ) : loading ? (
          <div className="flex items-center justify-center px-6 py-24 text-slate-400">
            <div className="flex items-center gap-3">
              <svg className="h-5 w-5 animate-spin text-emerald-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle className="opacity-25" cx="12" cy="12" r="10" />
                <path className="opacity-75" d="M4 12a8 8 0 018-8" />
              </svg>
              加载中...
            </div>
          </div>
        ) : error ? (
          <div className="px-6 py-24 text-center">
            <div className="mx-auto max-w-md space-y-4">
              <div className="inline-flex rounded-full bg-rose-50 px-4 py-1 text-sm font-semibold text-rose-500">提示</div>
              <p className="text-lg font-semibold text-slate-800">{error}</p>
              <p className="text-sm text-slate-500">请检查数据目录或稍后重试，如需帮助可联系系统管理员。</p>
            </div>
          </div>
        ) : response && response.items.length === 0 ? (
          <div className="px-6 py-24 text-center text-slate-500">暂无符合筛选条件的{productKind === 'etf' ? 'ETF' : '场外公募基金'}。</div>
        ) : (
          <div className="h-[520px] w-full overflow-auto">
            <table className="products-table min-w-[1280px] divide-y divide-slate-100">
              <thead className="bg-slate-50">
                <tr>
                  <th
                    scope="col"
                    ref={selectHeaderRef}
                    className="sticky left-0 top-0 z-50 border-r border-slate-100 bg-slate-50 px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500 whitespace-nowrap w-[120px] min-w-[120px]"
                  >
                    选择
                  </th>
                  <th
                    scope="col"
                    className="sticky top-0 z-50 border-r border-slate-100 bg-slate-50 px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500 whitespace-nowrap"
                    style={{ left: productLeft }}
                  >
                    产品
                  </th>
                  <th scope="col" className="sticky top-0 z-40 bg-slate-50 px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500 whitespace-nowrap">
                    基金类型 / 投资类型 / QDII
                  </th>
                  <th scope="col" className="sticky top-0 z-40 bg-slate-50 px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500 whitespace-nowrap">
                    风格 / 市场
                  </th>
                  <th scope="col" className="sticky top-0 z-40 bg-slate-50 px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500 whitespace-nowrap">
                    管理 / 托管
                  </th>
                  <th scope="col" className="sticky top-0 z-40 bg-slate-50 px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500 whitespace-nowrap">
                    <SortButton label="发行规模" activeKey={sortKey} columnKey="issue_amount" direction={sortDir} onClick={toggleSort} />
                  </th>
                  {selectedSnapshotMetricFields.map((field) => (
                    <th
                      key={field.field}
                      scope="col"
                      title={field.description}
                      className="sticky top-0 z-40 bg-violet-50 px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-violet-700 whitespace-nowrap"
                    >
                      {field.label}
                      <span className="ml-1 font-normal text-violet-400">快照</span>
                    </th>
                  ))}
                  <th scope="col" className="sticky top-0 z-40 bg-slate-50 px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500 whitespace-nowrap">
                    <SortButton label="费用 / 基准" activeKey={sortKey} columnKey="m_fee" direction={sortDir} onClick={toggleSort} />
                  </th>
                  <th scope="col" className="sticky top-0 z-40 bg-slate-50 px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500 whitespace-nowrap">
                    <SortButton label={productKind === 'etf' ? '上市日' : '成立日'} activeKey={sortKey} columnKey={productKind === 'etf' ? 'list_date' : 'found_date'} direction={sortDir} onClick={toggleSort} />
                  </th>
                  <th scope="col" className="sticky top-0 z-40 bg-slate-50 px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500 whitespace-nowrap">
                    状态
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white">
                {response?.items.map((item) => {
                  const code = item.ts_code ?? item.code ?? '--';
                  const detailPath = code && code !== '--'
                    ? `/product-research/products/${encodeURIComponent(code)}?kind=${productKind}`
                    : undefined;
                  const selectionId = item.ts_code ?? item.code ?? null;
                  const isSelected = selectionId ? isProductSelected(selectionId) : false;
                  return (
                    <tr key={`${code}-${item.name}`} className="group hover:bg-emerald-50/40">
                      <td
                        className="sticky left-0 z-40 border-r border-slate-100 bg-white px-6 py-4 whitespace-nowrap group-hover:bg-emerald-50/40 w-[120px] min-w-[120px]"
                      >
                        <input
                          type="checkbox"
                          className="h-4 w-4 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
                          checked={isSelected}
                          disabled={!selectionId}
                          aria-label={`选择 ${item.name ?? code}`}
                          onChange={() => toggleProductSelection(selectionId, item.name, code)}
                        />
                      </td>
                      <td
                        className="sticky z-40 border-r border-slate-100 bg-white px-6 py-4 group-hover:bg-emerald-50/40"
                        style={{ left: productLeft }}
                      >
                        {detailPath ? (
                          <Link
                            to={detailPath}
                            target="_blank"
                            rel="noreferrer"
                            aria-label={`进入${item.name ?? code}的单产品研究页面`}
                            title="点击进入单产品研究页面"
                            className="group block rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2"
                          >
                            <div className="text-sm font-semibold text-emerald-600 group-hover:text-emerald-700">
                              {item.name ?? '--'}
                            </div>
                            <div className="mt-1 text-xs text-emerald-500 group-hover:text-emerald-600">{code}</div>
                          </Link>
                        ) : (
                          <div>
                            <div className="text-sm font-semibold text-slate-900">{item.name ?? '--'}</div>
                            <div className="mt-1 text-xs text-slate-500">{code}</div>
                          </div>
                        )}
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-600">
                        <div className="font-medium text-slate-700">{formatText(item.type)}</div>
                        <div className="flex items-center gap-2 text-xs text-slate-400">
                          <span>{formatText(item.fund_type)}</span>
                          {item.qdii_type && (
                            <span className={`rounded-full px-2 py-0.5 font-semibold ${item.qdii_type === 'QDII' ? 'bg-violet-100 text-violet-700' : 'bg-slate-100 text-slate-500'}`}>
                              {item.qdii_type}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-600">
                        <div className="font-medium text-slate-700">{formatText(item.invest_type)}</div>
                        <div className="text-xs text-slate-400">{formatText(item.market)}</div>
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-600">
                        <div className="font-medium text-slate-700">{formatText(item.management)}</div>
                        <div className="text-xs text-slate-400">{formatText(item.custodian)}</div>
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-600">
                        <div className="font-semibold text-slate-800">{formatIssueAmount(item.issue_amount)}</div>
                        <div className="text-xs text-slate-400">发行披露口径（非当前 AUM）</div>
                      </td>
                      {selectedSnapshotMetricFields.map((field) => {
                        const value = item.snapshot_values?.[field.field];
                        const asOf = item.snapshot_value_dates?.[field.field];
                        const snapshotStatus = item.snapshot_statuses?.[field.field];
                        const snapshotWarning = item.snapshot_warnings?.[field.field];
                        return (
                          <td key={field.field} className="bg-violet-50/30 px-6 py-4 text-sm text-slate-600">
                            <div className="font-semibold text-slate-800">
                              {formatSnapshotValue(value, field.unit)}
                            </div>
                            <div className="text-xs text-slate-400">
                              {snapshotWarning
                                ? snapshotWarning
                                : asOf
                                  ? `快照截至 ${formatDate(asOf)}`
                                  : snapshotStatus === 'unavailable'
                                    ? '该产品当前不可计算此指标'
                                    : '快照暂无可用值'}
                            </div>
                          </td>
                        );
                      })}
                      <td className="px-6 py-4 text-sm text-slate-600">
                        <div className="font-medium text-slate-700">管理费 {formatPercent(item.m_fee)} / 托管费 {formatPercent(item.c_fee)}</div>
                        <div className="text-xs text-slate-400">基准 {formatText(item.benchmark)} · 产品类型 {formatText(item.fund_type)}</div>
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-600">
                        <div className="font-medium text-slate-700">{formatDate(productKind === 'etf' ? item.list_date : item.found_date)}</div>
                        <div className="text-xs text-slate-400">{productKind === 'etf' ? `成立：${formatDate(item.found_date ?? item.issue_date)}` : `发行：${formatDate(item.issue_date)}`}</div>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold ${statusTone(item.status)}`}>
                          {formatText(item.status)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {response && response.items.length > 0 && (
          <div className="flex flex-col gap-4 border-t border-slate-100 px-6 py-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="text-sm text-slate-500">
              第 {page} / {totalPages} 页
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setPage((prev) => Math.max(1, prev - 1))}
                disabled={page === 1}
                className="rounded-lg border border-slate-200 px-3 py-1 text-sm font-medium text-slate-600 disabled:cursor-not-allowed disabled:opacity-50 hover:border-emerald-400 hover:text-emerald-600"
              >
                上一页
              </button>
              <button
                type="button"
                onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
                disabled={page >= totalPages}
                className="rounded-lg border border-slate-200 px-3 py-1 text-sm font-medium text-slate-600 disabled:cursor-not-allowed disabled:opacity-50 hover:border-emerald-400 hover:text-emerald-600"
              >
                下一页
              </button>
            </div>
          </div>
        )}
        {selectedCount > 0 && (
          <div className="flex flex-wrap gap-2 border-t border-slate-100 px-6 py-4 text-xs text-emerald-600">
            {allMatchingSelected ? (
              <span className="rounded-full bg-emerald-50 px-3 py-1 font-semibold">
                已选择全部符合筛选条件的产品{excludedProductIds.size > 0 ? `，排除 ${excludedProductIds.size} 个` : ''}
              </span>
            ) : (
              <>
                {selectedList.slice(0, 20).map((item) => (
                  <span key={item.id} className="inline-flex items-center gap-2 rounded-full bg-emerald-50 px-3 py-1">
                    {item.name}
                    <button
                      type="button"
                      aria-label={`取消选择 ${item.name}`}
                      className="text-emerald-500 hover:text-emerald-700"
                      onClick={() => toggleProductSelection(item.id, item.name, item.code)}
                    >
                      ×
                    </button>
                  </span>
                ))}
                {selectedList.length > 20 && <span className="px-2 py-1">另有 {selectedList.length - 20} 个已选产品</span>}
              </>
            )}
          </div>
        )}
      </section>
      <MetricDefinitionDrawer indicator={definitionIndicator} onClose={() => setDefinitionIndicator(null)} />
    </div>
  );
}

const defaultConditionOperators: ProductConditionOperatorOption[] = [
  { value: 'gte', label: '大于等于', symbol: '≥' },
  { value: 'lte', label: '小于等于', symbol: '≤' },
  { value: 'gt', label: '大于', symbol: '>' },
  { value: 'lt', label: '小于', symbol: '<' },
  { value: 'eq', label: '等于', symbol: '=' },
];

interface ProductConditionBuilderProps {
  fields: ProductConditionField[];
  operators: ProductConditionOperatorOption[];
  conditions: ProductCondition[];
  snapshotStatus?: string | null;
  onAdd: (condition: ProductCondition) => void;
  onRemove: (index: number) => void;
}

function formatProductCondition(
  condition: ProductCondition,
  fields: ProductConditionField[],
  operators: ProductConditionOperatorOption[],
) {
  const field = fields.find((item) => item.field === condition.field);
  const operator = operators.find((item) => item.value === condition.operator);
  return `${field?.label ?? condition.field} ${operator?.symbol ?? condition.operator} ${condition.value}${field?.unit_label ?? ''}`;
}

function ProductConditionBuilder({
  fields,
  operators,
  conditions,
  snapshotStatus,
  onAdd,
  onRemove,
}: ProductConditionBuilderProps) {
  const availableFields = useMemo(() => fields.filter((field) => field.available), [fields]);
  const resolvedOperators = operators.length > 0 ? operators : defaultConditionOperators;
  const [fieldName, setFieldName] = useState('');
  const [operator, setOperator] = useState<ProductConditionOperator>('gte');
  const [value, setValue] = useState('');

  useEffect(() => {
    if (!availableFields.some((field) => field.field === fieldName)) {
      setFieldName(availableFields[0]?.field ?? '');
      setValue('');
    }
  }, [availableFields, fieldName]);

  const selectedField = fields.find((field) => field.field === fieldName);
  const snapshotReady = snapshotStatus === 'ready';
  const addCondition = () => {
    if (!selectedField || !value.trim()) return;
    onAdd({ field: selectedField.field, operator, value: value.trim() });
    setValue('');
  };

  return (
    <div className="space-y-4 border-t border-slate-100 pt-4">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h3 className="text-sm font-semibold text-slate-800">日期与快照指标筛选</h3>
          <p className="mt-1 text-xs leading-5 text-slate-500">
            指标直接读取已生成的分析快照，不实时扫描完整净值历史；指标为空的产品不会按 0 处理。
          </p>
        </div>
        <span className={`self-start rounded-full px-3 py-1 text-xs font-semibold ${snapshotReady ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700'}`}>
          指标快照：{snapshotReady ? '可用' : '未就绪'}
        </span>
      </div>
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)_minmax(0,1fr)_auto]">
        <label className="space-y-1 text-xs font-medium text-slate-600">
          筛选字段
          <select
            aria-label="筛选字段"
            value={fieldName}
            onChange={(event) => { setFieldName(event.target.value); setValue(''); }}
            className="min-h-11 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
          >
            {fields.map((field) => (
              <option key={field.field} value={field.field} disabled={!field.available}>
                {field.label}{field.source === 'instrument_metrics_snapshot' ? '（快照）' : ''}{!field.available ? ' · 未就绪' : ''}
              </option>
            ))}
          </select>
        </label>
        <label className="space-y-1 text-xs font-medium text-slate-600">
          比较方式
          <select
            aria-label="比较方式"
            value={operator}
            onChange={(event) => setOperator(event.target.value as ProductConditionOperator)}
            className="min-h-11 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
          >
            {resolvedOperators.map((item) => (
              <option key={item.value} value={item.value}>{item.label}（{item.symbol}）</option>
            ))}
          </select>
        </label>
        <label className="space-y-1 text-xs font-medium text-slate-600">
          筛选值{selectedField?.unit_label ? `（${selectedField.unit_label}）` : ''}
          <input
            aria-label="筛选值"
            type={selectedField?.data_type === 'date' ? 'date' : 'number'}
            step={selectedField?.data_type === 'number' ? 'any' : undefined}
            value={value}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') { event.preventDefault(); addCondition(); }
            }}
            placeholder={selectedField?.unit_label === '%' ? '如 10 表示 10%' : '请输入数值'}
            className="min-h-11 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-700 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
          />
        </label>
        <button
          type="button"
          onClick={addCondition}
          disabled={!selectedField || !value.trim()}
          className="min-h-11 self-end rounded-lg bg-slate-900 px-4 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-500"
        >
          添加条件
        </button>
      </div>
      {conditions.length > 0 && (
        <div className="flex flex-wrap gap-2" aria-label="已添加的日期与指标条件">
          {conditions.map((condition, index) => (
            <button
              key={`${condition.field}-${condition.operator}-${condition.value}-${index}`}
              type="button"
              onClick={() => onRemove(index)}
              aria-label={`移除条件 ${formatProductCondition(condition, fields, resolvedOperators)}`}
              className="inline-flex items-center gap-2 rounded-full bg-violet-50 px-3 py-1 text-xs font-semibold text-violet-700 hover:bg-violet-100"
            >
              {formatProductCondition(condition, fields, resolvedOperators)}
              <span aria-hidden="true">×</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

interface SortButtonProps {
  label: string;
  columnKey: string;
  activeKey: string;
  direction: 'asc' | 'desc';
  onClick: (key: string) => void;
}

function SortButton({ label, columnKey, activeKey, direction, onClick }: SortButtonProps) {
  const isActive = activeKey === columnKey;
  return (
    <button
      type="button"
      onClick={() => onClick(columnKey)}
      className={`inline-flex items-center gap-1 text-xs font-semibold uppercase tracking-wider whitespace-nowrap ${isActive ? 'text-emerald-600' : 'text-slate-500'}`}
    >
      {label}
      <svg className={`h-3 w-3 ${isActive ? 'text-emerald-500' : 'text-slate-400'}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M8 15l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M16 9l-4-4-4 4" strokeLinecap="round" strokeLinejoin="round" />
        {isActive && (
          <path d={direction === 'asc' ? 'M12 5v14' : 'M12 5v14'} strokeLinecap="round" strokeLinejoin="round" />
        )}
      </svg>
    </button>
  );
}

interface MetricCardProps {
  title: string;
  value: string;
  description?: string;
}

function MetricCard({ title, value, description }: MetricCardProps) {
  return (
    <div className="rounded-2xl border border-transparent bg-gradient-to-br from-white via-slate-50 to-emerald-50 p-5 shadow-sm">
      <div className="text-xs font-semibold uppercase tracking-wide text-emerald-500">{title}</div>
      <div className="mt-2 text-2xl font-bold text-slate-900">{value}</div>
      {description && <div className="mt-1 text-xs text-slate-500">{description}</div>}
    </div>
  );
}
