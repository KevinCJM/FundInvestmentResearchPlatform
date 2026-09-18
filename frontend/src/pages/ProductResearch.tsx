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
import { Badge, Button, Card, EmptyState } from '../components/ui';
import { systemText as s, useI18n } from '../i18n/runtime';

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
    return s('productResearch.unknown');
  }
  return value;
};

/** 状态用状态色阶。品牌蓝只表示链接与选中，不表示"这只基金还活着"。 */
const statusTone = (status?: string | null): 'neutral' | 'success' | 'danger' => {
  if (!status) {
    return 'neutral';
  }
  if (/终止|退市|清盘|暂停/.test(status)) {
    return 'danger';
  }
  if (/存续|上市/.test(status)) {
    return 'success';
  }
  return 'neutral';
};

const filterLabels = (): Record<string, string> => ({
  fund_type: s('productResearch.filterFundType'),
  type: s('productResearch.filterType'),
  invest_type: s('productResearch.filterInvestType'),
  qdii_type: s('productResearch.filterQdiiType'),
  market: s('productResearch.filterMarket'),
  status: s('productResearch.filterStatus'),
  management: s('productResearch.filterManagement'),
  custodian: s('productResearch.filterCustodian'),
});

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
  useI18n();
  const [searchParams, setSearchParams] = useSearchParams();
  const [productKind, setProductKind] = useState<'etf' | 'fund'>(() => readProductKind(searchParams.get('kind')));
  const [response, setResponse] = useState<ProductsResponse | null>(null);
  const [loading, setLoading] = useState(true);
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
  const [reloadToken, setReloadToken] = useState(0);
  const [pageInput, setPageInput] = useState('1');
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
            setError(s('productResearch.errorNotFound', { kind: productKind === 'etf' ? s('productResearch.kindEtf') : s('productResearch.kindFund') }));
            return;
          }
          const payload = await resp.json().catch(() => null) as { detail?: string } | null;
          throw new Error(payload?.detail || s('productResearch.errorLoadFailed'));
        }
        const data = (await resp.json()) as ProductsResponse;
        assertFixedNjitExecution(data.execution, '产品研究统计');
        setResponse(data);
      } catch (err) {
        if ((err as DOMException).name === 'AbortError') {
          return;
        }
        console.error('Failed to load products', err);
        setError(err instanceof Error ? err.message : s('productResearch.errorGeneric'));
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    };
    fetchData();
    return () => controller.abort();
  }, [conditions, page, pageSize, sortKey, sortDir, searchKeyword, filters, productKind, snapshotMetrics, reloadToken]);

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
    }).catch(() => { if (active) setResearchError(s('productResearch.errorCatalog')); });
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
      .catch(() => { if (active) { setResearchResults([]); setResearchError(s('productResearch.errorMetrics')); } })
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
    setSearchInput('');
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
  const tableScrollRef = useRef<HTMLDivElement | null>(null);
  const [selectColOffset, setSelectColOffset] = useState<number>(0);
  const [tableOverflowing, setTableOverflowing] = useState(false);

  useLayoutEffect(() => {
    const updateOffset = () => {
      if (selectHeaderRef.current) {
        setSelectColOffset(selectHeaderRef.current.offsetWidth);
      }
      const box = tableScrollRef.current;
      setTableOverflowing(Boolean(box && box.scrollWidth > box.clientWidth + 1));
    };
    updateOffset();
    window.addEventListener('resize', updateOffset);
    return () => window.removeEventListener('resize', updateOffset);
  }, [response, pageSize, sortKey, sortDir, snapshotMetrics]);

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
    ? s('productResearch.compareHintAll')
    : selectedCount > 10
      ? s('productResearch.compareHintLimit')
      : undefined;

  const kindLabel = productKind === 'etf' ? s('productResearch.kindEtf') : s('productResearch.kindFund');
  const labels = filterLabels();
  const dateColumnKey = productKind === 'etf' ? 'list_date' : 'found_date';
  const pageSelectRef = useRef<HTMLInputElement | null>(null);
  const pageSelectionPartial = !currentPageAllSelected
    && currentPageSelectableProducts.some((item) => isProductSelected(item.id));
  // indeterminate 只能用 DOM 属性设置，React 没有对应的 prop。
  useEffect(() => {
    if (pageSelectRef.current) {
      pageSelectRef.current.indeterminate = pageSelectionPartial;
    }
  });
  useEffect(() => { setPageInput(String(page)); }, [page]);

  const commitPageInput = () => {
    const next = Number(pageInput);
    if (Number.isInteger(next) && next >= 1 && next <= totalPages) {
      setPage(next);
      return;
    }
    setPageInput(String(page));
  };
  const clearSelection = () => {
    setSelectedProducts({});
    setAllMatchingSelected(false);
    setExcludedProductIds(new Set());
  };
  const reload = () => setReloadToken((current) => current + 1);
  const ariaSort = (columnKey: string): 'ascending' | 'descending' | 'none' => (
    sortKey === columnKey ? (sortDir === 'asc' ? 'ascending' : 'descending') : 'none'
  );
  const headerBase = 'px-5 py-3 text-xs font-semibold whitespace-nowrap';
  const headerCell = `${headerBase} bg-slate-50 text-left text-slate-600`;
  const headerCellNumeric = `${headerBase} bg-slate-50 text-right text-slate-600`;
  const headerCellSnapshot = `${headerBase} bg-accent-50 text-right text-accent-700`;
  const toggleClass = (active: boolean) => `min-h-10 rounded-lg px-4 text-sm font-semibold transition focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 ${
    active ? 'bg-accent-600 text-white' : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
  }`;

  return (
    // 宽度、外边距和面包屑都由 StageLayout 提供，这里不再套第二层容器。
    <div className="space-y-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div className="min-w-0">
          <h1 className="text-2xl font-bold text-slate-900 sm:text-3xl">{s('productResearch.title')}</h1>
          <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-600">{s('productResearch.description')}</p>
        </div>
        <div role="group" aria-label={s('productResearch.kindGroup')} className="inline-flex shrink-0 self-start rounded-xl border border-slate-200 bg-white p-1">
          {(['etf', 'fund'] as const).map((kind) => (
            <button
              key={kind}
              type="button"
              aria-pressed={productKind === kind}
              onClick={() => switchProductKind(kind)}
              className={toggleClass(productKind === kind)}
            >
              {kind === 'etf' ? s('productResearch.kindEtf') : s('productResearch.kindFund')}
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title={s('productResearch.summaryFiltered')}
          value={summary ? `${integerFormatter.format(summary.filtered_total)} / ${integerFormatter.format(summary.universe_total)}` : '--'}
          description={s('productResearch.summaryFilteredHint', { kind: kindLabel })}
        />
        <MetricCard
          title={s('productResearch.summaryActive')}
          value={summary?.active_count !== undefined && summary?.active_count !== null ? integerFormatter.format(summary.active_count) : '--'}
          description={summary?.active_rate !== undefined && summary?.active_rate !== null
            ? s('productResearch.summaryActiveRate', { rate: formatSnapshotValue(summary.active_rate, 'ratio') })
            : s('productResearch.summaryActiveUnavailable')}
        />
        <MetricCard
          title={s('productResearch.summaryIssue')}
          value={formatIssueAmount(summary?.total_issue_amount)}
          description={s('productResearch.summaryIssueHint')}
        />
        <MetricCard
          title={s('productResearch.summaryFee')}
          value={`${formatPercent(summary?.avg_m_fee)} · ${formatPercent(summary?.avg_c_fee)}`}
          description={s('productResearch.summaryFeeHint')}
        />
      </div>

      <Card className="space-y-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <label className="flex w-full max-w-xl items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 focus-within:border-accent-500 focus-within:ring-1 focus-within:ring-accent-500">
            <span className="sr-only">{s('productResearch.searchLabel')}</span>
            <svg aria-hidden="true" className="h-5 w-5 shrink-0 text-slate-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="7" />
              <line x1="20" y1="20" x2="16.65" y2="16.65" />
            </svg>
            <input
              value={searchInput}
              onChange={(event) => setSearchInput(event.target.value)}
              placeholder={s('productResearch.searchPlaceholder')}
              className="min-h-10 w-full bg-transparent text-sm text-slate-700 outline-none placeholder:text-slate-600 placeholder:opacity-100"
            />
          </label>
          <div className="flex flex-wrap items-center gap-3">
            <Badge>{s('productResearch.appliedCount', { count: appliedFiltersCount })}</Badge>
            {appliedFiltersCount > 0 && (
              <Button onClick={clearAllFilters}>{s('productResearch.resetFilters')}</Button>
            )}
          </div>
        </div>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          <FilterDropdown label={labels.fund_type} options={response?.available_filters?.fund_type ?? []} selected={filters.fund_type} onChange={handleFilterChange('fund_type')} />
          <FilterDropdown label={labels.type} options={response?.available_filters?.type ?? []} selected={filters.type} onChange={handleFilterChange('type')} />
          <FilterDropdown label={labels.invest_type} options={response?.available_filters?.invest_type ?? []} selected={filters.invest_type} onChange={handleFilterChange('invest_type')} />
          <FilterDropdown label={labels.qdii_type} options={response?.available_filters?.qdii_type ?? []} selected={filters.qdii_type} onChange={handleFilterChange('qdii_type')} />
          <FilterDropdown label={labels.market} options={response?.available_filters?.market ?? []} selected={filters.market} onChange={handleFilterChange('market')} />
          <FilterDropdown label={labels.status} options={response?.available_filters?.status ?? []} selected={filters.status} onChange={handleFilterChange('status')} />
          <FilterDropdown label={labels.management} options={response?.available_filters?.management ?? []} selected={filters.management} onChange={handleFilterChange('management')} />
          <FilterDropdown label={labels.custodian} options={response?.available_filters?.custodian ?? []} selected={filters.custodian} onChange={handleFilterChange('custodian')} />
        </div>
        <ProductConditionBuilder
          fields={conditionFields}
          operators={conditionOperators}
          conditions={conditions}
          snapshotStatus={response?.snapshot?.status}
          onAdd={addCondition}
          onRemove={removeCondition}
        />
        {/* 研究页面按研究日重算；这张筛选表读的是全历史快照，差异必须写在界面上。 */}
        {response?.pit?.snapshot_is_hindsight && (
          <p
            className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-900"
            data-testid="product-research-snapshot-hindsight"
          >
            {response.pit.warnings?.[0]}
          </p>
        )}
        {activeFilterChips.length > 0 && (
          <div className="flex flex-wrap gap-2 border-t border-slate-200 pt-4">
            {activeFilterChips.map((chip) => (
              <button
                key={`${chip.key}-${chip.value}`}
                type="button"
                onClick={() => removeChip(chip.key, chip.value)}
                aria-label={s('productResearch.removeFilter', { label: `${labels[chip.key] ?? chip.key} ${chip.label}` })}
                className="inline-flex min-h-10 items-center gap-2 rounded-full bg-slate-100 px-3 text-xs font-semibold text-slate-700 transition hover:bg-slate-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500"
              >
                <span className="text-slate-600">{labels[chip.key] ?? chip.key}</span>
                {chip.label}
                <span aria-hidden="true" className="text-slate-600">×</span>
              </button>
            ))}
          </div>
        )}
      </Card>

      {/* 与 Card 同一组令牌；这里不用 Card 是因为表格要贴边，而 Card 的内距不可覆盖。 */}
      <section className="rounded-xl border border-slate-200 bg-white shadow-sm">
        <div className="flex flex-col gap-3 border-b border-slate-200 px-5 py-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0">
            <h2 className="text-lg font-semibold text-slate-900">
              {viewMode === 'basic' ? s('productResearch.resultTitleBasic') : s('productResearch.resultTitleMetrics')}
            </h2>
            <p className="mt-1 text-xs leading-5 text-slate-600">
              {viewMode === 'basic' ? (
                <>
                  <span className="font-semibold text-accent-700">{s('productResearch.basicHintLink')}</span>
                  <span aria-hidden="true" className="mx-1">·</span>
                  <span>{s('productResearch.basicHintSelection')}</span>
                </>
              ) : s('productResearch.metricsHint')}
            </p>
          </div>
          <div role="group" aria-label={s('productResearch.viewGroup')} className="inline-flex shrink-0 self-start rounded-xl border border-slate-200 bg-white p-1">
            <button type="button" aria-pressed={viewMode === 'basic'} onClick={() => setViewMode('basic')} className={toggleClass(viewMode === 'basic')}>
              {s('productResearch.viewBasic')}
            </button>
            <button type="button" aria-pressed={viewMode === 'metrics'} onClick={() => setViewMode('metrics')} className={toggleClass(viewMode === 'metrics')}>
              {s('productResearch.viewMetrics')}
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-slate-200 bg-slate-50 px-5 py-3 text-sm text-slate-600">
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
          {viewMode === 'metrics' && (
            <>
              <MetricSelector
                indicators={researchIndicators}
                selectedIds={researchPreference.indicatorIds}
                onChange={(indicatorIds) => setResearchPreference((current) => withSelectedIndicators(current, indicatorIds, '1Y'))}
                maxSelected={5}
                label={s('productResearch.metricSelectorLabel')}
              />
              <label className="inline-flex items-center gap-2 text-xs font-medium text-slate-600">
                {s('productResearch.asOfLabel')}
                <input
                  type="date"
                  value={researchAsOf}
                  onChange={(event) => setResearchAsOf(event.target.value)}
                  className="min-h-10 rounded-lg border border-slate-200 px-3 text-sm text-slate-700 focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500"
                />
              </label>
            </>
          )}
          <span className="ml-auto inline-flex items-center gap-2">
            <span>{s('productResearch.pageSizeLabel')}</span>
            <select
              aria-label={s('productResearch.pageSizeAria')}
              value={pageSize}
              onChange={(event) => {
                setPageSize(Number(event.target.value));
                setPage(1);
              }}
              className="min-h-10 rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-700 focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500"
            >
              {PAGE_SIZE_OPTIONS.map((size) => (
                <option key={size} value={size}>{size}</option>
              ))}
            </select>
          </span>
          <span role="status" className="tabular-nums">{s('productResearch.totalCount', { count: integerFormatter.format(response?.total ?? 0) })}</span>
        </div>

        {viewMode === 'metrics' ? (
          <div className="p-4">
            {loading || researchLoading ? (
              <TableSkeleton columns={Math.max(2, selectedResearchIndicators.length + 1)} label={s('productResearch.metricsLoading')} />
            ) : error || researchError ? (
              <ResultError message={error ?? researchError ?? ''} onRetry={reload} />
            ) : currentPageTargets.length === 0 ? (
              <p className="py-16 text-center text-sm text-slate-600">{s('productResearch.metricsEmpty')}</p>
            ) : (
              <MetricMatrix
                indicators={selectedResearchIndicators}
                targets={currentPageTargets}
                results={researchResults}
                periodsByIndicator={researchPreference.periodsByIndicator}
                periodOptions={researchPeriods}
                onPeriodChange={(indicatorId, period) => setResearchPreference((current) => ({ ...current, periodsByIndicator: { ...current.periodsByIndicator, [indicatorId]: period } }))}
                onDefinition={setDefinitionIndicator}
              />
            )}
          </div>
        ) : loading ? (
          <TableSkeleton columns={6 + selectedSnapshotMetricFields.length} label={s('productResearch.loading')} />
        ) : error ? (
          <ResultError message={error} onRetry={reload} />
        ) : response && response.items.length === 0 ? (
          <div className="px-5 py-10">
            <EmptyState
              mascot={false}
              title={s('productResearch.emptyTitle', { kind: kindLabel })}
              hint={s('productResearch.emptyHint')}
              action={appliedFiltersCount > 0 || searchKeyword
                ? <Button tone="primary" onClick={clearAllFilters}>{s('productResearch.emptyAction')}</Button>
                : undefined}
            />
          </div>
        ) : (
          <>
            {tableOverflowing && <p className="px-5 pt-3 text-xs text-slate-600">{s('productResearch.scrollHint')}</p>}
            <div ref={tableScrollRef} className="w-full overflow-x-auto">
              <table className="w-full min-w-[1120px] divide-y divide-slate-200">
                <caption className="sr-only">
                  {s('productResearch.tableCaption', { kind: kindLabel, count: integerFormatter.format(response?.total ?? 0), page })}
                </caption>
                <thead>
                  <tr>
                    <th scope="col" ref={selectHeaderRef} className={`sticky left-0 z-20 w-[64px] min-w-[64px] border-r border-slate-200 ${headerCell}`}>
                      <input
                        ref={pageSelectRef}
                        type="checkbox"
                        className="h-4 w-4 rounded-lg border-slate-300 text-accent-600 focus:ring-accent-500"
                        aria-label={s('productResearch.selectAllPage')}
                        checked={currentPageAllSelected}
                        disabled={currentPageSelectableProducts.length === 0}
                        onChange={toggleCurrentPageSelection}
                      />
                    </th>
                    <th scope="col" style={{ left: productLeft }} className={`sticky z-20 border-r border-slate-200 ${headerCell}`}>
                      {s('productResearch.colProduct')}
                    </th>
                    <th scope="col" className={headerCell}>{s('productResearch.colType')}</th>
                    <th scope="col" className={headerCell}>{s('productResearch.colStyle')}</th>
                    <th scope="col" className={headerCell}>{s('productResearch.colManager')}</th>
                    <th scope="col" aria-sort={ariaSort('issue_amount')} className={headerCellNumeric}>
                      <SortButton label={s('productResearch.colIssue')} activeKey={sortKey} columnKey="issue_amount" direction={sortDir} onClick={toggleSort} />
                    </th>
                    {selectedSnapshotMetricFields.map((field) => (
                      <th key={field.field} scope="col" title={field.description} className={headerCellSnapshot}>
                        {field.label}
                        <span className="ml-1 font-normal">{s('productResearch.snapshotTag')}</span>
                      </th>
                    ))}
                    <th scope="col" aria-sort={ariaSort('m_fee')} className={headerCell}>
                      <SortButton label={s('productResearch.colFee')} activeKey={sortKey} columnKey="m_fee" direction={sortDir} onClick={toggleSort} />
                    </th>
                    <th scope="col" aria-sort={ariaSort(dateColumnKey)} className={headerCell}>
                      <SortButton
                        label={productKind === 'etf' ? s('productResearch.colListDate') : s('productResearch.colFoundDate')}
                        activeKey={sortKey}
                        columnKey={dateColumnKey}
                        direction={sortDir}
                        onClick={toggleSort}
                      />
                    </th>
                    <th scope="col" className={headerCell}>{s('productResearch.colStatus')}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 bg-white">
                  {response?.items.map((item) => {
                    const code = item.ts_code ?? item.code ?? '--';
                    const detailPath = code && code !== '--'
                      ? `/product-research/products/${encodeURIComponent(code)}?kind=${productKind}`
                      : undefined;
                    const selectionId = item.ts_code ?? item.code ?? null;
                    const isSelected = selectionId ? isProductSelected(selectionId) : false;
                    return (
                      <tr key={`${code}-${item.name}`} className={`group ${isSelected ? 'bg-accent-50' : 'hover:bg-slate-50'}`}>
                        <td className={`sticky left-0 z-10 w-[64px] min-w-[64px] border-r border-slate-200 px-5 py-4 whitespace-nowrap ${isSelected ? 'bg-accent-50' : 'bg-white group-hover:bg-slate-50'}`}>
                          <input
                            type="checkbox"
                            className="h-4 w-4 rounded-lg border-slate-300 text-accent-600 focus:ring-accent-500"
                            checked={isSelected}
                            disabled={!selectionId}
                            aria-label={s('productResearch.selectRow', { name: item.name ?? code })}
                            onChange={() => toggleProductSelection(selectionId, item.name, code)}
                          />
                        </td>
                        <td
                          className={`sticky z-10 border-r border-slate-200 px-5 py-4 ${isSelected ? 'bg-accent-50' : 'bg-white group-hover:bg-slate-50'}`}
                          style={{ left: productLeft }}
                        >
                          {detailPath ? (
                            <Link
                              to={detailPath}
                              target="_blank"
                              rel="noreferrer"
                              aria-label={s('productResearch.productLinkAria', { name: item.name ?? code })}
                              title={s('productResearch.productLinkTitle')}
                              className="block rounded-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 focus-visible:ring-offset-2"
                            >
                              <span className="block text-sm font-semibold text-accent-700 underline-offset-2 group-hover:underline">{item.name ?? '--'}</span>
                              <span className="mt-1 block text-xs tabular-nums text-slate-600">{code}</span>
                            </Link>
                          ) : (
                            <div>
                              <div className="text-sm font-semibold text-slate-900">{item.name ?? '--'}</div>
                              <div className="mt-1 text-xs tabular-nums text-slate-600">{code}</div>
                            </div>
                          )}
                        </td>
                        <td className="px-5 py-4 text-sm text-slate-600">
                          <div className="font-medium text-slate-700">{formatText(item.type)}</div>
                          <div className="mt-1 flex items-center gap-2 text-xs text-slate-600">
                            <span>{formatText(item.fund_type)}</span>
                            {item.qdii_type && <Badge>{item.qdii_type}</Badge>}
                          </div>
                        </td>
                        <td className="px-5 py-4 text-sm text-slate-600">
                          <div className="font-medium text-slate-700">{formatText(item.invest_type)}</div>
                          <div className="text-xs text-slate-600">{formatText(item.market)}</div>
                        </td>
                        <td className="px-5 py-4 text-sm text-slate-600">
                          <div className="font-medium text-slate-700">{formatText(item.management)}</div>
                          <div className="text-xs text-slate-600">{formatText(item.custodian)}</div>
                        </td>
                        <td className="px-5 py-4 text-right text-sm text-slate-600">
                          <div className="font-semibold tabular-nums text-slate-800">{formatIssueAmount(item.issue_amount)}</div>
                          <div className="text-xs text-slate-600">{s('productResearch.issueHint')}</div>
                        </td>
                        {selectedSnapshotMetricFields.map((field) => {
                          const value = item.snapshot_values?.[field.field];
                          const asOf = item.snapshot_value_dates?.[field.field];
                          const snapshotStatus = item.snapshot_statuses?.[field.field];
                          const snapshotWarning = item.snapshot_warnings?.[field.field];
                          return (
                            <td key={field.field} className="px-5 py-4 text-right text-sm text-slate-600">
                              <div className="font-semibold tabular-nums text-slate-800">{formatSnapshotValue(value, field.unit)}</div>
                              <div className="text-xs text-slate-600">
                                {snapshotWarning
                                  ? snapshotWarning
                                  : asOf
                                    ? s('productResearch.snapshotAsOf', { date: formatDate(asOf) })
                                    : snapshotStatus === 'unavailable'
                                      ? s('productResearch.snapshotUnavailable')
                                      : s('productResearch.snapshotEmpty')}
                              </div>
                            </td>
                          );
                        })}
                        <td className="px-5 py-4 text-sm text-slate-600">
                          <div className="font-medium tabular-nums text-slate-700">
                            {s('productResearch.feeCell', { management: formatPercent(item.m_fee), custody: formatPercent(item.c_fee) })}
                          </div>
                          <div className="text-xs text-slate-600">
                            {s('productResearch.feeBenchmark', { benchmark: formatText(item.benchmark), type: formatText(item.fund_type) })}
                          </div>
                        </td>
                        <td className="px-5 py-4 text-sm text-slate-600">
                          <div className="font-medium tabular-nums text-slate-700">{formatDate(productKind === 'etf' ? item.list_date : item.found_date)}</div>
                          <div className="text-xs tabular-nums text-slate-600">
                            {productKind === 'etf'
                              ? s('productResearch.foundedOn', { date: formatDate(item.found_date ?? item.issue_date) })
                              : s('productResearch.issuedOn', { date: formatDate(item.issue_date) })}
                          </div>
                        </td>
                        <td className="px-5 py-4">
                          <Badge tone={statusTone(item.status)}>{formatText(item.status)}</Badge>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}

        {response && response.items.length > 0 && (
          <nav aria-label={s('productResearch.paginationLabel')} className="flex flex-col gap-3 border-t border-slate-200 px-5 py-3 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-sm tabular-nums text-slate-600">{s('productResearch.pageStatus', { page, total: totalPages })}</p>
            <div className="flex flex-wrap items-center gap-2">
              <Button onClick={() => setPage(1)} disabled={page === 1}>{s('productResearch.firstPage')}</Button>
              <Button onClick={() => setPage((prev) => Math.max(1, prev - 1))} disabled={page === 1}>{s('productResearch.prevPage')}</Button>
              <input
                type="number"
                min={1}
                max={totalPages}
                inputMode="numeric"
                aria-label={s('productResearch.gotoPage')}
                value={pageInput}
                onChange={(event) => setPageInput(event.target.value)}
                onBlur={commitPageInput}
                onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); commitPageInput(); } }}
                className="min-h-10 w-20 rounded-lg border border-slate-200 px-3 text-sm tabular-nums text-slate-700 focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500"
              />
              <Button onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))} disabled={page >= totalPages}>{s('productResearch.nextPage')}</Button>
              <Button onClick={() => setPage(totalPages)} disabled={page >= totalPages}>{s('productResearch.lastPage')}</Button>
            </div>
          </nav>
        )}

        {selectedCount > 0 && (
          // 选中反馈只保留这一处：跟随滚动，不需要翻到表尾才看得到已选了什么。
          <div
            role="region"
            aria-label={s('productResearch.selectionBarLabel')}
            className="sticky bottom-0 z-10 flex flex-col gap-3 rounded-b-xl border-t border-slate-200 bg-white px-5 py-3 shadow-sm sm:flex-row sm:items-center sm:justify-between"
          >
            <div className="flex min-w-0 flex-wrap items-center gap-2 text-sm text-slate-600">
              <Badge>{s('productResearch.selectedCount', { count: integerFormatter.format(selectedCount) })}</Badge>
              {allMatchingSelected ? (
                <span>
                  {excludedProductIds.size > 0
                    ? s('productResearch.selectionAllMatchingExcluded', { count: excludedProductIds.size })
                    : s('productResearch.selectionAllMatching')}
                </span>
              ) : (
                <span className="min-w-0 truncate">
                  {selectedList.slice(0, 3).map((item) => item.name).join('、')}
                  {selectedList.length > 3 ? s('productResearch.selectionMore', { count: selectedList.length - 3 }) : ''}
                </span>
              )}
              {selectionActionHint && <span role="status" className="text-amber-800">{selectionActionHint}</span>}
            </div>
            <div className="flex shrink-0 flex-wrap items-center gap-2">
              {allMatchingSelected ? (
                <Button onClick={toggleAllMatchingSelection}>{s('productResearch.clearAllMatching')}</Button>
              ) : (
                <>
                  <Button onClick={toggleAllMatchingSelection} disabled={!response || response.total === 0}>
                    {s('productResearch.selectAllMatching', { count: integerFormatter.format(response?.total ?? 0) })}
                  </Button>
                  <Button onClick={clearSelection}>{s('productResearch.clearSelection')}</Button>
                </>
              )}
              <Button tone="primary" onClick={goToComparison} disabled={!canCompareSelection} title={selectionActionHint}>
                {s('productResearch.compare')}
              </Button>
            </div>
          </div>
        )}
      </section>
      <MetricDefinitionDrawer indicator={definitionIndicator} onClose={() => setDefinitionIndicator(null)} />
    </div>
  );
}

/** 骨架屏的行列数跟随真实表格，避免加载完成时布局跳一下。 */
function TableSkeleton({ columns, label }: { columns: number; label: string }) {
  return (
    <div role="status" aria-live="polite" className="space-y-3 px-5 py-5">
      <span className="sr-only">{label}</span>
      {Array.from({ length: 6 }).map((_, row) => (
        <div key={row} className="flex animate-pulse items-center gap-4 motion-reduce:animate-none">
          {Array.from({ length: columns }).map((_, column) => (
            <div key={column} className={`h-9 rounded-lg bg-slate-100 ${column === 0 ? 'w-16 shrink-0' : 'flex-1'}`} />
          ))}
        </div>
      ))}
    </div>
  );
}

/** 请求失败必须给重试入口，只说"请稍后再试"等于把重试交给整页刷新。 */
function ResultError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div role="alert" className="px-5 py-16">
      <div className="mx-auto max-w-md space-y-3 text-center">
        <p className="text-base font-semibold text-slate-900">{s('productResearch.errorTitle')}</p>
        <p className="text-sm leading-6 text-slate-600">{message}</p>
        <p className="text-sm leading-6 text-slate-600">{s('productResearch.errorHelp')}</p>
        <Button tone="primary" onClick={onRetry}>{s('productResearch.retry')}</Button>
      </div>
    </div>
  );
}

const defaultConditionOperators = (): ProductConditionOperatorOption[] => [
  { value: 'gte', label: s('productResearch.operatorGte'), symbol: '≥' },
  { value: 'lte', label: s('productResearch.operatorLte'), symbol: '≤' },
  { value: 'gt', label: s('productResearch.operatorGt'), symbol: '>' },
  { value: 'lt', label: s('productResearch.operatorLt'), symbol: '<' },
  { value: 'eq', label: s('productResearch.operatorEq'), symbol: '=' },
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
  const resolvedOperators = operators.length > 0 ? operators : defaultConditionOperators();
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
  const controlClass = 'min-h-10 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-700 placeholder:text-slate-600 placeholder:opacity-100 focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500';

  return (
    <div className="space-y-4 border-t border-slate-200 pt-4">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h3 className="text-sm font-semibold text-slate-800">{s('productResearch.conditionTitle')}</h3>
          <p className="mt-1 max-w-3xl text-xs leading-5 text-slate-600">{s('productResearch.conditionHint')}</p>
        </div>
        <span className="shrink-0 self-start">
          <Badge tone={snapshotReady ? 'success' : 'warning'}>
            {snapshotReady ? s('productResearch.snapshotReady') : s('productResearch.snapshotNotReady')}
          </Badge>
        </span>
      </div>
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)_minmax(0,1fr)_auto]">
        <label className="space-y-1 text-xs font-medium text-slate-600">
          {s('productResearch.conditionField')}
          <select
            aria-label={s('productResearch.conditionField')}
            value={fieldName}
            onChange={(event) => { setFieldName(event.target.value); setValue(''); }}
            className={controlClass}
          >
            {fields.map((field) => (
              <option key={field.field} value={field.field} disabled={!field.available}>
                {field.label}
                {field.source === 'instrument_metrics_snapshot' ? s('productResearch.conditionSnapshotSuffix') : ''}
                {!field.available ? s('productResearch.conditionUnavailableSuffix') : ''}
              </option>
            ))}
          </select>
        </label>
        <label className="space-y-1 text-xs font-medium text-slate-600">
          {s('productResearch.conditionOperator')}
          <select
            aria-label={s('productResearch.conditionOperator')}
            value={operator}
            onChange={(event) => setOperator(event.target.value as ProductConditionOperator)}
            className={controlClass}
          >
            {resolvedOperators.map((item) => (
              <option key={item.value} value={item.value}>{item.label}（{item.symbol}）</option>
            ))}
          </select>
        </label>
        <label className="space-y-1 text-xs font-medium text-slate-600">
          {selectedField?.unit_label
            ? s('productResearch.conditionValueUnit', { unit: selectedField.unit_label })
            : s('productResearch.conditionValue')}
          <input
            aria-label={s('productResearch.conditionValue')}
            type={selectedField?.data_type === 'date' ? 'date' : 'number'}
            step={selectedField?.data_type === 'number' ? 'any' : undefined}
            value={value}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') { event.preventDefault(); addCondition(); }
            }}
            placeholder={selectedField?.unit_label === '%' ? s('productResearch.conditionPlaceholderPercent') : s('productResearch.conditionPlaceholderNumber')}
            className={controlClass}
          />
        </label>
        <Button
          tone="primary"
          className="self-end"
          onClick={addCondition}
          disabled={!selectedField || !value.trim()}
          title={!selectedField || !value.trim() ? s('productResearch.conditionValue') : undefined}
        >
          {s('productResearch.conditionAdd')}
        </Button>
      </div>
      {conditions.length > 0 && (
        <div className="flex flex-wrap gap-2" aria-label={s('productResearch.conditionListLabel')}>
          {conditions.map((condition, index) => (
            <button
              key={`${condition.field}-${condition.operator}-${condition.value}-${index}`}
              type="button"
              onClick={() => onRemove(index)}
              aria-label={s('productResearch.conditionRemove', { condition: formatProductCondition(condition, fields, resolvedOperators) })}
              className="inline-flex min-h-10 items-center gap-2 rounded-full bg-slate-100 px-3 text-xs font-semibold text-slate-700 transition hover:bg-slate-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500"
            >
              {formatProductCondition(condition, fields, resolvedOperators)}
              <span aria-hidden="true" className="text-slate-600">×</span>
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
      aria-label={s('productResearch.sortAria', { label })}
      className={`inline-flex min-h-10 items-center gap-1 whitespace-nowrap rounded-lg text-xs font-semibold transition focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 ${isActive ? 'text-accent-700' : 'text-slate-600 hover:text-slate-900'}`}
    >
      {label}
      {/* 升序和降序必须是两个不同的图形；只换颜色等于没有方向指示。 */}
      <svg aria-hidden="true" className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        {isActive
          ? <path d={direction === 'asc' ? 'M12 19V5m-5 5l5-5 5 5' : 'M12 5v14m-5-5l5 5 5-5'} />
          : <path d="M8 10l4-4 4 4M8 14l4 4 4-4" />}
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
    <Card as="div">
      <p className="text-xs font-semibold text-slate-600">{title}</p>
      <p className="mt-2 text-2xl font-bold tabular-nums text-slate-900">{value}</p>
      {description && <p className="mt-1 text-xs leading-5 text-slate-600">{description}</p>}
    </Card>
  );
}
