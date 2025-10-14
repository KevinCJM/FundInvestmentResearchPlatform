import React, { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import FilterDropdown, { FilterOption } from '../components/FilterDropdown';

interface ProductItem {
  ts_code?: string | null;
  code?: string | null;
  name?: string | null;
  management?: string | null;
  custodian?: string | null;
  fund_type?: string | null;
  type?: string | null;
  invest_type?: string | null;
  market?: string | null;
  status?: string | null;
  issue_amount?: number | null;
  m_fee?: number | null;
  c_fee?: number | null;
  exp_return?: number | null;
  duration_year?: number | null;
  list_date?: string | null;
  found_date?: string | null;
  issue_date?: string | null;
}

interface ProductsSummary {
  universe_total: number;
  filtered_total: number;
  active_count?: number | null;
  recent_listings_12m?: number | null;
  avg_m_fee?: number | null;
  avg_c_fee?: number | null;
  avg_exp_return?: number | null;
  avg_duration_year?: number | null;
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
  sort_by: string;
  sort_dir: 'asc' | 'desc' | string;
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

const formatDuration = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  return `${decimalFormatter.format(value)} 年`;
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
  fund_type: '基金类型',
  type: '机构类型',
  invest_type: '投资风格',
  market: '交易市场',
  status: '产品状态',
  management: '管理人',
  custodian: '托管人',
};

type FilterState = {
  fund_type: string[];
  type: string[];
  invest_type: string[];
  market: string[];
  status: string[];
  management: string[];
  custodian: string[];
};

const initialFilterState: FilterState = {
  fund_type: [],
  type: [],
  invest_type: [],
  market: [],
  status: [],
  management: [],
  custodian: [],
};

const PAGE_SIZE_OPTIONS = [15, 30, 50];

export default function ProductResearch() {
  const [response, setResponse] = useState<ProductsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(PAGE_SIZE_OPTIONS[0]);
  const [filters, setFilters] = useState<FilterState>(initialFilterState);
  const [sortKey, setSortKey] = useState('issue_amount');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [searchInput, setSearchInput] = useState('');
  const [searchKeyword, setSearchKeyword] = useState('');

  useEffect(() => {
    const handler = window.setTimeout(() => {
      setSearchKeyword(searchInput.trim());
      setPage(1);
    }, 400);
    return () => window.clearTimeout(handler);
  }, [searchInput]);

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
        const resp = await fetch(`/api/etf/products?${params.toString()}`, { signal: controller.signal });
        if (!resp.ok) {
          if (resp.status === 404) {
            setResponse(null);
            setError('未找到ETF产品信息，请确认数据文件是否就绪。');
            return;
          }
          throw new Error('加载产品列表失败');
        }
        const data = (await resp.json()) as ProductsResponse;
        setResponse(data);
      } catch (err) {
        if ((err as DOMException).name === 'AbortError') {
          return;
        }
        console.error('Failed to load ETF products', err);
        setError('产品数据加载失败，请稍后重试。');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
    return () => controller.abort();
  }, [page, pageSize, sortKey, sortDir, searchKeyword, filters]);

  const summary = response?.summary;
  const totalPages = useMemo(() => {
    if (!response) {
      return 0;
    }
    return Math.max(1, Math.ceil(response.total / pageSize));
  }, [response, pageSize]);

  const handleFilterChange = (key: keyof FilterState) => (values: string[]) => {
    setFilters((prev) => ({ ...prev, [key]: values }));
    setPage(1);
  };

  const clearAllFilters = () => {
    setFilters(initialFilterState);
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
    setPage(1);
  };

  const toggleSort = (key: string) => {
    if (sortKey === key) {
      setSortDir((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir(key === 'list_date' ? 'desc' : 'asc');
    }
  };

  const appliedFiltersCount = activeFilterChips.length;

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <div className="mb-8 space-y-4">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">产品研究</h1>
          <p className="mt-2 text-base text-slate-600">
            浏览并筛选ETF产品，比较发行规模、费用结构与投资风格，为组合构建提供灵感支撑。
          </p>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard
            title="当前筛选产品"
            value={summary ? `${integerFormatter.format(summary.filtered_total)} / ${integerFormatter.format(summary.universe_total)}` : '--'}
            description="筛选结果 / 全部ETF"
          />
          <MetricCard
            title="有效存续产品"
            value={summary?.active_count !== undefined && summary?.active_count !== null ? integerFormatter.format(summary.active_count) : '--'}
            description={summary?.filtered_total ? `占比 ${decimalFormatter.format((summary.active_count ?? 0) / summary.filtered_total * 100)}%` : '存续状态估算'}
          />
          <MetricCard
            title="筛选合计发行规模"
            value={formatIssueAmount(summary?.total_issue_amount)}
            description="单位：按万转亿换算"
          />
          <MetricCard
            title="平均管理费 / 预期收益"
            value={`${formatPercent(summary?.avg_m_fee)} · ${formatPercent(summary?.avg_exp_return)}`}
            description="费用与收益均按筛选样本均值"
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
            label="基金类型"
            options={response?.available_filters?.fund_type ?? []}
            selected={filters.fund_type}
            onChange={handleFilterChange('fund_type')}
          />
          <FilterDropdown
            label="机构类型"
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
        <div className="flex items-center justify-between border-b border-slate-100 px-6 py-4">
          <h2 className="text-lg font-semibold text-slate-900">产品列表</h2>
          <div className="flex items-center gap-3 text-sm text-slate-500">
            <span>每页</span>
            <select
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
          </div>
        </div>

        {loading ? (
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
          <div className="px-6 py-24 text-center text-slate-500">暂无符合筛选条件的ETF。</div>
        ) : (
          <div className="h-[520px] w-full overflow-auto">
            <table className="min-w-[1200px] divide-y divide-slate-100">
              <thead className="bg-slate-50">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    产品
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    机构 / 类型
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    风格 / 市场
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    管理 / 托管
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    <SortButton label="发行规模" activeKey={sortKey} columnKey="issue_amount" direction={sortDir} onClick={toggleSort} />
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    <SortButton label="费用 / 预期" activeKey={sortKey} columnKey="m_fee" direction={sortDir} onClick={toggleSort} />
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    <SortButton label="上市日" activeKey={sortKey} columnKey="list_date" direction={sortDir} onClick={toggleSort} />
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                    状态
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white">
                {response?.items.map((item) => {
                  const code = item.ts_code ?? item.code ?? '--';
                  const detailPath = code && code !== '--' ? `/product/${encodeURIComponent(code)}` : undefined;
                  return (
                    <tr key={`${code}-${item.name}`} className="hover:bg-emerald-50/40">
                      <td className="px-6 py-4">
                        {detailPath ? (
                          <Link to={detailPath} target="_blank" rel="noreferrer" className="group block">
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
                        <div className="text-xs text-slate-400">{formatText(item.fund_type)}</div>
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
                        <div className="text-xs text-slate-400">中位规模：{formatIssueAmount(summary?.median_issue_amount)}</div>
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-600">
                        <div className="font-medium text-slate-700">管理费 {formatPercent(item.m_fee)} / 托管费 {formatPercent(item.c_fee)}</div>
                        <div className="text-xs text-slate-400">预期收益 {formatPercent(item.exp_return)} · 存续期 {formatDuration(item.duration_year)}</div>
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-600">
                        <div className="font-medium text-slate-700">{formatDate(item.list_date)}</div>
                        <div className="text-xs text-slate-400">成立：{formatDate(item.found_date ?? item.issue_date)}</div>
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
      </section>
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
      className={`inline-flex items-center gap-1 text-xs font-semibold uppercase tracking-wider ${isActive ? 'text-emerald-600' : 'text-slate-500'}`}
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
