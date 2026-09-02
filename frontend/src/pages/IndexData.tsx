import { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useDataRefresh } from '../components/dashboard/useDataRefresh';

interface IndexDatasetHealth {
  key: string;
  scope: string;
  file: string;
  exists: boolean;
  status: 'ready' | 'missing';
  rows: number;
  earliest_date?: string | null;
  latest_date?: string | null;
}

interface IndexSummary {
  schema_version: number;
  status: 'complete' | 'partial' | 'unavailable';
  catalog_count: number;
  source_count: number;
  covered_count: number;
  latest_date?: string | null;
  missing_count: number;
  stale_count: number;
  datasets: IndexDatasetHealth[];
}

interface IndexItem {
  source_api: string;
  quote_source_api?: string | null;
  ts_code: string;
  name?: string | null;
  category?: string | null;
  market?: string | null;
  publisher?: string | null;
  list_date?: string | null;
  exp_date?: string | null;
  status?: string | null;
  first_date?: string | null;
  latest_date?: string | null;
  rows?: number | null;
  coverage_status: 'ready' | 'stale' | 'missing';
}

interface IndexListResponse {
  schema_version: number;
  status: 'complete' | 'unavailable';
  page: number;
  page_size: number;
  total: number;
  items: IndexItem[];
  filters: Record<string, string[]>;
}

const integerFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 0 });
const scopeLabels: Record<string, string> = {
  domestic: '境内', industry: '行业', concept: '概念', global: '国际', futures: '期货',
  valuation: '估值', constituents: '成分权重', catalog: '目录', coverage: '覆盖快照',
};
const coverageLabels = { ready: '可用', stale: '陈旧', missing: '缺失' } as const;

export default function IndexData() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [summary, setSummary] = useState<IndexSummary | null>(null);
  const [catalog, setCatalog] = useState<IndexListResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const { status: refreshStatus } = useDataRefresh(() => undefined);
  const page = Math.max(Number(searchParams.get('page') ?? '1') || 1, 1);
  const pageSize = Math.min(Math.max(Number(searchParams.get('page_size') ?? '20') || 20, 10), 100);

  useEffect(() => {
    const controller = new AbortController();
    const params = new URLSearchParams(searchParams);
    params.set('page', String(page));
    params.set('page_size', String(pageSize));
    setLoading(true);
    Promise.all([
      fetch('/api/indices/summary', { signal: controller.signal }),
      fetch(`/api/indices?${params.toString()}`, { signal: controller.signal }),
    ])
      .then(async ([summaryResponse, catalogResponse]) => {
        if (!summaryResponse.ok || !catalogResponse.ok) {
          throw new Error('指数数据接口暂不可用。');
        }
        const [summaryPayload, catalogPayload] = await Promise.all([
          summaryResponse.json() as Promise<IndexSummary>,
          catalogResponse.json() as Promise<IndexListResponse>,
        ]);
        setSummary(summaryPayload);
        setCatalog(catalogPayload);
        setError(null);
      })
      .catch((requestError) => {
        if ((requestError as Error).name !== 'AbortError') {
          setError(requestError instanceof Error ? requestError.message : '指数数据加载失败。');
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [page, pageSize, searchParams]);

  const updateFilter = (key: string, value: string) => {
    const next = new URLSearchParams(searchParams);
    if (value) next.set(key, value); else next.delete(key);
    if (key !== 'page') next.set('page', '1');
    setSearchParams(next);
  };
  const totalPages = Math.max(Math.ceil((catalog?.total ?? 0) / pageSize), 1);
  const datasetGroups = useMemo(() => {
    const groups = new Map<string, IndexDatasetHealth[]>();
    (summary?.datasets ?? []).forEach((item) => groups.set(item.scope, [...(groups.get(item.scope) ?? []), item]));
    return [...groups.entries()].filter(([scope]) => !['catalog', 'coverage'].includes(scope));
  }, [summary?.datasets]);

  const kpis = [
    ['指数代码数', summary?.catalog_count],
    ['数据来源数', summary?.source_count],
    ['有行情代码数', summary?.covered_count],
    ['最新行情日', summary?.latest_date],
    ['缺失代码数', summary?.missing_count],
    ['陈旧代码数', summary?.stale_count],
  ];

  return (
    <div className="mx-auto min-w-0 max-w-7xl space-y-6 px-4 py-8 sm:px-6">
      <header className="flex flex-col gap-4 rounded-2xl bg-white p-6 shadow-sm sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-sm font-semibold text-indigo-600">Tushare 指数数据</p>
          <h1 className="mt-1 text-3xl font-bold text-slate-950">指数数据中心</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">浏览指数目录、行情覆盖和数据新鲜度，为后续市场状态识别与情景模拟准备可审计的数据底座。</p>
        </div>
        <div className="flex flex-col items-start gap-2 text-sm sm:items-end">
          <span role="status" aria-live="polite" className="rounded-full bg-slate-100 px-3 py-1.5 text-slate-700">
            Tushare：{refreshStatus?.job.status === 'running' ? '数据更新运行中' : refreshStatus?.job.message ?? '正在检查状态'}
          </span>
          <Link to="/" className="font-semibold text-indigo-700 hover:text-indigo-900">返回全景驾驶舱管理数据 →</Link>
        </div>
      </header>

      {error && <div role="alert" className="rounded-xl bg-rose-50 p-4 text-sm text-rose-700">{error}</div>}

      <section aria-label="指数数据概览" className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
        {kpis.map(([label, value]) => (
          <div key={String(label)} className="rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-100">
            <p className="text-xs font-semibold text-slate-500">{label}</p>
            <p className="mt-2 text-2xl font-bold tabular-nums text-slate-950">
              {value === null || value === undefined ? '--' : typeof value === 'number' ? integerFormatter.format(value) : value}
            </p>
          </div>
        ))}
      </section>

      <section aria-labelledby="index-health-title" className="rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-100">
        <h2 id="index-health-title" className="text-lg font-semibold text-slate-900">数据源健康</h2>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {datasetGroups.map(([scope, datasets]) => {
            const ready = datasets.filter((item) => item.exists).length;
            const rows = datasets.reduce((sum, item) => sum + (item.rows ?? 0), 0);
            return (
              <article key={scope} className="rounded-xl border border-slate-200 p-4">
                <div className="flex items-center justify-between gap-2"><h3 className="font-semibold text-slate-800">{scopeLabels[scope] ?? scope}</h3><span className={`h-2.5 w-2.5 rounded-full ${ready === datasets.length ? 'bg-emerald-500' : ready ? 'bg-amber-400' : 'bg-rose-500'}`} /></div>
                <p className="mt-2 text-sm tabular-nums text-slate-600">{ready}/{datasets.length} 个数据集 · {integerFormatter.format(rows)} 行</p>
                <p className="mt-1 text-xs tabular-nums text-slate-500">
                  {datasets.some((item) => item.earliest_date)
                    ? `${datasets.map((item) => item.earliest_date).filter(Boolean).sort()[0]} ～ ${datasets.map((item) => item.latest_date).filter(Boolean).sort().at(-1) ?? '--'}`
                    : '尚无日期覆盖'}
                </p>
              </article>
            );
          })}
        </div>
      </section>

      <section aria-labelledby="index-catalog-title" className="min-w-0 rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-100">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
          <div><h2 id="index-catalog-title" className="text-lg font-semibold text-slate-900">指数目录</h2><p className="mt-1 text-sm text-slate-500">共 {integerFormatter.format(catalog?.total ?? 0)} 条筛选结果</p></div>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-6">
            <label className="sm:col-span-2"><span className="sr-only">搜索指数</span><input aria-label="搜索指数" value={searchParams.get('q') ?? ''} onChange={(event) => updateFilter('q', event.target.value)} placeholder="代码、名称或发布方" className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500" /></label>
            {[
              ['source', '来源', catalog?.filters.source_api ?? []],
              ['category', '类别', catalog?.filters.category ?? []],
              ['market', '市场', catalog?.filters.market ?? []],
              ['coverage', '覆盖', ['ready', 'stale', 'missing']],
            ].map(([key, label, values]) => (
              <label key={String(key)}><span className="sr-only">{label}</span><select aria-label={String(label)} value={searchParams.get(String(key)) ?? ''} onChange={(event) => updateFilter(String(key), event.target.value)} className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"><option value="">全部{label}</option>{(values as string[]).map((value) => <option key={value} value={value}>{key === 'coverage' ? coverageLabels[value as keyof typeof coverageLabels] : value}</option>)}</select></label>
            ))}
            <label><span className="sr-only">存续状态</span><select aria-label="存续状态" value={searchParams.get('active') ?? ''} onChange={(event) => updateFilter('active', event.target.value)} className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"><option value="">全部状态</option><option value="active">存续</option><option value="inactive">终止</option></select></label>
          </div>
        </div>

        <div className="mt-4 max-w-full overflow-x-auto rounded-xl border border-slate-200" tabIndex={0} aria-label="指数目录表格滚动区域">
          <table className="min-w-[1320px] divide-y divide-slate-200 text-left text-sm">
            <caption className="sr-only">指数目录、来源与行情覆盖状态</caption>
            <thead className="bg-slate-50"><tr>{['代码', '名称', '类型', '来源', '市场', '发布方', '发布日期', '终止日期', '首个行情日', '最新行情日', '行数', '覆盖状态'].map((label) => <th key={label} scope="col" className="whitespace-nowrap px-3 py-3 text-xs font-semibold text-slate-600">{label}</th>)}</tr></thead>
            <tbody className="divide-y divide-slate-100">
              {catalog?.items.map((item) => <tr key={`${item.source_api}:${item.ts_code}`} className="hover:bg-slate-50"><td className="px-3 py-3 font-mono text-xs text-indigo-700">{item.ts_code}</td><td className="max-w-64 px-3 py-3 font-medium text-slate-800">{item.name ?? '--'}</td><td className="px-3 py-3 text-slate-600">{item.category ?? '--'}</td><td className="px-3 py-3 text-slate-600">{item.source_api}</td><td className="px-3 py-3 text-slate-600">{item.market ?? '--'}</td><td className="max-w-52 px-3 py-3 text-slate-600">{item.publisher ?? '--'}</td><td className="px-3 py-3 text-slate-600">{item.list_date ?? '--'}</td><td className="px-3 py-3 text-slate-600">{item.exp_date ?? '--'}</td><td className="px-3 py-3 text-slate-600">{item.first_date ?? '--'}</td><td className="px-3 py-3 text-slate-600">{item.latest_date ?? '--'}</td><td className="px-3 py-3 text-right tabular-nums text-slate-600">{item.rows === null || item.rows === undefined ? '--' : integerFormatter.format(item.rows)}</td><td className="px-3 py-3"><span className={`rounded-full px-2 py-1 text-xs font-semibold ${item.coverage_status === 'ready' ? 'bg-emerald-100 text-emerald-700' : item.coverage_status === 'stale' ? 'bg-amber-100 text-amber-800' : 'bg-rose-100 text-rose-700'}`}>{coverageLabels[item.coverage_status]}</span></td></tr>)}
              {!loading && !catalog?.items.length && <tr><td colSpan={12} className="px-4 py-12 text-center text-slate-500">没有符合筛选条件的指数。</td></tr>}
              {loading && <tr><td colSpan={12} className="px-4 py-12 text-center text-slate-500">正在加载指数目录...</td></tr>}
            </tbody>
          </table>
        </div>
        <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <label className="text-sm text-slate-600">每页 <select aria-label="每页数量" value={pageSize} onChange={(event) => updateFilter('page_size', event.target.value)} className="rounded-lg border border-slate-300 bg-white px-2 py-1.5"><option value="10">10</option><option value="20">20</option><option value="50">50</option><option value="100">100</option></select></label>
          <div className="flex items-center gap-2"><button type="button" disabled={page <= 1} onClick={() => updateFilter('page', String(page - 1))} className="rounded-lg border border-slate-300 px-3 py-2 text-sm disabled:opacity-40">上一页</button><span className="text-sm tabular-nums text-slate-600">{page} / {totalPages}</span><button type="button" disabled={page >= totalPages} onClick={() => updateFilter('page', String(page + 1))} className="rounded-lg border border-slate-300 px-3 py-2 text-sm disabled:opacity-40">下一页</button></div>
        </div>
      </section>
    </div>
  );
}
