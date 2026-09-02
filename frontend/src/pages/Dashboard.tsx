import { useEffect, useMemo } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import DashboardFilterBar from '../components/dashboard/DashboardFilters';
import DashboardOverviewCards from '../components/dashboard/DashboardOverviewCards';
import DashboardRankingTable from '../components/dashboard/DashboardRankingTable';
import DashboardResearchSearch from '../components/dashboard/DashboardResearchSearch';
import DashboardScopeTabs from '../components/dashboard/DashboardScopeTabs';
import { DistributionChartCard, TrendChartCard } from '../components/dashboard/DashboardChartCard';
import DataHealthRefreshPanel from '../components/dashboard/DataHealthRefreshPanel';
import type {
  DashboardFilterKey,
  DashboardFilters,
  DashboardKind,
  DashboardSegment,
  SegmentKind,
} from '../components/dashboard/types';
import { useInstrumentDashboard } from '../components/dashboard/useInstrumentDashboard';

const integerFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 0 });
const percentFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 1 });

const filterKeys: DashboardFilterKey[] = ['fund_type', 'invest_type', 'status', 'management', 'market'];
const isDashboardFilterKey = (value: string): value is DashboardFilterKey => (
  filterKeys.includes(value as DashboardFilterKey)
);

const commonMetrics = [
  ['return_1m', '近1月收益率'],
  ['return_3m', '近3月收益率'],
  ['return_1y', '近1年收益率'],
  ['return_3y', '近3年收益率'],
  ['annual_volatility_1y', '近1年年化波动率'],
  ['max_drawdown_3y', '近3年最大回撤'],
  ['sharpe_1y', '近1年 Sharpe'],
  ['calmar_3y', '近3年 Calmar'],
  ['issue_amount', '披露发行规模'],
  ['m_fee', '管理费率'],
  ['c_fee', '托管费率'],
] as const;

const etfMetrics = [
  ['amount_avg_20d', '近20日平均成交额'],
  ['volume_avg_20d', '近20日平均成交量'],
  ['premium_discount_latest', '最新价格净值偏离'],
] as const;

const returnPeriods = [
  ['return_3m', '3个月'],
  ['return_1y', '1年'],
  ['return_3y', '3年'],
] as const;

const segmentLabels: Record<SegmentKind, string> = {
  etf: 'ETF',
  fund: '场外公募基金',
};

const emptyFilters = (): DashboardFilters => ({
  fund_type: [],
  invest_type: [],
  status: [],
  management: [],
  market: [],
});

const parseKind = (value: string | null): DashboardKind => (
  value === 'etf' || value === 'fund' || value === 'all' ? value : 'all'
);

const selectedSegments = (kind: DashboardKind): SegmentKind[] => (
  kind === 'all' ? ['etf', 'fund'] : [kind]
);

const formatPercent = (value?: number | null) => value === null || value === undefined
  ? '--'
  : `${percentFormatter.format(value * 100)}%`;

const researchUrl = (kind: SegmentKind, filters: DashboardFilters, extra?: { key: DashboardFilterKey; value: string }) => {
  const params = new URLSearchParams({
    kind,
    sort_by: kind === 'etf' ? 'list_date' : 'found_date',
    sort_dir: 'desc',
  });
  filterKeys.forEach((key) => {
    filters[key].forEach((value) => params.append(key, value));
  });
  if (extra && !params.getAll(extra.key).includes(extra.value)) {
    params.append(extra.key, extra.value);
  }
  return `/research?${params.toString()}`;
};

interface SegmentLensProps {
  kind: SegmentKind;
  segment?: DashboardSegment;
  filters: DashboardFilters;
  loading?: boolean;
}

function SegmentLens({ kind, segment, filters, loading = false }: SegmentLensProps) {
  const label = segmentLabels[kind];
  if (loading && !segment) {
    return <section aria-label={`${label}市场镜头加载中`} className="min-w-0 h-52 animate-pulse rounded-2xl bg-white shadow-sm ring-1 ring-slate-100" />;
  }
  if (!segment || segment.availability === 'missing') {
    return (
      <section className="min-w-0 rounded-2xl border border-dashed border-amber-300 bg-amber-50 p-6">
        <h2 className="text-lg font-semibold text-amber-900">{label}数据尚未就绪</h2>
        <p className="mt-2 text-sm leading-6 text-amber-800">该品类会局部降级，不影响另一类产品的结构与排行。可在上方“数据管理”中更新对应模块。</p>
      </section>
    );
  }
  const summary = segment.summary;
  const latestProducts = segment.latest_products ?? [];
  const specialistMetrics = kind === 'etf'
    ? [
        { label: '指数标的覆盖率', value: formatPercent(summary.index_coverage_rate) },
        { label: '20日流动性覆盖率', value: formatPercent(summary.liquidity_coverage_rate) },
        { label: '最新交易行情日', value: summary.latest_candle_date ?? '--' },
      ]
    : [
        { label: '申赎起始日覆盖率', value: formatPercent(summary.purchase_redemption_coverage_rate) },
        { label: '最新净值日', value: summary.latest_nav_date ?? '--' },
      ];
  return (
    <section className="min-w-0 rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-100">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-indigo-500">{kind === 'etf' ? 'Exchange traded' : 'Off-exchange funds'}</p>
          <h2 className="mt-1 text-xl font-semibold text-slate-900">{label}市场镜头</h2>
          <p className="mt-1 text-sm text-slate-500">全部数量均按产品代码统计，同一母基金的不同份额类别分别计数。</p>
        </div>
        <Link to={researchUrl(kind, filters)} className="shrink-0 rounded-xl bg-indigo-50 px-4 py-2 text-sm font-semibold text-indigo-700 hover:bg-indigo-100">进入产品研究 →</Link>
      </div>
      <dl className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        <div className="rounded-xl bg-slate-50 p-4"><dt className="text-xs text-slate-500">产品代码</dt><dd className="mt-1 text-xl font-semibold tabular-nums text-slate-900">{integerFormatter.format(summary.share_code_count)}</dd></div>
        <div className="rounded-xl bg-slate-50 p-4"><dt className="text-xs text-slate-500">{kind === 'etf' ? '上市交易中' : '存续中'}</dt><dd className="mt-1 text-xl font-semibold tabular-nums text-slate-900">{integerFormatter.format(summary.active_count)}</dd></div>
        <div className="rounded-xl bg-slate-50 p-4"><dt className="text-xs text-slate-500">管理人</dt><dd className="mt-1 text-xl font-semibold tabular-nums text-slate-900">{integerFormatter.format(summary.unique_managements)}</dd></div>
        <div className="rounded-xl bg-slate-50 p-4"><dt className="text-xs text-slate-500">净值覆盖率</dt><dd className="mt-1 text-xl font-semibold tabular-nums text-slate-900">{formatPercent(summary.nav_coverage_rate)}</dd></div>
        {specialistMetrics.map((metric) => (
          <div key={metric.label} className="rounded-xl bg-slate-50 p-4">
            <dt className="text-xs text-slate-500">{metric.label}</dt>
            <dd className="mt-1 text-xl font-semibold tabular-nums text-slate-900">{metric.value}</dd>
          </div>
        ))}
      </dl>
      {latestProducts.length > 0 && (
        <div className="mt-5 overflow-x-auto rounded-xl border border-slate-200">
          <table className="min-w-[760px] divide-y divide-slate-200 text-left text-sm">
            <caption className="px-4 py-3 text-left text-sm font-semibold text-slate-800">
              {kind === 'etf' ? '最新上市产品' : '近期成立场外基金'}
              {kind === 'fund' && <span className="ml-2 text-xs font-normal text-slate-500">申购/赎回起始日仅为历史日期，不代表当前开放状态</span>}
            </caption>
            <thead className="bg-slate-50">
              <tr>
                <th scope="col" className="px-3 py-2 text-xs text-slate-500">产品</th>
                <th scope="col" className="px-3 py-2 text-xs text-slate-500">{kind === 'etf' ? '交易所 / 跟踪指数' : '类型 / 风格'}</th>
                <th scope="col" className="px-3 py-2 text-xs text-slate-500">{kind === 'etf' ? '上市日期' : '成立 / 终止日期'}</th>
                {kind === 'fund' && <th scope="col" className="px-3 py-2 text-xs text-slate-500">最低申购额</th>}
                {kind === 'fund' && <th scope="col" className="px-3 py-2 text-xs text-slate-500">申购 / 赎回起始日</th>}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {latestProducts.map((product) => (
                <tr key={product.ts_code}>
                  <td className="px-3 py-3">
                    <Link to={`/product/${encodeURIComponent(product.ts_code)}?kind=${kind}`} className="font-semibold text-indigo-700 hover:text-indigo-600">{product.name ?? product.ts_code}</Link>
                    <div className="text-xs text-slate-400">{product.ts_code}</div>
                  </td>
                  <td className="px-3 py-3 text-slate-600">
                    {kind === 'etf' ? (product.market ?? '--') : (product.fund_type ?? '--')}
                    <div className="text-xs text-slate-400">{kind === 'etf' ? (product.index_name ?? product.index_code ?? '--') : (product.invest_type ?? '--')}</div>
                  </td>
                  <td className="px-3 py-3 text-slate-600">
                    {kind === 'etf' ? (product.list_date ?? '--') : (product.found_date ?? '--')}
                    {kind === 'fund' && <div className="text-xs text-slate-400">{product.due_date ?? product.delist_date ?? '未披露终止日'}</div>}
                  </td>
                  {kind === 'fund' && <td className="px-3 py-3 tabular-nums text-slate-600">{product.min_amount ?? '--'}</td>}
                  {kind === 'fund' && <td className="px-3 py-3 text-slate-600">{product.purc_startdate ?? '--'}<div className="text-xs text-slate-400">{product.redm_startdate ?? '--'}</div></td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

const distributionConfig = (scope: DashboardKind, segment: SegmentKind) => {
  if (scope === 'all') {
    return segment === 'etf'
      ? [
          ['fund_type', 'ETF 产品类型', '按标准化产品类型展示前 10 项。'],
          ['market', 'ETF 交易市场', '仅描述上市交易所分布。'],
          ['index_name', 'ETF 跟踪指数 Top 10', '按跟踪指数对应的产品代码数排序。'],
          ['management', 'ETF 管理人 Top 10', '按产品代码数排序，不代表资产规模。'],
          ['m_fee', 'ETF 管理费率分布', '按披露管理费率区间统计。'],
        ] as const
      : [
          ['fund_type', '场外基金类型', '按产品代码数展示前 10 项。'],
          ['invest_type', '场外基金投资风格', '缺失值不按其他类型归并。'],
          ['management', '场外基金管理人 Top 10', '按产品代码数排序，不代表资产规模。'],
          ['status', '场外基金产品状态', '区分存续与到期终止等状态。'],
          ['m_fee', '场外基金管理费率分布', '按披露管理费率区间统计。'],
        ] as const;
  }
  return [
    ['fund_type', `${segmentLabels[segment]}产品类型`, '按产品代码数展示前 10 项。'],
    ['invest_type', `${segmentLabels[segment]}投资风格`, '基于基础信息中的投资类型。'],
    ['status', `${segmentLabels[segment]}产品状态`, segment === 'etf' ? '区分上市交易与摘牌等状态。' : '区分存续与到期终止等状态。'],
    ['management', `${segmentLabels[segment]}管理人 Top 10`, '按产品代码数排序，不代表资产规模。'],
    ['m_fee', `${segmentLabels[segment]}管理费率分布`, '按披露管理费率区间统计。'],
    ['c_fee', `${segmentLabels[segment]}托管费率分布`, '按披露托管费率区间统计。'],
    ...(segment === 'etf' ? [['index_name', 'ETF 跟踪指数 Top 10', '按跟踪指数对应的产品代码数排序。'] as const] : []),
    ...(segment === 'etf' ? [['market', 'ETF 交易市场', '仅描述上市交易所分布。'] as const] : []),
  ] as const;
};

export default function Dashboard() {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const kind = parseKind(searchParams.get('kind'));
  useEffect(() => {
    const currentKind = searchParams.get('kind');
    if (currentKind !== 'all' && currentKind !== 'etf' && currentKind !== 'fund') {
      const next = new URLSearchParams(searchParams);
      next.set('kind', 'all');
      setSearchParams(next, { replace: true });
    }
  }, [searchParams, setSearchParams]);
  const filters = useMemo(() => {
    const next = emptyFilters();
    filterKeys.forEach((key) => { next[key] = searchParams.getAll(key).filter(Boolean); });
    return next;
  }, [searchParams]);
  const validMetrics = useMemo(() => new Set<string>([
    ...commonMetrics.map(([value]) => value),
    ...(kind === 'etf' ? etfMetrics.map(([value]) => value) : []),
  ]), [kind]);
  const metricFromUrl = searchParams.get('metric');
  const rankingMetric = metricFromUrl && validMetrics.has(metricFromUrl) ? metricFromUrl : 'return_1y';
  const { overview, trend, rankings, reload } = useInstrumentDashboard({ kind, filters, rankingMetric });
  const analytics = overview.data?.kind === kind ? overview.data : null;
  const trendResponse = trend.data?.kind === kind ? trend.data : null;
  const segments = selectedSegments(kind);

  const updateParams = (mutate: (next: URLSearchParams) => void) => {
    const next = new URLSearchParams(searchParams);
    mutate(next);
    setSearchParams(next);
  };

  const changeKind = (nextKind: DashboardKind) => {
    updateParams((next) => {
      next.set('kind', nextKind);
      filterKeys.forEach((key) => next.delete(key));
      if (nextKind !== 'etf' && etfMetrics.some(([metric]) => metric === next.get('metric'))) {
        next.delete('metric');
      }
    });
  };

  const changeFilter = (key: DashboardFilterKey, values: string[]) => {
    updateParams((next) => {
      next.delete(key);
      values.forEach((value) => next.append(key, value));
    });
  };

  const resetFilters = () => updateParams((next) => filterKeys.forEach((key) => next.delete(key)));
  const changeMetric = (metric: string) => updateParams((next) => next.set('metric', metric));
  const analyticsStatus = analytics?.status ?? (overview.loading ? 'partial' : 'unavailable');
  const metricOptions = [
    ...commonMetrics,
    ...(kind === 'etf' ? etfMetrics : []),
  ];

  const openDistribution = (segment: SegmentKind, key: DashboardFilterKey, value: string) => {
    navigate(researchUrl(segment, filters, { key, value }));
  };

  return (
    <div className="min-h-screen bg-slate-50 pb-16">
      <header className="relative overflow-hidden bg-gradient-to-br from-slate-950 via-indigo-950 to-indigo-700 text-white">
        <div className="absolute -left-32 top-16 h-72 w-72 rounded-full bg-cyan-400/15 blur-3xl" />
        <div className="absolute -right-20 bottom-0 h-80 w-80 rounded-full bg-violet-300/15 blur-3xl" />
        <div className="relative mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8 lg:py-12">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <p className="text-xs font-semibold uppercase tracking-[0.3em] text-cyan-200">Fund Intelligence</p>
              <h1 className="mt-3 text-3xl font-semibold leading-tight sm:text-4xl">基金投研全景驾驶舱</h1>
              <p className="mt-3 text-sm leading-6 text-indigo-100 sm:text-base">ETF 与场外公募基金采用各自正确的生命周期、交易和净值口径，在同一入口完成市场扫描、候选发现和数据治理。</p>
              <p className="mt-2 text-xs text-indigo-200">数据截至：{analytics?.as_of ?? '--'} · 全部数量按产品代码统计</p>
              <DashboardResearchSearch scope={kind} />
            </div>
            <DashboardScopeTabs value={kind} onChange={changeKind} />
          </div>
          <div className="mt-7"><DashboardOverviewCards kind={kind} analytics={analytics} /></div>
        </div>
      </header>

      <main id="dashboard-content" role="tabpanel" aria-labelledby={`dashboard-tab-${kind}`} className="mx-auto max-w-7xl space-y-8 px-4 py-8 sm:px-6 lg:px-8">
        <DataHealthRefreshPanel analyticsStatus={analyticsStatus} asOf={analytics?.as_of} dataQuality={analytics?.data_quality} onRefreshCompleted={reload} />

        <DashboardFilterBar filters={filters} availableFilters={analytics?.available_filters} onChange={changeFilter} onReset={resetFilters} />

        {overview.error && (
          <div role="alert" className="flex flex-col gap-3 rounded-2xl border border-rose-200 bg-rose-50 p-5 text-sm text-rose-700 sm:flex-row sm:items-center sm:justify-between">
            <span>{overview.error}</span>
            <button type="button" onClick={reload} className="self-start rounded-lg bg-rose-600 px-4 py-2 font-semibold text-white hover:bg-rose-500">重试</button>
          </div>
        )}

        {analytics?.status === 'partial' && (
          <div role="status" className="rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">部分数据集未就绪，页面已保留可用区块；缺失值以 “--” 展示，不按 0 参与统计。</div>
        )}

        <div className={`grid gap-6 ${segments.length === 2 ? 'xl:grid-cols-2' : ''}`}>
          {segments.map((segmentKind) => <SegmentLens key={segmentKind} kind={segmentKind} segment={analytics?.segments[segmentKind]} filters={filters} loading={overview.loading && !analytics} />)}
        </div>

        <section aria-labelledby="trend-section-heading" className="space-y-4">
          <div>
            <h2 id="trend-section-heading" className="text-xl font-semibold text-slate-900">市场发展节奏</h2>
            <p className="mt-1 text-sm text-slate-500">ETF 使用上市日期，场外公募基金使用成立日期；两种日期不混算。</p>
          </div>
          <div className={`grid gap-6 ${segments.length === 2 ? 'xl:grid-cols-2' : ''}`}>
            {segments.map((segmentKind) => {
              const fallbackSeries = analytics?.segments[segmentKind]?.event_trend;
              const series = trendResponse?.series[segmentKind] ?? fallbackSeries;
              return <TrendChartCard key={segmentKind} series={series} loading={trend.loading && !series} error={!series ? trend.error : null} viewAllTo={researchUrl(segmentKind, filters)} />;
            })}
          </div>
        </section>

        <section aria-labelledby="structure-section-heading" className="space-y-5">
          <div>
            <h2 id="structure-section-heading" className="text-xl font-semibold text-slate-900">市场结构</h2>
            <p className="mt-1 text-sm text-slate-500">点击图表分类或数据表中的分类名称，可携带当前条件进入产品研究。</p>
          </div>
          {segments.map((segmentKind) => {
            const segment = analytics?.segments[segmentKind];
            if (!segment || segment.availability === 'missing') {
              return null;
            }
            return (
              <div key={segmentKind} className="space-y-3">
                <h3 className="text-sm font-semibold uppercase tracking-[0.16em] text-slate-500">{segmentLabels[segmentKind]}</h3>
                <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
                  {distributionConfig(kind, segmentKind).map(([key, title, description]) => (
                    <DistributionChartCard
                      key={`${segmentKind}-${key}`}
                      title={title}
                      description={description}
                      data={segment.distributions[key] ?? []}
                      loading={overview.loading && !analytics}
                      onSelect={isDashboardFilterKey(key) ? (value) => openDistribution(segmentKind, key, value) : undefined}
                      viewAllTo={researchUrl(segmentKind, filters)}
                    />
                  ))}
                </div>
              </div>
            );
          })}
        </section>

        <section aria-labelledby="ranking-section-heading" className="space-y-5">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <h2 id="ranking-section-heading" className="text-xl font-semibold text-slate-900">业绩、风险与交易指标</h2>
              <p className="mt-1 text-sm text-slate-500">共同收益风险指标统一来自复权净值；ETF 交易指标单独来自交易行情。</p>
            </div>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
              <div>
                <span className="block text-sm font-medium text-slate-700">收益排行周期</span>
                <div role="group" aria-label="收益排行周期" className="mt-1 inline-flex rounded-xl border border-slate-200 bg-white p-1 shadow-sm">
                  {returnPeriods.map(([metric, label]) => (
                    <button key={metric} type="button" aria-pressed={rankingMetric === metric} onClick={() => changeMetric(metric)} className={`rounded-lg px-3 py-1.5 text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-indigo-500 ${rankingMetric === metric ? 'bg-indigo-600 text-white' : 'text-slate-600 hover:bg-slate-50'}`}>{label}</button>
                  ))}
                </div>
              </div>
              <label className="flex flex-col gap-1 text-sm font-medium text-slate-700">
                其他排行指标
                <select value={rankingMetric} onChange={(event) => changeMetric(event.target.value)} className="min-w-56 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                  {metricOptions.map(([value, fallbackLabel]) => <option key={value} value={value}>{analytics?.metric_definitions?.[value]?.label ?? fallbackLabel}</option>)}
                </select>
              </label>
            </div>
          </div>
          <div className={`grid gap-6 ${segments.length === 2 ? 'xl:grid-cols-2' : ''}`}>
            {segments.map((segmentKind) => <DashboardRankingTable key={segmentKind} kind={segmentKind} state={rankings[segmentKind]} />)}
          </div>
          <p className="text-xs leading-5 text-slate-500">风险收益排行仅供投研筛选，不构成投资建议。当前版本不做母基金份额归并、实时 IOPV、持仓穿透或场外实时估值。</p>
        </section>
      </main>
    </div>
  );
}
