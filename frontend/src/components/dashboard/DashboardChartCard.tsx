import { useId, useMemo } from 'react';
import type { EChartsOption } from 'echarts';
import ReactECharts from 'echarts-for-react';
import { Link } from 'react-router-dom';
import type { DashboardTrendSeries, DistributionPoint } from './types';

const integerFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 0 });
const decimalFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 2 });

const formatIssueAmount = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  if (Math.abs(value) >= 10_000) {
    return `${decimalFormatter.format(value / 10_000)} 亿`;
  }
  return `${decimalFormatter.format(value)} 万`;
};

interface CardStateProps {
  loading?: boolean;
  error?: string | null;
}

function CardState({ loading, error }: CardStateProps) {
  if (loading) {
    return <div className="flex h-64 items-center justify-center text-sm text-slate-400">正在加载图表...</div>;
  }
  if (error) {
    return <div role="alert" className="flex h-64 items-center justify-center rounded-xl bg-rose-50 px-6 text-center text-sm text-rose-600">{error}</div>;
  }
  return null;
}

interface DistributionChartCardProps extends CardStateProps {
  title: string;
  description: string;
  data?: DistributionPoint[];
  onSelect?: (value: string) => void;
  viewAllTo?: string;
}

export function DistributionChartCard({
  title,
  description,
  data = [],
  loading = false,
  error = null,
  onSelect,
  viewAllTo,
}: DistributionChartCardProps) {
  const headingId = useId();
  const visibleData = useMemo(
    () => [...data].sort((left, right) => right.value - left.value).slice(0, 10),
    [data],
  );
  const option = useMemo<EChartsOption>(() => {
    const chartData = [...visibleData].reverse();
    return {
      animationDuration: 350,
      aria: { enabled: true, decal: { show: true }, description: `${title}，展示前 ${chartData.length} 项。` },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: 8, right: 24, top: 16, bottom: 12, containLabel: true },
      xAxis: { type: 'value', axisLabel: { color: '#64748b' } },
      yAxis: {
        type: 'category',
        data: chartData.map((item) => item.name),
        axisLabel: { color: '#475569', width: 120, overflow: 'truncate' },
      },
      series: [{
        name: '产品代码数',
        type: 'bar',
        data: chartData.map((item) => item.value),
        itemStyle: { color: '#6366f1', borderRadius: [0, 6, 6, 0] },
      }],
    };
  }, [title, visibleData]);

  const state = <CardState loading={loading} error={error} />;
  return (
    <section aria-labelledby={headingId} className="min-w-0 rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-100">
      <div className="flex items-start justify-between gap-3">
        <h3 id={headingId} className="text-base font-semibold text-slate-900">{title}</h3>
        {viewAllTo && <Link to={viewAllTo} className="shrink-0 text-xs font-semibold text-indigo-700 hover:text-indigo-600">查看全部 →</Link>}
      </div>
      <p className="mt-1 text-xs leading-5 text-slate-500">{description}</p>
      {loading || error ? state : visibleData.length === 0 ? (
        <div className="flex h-64 items-center justify-center rounded-xl border border-dashed border-slate-200 text-sm text-slate-400">暂无可展示数据</div>
      ) : (
        <>
          <ReactECharts
            option={option}
            style={{ height: Math.max(300, visibleData.length * 32) }}
            notMerge
            lazyUpdate
            onEvents={onSelect ? { click: (params: { name?: string }) => params.name && onSelect(params.name) } : undefined}
          />
          <details className="mt-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm">
            <summary className="cursor-pointer font-medium text-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500">查看数据表</summary>
            <div className="mt-3 overflow-x-auto">
              <table className="min-w-full divide-y divide-slate-200 text-left text-sm">
                <caption className="sr-only">{title}数据表</caption>
                <thead><tr><th scope="col" className="px-2 py-2 text-xs text-slate-500">分类</th><th scope="col" className="px-2 py-2 text-right text-xs text-slate-500">产品代码数</th></tr></thead>
                <tbody className="divide-y divide-slate-100">
                  {visibleData.map((item) => (
                    <tr key={item.name}>
                      <td className="px-2 py-2 text-slate-700">
                        {onSelect ? (
                          <button type="button" onClick={() => onSelect(item.name)} className="text-left font-medium text-indigo-700 underline-offset-2 hover:underline">{item.name}</button>
                        ) : item.name}
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums text-slate-600">{integerFormatter.format(item.value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </>
      )}
    </section>
  );
}

interface TrendChartCardProps extends CardStateProps {
  series?: DashboardTrendSeries | null;
  viewAllTo?: string;
}

export function TrendChartCard({ series, loading = false, error = null, viewAllTo }: TrendChartCardProps) {
  const headingId = useId();
  const points = useMemo(() => [...(series?.points ?? [])].sort((left, right) => left.year - right.year), [series]);
  const hasIssueAmount = points.some((point) => point.total_issue_amount !== null && point.total_issue_amount !== undefined);
  const option = useMemo<EChartsOption>(() => ({
    animationDuration: 350,
    aria: { enabled: true, decal: { show: true }, description: `${series?.label ?? '产品趋势'}，按年份展示产品代码数。` },
    tooltip: { trigger: 'axis' },
    legend: hasIssueAmount ? { bottom: 0 } : undefined,
    grid: { left: 12, right: hasIssueAmount ? 28 : 16, top: 20, bottom: hasIssueAmount ? 48 : 24, containLabel: true },
    xAxis: { type: 'category', data: points.map((point) => String(point.year)), axisLabel: { color: '#64748b' } },
    yAxis: hasIssueAmount
      ? [
          { type: 'value', name: '份额数', axisLabel: { color: '#64748b' } },
          { type: 'value', name: '发行规模(万)', axisLabel: { color: '#64748b' } },
        ]
      : { type: 'value', name: '份额数', axisLabel: { color: '#64748b' } },
    series: [
      {
        name: '新增产品代码数', type: 'line', smooth: true, data: points.map((point) => point.count),
        symbolSize: 7, lineStyle: { width: 3, color: '#4f46e5' }, itemStyle: { color: '#4f46e5' },
        areaStyle: { color: 'rgba(99, 102, 241, 0.14)' },
      },
      ...(hasIssueAmount ? [{
        name: '披露发行规模', type: 'bar' as const, yAxisIndex: 1,
        data: points.map((point) => point.total_issue_amount ?? null),
        itemStyle: { color: 'rgba(14, 165, 233, 0.72)', borderRadius: [5, 5, 0, 0] },
      }] : []),
    ],
  }), [hasIssueAmount, points, series?.label]);

  const title = series?.label ?? '发行与成立趋势';
  const dateDescription = series?.date_field === 'found_date'
    ? '严格按成立日期统计；场外基金不使用上市日期。'
    : '严格按交易所上市日期统计。';

  return (
    <section aria-labelledby={headingId} className="min-w-0 rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-100">
      <div className="flex items-start justify-between gap-3">
        <h3 id={headingId} className="text-base font-semibold text-slate-900">{title}</h3>
        {viewAllTo && <Link to={viewAllTo} className="shrink-0 text-xs font-semibold text-indigo-700 hover:text-indigo-600">查看全部 →</Link>}
      </div>
      <p className="mt-1 text-xs leading-5 text-slate-500">{dateDescription} 发行规模为披露口径，并非当前 AUM。</p>
      {loading || error ? <CardState loading={loading} error={error} /> : points.length === 0 ? (
        <div className="flex h-72 items-center justify-center rounded-xl border border-dashed border-slate-200 text-sm text-slate-400">暂无趋势数据</div>
      ) : (
        <>
          <ReactECharts option={option} style={{ height: 340 }} notMerge lazyUpdate />
          <details className="mt-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm">
            <summary className="cursor-pointer font-medium text-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500">查看数据表</summary>
            <div className="mt-3 overflow-x-auto">
              <table className="min-w-full divide-y divide-slate-200 text-left text-sm">
                <caption className="sr-only">{title}数据表</caption>
                <thead><tr><th scope="col" className="px-2 py-2 text-xs text-slate-500">年份</th><th scope="col" className="px-2 py-2 text-right text-xs text-slate-500">新增产品代码数</th><th scope="col" className="px-2 py-2 text-right text-xs text-slate-500">披露发行规模</th></tr></thead>
                <tbody className="divide-y divide-slate-100">
                  {points.map((point) => (
                    <tr key={point.year}><td className="px-2 py-2 text-slate-700">{point.year}</td><td className="px-2 py-2 text-right tabular-nums text-slate-600">{integerFormatter.format(point.count)}</td><td className="px-2 py-2 text-right text-slate-600">{formatIssueAmount(point.total_issue_amount)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </>
      )}
    </section>
  );
}
