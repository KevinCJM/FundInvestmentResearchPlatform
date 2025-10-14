import React, { useEffect, useMemo, useState } from 'react';
import * as echarts from 'echarts';
import ReactECharts from 'echarts-for-react';

interface DistributionItem {
  name: string;
  value: number;
}

interface FeeByFundTypeItem {
  fund_type: string;
  avg_m_fee?: number | null;
  avg_c_fee?: number | null;
}

interface MarketIssueItem {
  market: string;
  count: number;
  total_issue_amount?: number | null;
}

interface TrendItem {
  year: number;
  count: number;
  total_issue_amount?: number | null;
}

interface TableItem {
  ts_code?: string;
  name?: string;
  issue_amount?: number | null;
  list_date?: string | null;
  market?: string | null;
}

interface AnalyticsSummary {
  total_count: number;
  active_count?: number | null;
  unique_managements?: number | null;
  total_issue_amount?: number | null;
  avg_m_fee?: number | null;
  avg_c_fee?: number | null;
  avg_exp_return?: number | null;
  avg_duration_year?: number | null;
}

interface AnalyticsResponse {
  summary: AnalyticsSummary;
  top_management: { name: string; count: number; total_issue_amount?: number | null }[];
  organization_type_distribution: DistributionItem[];
  fund_type_distribution: DistributionItem[];
  invest_type_distribution: DistributionItem[];
  market_distribution: DistributionItem[];
  market_issue_summary: MarketIssueItem[];
  status_breakdown: DistributionItem[];
  list_trend: TrendItem[];
  fee_by_fund_type: FeeByFundTypeItem[];
  top_issue_amount: TableItem[];
  recent_listings: TableItem[];
}

const numberFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 0 });
const decimalFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 2 });

const formatIssueAmount = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  if (Math.abs(value) >= 10000) {
    return `${decimalFormatter.format(value / 10000)} 亿`;
  }
  return `${decimalFormatter.format(value)} 万`;
};

const createEmptyOption = (title: string) => ({
  title: {
    text: title,
    left: 'center',
    textStyle: { color: '#475569', fontSize: 16, fontWeight: 600 }
  },
  graphic: {
    type: 'text',
    left: 'center',
    top: 'middle',
    style: {
      text: '暂无数据',
      fill: '#94a3b8',
      fontSize: 16
    }
  }
});

export default function Dashboard() {
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        const resp = await fetch('/api/etf/analytics');
        if (!resp.ok) {
          throw new Error('无法获取ETF统计数据');
        }
        const data = (await resp.json()) as AnalyticsResponse;
        setAnalytics(data);
      } catch (err) {
        console.error('Failed to load ETF analytics', err);
        setError('统计信息加载失败，请稍后重试。');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const topManagementOption = useMemo(() => {
    if (!analytics?.top_management?.length) {
      return createEmptyOption('ETF管理人 Top 10');
    }
    const labels = analytics.top_management.map((item) => item.name).reverse();
    const counts = analytics.top_management.map((item) => item.count).reverse();
    return {
      title: { text: 'ETF管理人 Top 10', left: 'center', textStyle: { fontSize: 16 } },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: '3%', right: '6%', bottom: '3%', containLabel: true },
      xAxis: { type: 'value', boundaryGap: [0, 0.01], axisLabel: { color: '#475569' } },
      yAxis: { type: 'category', data: labels, axisLabel: { color: '#475569' } },
      series: [
        {
          name: '产品数量',
          type: 'bar',
          data: counts,
          itemStyle: {
            borderRadius: 6,
            color: new echarts.graphic.LinearGradient(1, 0, 0, 0, [
              { offset: 0, color: '#6366f1' },
              { offset: 1, color: '#22d3ee' }
            ])
          }
        }
      ]
    };
  }, [analytics]);

  const pieOption = (title: string, data?: DistributionItem[]) => {
    if (!data || data.length === 0) {
      return createEmptyOption(title);
    }
    return {
      title: { text: title, left: 'center', textStyle: { fontSize: 16 } },
      tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      legend: { bottom: 0, type: 'scroll' },
      series: [
        {
          type: 'pie',
          radius: ['35%', '65%'],
          center: ['50%', '45%'],
          itemStyle: { borderRadius: 8, borderColor: '#fff', borderWidth: 2 },
          data
        }
      ]
    };
  };

  const trendOption = useMemo(() => {
    if (!analytics?.list_trend?.length) {
      return createEmptyOption('ETF上市节奏');
    }
    const years = analytics.list_trend.map((item) => item.year);
    const counts = analytics.list_trend.map((item) => item.count);
    const issueAmounts = analytics.list_trend.map((item) => item.total_issue_amount ?? null);
    return {
      title: { text: 'ETF上市节奏', left: 'center', textStyle: { fontSize: 16 } },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const count = params[0]?.value ?? '--';
          const amount = params[1]?.value;
          const amountText = amount === null || amount === undefined ? '--' : formatIssueAmount(Number(amount));
          return `${params[0].axisValue}年<br />上市数量：${count}<br />发行规模：${amountText}`;
        }
      },
      grid: { left: '3%', right: '3%', bottom: '10%', containLabel: true },
      legend: { bottom: 0 },
      xAxis: { type: 'category', data: years, axisLabel: { color: '#475569' } },
      yAxis: [
        { type: 'value', name: '上市数量', axisLabel: { color: '#475569' } },
        { type: 'value', name: '发行规模(万)', axisLabel: { color: '#475569' } }
      ],
      series: [
        {
          name: '上市数量',
          type: 'line',
          smooth: true,
          data: counts,
          yAxisIndex: 0,
          areaStyle: {
            color: 'rgba(99, 102, 241, 0.15)'
          },
          lineStyle: { color: '#6366f1', width: 3 },
          symbol: 'circle',
          symbolSize: 8
        },
        {
          name: '发行规模',
          type: 'bar',
          data: issueAmounts,
          yAxisIndex: 1,
          itemStyle: {
            borderRadius: 6,
            color: 'rgba(14, 165, 233, 0.8)'
          }
        }
      ]
    };
  }, [analytics]);

  const feeOption = useMemo(() => {
    if (!analytics?.fee_by_fund_type?.length) {
      return createEmptyOption('管理/托管费率对比');
    }
    const labels = analytics.fee_by_fund_type.map((item) => item.fund_type).reverse();
    const mFees = analytics.fee_by_fund_type.map((item) => item.avg_m_fee ?? null).reverse();
    const cFees = analytics.fee_by_fund_type.map((item) => item.avg_c_fee ?? null).reverse();
    return {
      title: { text: '管理/托管费率对比', left: 'center', textStyle: { fontSize: 16 } },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { top: 0 },
      grid: { left: '3%', right: '6%', bottom: '3%', containLabel: true },
      xAxis: { type: 'value', axisLabel: { color: '#475569', formatter: '{value}%' } },
      yAxis: { type: 'category', data: labels, axisLabel: { color: '#475569' } },
      series: [
        {
          name: '管理费',
          type: 'bar',
          data: mFees,
          itemStyle: { color: '#22d3ee', borderRadius: 6 }
        },
        {
          name: '托管费',
          type: 'bar',
          data: cFees,
          itemStyle: { color: '#6366f1', borderRadius: 6 }
        }
      ]
    };
  }, [analytics]);

  const summary = analytics?.summary;

  if (loading) {
    return (
      <div className="flex h-full min-h-[50vh] items-center justify-center text-slate-500">
        正在加载ETF统计信息...
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 pb-16">
      <div className="relative overflow-hidden bg-gradient-to-r from-sky-500 via-indigo-500 to-purple-500 text-white">
        <div className="absolute -left-24 top-12 h-64 w-64 rounded-full bg-white/10 blur-3xl" />
        <div className="absolute -right-20 bottom-0 h-72 w-72 rounded-full bg-sky-300/20 blur-3xl" />
        <div className="relative mx-auto max-w-6xl px-6 py-12">
          <p className="text-sm uppercase tracking-[0.3em] text-white/80">ETF Intelligence</p>
          <h1 className="mt-2 text-3xl font-semibold leading-tight md:text-4xl">
            ETF全景智能驾驶舱
          </h1>
          <p className="mt-4 max-w-3xl text-base text-white/80">
            基于数据目录中的 etf_info_df.parquet 构建，全方位洞察中国ETF市场的发行规模、策略风格与市场分布，助力投研团队快速掌握行业脉动。
          </p>
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-2xl bg-white/10 p-4 backdrop-blur">
              <p className="text-xs uppercase tracking-wide text-white/70">ETF总数</p>
              <p className="mt-2 text-2xl font-semibold">{summary ? numberFormatter.format(summary.total_count) : '--'}</p>
              <p className="text-xs text-white/70">覆盖多市场多策略ETF产品</p>
            </div>
            <div className="rounded-2xl bg-white/10 p-4 backdrop-blur">
              <p className="text-xs uppercase tracking-wide text-white/70">在市产品</p>
              <p className="mt-2 text-2xl font-semibold">
                {summary && summary.active_count !== null && summary.active_count !== undefined
                  ? numberFormatter.format(summary.active_count)
                  : '--'}
              </p>
              <p className="text-xs text-white/70">剔除退市、清盘等非存续状态</p>
            </div>
            <div className="rounded-2xl bg-white/10 p-4 backdrop-blur">
              <p className="text-xs uppercase tracking-wide text-white/70">基金管理人</p>
              <p className="mt-2 text-2xl font-semibold">
                {summary && summary.unique_managements !== null && summary.unique_managements !== undefined
                  ? numberFormatter.format(summary.unique_managements)
                  : '--'}
              </p>
              <p className="text-xs text-white/70">头部机构集中度一目了然</p>
            </div>
            <div className="rounded-2xl bg-white/10 p-4 backdrop-blur">
              <p className="text-xs uppercase tracking-wide text-white/70">累计发行规模</p>
              <p className="mt-2 text-2xl font-semibold">
                {summary && summary.total_issue_amount !== null && summary.total_issue_amount !== undefined
                  ? formatIssueAmount(summary.total_issue_amount)
                  : '--'}
              </p>
              <p className="text-xs text-white/70">以基金合同披露的发行规模估算</p>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto -mt-12 max-w-6xl space-y-10 px-6">
        {error && (
          <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-600">
            {error}
          </div>
        )}

        {analytics && (
          <>
            <section className="grid gap-6 lg:grid-cols-2">
              <div className="rounded-2xl bg-white p-5 shadow-sm">
                <ReactECharts option={topManagementOption} style={{ height: 360 }} notMerge lazyUpdate />
              </div>
              <div className="rounded-2xl bg-white p-5 shadow-sm">
                <ReactECharts option={trendOption} style={{ height: 360 }} notMerge lazyUpdate />
              </div>
            </section>

            <section className="grid gap-6 md:grid-cols-2">
              <div className="rounded-2xl bg-white p-5 shadow-sm">
                <ReactECharts option={pieOption('基金机构类型', analytics.organization_type_distribution)} style={{ height: 360 }} notMerge lazyUpdate />
              </div>
              <div className="rounded-2xl bg-white p-5 shadow-sm">
                <ReactECharts option={pieOption('投资风格分布', analytics.invest_type_distribution)} style={{ height: 360 }} notMerge lazyUpdate />
              </div>
              <div className="rounded-2xl bg-white p-5 shadow-sm">
                <ReactECharts option={pieOption('产品类型分布', analytics.fund_type_distribution)} style={{ height: 360 }} notMerge lazyUpdate />
              </div>
              <div className="rounded-2xl bg-white p-5 shadow-sm">
                <ReactECharts option={pieOption('产品状态概览', analytics.status_breakdown)} style={{ height: 360 }} notMerge lazyUpdate />
              </div>
            </section>

            <section className="rounded-2xl bg-white p-5 shadow-sm">
              <ReactECharts option={feeOption} style={{ height: 360 }} notMerge lazyUpdate />
            </section>

            <section className="grid gap-6 lg:grid-cols-2">
              <div className="rounded-2xl bg-white p-5 shadow-sm">
                <h2 className="text-lg font-semibold text-slate-800">发行规模 Top 10</h2>
                <p className="text-xs text-slate-400">根据发行披露数据排序</p>
                <div className="mt-4 overflow-x-auto">
                  <table className="min-w-full divide-y divide-slate-200">
                    <thead>
                      <tr className="text-left text-xs uppercase tracking-wide text-slate-400">
                        <th className="px-2 py-2">基金代码</th>
                        <th className="px-2 py-2">名称</th>
                        <th className="px-2 py-2">发行规模</th>
                        <th className="px-2 py-2">上市日期</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 text-sm text-slate-700">
                      {analytics.top_issue_amount.length ? (
                        analytics.top_issue_amount.map((item, index) => (
                          <tr key={`${item.ts_code ?? 'NA'}-${item.name ?? 'NA'}-${index}`} className="hover:bg-slate-50">
                            <td className="px-2 py-2 font-medium text-slate-600">{item.ts_code ?? '--'}</td>
                            <td className="px-2 py-2">{item.name ?? '--'}</td>
                            <td className="px-2 py-2">{formatIssueAmount(item.issue_amount)}</td>
                            <td className="px-2 py-2">{item.list_date ?? '--'}</td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td className="px-2 py-6 text-center" colSpan={4}>
                            暂无数据
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="rounded-2xl bg-white p-5 shadow-sm">
                <h2 className="text-lg font-semibold text-slate-800">最新上市ETF</h2>
                <p className="text-xs text-slate-400">追踪近期登陆交易所的产品</p>
                <div className="mt-4 overflow-x-auto">
                  <table className="min-w-full divide-y divide-slate-200">
                    <thead>
                      <tr className="text-left text-xs uppercase tracking-wide text-slate-400">
                        <th className="px-2 py-2">上市日期</th>
                        <th className="px-2 py-2">基金代码</th>
                        <th className="px-2 py-2">名称</th>
                        <th className="px-2 py-2">市场</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 text-sm text-slate-700">
                      {analytics.recent_listings.length ? (
                        analytics.recent_listings.map((item, index) => (
                          <tr key={`${item.list_date ?? 'NA'}-${item.ts_code ?? 'NA'}-${item.name ?? 'NA'}-${index}`} className="hover:bg-slate-50">
                            <td className="px-2 py-2 font-medium text-slate-600">{item.list_date ?? '--'}</td>
                            <td className="px-2 py-2">{item.ts_code ?? '--'}</td>
                            <td className="px-2 py-2">{item.name ?? '--'}</td>
                            <td className="px-2 py-2">{item.market ?? '--'}</td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td className="px-2 py-6 text-center" colSpan={4}>
                            暂无数据
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
          </>
        )}
      </div>
    </div>
  );
}
