import type { DashboardKind, DashboardSummary, InstrumentAnalyticsResponse } from './types';

const integerFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 0 });
const percentFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 1 });

const count = (value?: number | null) => value === null || value === undefined
  ? '--'
  : integerFormatter.format(value);

const percent = (value?: number | null) => value === null || value === undefined
  ? '--'
  : `${percentFormatter.format(value * 100)}%`;

interface MetricCardProps {
  label: string;
  value: string;
  description: string;
}

function MetricCard({ label, value, description }: MetricCardProps) {
  return (
    <div className="rounded-xl border border-white/15 bg-white/10 p-4 backdrop-blur">
      <p className="text-xs font-semibold uppercase tracking-wide text-white">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-white">{value}</p>
      <p className="mt-1 text-xs leading-5 text-white">{description}</p>
    </div>
  );
}

const summaryFor = (analytics: InstrumentAnalyticsResponse, kind: Exclude<DashboardKind, 'all'>) => (
  analytics.summary[kind] ?? analytics.segments[kind]?.summary
);

const singleCards = (kind: 'etf' | 'fund', summary?: DashboardSummary) => {
  const label = kind === 'etf' ? 'ETF' : '场外基金';
  return [
    { label: `${label}产品代码`, value: count(summary?.share_code_count), description: '按 Tushare ts_code 计数' },
    { label: kind === 'etf' ? '上市交易中' : '存续中', value: count(summary?.active_count), description: '按标准化产品状态统计' },
    { label: '基金管理人', value: count(summary?.unique_managements), description: '管理人名称去重' },
    { label: '净值覆盖率', value: percent(summary?.nav_coverage_rate), description: `${count(summary?.nav_covered_count)} 个份额具有净值快照` },
    {
      label: kind === 'etf' ? '最新行情日' : '最新净值日',
      value: (kind === 'etf' ? summary?.latest_candle_date : summary?.latest_nav_date) ?? '--',
      description: kind === 'etf' ? '交易所日行情的最新日期' : '各份额最新复权净值日期的最大值',
    },
    {
      label: kind === 'etf' ? '20日流动性覆盖率' : '申赎信息覆盖率',
      value: percent(kind === 'etf' ? summary?.liquidity_coverage_rate : summary?.purchase_redemption_coverage_rate),
      description: kind === 'etf' ? '具备近20个交易观察值的份额' : '披露申购或赎回起始日的份额',
    },
  ];
};

interface DashboardOverviewCardsProps {
  kind: DashboardKind;
  analytics: InstrumentAnalyticsResponse | null;
}

export default function DashboardOverviewCards({ kind, analytics }: DashboardOverviewCardsProps) {
  if (!analytics) {
    return (
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6" aria-label="市场概览加载中">
        {Array.from({ length: 6 }, (_, index) => (
          <div key={index} className="h-28 animate-pulse rounded-xl bg-white/10" />
        ))}
      </div>
    );
  }

  const etf = summaryFor(analytics, 'etf');
  const fund = summaryFor(analytics, 'fund');
  const all = analytics.summary.all;
  const cards = kind === 'all'
    ? [
        { label: 'ETF产品代码', value: count(etf?.share_code_count), description: '交易所上市基金代码' },
        { label: '场外基金产品代码', value: count(fund?.share_code_count), description: '不同份额类别分别计数' },
        { label: 'ETF上市交易中', value: count(etf?.active_count), description: '剔除摘牌等非活跃状态' },
        { label: '场外基金存续中', value: count(fund?.active_count), description: '剔除到期终止等状态' },
        { label: '基金管理人', value: count(all?.unique_managements), description: '两类管理人名称合并去重' },
        { label: '净值覆盖率', value: percent(all?.nav_coverage_rate), description: `${count(all?.nav_covered_count)} 个份额具有净值快照` },
      ]
    : singleCards(kind, summaryFor(analytics, kind));

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6" aria-label="市场关键指标">
      {cards.map((card) => <MetricCard key={card.label} {...card} />)}
    </div>
  );
}
