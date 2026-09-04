import { useId } from 'react';
import { Link } from 'react-router-dom';
import { MetricValue, formatMetricValue as formatPresentedMetricValue } from '../metrics/MetricDisplay';
import type { MetricPresentation } from '../../services/customIndicators';
import type { InstrumentRankingsResponse, Loadable, MetricDefinition, SegmentKind } from './types';

const integerFormatter = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 0 });

const snapshotPresentation = (
  definition?: MetricDefinition | null,
  metric = 'snapshot_metric',
): MetricPresentation => {
  const rawUnit = definition?.unit?.toLowerCase() ?? '';
  const isScore = metric === 'sharpe_1y' || metric === 'calmar_3y';
  const isRatio = rawUnit === 'ratio' || rawUnit.includes('return') || rawUnit.includes('drawdown');
  const isPercent = rawUnit === 'percent' || rawUnit === 'percentage' || rawUnit === '%';
  const isAmount = rawUnit.includes('wan') || rawUnit.includes('10k') || rawUnit.includes('万元');
  return {
    indicator_id: null,
    revision: null,
    name: definition?.label ?? metric,
    source: definition?.source ?? 'instrument_metrics_snapshot',
    category: 'snapshot',
    category_label: '指标快照',
    context_kind: 'single_product',
    catalog_status: 'current',
    display_format: !isScore && (isRatio || isPercent) ? 'percent' : 'number',
    precision: isScore ? 2 : (isAmount ? 0 : 2),
    unit: isAmount ? '万元' : '',
    notation: isAmount ? 'compact' : 'standard',
    value_scale: isPercent ? 1 : (!isScore && isRatio ? 100 : 1),
    output_measure: isAmount ? 'currency_amount' : (isScore ? 'dimensionless' : 'return_decimal'),
    direction: 'higher_better',
    description: '由真实产品数据生成的只读研究快照。',
    methodology: '快照指标不属于工作区自定义指标。',
    data_basis: definition?.source ?? 'instrument_metrics_snapshot',
    minimum_observations: 1,
    applicable_product_kinds: ['etf', 'fund'],
  };
};

export const formatMetricValue = (value: number | null | undefined, definition?: MetricDefinition | null, metric?: string) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '--';
  return formatPresentedMetricValue(value, snapshotPresentation(definition, metric));
};

interface DashboardRankingTableProps {
  kind: SegmentKind;
  state: Loadable<InstrumentRankingsResponse>;
}

export default function DashboardRankingTable({ kind, state }: DashboardRankingTableProps) {
  const headingId = useId();
  const label = kind === 'etf' ? 'ETF' : '场外公募基金';
  const response = state.data;
  const definition = response?.metric_definition;

  return (
    <section aria-labelledby={headingId} className="min-w-0 rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-100">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h3 id={headingId} className="text-base font-semibold text-slate-900">{label} · 指标快照{definition ? ` · ${definition.label}` : ''}</h3>
          <p className="mt-1 text-xs leading-5 text-slate-500">来源：{definition?.source ?? '产品分析快照'} · 截至：{response?.as_of ?? '—'}。仅纳入存续、数据新鲜、完整覆盖所选区间且无异常跳点的产品；空值不按 0 处理。</p>
        </div>
        <Link to={`/product-research/products?kind=${kind}`} className="shrink-0 text-sm font-semibold text-indigo-700 hover:text-indigo-600">查看全部产品 →</Link>
      </div>

      {state.loading ? (
        <div className="flex h-64 items-center justify-center text-sm text-slate-400">正在加载排行榜...</div>
      ) : state.error ? (
        <div role="alert" className="mt-4 rounded-xl bg-rose-50 p-4 text-sm text-rose-600">{state.error}</div>
      ) : !response || response.status === 'unavailable' ? (
        <div className="mt-4 rounded-xl border border-dashed border-amber-200 bg-amber-50 p-6 text-center text-sm text-amber-700">业绩快照尚未生成；结构统计仍可正常使用。</div>
      ) : response.items.length === 0 ? (
        <div className="mt-4 rounded-xl border border-dashed border-slate-200 p-6 text-center text-sm text-slate-400">暂无满足排行榜口径的产品</div>
      ) : (
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-[1080px] divide-y divide-slate-200 text-left text-sm">
            <caption className="sr-only">{label}{definition?.label ?? '指标'}排行榜</caption>
            <thead>
              <tr className="text-xs uppercase tracking-wide text-slate-500">
                <th scope="col" className="px-3 py-3">排名</th>
                <th scope="col" className="px-3 py-3">产品</th>
                <th scope="col" className="px-3 py-3">类型 / 风格</th>
                <th scope="col" className="px-3 py-3">管理人</th>
                <th scope="col" className="px-3 py-3 text-right">{definition?.label ?? response.metric}</th>
                <th scope="col" className="px-3 py-3 text-right">1年年化波动</th>
                <th scope="col" className="px-3 py-3 text-right">3年最大回撤</th>
                <th scope="col" className="px-3 py-3 text-right">1年 Sharpe</th>
                <th scope="col" className="px-3 py-3 text-right">样本数</th>
                <th scope="col" className="px-3 py-3">截至日</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {response.items.map((item, index) => (
                <tr key={`${item.instrument_type}-${item.ts_code}`} className="hover:bg-indigo-50/50">
                  <td className="px-3 py-3 font-semibold tabular-nums text-slate-400">{index + 1}</td>
                  <td className="px-3 py-3">
                    <Link to={`/product-research/products/${encodeURIComponent(item.ts_code)}?kind=${item.instrument_type}`} className="font-semibold text-indigo-700 hover:text-indigo-600">
                      {item.name ?? item.ts_code}
                    </Link>
                    <div className="mt-0.5 text-xs text-slate-400">{item.ts_code}</div>
                  </td>
                  <td className="px-3 py-3 text-slate-600"><span className="font-medium text-slate-700">{item.fund_type ?? '--'}</span><div className="text-xs text-slate-400">{item.invest_type ?? '--'}</div></td>
                  <td className="px-3 py-3 text-slate-600">{item.management ?? '--'}</td>
                  <td className="px-3 py-3 text-right font-semibold text-slate-900"><MetricValue value={item.value} presentation={snapshotPresentation(definition, response.metric)} /></td>
                  <td className="px-3 py-3 text-right text-slate-600"><MetricValue value={item.metrics?.annual_volatility_1y} presentation={snapshotPresentation({ label: '1年年化波动', unit: 'ratio', source: definition?.source }, 'annual_volatility_1y')} /></td>
                  <td className="px-3 py-3 text-right text-slate-600"><MetricValue value={item.metrics?.max_drawdown_3y} presentation={snapshotPresentation({ label: '3年最大回撤', unit: 'ratio', source: definition?.source }, 'max_drawdown_3y')} /></td>
                  <td className="px-3 py-3 text-right text-slate-600"><MetricValue value={item.metrics?.sharpe_1y} presentation={snapshotPresentation({ label: '1年 Sharpe', unit: 'ratio', source: definition?.source }, 'sharpe_1y')} /></td>
                  <td className="px-3 py-3 text-right tabular-nums text-slate-600">{item.observation_count === null || item.observation_count === undefined ? '--' : integerFormatter.format(item.observation_count)}</td>
                  <td className="px-3 py-3 text-slate-600">{item.latest_date ?? '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
