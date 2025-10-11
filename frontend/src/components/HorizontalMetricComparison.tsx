import React from 'react';
import ReactECharts from 'echarts-for-react';

interface HorizontalMetricRow {
  label: string;
  values: Array<number | null | undefined>;
  formatter?: (value: number) => string;
  reverseScale?: boolean;
}

interface HorizontalMetricComparisonProps {
  columns: string[];
  rows: HorizontalMetricRow[];
  emptyText?: string;
}

function formatValue(value: number, formatter?: (value: number) => string): string {
  if (!Number.isFinite(value)) return '-';
  if (formatter) return formatter(value);
  return value.toFixed(2);
}

function computeCellStyle(
  value: number,
  min: number,
  max: number,
  reverse = false,
): { background: string; color: string } {
  if (!Number.isFinite(value)) return { background: '#f3f4f6', color: '#6b7280' };
  if (max <= min) return { background: '#d1fae5', color: '#065f46' };
  const ratioRaw = (value - min) / (max - min);
  const ratio = reverse ? 1 - ratioRaw : ratioRaw;
  const clamp = Math.max(0, Math.min(1, ratio));
  const from = [209, 250, 229];
  const to = [5, 150, 105];
  const interp = (a: number, b: number) => Math.round(a + (b - a) * clamp);
  const background = `rgb(${interp(from[0], to[0])},${interp(from[1], to[1])},${interp(from[2], to[2])})`;
  const color = clamp > 0.6 ? '#ffffff' : '#065f46';
  return { background, color };
}

export default function HorizontalMetricComparison({
  columns,
  rows,
  emptyText = '-',
}: HorizontalMetricComparisonProps) {
  if (!columns.length || !rows.length) {
    return <p className="text-xs text-gray-500">{emptyText}</p>;
  }

  return (
    <div className="overflow-auto">
      <table className="text-xs border" style={{ width: '100%', tableLayout: 'fixed' }}>
        <thead>
          <tr>
            <th className="border px-2 py-2">指标</th>
            {columns.map(col => (
              <th key={`metric-header-${col}`} className="border px-2 py-2">
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(row => {
            const numericValues = row.values.filter(v => Number.isFinite(v as number)) as number[];
            const min = numericValues.length ? Math.min(...numericValues) : 0;
            const max = numericValues.length ? Math.max(...numericValues) : 0;
            return (
              <tr key={`metric-row-${row.label}`}>
                <td className="border px-2 py-3 font-medium">{row.label}</td>
                {columns.map((_, idx) => {
                  const rawValue = row.values[idx];
                  if (!Number.isFinite(rawValue as number)) {
                    return (
                      <td key={`metric-cell-${row.label}-${idx}`} className="border px-2 py-3 text-right text-gray-400">
                        {emptyText}
                      </td>
                    );
                  }
                  const value = Number(rawValue);
                  const style = computeCellStyle(value, min, max, row.reverseScale);
                  return (
                    <td
                      key={`metric-cell-${row.label}-${idx}`}
                      className="border px-2 py-3 text-right"
                      style={style}
                    >
                      {formatValue(value, row.formatter)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export interface PerformanceQuadrantPoint {
  name: string;
  volatility: number;
  cumulativeReturn: number;
}

interface PerformanceQuadrantChartProps {
  points: PerformanceQuadrantPoint[];
  height?: number;
  emptyText?: string;
}

export function PerformanceQuadrantChart({
  points,
  height = 320,
  emptyText = '暂无可绘制的数据',
}: PerformanceQuadrantChartProps) {
  const validPoints = Array.isArray(points)
    ? points.filter(
        p =>
          Number.isFinite(p.volatility) &&
          Number.isFinite(p.cumulativeReturn),
      )
    : [];

  if (validPoints.length === 0) {
    return <p className="text-xs text-gray-500">{emptyText}</p>;
  }

  const data = validPoints.map(p => ({
    name: p.name,
    value: [p.volatility * 100, p.cumulativeReturn * 100],
  }));

  const vols = data.map(d => d.value[0]);
  const rets = data.map(d => d.value[1]);
  const computeBounds = (values: number[]) => {
    if (!values.length) return { min: 0, max: 0 };
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) return { min: 0, max: 0 };
    if (minVal === maxVal) {
      const pad = Math.abs(minVal) * 0.1 || 5;
      return { min: minVal - pad, max: maxVal + pad };
    }
    const pad = (maxVal - minVal) * 0.1;
    return { min: minVal - pad, max: maxVal + pad };
  };
  const xBounds = computeBounds(vols);
  if (xBounds.min < 0) xBounds.min = 0;
  const yBounds = computeBounds(rets);

  const option = {
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        const [x, y] = params.value || [];
        return `${params.name}<br/>波动率：${Number(x).toFixed(2)}%<br/>累计收益率：${Number(y).toFixed(2)}%`;
      },
    },
    grid: { top: 16, right: 16, left: 48, bottom: 40 },
    xAxis: {
      type: 'value',
      name: '波动率(%)',
      axisLabel: { formatter: (v: number) => `${Number(v).toFixed(1)}%` },
      splitLine: { lineStyle: { type: 'dashed' } },
      min: xBounds.min,
      max: xBounds.max,
    },
    yAxis: {
      type: 'value',
      name: '累计收益率(%)',
      axisLabel: { formatter: (v: number) => `${Number(v).toFixed(1)}%` },
      splitLine: { lineStyle: { type: 'dashed' } },
      min: yBounds.min,
      max: yBounds.max,
    },
    series: [
      {
        type: 'scatter',
        symbolSize: 12,
        data,
        label: {
          show: true,
          position: 'right',
          formatter: '{b}',
        },
        itemStyle: { color: '#2563eb' },
      },
    ],
  };

  return <ReactECharts style={{ height }} option={option} />;
}
