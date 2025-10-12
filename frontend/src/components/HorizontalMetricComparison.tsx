import React, { useCallback, useEffect, useMemo, useState } from 'react';
import ReactECharts from 'echarts-for-react';

export interface HorizontalMetricRow {
  label: string;
  values: Array<number | null | undefined>;
  formatter?: (value: number) => string;
  reverseScale?: boolean;
}

export const DEFAULT_METRIC_TABLE_HEIGHT = 320;

interface HorizontalMetricComparisonProps {
  columns: string[];
  rows: HorizontalMetricRow[];
  emptyText?: string;
  height?: number;
}

function formatExportCell(value: string): string {
  if (value.includes('"') || value.includes(',') || value.includes('\n')) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
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
  height = DEFAULT_METRIC_TABLE_HEIGHT,
}: HorizontalMetricComparisonProps) {
  const hasData = columns.length > 0 && rows.length > 0;

  const exportMatrix = useMemo(() => {
    if (!hasData) return null;
    const header = ['指标', ...columns];
    const data = rows.map(row => [
      row.label,
      ...columns.map((_, idx) => {
        const rawValue = row.values[idx];
        const numeric = Number(rawValue);
        if (Number.isFinite(numeric)) {
          return formatValue(numeric, row.formatter);
        }
        return emptyText;
      }),
    ]);
    return [header, ...data];
  }, [columns, emptyText, hasData, rows]);

  const handleExport = useCallback(() => {
    if (!exportMatrix) return;
    const lines = exportMatrix.map(line => line.map(formatExportCell).join(','));
    const csvContent = '\ufeff' + lines.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const filename = `指标横向对比_${new Date().toISOString().replace(/[:T]/g, '-').split('.')[0]}.csv`;
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [exportMatrix]);

  if (!hasData) {
    return <p className="text-xs text-gray-500">{emptyText}</p>;
  }

  return (
    <div className="space-y-2">
      <div className="flex justify-end">
        <button
          type="button"
          onClick={handleExport}
          className="rounded border border-gray-300 bg-white px-3 py-1 text-xs text-gray-700 shadow-sm transition hover:bg-gray-50"
        >
          导出表格
        </button>
      </div>
      <div className="overflow-x-auto">
        <div
          className="overflow-y-auto"
          style={{ maxHeight: height, minHeight: height, height }}
        >
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
      </div>
    </div>
  );
}

interface PerformanceQuadrantChartProps {
  columns: string[];
  rows: HorizontalMetricRow[];
  defaultXAxis?: string;
  defaultYAxis?: string;
  height?: number;
  emptyText?: string;
}

export function PerformanceQuadrantChart({
  columns,
  rows,
  defaultXAxis = '年化波动率(%)',
  defaultYAxis = '累计收益率(%)',
  height = 320,
  emptyText = '暂无可绘制的数据',
}: PerformanceQuadrantChartProps) {
  interface SanitizedMetricRow {
    label: string;
    values: Array<number | null>;
    formatter?: (value: number) => string;
  }

  const sanitizedRows: SanitizedMetricRow[] = useMemo(() => {
    if (!Array.isArray(rows)) return [];
    return rows
      .map((row) => {
        const values = columns.map((_, idx) => {
          const raw = row.values?.[idx];
          const num = Number(raw);
          return Number.isFinite(num) ? num : null;
        });
        const hasValid = values.some((v) => v !== null);
        if (!hasValid) return null;
        return {
          label: row.label,
          values,
          formatter: row.formatter,
        };
      })
      .filter((row): row is SanitizedMetricRow => row !== null);
  }, [columns, rows]);

  const metricLabels = sanitizedRows.map((row) => row.label);

  const resolveDefault = useCallback(
    (preferred: string, fallbacks: string[]): string => {
      for (const candidate of [preferred, ...fallbacks]) {
        if (candidate && metricLabels.includes(candidate)) {
          return candidate;
        }
      }
      return metricLabels[0] ?? '';
    },
    [metricLabels],
  );

  const [xMetric, setXMetric] = useState<string>(() =>
    resolveDefault(defaultXAxis, ['年化波动率(%)', '波动率(%)']),
  );
  const [yMetric, setYMetric] = useState<string>(() =>
    resolveDefault(defaultYAxis, ['累计收益率(%)', '年化收益率(%)']),
  );

  useEffect(() => {
    if (!metricLabels.length) {
      if (xMetric) setXMetric('');
      if (yMetric) setYMetric('');
      return;
    }

    const nextX = metricLabels.includes(xMetric)
      ? xMetric
      : resolveDefault(defaultXAxis, ['年化波动率(%)', '波动率(%)']);
    if (nextX !== xMetric) {
      setXMetric(nextX);
    }

    const nextY = metricLabels.includes(yMetric)
      ? yMetric
      : resolveDefault(defaultYAxis, ['累计收益率(%)', '年化收益率(%)']);
    if (nextY !== yMetric) {
      setYMetric(nextY);
    }
  }, [defaultXAxis, defaultYAxis, metricLabels, resolveDefault, xMetric, yMetric]);

  if (sanitizedRows.length === 0 || !metricLabels.length) {
    return <p className="text-xs text-gray-500">{emptyText}</p>;
  }

  const selectorControls = (
    <div className="flex flex-wrap gap-4 text-xs">
      <label className="flex flex-col gap-1 text-gray-600">
        <span>风险指标</span>
        <select
          className="rounded border border-gray-300 px-2 py-1"
          value={xMetric}
          onChange={(e) => setXMetric(e.target.value)}
        >
          {metricLabels.map((label) => (
            <option key={`quad-x-${label}`} value={label}>
              {label}
            </option>
          ))}
        </select>
      </label>
      <label className="flex flex-col gap-1 text-gray-600">
        <span>收益指标</span>
        <select
          className="rounded border border-gray-300 px-2 py-1"
          value={yMetric}
          onChange={(e) => setYMetric(e.target.value)}
        >
          {metricLabels.map((label) => (
            <option key={`quad-y-${label}`} value={label}>
              {label}
            </option>
          ))}
        </select>
      </label>
    </div>
  );

  const xRow = sanitizedRows.find((row) => row.label === xMetric);
  const yRow = sanitizedRows.find((row) => row.label === yMetric);

  const formatValue = (row: SanitizedMetricRow | undefined, value: number): string => {
    if (!Number.isFinite(value)) return '-';
    if (row?.formatter) {
      return row.formatter(value);
    }
    if (row && /%/.test(row.label)) {
      return `${value.toFixed(2)}%`;
    }
    return value.toFixed(2);
  };

  const scatterData = useMemo(() => {
    if (!xRow || !yRow) return [] as Array<{ name: string; value: [number, number] }>;
    return columns
      .map((name, idx) => {
        const xVal = xRow.values[idx];
        const yVal = yRow.values[idx];
        if (xVal === null || yVal === null) return null;
        return { name, value: [xVal, yVal] as [number, number] };
      })
      .filter((item): item is { name: string; value: [number, number] } => item !== null);
  }, [columns, xRow, yRow]);

  if (!xRow || !yRow || scatterData.length === 0) {
    return (
      <div className="space-y-3">
        {selectorControls}
        <p className="text-xs text-gray-500">{emptyText}</p>
      </div>
    );
  }

  const collectValues = (row: SanitizedMetricRow) =>
    row.values.filter((v): v is number => v !== null && Number.isFinite(v));

  const computeBounds = (values: number[]) => {
    if (!values.length) return { min: 0, max: 0 };
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) {
      return { min: 0, max: 0 };
    }
    if (minVal === maxVal) {
      const pad = Math.abs(minVal) * 0.1 || 1;
      return { min: minVal - pad, max: maxVal + pad };
    }
    const pad = (maxVal - minVal) * 0.1;
    return { min: minVal - pad, max: maxVal + pad };
  };

  const xBounds = computeBounds(collectValues(xRow));
  const yBounds = computeBounds(collectValues(yRow));

  const option = {
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        const [x, y] = params.value || [];
        return `${params.name}<br/>${xMetric}：${formatValue(xRow, Number(x))}<br/>${yMetric}：${formatValue(yRow, Number(y))}`;
      },
    },
    grid: { top: 48, right: 16, left: 60, bottom: 48 },
    xAxis: {
      type: 'value',
      name: xMetric,
      axisLabel: {
        formatter: (v: number) => formatValue(xRow, Number(v)),
      },
      splitLine: { lineStyle: { type: 'dashed' } },
      min: xBounds.min,
      max: xBounds.max,
    },
    yAxis: {
      type: 'value',
      name: yMetric,
      axisLabel: {
        formatter: (v: number) => formatValue(yRow, Number(v)),
      },
      splitLine: { lineStyle: { type: 'dashed' } },
      min: yBounds.min,
      max: yBounds.max,
    },
    series: [
      {
        type: 'scatter',
        symbolSize: 12,
        data: scatterData,
        label: {
          show: true,
          position: 'right',
          formatter: '{b}',
        },
        itemStyle: { color: '#2563eb' },
      },
    ],
  };

  return (
    <div className="space-y-3">
      {selectorControls}
      <ReactECharts style={{ height }} option={option} />
    </div>
  );
}
