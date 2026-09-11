export interface AnnualMetrics {
  cumulative: number | null;
  volatility: number | null;
  annualReturn: number | null;
  annualVolatility: number | null;
  sharpe: number | null;
  maxDrawdown: number | null;
  calmar: number | null;
}

export interface AnnualMetricsResult {
  years: number[];
  series: Record<string, Record<number, AnnualMetrics>>;
}

export function buildAnnualMetricRows(
  columns: string[],
  annual: AnnualMetricsResult,
): Array<{ label: string; values: Array<number | null | undefined> }> {
  if (!annual.years.length) return [];
  return annual.years.flatMap((year) => {
    const series = annual.series;
    const getValues = (key: keyof AnnualMetrics, scale = 1) =>
      columns.map((name) => {
        const metrics = series[name]?.[year];
        if (!metrics) return null;
        const value = metrics[key];
        if (value === null || value === undefined || !Number.isFinite(value)) return null;
        return value * scale;
      });

    return [
      { label: `${year}累计收益率(%)`, values: getValues('cumulative', 100) },
      { label: `${year}波动率(%)`, values: getValues('volatility', 100) },
      { label: `${year}年化收益率(%)`, values: getValues('annualReturn', 100) },
      { label: `${year}年化波动率(%)`, values: getValues('annualVolatility', 100) },
      { label: `${year}夏普比率`, values: getValues('sharpe') },
      { label: `${year}最大回撤(%)`, values: getValues('maxDrawdown', 100) },
      { label: `${year}卡玛比率`, values: getValues('calmar') },
    ];
  });
}
