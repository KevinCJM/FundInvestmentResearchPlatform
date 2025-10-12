export interface AnnualMetrics {
  cumulative: number | null;
  volatility: number | null;
  sharpe: number | null;
  maxDrawdown: number | null;
}

export interface AnnualMetricsResult {
  years: number[];
  series: Record<string, Record<number, AnnualMetrics>>;
}

const DEFAULT_ANN_FACTOR = 252;

function toFiniteNumber(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function computeDrawdown(values: number[]): number {
  if (values.length === 0) return Number.NaN;
  let peak = values[0];
  let worst = 0;
  for (const val of values) {
    if (val > peak) {
      peak = val;
    }
    if (peak !== 0) {
      const drawdown = val / peak - 1;
      if (drawdown < worst) {
        worst = drawdown;
      }
    }
  }
  return worst;
}

export function computeAnnualMetrics(
  dates: string[] | undefined,
  seriesMap: Record<string, Array<number | null | undefined>> | undefined,
  annFactor = DEFAULT_ANN_FACTOR,
): AnnualMetricsResult {
  const metricsBySeries: Record<string, Record<number, AnnualMetrics>> = {};
  const yearsObserved = new Set<number>();

  if (!Array.isArray(dates) || !seriesMap) {
    return { years: [], series: metricsBySeries };
  }

  const parsedDates = dates.map((d) => {
    const date = new Date(d);
    const time = date.getTime();
    if (Number.isNaN(time)) return null;
    return { date, year: date.getFullYear() };
  });

  for (const [name, rawValues] of Object.entries(seriesMap)) {
    if (!Array.isArray(rawValues)) continue;
    const limit = Math.min(parsedDates.length, rawValues.length);
    const entries: Array<{ year: number; value: number }> = [];
    for (let i = 0; i < limit; i += 1) {
      const meta = parsedDates[i];
      if (!meta) continue;
      const value = toFiniteNumber(rawValues[i]);
      if (value === null) continue;
      entries.push({ year: meta.year, value });
    }
    if (!entries.length) continue;

    const perYear: Record<number, { navs: number[]; returns: number[] }> = {};
    let previous: { year: number; value: number } | null = null;
    for (const item of entries) {
      const { year, value } = item;
      let bucket = perYear[year];
      if (!bucket) {
        bucket = { navs: [], returns: [] };
        perYear[year] = bucket;
      }
      bucket.navs.push(value);
      if (previous && previous.year === year) {
        const ret = value / previous.value - 1;
        if (Number.isFinite(ret)) {
          bucket.returns.push(ret);
        }
      }
      previous = { year, value };
    }

    const stats: Record<number, AnnualMetrics> = {};
    for (const [yearStr, bucket] of Object.entries(perYear)) {
      const year = Number(yearStr);
      yearsObserved.add(year);
      const navs = bucket.navs;
      const returns = bucket.returns;

      let cumulative: number | null = null;
      if (navs.length >= 2) {
        const first = navs[0];
        const last = navs[navs.length - 1];
        if (first !== 0) {
          const ratio = last / first - 1;
          if (Number.isFinite(ratio)) {
            cumulative = ratio;
          }
        }
      } else if (navs.length === 1) {
        cumulative = 0;
      }

      let volatility: number | null = null;
      let sharpe: number | null = null;
      if (returns.length >= 2) {
        const mean = returns.reduce((acc, cur) => acc + cur, 0) / returns.length;
        const variance = returns.reduce((acc, cur) => acc + (cur - mean) ** 2, 0) / (returns.length - 1);
        const dailyVol = Math.sqrt(Math.max(variance, 0));
        const vol = dailyVol * Math.sqrt(annFactor);
        if (Number.isFinite(vol)) {
          volatility = vol;
        }
        const annualMean = mean * annFactor;
        if (volatility && Math.abs(volatility) > 1e-12 && Number.isFinite(annualMean)) {
          sharpe = annualMean / volatility;
        }
      }

      let maxDrawdown: number | null = null;
      if (navs.length > 0) {
        const dd = computeDrawdown(navs);
        if (Number.isFinite(dd)) {
          maxDrawdown = dd;
        }
      }

      stats[year] = {
        cumulative,
        volatility,
        sharpe,
        maxDrawdown,
      };
    }

    metricsBySeries[name] = stats;
  }

  const observed = Array.from(yearsObserved).sort((a, b) => a - b);
  if (observed.length === 0) {
    return { years: [], series: metricsBySeries };
  }

  const firstYear = observed[0];
  const lastYear = observed[observed.length - 1];
  const years: number[] = [];
  for (let y = firstYear; y <= lastYear; y += 1) {
    years.push(y);
  }

  return { years, series: metricsBySeries };
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
      { label: `${year}年化波动率(%)`, values: getValues('volatility', 100) },
      { label: `${year}夏普比率`, values: getValues('sharpe') },
      { label: `${year}最大回撤(%)`, values: getValues('maxDrawdown', 100) },
    ];
  });
}
