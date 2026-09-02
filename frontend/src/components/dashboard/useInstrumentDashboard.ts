import { useCallback, useEffect, useMemo, useState } from 'react';
import type {
  DashboardFilters,
  DashboardKind,
  InstrumentAnalyticsResponse,
  InstrumentRankingsResponse,
  InstrumentTrendResponse,
  Loadable,
  SegmentKind,
} from './types';

const emptyLoadable = <T,>(): Loadable<T> => ({ data: null, loading: false, error: null });

const appendFilters = (
  params: URLSearchParams,
  filters: DashboardFilters,
  options: { includeStatus?: boolean } = {},
) => {
  (Object.keys(filters) as (keyof DashboardFilters)[]).forEach((key) => {
    if (key === 'status' && options.includeStatus === false) {
      return;
    }
    filters[key].forEach((value) => params.append(key, value));
  });
};

const fetchJson = async <T,>(url: string, signal: AbortSignal): Promise<T> => {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
};

interface UseInstrumentDashboardArgs {
  kind: DashboardKind;
  filters: DashboardFilters;
  rankingMetric: string;
}

export function useInstrumentDashboard({ kind, filters, rankingMetric }: UseInstrumentDashboardArgs) {
  const [overview, setOverview] = useState<Loadable<InstrumentAnalyticsResponse>>(() => emptyLoadable());
  const [trend, setTrend] = useState<Loadable<InstrumentTrendResponse>>(() => emptyLoadable());
  const [rankings, setRankings] = useState<Record<SegmentKind, Loadable<InstrumentRankingsResponse>>>(() => ({
    etf: emptyLoadable(),
    fund: emptyLoadable(),
  }));
  const [reloadVersion, setReloadVersion] = useState(0);

  const serializedFilters = useMemo(() => JSON.stringify(filters), [filters]);

  useEffect(() => {
    const controller = new AbortController();
    const analyticsParams = new URLSearchParams({ kind });
    appendFilters(analyticsParams, filters);
    const trendParams = new URLSearchParams({ kind, dimension: 'all' });
    appendFilters(trendParams, filters);

    setOverview((previous) => ({ ...previous, loading: true, error: null }));
    setTrend((previous) => ({ ...previous, loading: true, error: null }));

    fetchJson<InstrumentAnalyticsResponse>(
      `/api/instruments/analytics?${analyticsParams.toString()}`,
      controller.signal,
    )
      .then((data) => setOverview({ data, loading: false, error: null }))
      .catch((error: Error) => {
        if (error.name !== 'AbortError') {
          setOverview((previous) => ({ ...previous, loading: false, error: '市场总览加载失败，请稍后重试。' }));
        }
      });

    fetchJson<InstrumentTrendResponse>(
      `/api/instruments/analytics/trend?${trendParams.toString()}`,
      controller.signal,
    )
      .then((data) => setTrend({ data, loading: false, error: null }))
      .catch((error: Error) => {
        if (error.name !== 'AbortError') {
          setTrend((previous) => ({ ...previous, loading: false, error: '发行与成立趋势暂时不可用。' }));
        }
      });

    return () => controller.abort();
    // serializedFilters intentionally makes nested filter arrays observable.
  }, [kind, reloadVersion, serializedFilters]);

  useEffect(() => {
    const controller = new AbortController();
    const targetKinds: SegmentKind[] = kind === 'all' ? ['etf', 'fund'] : [kind];

    setRankings((previous) => ({
      etf: targetKinds.includes('etf')
        ? { ...previous.etf, loading: true, error: null }
        : emptyLoadable(),
      fund: targetKinds.includes('fund')
        ? { ...previous.fund, loading: true, error: null }
        : emptyLoadable(),
    }));

    targetKinds.forEach((targetKind) => {
      const params = new URLSearchParams({
        kind: targetKind,
        metric: rankingMetric,
        sort_dir: 'desc',
        page: '1',
        page_size: '10',
        active_only: 'true',
      });
      // Rankings deliberately ignore lifecycle status and always use active_only=true.
      appendFilters(params, filters, { includeStatus: false });
      fetchJson<InstrumentRankingsResponse>(
        `/api/instruments/analytics/rankings?${params.toString()}`,
        controller.signal,
      )
        .then((data) => {
          setRankings((previous) => ({
            ...previous,
            [targetKind]: { data, loading: false, error: null },
          }));
        })
        .catch((error: Error) => {
          if (error.name !== 'AbortError') {
            setRankings((previous) => ({
              ...previous,
              [targetKind]: {
                ...previous[targetKind],
                loading: false,
                error: `${targetKind === 'etf' ? 'ETF' : '场外公募基金'}排行榜加载失败。`,
              },
            }));
          }
        });
    });

    return () => controller.abort();
  }, [kind, rankingMetric, reloadVersion, serializedFilters]);

  const reload = useCallback(() => setReloadVersion((version) => version + 1), []);

  return { overview, trend, rankings, reload };
}
