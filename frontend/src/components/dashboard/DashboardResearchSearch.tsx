import { type FormEvent, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type { DashboardKind, SegmentKind } from './types';

interface DashboardResearchSearchProps {
  scope: DashboardKind;
}

export default function DashboardResearchSearch({ scope }: DashboardResearchSearchProps) {
  const [query, setQuery] = useState('');
  const navigate = useNavigate();

  const openResearch = (kind: SegmentKind) => {
    const params = new URLSearchParams({ kind });
    const keyword = query.trim();
    if (keyword) {
      params.set('q', keyword);
    }
    navigate(`/product-research/products?${params.toString()}`);
  };

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    openResearch(scope === 'fund' ? 'fund' : 'etf');
  };

  return (
    <form role="search" aria-label="进入产品研究" onSubmit={submit} className="mt-5 flex max-w-2xl flex-col gap-2 sm:flex-row">
      <label className="min-w-0 flex-1">
        <span className="sr-only">基金代码、名称或管理人</span>
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="搜索基金代码、名称或管理人"
          className="w-full rounded-xl border border-white/25 bg-white/10 px-4 py-2.5 text-sm text-white outline-none placeholder:text-indigo-200 focus:border-white/70 focus:ring-2 focus:ring-white/70"
        />
      </label>
      {scope === 'all' ? (
        <div className="grid shrink-0 grid-cols-2 gap-2">
          <button type="submit" className="rounded-xl bg-white px-4 py-2.5 text-sm font-semibold text-indigo-800 hover:bg-indigo-50 focus:outline-none focus:ring-2 focus:ring-white">搜 ETF</button>
          <button type="button" onClick={() => openResearch('fund')} className="rounded-xl border border-white/35 bg-white/10 px-4 py-2.5 text-sm font-semibold text-white hover:bg-white/20 focus:outline-none focus:ring-2 focus:ring-white">搜场外基金</button>
        </div>
      ) : (
        <button type="submit" className="shrink-0 rounded-xl bg-white px-4 py-2.5 text-sm font-semibold text-indigo-800 hover:bg-indigo-50 focus:outline-none focus:ring-2 focus:ring-white">
          搜索{scope === 'etf' ? 'ETF' : '场外基金'}
        </button>
      )}
    </form>
  );
}
