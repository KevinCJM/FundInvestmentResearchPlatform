import React, { useEffect, useMemo, useRef, useState } from 'react';

export interface FilterOption {
  value: string;
  label: string;
  count?: number;
}

interface FilterDropdownProps {
  label: string;
  options: FilterOption[];
  selected: string[];
  placeholder?: string;
  onChange: (next: string[]) => void;
}

const highlightMatch = (label: string, keyword: string) => {
  if (!keyword) {
    return label;
  }
  const lowerLabel = label.toLowerCase();
  const index = lowerLabel.indexOf(keyword.toLowerCase());
  if (index === -1) {
    return label;
  }
  const before = label.slice(0, index);
  const match = label.slice(index, index + keyword.length);
  const after = label.slice(index + keyword.length);
  return (
    <>
      {before}
      <span className="text-emerald-600 font-semibold">{match}</span>
      {after}
    </>
  );
};

export default function FilterDropdown({
  label,
  options,
  selected,
  placeholder = '全部',
  onChange,
}: FilterDropdownProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const handleClick = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    if (open) {
      document.addEventListener('mousedown', handleClick);
    }
    return () => {
      document.removeEventListener('mousedown', handleClick);
    };
  }, [open]);

  const filteredOptions = useMemo(() => {
    if (!search.trim()) {
      return options;
    }
    const keyword = search.trim().toLowerCase();
    return options.filter((option) => option.label.toLowerCase().includes(keyword));
  }, [options, search]);

  const toggleValue = (value: string) => {
    if (selected.includes(value)) {
      onChange(selected.filter((item) => item !== value));
    } else {
      onChange([...selected, value]);
    }
  };

  const reset = () => {
    setSearch('');
    onChange([]);
  };

  const summaryText = selected.length === 0 ? placeholder : `${selected.length} 项`;

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-600 shadow-sm hover:border-emerald-400 hover:text-emerald-600 focus:outline-none focus:ring-2 focus:ring-emerald-500"
      >
        <span>{label}</span>
        <span className="rounded bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-500">{summaryText}</span>
        <svg
          className={`h-4 w-4 transition-transform ${open ? 'rotate-180 text-emerald-500' : 'text-slate-400'}`}
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path
            fillRule="evenodd"
            d="M5.23 7.21a.75.75 0 011.06.02L10 10.585l3.71-3.354a.75.75 0 011.04 1.08l-4.25 3.85a.75.75 0 01-1.04 0l-4.25-3.85a.75.75 0 01.02-1.06z"
            clipRule="evenodd"
          />
        </svg>
      </button>

      {open && (
        <div className="absolute z-30 mt-2 w-64 rounded-xl border border-slate-200 bg-white shadow-xl">
          <div className="border-b border-slate-100 px-4 py-2">
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="搜索选项"
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-600 focus:border-emerald-500 focus:outline-none focus:ring-1 focus:ring-emerald-500"
            />
          </div>
          <div className="max-h-60 overflow-y-auto px-2 py-2">
            {filteredOptions.length === 0 ? (
              <div className="px-3 py-6 text-center text-sm text-slate-400">暂无匹配项</div>
            ) : (
              filteredOptions.map((option) => {
                const checked = selected.includes(option.value);
                return (
                  <label
                    key={option.value}
                    className="flex cursor-pointer items-center justify-between rounded-lg px-3 py-2 text-sm text-slate-600 hover:bg-emerald-50"
                  >
                    <div className="flex items-center gap-3">
                      <input
                        type="checkbox"
                        className="h-4 w-4 rounded border-slate-300 text-emerald-500 focus:ring-emerald-500"
                        checked={checked}
                        onChange={() => toggleValue(option.value)}
                      />
                      <span className="flex-1">
                        {highlightMatch(option.label, search)}
                      </span>
                    </div>
                    {typeof option.count === 'number' && (
                      <span className="text-xs font-medium text-slate-400">{option.count}</span>
                    )}
                  </label>
                );
              })
            )}
          </div>
          <div className="flex items-center justify-between border-t border-slate-100 px-4 py-2 text-xs text-slate-500">
            <span>已选 {selected.length} 项</span>
            <button type="button" onClick={reset} className="text-emerald-600 hover:text-emerald-500">
              清空
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
