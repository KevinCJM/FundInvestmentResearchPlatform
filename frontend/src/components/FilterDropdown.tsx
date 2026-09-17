import React, { useEffect, useId, useMemo, useRef, useState } from 'react';

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
      <span className="text-accent-600 font-semibold">{match}</span>
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
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const searchRef = useRef<HTMLInputElement | null>(null);
  const panelId = useId();

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

  useEffect(() => {
    if (!open) {
      return;
    }
    searchRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        setOpen(false);
        triggerRef.current?.focus();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
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
        ref={triggerRef}
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
        aria-controls={panelId}
        aria-haspopup="dialog"
        className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-600 shadow-sm hover:border-accent-400 hover:text-accent-600 focus:outline-none focus:ring-2 focus:ring-accent-500"
      >
        <span>{label}</span>
        <span className="rounded-lg bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-600">{summaryText}</span>
        <svg
          className={`h-4 w-4 transition-transform ${open ? 'rotate-180 text-accent-600' : 'text-slate-600'}`}
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
        <div id={panelId} role="dialog" aria-label={`${label}筛选`} className="absolute right-0 z-30 mt-2 w-64 max-w-[calc(100vw-2rem)] rounded-xl border border-slate-200 bg-white shadow-xl sm:left-0 sm:right-auto">
          <div className="border-b border-slate-100 px-4 py-2">
            <input
              ref={searchRef}
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="搜索选项"
              aria-label={`搜索${label}选项`}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-600 focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500"
            />
          </div>
          <div className="max-h-60 overflow-y-auto px-2 py-2">
            {filteredOptions.length === 0 ? (
              <div className="px-3 py-6 text-center text-sm text-slate-600">暂无匹配项</div>
            ) : (
              filteredOptions.map((option) => {
                const checked = selected.includes(option.value);
                return (
                  <label
                    key={option.value}
                    className="flex cursor-pointer items-center justify-between rounded-lg px-3 py-2 text-sm text-slate-600 hover:bg-accent-50"
                  >
                    <div className="flex items-center gap-3">
                      <input
                        type="checkbox"
                        className="h-4 w-4 rounded-lg border-slate-300 text-accent-600 focus:ring-accent-500"
                        checked={checked}
                        onChange={() => toggleValue(option.value)}
                      />
                      <span className="flex-1">
                        {highlightMatch(option.label, search)}
                      </span>
                    </div>
                    {typeof option.count === 'number' && (
                      <span className="text-xs font-medium text-slate-600">{option.count}</span>
                    )}
                  </label>
                );
              })
            )}
          </div>
          <div className="flex items-center justify-between border-t border-slate-100 px-4 py-2 text-xs text-slate-600">
            <span>已选 {selected.length} 项</span>
            <button type="button" onClick={reset} className="text-accent-600 hover:text-accent-600">
              清空
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export interface SnapshotMetricOption {
  field: string;
  label: string;
  available: boolean;
  metric_source?: string | null;
  metric_source_label?: string | null;
  metric_type?: string | null;
  metric_type_label?: string | null;
  indicator_id?: string | null;
  presentation?: {
    source?: string | null;
    indicator_type?: string | null;
    category?: string | null;
    category_label?: string | null;
  } | null;
}

interface SnapshotMetricSelectorProps {
  options: SnapshotMetricOption[];
  selected: string[];
  onChange: (next: string[]) => void;
  maxSelected?: number;
}

const SNAPSHOT_SOURCE_LABELS: Record<string, string> = {
  built_in: '内置指标',
  custom: '工作区指标',
  system_derived: '系统衍生指标',
};

const SNAPSHOT_TYPE_LABELS: Record<string, string> = {
  return: '收益型指标',
  risk: '风险型指标',
  risk_adjusted: '收益风险性价比指标',
  path: '路径与回撤指标',
  market_liquidity: '交易与流动性指标',
  scale: '规模指标',
  other: '其他指标',
};

const snapshotMetricSource = (option: SnapshotMetricOption) => (
  option.metric_source
  || option.presentation?.source
  || (option.indicator_id?.startsWith('builtin-') ? 'built_in' : null)
  || (option.indicator_id ? 'custom' : 'system_derived')
);

const snapshotMetricType = (option: SnapshotMetricOption) => (
  option.metric_type
  || option.presentation?.indicator_type
  || option.presentation?.category
  || (option.field.startsWith('return_') ? 'return' : null)
  || (option.field.includes('drawdown') ? 'path' : null)
  || (option.field.includes('volatility') ? 'risk' : null)
  || (option.field.includes('sharpe') || option.field.includes('calmar') ? 'risk_adjusted' : null)
  || (option.field === 'current_size' ? 'scale' : null)
  || 'other'
);

const snapshotSourceLabel = (option: SnapshotMetricOption) => (
  option.metric_source_label || SNAPSHOT_SOURCE_LABELS[snapshotMetricSource(option)] || '其他来源'
);

const snapshotTypeLabel = (option: SnapshotMetricOption) => (
  option.metric_type_label
  || option.presentation?.category_label
  || SNAPSHOT_TYPE_LABELS[snapshotMetricType(option)]
  || '其他指标'
);

export function SnapshotMetricSelector({
  options,
  selected,
  onChange,
  maxSelected = 8,
}: SnapshotMetricSelectorProps) {
  const [open, setOpen] = useState(false);
  const [source, setSource] = useState('all');
  const [type, setType] = useState('all');
  const [pendingMetric, setPendingMetric] = useState('');
  const [limitMessage, setLimitMessage] = useState('');
  const containerRef = useRef<HTMLDivElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const sourceRef = useRef<HTMLSelectElement | null>(null);
  const panelId = useId();

  useEffect(() => {
    if (!open) return undefined;
    sourceRef.current?.focus();
    const handlePointerDown = (event: MouseEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      setOpen(false);
      triggerRef.current?.focus();
    };
    document.addEventListener('mousedown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [open]);

  const sources = useMemo(() => Array.from(new Map(options.map((option) => [
    snapshotMetricSource(option),
    snapshotSourceLabel(option),
  ])).entries()), [options]);
  const sourceOptions = useMemo(() => (
    source === 'all' ? options : options.filter((option) => snapshotMetricSource(option) === source)
  ), [options, source]);
  const types = useMemo(() => Array.from(new Map(sourceOptions.map((option) => [
    snapshotMetricType(option),
    snapshotTypeLabel(option),
  ])).entries()), [sourceOptions]);
  const metricOptions = useMemo(() => sourceOptions.filter((option) => (
    type === 'all' || snapshotMetricType(option) === type
  )), [sourceOptions, type]);
  const selectedOptions = selected.flatMap((field) => {
    const option = options.find((item) => item.field === field);
    return option ? [option] : [];
  });

  const addMetric = (field: string) => {
    setPendingMetric('');
    if (!field || selected.includes(field)) return;
    if (selected.length >= maxSelected) {
      setLimitMessage(`最多展示 ${maxSelected} 个快照指标。`);
      return;
    }
    setLimitMessage('');
    onChange([...selected, field]);
  };

  return (
    <div ref={containerRef} className="relative">
      <button
        ref={triggerRef}
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        aria-haspopup="dialog"
        onClick={() => setOpen((current) => !current)}
        className="inline-flex min-h-10 items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-700 shadow-sm hover:border-accent-400 focus:outline-none focus:ring-2 focus:ring-accent-500"
      >
        <span>展示快照指标</span>
        <span className="rounded-lg bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
          {selected.length ? `已选 ${selected.length}/${maxSelected}` : '未选择'}
        </span>
        <svg
          className={`h-4 w-4 transition-transform ${open ? 'rotate-180 text-accent-600' : 'text-slate-600'}`}
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
        <div
          id={panelId}
          role="dialog"
          aria-label="展示快照指标选择"
          className="absolute right-0 z-40 mt-2 w-[34rem] max-w-[calc(100vw-2rem)] rounded-xl border border-slate-200 bg-white p-4 shadow-xl"
        >
          <div className="grid gap-3 sm:grid-cols-3">
            <label className="text-xs font-semibold text-slate-600">
              指标来源
              <select
                ref={sourceRef}
                aria-label="快照指标来源"
                value={source}
                onChange={(event) => {
                  setSource(event.target.value);
                  setType('all');
                  setPendingMetric('');
                }}
                className="mt-1 min-h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm font-medium text-slate-700 focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500"
              >
                <option value="all">全部来源</option>
                {sources.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
            </label>
            <label className="text-xs font-semibold text-slate-600">
              指标类型
              <select
                aria-label="快照指标类型"
                value={type}
                onChange={(event) => {
                  setType(event.target.value);
                  setPendingMetric('');
                }}
                className="mt-1 min-h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm font-medium text-slate-700 focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500"
              >
                <option value="all">全部类型</option>
                {types.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
            </label>
            <label className="text-xs font-semibold text-slate-600">
              指标名称
              <select
                aria-label="快照指标名称"
                value={pendingMetric}
                onChange={(event) => addMetric(event.target.value)}
                className="mt-1 min-h-11 w-full rounded-xl border border-slate-200 bg-white px-3 text-sm font-medium text-slate-700 focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500"
              >
                <option value="">选择指标名称</option>
                {metricOptions.map((option) => (
                  <option
                    key={option.field}
                    value={option.field}
                    disabled={!option.available || selected.includes(option.field)}
                  >
                    {option.label}{!option.available ? '（快照未就绪）' : selected.includes(option.field) ? '（已选择）' : ''}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="mt-4 border-t border-slate-100 pt-3">
            <div className="flex min-h-8 flex-wrap gap-2" aria-live="polite">
              {selectedOptions.length === 0 ? (
                <span className="text-xs text-slate-600">尚未选择展示指标</span>
              ) : selectedOptions.map((option) => (
                <button
                  key={option.field}
                  type="button"
                  aria-label={`移除快照指标 ${option.label}`}
                  onClick={() => onChange(selected.filter((field) => field !== option.field))}
                  className="rounded-full bg-accent-50 px-3 py-1 text-xs font-semibold text-accent-700 hover:bg-accent-100 focus:outline-none focus:ring-2 focus:ring-accent-500"
                >
                  {option.label} ×
                </button>
              ))}
            </div>
            {limitMessage && <p role="status" className="mt-2 text-xs font-medium text-amber-700">{limitMessage}</p>}
            <div className="mt-3 flex items-center justify-between text-xs">
              <span className="text-slate-600">按来源、类型逐级缩小范围，再选择指标名称。</span>
              <div className="flex gap-3">
                <button type="button" onClick={() => onChange([])} className="font-semibold text-slate-600 hover:text-slate-700">清空</button>
                <button type="button" onClick={() => setOpen(false)} className="font-semibold text-accent-700 hover:text-accent-600">完成</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export interface SearchDropdownProps<T> {
  label: string;
  value: string;
  items: T[];
  getItemKey: (item: T, index: number) => React.Key;
  renderItem: (item: T, index: number) => React.ReactNode;
  onChange: (value: string) => void;
  onSearch: () => void | Promise<void>;
  loading?: boolean;
  placeholder?: string;
  searchLabel?: string;
  emptyMessage?: string;
  className?: string;
}

/**
 * 可复用的显式搜索下拉框。
 *
 * 焦点在输入框、搜索按钮或结果列表中时保持展开；焦点离开整个控件、
 * 点击控件外部或按 Esc 时收起。搜索结果由调用方提供，便于复用远程搜索。
 */
export function SearchDropdown<T>({
  label,
  value,
  items,
  getItemKey,
  renderItem,
  onChange,
  onSearch,
  loading = false,
  placeholder,
  searchLabel = '搜索',
  emptyMessage = '未找到匹配结果',
  className = '',
}: SearchDropdownProps<T>) {
  const [open, setOpen] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const suppressNextFocusOpenRef = useRef(false);
  const listboxId = useId();

  useEffect(() => {
    if (!open) return undefined;
    const dismissOutside = (event: MouseEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const dismissWithEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      suppressNextFocusOpenRef.current = true;
      setOpen(false);
      inputRef.current?.focus();
    };
    document.addEventListener('mousedown', dismissOutside);
    document.addEventListener('keydown', dismissWithEscape);
    return () => {
      document.removeEventListener('mousedown', dismissOutside);
      document.removeEventListener('keydown', dismissWithEscape);
    };
  }, [open]);

  const runSearch = () => {
    setHasSearched(true);
    setOpen(true);
    void onSearch();
  };

  return (
    <div
      ref={containerRef}
      className={`relative min-w-0 ${className}`}
      onFocusCapture={() => {
        if (suppressNextFocusOpenRef.current) {
          suppressNextFocusOpenRef.current = false;
          return;
        }
        if (hasSearched) setOpen(true);
      }}
      onBlurCapture={(event) => {
        const nextTarget = event.relatedTarget as Node | null;
        if (!nextTarget || !event.currentTarget.contains(nextTarget)) setOpen(false);
      }}
    >
      <div className="flex min-w-0 gap-2">
        <input
          ref={inputRef}
          role="combobox"
          aria-label={label}
          aria-expanded={open}
          aria-controls={open ? listboxId : undefined}
          aria-autocomplete="list"
          autoComplete="off"
          value={value}
          onChange={(event) => {
            setHasSearched(false);
            setOpen(false);
            onChange(event.target.value);
          }}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault();
              runSearch();
            }
          }}
          placeholder={placeholder}
          className="min-w-0 flex-1 rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-accent-500 focus:outline-none focus:ring-2 focus:ring-accent-500"
        />
        <button
          type="button"
          onClick={runSearch}
          disabled={loading}
          className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-accent-500"
        >
          {loading ? '搜索中' : searchLabel}
        </button>
      </div>

      {open && (
        <ul
          id={listboxId}
          role="listbox"
          aria-label={`${label}结果`}
          className="absolute inset-x-0 top-full z-40 mt-2 max-h-64 overflow-auto rounded-xl border border-slate-200 bg-white shadow-xl"
        >
          {loading ? (
            <li role="status" className="px-3 py-4 text-center text-sm text-slate-600">正在搜索…</li>
          ) : items.length ? items.map((item, index) => (
            <li key={getItemKey(item, index)} role="option" aria-selected="false">
              {renderItem(item, index)}
            </li>
          )) : (
            <li role="status" className="px-3 py-4 text-center text-sm text-slate-600">{emptyMessage}</li>
          )}
        </ul>
      )}
    </div>
  );
}
