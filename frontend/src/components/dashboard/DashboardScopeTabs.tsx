import { useRef } from 'react';
import type { DashboardKind } from './types';

const options: { value: DashboardKind; label: string; description: string }[] = [
  { value: 'all', label: '全市场', description: 'ETF 与场外公募基金并列总览' },
  { value: 'etf', label: 'ETF', description: '交易所上市基金' },
  { value: 'fund', label: '场外公募基金', description: '非 ETF 的场外基金份额' },
];

interface DashboardScopeTabsProps {
  value: DashboardKind;
  onChange: (next: DashboardKind) => void;
}

export default function DashboardScopeTabs({ value, onChange }: DashboardScopeTabsProps) {
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);

  const moveFocus = (currentIndex: number, nextIndex: number) => {
    const normalized = (nextIndex + options.length) % options.length;
    const next = options[normalized];
    onChange(next.value);
    tabRefs.current[normalized]?.focus();
  };

  return (
    <div
      role="tablist"
      aria-label="驾驶舱产品范围"
      className="flex max-w-full gap-1 overflow-x-auto rounded-2xl border border-white/20 bg-white/10 p-1 backdrop-blur"
    >
      {options.map((option, index) => (
        <button
          key={option.value}
          ref={(node) => { tabRefs.current[index] = node; }}
          type="button"
          id={`dashboard-tab-${option.value}`}
          role="tab"
          aria-selected={value === option.value}
          aria-controls="dashboard-content"
          tabIndex={value === option.value ? 0 : -1}
          title={option.description}
          onClick={() => onChange(option.value)}
          onKeyDown={(event) => {
            if (event.key === 'ArrowRight') {
              event.preventDefault();
              moveFocus(index, index + 1);
            } else if (event.key === 'ArrowLeft') {
              event.preventDefault();
              moveFocus(index, index - 1);
            } else if (event.key === 'Home') {
              event.preventDefault();
              moveFocus(index, 0);
            } else if (event.key === 'End') {
              event.preventDefault();
              moveFocus(index, options.length - 1);
            }
          }}
          className={`shrink-0 rounded-xl px-4 py-2 text-sm font-semibold transition focus:outline-none focus:ring-2 focus:ring-white/80 ${
            value === option.value
              ? 'bg-white text-indigo-700 shadow-sm'
              : 'text-white/85 hover:bg-white/10 hover:text-white'
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
