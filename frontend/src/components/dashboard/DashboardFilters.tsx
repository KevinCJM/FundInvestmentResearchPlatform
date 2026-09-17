import FilterDropdown from '../FilterDropdown';
import type { DashboardFilterKey, DashboardFilters, DashboardFilterOption } from './types';

const fields: { key: DashboardFilterKey; label: string }[] = [
  { key: 'fund_type', label: '基金类型' },
  { key: 'invest_type', label: '投资风格' },
  { key: 'status', label: '产品状态' },
  { key: 'management', label: '管理人' },
  { key: 'market', label: '市场' },
];

interface DashboardFiltersProps {
  filters: DashboardFilters;
  availableFilters?: Record<string, DashboardFilterOption[]>;
  onChange: (key: DashboardFilterKey, values: string[]) => void;
  onReset: () => void;
}

export default function DashboardFilterBar({ filters, availableFilters = {}, onChange, onReset }: DashboardFiltersProps) {
  const activeCount = Object.values(filters).reduce((sum, values) => sum + values.length, 0);
  return (
    <section aria-labelledby="dashboard-filter-heading" className="rounded-xl bg-white p-5 shadow-sm ring-1 ring-slate-100">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h2 id="dashboard-filter-heading" className="text-base font-semibold text-slate-900">细分筛选</h2>
          <p className="mt-1 text-xs leading-5 text-slate-600">结构与趋势包含全部状态；排行榜始终仅含存续且样本有效的产品。</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {fields.map((field) => (
            <FilterDropdown
              key={field.key}
              label={field.label}
              options={availableFilters[field.key] ?? []}
              selected={filters[field.key]}
              onChange={(values) => onChange(field.key, values)}
            />
          ))}
          {activeCount > 0 && (
            <button type="button" onClick={onReset} className="rounded-lg px-3 py-2 text-sm font-semibold text-accent-700 hover:bg-accent-50 focus:outline-none focus:ring-2 focus:ring-accent-500">
              重置 {activeCount} 项
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
