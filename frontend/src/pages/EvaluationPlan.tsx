import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

type Direction = 'desc' | 'asc';

type Product = {
  id: string;
  name: string;
  category: string;
  type: string;
  company: string;
  establishedYear: number;
  size: number;
  return1Y: number;
  sharpe: number;
  maxDrawdown: number;
  expenseRatio: number;
};

type IndicatorKey = 'return1Y' | 'sharpe' | 'maxDrawdown' | 'expenseRatio';

type IndicatorConfig = {
  key: IndicatorKey;
  label: string;
  description: string;
  unit?: string;
  direction: Direction;
  defaultWeight: number;
  format?: (value: number) => string;
};

const PRODUCTS: Product[] = [
  {
    id: 'EQT-001',
    name: '成长先锋混合',
    category: '权益',
    type: '主动权益',
    company: '华夏基金',
    establishedYear: 2015,
    size: 18.2,
    return1Y: 14.6,
    sharpe: 1.21,
    maxDrawdown: -9.8,
    expenseRatio: 1.3,
  },
  {
    id: 'EQT-002',
    name: '行业精选增强',
    category: '权益',
    type: '指数增强',
    company: '易方达基金',
    establishedYear: 2018,
    size: 22.4,
    return1Y: 9.8,
    sharpe: 0.97,
    maxDrawdown: -12.4,
    expenseRatio: 1.1,
  },
  {
    id: 'FIX-001',
    name: '稳健回报债券',
    category: '固收',
    type: '纯债基金',
    company: '博时基金',
    establishedYear: 2012,
    size: 34.1,
    return1Y: 5.6,
    sharpe: 0.68,
    maxDrawdown: -3.7,
    expenseRatio: 0.6,
  },
  {
    id: 'ALT-001',
    name: '量化对冲策略',
    category: '另类',
    type: '量化对冲',
    company: '富国基金',
    establishedYear: 2019,
    size: 12.5,
    return1Y: 7.3,
    sharpe: 1.05,
    maxDrawdown: -6.1,
    expenseRatio: 1.8,
  },
];

const INDICATORS: IndicatorConfig[] = [
  {
    key: 'return1Y',
    label: '近一年收益率',
    description: '衡量过去一年产品的收益表现，越高越好。',
    unit: '%',
    direction: 'desc',
    defaultWeight: 40,
    format: (value) => `${value.toFixed(1)}%`,
  },
  {
    key: 'sharpe',
    label: '夏普比率',
    description: '反映单位风险所获得的超额收益，越高越好。',
    direction: 'desc',
    defaultWeight: 30,
    format: (value) => value.toFixed(2),
  },
  {
    key: 'maxDrawdown',
    label: '最大回撤',
    description: '衡量历史最大损失幅度，越低越好。',
    unit: '%',
    direction: 'asc',
    defaultWeight: 20,
    format: (value) => `${value.toFixed(1)}%`,
  },
  {
    key: 'expenseRatio',
    label: '管理费率',
    description: '综合考察产品的成本水平，越低越好。',
    unit: '%',
    direction: 'asc',
    defaultWeight: 10,
    format: (value) => `${value.toFixed(1)}%`,
  },
];

type TimeRange = '3M' | '6M' | '1Y' | '3Y' | '5Y';

const TIME_RANGE_OPTIONS: { value: TimeRange; label: string }[] = [
  { value: '3M', label: '近3个月' },
  { value: '6M', label: '近6个月' },
  { value: '1Y', label: '近1年' },
  { value: '3Y', label: '近3年' },
  { value: '5Y', label: '近5年' },
];

type IndicatorEntry = {
  id: string;
  key: IndicatorKey;
  weight: number;
  direction: Direction;
  timeRange: TimeRange;
};

type MultiSelectProps = {
  label: string;
  options: string[];
  selectedValues: string[];
  onChange: (values: string[]) => void;
  placeholder?: string;
  helperText?: string;
};

function useOnClickOutside<T extends HTMLElement>(ref: React.RefObject<T>, handler: () => void) {
  useEffect(() => {
    function handleClick(event: MouseEvent) {
      if (!ref.current || ref.current.contains(event.target as Node)) {
        return;
      }
      handler();
    }

    document.addEventListener('mousedown', handleClick);

    return () => {
      document.removeEventListener('mousedown', handleClick);
    };
  }, [handler, ref]);
}

function MultiSelect({
  label,
  options,
  selectedValues,
  onChange,
  placeholder = '全部',
  helperText,
}: MultiSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const closeDropdown = useCallback(() => setIsOpen(false), []);

  useOnClickOutside(containerRef, closeDropdown);

  const toggleValue = (value: string) => {
    onChange(
      selectedValues.includes(value)
        ? selectedValues.filter((item) => item !== value)
        : [...selectedValues, value],
    );
  };

  const selectionSummary = useMemo(() => {
    if (selectedValues.length === 0) {
      return placeholder;
    }
    if (selectedValues.length <= 2) {
      return selectedValues.join('、');
    }
    return `${selectedValues.slice(0, 2).join('、')} 等${selectedValues.length}项`;
  }, [placeholder, selectedValues]);

  return (
    <div className="relative space-y-2" ref={containerRef}>
      <label className="text-sm font-medium text-gray-700">{label}</label>
      <button
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        className="flex w-full items-center justify-between rounded-md border border-gray-200 px-3 py-2 text-sm text-gray-700 transition focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-100"
      >
        <span className={selectedValues.length === 0 ? 'text-gray-400' : ''}>{selectionSummary}</span>
        <svg
          className={`h-4 w-4 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          viewBox="0 0 20 20"
          fill="currentColor"
        >
          <path
            fillRule="evenodd"
            d="M5.23 7.21a.75.75 0 011.06.02L10 11.17l3.71-3.94a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z"
            clipRule="evenodd"
          />
        </svg>
      </button>
      {helperText && <p className="text-xs text-gray-400">{helperText}</p>}
      {isOpen && (
        <div className="absolute z-10 mt-1 w-full rounded-md border border-gray-200 bg-white shadow-lg">
          <div className="flex items-center justify-between border-b border-gray-100 px-3 py-2 text-xs text-gray-400">
            <button
              type="button"
              onClick={() => onChange([])}
              className="text-blue-600 hover:text-blue-700"
            >
              清除全部
            </button>
            <span>
              已选 {selectedValues.length} / {options.length}
            </span>
          </div>
          <div className="max-h-56 overflow-auto py-1 text-sm">
            {options.map((option) => {
              const checked = selectedValues.includes(option);
              return (
                <label
                  key={option}
                  className="flex cursor-pointer items-center justify-between px-3 py-2 hover:bg-gray-50"
                >
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggleValue(option)}
                      className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-gray-700">{option}</span>
                  </div>
                  {checked && <span className="text-xs text-blue-500">已选</span>}
                </label>
              );
            })}
            {options.length === 0 && (
              <div className="px-3 py-4 text-center text-xs text-gray-400">暂无可选项</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function normalizeScores(
  products: Product[],
  indicatorKey: IndicatorKey,
  direction: Direction,
): Record<string, number> {
  if (products.length === 0) {
    return {};
  }

  const values = products.map((product) => product[indicatorKey]);
  const dataMax = Math.max(...values);
  const dataMin = Math.min(...values);

  if (dataMax === dataMin) {
    return Object.fromEntries(products.map((product) => [product.id, 100]));
  }

  return Object.fromEntries(
    products.map((product) => {
      const raw = product[indicatorKey];
      const normalized =
        direction === 'desc'
          ? ((raw - dataMin) / (dataMax - dataMin)) * 100
          : ((dataMax - raw) / (dataMax - dataMin)) * 100;
      return [product.id, Number(normalized.toFixed(2))];
    }),
  );
}

export default function EvaluationPlan() {
  const categories = useMemo(() => Array.from(new Set(PRODUCTS.map((product) => product.category))), []);
  const productTypes = useMemo(() => Array.from(new Set(PRODUCTS.map((product) => product.type))), []);
  const fundCompanies = useMemo(() => Array.from(new Set(PRODUCTS.map((product) => product.company))), []);
  const yearBounds = useMemo(() => {
    const years = PRODUCTS.map((product) => product.establishedYear);
    return {
      min: Math.min(...years),
      max: Math.max(...years),
    };
  }, []);

  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [selectedTypes, setSelectedTypes] = useState<string[]>([]);
  const [selectedCompanies, setSelectedCompanies] = useState<string[]>([]);
  const [yearRange, setYearRange] = useState<[number, number]>(() => [yearBounds.min, yearBounds.max]);
  const [searchKeyword, setSearchKeyword] = useState('');
  const indicatorConfigMap = useMemo(() => {
    return INDICATORS.reduce((acc, indicator) => {
      acc[indicator.key] = indicator;
      return acc;
    }, {} as Record<IndicatorKey, IndicatorConfig>);
  }, []);

  const [indicatorEntries, setIndicatorEntries] = useState<IndicatorEntry[]>(() =>
    INDICATORS.map((indicator, index) => ({
      id: `indicator-${index}`,
      key: indicator.key,
      weight: indicator.defaultWeight,
      direction: indicator.direction,
      timeRange: '1Y',
    })),
  );
  const indicatorIdRef = useRef(indicatorEntries.length);

  const filteredProducts = useMemo(() => {
    return PRODUCTS.filter((product) => {
      const categoryMatch = selectedCategories.length === 0 || selectedCategories.includes(product.category);
      const typeMatch = selectedTypes.length === 0 || selectedTypes.includes(product.type);
      const companyMatch = selectedCompanies.length === 0 || selectedCompanies.includes(product.company);
      const yearMatch = product.establishedYear >= yearRange[0] && product.establishedYear <= yearRange[1];
      const keyword = searchKeyword.trim();
      const keywordMatch =
        keyword.length === 0 || product.name.includes(keyword) || product.id.includes(keyword) || product.company.includes(keyword);
      return categoryMatch && typeMatch && companyMatch && yearMatch && keywordMatch;
    });
  }, [selectedCategories, selectedTypes, selectedCompanies, yearRange, searchKeyword]);

  const totalWeight = indicatorEntries.reduce((sum, entry) => sum + entry.weight, 0);

  const normalizedScores = useMemo(() => {
    const result: Record<string, Record<IndicatorKey, number>> = {};

    indicatorEntries.forEach((entry) => {
      const scores = normalizeScores(filteredProducts, entry.key, entry.direction);
      Object.entries(scores).forEach(([productId, score]) => {
        if (!result[productId]) {
          result[productId] = {} as Record<IndicatorKey, number>;
        }
        result[productId][entry.key] = score;
      });
    });

    return result;
  }, [filteredProducts, indicatorEntries]);

  const aggregatedScores = useMemo(() => {
    return filteredProducts.map((product) => {
      const total = indicatorEntries.reduce((sum, entry) => {
        const indicatorWeightShare = totalWeight > 0 ? entry.weight / totalWeight : 0;
        const score = normalizedScores[product.id]?.[entry.key] ?? 0;
        return sum + score * indicatorWeightShare;
      }, 0);

      return {
        product,
        totalScore: Number(total.toFixed(2)),
      };
    });
  }, [filteredProducts, indicatorEntries, normalizedScores, totalWeight]);

  const sortedScores = useMemo(() => {
    return [...aggregatedScores].sort((a, b) => b.totalScore - a.totalScore);
  }, [aggregatedScores]);

  const usedIndicatorKeys = useMemo(() => indicatorEntries.map((entry) => entry.key), [indicatorEntries]);

  const availableIndicators = useMemo(
    () => INDICATORS.filter((indicator) => !usedIndicatorKeys.includes(indicator.key)),
    [usedIndicatorKeys],
  );

  const handleIndicatorKeyChange = (id: string, key: IndicatorKey) => {
    setIndicatorEntries((prev) =>
      prev.map((entry) => {
        if (entry.id !== id) {
          return entry;
        }
        const config = indicatorConfigMap[key];
        return {
          ...entry,
          key,
          direction: config.direction,
          weight: config.defaultWeight,
          timeRange: '1Y',
        };
      }),
    );
  };

  const handleIndicatorDirectionChange = (id: string, direction: Direction) => {
    setIndicatorEntries((prev) =>
      prev.map((entry) => (entry.id === id ? { ...entry, direction } : entry)),
    );
  };

  const handleIndicatorWeightChange = (id: string, weight: number) => {
    setIndicatorEntries((prev) =>
      prev.map((entry) => (entry.id === id ? { ...entry, weight: Number.isNaN(weight) ? entry.weight : weight } : entry)),
    );
  };

  const handleAddIndicator = () => {
    const nextIndicator = availableIndicators[0];
    if (!nextIndicator) {
      return;
    }
    setIndicatorEntries((prev) => [
      ...prev,
      {
        id: `indicator-${indicatorIdRef.current}`,
        key: nextIndicator.key,
        weight: nextIndicator.defaultWeight,
        direction: nextIndicator.direction,
        timeRange: '1Y',
      },
    ]);
    indicatorIdRef.current += 1;
  };

  const handleRemoveIndicator = (id: string) => {
    setIndicatorEntries((prev) => prev.filter((entry) => entry.id !== id));
  };

  const handleIndicatorTimeRangeChange = (id: string, value: TimeRange) => {
    setIndicatorEntries((prev) =>
      prev.map((entry) => (entry.id === id ? { ...entry, timeRange: value } : entry)),
    );
  };

  const handleYearRangeChange = (index: 0 | 1, rawValue: number) => {
    setYearRange((prev) => {
      const next: [number, number] = [...prev];
      const safeValue = Number.isNaN(rawValue) ? prev[index] : rawValue;
      const clamped = Math.round(Math.min(Math.max(safeValue, yearBounds.min), yearBounds.max));
      next[index] = clamped;
      if (next[0] > next[1]) {
        return [clamped, clamped];
      }
      return next;
    });
  };

  return (
    <div className="mx-auto max-w-6xl space-y-8 p-6">
      <header className="space-y-3">
        <div className="flex items-center gap-3">
          <h1 className="text-3xl font-semibold text-gray-900">评价方案</h1>
          <span className="rounded-full bg-blue-50 px-3 py-1 text-sm font-medium text-blue-600">产品优选</span>
        </div>
        <p className="text-gray-600">
          针对指定范围内的产品，支持配置评价指标、排名规则与权重，并实时生成各指标得分与总分，帮助快速发现表现优异的产品。
        </p>
      </header>

      <section className="rounded-lg bg-white p-6 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">1. 设定评价范围</h2>
            <p className="mt-1 text-sm text-gray-500">
              支持按产品类别、类型、基金公司、成立年份等多条件筛选候选池；下方的评分与排名将基于筛选后的产品集。
            </p>
          </div>
          <div className="rounded-md border border-blue-100 bg-blue-50 px-3 py-2 text-xs text-blue-700">
            当前候选产品：<span className="font-semibold">{filteredProducts.length}</span> 只
          </div>
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <MultiSelect
            label="基金类别"
            options={categories}
            selectedValues={selectedCategories}
            onChange={setSelectedCategories}
            helperText="可多选，未选择时默认包含全部类别。"
          />

          <MultiSelect
            label="基金类型"
            options={productTypes}
            selectedValues={selectedTypes}
            onChange={setSelectedTypes}
            helperText="可多选，未选择时默认包含所有类型。"
          />

          <MultiSelect
            label="基金公司"
            options={fundCompanies}
            selectedValues={selectedCompanies}
            onChange={setSelectedCompanies}
            helperText="支持多选，可快速定位目标管理人。"
          />

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">成立年份</label>
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <input
                type="number"
                min={yearBounds.min}
                max={yearBounds.max}
                value={yearRange[0]}
                onChange={(event) => handleYearRangeChange(0, Number(event.target.value))}
                className="w-24 rounded border border-gray-200 px-2 py-1 text-right focus:border-blue-500 focus:outline-none"
              />
              <span>至</span>
              <input
                type="number"
                min={yearBounds.min}
                max={yearBounds.max}
                value={yearRange[1]}
                onChange={(event) => handleYearRangeChange(1, Number(event.target.value))}
                className="w-24 rounded border border-gray-200 px-2 py-1 text-right focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div className="flex items-center justify-between text-xs text-gray-400">
              <span>
                数据范围 {yearBounds.min} - {yearBounds.max}
              </span>
              <button
                type="button"
                onClick={() => setYearRange([yearBounds.min, yearBounds.max])}
                className="text-blue-600 hover:underline"
              >
                重置
              </button>
            </div>
          </div>

          <div className="space-y-2 md:col-span-2 lg:col-span-4">
            <label className="text-sm font-medium text-gray-700">名称/代码检索</label>
            <input
              type="text"
              value={searchKeyword}
              onChange={(event) => setSearchKeyword(event.target.value)}
              placeholder="输入产品名称或代码"
              className="w-full rounded-md border border-gray-200 px-3 py-2 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
            />
          </div>
        </div>

        <div className="mt-6 overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="whitespace-nowrap px-3 py-2 text-left font-medium text-gray-500">产品</th>
                <th className="whitespace-nowrap px-3 py-2 text-left font-medium text-gray-500">类别</th>
                <th className="whitespace-nowrap px-3 py-2 text-left font-medium text-gray-500">类型</th>
                <th className="whitespace-nowrap px-3 py-2 text-left font-medium text-gray-500">基金公司</th>
                <th className="whitespace-nowrap px-3 py-2 text-left font-medium text-gray-500">成立年份</th>
                <th className="whitespace-nowrap px-3 py-2 text-left font-medium text-gray-500">规模 (亿元)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 bg-white">
              {filteredProducts.map((product) => (
                <tr key={product.id}>
                  <td className="whitespace-nowrap px-3 py-2">
                    <div className="font-medium text-gray-900">{product.name}</div>
                    <div className="text-xs text-gray-500">代码：{product.id}</div>
                  </td>
                  <td className="whitespace-nowrap px-3 py-2 text-gray-700">{product.category}</td>
                  <td className="whitespace-nowrap px-3 py-2 text-gray-700">{product.type}</td>
                  <td className="whitespace-nowrap px-3 py-2 text-gray-700">{product.company}</td>
                  <td className="whitespace-nowrap px-3 py-2 text-gray-700">{product.establishedYear}</td>
                  <td className="whitespace-nowrap px-3 py-2 text-gray-700">{product.size.toFixed(1)}</td>
                </tr>
              ))}
              {filteredProducts.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-3 py-6 text-center text-gray-500">
                    暂无符合条件的产品，请调整筛选条件。
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rounded-lg bg-white p-6 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">2. 配置评价指标</h2>
            <p className="mt-1 text-sm text-gray-500">
              可自由启用、调整排名规则与权重。得分按 0-100 归一化，最终总分为各指标得分乘以权重占比之和。
            </p>
          </div>
          <div className="text-sm text-gray-500">
            权重合计：<span className="font-semibold text-gray-900">{totalWeight}%</span>
          </div>
        </div>

        <div className="mt-6 overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="whitespace-nowrap px-3 py-2 text-left font-medium text-gray-500">指标</th>
                <th className="whitespace-nowrap px-3 py-2 text-left font-medium text-gray-500">排名规则</th>
                <th className="whitespace-nowrap px-3 py-2 text-left font-medium text-gray-500">权重</th>
                <th className="whitespace-nowrap px-3 py-2 text-left font-medium text-gray-500">时间区间</th>
                <th className="whitespace-nowrap px-3 py-2 text-left font-medium text-gray-500">操作</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 bg-white">
              {indicatorEntries.map((entry) => {
                const config = indicatorConfigMap[entry.key];
                const selectableIndicators = INDICATORS.filter(
                  (item) => item.key === entry.key || !usedIndicatorKeys.includes(item.key),
                );
                return (
                  <tr key={entry.id} className="align-top">
                    <td className="px-3 py-3">
                      <div className="space-y-2">
                        <select
                          value={entry.key}
                          onChange={(event) => handleIndicatorKeyChange(entry.id, event.target.value as IndicatorKey)}
                          className="w-full rounded border border-gray-200 px-2 py-1 text-sm focus:border-blue-500 focus:outline-none"
                        >
                          {selectableIndicators.map((option) => (
                            <option key={option.key} value={option.key}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                        <p className="text-xs text-gray-500">{config.description}</p>
                      </div>
                    </td>
                    <td className="whitespace-nowrap px-3 py-3">
                      <select
                        value={entry.direction}
                        onChange={(event) => handleIndicatorDirectionChange(entry.id, event.target.value as Direction)}
                        className="rounded border border-gray-200 px-2 py-1 text-sm focus:border-blue-500 focus:outline-none"
                      >
                        <option value="desc">数值高→得分高</option>
                        <option value="asc">数值低→得分高</option>
                      </select>
                    </td>
                    <td className="whitespace-nowrap px-3 py-3">
                      <div className="flex items-center gap-2">
                        <input
                          type="number"
                          min={0}
                          max={100}
                          step={1}
                          value={entry.weight}
                          onChange={(event) => handleIndicatorWeightChange(entry.id, Number(event.target.value))}
                          className="w-20 rounded border border-gray-200 px-2 py-1 text-right focus:border-blue-500 focus:outline-none"
                        />
                        <span className="text-gray-400">%</span>
                      </div>
                    </td>
                    <td className="whitespace-nowrap px-3 py-3">
                      <select
                        value={entry.timeRange}
                        onChange={(event) => handleIndicatorTimeRangeChange(entry.id, event.target.value as TimeRange)}
                        className="rounded border border-gray-200 px-2 py-1 text-sm focus:border-blue-500 focus:outline-none"
                      >
                        {TIME_RANGE_OPTIONS.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                      <p className="mt-2 text-xs text-gray-400">依据所选时间窗口汇总指标表现。</p>
                    </td>
                    <td className="whitespace-nowrap px-3 py-3">
                      <button
                        type="button"
                        onClick={() => handleRemoveIndicator(entry.id)}
                        className="text-sm text-red-500 hover:text-red-600"
                      >
                        删除
                      </button>
                    </td>
                  </tr>
                );
              })}
              {indicatorEntries.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-3 py-6 text-center text-gray-500">
                    尚未选择任何指标，请点击下方按钮添加。
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-dashed border-gray-200 pt-4">
          <div className="text-xs text-gray-500">
            权重可任意调整，系统将按相对占比进行加权汇总。可通过删除按钮移除不需要的指标。
          </div>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={handleAddIndicator}
              disabled={availableIndicators.length === 0}
              className={`rounded-md px-3 py-2 text-sm font-medium transition ${
                availableIndicators.length === 0
                  ? 'cursor-not-allowed bg-gray-100 text-gray-400'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
            >
              添加指标
            </button>
            {availableIndicators.length === 0 && (
              <span className="text-xs text-gray-400">已使用所有预设指标</span>
            )}
          </div>
        </div>
      </section>

      <section className="rounded-lg bg-white p-6 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">3. 结果预览与排名</h2>
            <p className="mt-1 text-sm text-gray-500">
              根据配置实时计算总分，并按从高到低排序。可用于导出至投研报告或进一步进行组合构建。
            </p>
          </div>
          <div className="rounded-md border border-emerald-100 bg-emerald-50 px-3 py-2 text-xs text-emerald-700">
            总分 = ∑ (指标得分 × 权重占比)
          </div>
        </div>

        <div className="mt-6 overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="whitespace-nowrap px-3 py-2 text-left font-medium text-gray-500">排名</th>
                <th className="whitespace-nowrap px-3 py-2 text-left font-medium text-gray-500">产品</th>
                {indicatorEntries.map((entry) => {
                  const config = indicatorConfigMap[entry.key];
                  const timeRangeLabel = TIME_RANGE_OPTIONS.find((option) => option.value === entry.timeRange)?.label;
                  return (
                    <th key={entry.id} className="whitespace-nowrap px-3 py-2 text-left font-medium text-gray-500">
                      <div className="flex flex-col">
                        <span>{config.label}</span>
                        {timeRangeLabel && <span className="text-xs font-normal text-gray-400">{timeRangeLabel}</span>}
                      </div>
                    </th>
                  );
                })}
                <th className="whitespace-nowrap px-3 py-2 text-left font-medium text-gray-500">总得分</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 bg-white">
              {sortedScores.map(({ product, totalScore }, index) => (
                <tr key={product.id}>
                  <td className="whitespace-nowrap px-3 py-2 text-gray-700">{index + 1}</td>
                  <td className="whitespace-nowrap px-3 py-2">
                    <div className="font-medium text-gray-900">{product.name}</div>
                    <div className="text-xs text-gray-500">代码：{product.id}</div>
                  </td>
                  {indicatorEntries.map((entry) => {
                    const score = normalizedScores[product.id]?.[entry.key];
                    const weightShare = totalWeight > 0 ? (entry.weight / totalWeight) * 100 : 0;
                    const timeRangeLabel = TIME_RANGE_OPTIONS.find((option) => option.value === entry.timeRange)?.label;
                    return (
                      <td key={entry.id} className="whitespace-nowrap px-3 py-2">
                        <div className="text-gray-700">得分 {score?.toFixed(1) ?? '—'}</div>
                        <div className="text-xs text-gray-400">权重占比 {weightShare.toFixed(0)}%</div>
                        {timeRangeLabel && <div className="text-xs text-gray-300">{timeRangeLabel}</div>}
                      </td>
                    );
                  })}
                  <td className="whitespace-nowrap px-3 py-2 font-semibold text-blue-600">{totalScore.toFixed(1)}</td>
                </tr>
              ))}
              {sortedScores.length === 0 && (
                <tr>
                  <td colSpan={indicatorEntries.length + 3} className="px-3 py-6 text-center text-gray-500">
                    当前筛选下暂无产品，请先调整评价范围。
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
