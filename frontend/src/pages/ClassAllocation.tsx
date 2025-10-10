import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';

// Helper component for section titles
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mt-6 rounded-2xl border border-gray-200 bg-white p-6">
      <h2 className="text-lg font-semibold text-gray-800">{title}</h2>
      <div className="mt-4">{children}</div>
    </div>
  );
}

interface ConfigDetail {
  className: string;
  code: string;
  name: string;
  weight: string;
}

export default function ClassAllocation() {
  // State for UI interaction
  const [returnMetric, setReturnMetric] = useState('annual');
  const [riskMetric, setRiskMetric] = useState('var');
  const [startDate, setStartDate] = useState('2020-01-01');
  const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);

  // State for results
  const [frontierData, setFrontierData] = useState<any>(null);
  const [isCalculating, setIsCalculating] = useState(false);

  // State for loading data
  const [allocations, setAllocations] = useState<string[]>([]);
  const [selectedAlloc, setSelectedAlloc] = useState('');
  const [configDetails, setConfigDetails] = useState<ConfigDetail[] | null>(null);
  const [assetNames, setAssetNames] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // Form states for dynamic inputs
  const [annualDaysRet, setAnnualDaysRet] = useState(252);
  const [ewmAlpha, setEwmAlpha] = useState(0.94);
  const [ewmWindow, setEwmWindow] = useState(60);
  const [annualDaysRisk, setAnnualDaysRisk] = useState(252);
  const [ewmAlphaRisk, setEwmAlphaRisk] = useState(0.94);
  const [ewmWindowRisk, setEwmWindowRisk] = useState(60);
  const [confidence, setConfidence] = useState(95);
  const [returnType, setReturnType] = useState('simple');
  // 年化无风险收益率（%）
  const [riskFreePct, setRiskFreePct] = useState(1.5);

  // 约束：单一上下限 per asset
  const [singleLimits, setSingleLimits] = useState<Record<string, { lo: number; hi: number }>>({});
  // 约束：组上下限列表
  type GroupLimit = { id: string; assets: string[]; lo: number; hi: number };
  const [groupLimits, setGroupLimits] = useState<GroupLimit[]>([]);

  // 随机探索参数：多轮
  type RoundConf = { id: string; samples: number; step: number; buckets: number };
  const [rounds, setRounds] = useState<RoundConf[]>([
    { id: 'r0', samples: 1000, step: 0.5, buckets: 10 }, // 第0轮不使用分桶参数
    { id: 'r1', samples: 2000, step: 0.4, buckets: 10 },
    { id: 'r2', samples: 3000, step: 0.3, buckets: 20 },
    { id: 'r3', samples: 4000, step: 0.2, buckets: 30 },
    { id: 'r4', samples: 5000, step: 0.15, buckets: 40 },
    { id: 'r5', samples: 5000, step: 0.1, buckets: 50 },
  ]);
  // 权重量化
  const [quantStep, setQuantStep] = useState<'none' | '0.001' | '0.002' | '0.005'>('none');
  // SLSQP 精炼
  const [useRefine, setUseRefine] = useState(false);
  const [refineCount, setRefineCount] = useState(20);

  // ---- 策略制定与回测 ----
  type StrategyType = 'fixed' | 'risk_budget' | 'target';
  type StrategyRow = { id: string; type: StrategyType; name: string; rows: { className: string; weight?: number | null; budget?: number }[]; cfg: any; rebalance?: any };
  const [strategies, setStrategies] = useState<StrategyRow[]>([]);
  const [btStart, setBtStart] = useState<string>('');
  const [navCount, setNavCount] = useState<number>(0);
  const [btSeries, setBtSeries] = useState<any>(null);
  const [scheduleMarkers, setScheduleMarkers] = useState<Record<string, {date:string, weights:number[]}[]>>({});
  const [busyStrategy, setBusyStrategy] = useState<string | null>(null);
  const [showAddPicker, setShowAddPicker] = useState(false);
  const [btBusy, setBtBusy] = useState(false);

  function computeEqualPercents(names: string[]): number[] {
    const n = Math.max(1, names.length);
    const base = Math.floor((100 / n) * 100) / 100; // 向下取两位
    const arr = Array(n).fill(base);
    const others = base * (n - 1);
    const first = parseFloat((100 - others).toFixed(2));
    arr[0] = first;
    return arr;
  }

  function uniqueStrategyName(base: string, list: StrategyRow[]): string {
    const exists = new Set(list.map(s => s.name));
    if (!exists.has(base)) return base;
    let k = 1;
    while (exists.has(`${base}${k}`)) k += 1;
    return `${base}${k}`;
  }


  // Fetch list of saved allocations on mount
  useEffect(() => {
    const fetchAllocations = async () => {
      try {
        setLoading(true);
        const res = await fetch('/api/list-allocations');
        if (!res.ok) throw new Error('无法获取方案列表');
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) {
          setAllocations(data);
          setSelectedAlloc(data[0]); // Default to the first one
        } else {
          setError('沒有找到已保存的大類構建方案。請先在“手動構建大類”頁面保存配置後，再進行大類資產配置。');
        }
      } catch (e: any) {
        setError(e.message || '获取方案列表失败');
      } finally {
        setLoading(false);
      }
    };
    fetchAllocations();
  }, []);

  // Handler to load details for the selected allocation
  const handleSelectAndLoad = async () => {
    if (!selectedAlloc) {
      alert('请选择一个方案');
      return;
    }
    try {
      setLoading(true);
      const res = await fetch(`/api/load-allocation?name=${encodeURIComponent(selectedAlloc)}`);
      if (!res.ok) throw new Error('加载方案详情失败');
      const data = await res.json();
      
      // Flatten the data for table display
      const details: ConfigDetail[] = [];
      data.forEach((ac: any) => {
        ac.etfs.forEach((etf: any) => {
          details.push({
            className: ac.name,
            code: etf.code,
            name: etf.name,
            weight: `${etf.weight.toFixed(2)}%`,
          });
        });
      });
      setConfigDetails(details);

      // 推导大类列表并初始化单项约束
      const aset = Array.from(new Set(details.map(d => d.className)));
      setAssetNames(aset);
      const initLimits: Record<string, { lo: number; hi: number }> = {};
      aset.forEach(n => { initLimits[n] = { lo: 0, hi: 1 }; });
      setSingleLimits(initLimits);

      // 初始化策略区域的行（按等权/空）
      setStrategies([]);

      // 获取默认回测开始日期（每个大类第一条净值的最大值）
      try {
        const r = await fetch(`/api/strategy/default-start?alloc_name=${encodeURIComponent(selectedAlloc)}`);
        const j = await r.json();
        if (r.ok) {
          if (j.default_start) setBtStart(j.default_start);
          if (typeof j.count === 'number') setNavCount(j.count);
        }
      } catch {}

    } catch (e: any) {
      alert(e.message || '加载失败');
      setConfigDetails(null);
    } finally {
      setLoading(false);
    }
  };

  // 当切换方案时，预取默认开始日期
  useEffect(() => {
    const fetchDefault = async () => {
      if (!selectedAlloc) return;
      try {
        const r = await fetch(`/api/strategy/default-start?alloc_name=${encodeURIComponent(selectedAlloc)}`);
        const j = await r.json();
        if (r.ok) {
          if (j.default_start) setBtStart(j.default_start);
          if (typeof j.count === 'number') setNavCount(j.count);
        }
      } catch {}
    };
    fetchDefault();
  }, [selectedAlloc]);

  const onCalculate = async () => {
    if (!selectedAlloc) {
      alert("请先选择一个大类构建方案并加载其详情");
      return;
    }
    
    const payload = {
      alloc_name: selectedAlloc,
      start_date: startDate,
      end_date: endDate,
      return_metric: {
        metric: returnMetric,
        type: returnType,
        days: annualDaysRet,
        alpha: ewmAlpha,
        window: ewmWindow,
      },
      risk_metric: {
        metric: riskMetric,
        type: returnType, // Risk metric uses the same return type
        days: annualDaysRisk,
        alpha: ewmAlphaRisk,
        window: ewmWindowRisk,
        confidence: confidence,
      },
      risk_free_rate: Number.isFinite(riskFreePct) ? riskFreePct / 100 : 0.0,
      constraints: {
        single_limits: singleLimits,
        group_limits: groupLimits.map(g => ({ assets: g.assets, lo: g.lo, hi: g.hi }))
      },
      exploration: {
        rounds: rounds.map((r, idx) => idx === 0 ? ({ samples: r.samples, step: r.step }) : ({ samples: r.samples, step: r.step, buckets: r.buckets }))
      },
      quantization: { step: quantStep === 'none' ? 'none' : Number(quantStep) },
      refine: { use_slsqp: useRefine, count: refineCount },
    };

    try {
      setIsCalculating(true);
      setFrontierData(null);
      const res = await fetch('/api/efficient-frontier', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '计算失败');
      setFrontierData(data);
    } catch (e: any) {
      alert(e.message);
    } finally {
      setIsCalculating(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl p-6 relative">
      {(loading || isCalculating || btBusy) && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/40 rounded-2xl">
          <div className="rounded-xl bg-white px-6 py-4 shadow text-sm">{btBusy ? '计算中...' : (isCalculating ? '正在计算，请稍候...' : '正在加载...')}</div>
        </div>
      )}

      
      <h1 className="text-2xl font-semibold">大类资产配置</h1>
      <p className="text-sm text-gray-500 mt-1">通过配置风险和收益指标，计算并可视化给定大类构建方案的可配置空间与有效前沿。</p>

      <Section title="选择大类构建方案">
        {loading && <p>正在加载方案列表...</p>}
        {error && <p className="text-red-600">{error}</p>}
        {!loading && !error && (
          <div className="flex items-center gap-4">
            <select 
              value={selectedAlloc}
              onChange={e => setSelectedAlloc(e.target.value)}
              className="flex-grow rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500">
              {allocations.map(name => <option key={name} value={name}>{name}</option>)}
            </select>
            <button 
              onClick={handleSelectAndLoad}
              className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700">
              选择该方案
            </button>
          </div>
        )}
        {configDetails && (
          <div className="mt-4 rounded-lg border">
            <div className="max-h-96 overflow-y-auto">
              <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">大类名称</th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">ETF代码</th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">ETF名称</th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">资金权重</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 bg-white">
                {configDetails.map((item, index) => (
                  <tr key={index}>
                    <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-900">{item.className}</td>
                    <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500 font-mono">{item.code}</td>
                    <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-900">{item.name}</td>
                    <td className="whitespace-nowrap px-6 py-4 text-sm text-gray-500">{item.weight}</td>
                  </tr>
                ))}
              </tbody>
              </table>
            </div>
          </div>
        )}
      </Section>

      <Section title="刻画可配置空间与有效前沿的参数">
        <div className="grid grid-cols-1 gap-x-8 gap-y-6 md:grid-cols-2">
          {/* 收益指标 */}
          <div className="space-y-3 rounded-lg border p-4">
            <h3 className="font-medium text-gray-700">📈 收益指标</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-600">收益指标</label>
                <select onChange={e => setReturnMetric(e.target.value)} value={returnMetric} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm">
                  <option value="annual">年化收益率</option>
                  <option value="annual_mean">年化收益率均值</option>
                  <option value="cumulative">累计收益率</option>
                  <option value="mean">收益率均值</option>
                  <option value="ewm">指数加权收益率</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-600">收益类型</label>
                <select value={returnType} onChange={e => setReturnType(e.target.value)} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm">
                  <option value="simple">普通收益率</option>
                  <option value="log">对数收益率</option>
                </select>
              </div>
            </div>
            {(returnMetric === 'annual' || returnMetric === 'annual_mean') && (
              <div>
                <label className="block text-sm font-medium text-gray-600">年化天数</label>
                <input type="number" value={annualDaysRet} onChange={e => setAnnualDaysRet(Number(e.target.value))} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm" />
              </div>
            )}
            {returnMetric === 'ewm' && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-600">衰减因子 λ</label>
                  <input type="number" step="0.01" value={ewmAlpha} onChange={e => setEwmAlpha(Number(e.target.value))} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-600">窗口长度</label>
                  <input type="number" value={ewmWindow} onChange={e => setEwmWindow(Number(e.target.value))} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm" />
                </div>
              </div>
            )}
          </div>

          {/* 风险指标 */}
          <div className="space-y-3 rounded-lg border p-4">
            <h3 className="font-medium text-gray-700">⚠️ 风险指标</h3>
            <div className="grid grid-cols-2 gap-4">
                <div>
                    <label className="block text-sm font-medium text-gray-600">风险指标</label>
                    <select onChange={e => setRiskMetric(e.target.value)} value={riskMetric} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm">
                        <option value="vol">波动率</option>
                        <option value="annual_vol">年化波动率</option>
                        <option value="ewm_vol">指数加权波动率</option>
                        <option value="var">VaR</option>
                        <option value="es">ES</option>
                        <option value="max_drawdown">最大回撤</option>
                        <option value="downside_vol">下行波动率</option>
                    </select>
                </div>
                {(riskMetric === 'var' || riskMetric === 'es') && (
                    <div>
                        <label className="block text-sm font-medium text-gray-600">置信度 %</label>
                        <input type="number" value={confidence} onChange={e => setConfidence(Number(e.target.value))} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm" />
                    </div>
                )}
            </div>
            {(riskMetric === 'annual_vol') && (
                <div>
                    <label className="block text-sm font-medium text-gray-600">年化天数</label>
                    <input type="number" value={annualDaysRisk} onChange={e => setAnnualDaysRisk(Number(e.target.value))} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm" />
                </div>
            )}
            {(riskMetric === 'ewm_vol') && (
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-600">衰减因子 λ</label>
                        <input type="number" step="0.01" value={ewmAlphaRisk} onChange={e => setEwmAlphaRisk(Number(e.target.value))} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm" />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-600">窗口长度</label>
                        <input type="number" value={ewmWindowRisk} onChange={e => setEwmWindowRisk(Number(e.target.value))} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm" />
                    </div>
                </div>
            )}
            <div>
              <label className="block text-sm font-medium text-gray-600">年化无风险收益率(%)</label>
              <input
                type="number"
                step="0.1"
                value={riskFreePct}
                onChange={e => setRiskFreePct(Number(e.target.value))}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
              />
              <p className="mt-1 text-xs text-gray-500">用于计算最大夏普率，默认 1.5%</p>
            </div>
          </div>
        </div>

        {/* 权重约束设置 */}
        <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
          <div className="rounded-lg border p-4">
            <h3 className="font-medium text-gray-700">🔒 单个大类上下限</h3>
            {assetNames.length === 0 ? (
              <p className="text-sm text-gray-500 mt-2">请先选择并加载方案</p>
            ) : (
              <div className="mt-3 space-y-2">
                {assetNames.map(name => (
                  <div key={name} className="grid grid-cols-3 items-center gap-2">
                    <div className="text-sm text-gray-700">{name}</div>
                    <input type="number" min={0} max={1} step={0.01}
                      value={singleLimits[name]?.lo ?? 0}
                      onChange={e => setSingleLimits(prev => ({ ...prev, [name]: { ...(prev[name]||{lo:0,hi:1}), lo: Number(e.target.value) } }))}
                      className="rounded-md border-gray-300 px-2 py-1 text-sm" placeholder="下限(0-1)" />
                    <input type="number" min={0} max={1} step={0.01}
                      value={singleLimits[name]?.hi ?? 1}
                      onChange={e => setSingleLimits(prev => ({ ...prev, [name]: { ...(prev[name]||{lo:0,hi:1}), hi: Number(e.target.value) } }))}
                      className="rounded-md border-gray-300 px-2 py-1 text-sm" placeholder="上限(0-1)" />
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="rounded-lg border p-4">
            <h3 className="font-medium text-gray-700">🧩 多大类联合上下限</h3>
            <div className="mt-2 space-y-3">
              {groupLimits.map((g, idx) => (
                <div key={g.id} className="rounded border p-2">
                  <div className="flex flex-wrap gap-2">
                    {assetNames.map(n => (
                      <label key={n} className="flex items-center gap-1 text-xs">
                        <input type="checkbox" checked={g.assets.includes(n)} onChange={e => {
                          setGroupLimits(prev => prev.map(x => x.id===g.id ? { ...x, assets: e.target.checked ? [...x.assets, n] : x.assets.filter(a => a!==n) } : x))
                        }} />{n}
                      </label>
                    ))}
                  </div>
                  <div className="mt-2 grid grid-cols-3 gap-2">
                    <input type="number" min={0} max={1} step={0.01} value={g.lo} onChange={e => setGroupLimits(prev => prev.map(x => x.id===g.id ? { ...x, lo: Number(e.target.value) } : x))} className="rounded-md border-gray-300 px-2 py-1 text-sm" placeholder="下限(0-1)" />
                    <input type="number" min={0} max={1} step={0.01} value={g.hi} onChange={e => setGroupLimits(prev => prev.map(x => x.id===g.id ? { ...x, hi: Number(e.target.value) } : x))} className="rounded-md border-gray-300 px-2 py-1 text-sm" placeholder="上限(0-1)" />
                    <button onClick={() => setGroupLimits(prev => prev.filter(x => x.id !== g.id))} className="rounded bg-red-50 text-red-700 text-xs px-2">删除</button>
                  </div>
                </div>
              ))}
              <button onClick={() => setGroupLimits(prev => [...prev, { id: `g${Date.now()}`, assets: [], lo: 0, hi: 1 }])} className="rounded bg-gray-100 px-3 py-1 text-xs">+ 添加联合约束</button>
            </div>
          </div>
        </div>

        {/* 随机探索设置 */}
        <div className="mt-6 rounded-lg border p-4">
          <h3 className="font-medium text-gray-700">🎲 随机探索设置</h3>
          <p className="text-xs text-gray-500 mt-1">默认提供第0至第5轮，样本点从 1000~5000，步长从 0.5~0.1，分桶从 10~50（第0轮不分桶，可删除第1-5轮）。</p>
          <div className="mt-2 space-y-2">
            {rounds.map((r, idx) => (
              <div key={r.id} className="grid grid-cols-12 items-center gap-2">
                <div className="col-span-2 text-sm text-gray-600">第{idx}轮</div>
                <label className="col-span-3 text-xs text-gray-600 flex items-center gap-1">样本点
                  <input type="number" min={1} value={r.samples} onChange={e => setRounds(prev => prev.map(x => x.id===r.id ? { ...x, samples: Number(e.target.value) } : x))} className="ml-1 w-full rounded-md border-gray-300 px-2 py-1 text-xs" />
                </label>
                <label className="col-span-3 text-xs text-gray-600 flex items-center gap-1">步长
                  <input type="number" step={0.01} min={0} max={1} value={r.step} onChange={e => setRounds(prev => prev.map(x => x.id===r.id ? { ...x, step: Number(e.target.value) } : x))} className="ml-1 w-full rounded-md border-gray-300 px-2 py-1 text-xs" />
                </label>
                {idx > 0 && (
                  <label className="col-span-3 text-xs text-gray-600 flex items-center gap-1">分桶
                    <input type="number" min={1} value={(r as any).buckets ?? 50} onChange={e => setRounds(prev => prev.map(x => x.id===r.id ? { ...x, buckets: Number(e.target.value) } : x))} className="ml-1 w-full rounded-md border-gray-300 px-2 py-1 text-xs" />
                  </label>
                )}
                {idx > 0 && (
                  <button onClick={() => setRounds(prev => prev.filter(x => x.id !== r.id))} className="col-span-1 rounded bg-red-50 text-red-700 text-xs px-2">删</button>
                )}
              </div>
            ))}
            <button onClick={() => setRounds(prev => [...prev, { id: `r${Date.now()}`, samples: 200, step: 0.5, buckets: 50 }])} className="rounded bg-gray-100 px-3 py-1 text-xs">+ 增加一轮</button>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-600">权重量化</label>
              <select value={quantStep} onChange={e => setQuantStep(e.target.value as any)} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm">
                <option value="none">不量化</option>
                <option value="0.001">0.1%</option>
                <option value="0.002">0.2%</option>
                <option value="0.005">0.5%</option>
              </select>
            </div>
            <div className="flex items-end gap-2">
              <label className="flex items-center gap-2 text-sm text-gray-700">
                <input type="checkbox" checked={useRefine} onChange={e => setUseRefine(e.target.checked)} /> 使用 SLSQP 最终精炼
              </label>
              {useRefine && (
                <input type="number" min={1} value={refineCount} onChange={e => setRefineCount(Number(e.target.value))} className="w-28 rounded-md border-gray-300 px-2 py-1 text-sm" placeholder="精炼数量" />
              )}
            </div>
          </div>
        </div>
      </Section>

      <Section title="选择模型的构建区间">
        <div className="flex items-center gap-4">
            <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} className="w-full rounded-md border-gray-300 shadow-sm" />
            <span className="text-gray-500">至</span>
            <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} className="w-full rounded-md border-gray-300 shadow-sm" />
        </div>
      </Section>

      <div className="mt-8 flex justify-center">
        <button 
          onClick={onCalculate}
          disabled={isCalculating}
          className="rounded-lg bg-indigo-600 px-6 py-3 text-base font-semibold text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50">
          {isCalculating ? '计算中...' : '计算可配置空间与有效前沿'}
        </button>
      </div>

      {frontierData && (
        <Section title="结果展示">
          <ReactECharts
            style={{ height: 500 }}
            option={{
              title: {
                text: '可配置空间与有效前沿',
                left: 'center'
              },
              tooltip: {
                trigger: 'item',
                confine: true,
                formatter: (params: any) => {
                  const val = Array.isArray(params.value) ? params.value : (params.data?.value ?? params.value);
                  const risk = val?.[0];
                  const ret = val?.[1];
                  const names: string[] = frontierData.asset_names || [];
                  const ws: number[] | undefined = params.data?.weights;
                  const header = params.seriesName ? `${params.seriesName}<br/>` : '';
                  const rr = (Number.isFinite(risk) ? Number(risk).toFixed(4) : '-') + ' (风险)';
                  const re = (Number.isFinite(ret) ? Number(ret).toFixed(4) : '-') + ' (收益)';
                  if (ws && names && names.length === ws.length) {
                    const lines = names.map((n, i) => `${n}: ${(ws[i] * 100).toFixed(2)}%`);
                    return `${header}${lines.join('<br/>')}<br/>${rr}<br/>${re}`;
                  }
                  return `${header}${rr}<br/>${re}`;
                }
              },
              dataZoom: [
                { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
                { type: 'inside', yAxisIndex: 0, filterMode: 'none' },
                { type: 'slider', xAxisIndex: 0, filterMode: 'none' },
                { type: 'slider', yAxisIndex: 0, filterMode: 'none' },
              ],
              legend: {
                top: 36,
                left: 'center',
                data: ['其他组合', '有效前沿', '最大夏普率', '最小方差', '最大收益']
              },
              grid: { top: 80 },
              xAxis: { type: 'value', name: `风险（${riskMetric}）`, scale: true },
              yAxis: { type: 'value', name: `收益（${returnMetric}/${returnType === 'log' ? 'log' : 'simple'}）`, scale: true },
              series: [
                ...(frontierData.scatter ? [{
                  name: '其他组合',
                  type: 'scatter',
                  symbolSize: 3,
                  data: frontierData.scatter,
                  itemStyle: { color: 'rgba(128, 128, 128, 0.35)' }
                }] : []),
                ...(frontierData.frontier ? [{
                  name: '有效前沿',
                  type: 'scatter',
                  symbolSize: 6,
                  data: frontierData.frontier,
                  itemStyle: { color: '#2563eb' } // Tailwind indigo-600
                }] : []),
                ...(frontierData.max_sharpe ? [{
                  name: '最大夏普率',
                  type: 'scatter',
                  symbolSize: 10,
                  data: [frontierData.max_sharpe]
                }] : []),
                ...(frontierData.min_variance ? [{
                  name: '最小方差',
                  type: 'scatter',
                  symbolSize: 10,
                  data: [frontierData.min_variance]
                }] : []),
                ...(frontierData.max_return ? [{
                  name: '最大收益',
                  type: 'scatter',
                  symbolSize: 10,
                  data: [frontierData.max_return]
                }] : []),
              ]
            }}
          />
        </Section>
      )}

      {/* 大类资产策略制定与回测 */}
      <Section title="大类资产策略制定与回测">
        <div className="space-y-4">
          {/* 顶部不再显示“添加策略”按钮，统一放在策略列表与回测之间 */}

          {strategies.map((s, idx) => (
            <div key={s.id} className="rounded-lg border p-4">
              <div className="flex flex-wrap items-center gap-3">
                <input value={s.name} onChange={e => setStrategies(prev => prev.map(x => x.id===s.id? { ...x, name: e.target.value } : x))} className="rounded-md border-gray-300 px-2 py-1 text-sm" />
                <span className="rounded bg-gray-100 px-2 py-1 text-xs text-gray-700">
                  {s.type === 'fixed' ? '固定比例' : s.type === 'risk_budget' ? '风险预算' : '指定目标'}
                </span>
                <button onClick={() => setStrategies(prev => prev.filter(x => x.id !== s.id))} className="ml-auto rounded bg-red-50 px-2 py-1 text-xs text-red-700">删除</button>
              </div>
              {/* 再平衡设置（通用） */}
              <div className="mt-3 rounded border p-3 text-sm">
                <label className="flex items-center gap-2"><input type="checkbox" checked={!!s.rebalance?.enabled} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), enabled: e.target.checked } } : x))}/> 是否启用再平衡</label>
                {s.rebalance?.enabled && (
                  <div className="mt-2 grid grid-cols-12 items-center gap-2">
                    <div className="col-span-3">
                      <label className="block text-xs text-gray-600">再平衡方式</label>
                      <select value={s.rebalance?.mode||'monthly'} onChange={e=> setStrategies(prev=> prev.map(x=> {
                        if (x.id!==s.id) return x as any;
                        const mode = e.target.value;
                        const rb = { ...(x.rebalance||{}), mode } as any;
                        if (mode === 'fixed' && !rb.fixedInterval) {
                          rb.fixedInterval = Math.max(1, navCount||1);
                        }
                        return { ...x, rebalance: rb } as any;
                      }))} className="mt-1 w-full rounded border-gray-300">
                        <option value="weekly">每周</option>
                        <option value="monthly">每月</option>
                        <option value="yearly">每年</option>
                        <option value="fixed">固定区间</option>
                      </select>
                    </div>
                    {s.rebalance?.mode !== 'fixed' ? (
                      <>
                    <div className="col-span-2">
                      <label className="block text-xs text-gray-600">第N</label>
                      <input type="number" min={1} max={ s.rebalance?.mode==='weekly'?5: s.rebalance?.mode==='monthly'?30:360 } value={s.rebalance?.N ?? 1} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), which:'nth', N: Number(e.target.value) } } : x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                    </div>
                        <div className="col-span-2">
                        <label className="block text-xs text-gray-600">个</label>
                        <div className="mt-1 text-sm text-gray-500">&nbsp;</div>
                        </div>
                        <div className="col-span-2">
                          <label className="block text-xs text-gray-600">单位</label>
                          <select value={s.rebalance?.unit||'trading'} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), unit: e.target.value } } : x))} className="mt-1 w-full rounded border-gray-300">
                            <option value="trading">交易日</option>
                            <option value="natural">自然日</option>
                          </select>
                        </div>
                      </>
                    ) : (
                      <div className="col-span-3">
                        <label className="block text-xs text-gray-600">固定区间(天)</label>
                        <input type="number" min={1} value={s.rebalance?.fixedInterval ?? 20} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), fixedInterval: Number(e.target.value) } } : x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                      </div>
                    )}
                    {s.type !== 'fixed' && (
                      <div className="col-span-12">
                        <label className="flex items-center gap-2"><input type="checkbox" checked={!!s.rebalance?.recalc} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, rebalance: { ...(x.rebalance||{}), recalc: e.target.checked } } : x))}/> 再平衡时是否重新模型计算</label>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* 固定比例 */}
              {s.type === 'fixed' && (
                <div className="mt-3 space-y-3">
                  <div className="flex items-center gap-3 text-sm">
                    <label className="flex items-center gap-2"><input type="radio" checked={(s.cfg?.mode||'equal')==='equal'} onChange={() => setStrategies(prev => prev.map(x=>{
                      if (x.id!==s.id) return x;
                      const eqArr = computeEqualPercents(assetNames);
                      return { ...x, cfg:{...x.cfg, mode:'equal'}, rows: assetNames.map((n,i)=> ({ className:n, weight:eqArr[i] })) } as StrategyRow;
                    }))}/> 等权重</label>
                    <label className="flex items-center gap-2"><input type="radio" checked={(s.cfg?.mode||'equal')==='custom'} onChange={() => setStrategies(prev => prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, mode:'custom'}, rows: x.rows.map((rr,i)=> ({...rr, weight: i===0?100:0})) }:x))}/> 自定义权重</label>
                  </div>
                  <div className="rounded border">
                    <table className="min-w-full">
                      <thead className="bg-gray-50 text-xs text-gray-600"><tr><th className="px-3 py-2 text-left">大类名称</th><th className="px-3 py-2 text-left">资金权重(%)</th></tr></thead>
                      <tbody className="text-sm">
                        {s.rows.map((r,i)=> (
                          <tr key={i} className="border-t">
                            <td className="px-3 py-2">{r.className}</td>
                            <td className="px-3 py-2"><input type="number" value={r.weight ?? ''} onChange={e=>{
                              const v = e.target.value === '' ? undefined : Number(e.target.value);
                              setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, rows:x.rows.map((rr,j)=> j===i?{...rr, weight:v}:rr)}:x))
                            }} className={`w-28 rounded border px-2 py-1 ${ (s.cfg?.mode||'equal')==='equal' ? 'bg-gray-50 text-gray-500 border-gray-200' : 'border-gray-300' }`} disabled={(s.cfg?.mode||'equal')==='equal'}/></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* 风险预算 */}
              {s.type === 'risk_budget' && (
                <div className="mt-3 space-y-3">
                  <div className="rounded border">
                    <table className="min-w-full">
                      <thead className="bg-gray-50 text-xs text-gray-600"><tr><th className="px-3 py-2 text-left">大类名称</th><th className="px-3 py-2 text-left">风险预算(%)</th><th className="px-3 py-2 text-left">资金权重(%)</th></tr></thead>
                      <tbody className="text-sm">
                        {s.rows.map((r,i)=> (
                          <tr key={i} className="border-t">
                            <td className="px-3 py-2">{r.className}</td>
                            <td className="px-3 py-2"><input type="number" value={r.budget ?? 100} onChange={e=>{
                              const v = Number(e.target.value);
                              setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, rows:x.rows.map((rr,j)=>j===i?{...rr, budget:v}:rr)}:x))
                            }} className="w-28 rounded border-gray-300 px-2 py-1"/></td>
                            <td className="px-3 py-2">{r.weight==null? '-' : (r.weight?.toFixed(2))}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <label className="block text-xs text-gray-600">风险指标</label>
                      <select value={s.cfg?.risk_metric||'vol'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_metric:e.target.value}}:x))} className="mt-1 w-full rounded border-gray-300">
                        <option value="vol">波动率</option>
                        <option value="var">VaR</option>
                        <option value="es">ES</option>
                        <option value="downside_vol">下行波动率</option>
                        <option value="max_drawdown">最大回撤</option>
                      </select>
                    </div>
                    {['var','es'].includes(s.cfg?.risk_metric) && (
                      <><div>
                        <label className="block text-xs text-gray-600">置信度(%)</label>
                        <input type="number" value={s.cfg?.confidence ?? 95} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, confidence:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                      </div>
                      <div>
                        <label className="block text-xs text-gray-600">天数</label>
                        <input type="number" value={s.cfg?.days ?? 252} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, days:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                      </div></>
                    )}
                  </div>
                  {/* 模型计算区间 */}
                  <div className="rounded border p-3 text-sm">
                    <div className="grid grid-cols-3 gap-3 items-end">
                      <div>
                        <label className="block text-xs text-gray-600">窗口模式</label>
                        <select value={s.cfg?.window_mode || 'rollingN'} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, cfg: { ...(x.cfg||{}), window_mode: e.target.value } } : x))} className="mt-1 w-full rounded border-gray-300">
                          <option value="all">所有数据</option>
                          <option value="rollingN">最近N条</option>
                        </select>
                      </div>
                      { (s.cfg?.window_mode==='rollingN') && (
                        <div>
                          <label className="block text-xs text-gray-600">N（交易日）</label>
                          <input type="number" min={2} value={s.cfg?.data_len ?? 60} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, cfg: { ...(x.cfg||{}), data_len: Number(e.target.value) } } : x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                        </div>
                      )}
                    </div>
                    <p className="mt-2 text-xs text-gray-500">
                      模式说明：
                      <span className="ml-1 font-medium">所有数据</span> 使用回测开始至当期的全部样本；
                      <span className="ml-1 font-medium">最近N条</span> 使用当期之前最近 N 条的滚动窗口（推荐）。
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    <button disabled={busyStrategy===s.id} className="rounded bg-gray-100 px-3 py-1 text-sm disabled:opacity-50" onClick={async ()=>{
                      try{
                        setBusyStrategy(s.id);
                        if (s.rebalance?.enabled && s.rebalance?.recalc) {
                          // 调用批量时点权重
                          setBtBusy(true);
                          const payload = { alloc_name: selectedAlloc, start_date: btStart || undefined, strategy: { type:'risk_budget', name:s.name, classes: s.rows.map(r=>({ name:r.className, budget:r.budget??100 })), rebalance: s.rebalance, model: { risk_metric: s.cfg?.risk_metric, days: s.cfg?.days, confidence: s.cfg?.confidence, window_mode: s.cfg?.window_mode || 'rollingN', data_len: s.cfg?.data_len, } } };
                          const res = await fetch('/api/strategy/compute-schedule-weights',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
                          const dat = await res.json();
                          if(!res.ok) throw new Error(dat.detail||'权重计算失败');
                          const markers = (dat.dates||[]).map((d:string, idx:number)=> ({ date: d, weights: dat.weights[idx]||[] }));
                          setScheduleMarkers(prev => ({ ...prev, [s.name]: markers }));
                          // 同时回填当前权重为最新一列
                          const last = dat.weights && dat.weights.length ? dat.weights[dat.weights.length-1] : [];
                          setStrategies(prev=> prev.map(x=> x.id===s.id?{...x, rows: x.rows.map((r,i)=> ({...r, weight: last[i]!=null? Number((last[i]*100).toFixed(2)) : r.weight }))}:x));
                          setBtBusy(false);
                        } else {
                          const payload = { alloc_name: selectedAlloc, data_len: (s.cfg?.window_mode==='all') ? undefined : (s.cfg?.data_len ?? 60), window_mode: s.cfg?.window_mode || 'rollingN', strategy: { type:'risk_budget', name:s.name, classes: s.rows.map(r=>({ name:r.className, budget:r.budget??100 })), risk_metric: s.cfg?.risk_metric, confidence: s.cfg?.confidence, days: s.cfg?.days } };
                          const res = await fetch('/api/strategy/compute-weights',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
                          const dat = await res.json();
                          if(!res.ok) throw new Error(dat.detail||'计算失败');
                          const weights: number[] = dat.weights || [];
                          setStrategies(prev=> prev.map(x=> x.id===s.id?{...x, rows: x.rows.map((r,i)=> ({...r, weight: Number((weights[i]*100).toFixed(2)) }))}:x));
                        }
                      }catch(e:any){ alert(e.message||'反推失败'); }
                      finally { setBusyStrategy(null); }
                    }}>反推资金权重</button>
                  </div>
                  {busyStrategy===s.id && <div className="text-xs text-gray-500">计算中，请稍候…</div>}

                  {/* 再平衡横向权重表（来自回测后的 markers） */}
                  {s.rebalance?.enabled && s.rebalance?.recalc && (scheduleMarkers[s.name] || (btSeries && (btSeries.markers?.[s.name]))) && (
                    <div className="mt-3">
                      <div className="rounded border overflow-x-auto">
                        {(() => {
                          const markers = scheduleMarkers[s.name] || (btSeries.markers[s.name]||[]);
                          const dates: string[] = markers.map((m:any)=> m.date);
                          const names: string[] = (btSeries?.asset_names || assetNames);
                          const weightsByAsset: number[][] = names.map((_:any, i:number)=> markers.map((m:any)=> (m.weights?.[i] ?? 0)));
                          return (
                            <table className="min-w-full whitespace-nowrap text-sm">
                              <thead className="bg-gray-50">
                                <tr>
                                  <th className="px-3 py-2 text-left text-xs text-gray-600">大类名称</th>
                                  {dates.map((d)=> (
                                    <th key={d} className="px-3 py-2 text-left text-xs text-gray-600">{d}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {names.map((n, rIdx)=> (
                                  <tr key={n} className="border-t">
                                    <td className="px-3 py-2">{n}</td>
                                    {weightsByAsset[rIdx].map((w, cIdx)=> (
                                      <td key={cIdx} className="px-3 py-2">{(w*100).toFixed(2)}%</td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          );
                        })()}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* 指定目标 */}
              {s.type === 'target' && (
                <div className="mt-3 space-y-3">
                  {/* 目标类型 + 收益率类型 */}
                  <div className="grid grid-cols-2 gap-4 text-sm rounded border p-3">
                    <div>
                      <label className="block text-xs text-gray-600">目标类型</label>
                      <select value={s.cfg?.target||'min_risk'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, target:e.target.value}}:x))} className="mt-1 w-full rounded border-gray-300">
                        <option value="min_risk">最小风险</option>
                        <option value="max_return">最大收益</option>
                        <option value="max_sharpe">最大化收益风险性价比</option>
                        <option value="max_sharpe_traditional">最大化夏普比率</option>
                        <option value="risk_min_given_return">指定收益下最小风险</option>
                        <option value="return_max_given_risk">指定风险下最大收益</option>
                      </select>
                      
                      {/* Explanations for each target type */}
                      {(s.cfg?.target === 'min_risk' || !s.cfg?.target) && (
                        <p className="mt-2 text-xs text-gray-500 bg-gray-50 p-2 rounded-md">
                          目标：在满足所有约束条件下，寻找使组合风险（由指定的<strong>风险指标</strong>衡量）最小化的权重。
                        </p>
                      )}
                      {s.cfg?.target === 'max_return' && (
                        <p className="mt-2 text-xs text-gray-500 bg-gray-50 p-2 rounded-md">
                          目标：在满足所有约束条件下，寻找使组合收益（由指定的<strong>收益指标</strong>衡量）最大化的权重。
                        </p>
                      )}
                      {s.cfg?.target === 'max_sharpe' && (
                        <p className="mt-2 text-xs text-gray-500 bg-gray-50 p-2 rounded-md">
                          目标：寻找使 <strong>(指定收益指标) / (指定风险指标)</strong> 比值最大化的权重。这是一个广义的收益风险性价比优化。
                        </p>
                      )}
                      {s.cfg?.target === 'max_sharpe_traditional' && (
                        <p className="mt-2 text-xs text-gray-500 bg-gray-50 p-2 rounded-md">
                          目标：寻找使传统夏普比率 <code>(年化收益 - 无风险利率) / 年化波动率</code> 最大化的权重。
                        </p>
                      )}
                      {s.cfg?.target === 'risk_min_given_return' && (
                        <p className="mt-2 text-xs text-gray-500 bg-gray-50 p-2 rounded-md">
                          目标：在组合收益等于<strong>目标收益值</strong>的前提下，寻找使组合风险最小化的权重。
                        </p>
                      )}
                      {s.cfg?.target === 'return_max_given_risk' && (
                        <p className="mt-2 text-xs text-gray-500 bg-gray-50 p-2 rounded-md">
                          目标：在组合风险不高于<strong>目标风险值</strong>的前提下，寻找使组合收益最大化的权重。
                        </p>
                      )}
                    </div>
                    <div>
                      <label className="block text-xs text-gray-600">收益率类型</label>
                      <select value={s.cfg?.return_type||'simple'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, return_type:e.target.value}}:x))} className="mt-1 w-full rounded border-gray-300">
                        <option value="simple">普通收益率</option>
                        <option value="log">对数收益率</option>
                      </select>
                    </div>
                  </div>

                  {/* 根据目标类型显示不同UI */}
                  {s.cfg?.target === 'max_sharpe_traditional' ? (
                    <div className="space-y-3 rounded border p-3 text-sm">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs text-gray-600">收益指标 (固定)</label>
                          <input type="text" value="年化收益率均值" disabled className="mt-1 w-full rounded border-gray-200 bg-gray-100 px-2 py-1"/>
                        </div>
                        <div>
                          <label className="block text-xs text-gray-600">风险指标 (固定)</label>
                          <input type="text" value="年化波动率" disabled className="mt-1 w-full rounded border-gray-200 bg-gray-100 px-2 py-1"/>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs text-gray-600">年化天数</label>
                          <input type="number" value={s.cfg?.days ?? 252} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, days:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                        </div>
                        <div>
                          <label className="block text-xs text-gray-600">年化无风险利率(%)</label>
                          <input type="number" step="0.1" value={s.cfg?.risk_free_rate_pct ?? 1.5} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_free_rate_pct:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <>
                      {/* 收益指标配置 */}
                      <div className="rounded border p-3 text-sm space-y-3">
                        <div>
                          <label className="block text-xs text-gray-600">收益指标</label>
                          <select value={s.cfg?.return_metric||'annual'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, return_metric:e.target.value}}:x))} className="mt-1 w-full rounded border-gray-300">
                            <option value="annual">年化收益率</option>
                            <option value="annual_mean">年化收益率均值</option>
                            <option value="cumulative">累计收益率</option>
                            <option value="mean">收益率均值</option>
                            <option value="ewm">指数加权收益率</option>
                          </select>
                        </div>
                        {(s.cfg?.return_metric==='annual' || s.cfg?.return_metric==='annual_mean') && (
                          <div>
                            <label className="block text-xs text-gray-600">年化天数</label>
                            <input type="number" value={s.cfg?.days ?? 252} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, days:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                          </div>
                        )}
                        {s.cfg?.return_metric==='ewm' && (
                          <div className="grid grid-cols-2 gap-3">
                            <div>
                              <label className="block text-xs text-gray-600">衰减因子 λ</label>
                              <input type="number" step={0.01} value={s.cfg?.ret_alpha ?? 0.94} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, ret_alpha:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                            </div>
                            <div>
                              <label className="block text-xs text-gray-600">窗口长度</label>
                              <input type="number" value={s.cfg?.ret_window ?? 60} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, ret_window:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                            </div>
                          </div>
                        )}
                      </div>

                      {/* 风险指标配置 */}
                      <div className="rounded border p-3 text-sm space-y-3">
                        <div>
                          <label className="block text-xs text-gray-600">风险指标</label>
                          <select value={s.cfg?.risk_metric||'vol'} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_metric:e.target.value}}:x))} className="mt-1 w-full rounded border-gray-300">
                            <option value="vol">波动率</option>
                            <option value="annual_vol">年化波动率</option>
                            <option value="ewm_vol">指数加权波动率</option>
                            <option value="var">VaR</option>
                            <option value="es">ES</option>
                            <option value="max_drawdown">最大回撤</option>
                            <option value="downside_vol">下行波动率</option>
                          </select>
                        </div>
                        {s.cfg?.risk_metric==='annual_vol' && (
                          <div>
                            <label className="block text-xs text-gray-600">年化天数</label>
                            <input type="number" value={s.cfg?.risk_days ?? 252} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_days:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                          </div>
                        )}
                        {s.cfg?.risk_metric==='ewm_vol' && (
                          <div className="grid grid-cols-2 gap-3">
                            <div>
                              <label className="block text-xs text-gray-600">衰减因子 λ</label>
                              <input type="number" step={0.01} value={s.cfg?.risk_alpha ?? 0.94} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_alpha:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                            </div>
                            <div>
                              <label className="block text-xs text-gray-600">窗口长度</label>
                              <input type="number" value={s.cfg?.risk_window ?? 60} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_window:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                            </div>
                          </div>
                        )}
                        {(s.cfg?.risk_metric==='var' || s.cfg?.risk_metric==='es') && (
                          <div>
                            <label className="block text-xs text-gray-600">置信度%</label>
                            <input type="number" value={s.cfg?.risk_confidence ?? 95} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, risk_confidence:Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                          </div>
                        )}
                      </div>
                    </>
                  )}
                  {(s.cfg?.target==='risk_min_given_return') && (
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <label className="block text-xs text-gray-600">目标收益值</label>
                        <input type="number" value={s.cfg?.target_return ?? ''} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, target_return: Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                      </div>
                    </div>
                  )}
                  {(s.cfg?.target==='return_max_given_risk') && (
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <label className="block text-xs text-gray-600">目标风险值</label>
                        <input type="number" value={s.cfg?.target_risk ?? ''} onChange={e=> setStrategies(prev=>prev.map(x=>x.id===s.id?{...x, cfg:{...x.cfg, target_risk: Number(e.target.value)}}:x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                      </div>
                    </div>
                  )}

                  {/* 结果表格：若开启 recalc 且有回测数据，则横向展示每次再平衡的权重，否则展示当前权重 */}
                  {s.rebalance?.enabled && s.rebalance?.recalc && (scheduleMarkers[s.name] || (btSeries && (btSeries.markers?.[s.name]))) ? (
                    <div className="rounded border overflow-x-auto">
                      {(() => {
                        const markers = scheduleMarkers[s.name] || (btSeries.markers[s.name]||[]);
                        const dates: string[] = markers.map((m:any)=> m.date);
                        const names: string[] = (btSeries?.asset_names || assetNames);
                        const weightsByAsset: number[][] = names.map((_:any, i:number)=> markers.map((m:any)=> (m.weights?.[i] ?? 0)));
                        return (
                          <table className="min-w-full whitespace-nowrap text-sm">
                            <thead className="bg-gray-50">
                              <tr>
                                <th className="px-3 py-2 text-left text-xs text-gray-600">大类名称</th>
                                {dates.map((d)=> (
                                  <th key={d} className="px-3 py-2 text-left text-xs text-gray-600">{d}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {names.map((n, rIdx)=> (
                                <tr key={n} className="border-t">
                                  <td className="px-3 py-2">{n}</td>
                                  {weightsByAsset[rIdx].map((w, cIdx)=> (
                                    <td key={cIdx} className="px-3 py-2">{(w*100).toFixed(2)}%</td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        );
                      })()}
                    </div>
                  ) : (
                    <div className="rounded border">
                      <table className="min-w-full">
                        <thead className="bg-gray-50 text-xs text-gray-600"><tr><th className="px-3 py-2 text-left">大类名称</th><th className="px-3 py-2 text-left">资金权重(%)</th></tr></thead>
                        <tbody className="text-sm">
                          {s.rows.map((r,i)=> (
                            <tr key={i} className="border-t">
                              <td className="px-3 py-2">{r.className}</td>
                              <td className="px-3 py-2">{r.weight==null? '-' : (r.weight?.toFixed(2))}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  {/* 模型计算区间 */}
                  <div className="rounded border p-3 text-sm">
                    <div className="grid grid-cols-3 gap-3 items-end">
                      <div>
                        <label className="block text-xs text-gray-600">窗口模式</label>
                        <select value={s.cfg?.window_mode || 'rollingN'} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, cfg: { ...(x.cfg||{}), window_mode: e.target.value } } : x))} className="mt-1 w-full rounded border-gray-300">
                          <option value="all">所有数据</option>
                          <option value="rollingN">最近N条</option>
                        </select>
                      </div>
                      { (s.cfg?.window_mode==='rollingN') && (
                        <div>
                          <label className="block text-xs text-gray-600">N（交易日）</label>
                          <input type="number" min={2} value={s.cfg?.data_len ?? 60} onChange={e=> setStrategies(prev=> prev.map(x=> x.id===s.id? { ...x, cfg: { ...(x.cfg||{}), data_len: Number(e.target.value) } } : x))} className="mt-1 w-full rounded border-gray-300 px-2 py-1"/>
                        </div>
                      )}
                    </div>
                    <p className="mt-2 text-xs text-gray-500">
                      模式说明：
                      <span className="ml-1 font-medium">所有数据</span> 使用回测开始至当期的全部样本；
                      <span className="ml-1 font-medium">最近N条</span> 使用当期之前最近 N 条的滚动窗口（推荐）。
                    </p>
                  </div>

                  <div>
                    <button disabled={busyStrategy===s.id} className="rounded bg-gray-100 px-3 py-1 text-sm disabled:opacity-50" onClick={async ()=>{
                      try{
                        setBusyStrategy(s.id);
                        if (s.rebalance?.enabled && s.rebalance?.recalc) {
                          setBtBusy(true);
                          const payload = { alloc_name: selectedAlloc, start_date: btStart || undefined, strategy: { type:'target', name:s.name, classes: s.rows.map(r=>({ name:r.className })), rebalance: s.rebalance, model: { target: s.cfg?.target, return_metric: s.cfg?.return_metric, return_type: s.cfg?.return_type, days: s.cfg?.ret_days ?? 252, ret_alpha: s.cfg?.ret_alpha, ret_window: s.cfg?.ret_window, risk_metric: s.cfg?.risk_metric || 'vol', risk_days: s.cfg?.risk_days, risk_alpha: s.cfg?.risk_alpha, risk_window: s.cfg?.risk_window, risk_confidence: s.cfg?.risk_confidence, risk_free_rate: (riskFreePct/100), constraints: { single_limits: singleLimits, group_limits: groupLimits }, window_mode: s.cfg?.window_mode || 'rollingN', data_len: s.cfg?.data_len, target_return: s.cfg?.target_return, target_risk: s.cfg?.target_risk } } };
                          const res = await fetch('/api/strategy/compute-schedule-weights',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
                          const dat = await res.json();
                          if(!res.ok) throw new Error(dat.detail||'权重计算失败');
                          const markers = (dat.dates||[]).map((d:string, idx:number)=> ({ date: d, weights: dat.weights[idx]||[] }));
                          setScheduleMarkers(prev => ({ ...prev, [s.name]: markers }));
                          const last = dat.weights && dat.weights.length ? dat.weights[dat.weights.length-1] : [];
                          setStrategies(prev=> prev.map(x=> x.id===s.id?{...x, rows: x.rows.map((r,i)=> ({...r, weight: last[i]!=null? Number((last[i]*100).toFixed(2)) : r.weight }))}:x));
                          setBtBusy(false);
                        } else {
                          const payload = { alloc_name: selectedAlloc, data_len: (s.cfg?.window_mode==='all') ? undefined : (s.cfg?.data_len ?? 60), window_mode: s.cfg?.window_mode || 'rollingN', strategy: { type:'target', name:s.name, classes: s.rows.map(r=>({ name:r.className })), target: s.cfg?.target, return_metric: s.cfg?.return_metric, return_type: s.cfg?.return_type, days: s.cfg?.ret_days ?? 252, risk_metric: s.cfg?.risk_metric || 'vol', window: s.cfg?.risk_window, confidence: s.cfg?.risk_confidence, risk_free_rate: (riskFreePct/100), constraints: { single_limits: singleLimits, group_limits: groupLimits }, target_return: s.cfg?.target_return, target_risk: s.cfg?.target_risk } };
                          const res = await fetch('/api/strategy/compute-weights',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
                          const dat = await res.json();
                          if(!res.ok) throw new Error(dat.detail||'计算失败');
                          const weights: number[] = dat.weights || [];
                          setStrategies(prev=> prev.map(x=> x.id===s.id?{...x, rows: x.rows.map((r,i)=> ({...r, weight: Number((weights[i]*100).toFixed(2)) }))}:x));
                        }
                      }catch(e:any){ alert(e.message||'反推失败'); }
                      finally { setBusyStrategy(null); }
                    }}>反推资金权重</button>
                  </div>

                  {/* 已替换为横向表格展示（见上）*/}
                </div>
              )}
            </div>
          ))}

          {/* 添加策略按钮与类型选择器，位于所有策略块的下方且位于回测模块上方 */}
          {!showAddPicker && (
            <div className="mt-4">
              <button
                onClick={() => {
                  if (assetNames.length === 0) { alert('请先加载方案'); return; }
                  setShowAddPicker(true);
                }}
                className="rounded bg-indigo-600 text-white px-3 py-2 text-sm">
                + 添加新的组合策略
              </button>
            </div>
          )}
              {showAddPicker && (
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <span className="text-sm text-gray-700">选择策略类型：</span>
                  {(['fixed','risk_budget','target'] as StrategyType[]).map(t => (
                    <button key={t} className="rounded bg-gray-100 px-3 py-1 text-sm" onClick={() => {
                      const id = `s${Date.now()}`;
                      if (t === 'fixed') {
                        setStrategies(prev => {
                          const eqArr = computeEqualPercents(assetNames);
                          const rows = assetNames.map((n,i)=> ({ className:n, weight: eqArr[i], budget: 100 }));
                          const name = uniqueStrategyName('固定比例策略', prev);
                          return [...prev, { id, type: t, name, rows, cfg: { mode: 'equal' } }];
                        });
                      } else if (t==='risk_budget') {
                        setStrategies(prev => {
                          const rows = assetNames.map(n=> ({ className:n, budget:100, weight: null }));
                          const name = uniqueStrategyName('风险预算策略', prev);
                          return [...prev, { id, type: t, name, rows, cfg: { risk_metric:'vol', window_mode:'rollingN', data_len:60 } }];
                        });
                      } else {
                        setStrategies(prev => {
                          const rows = assetNames.map(n=> ({ className:n, weight: null }));
                          const name = uniqueStrategyName('指定目标策略', prev);
                          return [...prev, { id, type: t, name, rows, cfg: { target:'min_risk', return_metric:'annual', window_mode:'rollingN', data_len:60 } }];
                        });
                      }
                      setShowAddPicker(false);
                    }}>{t==='fixed'?'固定比例': t==='risk_budget'?'风险预算':'指定目标'}</button>
                  ))}
                  <button className="ml-2 rounded bg-white border px-2 py-1 text-xs" onClick={()=> setShowAddPicker(false)}>取消</button>
                </div>
              )}

          {/* 策略回测 */}
          <div className="rounded-lg border p-4">
            <h3 className="font-medium text-gray-700">策略回测</h3>
            <div className="mt-2 flex items-center gap-3">
              <label className="text-sm text-gray-600">选择开始日期</label>
              <input type="date" value={btStart} onChange={e=> setBtStart(e.target.value)} className="rounded border-gray-300 px-2 py-1"/>
              <button disabled={btBusy} className="rounded bg-indigo-600 px-3 py-2 text-sm text-white disabled:opacity-50" onClick={async ()=>{
                try{
                  if(!selectedAlloc){ alert('请先选择方案'); return; }
                  setBtBusy(true);
                  // 1) 预计算需要模型计算的策略权重（风险预算/指定目标）
                  const updated = await Promise.all(strategies.map(async (s)=>{
                    if (s.type === 'risk_budget') {
                      const payloadCW = { alloc_name: selectedAlloc, data_len: (s.cfg?.window_mode==='all') ? undefined : (s.cfg?.data_len ?? 60), window_mode: s.cfg?.window_mode || 'all', strategy: { type:'risk_budget', name:s.name, classes: s.rows.map(r=>({ name:r.className, budget:r.budget??100 })), risk_metric: s.cfg?.risk_metric, confidence: s.cfg?.confidence, days: s.cfg?.days } };
                      const res = await fetch('/api/strategy/compute-weights',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payloadCW)});
                      const dat = await res.json();
                      if(!res.ok) throw new Error(dat.detail||'风险预算权重计算失败');
                      const ws: number[] = dat.weights || [];
                      return { ...s, rows: s.rows.map((r,i)=> ({...r, weight: Number((ws[i]*100).toFixed(2)) })) } as StrategyRow;
                    }
                    if (s.type === 'target') {
                      const payloadCW = { alloc_name: selectedAlloc, data_len: (s.cfg?.window_mode==='all') ? undefined : (s.cfg?.data_len ?? 60), window_mode: s.cfg?.window_mode || 'all', strategy: { type:'target', name:s.name, classes: s.rows.map(r=>({ name:r.className })), target: s.cfg?.target, return_metric: s.cfg?.return_metric, return_type: s.cfg?.return_type, days: s.cfg?.ret_days ?? 252, risk_metric: s.cfg?.risk_metric || 'vol', window: s.cfg?.risk_window, confidence: s.cfg?.risk_confidence, risk_free_rate: (riskFreePct/100), constraints: { single_limits: singleLimits, group_limits: groupLimits }, target_return: s.cfg?.target_return, target_risk: s.cfg?.target_risk } };
                      const res = await fetch('/api/strategy/compute-weights',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payloadCW)});
                      const dat = await res.json();
                      if(!res.ok) throw new Error(dat.detail||'指定目标权重计算失败');
                      const ws: number[] = dat.weights || [];
                      return { ...s, rows: s.rows.map((r,i)=> ({...r, weight: Number((ws[i]*100).toFixed(2)) })) } as StrategyRow;
                    }
                    return s;
                  }));
                  setStrategies(updated);
                  // 2) 发送回测
                  const payload = { 
                    alloc_name: selectedAlloc, 
                    start_date: btStart || undefined, 
                    strategies: updated.map(s=> ({ 
                      type:s.type, 
                      name:s.name, 
                      classes: s.rows.map(r=> ({ name: r.className, weight: (r.weight??0)/100, budget: r.budget })), 
                      rebalance: s.rebalance,
                      model: (s.type==='risk_budget') ? {
                        risk_metric: s.cfg?.risk_metric, 
                        days: s.cfg?.days, 
                        confidence: s.cfg?.confidence,
                        window_mode: s.cfg?.window_mode || 'all', 
                        data_len: s.cfg?.data_len
                      } : (s.type==='target') ? {
                        target: s.cfg?.target,
                        return_metric: s.cfg?.return_metric,
                        return_type: s.cfg?.return_type,
                        days: s.cfg?.ret_days ?? 252,
                        ret_alpha: s.cfg?.ret_alpha,
                        ret_window: s.cfg?.ret_window,
                        risk_metric: s.cfg?.risk_metric,
                        risk_days: s.cfg?.risk_days,
                        risk_alpha: s.cfg?.risk_alpha,
                        risk_window: s.cfg?.risk_window,
                        risk_confidence: s.cfg?.risk_confidence,
                        risk_free_rate: (riskFreePct/100),
                        constraints: { single_limits: singleLimits, group_limits: groupLimits },
                        target_return: s.cfg?.target_return,
                        target_risk: s.cfg?.target_risk,
                        window_mode: s.cfg?.window_mode || 'all',
                        data_len: s.cfg?.data_len
                      } : undefined
                    })) 
                  };
                  const res = await fetch('/api/strategy/backtest',{ method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
                  const dat = await res.json();
                  if(!res.ok) throw new Error(dat.detail||'回测失败');
                  setBtSeries(dat);
                }catch(e:any){ alert(e.message||'回测失败'); }
                finally { setBtBusy(false); }
              }}>开始策略回测</button>
            </div>
            {btSeries && (
              <div className="mt-4">
                <ReactECharts style={{height: 360}} option={{
                  tooltip: { 
                    trigger:'axis',
                    formatter: (params: any) => {
                      const arr = Array.isArray(params) ? params : [params];
                      const header = arr[0]?.axisValue || '';
                      const lines = arr.map((p: any) => {
                        if (p.seriesType === 'scatter' && p.data && p.data.weights) {
                          const names = btSeries.asset_names || [];
                          const ws: number[] = p.data.weights || [];
                          const weightLines = names.map((n: string, i: number) => `${n}: ${(ws[i]*100).toFixed(2)}%`).join('<br/>');
                          return `${p.seriesName}: ${Number(p.data.value[1]).toFixed(2)}<br/>${weightLines}`;
                        }
                        return `${p.seriesName}: ${Number(p.data).toFixed(2)}`;
                      });
                      return `${header}<br/>${lines.join('<br/>')}`;
                    }
                  },
                  legend: { top: 0 },
                  dataZoom: [
                    { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
                    { type: 'slider', xAxisIndex: 0, filterMode: 'none', bottom: 24, height: 20 }
                  ],
                  grid: { top: 40, right: 10, bottom: 80, left: 60 },
                  xAxis: { type:'category', data: btSeries.dates },
                  yAxis: { type:'value', name:'组合净值', axisLabel: { formatter: (v: any) => Number(v).toFixed(2) } },
                  series: [
                    ...Object.keys(btSeries.series||{}).map((k:string)=> ({ name:k, type:'line', showSymbol:false, data: btSeries.series[k] })),
                    ...Object.keys(btSeries.markers||{}).flatMap((k:string)=> {
                      const arr = (btSeries.markers[k]||[]).map((m:any)=> ({ value: [m.date, m.value], weights: m.weights }));
                      return [{ name: `${k}-rebal`, type:'scatter', symbolSize:6, data: arr }];
                    })
                  ]
                }}/>
              </div>
            )}

            {/*（移除：回测模块内的重复添加按钮）*/}
          </div>
        </div>
      </Section>
    </div>
  );
}
