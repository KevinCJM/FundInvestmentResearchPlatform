import React, { useMemo, useState, useEffect } from 'react'
import ReactECharts from 'echarts-for-react'

type WeightMode = 'custom' | 'equal' | 'risk'
type RiskMetric = 'vol' | 'var' | 'es'

interface ETFItem {
  code: string
  name: string
  weight?: number
  riskContribution?: number
  solved?: boolean
  management?: string
  found_date?: string
}

interface AssetClass {
  id: string
  name: string
  mode: WeightMode
  etfs: ETFItem[]
  riskMetric?: RiskMetric
  maxLeverage?: number
}

const ETF_UNIVERSE: { code: string; name: string }[] = [
  { code: '510300', name: '沪深300ETF' },
  { code: '510500', name: '中证500ETF' },
  { code: '159919', name: '沪深300ETF 易方达' },
  { code: '159922', name: '中证500ETF 嘉实' },
  { code: '510050', name: '上证50ETF' },
  { code: '159915', name: '创业板ETF' },
  { code: 'TLT', name: 'iShares 20+ Year Treasury' },
  { code: 'IEF', name: 'iShares 7-10 Year Treasury' },
  { code: 'SHY', name: 'iShares 1-3 Year Treasury' },
  { code: 'AGG', name: 'iShares Core US Aggregate Bond' },
  { code: 'LQD', name: 'iShares iBoxx $ Inv Grade Corp Bd' },
  { code: 'HYG', name: 'iShares iBoxx $ High Yield Corp Bd' },
  { code: 'SPY', name: 'SPDR S&P 500 ETF Trust' },
  { code: 'QQQ', name: 'Invesco QQQ Trust' },
]

function fuzzySearchTop(q: string, k = 10) {
  const query = q.trim().toLowerCase()
  if (!query) return ETF_UNIVERSE.slice(0, k)
  const scored = ETF_UNIVERSE
    .map((x) => {
      const hay = (x.code + ' ' + x.name).toLowerCase()
      const idx = hay.indexOf(query)
      const score = idx === -1 ? Infinity : idx + query.length * 0.2
      return { x, score }
    })
    .filter((s) => s.score !== Infinity)
    .sort((a, b) => a.score - b.score)
    .slice(0, k)
    .map((s) => s.x)
  return scored
}

function uid() {
  return Math.random().toString(36).slice(2, 10)
}

function clamp(n: number, a: number, b: number) {
  return Math.max(a, Math.min(b, n))
}

function round2(n: number) {
  return Math.round(n * 100) / 100
}

function equalWeights(n: number): number[] {
  if (n <= 0) return []
  const w = 100 / n
  const arr = Array(n).fill(Math.floor(w * 100) / 100) as number[]
  let rem = round2(100 - arr.reduce((a, b) => a + b, 0))
  let i = 0
  while (rem > 0 && i < n) {
    arr[i] = round2(arr[i] + 0.01)
    rem = round2(rem - 0.01)
    i++
  }
  return arr
}

export default function AssetClassConstructionPage() {
  const [classes, setClasses] = useState<AssetClass[]>([])

  const [loading, setLoading] = useState(false)
  const [searchOpen, setSearchOpen] = useState<{ open: boolean; classId?: string }>({ open: false })
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<ETFItem[]>([])
  const [sortBy, setSortBy] = useState<'name' | 'code' | 'management' | 'found_date'>('name')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc')
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [total, setTotal] = useState(0)
  const [fitLoading, setFitLoading] = useState(false)
  const [startDate, setStartDate] = useState<string>('2020-01-01')
  const [fitResult, setFitResult] = useState<null | { dates: string[]; navs: Record<string, number[]>; corr: number[][]; corr_labels: string[]; metrics: { name: string; annual_return?: number; annual_vol?: number; sharpe?: number; var99?: number; es99?: number; max_drawdown?: number; calmar?: number }[]; consistency: { name: string; mean_corr?: number; pca_evr1?: number; max_te?: number }[] }>(null)
  const [rollLoading, setRollLoading] = useState(false)
  const [rollResult, setRollResult] = useState<null | { dates: string[]; series: Record<string, number[]>; metrics: { name: string; overall:number; mean:number; median:number; std:number; skew:number; kurtosis:number }[] }>(null)
  const [rollWindow, setRollWindow] = useState<number>(60)
  const classOptions = useMemo(()=> classes.map(c=> c.name), [classes])
  const [rollTargetClass, setRollTargetClass] = useState<string>('')

  // --- Save/Load States ---
  const [saveModal, setSaveModal] = useState(false)
  const [loadModal, setLoadModal] = useState(false)
  const [allocName, setAllocName] = useState('')
  const [allocList, setAllocList] = useState<string[]>([])
  const [allocSearch, setAllocSearch] = useState('')

  function updateClass(id: string, updater: (c: AssetClass) => AssetClass) {
    setClasses((prev) => prev.map((c) => (c.id === id ? updater(c) : c)))
  }

  function setCustomWeight(classId: string, idx: number, val: number) {
    updateClass(classId, (c) => {
      const etfs = c.etfs.map((e, i) => (i === idx ? { ...e, weight: clamp(val, 0, 100) } : e))
      return { ...c, etfs }
    })
  }

  function setRiskContribution(classId: string, idx: number, val: number) {
    updateClass(classId, (c) => {
      const etfs = c.etfs.map((e, i) => (i === idx ? { ...e, riskContribution: clamp(val, 0, 100) } : e))
      return { ...c, etfs }
    })
  }

  function setMaxLeverage(classId: string, val: number) {
    updateClass(classId, (c) => ({ ...c, maxLeverage: clamp(val, 0, 100) }))
  }

  function addETFToClass(classId: string, etf: { code: string; name: string }) {
    updateClass(classId, (c) => {
      const exists = c.etfs.some((x) => x.code === etf.code && x.name === etf.name)
      const etfs = exists
        ? c.etfs
        : [
            ...c.etfs,
            {
              ...etf,
              weight: c.mode === 'custom' ? 0 : undefined,
              riskContribution: c.mode === 'risk' ? 0 : undefined,
              solved: false,
            },
          ]
      return { ...c, etfs }
    })
  }

  function removeETF(classId: string, idx: number) {
    updateClass(classId, (c) => ({ ...c, etfs: c.etfs.filter((_, i) => i !== idx) }))
  }

  function addAssetClass() {
    setClasses((prev) => [...prev, { id: uid(), name: '新大类', mode: 'custom', etfs: [], riskMetric: 'vol', maxLeverage: 0 }])
  }

  function deleteAssetClass(id: string) {
    setClasses((prev) => prev.filter((c) => c.id !== id))
  }

  async function onSolveRiskWeights(classId: string) {
    const ac = classes.find((c) => c.id === classId)
    if (!ac) return
    if (ac.mode !== 'risk') {
      alert('请先切换到“风险平价”模式')
      return
    }
    const sumRisk = ac.etfs.reduce((a, e) => a + (e.riskContribution ?? 0), 0)
    if (Math.abs(sumRisk - 100) > 1e-6) {
      alert('风险贡献合计需等于 100%，请调整后再计算')
      return
    }
    try {
      setLoading(true)
      const payload = {
        assetClassId: ac.id,
        riskMetric: ac.riskMetric ?? 'vol',
        maxLeverage: ac.maxLeverage ?? 0,
        etfs: ac.etfs.map((e) => ({ code: e.code, name: e.name, riskContribution: e.riskContribution ?? 0 })),
      }
      const resp = await fetch('/api/risk-parity/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!resp.ok) throw new Error(`后端返回错误状态 ${resp.status}`)
      const data: { weights: number[] } = await resp.json()
      if (!Array.isArray(data.weights) || data.weights.length !== ac.etfs.length) {
        throw new Error('返回的权重数量与ETF数量不一致')
      }
      updateClass(classId, (c) => ({
        ...c,
        etfs: c.etfs.map((e, i) => ({ ...e, weight: round2(Math.max(0, data.weights[i])), solved: true })),
      }))
      // 成功后直接回显（不弹窗）
    } catch (err: any) {
      console.error(err)
      alert('计算失败：' + err.message + '\n请确认已启动 Python 后端 (POST /api/risk-parity/solve)。')
    } finally {
      setLoading(false)
    }
  }

  async function onFit() {
    // 校验参数
    if (!startDate) {
      alert('请选择开始日期')
      return
    }
    const payloadClasses = classes.map((ac) => {
      // 生成资金权重
      let weights: number[] = []
      if (ac.mode === 'equal') {
        weights = equalWeights(ac.etfs.length)
      } else {
        weights = ac.etfs.map((e) => Number(e.weight || 0))
      }
      const sumW = weights.reduce((a, b) => a + b, 0)
      if (ac.mode !== 'equal' && Math.abs(sumW - 100) > 1e-4) {
        throw new Error(`大类【${ac.name}】资金权重合计应为 100%`)
      }
      if (ac.mode === 'risk' && sumW <= 0) {
        throw new Error(`大类【${ac.name}】请先完成“反推资金权重”计算`)
      }
      return {
        id: ac.id,
        name: ac.name,
        etfs: ac.etfs.map((e, i) => ({ code: e.code, name: e.name, weight: weights[i] || 0 })),
      }
    })
    try {
      setFitLoading(true)
      setFitResult(null)
      const resp = await fetch('/api/fit-classes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ startDate, classes: payloadClasses }),
      })
      if (!resp.ok) throw new Error(`后端错误 ${resp.status}`)
      const data = await resp.json()
      setFitResult(data)
    } catch (e: any) {
      alert('拟合失败：' + (e?.message || e))
    } finally {
      setFitLoading(false)
    }
  }

  async function onRoll() {
    if (!rollTargetClass) {
      alert('请选择研究对象大类')
      return
    }
    try {
      setRollLoading(true)
      setRollResult(null)
      // 准备大类及资金权重
      const payloadClasses = classes.map((ac) => {
        let weights: number[] = []
        if (ac.mode === 'equal') weights = equalWeights(ac.etfs.length)
        else weights = ac.etfs.map((e)=> Number(e.weight||0))
        return {
          id: ac.id,
          name: ac.name,
          etfs: ac.etfs.map((e,i)=> ({ code: e.code, name: e.name, weight: weights[i]||0 }))
        }
      })
      const resp = await fetch('/api/rolling-corr-classes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ startDate, window: rollWindow, targetClassName: rollTargetClass, classes: payloadClasses })
      })
      if (!resp.ok) throw new Error(`后端错误 ${resp.status}`)
      const data = await resp.json()
      setRollResult(data)
    } catch (e:any) {
      alert('滚动相关性计算失败：' + (e?.message||e))
    } finally {
      setRollLoading(false)
    }
  }

  useEffect(() => {
    try {
      const ew = equalWeights(3)
      console.assert(ew.length === 3, 'equalWeights 长度应为 3')
      console.assert(Math.abs(ew.reduce((a, b) => a + b, 0) - 100) < 1e-6, 'equalWeights 合计应为 100')
      const riskSum = [{ v: 60 }, { v: 40 }].reduce((a, e) => a + e.v, 0)
      console.assert(riskSum === 100, '手动风险贡献合计应为 100')
      console.assert(clamp(150, 0, 100) === 100, 'clamp 上界应生效')
      console.assert(round2(1.234) === 1.23 && round2(1.235) === 1.24, 'round2 四舍五入应生效')
      console.assert(ETF_UNIVERSE.some((x) => x.code === 'SPY'), '搜索应包含 SPY')
    } catch (e) {
      console.error('内置测试失败:', e)
    }
  }, [])

  // 页面初始化：从后端读取默认两类
  useEffect(() => {
    const init = async () => {
      try {
        const fetchTop = async (keyword: string) => {
          const resp = await fetch(`/api/etf/search?q=${encodeURIComponent(keyword)}&page=1&page_size=2&sort_by=name&sort_dir=asc`)
          if (!resp.ok) throw new Error('search failed')
          const data = await resp.json()
          const items = (data.items || []) as ETFItem[]
          return items
        }
        const eq = await fetchTop('沪深300')
        const bond = await fetchTop('国债')
        setClasses([
          {
            id: uid(),
            name: '权益类',
            mode: 'custom',
            etfs: eq.map((e, i) => ({ ...e, weight: i === 0 ? 80 : 20 })),
            riskMetric: 'vol',
            maxLeverage: 0,
          },
          {
            id: uid(),
            name: '固收类',
            mode: 'equal',
            etfs: bond,
            riskMetric: 'vol',
            maxLeverage: 0,
          },
        ])
      } catch {
        // 回退：保留空列表，用户自行添加
        setClasses([
          { id: uid(), name: '权益类', mode: 'custom', etfs: [], riskMetric: 'vol', maxLeverage: 0 },
          { id: uid(), name: '固收类', mode: 'equal', etfs: [], riskMetric: 'vol', maxLeverage: 0 },
        ])
      }
    }
    if (classes.length === 0) init()
  }, [])

  // 搜索：优先从后端读取 data 下的 ETF 列表（JSON/Parquet），失败时回退本地模糊匹配；支持排序与分页
  useEffect(() => {
    const controller = new AbortController()
    const doFetch = async () => {
      try {
        const params = new URLSearchParams({
          q: searchQuery,
          sort_by: sortBy,
          sort_dir: sortDir,
          page: String(page),
          page_size: String(pageSize),
        })
        const url = `/api/etf/search?${params.toString()}`
        const resp = await fetch(url, { signal: controller.signal })
        if (!resp.ok) throw new Error(`status ${resp.status}`)
        const data = await resp.json()
        if (Array.isArray(data?.items)) {
          setSearchResults(data.items)
          setTotal(Number(data.total || 0))
        } else {
          const local = fuzzySearchTop(searchQuery, pageSize)
          setSearchResults(local)
          setTotal(local.length)
        }
      } catch {
        const local = fuzzySearchTop(searchQuery, pageSize)
        setSearchResults(local)
        setTotal(local.length)
      }
    }
    doFetch()
    return () => controller.abort()
  }, [searchQuery, sortBy, sortDir, page, pageSize])

  const busy = loading || fitLoading || rollLoading

  // --- Save/Load Handlers ---
  async function handleSave() {
    const name = allocName.trim()
    if (!name) {
      alert('配置名称不能为空')
      return
    }
    if (allocList.includes(name)) {
      if (!confirm(`配置名称 “${name}” 已存在，要覆盖吗？（此操作不可逆）`)) {
        return
      }
    }
    // 准备权重数据
    const payloadClasses = classes.map((ac) => {
      let weights: number[] = []
      if (ac.mode === 'equal') weights = equalWeights(ac.etfs.length)
      else weights = ac.etfs.map((e) => Number(e.weight || 0))
      return {
        id: ac.id,
        name: ac.name,
        etfs: ac.etfs.map((e, i) => ({ code: e.code, name: e.name, weight: weights[i] || 0 })),
      }
    })

    try {
      setLoading(true)
      const resp = await fetch('/api/save-allocation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ asset_alloc_name: name, classes: payloadClasses }),
      })
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.detail || `错误 ${resp.status}`)
      alert(`配置 “${name}” 保存成功！`)
      setSaveModal(false)
      setAllocName('')
      setAllocList(p => Array.from(new Set([...p, name])).sort())
    } catch (e: any) {
      alert('保存失败：' + (e.message || e))
    } finally {
      setLoading(false)
    }
  }

  async function handleLoad(name: string) {
    if (!name) return
    try {
      setLoading(true)
      const resp = await fetch(`/api/load-allocation?name=${encodeURIComponent(name)}`)
      const data = await resp.json()
      if (!resp.ok) throw new Error(data.detail || `错误 ${resp.status}`)
      setClasses(data)
      setLoadModal(false)
      setAllocSearch('')
    } catch (e: any) {
      alert('加载失败：' + (e.message || e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    const fetchAllocations = async () => {
      try {
        const resp = await fetch('/api/list-allocations')
        if (resp.ok) {
          const data = await resp.json()
          if (Array.isArray(data)) setAllocList(data.sort())
        }
      } catch (e) {
        console.error("Failed to fetch allocation list", e)
      }
    }
    fetchAllocations()
  }, [])


  return (
    <div className="mx-auto max-w-5xl p-6 relative">
      {busy && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="rounded-xl bg-white px-6 py-4 shadow text-sm">正在计算，请稍候...</div>
        </div>
      )}
      <h1 className="text-2xl font-semibold">资产大类构建模块</h1>
      <p className="text-sm text-gray-500 mt-1">配置资产大类、ETF 选择与权重；风险平价支持手动风险贡献、最大杠杆与后端反推权重</p>

      <div className="mt-5 rounded-2xl border border-gray-200 bg-white p-4">
        <div className="flex justify-between items-center">
          <SectionTitle title="构建资产大类" />
          <button 
            className="rounded-md bg-gray-100 px-3 py-1 text-xs text-gray-700 hover:bg-gray-200"
            onClick={() => setLoadModal(true)} >
              导入大类配置
          </button>
        </div>
        <div className="space-y-6">
          {classes.map((ac) => (
            <AssetClassCard
              key={ac.id}
              ac={ac}
              on重命名={(name) => updateClass(ac.id, (c) => ({ ...c, name }))}
              onModeChange={(mode) => updateClass(ac.id, (c) => ({ ...c, mode }))}
              onRiskMetricChange={(metric) => updateClass(ac.id, (c) => ({ ...c, riskMetric: metric }))}
              on删除={() => deleteAssetClass(ac.id)}
              onAddETF={() => setSearchOpen({ open: true, classId: ac.id })}
              onRemoveETF={(idx) => removeETF(ac.id, idx)}
              onSetCustomWeight={(idx, v) => setCustomWeight(ac.id, idx, v)}
              onSetRiskContribution={(idx, v) => setRiskContribution(ac.id, idx, v)}
              onSetMaxLeverage={(v) => setMaxLeverage(ac.id, v)}
              onSolve={() => onSolveRiskWeights(ac.id)}
              loading={loading}
            />
          ))}

          <button className="w-full rounded-xl border border-dashed border-gray-300 py-3 text-sm bg-blue-50/50 hover:bg-blue-50" onClick={addAssetClass}>
            + 添加新大类
          </button>
        </div>
      </div>

      <div className="mt-6 text-center">
          <button 
            className="rounded-lg bg-blue-600 px-6 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-700"
            onClick={() => setSaveModal(true)} >
              保存当前大类配置
          </button>
      </div>

      {/* Save Modal */}
      {saveModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-md rounded-2xl bg-white p-5 shadow-xl">
            <h3 className="text-lg font-semibold">保存大类配置</h3>
            <input
              autoFocus
              value={allocName}
              onChange={(e) => setAllocName(e.target.value)}
              placeholder="请输入配置名称..."
              className="mt-3 w-full rounded-lg border border-gray-300 px-3 py-2 outline-none focus:ring-2 focus:ring-blue-500"
            />
            <div className="mt-4 flex justify-end gap-3">
              <button className="rounded-lg bg-gray-100 px-4 py-2 text-sm hover:bg-gray-200" onClick={() => setSaveModal(false)}>取消</button>
              <button className="rounded-lg bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700" onClick={handleSave}>保存</button>
            </div>
          </div>
        </div>
      )}

      {/* Load Modal */}
      {loadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-xl rounded-2xl bg-white p-5 shadow-xl">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">导入大类配置</h3>
              <button className="text-gray-500" onClick={() => setLoadModal(false)}>✕</button>
            </div>
            <input
              autoFocus
              value={allocSearch}
              onChange={(e) => setAllocSearch(e.target.value)}
              placeholder="按名称搜索..."
              className="mt-3 w-full rounded-lg border border-gray-300 px-3 py-2 outline-none focus:ring-2 focus:ring-blue-500"
            />
            <div className="mt-3 max-h-80 overflow-auto rounded-lg border border-gray-100">
              {allocList.filter(name => name.toLowerCase().includes(allocSearch.toLowerCase())).map(name => (
                <button
                  key={name}
                  onClick={() => handleLoad(name)}
                  className="flex w-full items-center justify-between border-b px-4 py-2 text-left hover:bg-gray-50"
                >
                  <span className="text-sm text-gray-800">{name}</span>
                  <span className="text-xs text-gray-400">导入</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {searchOpen.open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-xl rounded-2xl bg-white p-5 shadow-xl">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">添加 ETF</h3>
              <button className="text-gray-500" onClick={() => setSearchOpen({ open: false })}>✕</button>
            </div>
            <input
              autoFocus
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="按代码或名称搜索..."
              className="mt-3 w-full rounded-lg border border-gray-300 px-3 py-2 outline-none focus:ring-2 focus:ring-blue-500"
            />
            <div className="mt-3 max-h-80 overflow-auto rounded-lg border border-gray-100">
              {searchResults.length === 0 && <div className="p-4 text-sm text-gray-500">无匹配结果</div>}
              {searchResults.map((etf) => (
                <button
                  key={etf.code + etf.name}
                  onClick={() => {
                    if (searchOpen.classId) addETFToClass(searchOpen.classId, etf)
                    setSearchQuery('')
                    setSearchOpen({ open: false })
                  }}
                  className="flex w-full items-center justify-between border-b px-4 py-2 text-left hover:bg-gray-50"
                >
                  <div className="flex-1 flex items-center gap-2">
                    <span className="font-mono text-sm">{etf.code}</span>
                    <span className="truncate px-2 text-sm text-gray-700">{etf.name}</span>
                  </div>
                  <div className="hidden md:block text-right text-xs text-gray-500 mr-3">
                    <div>基金公司：{etf.management || '—'}</div>
                    <div>成立日期：{etf.found_date || '—'}</div>
                  </div>
                  <span className="text-xs text-gray-400">添加</span>
                </button>
              ))}
            </div>
            {/* 分页与排序控制 */}
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <label className="text-xs text-gray-600">排序</label>
                <select className="border rounded px-2 py-1 text-xs" value={sortBy} onChange={(e) => { setSortBy(e.target.value as any); setPage(1) }}>
                  <option value="name">名称</option>
                  <option value="code">代码</option>
                  <option value="management">基金公司</option>
                  <option value="found_date">成立日期</option>
                </select>
                <select className="border rounded px-2 py-1 text-xs" value={sortDir} onChange={(e) => { setSortDir(e.target.value as any); setPage(1) }}>
                  <option value="asc">升序</option>
                  <option value="desc">降序</option>
                </select>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-xs text-gray-600">每页</label>
                <select className="border rounded px-2 py-1 text-xs" value={pageSize} onChange={(e) => { setPageSize(Number(e.target.value)); setPage(1) }}>
                  {[5,10,20,50].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
                <span className="text-xs text-gray-600">共 {total} 条</span>
                <button className="border rounded px-2 py-1 text-xs" disabled={page<=1} onClick={() => setPage(p=>Math.max(1,p-1))}>上一页</button>
                <span className="text-xs">第 {page} 页</span>
                <button className="border rounded px-2 py-1 text-xs" disabled={page*pageSize>=total} onClick={() => setPage(p=>p+1)}>下一页</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 拟合区域 */}
      <div className="mt-6 rounded-2xl border border-gray-200 bg-white p-4">
        <SectionTitle title="大类收益率拟合" />
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 text-sm">
            <label className="text-gray-700">选择开始日期</label>
            <input type="date" className="border rounded px-2 py-1" value={startDate} onChange={(e)=>setStartDate(e.target.value)} />
          </div>
          <button className="rounded-md bg-blue-600 px-3 py-1 text-xs text-white hover:bg-blue-700" onClick={onFit} disabled={busy}>
            拟合大类收益率
          </button>
        </div>
        {fitResult && (
          <div className="mt-4 space-y-6">
            <div>
              {(() => {
                const keys = Object.keys(fitResult.navs)
                return (
                  <ReactECharts style={{ height: 360 }} option={{
                    title: { text: '虚拟净值走势（起始=1）', left: 0, top: 0, textStyle: { fontSize: 13, fontWeight: 600 } },
                    tooltip: { trigger: 'axis', valueFormatter: (v:any)=> Number(v).toFixed(2) },
                    legend: { top: 0, right: 0 },
                    grid: { left: 56, right: 16, top: 36, bottom: 86 },
                    xAxis: { type: 'category', data: fitResult.dates, axisLabel: { showMaxLabel: true, hideOverlap: true, margin: 12 } },
                    // 自动范围 + padding，通过函数形式按数据动态设置范围
                    yAxis: {
                      type: 'value',
                      scale: true,
                      min: (v:any) => (Number.isFinite(v.min) && Number.isFinite(v.max)) ? v.min - (v.max - v.min) * 0.05 : 'dataMin',
                      max: (v:any) => (Number.isFinite(v.min) && Number.isFinite(v.max)) ? v.max + (v.max - v.min) * 0.05 : 'dataMax',
                      axisLabel: { formatter: (val:any)=> Number(val).toFixed(2) },
                    },
                    dataZoom: [
                      { type: 'inside' },
                      { type: 'slider', bottom: 36, height: 18 },
                    ],
                    series: keys.map((k)=>({
                      name: k,
                      type: 'line',
                      smooth: false, // 不使用平滑曲线
                      symbol: 'none',
                      lineStyle: { width: 2 },
                      data: fitResult.navs[k]
                    }))
                  }} />
                )
              })()}
            </div>
            <div>
              <h3 className="text-sm font-semibold mb-2">相关系数矩阵</h3>
              <ReactECharts style={{ height: 320 }} option={(function(){
                const labels = fitResult.corr_labels
                const data: any[] = []
                for(let i=0;i<labels.length;i++){
                  for(let j=0;j<labels.length;j++){
                    data.push([i, j, Number(fitResult.corr[i][j])])
                  }
                }
                return {
                  tooltip: { position: 'top', formatter: (p:any)=> `${labels[p.data[1]]} vs ${labels[p.data[0]]}: ${Number(p.data[2]).toFixed(2)}` },
                  grid: { left: 80, right: 16, top: 16, bottom: 40 },
                  xAxis: { type: 'category', data: labels, axisLabel: { rotate: 30 } },
                  yAxis: { type: 'category', data: labels },
                  // 配色：白色 → 天蓝色，隐藏色度图例
                  visualMap: {
                    min: -1, max: 1, show: false,
                    inRange: { color: ['#ffffff','#e0f2fe','#bae6fd','#7dd3fc','#38bdf8','#0ea5e9'] }
                  },
                  series: [{
                    type: 'heatmap',
                    data,
                    label: { show: true, formatter: (p:any)=> Number(p.data[2]).toFixed(2), color: '#111827' },
                    emphasis: { itemStyle: { shadowBlur: 5, shadowColor: 'rgba(0,0,0,0.3)' } }
                  }]
                }
              })()} />
            </div>
            <div>
              <h3 className="text-sm font-semibold mb-2">横向指标对比</h3>
              {(() => {
                const classes = fitResult.metrics.map(m => m.name)
                const rows = [
                  { label: '年化收益率(%)', values: fitResult.metrics.map(m=> Number((m.annual_return ?? NaN) * 100)) },
                  { label: '年化波动率(%)', values: fitResult.metrics.map(m=> Number((m.annual_vol ?? NaN) * 100)) },
                  { label: '夏普比率', values: fitResult.metrics.map(m=> Number(m.sharpe ?? NaN)) },
                  { label: '99%VaR(日)(%)', values: fitResult.metrics.map(m=> Number((m.var99 ?? NaN) * 100)) },
                  { label: '99%ES(日)(%)', values: fitResult.metrics.map(m=> Number((m.es99 ?? NaN) * 100)) },
                  { label: '最大回撤(%)', values: fitResult.metrics.map(m=> Number((m.max_drawdown ?? NaN) * 100)) },
                  { label: '卡玛比率', values: fitResult.metrics.map(m=> Number(m.calmar ?? NaN)) },
                ]
                const color = (val:number, min:number, max:number) => {
                  if (!Number.isFinite(val)) return { background: '#f3f4f6', color: '#6b7280' }
                  if (max <= min) return { background: '#d1fae5', color: '#065f46' }
                  const t = (val - min) / (max - min)
                  // 绿色渐变：低值浅， 高值深
                  const start = [209, 250, 229]
                  const end = [5, 150, 105]
                  const mix = (a:number,b:number)=> Math.round(a + (b-a)*t)
                  const bg = `rgb(${mix(start[0],end[0])},${mix(start[1],end[1])},${mix(start[2],end[2])})`
                  const txt = t > 0.6 ? '#ffffff' : '#065f46'
                  return { background: bg, color: txt }
                }
                return (
                  <div className="overflow-auto">
                    <table className="text-xs border" style={{ width: '100%', tableLayout: 'fixed' }}>
                      <thead>
                        <tr>
                          <th className="border px-2 py-2">指标</th>
                          {classes.map(c => <th key={'mh'+c} className="border px-2 py-2">{c}</th>)}
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map((row) => {
                          const finiteVals = row.values.filter(v=> Number.isFinite(v))
                          const min = finiteVals.length ? Math.min(...finiteVals) : 0
                          const max = finiteVals.length ? Math.max(...finiteVals) : 0
                          return (
                            <tr key={'mr'+row.label}>
                              <td className="border px-2 py-3 font-medium">{row.label}</td>
                              {row.values.map((v,i)=> {
                                const st = color(v, min, max)
                                return <td key={'mc'+row.label+'-'+i} className="border px-2 py-3 text-right" style={{ background: st.background, color: st.color }}>{Number.isFinite(v)? v.toFixed(2): '-'}</td>
                              })}
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                )
              })()}
            </div>
          </div>
        )}
      </div>

      {/* 同类资产一致性 */}
      {fitResult && Array.isArray(fitResult.consistency) && fitResult.consistency.length > 0 && (
        <div className="mt-4 rounded-2xl border border-gray-200 bg-white p-4">
          <SectionTitle title="同类资产一致性" />
          <div className="overflow-auto">
            <table className="text-xs border" style={{ width: '100%', tableLayout: 'fixed' }}>
              <thead>
                <tr>
                  <th className="border px-2 py-2">大类</th>
                  <th className="border px-2 py-2">相关性均值</th>
                  <th className="border px-2 py-2">主成分解释度(%)</th>
                  <th className="border px-2 py-2">最大跟踪误差(%)</th>
                </tr>
              </thead>
              <tbody>
                {fitResult.consistency.map((r)=> {
                  const meanCorr = r.mean_corr as number
                  const pcaEvr1 = r.pca_evr1 as number
                  const maxTe = r.max_te as number
                  const meanCorrStyle = Number.isFinite(meanCorr) && meanCorr < 0.6 ? { backgroundColor: '#fee2e2' } : {}
                  const pcaEvr1Style = Number.isFinite(pcaEvr1) && pcaEvr1 < 0.8 ? { backgroundColor: '#fee2e2' } : {}
                  const maxTeStyle = Number.isFinite(maxTe) && maxTe * 100 > 5 ? { backgroundColor: '#fee2e2' } : {}

                  return (
                    <tr key={r.name}>
                      <td className="border px-2 py-2">{r.name}</td>
                      <td className="border px-2 py-2 text-right" style={meanCorrStyle}>{Number.isFinite(meanCorr) ? meanCorr.toFixed(3) : '-'}</td>
                      <td className="border px-2 py-2 text-right" style={pcaEvr1Style}>{Number.isFinite(pcaEvr1) ? (pcaEvr1 * 100).toFixed(2) : '-'}</td>
                      <td className="border px-2 py-2 text-right" style={maxTeStyle}>{Number.isFinite(maxTe) ? (maxTe * 100).toFixed(2) : '-'}</td>
                    </tr>
                  )
                })}</tbody>
            </table>
          </div>
        </div>
      )}

      {/* 滚动相关性研究 */}
      <div className="mt-6 rounded-2xl border border-gray-200 bg-white p-4">
        <SectionTitle title="滚动相关性研究" />
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 text-sm">
            <label className="text-gray-700">滚动天数</label>
            <input type="number" min={5} step={1} className="border rounded px-2 py-1 w-24 text-right" value={rollWindow} onChange={(e)=> setRollWindow(Number(e.target.value)||60)} />
          </div>
          <div className="flex items-center gap-2 text-sm">
            <label className="text-gray-700">研究对象</label>
            <select className="border rounded px-2 py-1" value={rollTargetClass} onChange={(e)=> setRollTargetClass(e.target.value)}>
              <option value="">请选择大类</option>
              {classOptions.map(n=> <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <button className="rounded-md bg-blue-600 px-3 py-1 text-xs text-white hover:bg-blue-700" onClick={onRoll} disabled={busy}>
            计算滚动相关性
          </button>
        </div>
        {rollResult && (
          <div className="mt-4 space-y-4">
            <ReactECharts style={{ height: 320 }} option={{
              tooltip: { trigger: 'axis', valueFormatter: (v:any)=> Number(v).toFixed(2) },
              legend: { top: 0 },
              grid: { left: 48, right: 16, top: 16, bottom: 40 },
              xAxis: { type: 'category', data: rollResult.dates },
              yAxis: { type: 'value', min: -1, max: 1 },
              dataZoom: [{ type: 'inside' }, { type: 'slider' }],
              series: Object.keys(rollResult.series).map(k=> ({ name:k, type:'line', symbol:'none', data: rollResult.series[k] }))
            }} />
            <div className="overflow-auto">
              <table className="text-xs border" style={{ width: '100%', tableLayout: 'fixed' }}>
                <thead>
                  <tr>
                    <th className="border px-2 py-2">大类</th>
                    <th className="border px-2 py-2">整体相关系数</th>
                    <th className="border px-2 py-2">均值</th>
                    <th className="border px-2 py-2">中位数</th>
                    <th className="border px-2 py-2">标准差</th>
                    <th className="border px-2 py-2">偏度</th>
                    <th className="border px-2 py-2">峰度</th>
                  </tr>
                </thead>
                <tbody>
                  {rollResult.metrics.map(m=> (
                    <tr key={m.name}>
                      <td className="border px-2 py-2">{m.name}</td>
                      <td className="border px-2 py-2 text-right">{Number.isFinite(m.overall)? m.overall.toFixed(2): '-'}</td>
                      <td className="border px-2 py-2 text-right">{Number.isFinite(m.mean)? m.mean.toFixed(2): '-'}</td>
                      <td className="border px-2 py-2 text-right">{Number.isFinite(m.median)? m.median.toFixed(2): '-'}</td>
                      <td className="border px-2 py-2 text-right">{Number.isFinite(m.std)? m.std.toFixed(2): '-'}</td>
                      <td className="border px-2 py-2 text-right">{Number.isFinite(m.skew)? m.skew.toFixed(2): '-'}</td>
                      <td className="border px-2 py-2 text-right">{Number.isFinite(m.kurtosis)? m.kurtosis.toFixed(2): '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function SectionTitle({ title }: { title: string }) {
  return (
    <div className="mb-3 flex items-center gap-2">
      <span className="text-blue-600">◆</span>
      <h2 className="text-lg font-semibold">{title}</h2>
    </div>
  )
}

function ModePill({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`rounded-md px-2 py-1 text-xs ${active ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
      style={{ padding: '0.25rem 0.5rem' }}
    >
      {label}
    </button>
  )
}

function AssetClassCard({
  ac,
  on重命名,
  onModeChange,
  onRiskMetricChange,
  on删除,
  onAddETF,
  onRemoveETF,
  onSetCustomWeight,
  onSetRiskContribution,
  onSetMaxLeverage,
  onSolve,
  loading,
}: {
  ac: AssetClass
  on重命名: (name: string) => void
  onModeChange: (mode: WeightMode) => void
  onRiskMetricChange: (metric: RiskMetric) => void
  on删除: () => void
  onAddETF: () => void
  onRemoveETF: (idx: number) => void
  onSetCustomWeight: (idx: number, val: number) => void
  onSetRiskContribution: (idx: number, val: number) => void
  onSetMaxLeverage: (val: number) => void
  onSolve: () => void
  loading: boolean
}) {
  const [editing, setEditing] = useState(false)
  const [tempName, setTempName] = useState(ac.name)
  useEffect(() => setTempName(ac.name), [ac.name])

  const isRisk = ac.mode === 'risk'
  const showSolved = isRisk && ac.etfs.some((e) => e.solved)

  const equal = useMemo(() => (ac.mode === 'equal' ? equalWeights(ac.etfs.length) : []), [ac.mode, ac.etfs.length])

  const sumWeight = useMemo(() => {
    if (ac.mode === 'equal') return 100
    if (ac.mode === 'custom') return round2(ac.etfs.reduce((a, e) => a + (e.weight ?? 0), 0))
    return round2(ac.etfs.reduce((a, e) => a + (e.weight ?? 0), 0))
  }, [ac])

  const sumRisk = useMemo(() => {
    if (!isRisk) return 0
    return round2(ac.etfs.reduce((a, e) => a + (e.riskContribution ?? 0), 0))
  }, [ac, isRisk])

  return (
    <div className="rounded-xl border border-gray-200">
      <div className="flex items-center justify-between border-b bg-gray-50/80 px-3 py-2">
        <div className="flex items-center gap-2">
          {editing ? (
            <input
              value={tempName}
              onChange={(e) => setTempName(e.target.value)}
              onBlur={() => {
                on重命名(tempName.trim() || ac.name)
                setEditing(false)
              }}
              className="rounded-md border border-gray-300 px-2 py-1 text-sm"
            />
          ) : (
            <div className="text-sm font-medium">{ac.name}</div>
          )}
          <button className="text-xs text-blue-600 underline" onClick={() => setEditing((v) => !v)} title="重命名">
            {editing ? '保存' : '重命名'}
          </button>

          <div className="ml-3 flex items-center gap-2">
            <ModePill label="自定义权重" active={ac.mode === 'custom'} onClick={() => onModeChange('custom')} />
            <ModePill label="等权重" active={ac.mode === 'equal'} onClick={() => onModeChange('equal')} />
            <ModePill label="风险平价" active={ac.mode === 'risk'} onClick={() => onModeChange('risk')} />

            {isRisk && (
              <>
                <select
                  className="ml-2 rounded-md border border-gray-300 px-2 py-1 text-xs"
                  value={ac.riskMetric || 'vol'}
                  onChange={(e) => onRiskMetricChange(e.target.value as RiskMetric)}
                >
                  <option value="vol">波动率</option>
                  <option value="var">VaR</option>
                  <option value="es">ES</option>
                </select>

                <div className="ml-2 flex items-center gap-2 text-xs">
                  <span className="text-gray-600">最大杠杆</span>
                  <input
                    type="number"
                    min={0}
                    step={0.01}
                    value={ac.maxLeverage ?? 0}
                    onChange={(e) => onSetMaxLeverage(Number(e.target.value))}
                    className="w-20 rounded-md border border-gray-300 px-2 py-1 text-right"
                    title="允许的组合最大杠杆率，例如 0 表示不允许杠杆；2 表示最多 2x"
                  />
                </div>

                <button
                  className="ml-2 rounded-md bg-blue-600 px-3 py-1 text-xs text-white hover:bg-blue-700 disabled:opacity-60"
                  onClick={onSolve}
                  disabled={loading || sumRisk !== 100}
                  title={sumRisk !== 100 ? '风险贡献合计需等于 100% 才能计算' : ''}
                >
                  {loading ? '计算中...' : '反推资金权重'}
                </button>
              </>
            )}
          </div>
        </div>
        <button className="rounded-md border border-red-200 px-3 py-1 text-xs text-red-600 hover:bg-red-50" onClick={on删除}>
          删除这个大类
        </button>
      </div>

      <div className="grid grid-cols-12 items-center gap-2 px-3 py-2 text-xs text-gray-500">
        <div className="col-span-7">ETF</div>
        <div className="col-span-3 text-right">{isRisk ? (showSolved ? '风险贡献（%） / 资金权重（%）' : '风险贡献（%）') : '权重（%）'}</div>
        <div className="col-span-2 text-right">操作</div>
      </div>

      <div className="divide-y">
        {ac.etfs.map((e, idx) => (
          <div key={idx} className="grid grid-cols-12 items-center gap-2 px-3 py-2">
            <div className="col-span-7">
              <div className="flex items-center gap-2">
                <span className="w-5 text-xs text-gray-500">{idx + 1}.</span>
                <div className="min-w-0 flex-1 truncate text-sm">
                  <div className="flex items-center gap-2">
                    <span className="font-mono">{e.code}</span>
                    <span className="text-gray-700">{e.name}</span>
                  </div>
                  <div className="text-xs text-gray-500 mt-0.5">基金公司：{e.management || '—'} ｜ 成立日期：{e.found_date || '—'}</div>
                </div>
              </div>
            </div>

            <div className="col-span-3 text-right">
              {ac.mode === 'custom' ? (
                <div className="inline-flex items-center gap-2">
                  <input
                    type="number"
                    min={0}
                    max={100}
                    step={0.01}
                    value={e.weight ?? 0}
                    onChange={(ev) => onSetCustomWeight(idx, Number(ev.target.value))}
                    className="w-24 rounded-md border border-gray-300 px-2 py-1 text-right text-sm"
                  />
                  <span className="text-sm text-gray-500">%</span>
                </div>
              ) : ac.mode === 'equal' ? (
                <div className="pr-2 text-sm text-gray-700">{(equal[idx] ?? 0).toFixed(2)}%</div>
              ) : (
                <div className="inline-flex items-center gap-3 justify-end">
                  <input
                    type="number"
                    min={0}
                    max={100}
                    step={0.01}
                    value={e.riskContribution ?? 0}
                    onChange={(ev) => onSetRiskContribution(idx, Number(ev.target.value))}
                    className="w-24 rounded-md border border-gray-300 px-2 py-1 text-right text-sm"
                  />
                  <span className="text-sm text-gray-500">%</span>
                  {showSolved && <span className="text-xs text-gray-500">/ 资金权重 {(e.weight ?? 0).toFixed(2)}%</span>}
                </div>
              )}
            </div>

            <div className="col-span-2 text-right">
              <button onClick={() => onRemoveETF(idx)} className="rounded-md border border-gray-300 px-3 py-1 text-xs hover:bg-gray-50">
                删除
              </button>
            </div>
          </div>
        ))}

        <div className="px-3 py-2">
          <button onClick={onAddETF} className="w-full rounded-lg border border-dashed border-gray-300 py-2 text-sm hover:bg-gray-50">
            + 添加新的 ETF
          </button>
        </div>
      </div>

      <div className="border-t px-3 py-2 text-xs">
        {ac.mode === 'custom' ? (
          <>
            <span
              className={
                'rounded-md px-2 py-0.5 ' + (sumWeight === 100 ? 'bg-green-50 text-green-700' : 'bg-yellow-50 text-yellow-700')
              }
            >
              权重合计： {sumWeight.toFixed(2)}%
            </span>
            {sumWeight !== 100 && <span className="ml-2 text-yellow-700">（需等于 100%）</span>}
          </>
        ) : ac.mode === 'equal' ? (
          <span className="rounded-md bg-green-50 px-2 py-0.5 text-green-700">权重合计： 100.00%</span>
        ) : (
          <>
            <span
              className={
                'rounded-md px-2 py-0.5 ' + (sumRisk === 100 ? 'bg-green-50 text-green-700' : 'bg-yellow-50 text-yellow-700')
              }
            >
              风险贡献合计： {sumRisk.toFixed(2)}%
            </span>
            {sumRisk !== 100 && <span className="ml-2 text-yellow-700">（需等于 100%）</span>}
            {showSolved && (
              <span className="ml-3 rounded-md bg-blue-50 px-2 py-0.5 text-blue-700">资金权重合计： {sumWeight.toFixed(2)}%</span>
            )}
          </>
        )}
      </div>
    </div>
  )
}
