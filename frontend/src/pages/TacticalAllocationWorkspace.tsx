import { useEffect, useMemo, useRef, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { Link } from 'react-router-dom'
import {
  getHistoricalRegimeRun,
  listHistoricalRegimeRuns,
  type HistoricalRegimeRun,
} from '../services/historicalRegimes'
import {
  backtestHistoricalRegimeTaa,
  type TaaBacktestRequest,
  type TaaBacktestResult,
} from '../services/taaBacktest'

const SAMPLE_BASE = JSON.stringify({ '沪深300': 0.6, '中债综合': 0.4 }, null, 2)
const SAMPLE_RETURNS = `date,period_start,沪深300,中债综合
2024-01-02,2024-01-01,0.0042,0.0008
2024-01-03,2024-01-02,-0.0061,0.0011
2024-01-04,2024-01-03,0.0028,-0.0003
2024-01-05,2024-01-04,0.0074,-0.0012
2024-01-08,2024-01-05,-0.0035,0.0006
2024-01-09,2024-01-08,0.0051,-0.0004
2024-01-10,2024-01-09,-0.0082,0.0014
2024-01-11,2024-01-10,0.0039,0.0002
2024-01-12,2024-01-11,0.0063,-0.0007
2024-01-15,2024-01-12,-0.0027,0.0009`

function isEligible(run: HistoricalRegimeRun) {
  const published = Boolean(run.content_hash) && (run.publications ?? []).some((item) => (
    (item.usage === 'taa' || item.usage === 'formal_backtest')
    && item.run_id === run.id
    && item.definition_revision === run.definition_revision
    && item.run_content_hash === run.content_hash
    && item.gate === 'causality_passed'
  ))
  return run.immutable === true
    && run.mode === 'realtime'
    && run.causality?.is_causal === true
    && run.causality?.uses_future_data === false
    && run.causality?.repaints !== true
    && run.causality?.realtime_eligible === true
    && published
}

function numericRecord(text: string, label: string) {
  let value: unknown
  try {
    value = JSON.parse(text)
  } catch {
    throw new Error(`${label}不是有效 JSON。`)
  }
  if (!value || Array.isArray(value) || typeof value !== 'object') throw new Error(`${label}必须是 JSON 对象。`)
  const result: Record<string, number> = {}
  Object.entries(value as Record<string, unknown>).forEach(([key, item]) => {
    const number = Number(item)
    if (!key.trim() || !Number.isFinite(number)) throw new Error(`${label}中的 ${key || '空键'} 必须是有限数值。`)
    result[key] = number
  })
  return result
}

function parseStateTilts(text: string) {
  let value: unknown
  try {
    value = JSON.parse(text)
  } catch {
    throw new Error('状态偏移不是有效 JSON。')
  }
  if (!value || Array.isArray(value) || typeof value !== 'object') throw new Error('状态偏移必须是两层 JSON 对象。')
  return Object.fromEntries(Object.entries(value as Record<string, unknown>).map(([state, row]) => {
    if (!row || Array.isArray(row) || typeof row !== 'object') throw new Error(`状态 ${state} 的偏移必须是 JSON 对象。`)
    return [state, numericRecord(JSON.stringify(row), `状态 ${state} 的偏移`)]
  }))
}

function parseReturns(text: string): Array<Record<string, string | number>> {
  const trimmed = text.trim()
  if (!trimmed) throw new Error('请提供资产收益数据。')
  if (trimmed.startsWith('[')) {
    let rows: unknown
    try {
      rows = JSON.parse(trimmed)
    } catch {
      throw new Error('资产收益不是有效 JSON。')
    }
    if (!Array.isArray(rows) || !rows.length) throw new Error('资产收益 JSON 必须是非空数组。')
    return rows as Array<Record<string, string | number>>
  }
  const lines = trimmed.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
  if (lines.length < 2) throw new Error('CSV 至少需要表头和一行收益。')
  const headers = lines[0].split(',').map((item) => item.trim())
  if (headers[0] !== 'date' || headers.length < 2 || new Set(headers).size !== headers.length) {
    throw new Error('CSV 第一列必须是 date，且资产列名不能重复。')
  }
  const assetHeaders = headers.filter((header) => header !== 'date' && header !== 'period_start')
  if (!assetHeaders.length) throw new Error('CSV 至少需要一个资产收益列。')
  return lines.slice(1).map((line, rowIndex) => {
    const cells = line.split(',').map((item) => item.trim())
    if (cells.length !== headers.length) throw new Error(`CSV 第 ${rowIndex + 2} 行列数不一致。`)
    const row: Record<string, string | number> = { date: cells[0] }
    const periodStartIndex = headers.indexOf('period_start')
    if (periodStartIndex >= 0 && cells[periodStartIndex]) row.period_start = cells[periodStartIndex]
    assetHeaders.forEach((asset) => {
      const value = Number(cells[headers.indexOf(asset)])
      if (!Number.isFinite(value)) throw new Error(`CSV 第 ${rowIndex + 2} 行的 ${asset} 不是有效收益率。`)
      row[asset] = value
    })
    return row
  })
}

function emptyTilts(run: HistoricalRegimeRun, assets: string[]) {
  return Object.fromEntries(run.states.map((state) => [
    state.id,
    Object.fromEntries(assets.map((asset) => [asset, 0])),
  ]))
}

function pct(value: number | null | undefined, digits = 2) {
  return value == null || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(digits)}%`
}

function number(value: number | null | undefined, digits = 3) {
  return value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits)
}

function Metric({ label, value, detail, tone = 'slate' }: { label: string; value: string; detail?: string; tone?: 'slate' | 'emerald' | 'rose' }) {
  const color = tone === 'emerald' ? 'text-emerald-700' : tone === 'rose' ? 'text-rose-700' : 'text-slate-950'
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="text-xs font-medium text-slate-500">{label}</div>
      <div className={`mt-1 text-2xl font-semibold tabular-nums ${color}`}>{value}</div>
      {detail && <div className="mt-1 text-xs text-slate-500">{detail}</div>}
    </div>
  )
}

export default function TacticalAllocationWorkspace() {
  const [runs, setRuns] = useState<HistoricalRegimeRun[]>([])
  const [selectedRunId, setSelectedRunId] = useState('')
  const [selectedRunDetail, setSelectedRunDetail] = useState<HistoricalRegimeRun | null>(null)
  const [loadingSelectedRun, setLoadingSelectedRun] = useState(false)
  const [baseText, setBaseText] = useState(SAMPLE_BASE)
  const [tiltsText, setTiltsText] = useState('{}')
  const [returnsText, setReturnsText] = useState(SAMPLE_RETURNS)
  const [transactionCost, setTransactionCost] = useState(5)
  const [confidenceFloor, setConfidenceFloor] = useState(0.55)
  const [maxAbsTilt, setMaxAbsTilt] = useState(0.2)
  const [periodsPerYear, setPeriodsPerYear] = useState(252)
  const [maxSignalAgeDays, setMaxSignalAgeDays] = useState(31)
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<TaaBacktestResult | null>(null)
  const [resultTab, setResultTab] = useState<'performance' | 'weights' | 'audit'>('performance')
  const requestVersionRef = useRef(0)

  const invalidateResult = () => {
    requestVersionRef.current += 1
    setResult(null)
  }

  const eligibleRuns = useMemo(() => runs.filter(isEligible), [runs])
  const selectedRunSummary = eligibleRuns.find((run) => run.id === selectedRunId) ?? null
  const selectedRun = selectedRunDetail?.id === selectedRunId ? selectedRunDetail : null

  useEffect(() => {
    let active = true
    void listHistoricalRegimeRuns()
      .then((items) => {
        if (!active) return
        setRuns(items)
        const eligible = items.filter(isEligible)
        setSelectedRunId((current) => current || eligible[0]?.id || '')
      })
      .catch((failure) => { if (active) setError(failure instanceof Error ? failure.message : '历史情景版本加载失败。') })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [])

  useEffect(() => {
    if (!selectedRunSummary) { setSelectedRunDetail(null); setLoadingSelectedRun(false); return undefined }
    if (Array.isArray(selectedRunSummary.states) && Array.isArray(selectedRunSummary.publications)) {
      setSelectedRunDetail(selectedRunSummary); setLoadingSelectedRun(false); return undefined
    }
    let active = true
    setSelectedRunDetail(null); setLoadingSelectedRun(true)
    void getHistoricalRegimeRun(selectedRunSummary.id)
      .then((detail) => { if (active) setSelectedRunDetail(detail) })
      .catch((failure) => { if (active) setError(failure instanceof Error ? failure.message : '所选历史情景详情加载失败。') })
      .finally(() => { if (active) setLoadingSelectedRun(false) })
    return () => { active = false }
  }, [selectedRunSummary])

  useEffect(() => {
    if (!selectedRun) return
    try {
      const assets = Object.keys(numericRecord(baseText, '基础权重'))
      setTiltsText(JSON.stringify(emptyTilts(selectedRun, assets), null, 2))
    } catch {
      setTiltsText(JSON.stringify(emptyTilts(selectedRun, ['沪深300', '中债综合']), null, 2))
    }
    invalidateResult()
  }, [selectedRun?.id]) // switching the immutable regime version invalidates the prior result

  const runBacktest = async () => {
    if (!selectedRun) return
    const requestRunId = selectedRun.id
    const requestVersion = requestVersionRef.current + 1
    requestVersionRef.current = requestVersion
    try {
      setRunning(true)
      setError(null)
      setResult(null)
      const baseWeights = numericRecord(baseText, '基础权重')
      const payload: TaaBacktestRequest = {
        asset_returns: parseReturns(returnsText),
        base_weights: baseWeights,
        state_tilts: parseStateTilts(tiltsText),
        limits: { min_weight: 0, max_weight: 1, max_abs_tilt: maxAbsTilt },
        transaction_cost_bps: transactionCost,
        confidence_floor: confidenceFloor,
        periods_per_year: periodsPerYear,
        max_signal_age_days: maxSignalAgeDays,
      }
      const response = await backtestHistoricalRegimeTaa(requestRunId, payload)
      if (requestVersion === requestVersionRef.current && response.run_id === requestRunId) {
        setResult(response)
        setResultTab('performance')
      }
    } catch (failure) {
      if (requestVersion === requestVersionRef.current) {
        setError(failure instanceof Error ? failure.message : 'TAA 回测失败。')
      }
    } finally {
      setRunning(false)
    }
  }

  const rebuildTilts = () => {
    if (!selectedRun) return
    try {
      const assets = Object.keys(numericRecord(baseText, '基础权重'))
      setTiltsText(JSON.stringify(emptyTilts(selectedRun, assets), null, 2))
      setError(null)
      invalidateResult()
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : '无法读取基础权重。')
    }
  }

  const exportResult = () => {
    if (!result) return
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `taa-backtest-${result.run_id}.json`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  const chartOption = useMemo(() => {
    if (!result) return null
    return {
      animationDuration: 350,
      tooltip: { trigger: 'axis', valueFormatter: (value: number) => value.toFixed(4) },
      legend: { top: 0, data: ['SAA 基准', '情景驱动 TAA'] },
      grid: { left: 52, right: 20, top: 48, bottom: 44 },
      xAxis: { type: 'category', boundaryGap: false, data: result.taa.nav.map((point) => point.date), axisLabel: { hideOverlap: true } },
      yAxis: { type: 'value', scale: true, axisLabel: { formatter: (value: number) => value.toFixed(2) } },
      series: [
        { name: 'SAA 基准', type: 'line', showSymbol: false, data: result.baseline.nav.map((point) => point.value), lineStyle: { color: '#64748b', width: 2 } },
        { name: '情景驱动 TAA', type: 'line', showSymbol: false, data: result.taa.nav.map((point) => point.value), lineStyle: { color: '#0f766e', width: 2.5 }, areaStyle: { color: 'rgba(13,148,136,.08)' } },
      ],
    }
  }, [result])

  return (
    <div className="min-h-screen bg-slate-50 pb-16 text-slate-800">
      <section className="border-b border-slate-800 bg-slate-950 px-5 py-8 text-white sm:px-8">
        <div className="mx-auto max-w-[1500px]">
          <div className="text-xs font-semibold uppercase tracking-[0.22em] text-teal-300">Pre-investment · Tactical allocation</div>
          <div className="mt-3 flex flex-col justify-between gap-4 lg:flex-row lg:items-end">
            <div>
              <h1 className="text-3xl font-semibold">战术资产配置工作台</h1>
              <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">把已发布的历史情景概率映射为相对 SAA 的权重偏移，严格使用上一可用信号，计入换手成本，并与静态基准逐期比较。</p>
            </div>
            <div className="flex flex-wrap gap-2 text-xs">
              {['不可变版本', '实时因果', '概率加权', 'NJIT 数值核心'].map((item) => <span key={item} className="rounded-full border border-teal-700 bg-teal-950/60 px-3 py-1.5 text-teal-200">✓ {item}</span>)}
            </div>
          </div>
        </div>
      </section>

      <main className="mx-auto max-w-[1500px] space-y-5 px-4 py-6 sm:px-8">
        {error && <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">{error}</div>}

        <section className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_340px]">
          <div className="space-y-5">
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="text-xs font-semibold text-teal-700">01 · 选择可信情景版本</div>
                  <h2 className="mt-1 text-lg font-semibold text-slate-950">历史情景信号源</h2>
                </div>
                <Link to="/settings/scenario-algorithms" className="text-sm font-medium text-teal-700 hover:text-teal-900">去情景算法中心 →</Link>
              </div>
              {loading ? <p className="mt-4 text-sm text-slate-500">正在核验情景版本…</p> : eligibleRuns.length ? (
                <div className="mt-4 grid gap-4 md:grid-cols-[minmax(0,1fr)_220px]">
                  <label className="text-sm font-medium text-slate-700">已发布的实时因果版本
                    <select aria-label="已发布的实时因果版本" value={selectedRunId} disabled={running} onChange={(event) => setSelectedRunId(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm disabled:bg-slate-100">
                      {eligibleRuns.map((run) => <option key={run.id} value={run.id}>{run.name} · R{run.definition_revision ?? '—'} · {run.id}</option>)}
                    </select>
                  </label>
                  <div className="rounded-lg bg-emerald-50 px-3 py-2.5 text-xs leading-5 text-emerald-800">
                    <div className="font-semibold">{loadingSelectedRun ? '正在按需读取详情' : '门禁通过'}</div>
                    <div>immutable · realtime · causal</div>
                    <div>{selectedRun?.publications.filter((item) => item.usage === 'taa' || item.usage === 'formal_backtest').map((item) => item.usage).join(' / ')}</div>
                  </div>
                </div>
              ) : (
                <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
                  暂无可用版本。请先在情景算法中心保存并运行 realtime 因果模型，再发布到“TAA”或“正式回测”。回溯式、重绘或未发布结果不会出现在这里。
                </div>
              )}
            </div>

            <div className="grid gap-5 lg:grid-cols-2">
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="text-xs font-semibold text-teal-700">02 · 定义政策组合</div>
                <h2 className="mt-1 text-lg font-semibold text-slate-950">SAA 基准权重</h2>
                <p className="mt-1 text-xs leading-5 text-slate-500">JSON 键即资产列名，权重和必须为 1。回测不自动补齐缺失资产。</p>
                <textarea aria-label="SAA 基准权重" value={baseText} disabled={running} onChange={(event) => { setBaseText(event.target.value); invalidateResult() }} rows={8} spellCheck={false} className="mt-3 w-full rounded-lg border border-slate-300 bg-slate-950 p-3 font-mono text-xs leading-5 text-slate-100 disabled:opacity-60" />
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="text-xs font-semibold text-teal-700">03 · 状态到权重</div>
                <div className="mt-1 flex items-center justify-between gap-3"><h2 className="text-lg font-semibold text-slate-950">各情景相对偏移</h2><button type="button" onClick={rebuildTilts} disabled={!selectedRun} className="text-xs font-semibold text-teal-700 hover:text-teal-900 disabled:text-slate-400">按基准重建零偏移</button></div>
                <p className="mt-1 text-xs leading-5 text-slate-500">每个状态必须覆盖全部资产，单个状态内偏移和为 0。平台不会根据“牛/熊”等标签猜测资产方向，请由研究员明确填写。</p>
                <textarea aria-label="状态权重偏移" value={tiltsText} disabled={running} onChange={(event) => { setTiltsText(event.target.value); invalidateResult() }} rows={8} spellCheck={false} className="mt-3 w-full rounded-lg border border-slate-300 bg-slate-950 p-3 font-mono text-xs leading-5 text-slate-100 disabled:opacity-60" />
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="text-xs font-semibold text-teal-700">04 · 收益与执行假设</div>
              <div className="mt-1 flex flex-col justify-between gap-3 md:flex-row md:items-end">
                <div>
                  <h2 className="text-lg font-semibold text-slate-950">资产逐期收益</h2>
                  <p className="mt-1 text-xs text-slate-500">支持 CSV 或 JSON 数组；日期相同的情景信号不会用于当期，杜绝同日偷看。</p>
                </div>
                <div className="grid grid-cols-2 gap-2 lg:grid-cols-5">
                  <label className="text-xs text-slate-600">成本（bp）<input aria-label="单边交易成本" disabled={running} type="number" min="0" max="1000" step="1" value={transactionCost} onChange={(event) => { setTransactionCost(Number(event.target.value)); invalidateResult() }} className="mt-1 w-full rounded-lg border border-slate-300 px-2 py-2" /></label>
                  <label className="text-xs text-slate-600">置信度门槛<input aria-label="置信度门槛" disabled={running} type="number" min="0" max="1" step="0.05" value={confidenceFloor} onChange={(event) => { setConfidenceFloor(Number(event.target.value)); invalidateResult() }} className="mt-1 w-full rounded-lg border border-slate-300 px-2 py-2" /></label>
                  <label className="text-xs text-slate-600">最大偏移<input aria-label="最大绝对偏移" disabled={running} type="number" min="0" max="1" step="0.05" value={maxAbsTilt} onChange={(event) => { setMaxAbsTilt(Number(event.target.value)); invalidateResult() }} className="mt-1 w-full rounded-lg border border-slate-300 px-2 py-2" /></label>
                  <label className="text-xs text-slate-600">年化期数<input aria-label="年化期数" disabled={running} type="number" min="1" max="3660" step="1" value={periodsPerYear} onChange={(event) => { setPeriodsPerYear(Number(event.target.value)); invalidateResult() }} className="mt-1 w-full rounded-lg border border-slate-300 px-2 py-2" /></label>
                  <label className="text-xs text-slate-600">信号最长天数<input aria-label="信号最长有效天数" disabled={running} type="number" min="1" max="3650" step="1" value={maxSignalAgeDays} onChange={(event) => { setMaxSignalAgeDays(Number(event.target.value)); invalidateResult() }} className="mt-1 w-full rounded-lg border border-slate-300 px-2 py-2" /></label>
                </div>
              </div>
              <textarea aria-label="资产收益数据" value={returnsText} disabled={running} onChange={(event) => { setReturnsText(event.target.value); invalidateResult() }} rows={11} spellCheck={false} className="mt-4 w-full rounded-lg border border-slate-300 bg-slate-50 p-3 font-mono text-xs leading-5 text-slate-800 disabled:opacity-60" />
              <p className="mt-2 text-xs text-amber-700">当前预填内容仅用于说明 CSV 格式，是合成收益样例；正式研究请替换为版本化资产收益数据。</p>
              <div className="mt-4 flex items-center justify-between gap-4">
                <p className="text-xs text-slate-500">低于置信度门槛或尚无生效信号时，明确回退到 SAA，不会臆造状态。</p>
                <button type="button" disabled={!selectedRun || running} onClick={() => void runBacktest()} className="shrink-0 rounded-lg bg-teal-700 px-5 py-2.5 text-sm font-semibold text-white hover:bg-teal-800 disabled:cursor-not-allowed disabled:bg-slate-300">{running ? '计算中…' : '运行 TAA 回测'}</button>
              </div>
            </div>
          </div>

          <aside className="space-y-4">
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h2 className="font-semibold text-slate-950">信号时序</h2>
              <div className="mt-4 space-y-4 text-sm">
                {[['t 日收盘', '情景指标形成观察值'], ['t + 滞后', '状态被确认并生效'], ['下一收益期', '用概率加权偏移调仓'], ['期末', '扣成本并归因']].map(([time, action], index) => (
                  <div key={time} className="flex gap-3"><span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-teal-100 text-xs font-semibold text-teal-800">{index + 1}</span><div><div className="font-medium text-slate-800">{time}</div><div className="text-xs text-slate-500">{action}</div></div></div>
                ))}
              </div>
            </div>
            <div className="rounded-2xl border border-slate-200 bg-slate-900 p-5 text-slate-200 shadow-sm">
              <h2 className="font-semibold text-white">研究边界</h2>
              <ul className="mt-3 space-y-2 text-xs leading-5 text-slate-300">
                <li>• 这是概率驱动的研究回测，不生成交易指令。</li>
                <li>• SAA 与 TAA 都按同一成本模型再平衡，避免比较口径偏置。</li>
                <li>• 缺失收益、未知资产和不完整状态映射直接阻断。</li>
                <li>• 每次结果保留输入哈希、版本哈希和发布门禁。</li>
              </ul>
            </div>
          </aside>
        </section>

        {result && (
          <section className="rounded-2xl border border-slate-200 bg-white shadow-sm">
            <div className="border-b border-slate-200 px-5 pt-5">
              <div className="flex flex-col justify-between gap-3 md:flex-row md:items-center">
                <div>
                  <div className="text-xs font-semibold text-emerald-700">完整结果快照 · {result.snapshot_hash.slice(0, 12)}</div>
                  <h2 className="mt-1 text-xl font-semibold text-slate-950">SAA 与情景驱动 TAA 对照</h2>
                </div>
                <div className="flex items-center gap-3"><div className="text-xs text-slate-500">{result.input_snapshot.start_date} — {result.input_snapshot.end_date} · {result.input_snapshot.observations} 期</div><button type="button" onClick={exportResult} className="rounded-lg border border-slate-300 px-3 py-2 text-xs font-semibold text-slate-700 hover:border-teal-500 hover:text-teal-800">导出完整 JSON</button></div>
              </div>
              <div role="tablist" className="mt-5 flex gap-1">
                {[['performance', '绩效与净值'], ['weights', '权重与归因'], ['audit', '时序与审计']].map(([id, label]) => <button key={id} role="tab" aria-selected={resultTab === id} onClick={() => setResultTab(id as typeof resultTab)} className={`rounded-t-lg px-4 py-2 text-sm font-medium ${resultTab === id ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-100'}`}>{label}</button>)}
              </div>
            </div>

            {resultTab === 'performance' && <div className="space-y-5 p-5">
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
                <Metric label="TAA 累计收益" value={pct(result.taa.metrics.total_return)} />
                <Metric label="SAA 累计收益" value={pct(result.baseline.metrics.total_return)} />
                <Metric label="累计收益差" value={pct(result.excess.total_return_difference)} tone={result.excess.total_return_difference >= 0 ? 'emerald' : 'rose'} />
                <Metric label="TAA 最大回撤" value={pct(result.taa.metrics.max_drawdown)} />
                <Metric label="TAA 夏普" value={number(result.taa.metrics.sharpe)} detail={`无风险利率 0；年化 ${result.metrics_policy?.annualization_periods ?? periodsPerYear} 期`} />
              </div>
              {chartOption && <ReactECharts option={chartOption} style={{ height: 360 }} />}
              <div className="grid gap-3 md:grid-cols-3">
                <Metric label="累计换手" value={pct(result.turnover_and_cost.total_turnover)} />
                <Metric label="平均单期换手" value={pct(result.turnover_and_cost.average_turnover)} />
                <Metric label="交易成本金额" value={number(result.turnover_and_cost.total_transaction_cost, 6)} detail="按期初净值计" />
              </div>
            </div>}

            {resultTab === 'weights' && <div className="grid gap-5 p-5 xl:grid-cols-[minmax(0,1fr)_380px]">
              <div className="overflow-x-auto">
                <table className="w-full min-w-[760px] text-left text-xs">
                  <thead className="bg-slate-50 text-slate-500"><tr><th className="px-3 py-2">收益日</th><th className="px-3 py-2">生效信号</th>{result.input_snapshot.assets.map((asset) => <th key={asset} className="px-3 py-2">{asset}</th>)}<th className="px-3 py-2">换手</th><th className="px-3 py-2">TAA 收益</th><th className="px-3 py-2">状态</th></tr></thead>
                  <tbody className="divide-y divide-slate-100">{result.weights.slice(-30).map((row) => <tr key={row.date}><td className="px-3 py-2 tabular-nums">{row.date}</td><td className="px-3 py-2 tabular-nums text-slate-500">{row.regime_effective_date ?? '—'}</td>{result.input_snapshot.assets.map((asset) => <td key={asset} className="px-3 py-2 tabular-nums">{pct(row.weights[asset])}</td>)}<td className="px-3 py-2 tabular-nums">{pct(row.turnover)}</td><td className="px-3 py-2 tabular-nums">{pct(row.net_taa_return)}</td><td className="px-3 py-2">{row.fallback_to_base ? <span className="text-amber-700">回退 SAA</span> : <span className="text-emerald-700">概率偏移</span>}</td></tr>)}</tbody>
                </table>
              </div>
              <div>
                <h3 className="font-semibold text-slate-950">状态超额贡献</h3>
                <div className="mt-3 space-y-3">{result.state_contributions.map((item) => <div key={item.state_id} className="rounded-lg border border-slate-200 p-3"><div className="flex justify-between text-sm"><span className="font-medium">{selectedRun?.states.find((state) => state.id === item.state_id)?.label ?? item.state_id}</span><span className={item.gross_excess_return_contribution >= 0 ? 'text-emerald-700' : 'text-rose-700'}>{pct(item.gross_excess_return_contribution)}</span></div><div className="mt-1 text-xs text-slate-500">概率权重累计 {number(item.probability_weight, 2)} · 活跃 {item.active_periods} 期</div></div>)}</div>
                {result.fallbacks.periods > 0 && <div className="mt-4 rounded-lg bg-amber-50 p-3 text-xs text-amber-900">{result.fallbacks.periods} 期回退 SAA：{Object.entries(result.fallbacks.reasons).map(([reason, count]) => `${reason} ${count}`).join('；')}</div>}
              </div>
            </div>}

            {resultTab === 'audit' && <div className="grid gap-4 p-5 md:grid-cols-2 xl:grid-cols-4">
              <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-900"><div className="font-semibold">发布与因果门禁</div><div className="mt-2 text-xs leading-5">通过：{String(result.gate.passed)}<br />模式：{result.gate.mode}<br />因果：{String(result.gate.causal)}<br />用途：{result.gate.publication_usages.join(' / ')}</div></div>
              <div className="rounded-xl border border-slate-200 p-4 text-sm"><div className="font-semibold text-slate-950">严格时序</div><div className="mt-2 break-words font-mono text-xs leading-5 text-slate-600">{result.timing_policy.signal_rule}<br />same_day = {String(result.timing_policy.same_day_signal_allowed)}</div></div>
              <div className="rounded-xl border border-slate-200 p-4 text-sm"><div className="font-semibold text-slate-950">运行版本</div><div className="mt-2 break-all font-mono text-xs leading-5 text-slate-600">{result.run_id}<br />R{result.definition_revision ?? '—'}<br />{result.gate.run_content_hash}</div></div>
              <div className="rounded-xl border border-slate-200 p-4 text-sm"><div className="font-semibold text-slate-950">输入与执行证据链</div><div className="mt-2 break-all font-mono text-xs leading-5 text-slate-600">returns {result.input_snapshot.asset_returns_hash}<br />params {result.input_snapshot.parameters_hash}<br />engine {result.execution?.engine ?? '—'} / {result.execution?.backend ?? '—'}<br />nopython {String(result.execution?.nopython ?? false)} · fallback {result.execution?.python_fallback ?? '—'}</div></div>
            </div>}
          </section>
        )}
      </main>
    </div>
  )
}
