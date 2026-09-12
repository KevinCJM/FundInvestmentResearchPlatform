import { useEffect, useRef, useState } from 'react'
import type { ConfigRecord, InterfaceConfig, SourceConfig, SourceSyncJob } from '../../services/dataSources'
import { listSourceSyncJobs, startSourceSync } from '../../services/dataSources'
import { buttonClass, inputClass, JsonField, primaryClass, validateEditor } from './EditorFields'

export default function SourceSyncPanel({ record, source, disabled }: {
  record: ConfigRecord<InterfaceConfig>; source: SourceConfig; disabled: boolean
}) {
  const [open, setOpen] = useState(false)
  const [mode, setMode] = useState<'full' | 'incremental'>('incremental')
  const [symbol, setSymbol] = useState(String(record.config.params.symbol ?? record.config.params.ts_code ?? ''))
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [extra, setExtra] = useState<Record<string, unknown>>({})
  const [job, setJob] = useState<SourceSyncJob | null>(null)
  const [error, setError] = useState('')
  const [ready, setReady] = useState(false)
  const [busy, setBusy] = useState(false)
  const [otherJobRunning, setOtherJobRunning] = useState(false)
  const container = useRef<HTMLDetailsElement>(null)
  const [retry, setRetry] = useState(0)
  const submitting = useRef(false)
  useEffect(() => {
    if (!open) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    const poll = async () => {
      try {
        const jobs = await listSourceSyncJobs()
        if (cancelled) return
        setJob(jobs.find(item => item.interface_id === record.config.id) ?? null)
        setOtherJobRunning(jobs.some(item => item.status === 'RUNNING' && item.interface_id !== record.config.id))
        setReady(true); setError('')
        if (jobs.some(item => item.status === 'RUNNING')) timer = setTimeout(poll, 2000)
      } catch (reason) {
        if (!cancelled) { setReady(false); setError(reason instanceof Error ? reason.message : '无法读取任务状态。') }
      }
    }
    void poll()
    return () => { cancelled = true; clearTimeout(timer) }
  }, [open, record.config.id, retry])
  const codeParam = source.transport === 'akshare' || 'symbol' in record.config.params ? 'symbol' : 'ts_code'
  const hasCode = source.transport !== 'http' || codeParam in record.config.params
  const run = async () => {
    if (submitting.current || disabled || !ready || otherJobRunning || job?.status === 'RUNNING') return
    if (!validateEditor(container.current?.closest('form') ?? null)) { setError('请先修正无效的参数输入。'); return }
    if (start && end && start > end) { setError('开始日期不能晚于结束日期。'); return }
    if (source.transport === 'akshare' && !/^\d{6}$/.test(symbol)) { setError('请输入六位产品代码，保留前导零。'); return }
    if (!window.confirm(`${source.name} / ${record.config.name}\n${mode === 'full' ? '重新获取所选范围' : '补充更新所选范围'}\n代码：${symbol || '使用已保存参数'}\n日期：${start || '接口默认起点'} 至 ${end || '接口默认终点'}\n将访问外部接口；结果保存为候选，暂不覆盖正式研究数据。继续？`)) return
    submitting.current = true; setBusy(true); setError('')
    try {
      const params = { ...extra, ...(hasCode && symbol ? { [codeParam]: symbol } : {}),
        ...(start ? { [record.config.start_param]: start.replace(/-/g, '') } : {}),
        ...(end ? { [record.config.end_param]: end.replace(/-/g, '') } : {}) }
      const next = await startSourceSync(record.config.id, record.revision, params, mode)
      setJob(next); setRetry(value => value + 1)
    } catch (reason) { setError(reason instanceof Error ? reason.message : '无法启动下载。') }
    finally { submitting.current = false; setBusy(false) }
  }
  return <details ref={container} className="rounded-xl border border-accent-200 p-4" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary className="cursor-pointer text-sm font-bold text-accent-900">5. 下载与更新此接口</summary>
    <p className="mt-3 text-xs leading-6 text-slate-600">按此接口的已保存映射、分页和限流下载。增量模式复用同配置同产品的断点，并回查最近 3 天；不代表全市场下载。{source.transport === 'akshare' ? '基金净值接口可能先取得单基金历史，再在本地按日期截取。' : ''}</p>
    <fieldset disabled={disabled || busy || job?.status === 'RUNNING'} className="mt-4 grid gap-3 sm:grid-cols-2">
      {hasCode ? <label className="text-xs font-semibold">产品代码<input className={inputClass} value={symbol} onChange={e => setSymbol(e.target.value)} placeholder={source.transport === 'akshare' ? '510300 / 000001' : '510300.SH'} /></label> : null}
      <label className="text-xs font-semibold">此次更新方式<select className={inputClass} value={mode} onChange={e => setMode(e.target.value as typeof mode)}><option value="incremental">增量补充</option><option value="full">重新下载所选范围</option></select></label>
      <label className="text-xs font-semibold">下载开始日期<input className={inputClass} type="date" value={start} onChange={e => setStart(e.target.value)} /></label>
      <label className="text-xs font-semibold">下载结束日期<input className={inputClass} type="date" value={end} onChange={e => setEnd(e.target.value)} /></label>
    </fieldset>
    <details className="mt-3"><summary className="cursor-pointer text-xs">高级：其他本次请求参数</summary><JsonField label="本次下载参数" value={extra} disabled={disabled || busy} onChange={value => setExtra(value as Record<string, unknown>)} /></details>
    {error ? <p role="alert" className="mt-3 text-sm text-rose-700">{error}<button type="button" className={`${buttonClass} ml-2`} onClick={() => setRetry(value => value + 1)}>重新读取状态</button></p> : null}
    {disabled ? <p className="mt-3 text-xs text-amber-800">先保存修改并启用数据源和接口，再执行下载。</p> : null}
    {otherJobRunning ? <p className="mt-3 text-xs text-amber-800">另一个接口正在下载，请等待后台任务结束。</p> : null}
    <button type="button" className={`${primaryClass} mt-4`} disabled={disabled || busy || !ready || otherJobRunning || job?.status === 'RUNNING'} onClick={() => void run()}>{job?.status === 'RUNNING' ? '此接口正在下载' : busy ? '正在启动…' : '开始此接口下载'}</button>
    {job ? <section aria-label="接口下载结果" className="mt-4 space-y-2 rounded-lg bg-slate-50 p-3 text-sm">
      <p>{job.status === 'RUNNING' ? '后台下载中' : job.status === 'FAILED' ? '需要处理' : job.status === 'EMPTY' ? '未取得数据，请核对范围' : '下载结束'} · {job.rows} 行 · {job.pages} 个下载分页</p>
      <p className={job.error ? 'text-rose-700' : 'text-slate-600'}>{job.error || job.message}</p>
      {job.resolutions?.map(result => <p key={result.table_id} className="text-xs">{result.table_id}：{result.status === 'CANDIDATE_READY' ? '多源取值完成 · 未发布' : '多源取值待检查'}{result.summary ? ` · 选中 ${result.summary.selected_rows} 行` : ''}{result.message ? ` · ${result.message}` : ''}</p>)}
    </section> : null}
  </details>
}
