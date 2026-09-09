import { useEffect, useRef, useState } from 'react'
import { cancelStoragePlan, getDataStorage, planDataStorage, probeDataStorage, type StorageProbe, type StorageStatus } from '../../services/dataStorage'
import { buttonClass, inputClass, primaryClass } from './EditorFields'

export function storageSize(bytes: number | null | undefined) {
  if (bytes === null || bytes === undefined) return '未知'
  return `${(bytes / 1024 ** 3).toFixed(2)} GiB`
}
function Help({ text }: { text: string }) {
  return <span tabIndex={0} role="img" aria-label={text} title={text} className="ml-1 cursor-help rounded-full border px-1 text-xs font-normal text-slate-500">?</span>
}

export default function DataStoragePanel() {
  const [status, setStatus] = useState<StorageStatus | null>(null)
  const [path, setPath] = useState('')
  const [checked, setChecked] = useState<{ path: string; value: StorageProbe } | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [retry, setRetry] = useState(0)
  const [confirm, setConfirm] = useState(false)
  const requestId = useRef(0)
  const locked = useRef(false)
  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try {
        const value = await getDataStorage(controller.signal)
        if (!controller.signal.aborted) setStatus(value)
      } catch (reason) {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '存储状态暂不可用。')
      }
      if (!controller.signal.aborted) timer = setTimeout(poll, 10000)
    }
    void poll()
    return () => { controller.abort(); clearTimeout(timer); requestId.current += 1 }
  }, [retry])
  const changePath = (value: string) => { requestId.current += 1; setPath(value); setChecked(null); setConfirm(false); setNotice('') }
  const perform = async (operation: () => Promise<void>) => {
    if (locked.current) return
    locked.current = true; setBusy(true); setError(''); setNotice('')
    try { await operation() } catch (reason) { setError(reason instanceof Error ? reason.message : '存储操作失败。') }
    finally { locked.current = false; setBusy(false) }
  }
  const probe = () => void perform(async () => {
    const id = ++requestId.current
    setChecked(null)
    const result = await probeDataStorage(path)
    if (id === requestId.current) setChecked({ path, value: result })
  })
  const save = () => void perform(async () => {
    if (!checked || checked.path !== path || !confirm || status?.revision === undefined) return
    setStatus(await planDataStorage(checked.value.target, status.revision))
    setNotice('计划已保存，尚未迁移或释放空间。请先停止下载，然后在终端重启服务。')
    setChecked(null); setConfirm(false)
  })
  const cancel = () => void perform(async () => {
    if (status?.revision === undefined) return
    setStatus(await cancelStoragePlan(status.revision))
    setNotice('计划已取消，原数据与已有暂存文件均未删除。')
  })
  const plan = status?.pending
  const transferable = Boolean(status?.online && status.editing_enabled && !status.active && !plan)
  return <section aria-label="数据存储位置" className="space-y-3 rounded-xl border border-slate-200 bg-white p-4 sm:p-5">
    <div className="flex flex-wrap items-center justify-between gap-2"><h2 className="text-lg font-bold">数据存储位置<Help text="包括下载分片、原始响应、检查点、快照和研究数据。源码、依赖及服务日志不会迁移。" /></h2><button type="button" className={buttonClass} disabled={busy} onClick={() => { setError(''); setRetry(v => v + 1) }}>刷新存储状态</button></div>
    {!status ? <p role="status" className="text-sm text-slate-600">正在检查存储目录…</p> : <>
      <dl className="grid min-w-0 gap-3 text-sm sm:grid-cols-3"><div className="min-w-0 sm:col-span-2"><dt className="text-slate-500">实际数据目录</dt><dd className="mt-1 break-all font-medium">{status.actual_path || status.active?.target || status.logical_path}</dd></div><div><dt className="text-slate-500">磁盘剩余空间</dt><dd className="mt-1 font-medium">{storageSize(status.free_bytes)} / {storageSize(status.total_bytes)}</dd></div></dl>
      {!status.online ? <p role="alert" className="rounded-lg bg-rose-50 p-3 text-sm text-rose-800">{status.error || '数据磁盘不可用。'} 不会自动写回本机旧副本。</p> : status.free_bytes !== null && status.free_bytes < 5 * 1024 ** 3 ? <p className="rounded-lg bg-amber-50 p-3 text-sm text-amber-900">当前磁盘空间偏少。迁移后还需确认清理原本机副本，才能释放本机容量。</p> : null}
      {status.active ? <div className="space-y-2 text-sm"><p className="text-emerald-800">已启用指定目录；新的数据下载和处理均使用该数据区。</p><p className="text-xs text-slate-500">本期支持首次迁移，不支持再次跨盘切换，避免破坏历史任务中的物理路径引用。</p>{!status.active.backup_removed ? <details className="rounded-lg bg-amber-50 p-3"><summary className="cursor-pointer font-semibold">原本机副本仍保留：确认后清理释放空间</summary><p className="my-2 break-all">{status.active.backup}</p><p>确认数据可用后，先停止服务和下载，再执行以下命令。删除不可恢复，不影响指定磁盘上的活动数据。</p><code className="my-2 block break-all text-xs">./start_services.sh storage-cleanup {status.active.id}</code></details> : <p>原本机副本已清理。</p>}</div> : null}
    </>}
    {plan ? <div className="space-y-2 rounded-lg border border-indigo-200 bg-indigo-50 p-3 text-sm"><p className="font-semibold">待迁移目录：<span className="break-all">{plan.target}</span></p><p>{plan.message}</p>{plan.inventory ? <><progress aria-label="存储迁移进度" className="w-full" value={plan.bytes} max={Math.max(plan.inventory.bytes, 1)} /><p>{plan.files.toLocaleString()} / {plan.inventory.files.toLocaleString()} 个文件，{storageSize(plan.bytes)} / {storageSize(plan.inventory.bytes)}</p></> : null}<p>1. 在运行记录中停止下载；2. 在项目终端执行：</p><code className="block break-all">./start_services.sh restart</code><p className="text-xs">迁移期间后端离线，详细进度显示在终端。复制和校验可能耗时较长，请保持磁盘连接。不会自动重新下载。</p><button type="button" className={buttonClass} disabled={busy || ['SWITCHING', 'LINKED'].includes(plan.phase)} onClick={cancel}>取消迁移计划（不删除数据）</button></div> : null}
    {!status?.active && !plan ? <details className="rounded-lg border border-slate-200 p-3"><summary className="cursor-pointer text-sm font-semibold">设置外接磁盘或其他目录</summary><div className="mt-3 space-y-3">
      <label className="block text-sm font-semibold">已挂载磁盘<Help text="只列出后端所在电脑的挂载磁盘。选择后仍需检查目录；系统不会格式化磁盘。" /><select className={inputClass} aria-label="已挂载磁盘" disabled={!transferable || busy} value="" onChange={e => { if (e.target.value) changePath(`${e.target.value}/FundResearchData`) }}><option value="">选择磁盘以填入推荐路径</option>{status?.volumes.map(v => <option key={v.path} value={v.path}>{v.name} · 剩余 {storageSize(v.free_bytes)}</option>)}</select></label>
      <label className="block text-sm font-semibold">目标绝对目录<Help text="例如 /Volumes/研究磁盘/FundResearchData。必须是专用空目录，父目录需已存在。不是浏览器的下载文件夹。" /><input aria-label="目标绝对目录" className={inputClass} placeholder="/Volumes/你的磁盘/FundResearchData" value={path} disabled={!transferable} onChange={e => changePath(e.target.value)} /></label>
      <p className="text-xs leading-5 text-slate-500">整库迁移包括本地配置与凭据，目标文件系统必须支持私有权限、文件锁与原子替换。目录检查会写入少量临时探测文件并清除，不会移动现有数据。</p>
      <button type="button" className={buttonClass} disabled={!transferable || busy || !path.trim()} onClick={probe}>{busy ? '正在处理…' : '检查目录'}</button>
      {checked ? <div className="space-y-2 rounded-lg bg-slate-50 p-3 text-sm"><p>{checked.value.message}</p><p>剩余 {storageSize(checked.value.free_bytes)}，预留 {storageSize(checked.value.reserve_bytes)}。</p>{checked.value.same_device ? <p className="text-amber-900">目标与当前数据在同一文件系统；搬到这里通常不能解决本机容量不足。</p> : null}<label className="flex items-start gap-2"><input type="checkbox" aria-label="确认保存迁移计划" checked={confirm} onChange={e => setConfirm(e.target.checked)} /><span>我了解：保存只生成计划，重启前迁移；原本机副本不会自动删除。</span></label><button type="button" className={primaryClass} disabled={busy || !confirm || !transferable} onClick={save}>保存迁移计划</button></div> : null}
    </div></details> : null}
    {error ? <p role="alert" className="text-sm text-rose-700">{error}</p> : null}
    {notice ? <p role="status" className="text-sm text-emerald-800">{notice}</p> : null}
  </section>
}
