import type { DataRefreshStatus } from './types'

export default function RefreshRecoveryActions({ status, locked, rebuilding, recoverableCandidate, canResume, onResume, onRebuild }: {
  status: DataRefreshStatus | null
  locked: boolean
  rebuilding: boolean
  recoverableCandidate: boolean
  canResume: boolean
  onResume: () => void
  onRebuild: () => void
}) {
  if (status?.job.analytics_snapshot?.status === 'failed') {
    return <section aria-label="恢复数据处理" className="rounded-xl border border-amber-200 bg-amber-50 p-5 text-amber-950">
      <h2 className="font-semibold">数据已拉取，无需重新下载</h2>
      <p className="mt-2 text-sm leading-6">分析数据整理未完成。{recoverableCandidate ? '重建并验收已保留的候选快照' : '重建当前分析快照'}即可继续，不会再次请求 Tushare。</p>
      {status.job.analytics_snapshot.message ? <p className="mt-2 text-xs">{status.job.analytics_snapshot.message}</p> : null}
      <button type="button" onClick={onRebuild} disabled={locked || !status.enabled} className="mt-3 min-h-11 rounded-xl bg-amber-900 px-4 text-sm font-semibold text-white disabled:opacity-50">
        {rebuilding ? '正在重建...' : recoverableCandidate ? '重建并接入候选快照' : '重建分析快照'}
      </button>
    </section>
  }
  if (!canResume || !status) return null
  return <section aria-label="恢复上次下载" className="rounded-xl border border-amber-200 bg-amber-50 p-5 text-amber-950">
    <h2 className="font-semibold">上次下载可以继续</h2>
    <p className="mt-2 text-sm leading-6">保留上次选择的内容与更新方式，无需重新勾选。系统按检查点或增量规则恢复，不会盲目从头下载。</p>
    <button type="button" onClick={onResume} disabled={locked || !status.enabled || !status.token_configured || (status.job.mode === 'full' && !status.full_refresh_enabled)} className="mt-3 min-h-11 rounded-xl bg-amber-900 px-4 text-sm font-semibold text-white disabled:opacity-50">按原配置继续上次更新</button>
    {!status.token_configured ? <p className="mt-2 text-xs">先在下方补充连接凭据，再继续下载。</p> : null}
  </section>
}
