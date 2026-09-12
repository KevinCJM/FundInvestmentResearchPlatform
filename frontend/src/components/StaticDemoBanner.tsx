export default function StaticDemoBanner({ compact = false }: { compact?: boolean }) {
  if (compact) return (
    <details className="rounded-lg border border-accent-200 bg-accent-50 px-3 py-2 text-xs leading-5 text-accent-950">
      <summary className="cursor-pointer font-semibold">静态功能演示｜未接入真实数据与后端服务</summary>
      <p className="mt-1 text-accent-800">本页操作只在当前会话展示，刷新后恢复预置示例。</p>
    </details>
  )
  return (
    <div className="rounded-xl border border-accent-300 bg-accent-50 px-4 py-3 text-sm text-accent-950" role="note">
      <span className="font-bold">静态功能演示｜未接入真实数据与后端服务</span>
      <span className="ml-2 text-accent-800">本页操作只在当前会话展示，刷新后恢复预置示例。</span>
    </div>
  )
}
