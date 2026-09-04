export default function StaticDemoBanner() {
  return (
    <div className="rounded-xl border border-sky-300 bg-sky-50 px-4 py-3 text-sm text-sky-950" role="note">
      <span className="font-bold">静态功能演示｜未接入真实数据与后端服务</span>
      <span className="ml-2 text-sky-800">本页操作只在当前会话展示，刷新后恢复预置示例。</span>
    </div>
  )
}
