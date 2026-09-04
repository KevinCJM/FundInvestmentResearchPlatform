import { useEffect, useRef, useState } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { getStage, type StageId } from '../app/processRegistry'
import ActualPortfolioSelector from '../components/ActualPortfolioSelector'

export default function StageLayout({ stageId }: { stageId: StageId }) {
  const stage = getStage(stageId)
  const location = useLocation()
  const [isStageNavOpen, setIsStageNavOpen] = useState(false)
  const stageNavRef = useRef<HTMLDivElement>(null)
  const stageNavTriggerRef = useRef<HTMLButtonElement>(null)
  const previousPathRef = useRef(location.pathname)
  const navigationOptions = [
    { label: `${stage.label}总览`, path: stage.path },
    ...stage.nodes.map((node) => ({ label: node.label, path: node.path })),
    ...(stage.tools ?? []).map((tool) => ({ label: `工具 · ${tool.label}`, path: tool.path })),
  ]
  const selectedOption = navigationOptions.find((item) => location.pathname === item.path)
    ?? [...navigationOptions]
      .sort((left, right) => right.path.length - left.path.length)
      .find((item) => location.pathname.startsWith(`${item.path}/`))
    ?? navigationOptions[0]
  const isCrossPortfolioWorkspace = location.pathname === '/investment-execution/trade-allocation' || location.pathname === '/fund-accounting/account-statements'
  const isFullBleedWorkspace = location.pathname.startsWith('/settings/scenario-algorithms/workbench')

  useEffect(() => {
    if (previousPathRef.current !== location.pathname) {
      previousPathRef.current = location.pathname
      setIsStageNavOpen(false)
    }
  }, [location.pathname])

  useEffect(() => {
    if (!isStageNavOpen) return undefined

    const handlePointerDown = (event: PointerEvent) => {
      if (event.target instanceof Node && !stageNavRef.current?.contains(event.target)) {
        setIsStageNavOpen(false)
      }
    }
    document.addEventListener('pointerdown', handlePointerDown)
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown)
    }
  }, [isStageNavOpen])

  const stageContext = stage.id === 'portfolio-solutions' ? (
    <div className="rounded-xl border border-fuchsia-200 bg-fuchsia-50 px-4 py-3 text-sm">
      <span className="font-semibold text-fuchsia-900">当前展示版本：</span>
      <span className="text-fuchsia-700">稳健多资产方案 · V3.2</span>
    </div>
  ) : stage.id === 'pre-investment' ? (
    <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm">
      <span className="font-semibold text-slate-800">当前研究方案：</span>
      <span className="text-slate-500">稳健 FOF 配置研究 · RS-2026-018 · V4</span>
    </div>
  ) : stage.id === 'portfolio-center' ? (
    <div className="rounded-xl border border-indigo-200 bg-indigo-50 px-4 py-3 text-sm text-indigo-900">
      <span className="font-semibold">主数据范围：</span>真实组合、账户、核算主体与目标版本
    </div>
  ) : isCrossPortfolioWorkspace ? (
    <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950">
      <span className="font-semibold">当前业务范围：</span>跨组合账户与分配工作区
    </div>
  ) : ['investment-execution', 'fund-accounting', 'post-investment', 'feedback'].includes(stage.id) ? (
    <ActualPortfolioSelector compact />
  ) : null

  const stageNavigation = (
    <div
      ref={stageNavRef}
      onKeyDown={(event) => {
        if (event.key === 'Escape' && isStageNavOpen) {
          event.stopPropagation()
          setIsStageNavOpen(false)
          stageNavTriggerRef.current?.focus()
        }
      }}
      className="relative w-full sm:w-auto"
    >
      <button
        ref={stageNavTriggerRef}
        type="button"
        aria-expanded={isStageNavOpen}
        aria-controls="stage-subpage-navigation"
        aria-label={`${stage.label}阶段导航`}
        onClick={() => setIsStageNavOpen((current) => !current)}
        className="inline-flex min-h-11 w-full items-center gap-3 rounded-xl border border-slate-200 bg-white px-3 py-2 text-left shadow-sm transition hover:border-slate-300 hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 sm:w-auto sm:max-w-[280px]"
      >
        <span aria-hidden="true" className={`grid h-8 w-8 shrink-0 place-items-center rounded-lg ${stage.accent.soft} ${stage.accent.text}`}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className="h-4 w-4">
            <path strokeLinecap="round" d="M5 7h14M5 12h14M5 17h14" />
          </svg>
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">阶段导航</span>
          <span className="block truncate text-sm font-semibold text-slate-800">{selectedOption.label}</span>
        </span>
        <svg aria-hidden="true" viewBox="0 0 20 20" fill="currentColor" className={`ml-1 h-4 w-4 shrink-0 text-slate-400 transition-transform motion-reduce:transition-none ${isStageNavOpen ? 'rotate-180' : ''}`}>
          <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 0 1 1.06.02L10 11.168l3.71-3.938a.75.75 0 1 1 1.08 1.04l-4.25 4.5a.75.75 0 0 1-1.08 0l-4.25-4.5a.75.75 0 0 1 .02-1.06Z" clipRule="evenodd" />
        </svg>
      </button>

      {isStageNavOpen ? (
        <aside id="stage-subpage-navigation" className="absolute left-0 top-full z-40 mt-2 max-h-[min(70vh,720px)] w-full min-w-0 overflow-y-auto rounded-2xl border border-slate-200 bg-white p-3 shadow-xl sm:left-auto sm:right-0 sm:w-[340px] sm:max-w-[calc(100vw-3rem)]" aria-label={`${stage.label}子页面导航`}>
          <Link to={stage.path} onClick={() => setIsStageNavOpen(false)} className="block rounded-xl px-3 py-3 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-indigo-500">
            <span className="text-sm font-semibold text-slate-900">{stage.label}总览</span>
            <span className="mt-1 block text-xs text-slate-500">查看本阶段全部流程节点</span>
          </Link>
          <div className="my-2 border-t border-slate-100" />
          <nav className="grid gap-1">
            {stage.nodes.map((node, index) => (
              <NavLink
                key={node.id}
                to={node.path}
                onClick={() => setIsStageNavOpen(false)}
                className={({ isActive }) => `rounded-xl px-3 py-2.5 text-sm transition focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                  isActive ? `${stage.accent.soft} font-semibold ${stage.accent.text}` : 'text-slate-600 hover:bg-slate-50 hover:text-slate-950'
                }`}
              >
                <span className="mr-2 text-xs tabular-nums text-slate-400">{String(index + 1).padStart(2, '0')}</span>
                {node.label}
              </NavLink>
            ))}
          </nav>
          {stage.tools?.length ? (
            <div className="mt-4 border-t border-slate-100 pt-3">
              <p className="px-3 text-xs font-semibold uppercase tracking-wide text-slate-400">现有工具</p>
              <nav className="mt-2 grid gap-1">
                {stage.tools.map((tool) => (
                  <NavLink
                    key={tool.path}
                    to={tool.path}
                    onClick={() => setIsStageNavOpen(false)}
                    className={({ isActive }) => `rounded-xl px-3 py-2.5 text-sm transition focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                      isActive ? `${stage.accent.soft} font-semibold ${stage.accent.text}` : 'text-slate-600 hover:bg-slate-50 hover:text-slate-950'
                    }`}
                  >
                    {tool.label}
                  </NavLink>
                ))}
              </nav>
            </div>
          ) : null}
        </aside>
      ) : null}
    </div>
  )

  return (
    <div className="min-h-[calc(100vh-64px)] bg-slate-50">
      {!isFullBleedWorkspace ? <section className="relative z-30 border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-[1600px] px-4 py-5 sm:px-6 lg:px-8">
          <nav aria-label="面包屑" className="flex items-center gap-2 text-xs text-slate-500">
            <Link to="/" className="rounded hover:text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500">流程首页</Link>
            <span aria-hidden="true">/</span>
            <span className="font-medium text-slate-800">{stage.label}</span>
          </nav>
          <div className="mt-3 flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
            <div className="min-w-0">
              <p className={`text-xs font-semibold uppercase tracking-[0.22em] ${stage.accent.text}`}>{stage.eyebrow}</p>
              <h1 className="mt-1 text-2xl font-bold text-slate-950 sm:text-3xl">{stage.label}</h1>
              <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-600">{stage.description}</p>
            </div>
            <div className="flex w-full flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center xl:w-auto xl:max-w-[62%] xl:justify-end">
              {stageContext}
              {stageNavigation}
            </div>
          </div>
        </div>
      </section> : null}

      <div className={isFullBleedWorkspace ? 'w-full' : 'mx-auto max-w-[1600px] px-4 py-6 sm:px-6 lg:px-8'}>
        <section className="min-w-0" aria-label={`${stage.label}工作区`}>
          <Outlet />
        </section>
      </div>
    </div>
  )
}
