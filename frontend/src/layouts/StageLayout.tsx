import { useEffect, useRef, useState } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { type StageId } from '../app/processRegistry'
import { useLocalizedStage } from '../i18n/navigation'
import { useI18n } from '../i18n/runtime'
import ActualPortfolioSelector from '../components/ActualPortfolioSelector'
import FactorEvidencePanel from '../components/FactorEvidencePanel'
import type { ContextType } from '../services/factorResearch'
import { allocationJourneyPath, useAllocationJourney, type AllocationJourneyStep } from '../app/allocationJourney'

export default function StageLayout({ stageId }: { stageId: StageId }) {
  const { s } = useI18n()
  const stage = useLocalizedStage(stageId)
  const location = useLocation()
  const [journey] = useAllocationJourney()
  const isAllocationJourney = stageId === 'pre-investment' || location.pathname === '/product-research/pools'
  const journeySteps: { key: AllocationJourneyStep; label: string; ready: boolean; done: boolean; current: boolean }[] = [
    { key: 'pool', label: '产品范围', ready: true, done: Boolean(journey.universeId), current: /product-pool|\/pools$/.test(location.pathname) },
    { key: 'classes', label: '构建大类', ready: Boolean(journey.universeId), done: Boolean(journey.universeId && journey.allocationName), current: /asset-classes|auto-classification/.test(location.pathname) },
    { key: 'saa', label: '长期配置 SAA', ready: Boolean(journey.universeId && journey.allocationName), done: Boolean(journey.universeId && journey.allocationName && journey.baselineId), current: location.pathname.endsWith('/allocation-lab') },
    { key: 'taa', label: '战术研究 TAA', ready: Boolean(journey.universeId && journey.allocationName && journey.baselineId), done: Boolean(journey.universeId && journey.baselineId && journey.taaRunId), current: location.pathname.includes('/taa') },
    { key: 'products', label: '产品配置', ready: Boolean(journey.universeId && journey.taaRunId), done: false, current: location.pathname.includes('/product-allocation-timing') },
  ]
  const [isStageNavOpen, setIsStageNavOpen] = useState(false)
  const stageNavRef = useRef<HTMLElement>(null)
  const stageNavTriggerRef = useRef<HTMLButtonElement>(null)
  const previousPathRef = useRef(location.pathname)
  const navigationOptions = [
    { label: s('navigation.overview', { stage: stage.label }), path: stage.path },
    ...stage.nodes.map((node) => ({ label: node.label, path: node.path })),
    ...(stage.tools ?? []).map((tool) => ({ label: s('navigation.tool', { name: tool.label }), path: tool.path })),
  ]
  const selectedOption = navigationOptions.find((item) => location.pathname === item.path)
    ?? [...navigationOptions]
      .sort((left, right) => right.path.length - left.path.length)
      .find((item) => location.pathname.startsWith(`${item.path}/`))
    ?? navigationOptions[0]
  const isCrossPortfolioWorkspace = location.pathname === '/investment-execution/trade-allocation' || location.pathname === '/fund-accounting/account-statements'
  const isFullBleedWorkspace = location.pathname.startsWith('/settings/scenario-algorithms/workbench')
  const isStageOverview = location.pathname.replace(/\/$/, '') === stage.path
  const factorContext: ContextType | undefined = stage.id === 'product-research'
    ? (location.pathname.startsWith('/product-research/products/') ? undefined : 'product_research')
    : stage.id === 'pre-investment' ? (location.pathname.includes('/taa') ? 'taa' : location.pathname.includes('/saa') ? 'saa' : 'allocation')
    : stage.id === 'portfolio-center' ? 'portfolio'
    : stage.id === 'post-investment' ? 'post_investment'
    : location.pathname === '/settings/scenario-algorithms' ? 'regime' : undefined

  useEffect(() => {
    if (previousPathRef.current !== location.pathname) {
      previousPathRef.current = location.pathname
      setIsStageNavOpen(false)
    }
  }, [location.pathname])

  useEffect(() => {
    if (!isStageNavOpen) return undefined

    const handlePointerDown = (event: PointerEvent) => {
      if (event.target instanceof Node && !stageNavRef.current?.contains(event.target) && !stageNavTriggerRef.current?.contains(event.target)) {
        setIsStageNavOpen(false)
      }
    }
    document.addEventListener('pointerdown', handlePointerDown)
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown)
    }
  }, [isStageNavOpen])

  const stageContext = stage.id === 'portfolio-solutions' ? (
    <div className="rounded-xl border border-accent-200 bg-accent-50 px-4 py-3 text-sm">
      <span className="font-semibold text-accent-900">当前展示版本：</span>
      <span className="text-accent-700">稳健多资产方案 · V3.2</span>
    </div>
  ) : stage.id === 'pre-investment' ? (
    <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm">
      <span className="font-semibold text-slate-800">当前研究方案：</span>
      <span className="break-words text-slate-700">{journey.name || (journey.universeId ? '产品范围已载入（名称待确认）' : '尚未选择产品范围')}</span>
      {journey.researchDate && <span className="mt-1 block text-xs text-slate-600">范围研究日 {journey.researchDate}</span>}
    </div>
  ) : stage.id === 'portfolio-center' ? (
    <div className="rounded-xl border border-accent-200 bg-accent-50 px-4 py-3 text-sm text-accent-900">
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
      onKeyDown={(event) => {
        if (event.key === 'Escape' && isStageNavOpen) {
          event.stopPropagation()
          setIsStageNavOpen(false)
          stageNavTriggerRef.current?.focus()
        }
      }}
      className="relative w-full sm:w-auto xl:hidden"
    >
      <button
        ref={stageNavTriggerRef}
        type="button"
        aria-expanded={isStageNavOpen}
        aria-controls="stage-subpage-navigation"
        aria-label={s('navigation.stageAria', { stage: stage.label })}
        onClick={() => setIsStageNavOpen((current) => !current)}
        className="inline-flex min-h-11 w-full items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-1 text-left shadow-sm transition hover:border-slate-300 hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 sm:w-auto sm:max-w-[280px]"
      >
        <span aria-hidden="true" className={`grid h-6 w-6 shrink-0 place-items-center rounded-lg ${stage.accent.soft} ${stage.accent.text}`}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className="h-4 w-4">
            <path strokeLinecap="round" d="M5 7h14M5 12h14M5 17h14" />
          </svg>
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-xs font-semibold uppercase tracking-[0.14em] text-slate-600">{s('navigation.stage')}</span>
          <span className="block truncate text-sm font-semibold text-slate-800">{selectedOption.label}</span>
        </span>
        <svg aria-hidden="true" viewBox="0 0 20 20" fill="currentColor" className={`ml-1 h-4 w-4 shrink-0 text-slate-600 transition-transform motion-reduce:transition-none ${isStageNavOpen ? 'rotate-180' : ''}`}>
          <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 0 1 1.06.02L10 11.168l3.71-3.938a.75.75 0 1 1 1.08 1.04l-4.25 4.5a.75.75 0 0 1-1.08 0l-4.25-4.5a.75.75 0 0 1 .02-1.06Z" clipRule="evenodd" />
        </svg>
      </button>

    </div>
  )

  const stageNavigationPanel = (
        <aside ref={stageNavRef} id="stage-subpage-navigation" className={`${isStageNavOpen ? 'block' : 'hidden'} absolute inset-x-4 top-0 z-40 max-h-[70vh] overflow-y-auto rounded-xl border border-slate-200 bg-white p-3 shadow-xl sm:left-auto sm:right-6 sm:w-[340px] xl:sticky xl:inset-auto xl:top-4 xl:block xl:max-h-[calc(100dvh-96px)] xl:w-auto xl:self-start xl:shadow-none`} aria-label={s('navigation.subpages', { stage: stage.label })} onKeyDown={(event) => {
          if (event.key === 'Escape' && isStageNavOpen) {
            event.stopPropagation()
            setIsStageNavOpen(false)
            stageNavTriggerRef.current?.focus()
          }
        }}>
          <Link to={stage.path} onClick={() => setIsStageNavOpen(false)} className="block rounded-xl px-3 py-3 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-accent-500">
            <span className="text-sm font-semibold text-slate-900">{s('navigation.overview', { stage: stage.label })}</span>
            <span className="mt-1 block text-xs text-slate-600">{s('navigation.viewAll')}</span>
          </Link>
          <div className="my-2 border-t border-slate-100" />
          <nav className="grid gap-1">
            {stage.nodes.map((node, index) => (
              <NavLink
                key={node.id}
                to={node.path}
                onClick={() => setIsStageNavOpen(false)}
                className={({ isActive }) => `rounded-xl px-3 py-2.5 text-sm transition focus:outline-none focus:ring-2 focus:ring-accent-500 ${
                  isActive ? `${stage.accent.soft} font-semibold ${stage.accent.text}` : 'text-slate-600 hover:bg-slate-50 hover:text-slate-950'
                }`}
              >
                <span className="mr-2 text-xs tabular-nums text-slate-600">{String(index + 1).padStart(2, '0')}</span>
                {node.label}
              </NavLink>
            ))}
          </nav>
          {stage.tools?.length ? (
            <div className="mt-4 border-t border-slate-100 pt-3">
              <p className="px-3 text-xs font-semibold uppercase tracking-wide text-slate-600">{s('navigation.tools')}</p>
              <nav className="mt-2 grid gap-1">
                {stage.tools.map((tool) => (
                  <NavLink
                    key={tool.path}
                    to={tool.path}
                    onClick={() => setIsStageNavOpen(false)}
                    className={({ isActive }) => `rounded-xl px-3 py-2.5 text-sm transition focus:outline-none focus:ring-2 focus:ring-accent-500 ${
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
  )

  return (
    <div className="min-h-[calc(100vh-64px)] bg-slate-50">
      {!isFullBleedWorkspace ? <section className="relative z-30 border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-[1600px] px-4 py-3 sm:px-6 lg:px-8">
          <nav aria-label={s('navigation.breadcrumb')} className="flex items-center gap-2 text-xs text-slate-600">
            <Link to="/" className="rounded-lg hover:text-slate-900 focus:outline-none focus:ring-2 focus:ring-accent-500">{s('navigation.workflowHome')}</Link>
            <span aria-hidden="true">/</span>
            <span className="font-medium text-slate-800">{stage.label}</span>
          </nav>
          <div className={`flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between ${isStageOverview ? 'mt-3' : 'mt-2'}`}>
            {isStageOverview && <div className="min-w-0">
              <p className={`text-xs font-semibold uppercase tracking-[0.22em] ${stage.accent.text}`}>{stage.eyebrow}</p>
              <h1 className="mt-1 text-2xl font-bold text-slate-950 sm:text-3xl">{stage.label}</h1>
              {!isAllocationJourney && <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-600">{stage.description}</p>}
            </div>}
            <div className="flex w-full min-w-0 flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center xl:w-auto">
              {stageContext}
              {stageNavigation}
            </div>
          </div>
        </div>
      </section> : null}

      {isAllocationJourney && <nav aria-label="配置研究流程" className="border-b border-slate-200 bg-white px-4 py-3 sm:px-6">
        <ol className="mx-auto grid max-w-[1536px] grid-cols-5 gap-1 sm:gap-2">{journeySteps.map((step, index) => <li key={step.key} className="min-w-0">
          {step.ready || step.current ? <Link to={allocationJourneyPath(step.key, journey)} aria-current={step.current ? 'step' : undefined} aria-label={`${index + 1}. ${step.label}`} className={`block rounded-lg border px-1 py-2 text-xs sm:px-3 sm:text-sm ${step.current ? 'border-accent-500 bg-accent-50 font-semibold text-accent-900' : 'border-slate-200 text-slate-700 hover:bg-slate-50'}`}>
            <span className="sm:hidden">{index + 1}. {{ pool: '范围', classes: '大类', saa: 'SAA', taa: 'TAA', products: '产品' }[step.key]}</span><span className="hidden sm:inline">{index + 1}. {step.label}</span><span className="mt-0.5 hidden text-xs font-normal sm:block text-slate-600">{step.current ? '当前步骤' : step.done ? '已保存，可返回' : step.key === 'products' ? '从 TAA 保存页交接' : '继续研究'}</span>
          </Link> : <span className="block rounded-lg border border-dashed border-slate-200 px-1 py-2 text-xs text-slate-600 sm:px-3 sm:text-sm" title="先完成上一步"><span className="sm:hidden">{index + 1}. {{ pool: '范围', classes: '大类', saa: 'SAA', taa: 'TAA', products: '产品' }[step.key]}</span><span className="hidden sm:inline">{index + 1}. {step.label}</span><span className="mt-0.5 hidden text-xs sm:block">先完成上一步</span></span>}
        </li>)}</ol>
      </nav>}
      <div className={isFullBleedWorkspace ? 'w-full' : 'relative mx-auto grid max-w-[1600px] gap-5 px-4 py-4 sm:px-6 lg:px-8 xl:grid-cols-[220px_minmax(0,1fr)]'}>
        {!isFullBleedWorkspace && stageNavigationPanel}
        <section className="min-w-0" aria-label={s('navigation.workspace', { stage: stage.label })}>
          {factorContext && <details key={location.pathname} className="mb-3 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm"><summary className="min-h-6 cursor-pointer text-slate-600">参考：因子证据</summary><FactorEvidencePanel contextType={factorContext} /></details>}
          <Outlet />
        </section>
      </div>
    </div>
  )
}
