import { Link } from 'react-router-dom'
import { ArrowLongLeftIcon } from '@heroicons/react/24/outline'
import { businessStages, getStage } from '../app/processRegistry'
import { localizeStage } from '../i18n/navigation'
import { useI18n } from '../i18n/runtime'

export default function ProcessHome() {
  const { s } = useI18n()
  const settings = getStage('settings')
  const portfolioCenter = getStage('portfolio-center')
  const portfolioSolutions = getStage('portfolio-solutions')
  const accounting = getStage('fund-accounting')

  return (
    <div className="min-h-[calc(100vh-64px)] bg-slate-950 text-white">
      <section className="relative overflow-hidden border-b border-white/10">
        <div className="absolute -left-24 top-8 h-72 w-72 rounded-full bg-sky-500/20 blur-3xl" />
        <div className="absolute right-0 top-16 h-80 w-80 rounded-full bg-violet-500/20 blur-3xl" />
        <div className="relative mx-auto max-w-[1600px] px-4 py-12 sm:px-6 lg:px-8 lg:py-16">
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-sky-300">Fund investment research platform</p>
          <h1 className="mt-4 max-w-4xl text-3xl font-bold leading-tight sm:text-5xl">{s('home.title')}</h1>
          <p className="mt-4 max-w-5xl text-sm leading-7 text-slate-300 sm:text-base">{s('home.description')}</p>
        </div>
      </section>

      <main className="mx-auto max-w-[1600px] px-4 py-10 sm:px-6 lg:px-8">
        <section aria-labelledby="main-process-title">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">Single portfolio lifecycle</p>
              <h2 id="main-process-title" className="mt-1 text-2xl font-bold">{s('home.main')}</h2>
            </div>
            <p className="text-sm text-slate-400">{s('home.enterHint')}</p>
          </div>

          <ol className="mt-7 grid gap-3 xl:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr_auto_1fr] xl:items-stretch">
            {businessStages.map(localizeStage).map((stage, index) => (
              <li key={stage.id} className="contents">
                <Link to={stage.path} className="group flex min-h-64 flex-col rounded-2xl border border-white/10 bg-white/[0.06] p-5 transition hover:-translate-y-1 hover:border-white/25 hover:bg-white/[0.1] focus:outline-none focus:ring-2 focus:ring-sky-400">
                  <span className="text-3xl font-bold tabular-nums text-white/20">{String(stage.order).padStart(2, '0')}</span>
                  <p className="mt-6 text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">{stage.eyebrow}</p>
                  <h3 className="mt-2 text-xl font-bold text-white group-hover:text-sky-200">{stage.label}</h3>
                  <p className="mt-3 text-sm leading-6 text-slate-300">{stage.description}</p>
                  <span className="mt-auto pt-5 text-sm font-semibold text-sky-300">{s('home.enter', { stage: stage.label })} →</span>
                </Link>
                {index < businessStages.length - 1 ? (
                  <div className="hidden items-center text-2xl text-white/30 xl:flex" aria-hidden="true">→</div>
                ) : null}
              </li>
            ))}
          </ol>

          <div className="mt-4 flex items-center gap-3 px-3 text-rose-200" aria-label={s('home.feedback')}>
            <ArrowLongLeftIcon className="h-5 w-5 shrink-0" aria-hidden="true" />
            <div className="h-px flex-1 bg-rose-300/35" aria-hidden="true" />
            <span className="shrink-0 text-xs font-semibold tracking-wide">{s('home.feedback')}</span>
          </div>

          <div className="mt-5 rounded-2xl border border-rose-300/20 bg-rose-300/[0.06] px-5 py-4 text-sm leading-6 text-rose-100">
            <span className="font-semibold">{s('home.feedbackTitle')}</span>{s('home.feedbackHint')}
          </div>

          <Link to={portfolioCenter.path} className="group mt-5 block rounded-2xl border border-indigo-300/25 bg-gradient-to-r from-indigo-950/75 via-slate-900 to-blue-950/55 p-5 transition hover:border-indigo-300/55 focus:outline-none focus:ring-2 focus:ring-indigo-300 sm:p-6">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-indigo-300">Actual portfolio registry · 投中至投后</p>
                <h3 className="mt-2 text-2xl font-bold">{s('home.portfolioTitle')}</h3>
                <p className="mt-2 max-w-5xl text-sm leading-6 text-slate-300">{s('home.portfolioDescription')}</p>
              </div>
              <span className="shrink-0 text-sm font-semibold text-indigo-300 group-hover:text-indigo-200">{s('home.enter', { stage: s('navigation.routes.portfolio-center') })} →</span>
            </div>
            <p className="mt-5 border-t border-white/10 pt-4 text-xs leading-6 text-indigo-100/80">{s('home.portfolioScope')}</p>
          </Link>

          <Link to={accounting.path} className="group mt-5 block rounded-2xl border border-cyan-300/25 bg-gradient-to-r from-cyan-950/70 via-slate-900 to-emerald-950/50 p-5 transition hover:border-cyan-300/55 focus:outline-none focus:ring-2 focus:ring-cyan-300 sm:p-6">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-300">Parallel accounting lifecycle · 投中至投后</p>
                <h3 className="mt-2 text-2xl font-bold">{s('home.accountingTitle')}</h3>
                <p className="mt-2 max-w-5xl text-sm leading-6 text-slate-300">{s('home.accountingDescription')}</p>
              </div>
              <span className="shrink-0 text-sm font-semibold text-cyan-300 group-hover:text-cyan-200">{s('home.enterAccounting')} →</span>
            </div>
            <p className="mt-5 border-t border-white/10 pt-4 text-xs leading-6 text-cyan-100/80">{s('home.accountingScope')}</p>
          </Link>
        </section>

        <section className="mt-10 border-t border-white/10 pt-8" aria-labelledby="extended-capabilities-title">
          <div className="mb-5">
            <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">Beyond the lifecycle</p>
            <h2 id="extended-capabilities-title" className="mt-1 text-xl font-bold">{s('home.supportTitle')}</h2>
            <p className="mt-2 text-sm text-slate-400">{s('home.supportDescription')}</p>
          </div>
          <div className="space-y-4" data-testid="extended-capabilities-stack">
            <Link to={portfolioSolutions.path} className="group block rounded-2xl border border-fuchsia-300/20 bg-gradient-to-r from-fuchsia-950/60 to-slate-900 p-6 transition hover:border-fuchsia-300/50 focus:outline-none focus:ring-2 focus:ring-fuchsia-300">
              <div className="grid gap-5 lg:grid-cols-[1fr_auto] lg:items-center">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.22em] text-fuchsia-300">Portfolio solutions</p>
                  <h3 className="mt-2 text-2xl font-bold">{s('home.solutionsTitle')}</h3>
                  <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-300">{s('home.solutionsDescription')}</p>
                </div>
                <span className="text-sm font-semibold text-fuchsia-300 group-hover:text-fuchsia-200">{s('home.viewSolutions')} →</span>
              </div>
              <p className="mt-5 border-t border-white/10 pt-4 text-xs leading-6 text-fuchsia-100/80">{s('home.solutionsScope')}</p>
            </Link>
            <Link to={settings.path} className="group block rounded-2xl border border-white/10 bg-gradient-to-r from-slate-900 to-slate-800 p-6 transition hover:border-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-400">
              <div className="grid gap-5 lg:grid-cols-[1fr_auto] lg:items-center">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">Platform foundation</p>
                  <h3 className="mt-2 text-2xl font-bold">{s('home.settingsTitle')}</h3>
                  <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-300">{s('home.settingsDescription')}</p>
                </div>
                <span className="text-sm font-semibold text-sky-300 group-hover:text-sky-200">{s('home.enter', { stage: s('navigation.routes.settings') })} →</span>
              </div>
              <p className="mt-5 border-t border-white/10 pt-4 text-xs leading-6 text-slate-300">{s('home.settingsScope')}</p>
            </Link>
          </div>
        </section>
      </main>
    </div>
  )
}
