import { useEffect, useState } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import { definitionForRequest, resolveRegimeAuthoring, type RegimeAuthoringResolution, type RegimeGraphDefinition, type RegimeMode } from '../../services/regimeGraph'

function MathNotation({ latex, label }: { latex: string; label: string }) {
  try {
    const markup = katex.renderToString(latex, { displayMode: true, throwOnError: true, trust: false, strict: 'ignore', maxExpand: 1000 })
    return <div aria-label={label} tabIndex={0} className="min-w-0 overflow-x-auto py-2 text-slate-900 focus:outline-none focus:ring-2 focus:ring-violet-300" dangerouslySetInnerHTML={{ __html: markup }} />
  } catch {
    return <p role="status" className="text-sm text-amber-700">这条公式暂时无法排版，请检查公式定义。</p>
  }
}

export function RegimeMathDisplay({ resolution, outputId = 'state', message }: {
  resolution: RegimeAuthoringResolution | null; outputId?: string; message?: string
}) {
  const latex = resolution?.display_latex?.[outputId]
  const steps = resolution?.formula_steps?.[outputId] || []
  return <section aria-label="情景数学公式预览" className="mt-4 min-w-0 rounded-xl border border-violet-100 bg-violet-50/40 p-4">
    <h3 className="text-sm font-semibold text-slate-800">数学公式</h3>
    {latex && !message && <p className="mt-1 text-xs text-slate-500 sm:hidden">较长公式可左右滑动查看。</p>}
    {message || !latex ? <p role="status" className="mt-2 text-sm text-slate-500">{message || resolution?.math_error || '连接输出并完成公式检查后，将显示数学公式。'}</p> : <>
      <MathNotation latex={latex} label="当前输出的数学公式" />
      {steps.length > 0 && <details className="mt-3 rounded-lg border border-slate-200 bg-white p-3"><summary className="cursor-pointer text-sm font-semibold text-slate-700">查看分步公式与参数</summary>
        <p className="mt-2 text-xs leading-5 text-slate-500">每个 z 对应一个步骤的输出序列，θ 表示该步骤的参数。具名算子的含义见各步说明。</p>
        <ol className="mt-3 space-y-3">{steps.map((step, index) => <li key={`${step.node_id}:${step.port}`} className="min-w-0 rounded-lg bg-slate-50 p-3">
          <p className="text-sm font-semibold text-slate-800">{index + 1}. {step.label}</p>
          <MathNotation latex={step.latex} label={`第 ${index + 1} 步的数学公式`} />
          {step.description && <p className="text-xs leading-5 text-slate-600">{step.description}</p>}
          {step.parameters.length > 0 && <dl className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600">{step.parameters.map((parameter, i) => <div key={i} className="flex min-w-0 gap-1"><dt>{parameter.label}：</dt><dd className="break-all">{parameter.value}</dd></div>)}</dl>}
        </li>)}</ol>
      </details>}
    </>}
  </section>
}

export default function RegimeMathPreview({ definition, mode, outputId }: { definition: RegimeGraphDefinition; mode: RegimeMode; outputId: string }) {
  const key = JSON.stringify([definitionForRequest(definition), mode])
  const [snapshot, setSnapshot] = useState<{ key: string; resolution: RegimeAuthoringResolution | null; error?: string } | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      void resolveRegimeAuthoring(definition, mode, 'graph', '', controller.signal)
        .then(resolution => { if (!controller.signal.aborted) setSnapshot({ key, resolution }) })
        .catch(() => { if (!controller.signal.aborted) setSnapshot({ key, resolution: null, error: '数学公式载入失败，请稍后重试。' }) })
    }, 200)
    return () => { window.clearTimeout(timer); controller.abort() }
  }, [key])
  const current = snapshot?.key === key ? snapshot : null
  return <RegimeMathDisplay resolution={current?.resolution || null} outputId={outputId} message={!current ? '正在生成数学公式…' : current.error} />
}
