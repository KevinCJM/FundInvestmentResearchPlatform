import {useId,useState,type ReactNode} from 'react'
import {CheckIcon,ClipboardDocumentIcon} from '@heroicons/react/24/outline'
import {IconButton,ActionFeedback,type Feedback} from '../ActionControls'
import {useI18n} from '../../i18n/runtime'
import type {SeriesOutputDefinition,SeriesParameterDefinition} from '../../services/customIndicators'

export default function DefinitionDetails({ definition, children }: { definition: Record<string, unknown>; children?: ReactNode }) {
  const { s } = useI18n()
  const [section, setSection] = useState<'formula' | 'parameters' | null>(null)
  const sectionId = useId()
  const [originalLines, setOriginalLines] = useState(false)
  const [copyFeedback, setCopyFeedback] = useState<Feedback>()
  const series = definition.result_kind === 'time_series' && Array.isArray(definition.series_outputs)
    ? definition.series_outputs as SeriesOutputDefinition[] : null
  const outputs = series || [{ id: 'scalar', label: '', expression: String(definition.expression || '') }]
  const parameters = Array.isArray(definition.parameter_schema) ? definition.parameter_schema as SeriesParameterDefinition[] : []
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(outputs.map(output => series ? `${output.label || output.id} = ${output.expression}` : output.expression).join('\n\n'))
      setCopyFeedback({ text: s('agent.formulaCopied') })
    } catch { setCopyFeedback({ text: s('agent.formulaCopyFailed'), error: true }) }
  }
  return <div className="mt-2">
    <div className="flex flex-wrap items-center gap-1">
      <button type="button" aria-label={s('agent.fullFormula', { count: outputs.length })} aria-expanded={section === 'formula'} aria-controls={`${sectionId}-formula`} onClick={() => setSection(value => value === 'formula' ? null : 'formula')} className="min-h-10 rounded-lg px-2 text-xs font-semibold text-accent-700 hover:bg-accent-50">{s('agent.formulaShort')}<span aria-hidden="true"> {section === 'formula' ? '▴' : '▾'}</span></button>
      {parameters.length > 0 && <button type="button" aria-label={s('agent.parameters', { count: parameters.length })} aria-expanded={section === 'parameters'} aria-controls={`${sectionId}-parameters`} onClick={() => setSection(value => value === 'parameters' ? null : 'parameters')} className="min-h-10 rounded-lg px-2 text-xs font-semibold text-slate-600 hover:bg-slate-100">{s('agent.parametersShort')}<span aria-hidden="true"> {section === 'parameters' ? '▴' : '▾'}</span></button>}
      <div className="ml-auto flex flex-wrap gap-1">{children}</div>
    </div>
    <div id={`${sectionId}-formula`} hidden={section !== 'formula'}>
    <div className="flex flex-wrap items-center justify-between gap-2">
      <label className="flex min-h-10 items-center gap-2 text-xs text-slate-600"><input type="checkbox" checked={originalLines} onChange={event => setOriginalLines(event.target.checked)} />{s('agent.originalLines')}</label>
      <IconButton label={s('agent.copyFormula')} hint={copyFeedback && !copyFeedback.error ? s('agent.formulaCopied') : s('agent.copyFormula')} onClick={() => void copy()} className={copyFeedback && !copyFeedback.error ? 'agent-copy-confirmed' : ''}>{copyFeedback && !copyFeedback.error ? <CheckIcon className="h-5 w-5" aria-hidden="true" /> : <ClipboardDocumentIcon className="h-5 w-5" aria-hidden="true" />}</IconButton>
    </div>
    {outputs.map(output => <pre key={output.id} tabIndex={0} aria-label={s('agent.formulaLabel', { name: output.label || output.id })} className={`mb-3 max-w-full overflow-x-auto rounded-lg bg-slate-950 p-3 text-xs leading-6 text-slate-200 focus-visible:ring-2 focus-visible:ring-accent-500 ${originalLines ? 'whitespace-pre' : 'whitespace-pre-wrap break-words [overflow-wrap:anywhere]'}`}>{output.expression ? (series ? `${output.label || output.id} = ${output.expression}` : output.expression) : s('agent.formulaMissing')}</pre>)}
    <ActionFeedback value={copyFeedback} />
    </div>
    <dl id={`${sectionId}-parameters`} hidden={section !== 'parameters'} className="space-y-2 pt-2 text-xs text-slate-600">{parameters.map(parameter => <div key={parameter.id}><dt className="font-semibold">{parameter.label || parameter.id}</dt><dd className="tabular-nums">{s('agent.parameterDetails', { defaultValue: parameter.default, minComparison: s(parameter.exclusive_minimum ? 'agent.greaterThan' : 'agent.atLeast'), minimum: parameter.minimum, maxComparison: s(parameter.exclusive_maximum ? 'agent.lessThan' : 'agent.atMost'), maximum: parameter.maximum, step: parameter.step })}</dd></div>)}</dl>
  </div>
}
