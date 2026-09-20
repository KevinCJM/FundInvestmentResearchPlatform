import { useI18n } from '../../i18n/runtime'
export { control, linkClass, useLtcmaTask as useResearchTask } from '../ltcma/shared'
export function useImplementationText() {
  const {s} = useI18n()
  return (key: string, values?: Record<string, string | number>) => s(`implementation.${key}`, values)
}
export const percent = (value: number | null | undefined) => value == null ? '—' : `${(value*100).toFixed(2)}%`
export const amount = (value: number | null | undefined, digits=2) => value == null || !Number.isFinite(value) ? '—' : (Math.abs(value)<.5*10**-digits?0:value).toLocaleString(undefined, {maximumFractionDigits:digits})
