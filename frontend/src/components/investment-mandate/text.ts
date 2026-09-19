import { useI18n } from '../../i18n/runtime'

export function useMandateText() {
  const { s, locale } = useI18n()
  return { t: (key: string, values?: Record<string, string | number>) => s(`mandate.${key}`, values), locale }
}
