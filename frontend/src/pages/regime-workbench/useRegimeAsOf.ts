import { useState } from 'react'
import { useResearchContextIdentity, useResearchDay } from '../../app/ResearchContext'

/** A run may override its date; clearing it or changing PIT restores the shared default. */
export function useRegimeAsOf(initialAsOf = '') {
  const researchDay = useResearchDay()
  const context = useResearchContextIdentity()
  const [selection, setSelection] = useState<{ context: string; date: string } | null>(null)
  const asOf = (selection?.context === context ? selection.date : '') || initialAsOf || researchDay || ''
  const setAsOf = (date: string) => setSelection({ context, date })
  return [asOf, setAsOf] as const
}
