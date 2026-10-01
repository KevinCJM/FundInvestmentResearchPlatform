import {useCallback,useEffect,useRef,useState} from 'react'
import type {IndicatorDraft} from '../../services/customIndicators'
import type {IndicatorPreview} from '../../services/researchContracts'

/** Indicator-workbench ownership: clearing a result must not discard its active definition. */
export function useAdoptedIndicatorPreview(draft: IndicatorDraft, selectedId: string | null, onAdopt: (value: IndicatorPreview) => void) {
  const [preview, setPreview] = useState<IndicatorPreview | null>(null)
  const [definition, setDefinition] = useState<IndicatorDraft | null>(null)
  const adoptedId = useRef('')
  const adoptRef = useRef(onAdopt)
  adoptRef.current = onAdopt
  const receive = useCallback((value: IndicatorPreview | null, explicit = false) => {
    if (!explicit && value && adoptedId.current === value.preview_id) return
    setPreview(value)
    if (!value) return
    adoptedId.current = value.preview_id
    setDefinition(value.definition)
    adoptRef.current(value)
  }, [])
  useEffect(() => { setDefinition(null); setPreview(null) }, [draft, selectedId])
  return { preview, definition, setPreview, receive }
}
