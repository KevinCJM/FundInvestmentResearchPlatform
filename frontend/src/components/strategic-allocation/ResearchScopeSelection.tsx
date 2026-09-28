import type { ReactNode } from 'react'
import { useI18n } from '../../i18n/runtime'
import type { StrategicCatalog } from '../../services/strategicAllocation'
import { Field, inputClass } from '../risk-models/ResearchUI'

type ScopeChoice = { allocationName: string; strategicUniverseId: string; implementationMappingId: string }

export default function ResearchScopeSelection({ catalog, allocationName, strategicUniverseId, onChange, children }: {
  catalog: StrategicCatalog; allocationName: string; strategicUniverseId: string
  onChange: (value: Partial<ScopeChoice>) => void; children: ReactNode
}) {
  const { s } = useI18n()
  const universe = catalog.strategic_universes?.find(item => item.id === strategicUniverseId)
  const allocation = catalog.allocations.find(item => item.alloc_name === allocationName)
  const assets = universe?.definition.assets ?? allocation?.assets ?? []

  function select(value: string) {
    // One visible choice, with mutually exclusive internal source identities.
    onChange({ strategicUniverseId: value.startsWith('strategic:') ? value.slice(10) : '',
      allocationName: value.startsWith('allocation:') ? value.slice(11) : '', implementationMappingId: '' })
  }

  return <div className="min-w-0 space-y-3">
    <div className="grid gap-4 sm:grid-cols-2">
      {children}
      <Field required label={s('saaScope.scope')}>
        <select required className={inputClass} value={strategicUniverseId ? `strategic:${strategicUniverseId}` : allocationName ? `allocation:${allocationName}` : ''} onChange={event => select(event.target.value)}>
          <option value="">{s('saaScope.choose')}</option>
          {catalog.strategic_universes?.length ? <optgroup label={s('saaScope.strategicGroup')}>{catalog.strategic_universes.map(item => <option key={item.id} value={`strategic:${item.id}`}>{item.name} · {item.definition.as_of}</option>)}</optgroup> : null}
          {catalog.allocations.length ? <optgroup label={s('saaScope.productGroup')}>{catalog.allocations.map(item => <option key={item.alloc_name} value={`allocation:${item.alloc_name}`}>{item.alloc_name}</option>)}</optgroup> : null}
        </select>
      </Field>
    </div>
    {assets.length > 0 && <p className="text-sm text-slate-600">{s('saaScope.assets', { assets: assets.map(asset => asset.name).join('、') })}</p>}
    {!catalog.allocations.length && !catalog.strategic_universes?.length && <p className="text-sm text-slate-600">{s('saaScope.empty')}</p>}
    {universe && <p className="text-sm text-slate-600">{s('saaScope.researchOnly')}</p>}
  </div>
}
