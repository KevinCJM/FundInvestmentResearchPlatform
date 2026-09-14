import { beforeEach, expect, it } from 'vitest'
import { allocationJourneyPath, readAllocationJourney, updateAllocationJourney } from './allocationJourney'
beforeEach(() => { localStorage.clear(); sessionStorage.clear() })
it('更换产品域保留独立战略定义与目标，同时清除依赖旧映射的基线', () => {
  updateAllocationJourney({ mandateId: 'm', strategicUniverseId: 's', universeId: 'domain-one', implementationMappingId: 'map-one', baselineId: 'b', taaRunId: 't' })
  updateAllocationJourney({ universeId: 'domain-two' })
  expect(readAllocationJourney()).toMatchObject({ mandateId: 'm', strategicUniverseId: 's', universeId: 'domain-two' })
  expect(readAllocationJourney().baselineId).toBeUndefined()
  expect(readAllocationJourney().implementationMappingId).toBeUndefined()
})
it('不依赖产品域的前瞻政策不会因为选产品域丢失，但换战略轴或目标会失效', () => {
  updateAllocationJourney({ mandateId: 'm', strategicUniverseId: 's', baselineId: 'b' })
  updateAllocationJourney({ universeId: 'domain' })
  expect(readAllocationJourney().baselineId).toBe('b')
  updateAllocationJourney({ mandateId: 'm2' })
  expect(readAllocationJourney().baselineId).toBeUndefined()
  updateAllocationJourney({ implementationMappingId: 'map', baselineId: 'b2' })
  updateAllocationJourney({ strategicUniverseId: 's2' })
  expect(readAllocationJourney().implementationMappingId).toBeUndefined()
  expect(readAllocationJourney().baselineId).toBeUndefined()
})
it('前瞻导航带明确目标范围及映射，不发虚构的大类名', () => {
  const path = allocationJourneyPath('saa', { mandateId: 'm', strategicUniverseId: 's', implementationMappingId: 'map', allocationName: '产品代理' })
  expect(path).toBe('/pre-investment/saa/policy?mandate=m&strategic_universe=s&mapping=map')
})
