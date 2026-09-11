import { describe, expect, it } from 'vitest'
import { prototypeConfigs } from './prototypeRegistry'
import { prototypeBlueprints } from './prototypeBlueprintRegistry'

describe('静态节点功能蓝图', () => {
  it('每个静态页面都有一份职责蓝图，且不存在孤立蓝图', () => {
    expect(Object.keys(prototypeBlueprints).sort()).toEqual(Object.keys(prototypeConfigs).sort())
  })

  it('每份蓝图均包含步骤、功能、产出和三段职责边界', () => {
    Object.values(prototypeBlueprints).forEach((blueprint) => {
      expect(blueprint.steps.length).toBeGreaterThanOrEqual(4)
      expect(blueprint.capabilities.length).toBeGreaterThan(0)
      expect(blueprint.outputs.length).toBeGreaterThanOrEqual(3)
      expect(blueprint.boundary.owns).toBeTruthy()
      expect(blueprint.boundary.reuses).toBeTruthy()
      expect(blueprint.boundary.excludes).toBeTruthy()
    })
  })
})
