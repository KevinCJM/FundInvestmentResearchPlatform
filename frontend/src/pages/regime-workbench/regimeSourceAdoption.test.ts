import { expect, it } from 'vitest'
import { adoptStudyQualification } from './regimeStudy'
import { reliabilityDefinition } from './regimeReliabilityFixtures'
import type { IndexSnapshotBindings } from '../../services/regimeProspective'

const bindings: IndexSnapshotBindings = { source: { snapshot_id: 'next', snapshot_generation: 'next', source_file: 'index_daily_df.parquet', file_checksum: 'a'.repeat(64) } }
it('adoption creates a source-bound draft without mutating the old graph or model parameters', () => {
  const definition = structuredClone(reliabilityDefinition)
  definition.graph.nodes[0].type = 'source.index'
  definition.graph.nodes[0].parameters.ts_code = '000300.SH'
  const original = structuredClone(definition)
  const adopted = adoptStudyQualification(definition, 'cal', 'qual', bindings)
  expect(adopted.study).toMatchObject({ calibration_id: 'cal', qualification_id: 'qual' })
  expect(adopted.graph.nodes[0].parameters).toMatchObject({ ...original.graph.nodes[0].parameters, ...bindings.source })
  expect(definition).toEqual(original)
  expect(adopted.graph.outputs).toBe(definition.graph.outputs)
})
it('a data successor cannot change algorithms, unrelated nodes, or add non-binding fields', () => {
  const definition = structuredClone(reliabilityDefinition)
  expect(() => adoptStudyQualification(definition, 'cal', 'qual', bindings)).toThrow('续接来源')
  definition.graph.nodes[0].type = 'source.index'
  expect(() => adoptStudyQualification(definition, 'cal', undefined, bindings)).toThrow('续接来源')
  expect(() => adoptStudyQualification(definition, 'cal', 'qual', { unknown: bindings.source })).toThrow('续接来源')
  expect(() => adoptStudyQualification(definition, 'cal', 'qual', { source: { ...bindings.source, threshold: 999 } } as never)).toThrow('续接来源')
})
