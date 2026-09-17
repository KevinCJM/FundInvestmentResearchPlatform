import { describe, expect, it } from 'vitest'
import { createBlankRegimeDefinition, definitionForRequest } from '../../services/regimeGraph'
import { bindStudyReference, scenarioCenterFromQuery, studyDraft, studyMappingIssue } from './regimeStudy'

describe('研究用途与精确参考', () => {
  it('旧实时链接保持精确目标且打开实时入口', () => {
    const query = new URLSearchParams('center=historical&mode=realtime&definition=d&revision=7')
    expect(scenarioCenterFromQuery(query)).toBe('realtime')
    expect(query.get('definition')).toBe('d')
    expect(query.get('revision')).toBe('7')
    expect(scenarioCenterFromQuery(new URLSearchParams('center=events&mode=realtime'))).toBe('events')
  })
  it('旧序列化不增加 study，载入保存版本不自动迁移', () => {
    const old = { ...createBlankRegimeDefinition(), id: 'old', revision: 3 }
    expect(definitionForRequest(old)).not.toHaveProperty('study')
    expect(studyDraft(old, 'historical_reference')).toBe(old)
    expect(studyDraft(createBlankRegimeDefinition(), 'historical_reference')).toMatchObject({ default_mode: 'retrospective', study: { purpose: 'historical_reference' } })
  })
  it('更换参考清除映射和校准，保留图；不猜名称或排列', () => {
    const definition = studyDraft(createBlankRegimeDefinition(), 'realtime_recognition')
    definition.states = [{ id: 'custom', label: '牛', color: '#000000' }]
    definition.study = { ...definition.study!, calibration_id: 'cal', state_mapping: { custom: 'bull' } }
    const next = bindStudyReference(definition, { run_id: 'r', publication_id: 'p', content_hash: 'h' })
    expect(next.graph).toBe(definition.graph)
    expect(next.study).not.toHaveProperty('calibration_id')
    expect(next.study).not.toHaveProperty('state_mapping')
    expect(studyMappingIssue(next, [{ id: 'bull' }])).not.toBe('')
    expect(studyMappingIssue(next, [{ id: 'custom' }])).toBe('')
  })
})

it('explicit adoption sets both ids; edits remove both keys and preserve old payloads', async () => {
  const { invalidateStudyQualification, adoptStudyQualification } = await import('./regimeStudy')
  const old = studyDraft(createBlankRegimeDefinition(), 'realtime_recognition')
  const before = JSON.stringify(definitionForRequest(old))
  expect(invalidateStudyQualification(old)).toBe(old)
  expect(JSON.stringify(definitionForRequest(old))).toBe(before)
  const adopted = adoptStudyQualification(old, 'cal', 'qualified')
  expect(adopted.study).toMatchObject({ calibration_id: 'cal', qualification_id: 'qualified' })
  expect(invalidateStudyQualification(adopted).study).toEqual(old.study)
  expect(bindStudyReference(adopted).study).not.toHaveProperty('qualification_id')
  expect(adoptStudyQualification(adopted, 'old-compatible').study).not.toHaveProperty('qualification_id')
})
