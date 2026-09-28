import { afterEach, expect, it } from 'vitest'
import { i18n } from './runtime'
import { researchMessage } from './researchMessages'
import { localizeStage } from './navigation'
import { getStage } from '../app/processRegistry'

afterEach(async () => { await i18n.changeLanguage('zh-CN') })
it('translates frozen diagnostics in both directions without changing names, numbers or unknown messages', async () => {
  const source = '训练区 540 条、验证区 260 条共同收益；各至少需要 20 条。'
  await i18n.changeLanguage('en-US')
  const translated = researchMessage(source)
  expect(translated).toBe('540 common training returns and 260 validation returns; at least 20 are required in each.')
  expect(researchMessage('家庭长期目标')).toBe('家庭长期目标')
  expect(researchMessage('未登记的新诊断：甲资产')).toBe('未登记的新诊断：甲资产')
  expect(researchMessage('甲资产 缺少明确的产品映射。')).toBe('甲资产 lacks an explicit product mapping.')
  await i18n.changeLanguage('zh-CN')
  expect(researchMessage(translated)).toBe(source)
})
it('all pre-investment directory descriptions and tool labels follow the active language', async () => {
  await i18n.changeLanguage('en-US')
  const stage = localizeStage(getStage('pre-investment'))
  for (const item of [...stage.nodes, ...(stage.tools ?? [])]) {
    expect(item.label).not.toMatch(/[\u3400-\u9fff]/u)
    expect(item.description).not.toMatch(/[\u3400-\u9fff]/u)
  }
  await i18n.changeLanguage('zh-CN')
  expect(localizeStage(getStage('pre-investment')).nodes[0].description).toContain('设置收益目标')
})
