import { act, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { builtinBundle, builtinCatalogs, DEFAULT_LANGUAGES, normalizeLocale, routeTranslationKey } from './catalogs'
import { businessText, chooseLocale, formatDate, formatNumber, i18n, installBusinessBundle, installLanguageRegistry, resolveActiveLocale, LANGUAGE_KEY, systemText, useI18n } from './runtime'
import { localizeIndicatorMeta } from './indicatorMetadata'
import { allStages } from '../app/processRegistry'
import type { IndicatorMeta } from '../services/customIndicators'
import { canvasModel, emptyOutput, graphSignature } from '../components/indicator-graph/indicatorGraphAdapter'
import type { GraphDocument } from '../services/indicatorGraph'

async function reset() {
  localStorage.clear()
  installLanguageRegistry(DEFAULT_LANGUAGES, false)
  installBusinessBundle('zh-CN', builtinBundle('zh-CN').business)
  installBusinessBundle('en-US', builtinBundle('en-US').business)
  await i18n.changeLanguage('zh-CN')
}
beforeEach(reset)
afterEach(reset)

describe('isolated system and business translations', () => {
  it('dynamic languages use their own business terms and bundled system fallback only', async () => {
    installLanguageRegistry([...DEFAULT_LANGUAGES, { id: 'ja-JP', label: '日本語', enabled: true, fallback_locale: 'en-US' }])
    installBusinessBundle('en-US', { 'axes.asset': 'Private English override' })
    installBusinessBundle('ja-JP', { 'variables.periods_per_year.label': '年間観測数' })
    await chooseLocale('ja-JP')
    expect(i18n.language).toBe('ja-JP')
    expect(systemText('common.save')).toBe('Save')
    expect(businessText('variables.periods_per_year.label')).toBe('年間観測数')
    expect(businessText('axes.asset')).toBe('Asset')
    expect(localStorage.getItem(LANGUAGE_KEY)).toBe('ja-JP')
  })
  it('disabled and unregistered browser languages fall back without destroying translations', async () => {
    const language = { id: 'ja-JP', label: '日本語', enabled: true, fallback_locale: 'en-US' }
    installLanguageRegistry([...DEFAULT_LANGUAGES, language])
    installBusinessBundle('ja-JP', { 'axes.asset': '資産' })
    installLanguageRegistry([...DEFAULT_LANGUAGES, { ...language, enabled: false }])
    expect(resolveActiveLocale('ja-JP', 'en-US')).toBe('en-US')
    expect(resolveActiveLocale('unknown', 'unknown')).toBe('zh-CN')
    installLanguageRegistry([...DEFAULT_LANGUAGES, language])
    await chooseLocale('ja-JP')
    expect(businessText('axes.asset')).toBe('資産')
  })
  it('normalizes locale codes and rejects unbounded or unsupported locale syntax', () => {
    expect(normalizeLocale('JA-jp')).toBe('ja-JP')
    expect(normalizeLocale('zh-hANT-tw')).toBe('zh-Hant-TW')
    for (const invalid of ['cimode', 'zh_CN', '../en-US', 'en-US-u-ca-gregory', 'a', '']) expect(normalizeLocale(invalid)).toBeNull()
    expect(() => installLanguageRegistry([...DEFAULT_LANGUAGES, { ...DEFAULT_LANGUAGES[0], id: 'ZH-cn' }])).toThrow()
  })
  it('business resources can never overwrite system strings', () => {
    installBusinessBundle('zh-CN', { 'common.save': '不得影响系统', 'variables.periods_per_year.label': '每年观察期数' })
    expect(systemText('common.save')).toBe('保存')
    expect(businessText('variables.periods_per_year.label')).toBe('每年观察期数')
    expect(businessText('common.cancel', '业务缺失')).toBe('业务缺失')
    expect(systemText('variables.periods_per_year.label', {}, '系统缺失')).toBe('系统缺失')
  })
  it('language changes are reactive and persist only the preference', async () => {
    function Labels() { const { s, b } = useI18n(); return <p>{s('common.save')} · {b('valueTypes.scalar.numeric')}</p> }
    render(<Labels />)
    expect(screen.getByText('保存 · 单个数值')).toBeInTheDocument()
    await act(async () => { await chooseLocale('en-US') })
    expect(screen.getByText('Save · Single number')).toBeInTheDocument()
    expect(localStorage.getItem(LANGUAGE_KEY)).toBe('en-US')
    expect(document.documentElement.lang).toBe('en-US')
    await act(async () => { await chooseLocale(null, 'zh-CN') })
    expect(localStorage.getItem(LANGUAGE_KEY)).toBeNull()
    expect(screen.getByText('保存 · 单个数值')).toBeInTheDocument()
  })
  it('changing a business term updates subscribers without changing graph identity', async () => {
    const document: GraphDocument = { graph: { graph_version: 1, nodes: [{ id: 'input', kind: 'variable', variable_id: 'periods_per_year' }], outputs: [{ ...emptyOutput('result', '自定义标题 returns'), node_id: 'input' }] }, positions: {} }
    const signature = graphSignature(document.graph)
    function Labels() { useI18n(); const model = canvasModel(document, [], [], { input: { kind: 'scalar' } }); return <div>{model.nodes.map(node => <p key={node.id}>{node.label}</p>)}</div> }
    render(<Labels />)
    expect(screen.getByText('年化因子')).toBeInTheDocument()
    act(() => installBusinessBundle('zh-CN', { 'variables.periods_per_year.label': '每年观察期数' }))
    await waitFor(() => expect(screen.getByText('每年观察期数')).toBeInTheDocument())
    expect(screen.getByText('自定义标题 returns')).toBeInTheDocument()
    expect(graphSignature(document.graph)).toBe(signature)
  })
  it('locale overrides do not leak across languages and reset removes old labels', async () => {
    installBusinessBundle('zh-CN', { 'variables.returns.label': '中文专属' })
    await i18n.changeLanguage('en-US')
    expect(businessText('variables.returns.label')).toBe('Adjusted NAV simple returns')
    installBusinessBundle('zh-CN', {})
    await i18n.changeLanguage('zh-CN')
    expect(businessText('variables.returns.label')).toBe('复权净值普通收益率')
  })
  it('interpolation arguments cannot hijack translation options or execute markup', () => {
    const name = '<img src=x onerror=alert(1)>'
    const translated = systemText('navigation.tool', { name, ns: 'business', lng: 'en-US' })
    expect(translated).toBe(`工具 · ${name}`)
    const { container } = render(<p>{translated}</p>)
    expect(container.querySelector('img')).toBeNull()
    expect(container).toHaveTextContent(name)
  })
  it('number/date formatting preserves missing values and calendar dates', () => {
    expect(formatNumber(null)).toBe('—')
    expect(formatNumber(NaN)).toBe('—')
    expect(formatNumber(Infinity)).toBe('—')
    expect(formatNumber(0.125, { style: 'percent', minimumFractionDigits: 1 }, 'en-US')).toBe('12.5%')
    expect(formatDate('2026-09-07', 'en-US')).toBe('09/07/2026')
    expect(formatDate('invalid')).toBe('—')
  })
  it('all navigation labels are registered in the system catalog', () => {
    for (const stage of allStages) {
      expect(Object.prototype.hasOwnProperty.call(builtinCatalogs.system, routeTranslationKey(stage.path)), stage.path).toBe(true)
      for (const entry of [...stage.nodes, ...(stage.tools || [])]) {
        expect(Object.prototype.hasOwnProperty.call(builtinCatalogs.system, routeTranslationKey(entry.path)), entry.path).toBe(true)
      }
    }
  })
  it('metadata projection changes only display fields', () => {
    const metadata: IndicatorMeta = { engine_version: 'test', workspace_scope: 'shared', periods: [], templates: [], limits: {}, variables: [{ name: 'periods_per_year', label: '年化因子', value_type: 'scalar', dtype: 'float64', latex: 'p', axes: [] }], operators: [] }
    const before = JSON.stringify(metadata)
    installBusinessBundle('zh-CN', { 'variables.periods_per_year.label': '每年观察期数' })
    const localized = localizeIndicatorMeta(metadata)!
    expect(localized.variables[0].label).toBe('每年观察期数')
    expect(localized.variables[0]).toMatchObject({ name: 'periods_per_year', value_type: 'scalar', latex: 'p' })
    expect(JSON.stringify(metadata)).toBe(before)
  })
})
