import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import LanguageTerminology from './LanguageTerminology'
import * as api from '../services/localization'
import { DEFAULT_LANGUAGES } from '../i18n/catalogs'
import { i18n, installLanguageRegistry } from '../i18n/runtime'
import { matrixFixture, translationState } from '../test/localizationFixtures'

vi.mock('../services/localization', async original => ({
  ...await original<typeof import('../services/localization')>(),
  getTranslationState: vi.fn(), getTranslationMatrix: vi.fn(), getTranslationHistory: vi.fn(), updateLanguages: vi.fn(),
  updateBusinessTranslations: vi.fn(), updateLanguagePreferences: vi.fn(), exportBusinessTranslations: vi.fn(),
  validateTranslationImport: vi.fn(), applyTranslationImport: vi.fn(), restoreTranslations: vi.fn(),
}))
const key = 'variables.periods_per_year.label'
const jsonFile = (value: unknown, name: string) => {
  const content = JSON.stringify(value)
  const file = new File([content], name, { type: 'application/json' })
  Object.defineProperty(file, 'text', { value: async () => content })
  return file
}
let state: api.TranslationState
beforeEach(async () => {
  vi.resetAllMocks(); localStorage.clear(); installLanguageRegistry(DEFAULT_LANGUAGES, false); await i18n.changeLanguage('zh-CN')
  state = translationState()
  vi.mocked(api.getTranslationState).mockImplementation(async () => structuredClone(state))
  vi.mocked(api.getTranslationMatrix).mockImplementation(async options => ({ ...matrixFixture(options.scope, options.scope === 'business' ? [key] : ['common.save'], state), sort_by: options.sortBy ?? 'code', sort_dir: options.sortDir ?? 'asc' }))
  vi.mocked(api.getTranslationHistory).mockResolvedValue({ revision: 0, items: [{ revision: 0, at: null, action: 'initial', reason: '', change_count: 0 }] })
  vi.mocked(api.updateLanguages).mockImplementation(async (expected, locales) => {
    state = { ...state, preferences_revision: expected + 1, locales }
    return { locales, preferences_revision: state.preferences_revision, default_locale: state.default_locale }
  })
  vi.mocked(api.updateBusinessTranslations).mockImplementation(async (expected, changes) => {
    const overrides = structuredClone(state.overrides)
    for (const change of changes) {
      const entries = overrides[change.locale] ??= {}
      if (change.value === null) delete entries[change.key]; else entries[change.key] = change.value
    }
    state = { ...state, revision: expected + 1, overrides }
    return { revision: state.revision, overrides, changes: [] }
  })
})
afterEach(async () => { installLanguageRegistry(DEFAULT_LANGUAGES, false); localStorage.clear(); await i18n.changeLanguage('zh-CN') })
async function openBusiness() {
  const user = userEvent.setup()
  render(<LanguageTerminology />)
  await screen.findByRole('grid', { name: '系统翻译（内置只读）' })
  await user.click(screen.getByRole('tab', { name: '业务翻译（可自定义）' }))
  await screen.findByRole('gridcell', { name: `${key} zh-CN` })
  return user
}
async function editCell(user: ReturnType<typeof userEvent.setup>, value: string, locale = 'zh-CN') {
  await user.dblClick(screen.getByRole('gridcell', { name: `${key} ${locale}` }))
  await user.clear(screen.getByRole('textbox', { name: `编辑译文 ${key} ${locale}` }))
  await user.type(screen.getByRole('textbox', { name: `编辑译文 ${key} ${locale}` }), value)
  await user.keyboard('{Enter}')
}

describe('code-first multilingual management', () => {
  it('system rows expose code keys and separate languages, but no write controls', async () => {
    const user = userEvent.setup(); render(<LanguageTerminology />)
    const table = await screen.findByRole('grid', { name: '系统翻译（内置只读）' })
    expect(within(table).getByRole('rowheader')).toHaveTextContent('common.save')
    expect(within(table).getByRole('gridcell', { name: 'common.save zh-CN' })).toHaveTextContent('保存')
    expect(within(table).getByRole('gridcell', { name: 'common.save en-US' })).toHaveTextContent('Save')
    await user.dblClick(within(table).getByRole('gridcell', { name: 'common.save zh-CN' }))
    expect(screen.queryByRole('textbox', { name: /编辑译文/ })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '保存业务翻译' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '导入业务覆盖' })).not.toBeInTheDocument()
  })
  it('searches across the table and sorts by code or language columns', async () => {
    const user = await openBusiness()
    vi.mocked(api.getTranslationMatrix).mockClear()
    await user.type(screen.getByRole('searchbox', { name: '搜索词条' }), 'periods')
    await waitFor(() => expect(api.getTranslationMatrix).toHaveBeenCalledWith(expect.objectContaining({ scope: 'business', q: 'periods', sortBy: 'code', sortDir: 'asc', page: 1 }), expect.any(AbortSignal)))
    const codeHeader = screen.getByRole('columnheader', { name: /系统代码/ })
    await user.click(within(codeHeader).getByRole('button', { name: /降序排序/ }))
    await waitFor(() => expect(api.getTranslationMatrix).toHaveBeenCalledWith(expect.objectContaining({ sortBy: 'code', sortDir: 'desc', page: 1 }), expect.any(AbortSignal)))
    const english = screen.getByRole('columnheader', { name: /English/ })
    await user.click(within(english).getByRole('button', { name: /升序排序/ }))
    await waitFor(() => expect(api.getTranslationMatrix).toHaveBeenCalledWith(expect.objectContaining({ sortBy: 'en-US', sortDir: 'asc', page: 1 }), expect.any(AbortSignal)))
    await user.click(screen.getByRole('button', { name: '清除筛选' }))
    expect(screen.getByRole('searchbox', { name: '搜索词条' })).toHaveValue('')
  })
  it('commits multiple language cells to one preview before explicit save', async () => {
    const user = await openBusiness()
    await editCell(user, '每年观察期数'); await editCell(user, 'Annual observations', 'en-US')
    expect(screen.getByRole('region', { name: '修改预览' })).toHaveTextContent('每年观察期数')
    expect(api.updateBusinessTranslations).not.toHaveBeenCalled()
    await user.click(screen.getByRole('button', { name: '保存业务翻译' }))
    await waitFor(() => expect(api.updateBusinessTranslations).toHaveBeenCalledWith(0, [{ key, locale: 'zh-CN', value: '每年观察期数' }, { key, locale: 'en-US', value: 'Annual observations' }], ''))
    expect(await screen.findByText('业务翻译已保存，修订 1。')).toBeInTheDocument()
  })
  it('conflicts preserve the draft and an explicit refresh rebases the preview', async () => {
    vi.mocked(api.updateBusinessTranslations).mockRejectedValueOnce(new api.LocalizationApiError(409, 'I18N_REVISION_CONFLICT', '冲突'))
    const user = await openBusiness(); await editCell(user, '本次编辑')
    await user.click(screen.getByRole('button', { name: '保存业务翻译' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('草稿已保留')
    expect(screen.getByRole('gridcell', { name: `${key} zh-CN` })).toHaveTextContent('本次编辑')
    state = { ...state, revision: 3, overrides: { 'zh-CN': { [key]: '别人已编辑' } } }
    await user.click(within(screen.getByRole('alert')).getByRole('button', { name: '刷新' }))
    await waitFor(() => expect(screen.getByRole('region', { name: '修改预览' })).toHaveTextContent('别人已编辑'))
    expect(await screen.findByRole('gridcell', { name: `${key} zh-CN` })).toHaveTextContent('本次编辑')
    await user.click(screen.getByRole('button', { name: '保存业务翻译' }))
    await waitFor(() => expect(api.updateBusinessTranslations).toHaveBeenLastCalledWith(3, [{ key, locale: 'zh-CN', value: '本次编辑' }], ''))
  })
  it('switching interface language leaves both language columns and pending text intact', async () => {
    const user = await openBusiness(); await editCell(user, '保留中文草稿')
    await user.selectOptions(screen.getByRole('combobox', { name: '当前浏览器语言' }), 'en-US')
    expect(screen.getByRole('heading', { name: 'Language and terminology' })).toBeInTheDocument()
    expect(screen.getByRole('gridcell', { name: `${key} zh-CN` })).toHaveTextContent('保留中文草稿')
    expect(screen.getByRole('gridcell', { name: `${key} en-US` })).toHaveTextContent('Periods per year')
    expect(api.updateBusinessTranslations).not.toHaveBeenCalled()
  })
  it('reset deletes an override instead of storing an empty translation', async () => {
    state.overrides = { 'zh-CN': { [key]: '自定义名称' } }
    const user = await openBusiness()
    await user.click(screen.getByRole('button', { name: '恢复默认' }))
    expect(screen.getByRole('gridcell', { name: `${key} zh-CN` })).toHaveTextContent('年化因子')
    await user.click(screen.getByRole('button', { name: '保存业务翻译' }))
    await waitFor(() => expect(api.updateBusinessTranslations).toHaveBeenCalledWith(0, [{ key, locale: 'zh-CN', value: null }], ''))
    expect(api.updateLanguagePreferences).not.toHaveBeenCalled()
  })
  it('invalid active edits cannot be lost by switching filters or publishing other cells', async () => {
    const user = await openBusiness(); await editCell(user, '先前有效修改', 'en-US')
    await user.dblClick(screen.getByRole('gridcell', { name: `${key} zh-CN` }))
    const editor = screen.getByRole('textbox', { name: `编辑译文 ${key} zh-CN` })
    await user.clear(editor); await user.type(editor, '<script>bad</script>'); await user.keyboard('{Enter}')
    expect(await screen.findByRole('alert')).toHaveTextContent('译文')
    expect(editor).toHaveValue('<script>bad</script>')
    expect(screen.getByRole('searchbox', { name: '搜索词条' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '保存业务翻译' })).toBeDisabled()
    await user.keyboard('{Escape}')
    expect(screen.getByRole('gridcell', { name: `${key} zh-CN` })).toHaveTextContent('年化因子')
    expect(screen.getByRole('button', { name: '保存业务翻译' })).toBeEnabled()
  })
  it('adding a language creates a column with missing cells and an explicit fallback', async () => {
    const user = await openBusiness()
    await user.click(screen.getByRole('button', { name: '＋ 添加语言' }))
    const manager = screen.getByRole('region', { name: '管理语言列' })
    await user.type(within(manager).getByRole('textbox', { name: /^语言代码$/ }), 'JA-jp')
    await user.type(within(manager).getByRole('textbox', { name: /^语言名称$/ }), '日本語')
    await user.selectOptions(within(manager).getByRole('combobox', { name: /^缺少译文时回退到$/ }), 'en-US')
    await user.click(within(manager).getByRole('button', { name: /^保存$/ }))
    const cell = await screen.findByRole('gridcell', { name: `${key} ja-JP` })
    expect(cell).toHaveTextContent('未翻译'); expect(cell).toHaveTextContent('Periods per year')
    expect(api.updateLanguages).toHaveBeenCalledWith(0, expect.arrayContaining([expect.objectContaining({ id: 'ja-JP', fallback_locale: 'en-US' })]))
    await editCell(user, '年間観測数', 'ja-JP')
    await user.click(screen.getByRole('button', { name: '保存业务翻译' }))
    await waitFor(() => expect(api.updateBusinessTranslations).toHaveBeenCalledWith(0, [{ key, locale: 'ja-JP', value: '年間観測数' }], ''))
  })
  it('import requires a review and rejects system packages before calling the API', async () => {
    const user = await openBusiness()
    const fileInput = screen.getByLabelText('导入业务覆盖', { selector: 'input' })
    await user.upload(fileInput, jsonFile({ format_version: 1, scope: 'system', entries: [] }, 'invalid.json'))
    expect(await screen.findByRole('alert')).toHaveTextContent('不是有效')
    expect(api.validateTranslationImport).not.toHaveBeenCalled()
    const pack: api.TranslationPackage = { format_version: 1, scope: 'business', entries: [{ key, locale: 'zh-CN', value: '导入名称' }] }
    vi.mocked(api.validateTranslationImport).mockResolvedValue({ valid: true, revision: 0, confirmation_digest: 'a'.repeat(64), catalog_changed: false, changes: [{ key, locale: 'zh-CN', before: null, after: '导入名称' }] })
    vi.mocked(api.applyTranslationImport).mockResolvedValue({ revision: 1, changes: [], overrides: {} })
    await user.upload(fileInput, jsonFile(pack, 'valid.json'))
    const preview = await screen.findByRole('region', { name: '确认导入差异' })
    expect(preview).toHaveTextContent('导入名称'); expect(api.applyTranslationImport).not.toHaveBeenCalled()
    await user.click(within(preview).getByRole('button', { name: '确认并应用导入' }))
    await waitFor(() => expect(api.applyTranslationImport).toHaveBeenCalledWith(0, pack, 'a'.repeat(64)))
  })
})
