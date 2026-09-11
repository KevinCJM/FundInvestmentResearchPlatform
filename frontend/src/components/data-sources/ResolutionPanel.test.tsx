import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ResolutionPanel from './ResolutionPanel'
import type { ResolutionPolicyRecord, SourceCatalog } from '../../services/dataSources'
import * as api from '../../services/dataSources'

vi.mock('../../services/dataSources', () => ({ getResolutionPolicy: vi.fn(), saveResolutionPolicy: vi.fn(), runResolution: vi.fn(), previewResolution: vi.fn() }))
const catalog = {
  editing_enabled: true,
  sources: [{ config: { id: 'tushare', name: 'Tushare', enabled: true } }, { config: { id: 'akshare', name: 'AKShare', enabled: true } }],
  targets: { categories: [{ category_id: 'market', label: '行情与净值' }], tables: [{ table_id: 'market.quote_daily', category_id: 'market', label: '日行情', source_mappable: true, usage: 'external_import', fields: [{ name: 'close', label: '收盘价', source_mappable: true, data_type: 'float64' }] }] },
} as unknown as SourceCatalog
let saved: ResolutionPolicyRecord
beforeEach(() => {
  vi.resetAllMocks()
  saved = { revision: 1, updated_at: '', runs: [], config: { default_source_priority: ['tushare', 'akshare'], tables: [{ table_id: 'market.quote_daily', source_priority: [], fallback_on_missing: true, fallback_on_invalid: false, conflict_action: 'quarantine', required_fields: ['close'], compare_fields: ['close'], absolute_tolerance: .000001, relative_tolerance: .0001, max_relative_jump: null, field_rules: [] }] } }
  vi.mocked(api.getResolutionPolicy).mockImplementation(async () => structuredClone(saved))
  vi.mocked(api.saveResolutionPolicy).mockImplementation(async config => ({ ...saved, revision: 2, config }))
})

describe('ResolutionPanel', () => {
  it('调整全局优先级后须保存，保存使用原修订号', async () => {
    const dirty = vi.fn()
    render(<ResolutionPanel catalog={catalog} onDirty={dirty} />)
    fireEvent.click(await screen.findByRole('button', { name: '全局来源优先级 akshare 上移' }))
    expect(dirty).toHaveBeenCalledWith(true)
    expect(screen.getByRole('button', { name: '应用规则到已下载候选' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '保存取值规则' }))
    await waitFor(() => expect(api.saveResolutionPolicy).toHaveBeenCalledWith(expect.objectContaining({ default_source_priority: ['akshare', 'tushare'] }), 1))
    expect(await screen.findByText(/规则修订 2/)).toBeVisible()
    expect(screen.getByRole('button', { name: '应用规则到已下载候选' })).toBeEnabled()
  })

  it('异常替代与合法冲突处理分别配置', async () => {
    render(<ResolutionPanel catalog={catalog} onDirty={vi.fn()} />)
    fireEvent.click(await screen.findByLabelText(/主来源异常时使用下一来源/))
    fireEvent.change(screen.getByLabelText('双方合法但数值不一致时'), { target: { value: 'prefer_priority' } })
    fireEvent.click(screen.getByRole('button', { name: '保存取值规则' }))
    await waitFor(() => expect(api.saveResolutionPolicy).toHaveBeenCalledWith(expect.objectContaining({ tables: [expect.objectContaining({ fallback_on_invalid: true, conflict_action: 'prefer_priority' })] }), 1))
  })

  it('合并只作用于已下载候选，并显示冲突和未发布状态', async () => {
    vi.mocked(api.runResolution).mockResolvedValue({ table_id: 'market.quote_daily', published: false, summary: { selected_rows: 0, CONFLICT: 1 }, decisions: [{ key: { instrument_id: 'x' }, status: 'CONFLICT', selected_source: null, skipped: [], conflicts: [{ source_id: 'akshare', fields: ['close'] }] }] })
    render(<ResolutionPanel catalog={catalog} onDirty={vi.fn()} />)
    fireEvent.click(await screen.findByRole('button', { name: '应用规则到已下载候选' }))
    expect(await screen.findByRole('region', { name: '多源取值结果' })).toHaveTextContent('取值结果 · 未发布')
    expect(screen.getByText(/数据冲突，等待处理/)).toBeVisible()
    expect(api.runResolution).toHaveBeenCalledWith('market.quote_daily', 1, '', '', '')
  })

  it('修订冲突保留未保存配置', async () => {
    vi.mocked(api.saveResolutionPolicy).mockRejectedValue(new Error('规则已被修改，请重新加载。'))
    render(<ResolutionPanel catalog={catalog} onDirty={vi.fn()} />)
    fireEvent.click(await screen.findByLabelText(/主来源异常时使用下一来源/))
    fireEvent.click(screen.getByRole('button', { name: '保存取值规则' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('规则已被修改')
    expect(screen.getByLabelText(/主来源异常时使用下一来源/)).toBeChecked()
  })

  it('只读环境不可保存或执行候选合并', async () => {
    render(<ResolutionPanel catalog={{ ...catalog, editing_enabled: false }} onDirty={vi.fn()} />)
    expect(await screen.findByRole('button', { name: '全局来源优先级 akshare 上移' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '应用规则到已下载候选' })).toBeDisabled()
  })
})
