import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import DataManagement from './DataManagement'

vi.mock('../components/data-sources/DataDownloadWorkspace', () => ({ default: () => <section aria-label="多源下载与ETL">选择数据源与下载内容</section> }))
vi.mock('../components/data-sources/DataStoragePanel', () => ({ default: () => <section aria-label="数据存储位置">选择数据磁盘</section> }))
vi.mock('../components/dashboard/DataHealthRefreshPanel', () => ({ default: () => <section aria-label="原Tushare下载">原模块级全市场任务</section> }))

describe('DataManagement', () => {
  it('默认入口为多源下载及ETL，旧下载面板按需展开', () => {
    render(<MemoryRouter><DataManagement /></MemoryRouter>)
    expect(screen.getByRole('heading', { name: '数据下载与更新' })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: '多源下载与ETL' })).toBeVisible()
    expect(screen.getByRole('region', { name: '数据存储位置' })).toBeVisible()
    expect(screen.getByRole('link', { name: '查看数据质量 →' })).toHaveAttribute('href', '/settings/data-quality')
    expect(screen.queryByRole('region', { name: '原Tushare下载' })).not.toBeInTheDocument()
    const details = screen.getByText('原 Tushare 全市场任务与旧快照维护').closest('details')!
    details.open = true; fireEvent(details, new Event('toggle'))
    expect(screen.getByRole('region', { name: '原Tushare下载' })).toBeVisible()
  })
})
