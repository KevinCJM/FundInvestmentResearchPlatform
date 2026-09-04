import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { RegimePreviewRun } from '../../services/regimeGraph'
import RegimeResultDock from './RegimeResultDock'

describe('RegimeResultDock evaluation targets', () => {
  it('只展示试算接口返回的多评价目标摘要，不伪造条件收益', () => {
    const run: RegimePreviewRun = {
      id: 'PREVIEW-1',
      status: 'completed',
      result: {
        row_count: 500,
        evaluation_results: {
          equity: {
            id: 'equity',
            name: '沪深300全收益',
            primary: true,
            snapshot: {
              kind: 'index',
              selected_observations: 500,
              first_observation_date: '2022-01-04',
              last_observation_date: '2023-12-29',
              fingerprint: 'index-fingerprint-from-api',
            },
          },
          commodity: {
            id: 'commodity',
            name: '南华商品指数',
            primary: false,
            snapshot: { kind: 'index', fingerprint: 'commodity-fingerprint-from-api' },
          },
        },
      },
    }

    render(<RegimeResultDock run={run} page={null} nodes={[]} previewNodeId="" loadingSeries={false} height={420} collapsed={false} onPreviewNode={vi.fn()} onLoadSeries={vi.fn()} onHeight={vi.fn()} onCollapsed={vi.fn()} />)

    expect(screen.getByRole('region', { name: '试算评价目标结果' })).toBeInTheDocument()
    expect(screen.getByText('沪深300全收益')).toBeInTheDocument()
    expect(screen.getByText('南华商品指数')).toBeInTheDocument()
    expect(screen.getByText('主要评价目标')).toBeInTheDocument()
    expect(screen.getByText('2022-01-04 → 2023-12-29')).toBeInTheDocument()
    expect(screen.getByText('index-fingerprint-from-api')).toBeInTheDocument()
    expect(screen.getAllByText('试算仅返回评价目标血缘；分状态条件表现由正式运行在服务端计算。')).toHaveLength(2)
    expect(screen.getAllByRole('link', { name: '进入正式实验生成条件表现' })).toHaveLength(2)
    expect(screen.getAllByRole('link', { name: '进入正式实验生成条件表现' })[0]).toHaveAttribute('href', '#regime-formal-experiments')
    expect(screen.queryByRole('table', { name: /分状态条件表现/ })).not.toBeInTheDocument()
  })
})
