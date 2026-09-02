import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ManualConstruction from './ManualConstruction'

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="chart" /> }))

function LocationProbe() {
  const location = useLocation()
  const params = new URLSearchParams(location.search)
  const state = location.state as { returnTo?: string; returnLabel?: string } | null
  return (
    <div data-testid="location-probe">
      {location.pathname}|{params.get('ids')}|{params.get('kinds')}|{state?.returnTo}|{state?.returnLabel}
    </div>
  )
}

describe('ManualConstruction product navigation', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url === '/api/list-allocations') {
        return { ok: true, json: async () => [] }
      }
      if (url.startsWith('/api/instruments/search?')) {
        const params = new URL(url, 'http://localhost').searchParams
        const keyword = params.get('q') ?? ''
        const items = keyword === '沪深300'
          ? [
              { code: '159393.SZ', name: '万家沪深300ETF', instrument_type: 'etf', management: '万家基金' },
              { code: '024011.OF', name: '万家沪深300ETF联接-A', instrument_type: 'fund', management: '万家基金' },
            ]
          : []
        return { ok: true, json: async () => ({ items, total: items.length }) }
      }
      return { ok: false, status: 404, json: async () => ({}) }
    }))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('产品名称进入对应类别的产品研究，并可对比同一大类的全部产品', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter initialEntries={['/manual-construction']}>
        <Routes>
          <Route path="/manual-construction" element={<ManualConstruction />} />
          <Route path="/product-compare" element={<LocationProbe />} />
        </Routes>
      </MemoryRouter>,
    )

    const etfLink = await screen.findByRole('link', { name: '万家沪深300ETF' })
    const fundLink = screen.getByRole('link', { name: '万家沪深300ETF联接-A' })
    expect(etfLink).toHaveAttribute('href', '/product/159393.SZ?kind=etf')
    expect(fundLink).toHaveAttribute('href', '/product/024011.OF?kind=fund')

    await act(async () => {
      await user.click(screen.getByRole('button', { name: '产品对比（2）' }))
    })
    await waitFor(() => expect(screen.getByTestId('location-probe')).toHaveTextContent(
      '/product-compare|159393.SZ,024011.OF|etf,fund|/manual-construction|返回手动构建大类',
    ))
  })
})
