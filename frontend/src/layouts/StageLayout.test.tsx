import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, expect, it, vi } from 'vitest'
import StageLayout from './StageLayout'
import { updateAllocationJourney } from '../app/allocationJourney'

vi.mock('../i18n/runtime', () => ({ useI18n: () => ({ s: (key: string) => key }) }))
vi.mock('../i18n/navigation', () => ({ useLocalizedStage: () => ({ id: 'pre-investment', label: '投前研究', eyebrow: '', description: '', path: '/pre-investment', nodes: [], tools: [], accent: { soft: '', text: '' } }) }))
vi.mock('../components/FactorEvidencePanel', () => ({ default: () => null }))
vi.mock('../components/ActualPortfolioSelector', () => ({ default: () => null }))

beforeEach(() => { localStorage.clear(); sessionStorage.clear() })

it('shows the actual selected research and preserves the allocation in a return link', () => {
  updateAllocationJourney({ name: '股债研究甲', universeId: 'u1', allocationName: '60/40', baselineId: 'b1' })
  render(<MemoryRouter initialEntries={['/pre-investment/taa']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  expect(screen.getByText('股债研究甲')).toBeInTheDocument()
  expect(screen.queryByText(/RS-2026-018/)).not.toBeInTheDocument()
  const flow = within(screen.getByRole('navigation', { name: '配置研究流程' }))
  expect(flow.getByRole('link', { name: '3. 长期配置 SAA' })).toHaveAttribute('href', '/pre-investment/saa/policy?baseline=b1&alloc=60%2F40&universe=u1')
  expect(flow.getByRole('link', { name: '4. 战术研究 TAA' })).toHaveAttribute('aria-current', 'step')
  expect(flow.getByRole('link', { name: '1. 产品范围' })).toHaveAttribute('href', '/pre-investment/product-pool?universe=u1')
})

it('does not invent a research name while a different range awaits its metadata', () => {
  updateAllocationJourney({ universeId: 'new-range', baselineId: 'b2' })
  render(<MemoryRouter initialEntries={['/pre-investment/taa']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  expect(screen.getByText('产品范围已载入（名称待确认）')).toBeInTheDocument()
  expect(screen.queryByText('尚未选择产品范围')).not.toBeInTheDocument()
})

it('does not mark downstream steps complete when no product range has been locked', () => {
  localStorage.setItem('allocation-journey:v1', JSON.stringify({ baselineId: 'old-baseline', allocationName: 'orphaned allocation' }))
  render(<MemoryRouter initialEntries={['/pre-investment/product-pool']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  expect(screen.getByText('尚未选择产品范围')).toBeInTheDocument()
  const flow = within(screen.getByRole('navigation', { name: '配置研究流程' }))
  expect(flow.queryByRole('link', { name: '3. 长期配置 SAA' })).not.toBeInTheDocument()
  expect(flow.queryByRole('link', { name: '4. 战术研究 TAA' })).not.toBeInTheDocument()
  expect(flow.queryByText('已保存，可返回')).not.toBeInTheDocument()
})

it('allows independent strategic research without pretending products or mappings exist', () => {
  updateAllocationJourney({ mandateId: 'm1', strategicUniverseId: 's1' })
  render(<MemoryRouter initialEntries={['/pre-investment/product-pool?scope=strategic']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  expect(screen.getByText('战略范围已载入（名称待确认）')).toBeInTheDocument()
  const flow = within(screen.getByRole('navigation', { name: '配置研究流程' }))
  expect(flow.getByRole('link', { name: '3. 长期配置 SAA' })).toHaveAttribute('href', '/pre-investment/saa/policy?mandate=m1&strategic_universe=s1')
  expect(flow.getByRole('link', { name: '2. 产品映射' })).toHaveAttribute('href', '/pre-investment/product-pool?mandate=m1&strategic_universe=s1&scope=strategic')
  expect(flow.queryByRole('link', { name: '4. 战术研究 TAA' })).not.toBeInTheDocument()
})

it('highlights saved strategic policy and only links TAA after an explicit mapping reference', () => {
  updateAllocationJourney({ strategicUniverseId: 's1', baselineId: 'b1', implementationMappingId: 'map1' })
  render(<MemoryRouter initialEntries={['/pre-investment/saa/policy?baseline=b1']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  const flow = within(screen.getByRole('navigation', { name: '配置研究流程' }))
  expect(flow.getByRole('link', { name: '3. 长期配置 SAA' })).toHaveAttribute('aria-current', 'step')
  expect(flow.getByRole('link', { name: '4. 战术研究 TAA' })).toHaveAttribute('href', '/pre-investment/taa?baseline=b1')
  expect(flow.queryByRole('link', { name: '5. 产品配置' })).not.toBeInTheDocument()
})
