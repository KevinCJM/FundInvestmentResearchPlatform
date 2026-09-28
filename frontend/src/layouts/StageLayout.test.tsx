import { act, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, expect, it, vi } from 'vitest'
import StageLayout from './StageLayout'
import { updateAllocationJourney } from '../app/allocationJourney'

// 不再假造 stage：流程条的名字和编号就是从真实注册表推出来的，假 nodes 会让这些断言失去意义。
vi.mock('../i18n/runtime', async (importOriginal) => ({
  ...await importOriginal<typeof import('../i18n/runtime')>(),
  useI18n: () => ({ s: (key: string) => key }),
}))
vi.mock('../components/ActualPortfolioSelector', () => ({ default: () => null }))

beforeEach(() => { localStorage.clear(); sessionStorage.clear() })

it('shows the actual selected research and preserves the allocation in a return link', () => {
  updateAllocationJourney({ name: '股债研究甲', mandateId: 'm1', universeId: 'u1', allocationName: '60/40', baselineId: 'b1' })
  render(<MemoryRouter initialEntries={['/pre-investment/taa?baseline=b1']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  expect(screen.getByText('股债研究甲')).toBeInTheDocument()
  expect(screen.queryByText(/RS-2026-018/)).not.toBeInTheDocument()
  const flow = within(screen.getByRole('navigation', { name: 'navigation.allocationFlow' }))
  expect(flow.getByRole('link', { name: '04 战略资产配置（SAA）' })).toHaveAttribute('href', '/pre-investment/saa/policy?baseline=b1&alloc=60%2F40&universe=u1&mandate=m1')
  expect(flow.getByRole('link', { name: '05 战术资产配置（TAA）' })).toHaveAttribute('aria-current', 'step')
  expect(flow.getByRole('link', { name: '02 选择研究路径与范围' })).toHaveAttribute('href', '/pre-investment/product-pool/new?universe=u1&mandate=m1')
})

it('does not invent a research name while a different range awaits its metadata', () => {
  updateAllocationJourney({ universeId: 'new-range', baselineId: 'b2' })
  render(<MemoryRouter initialEntries={['/pre-investment/taa?baseline=b2']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  expect(screen.getByText('产品范围已载入（名称待确认）')).toBeInTheDocument()
  expect(screen.queryByText('尚未选择产品范围')).not.toBeInTheDocument()
})

it('does not mark downstream steps complete when no product range has been locked', () => {
  localStorage.setItem('allocation-journey:v1', JSON.stringify({ baselineId: 'old-baseline', allocationName: 'orphaned allocation' }))
  render(<MemoryRouter initialEntries={['/pre-investment/product-pool']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  expect(screen.getByText('尚未选择产品范围')).toBeInTheDocument()
  const flow = within(screen.getByRole('navigation', { name: 'navigation.allocationFlow' }))
  expect(flow.queryByRole('link', { name: '04 战略资产配置（SAA）' })).not.toBeInTheDocument()
  expect(flow.queryByRole('link', { name: '05 战术资产配置（TAA）' })).not.toBeInTheDocument()
  expect(flow.queryByText('已保存，可返回')).not.toBeInTheDocument()
})

it('allows independent strategic research without pretending products or mappings exist', () => {
  updateAllocationJourney({ mandateId: 'm1', strategicUniverseId: 's1' })
  render(<MemoryRouter initialEntries={['/pre-investment/product-pool?mandate=m1&strategic_universe=s1&scope=strategic']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  expect(screen.getByText('战略范围已载入（名称待确认）')).toBeInTheDocument()
  const flow = within(screen.getByRole('navigation', { name: 'navigation.allocationFlow' }))
  expect(flow.queryByRole('link', { name: '04 战略资产配置（SAA）' })).not.toBeInTheDocument()
  expect(flow.getByLabelText('04 战略资产配置（SAA） · navigation.journeyBlockedLtcma')).toBeInTheDocument()
  expect(flow.getByRole('link', { name: 'navigation.journeyMapping' })).toHaveAttribute('href', '/pre-investment/product-pool/new?mandate=m1&strategic_universe=s1&scope=strategic')
  expect(flow.queryByRole('link', { name: '05 战术资产配置（TAA）' })).not.toBeInTheDocument()
  act(() => updateAllocationJourney({ ltcmaId: 'cma-7' }))
  expect(flow.getByRole('link', { name: '04 战略资产配置（SAA）' })).toHaveAttribute('href', '/pre-investment/saa/policy?mandate=m1&cma=cma-7&strategic_universe=s1')
})

it('links saved strategic policy to TAA without product mapping and keeps product handoff gated', () => {
  updateAllocationJourney({ strategicUniverseId: 's1', baselineId: 'b1' })
  render(<MemoryRouter initialEntries={['/pre-investment/saa/policy?baseline=b1']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  const flow = within(screen.getByRole('navigation', { name: 'navigation.allocationFlow' }))
  expect(flow.getByRole('link', { name: '04 战略资产配置（SAA）' })).toHaveAttribute('aria-current', 'step')
  expect(flow.getByRole('link', { name: '05 战术资产配置（TAA）' })).toHaveAttribute('href', '/pre-investment/taa?baseline=b1')
  expect(flow.queryByRole('link', { name: '06 产品配置与择时' })).not.toBeInTheDocument()
})

it('顶部流程条与侧栏用同一套名字和编号，工具步不编号', () => {
  updateAllocationJourney({ mandateId: 'm1', universeId: 'u1', allocationName: '60/40', ltcmaId: 'cma-7', baselineId: 'b1', taaRunId: 't1' })
  localStorage.setItem('allocation-draft:v1:products:u1:t1', JSON.stringify({ name: '已交接' }))
  render(<MemoryRouter initialEntries={['/pre-investment/ltcma/cma-7']}><StageLayout stageId="pre-investment" /></MemoryRouter>)

  const flow = within(screen.getByRole('navigation', { name: 'navigation.allocationFlow' }))
  expect(flow.getAllByRole('link').map((link) => link.getAttribute('aria-label'))).toEqual([
    '01 投资目标与约束', '02 选择研究路径与范围', '大类资产构建', '03 LTCMA 中心', '04 战略资产配置（SAA）', '05 战术资产配置（TAA）', '06 产品配置与择时',
  ])
  // 同一个节点在侧栏里是第几个，顶部就印第几个；两处不再各数各的。
  const sidebar = within(screen.getByRole('complementary', { name: 'navigation.subpages' }))
  expect(sidebar.getByRole('link', { name: '04 战略资产配置（SAA）' })).toHaveTextContent('04')
  expect(sidebar.getByRole('link', { name: /LTCMA/ })).toHaveTextContent('03')
})

it.each(['/pre-investment', '/pre-investment/', '/pre-investment?baseline=b1'])(
  '总览 %s 不把上次研究显示为当前方案或进度，也不清除书签', (path) => {
    updateAllocationJourney({ name: '基础四类配置', mandateId: 'm1', strategicUniverseId: 's1', ltcmaId: 'cma1', baselineId: 'b1', researchDate: '2019-12-31' })
    const saved = localStorage.getItem('allocation-journey:v1')
    const active = sessionStorage.getItem('allocation-journey:v1')
    render(<MemoryRouter initialEntries={[path]}><StageLayout stageId="pre-investment" /></MemoryRouter>)
    expect(screen.getByRole('heading', { level: 1, name: '投前决策' })).toBeInTheDocument()
    expect(screen.queryByText('基础四类配置')).not.toBeInTheDocument()
    expect(screen.queryByText('尚未选择产品范围')).not.toBeInTheDocument()
    expect(screen.queryByRole('navigation', { name: 'navigation.allocationFlow' })).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'navigation.resumeAction' })).not.toBeInTheDocument()
    const sidebar = within(screen.getByRole('complementary', { name: 'navigation.subpages' }))
    expect(sidebar.getByRole('link', { name: '04 战略资产配置（SAA）' })).toHaveAttribute('href', '/pre-investment/saa')
    expect(localStorage.getItem('allocation-journey:v1')).toBe(saved)
    expect(sessionStorage.getItem('allocation-journey:v1')).toBe(active)
  },
)

it('上游没选投资目标，模块内的范围步就不可点，并说明卡在哪', () => {
  render(<MemoryRouter initialEntries={['/pre-investment/objectives']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  const flow = within(screen.getByRole('navigation', { name: 'navigation.allocationFlow' }))
  expect(flow.getByRole('link', { name: '01 投资目标与约束' })).toBeInTheDocument()
  expect(flow.queryByRole('link', { name: '02 选择研究路径与范围' })).not.toBeInTheDocument()
  expect(flow.getByLabelText('02 选择研究路径与范围 · navigation.journeyBlockedMandate')).toBeInTheDocument()
})

it('选定 LTCMA 之后 03 才算完成，并把那一版假设带回 SAA', () => {
  updateAllocationJourney({ mandateId: 'm1', universeId: 'u1', allocationName: '60/40', ltcmaId: 'cma-7' })
  render(<MemoryRouter initialEntries={['/pre-investment/saa/policy?alloc=60%2F40&universe=u1&mandate=m1&cma=cma-7']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  const flow = within(screen.getByRole('navigation', { name: 'navigation.allocationFlow' }))
  // 已选的版本直接回到那一版，不是回中心页重找。
  expect(flow.getByRole('link', { name: '03 LTCMA 中心' })).toHaveAttribute('href', '/pre-investment/ltcma/cma-7')
  expect(flow.getByRole('link', { name: '04 战略资产配置（SAA）' })).toHaveAttribute('href', '/pre-investment/saa/policy?alloc=60%2F40&universe=u1&mandate=m1&cma=cma-7')
})

it('产品大类尚未确认 LTCMA 时 03 未完成、04 不放行', () => {
  updateAllocationJourney({ mandateId: 'm1', universeId: 'u1', allocationName: '60/40' })
  render(<MemoryRouter initialEntries={['/pre-investment/product-pool/new?universe=u1&mandate=m1']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  const flow = within(screen.getByRole('navigation', { name: 'navigation.allocationFlow' }))
  const ltcma = flow.getByRole('link', { name: '03 LTCMA 中心' })
  expect(ltcma).toHaveAttribute('href', '/pre-investment/ltcma')
  expect(ltcma).toHaveTextContent('navigation.journeyContinue')
  expect(flow.queryByRole('link', { name: '04 战略资产配置（SAA）' })).not.toBeInTheDocument()
  expect(flow.getByLabelText('04 战略资产配置（SAA） · navigation.journeyBlockedLtcma')).toBeInTheDocument()
})

it('TAA 存了但没交接时，产品步不可点；交接过就点得开', () => {
  updateAllocationJourney({ mandateId: 'm1', universeId: 'u1', allocationName: '60/40', baselineId: 'b1', taaRunId: 't1' })
  const { unmount } = render(<MemoryRouter initialEntries={['/pre-investment/taa?decision=t1']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  const blocked = within(screen.getByRole('navigation', { name: 'navigation.allocationFlow' }))
  expect(blocked.queryByRole('link', { name: '06 产品配置与择时' })).not.toBeInTheDocument()
  expect(blocked.getByLabelText('06 产品配置与择时 · navigation.journeyHandoff')).toBeInTheDocument()
  unmount()

  localStorage.setItem('allocation-draft:v1:products:u1:t1', JSON.stringify({ name: '已交接' }))
  render(<MemoryRouter initialEntries={['/pre-investment/taa?decision=t1']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  const ready = within(screen.getByRole('navigation', { name: 'navigation.allocationFlow' }))
  expect(ready.getByRole('link', { name: '06 产品配置与择时' })).toHaveAttribute('href', '/pre-investment/product-allocation-timing/construction?universe=u1&decision=t1')
})

it('02 的续接目标是独立的编辑页，裸列表页不重复给续接条——列表本身已经是完整入口', () => {
  // 02 拆分列表/编辑页后和 01 同构：续接目标指向 .../new，和当前裸列表页 pathname 对不上，
  // ResumeResearch 的同页匹配就不会触发；已保存范围库本身就是完整的“继续研究”入口，不需要再提示一次。
  updateAllocationJourney({ mandateId: 'm1', universeId: 'u1' })
  render(<MemoryRouter initialEntries={['/pre-investment/product-pool']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  expect(screen.queryByRole('link', { name: 'navigation.resumeAction' })).not.toBeInTheDocument()
})

it('地址栏已经带齐身份时不再重复提示续接', () => {
  updateAllocationJourney({ mandateId: 'm1', universeId: 'u1' })
  render(<MemoryRouter initialEntries={['/pre-investment/product-pool?universe=u1&mandate=m1']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  expect(screen.queryByRole('link', { name: 'navigation.resumeAction' })).not.toBeInTheDocument()
})

it('从侧栏裸路径进入 01 时，流程条不拿上次的书签解锁下游', () => {
  updateAllocationJourney({ name: '股债研究甲', mandateId: 'm1', universeId: 'u1' })
  render(<MemoryRouter initialEntries={['/pre-investment/objectives']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  expect(screen.getByText('尚未选择产品范围')).toBeInTheDocument()
  const flow = within(screen.getByRole('navigation', { name: 'navigation.allocationFlow' }))
  expect(flow.queryByRole('link', { name: '02 选择研究路径与范围' })).not.toBeInTheDocument()
  expect(flow.getByLabelText('02 选择研究路径与范围 · navigation.journeyBlockedMandate')).toBeInTheDocument()
  // 01 自己也回到列表页重新选，而不是悄悄带回上次那一版。
  expect(flow.getByRole('link', { name: '01 投资目标与约束' })).toHaveAttribute('href', '/pre-investment/objectives')
})

it('地址栏选定目标之后 02 才解锁，并带着该目标继续', () => {
  updateAllocationJourney({ name: '股债研究甲', mandateId: 'm1', universeId: 'u1' })
  render(<MemoryRouter initialEntries={['/pre-investment/objectives/new?view=m1']}><StageLayout stageId="pre-investment" /></MemoryRouter>)
  expect(screen.getByText('股债研究甲')).toBeInTheDocument()
  const flow = within(screen.getByRole('navigation', { name: 'navigation.allocationFlow' }))
  expect(flow.getByRole('link', { name: '02 选择研究路径与范围' })).toHaveAttribute('href', '/pre-investment/product-pool/new?universe=u1&mandate=m1')
})
