import { act, cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it } from 'vitest'
import { getStage } from '../app/processRegistry'
import { chooseLocale } from '../i18n/runtime'
import StageOverview from './StageOverview'

afterEach(async () => { cleanup(); await act(() => chooseLocale('zh-CN')) })

it('connects every registered step in order, preserving identity-free module entries', () => {
  render(<MemoryRouter><StageOverview stageId="pre-investment" /></MemoryRouter>)
  const links = within(screen.getByRole('list', { name: '投前研究流程' })).getAllByRole('link')
  const nodes = getStage('pre-investment').nodes
  expect(links.map(link => link.getAttribute('href'))).toEqual(nodes.map(node => node.path))
  nodes.forEach((node, index) => expect(links[index]).toHaveAccessibleName(`${String(index + 1).padStart(2, '0')} ${node.label}`))
  expect(screen.getByText(/先确认长期收益与风险，再研究配置比例/)).toBeVisible()
  expect(screen.getByText(/不做战术调整时，可从 SAA 进入产品配置/)).toBeVisible()
  expect(screen.getByRole('link', { name: '查看已保存 SAA' })).toHaveAttribute('href', '/pre-investment/saa')
  expect(screen.queryByText(/已保存，可返回|当前步骤|部分实现/)).not.toBeInTheDocument()
})

it('localizes workflow explanations and registry names together', async () => {
  await act(() => chooseLocale('en-US'))
  render(<MemoryRouter><StageOverview stageId="pre-investment" /></MemoryRouter>)
  const flow = screen.getByRole('list', { name: 'Pre-investment research workflow' })
  expect(within(flow).getAllByRole('link')).toHaveLength(9)
  expect(screen.getByText('Optional')).toBeVisible()
  expect(screen.getByText('How do the two research paths converge?')).toBeVisible()
  expect(screen.queryByText(/名称待补充|Text unavailable/)).not.toBeInTheDocument()
})
