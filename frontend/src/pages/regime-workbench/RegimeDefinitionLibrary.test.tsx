import { fireEvent, render, screen, within } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import type { RegimeGraphTemplate } from '../../services/regimeGraph'
import RegimeDefinitionLibrary from './RegimeDefinitionLibrary'
import { reliabilityDefinition } from './regimeReliabilityFixtures'

const templates: RegimeGraphTemplate[] = [
  { id: 'dual', name: '趋势模板', description: '可复用趋势计算',
    default_mode: 'realtime', supported_modes: ['realtime', 'retrospective'], definition: reliabilityDefinition },
  { id: 'history', name: '峰谷模板', description: '依赖后续信息',
    default_mode: 'retrospective', supported_modes: ['retrospective'], definition: reliabilityDefinition },
]
const props = { definitions: [], templates, schemas: [], selectedTemplate: '', loading: false,
  busy: false, onDefinition: vi.fn(), onTemplate: vi.fn(), onLegacy: vi.fn() }

it('历史工作区显示当前用途，不把可复用模板标成实时任务', () => {
  render(<RegimeDefinitionLibrary {...props} fixedMode="retrospective" />)
  const trend = screen.getByRole('button', { name: '选择算法：趋势模板' })
  expect(within(trend).getByText('历史状态定义')).toBeVisible()
  expect(screen.queryByText('实时识别')).not.toBeInTheDocument()
  expect(screen.queryByLabelText('算法库识别方式')).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: '选择算法：峰谷模板' })).toBeVisible()
  fireEvent.click(trend)
  expect(props.onTemplate).toHaveBeenCalledWith('dual')
})

it('实时工作区仍排除只支持事后的模板', () => {
  render(<RegimeDefinitionLibrary {...props} fixedMode="realtime" />)
  expect(screen.queryByRole('button', { name: '选择算法：峰谷模板' })).not.toBeInTheDocument()
  expect(within(screen.getByRole('button', { name: '选择算法：趋势模板' })).getByText('实时识别')).toBeVisible()
})

it('新历史模板直接使用服务端目录，名称与精确 ID 不维护第二份列表', () => {
  const onTemplate = vi.fn()
  render(<RegimeDefinitionLibrary {...props} fixedMode="retrospective" onTemplate={onTemplate} templates={[{ ...templates[1], id: 'server-new-template-r7', name: '服务端新增风险分组', description: '参数在原计算图编辑' }]} />)
  fireEvent.click(screen.getByRole('button', { name: '选择算法：服务端新增风险分组' }))
  expect(onTemplate).toHaveBeenCalledWith('server-new-template-r7')
  expect(screen.queryByRole('button', { name: '选择算法：峰谷模板' })).not.toBeInTheDocument()
})
