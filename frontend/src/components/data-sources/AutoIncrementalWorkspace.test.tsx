import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, afterEach, expect, it, vi } from 'vitest'
import AutoIncrementalWorkspace, { autoDefinition } from './AutoIncrementalWorkspace'
import EtlRunOptionsEditor from './EtlRunOptionsEditor'
import * as api from '../../services/etl'
import type { EtlTaskSpec, SourceCatalog } from '../../services/dataSources'

vi.mock('../../services/etl', async original => ({ ...await original<typeof import('../../services/etl')>(), validateEtl: vi.fn(), runEtl: vi.fn() }))
const tasks = [
  { id: 'tushare.fund_info', name: '基金目录', requires: [], provides: ['fund_info'] },
  { id: 'tushare.calendar', name: '交易日历', requires: [], provides: ['calendar'] },
  { id: 'tushare.fund_nav', name: '场外公募基金净值', requires: ['fund_info', 'calendar'], provides: ['fund_nav'] },
].map(t => ({ ...t, auto_incremental_supported: true, source_ids: ['tushare'], requires_source: true, parameters: [], category: '公募基金', description: '', output_type: 'market_files_v1', network: true })) as EtlTaskSpec[]
const catalog = { etl_tasks: tasks } as SourceCatalog
const plan: api.AutoIncrementalPlan = { plan_id: 'frozen', snapshot: 'snapshot_1', cutoff_date: '2026-09-06', lookback_trade_days: 5, ready: true, errors: [],
  steps: [{ id: 'auto_2', name: '场外公募基金净值', strategy: 'incremental', latest_date: '2026-09-03', start_date: '20260828', end_date: '20260906', message: '回查修订并补充新日期' }] }
beforeEach(() => { vi.clearAllMocks(); vi.spyOn(window, 'confirm').mockReturnValue(true); vi.mocked(api.validateEtl).mockResolvedValue({ valid: true, errors: [], steps: [], auto_plan: plan }) })
afterEach(() => { vi.restoreAllMocks() })

it('自动展开目录与日历依赖，不要求填写日期', async () => {
  render(<AutoIncrementalWorkspace catalog={catalog} canRun onStarted={vi.fn()} />)
  expect(document.querySelector('input[type="date"]')).toBeNull()
  fireEvent.click(screen.getByLabelText('场外公募基金净值'))
  fireEvent.click(screen.getByRole('button', { name: '分析快照并预览区间' }))
  await screen.findByText('基线快照：snapshot_1')
  const definition = vi.mocked(api.validateEtl).mock.calls[0][0]
  expect(definition.steps.map(s => s.task_id)).toEqual(['tushare.fund_info', 'tushare.calendar', 'tushare.fund_nav'])
  expect(api.validateEtl).toHaveBeenCalledWith(definition, { mode: 'auto_incremental', parameters: {} })
  expect(screen.getByText(/2026-08-28 至 2026-09-06/)).toBeInTheDocument()
})

it('仅确认已预览计划才启动，并携带基线计划 ID', async () => {
  const started = vi.fn()
  const result = { run_id: 'run_1' } as api.EtlRun
  vi.mocked(api.runEtl).mockResolvedValue(result)
  render(<AutoIncrementalWorkspace catalog={catalog} canRun onStarted={started} />)
  fireEvent.click(screen.getByLabelText('场外公募基金净值'))
  expect(screen.queryByText('确认计划并开始自动增量')).not.toBeInTheDocument()
  fireEvent.click(screen.getByText('分析快照并预览区间'))
  fireEvent.click(await screen.findByText('确认计划并开始自动增量'))
  await waitFor(() => expect(api.runEtl).toHaveBeenCalledWith(expect.anything(), expect.any(String), { mode: 'auto_incremental', parameters: {} }, 'frozen'))
  expect(started).toHaveBeenCalledWith(result)
})

it('修改数据集立即清除过期计划，基线缺失不能启动', async () => {
  render(<AutoIncrementalWorkspace catalog={catalog} canRun onStarted={vi.fn()} />)
  fireEvent.click(screen.getByLabelText('场外公募基金净值'))
  fireEvent.click(screen.getByText('分析快照并预览区间'))
  await screen.findByText('确认计划并开始自动增量')
  fireEvent.click(screen.getByLabelText('基金目录'))
  expect(screen.queryByText('确认计划并开始自动增量')).not.toBeInTheDocument()
  vi.mocked(api.validateEtl).mockResolvedValue({ valid: false, errors: [{ code: 'BASE', message: '尚未初始化，不会改成全量' }], steps: [] })
  fireEvent.click(screen.getByText('分析快照并预览区间'))
  await screen.findByRole('alert')
  expect(api.runEtl).not.toHaveBeenCalled()
})

it('另一个任务执行期间允许只读预览，禁止新下载', async () => {
  render(<AutoIncrementalWorkspace catalog={catalog} canRun={false} onStarted={vi.fn()} />)
  fireEvent.click(screen.getByLabelText('场外公募基金净值'))
  fireEvent.click(screen.getByText('分析快照并预览区间'))
  expect(await screen.findByText('确认计划并开始自动增量')).toBeDisabled()
})

it('ETL 自动模式隐藏运行日期，保留其他参数', () => {
  const definition = autoDefinition(tasks, ['tushare.fund_nav'])
  definition.parameters = [{ id: 'end', label: '截止日', data_type: 'date', default: '', date_format: 'compact', required: true, description: '' }]
  render(<EtlRunOptionsEditor definition={definition} value={{ mode: 'auto_incremental', parameters: {} }} onChange={vi.fn()} disabled={false} />)
  expect(screen.queryByLabelText('截止日')).not.toBeInTheDocument()
  expect(screen.getByLabelText('本次运行模式')).toHaveValue('auto_incremental')
})
