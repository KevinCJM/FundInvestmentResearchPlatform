import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, expect, it, vi } from 'vitest'
import DataStoragePanel from './DataStoragePanel'
import * as api from '../../services/dataStorage'

vi.mock('../../services/dataStorage')
const status: api.StorageStatus = { revision: 0, online: true, logical_path: '/project/data', actual_path: '/project/data', editing_enabled: true, active: null, pending: null, free_bytes: 3 * 1024 ** 3, total_bytes: 500 * 1024 ** 3, volumes: [{ name: '研究磁盘', path: '/Volumes/研究磁盘', free_bytes: 100 * 1024 ** 3 }] }
const probe: api.StorageProbe = { target: '/Volumes/研究磁盘/FundResearchData', mount: '/Volumes/研究磁盘', free_bytes: 100 * 1024 ** 3, total_bytes: 1000 * 1024 ** 3, reserve_bytes: 2 * 1024 ** 3, same_device: false, message: '目录能力检查通过' }
const pending: api.StoragePlan = { id: 'abc', target: probe.target, mount: probe.mount, phase: 'PLANNED', files: 0, bytes: 0, message: '计划已保存' }
beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(api.getDataStorage).mockResolvedValue(status)
  vi.mocked(api.probeDataStorage).mockResolvedValue(probe)
  vi.mocked(api.planDataStorage).mockResolvedValue({ ...status, revision: 1, pending })
  vi.mocked(api.cancelStoragePlan).mockResolvedValue({ ...status, revision: 2 })
})
async function open() {
  render(<DataStoragePanel />)
  await screen.findByText('/project/data')
  const details = screen.getByText('设置外接磁盘或其他目录').closest('details')!
  details.open = true
  fireEvent.change(screen.getByLabelText('已挂载磁盘'), { target: { value: '/Volumes/研究磁盘' } })
}

it('显示容量及存储范围，选择挂载盘后填入目录，不自动迁移', async () => {
  await open()
  expect(screen.getByLabelText('目标绝对目录')).toHaveValue(probe.target)
  expect(screen.getByText('3.00 GiB / 500.00 GiB')).toBeVisible()
  expect(screen.getByText(/当前磁盘空间偏少/)).toBeVisible()
  expect(api.planDataStorage).not.toHaveBeenCalled()
})

it('先检查、再确认保存，并展示离线重启步骤', async () => {
  await open()
  fireEvent.click(screen.getByText('检查目录'))
  await screen.findByText('目录能力检查通过')
  expect(screen.getByText('保存迁移计划')).toBeDisabled()
  fireEvent.click(screen.getByLabelText('确认保存迁移计划'))
  fireEvent.click(screen.getByText('保存迁移计划'))
  await screen.findByText('./start_services.sh restart')
  expect(api.planDataStorage).toHaveBeenCalledWith(probe.target, 0)
  expect(screen.getByText(/尚未迁移或释放空间/)).toBeVisible()
})

it('修改路径会清除检查结果，过期检查响应不能用于保存', async () => {
  let resolve!: (value: api.StorageProbe) => void
  vi.mocked(api.probeDataStorage).mockImplementation(() => new Promise(r => { resolve = r }))
  await open()
  fireEvent.click(screen.getByText('检查目录'))
  fireEvent.change(screen.getByLabelText('目标绝对目录'), { target: { value: '/Volumes/other/data' } })
  await act(async () => resolve(probe))
  expect(screen.queryByText('保存迁移计划')).not.toBeInTheDocument()
  expect(api.planDataStorage).not.toHaveBeenCalled()
})

it('同一磁盘警告和后端校验错误清晰可见', async () => {
  vi.mocked(api.probeDataStorage).mockResolvedValue({ ...probe, same_device: true })
  await open()
  fireEvent.click(screen.getByText('检查目录'))
  expect(await screen.findByText(/通常不能解决本机容量不足/)).toBeVisible()
  fireEvent.change(screen.getByLabelText('目标绝对目录'), { target: { value: '/missing/data' } })
  vi.mocked(api.probeDataStorage).mockRejectedValue(new Error('请先挂载磁盘'))
  fireEvent.click(screen.getByText('检查目录'))
  expect(await screen.findByRole('alert')).toHaveTextContent('请先挂载磁盘')
})

it('可取消计划但不会触发删除或下载', async () => {
  vi.mocked(api.getDataStorage).mockResolvedValue({ ...status, revision: 1, pending })
  render(<DataStoragePanel />)
  fireEvent.click(await screen.findByText('取消迁移计划（不删除数据）'))
  await waitFor(() => expect(api.cancelStoragePlan).toHaveBeenCalledWith(1))
  expect(await screen.findByText(/计划已取消，原数据与已有暂存文件均未删除/)).toBeVisible()
})

it('掉线显示明确警告，不允许提交新目标', async () => {
  vi.mocked(api.getDataStorage).mockResolvedValue({ ...status, online: false, error: '数据磁盘未挂载' })
  render(<DataStoragePanel />)
  expect(await screen.findByRole('alert')).toHaveTextContent('不会自动写回本机旧副本')
  expect(screen.getByLabelText('目标绝对目录')).toBeDisabled()
})

it('活动目录显示副本及精确清理命令，不提供再次跨盘切换', async () => {
  vi.mocked(api.getDataStorage).mockResolvedValue({ ...status, actual_path: probe.target, active: { id: 'abc', target: probe.target, backup: '/project/.storage/backups/abc/data', backup_removed: false } })
  render(<DataStoragePanel />)
  await screen.findByText(probe.target)
  expect(screen.getByText(/storage-cleanup abc/)).toBeInTheDocument()
  expect(screen.queryByText('设置外接磁盘或其他目录')).not.toBeInTheDocument()
})

it('迁移阶段显示文件和字节进度，切换阶段不能取消', async () => {
  vi.mocked(api.getDataStorage).mockResolvedValue({ ...status, pending: { ...pending, phase: 'SWITCHING', files: 3, bytes: 1024, inventory: { files: 4, bytes: 2048 } } })
  render(<DataStoragePanel />)
  const progress = await screen.findByLabelText('存储迁移进度')
  expect(progress).toHaveAttribute('value', '1024')
  expect(screen.getByText('取消迁移计划（不删除数据）')).toBeDisabled()
})
