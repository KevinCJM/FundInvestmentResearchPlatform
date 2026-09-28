import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Badge, Button, DataTable, EmptyState, SectionHeader, actionClass, type TableColumn, ErrorPanel } from '../../components/ui'
import { formatDate } from '../../i18n/runtime'
import {
  listHistoricalReferences,
  listRegimeFormalRuns,
  listRegimeGraphDefinitions,
  listRegimeReliability,
  type HistoricalReference,
  type RegimeFormalRun,
  type RegimeGraphDefinition,
  type RegimeReliabilityCatalogItem,
} from '../../services/regimeGraph'
import { referenceKey } from './RegimeReferenceBinding'
import { studyBelongsToPurpose, studyMode, type MarketStateStage } from './regimeStudy'
import { isManualEventDefinition } from './regimeWorkspace'

const COPY: Record<MarketStateStage, { title: string; description: string; action?: string; empty: string; hint: string }> = {
  historical: {
    title: '定义历史参考',
    description: '用完整历史数据定义可解释、可复用的市场状态参考。已保存的研究都在这里，点名称继续；确认为历史参考后才能被实时模型引用。',
    action: '新建历史参考算法',
    empty: '还没有历史参考算法',
    hint: '新建一个，从内置算法或空白画布开始；跑出历史区间并确认后，它才会成为可被引用的历史参考。',
  },
  realtime: {
    title: '建立实时识别',
    description: '先固定一份历史参考，再用当时可得信息识别同一组状态。已保存的实时识别模型都在这里。',
    action: '新建实时识别模型',
    empty: '还没有实时识别模型',
    hint: '新建一个，先选定历史参考，再建立只用当时可得信息的识别规则。',
  },
  validation: {
    title: '验证识别能力与应用',
    description: '检查准确性、校准与前瞻证据，再判断可用范围。选择一个已保存的实时识别模型开始验证。',
    empty: '还没有可验证的实时识别模型',
    hint: '先在「建立实时识别」保存一个模型，再回到这里验证它是否真的能识别同一组状态。',
  },
}

interface StudyRow {
  definition: RegimeGraphDefinition
  runCount: number
  latestVersionRun?: RegimeFormalRun
  published: boolean
  reference?: HistoricalReference
  report?: RegimeReliabilityCatalogItem
}

function statusBadge(stage: MarketStateStage, row: StudyRow) {
  if (stage === 'validation') {
    if (!row.report) return <Badge>尚未验证</Badge>
    return row.report.verification?.recognition_ready
      ? <Badge tone="success">已验证可识别</Badge>
      : <Badge tone="warning">已出报告 · 未达可识别</Badge>
  }
  if (stage === 'historical') {
    if (row.reference) return <Badge tone="success">已确认参考</Badge>
    return row.runCount ? <Badge tone="warning">已运行 · 未确认参考</Badge> : <Badge>尚未运行</Badge>
  }
  if (row.published) return <Badge tone="success">已发布</Badge>
  return row.runCount ? <Badge tone="warning">已运行 · 未发布</Badge> : <Badge>尚未运行</Badge>
}

/**
 * 市场状态研究每一步的已保存清单。中心路由默认停在这里，带上 definition / template / new
 * 这类身份参数才进入工作台，避免一进来就是编辑器。
 */
export default function RegimeStudyList({ stage, studyHref, newHref }: {
  stage: MarketStateStage
  /** 打开某一条已保存研究的查询串。 */
  studyHref: (definition: RegimeGraphDefinition) => string
  /** 空白新建的查询串；验证步骤不新建研究，不传。 */
  newHref?: string
}) {
  const copy = COPY[stage]
  const purpose = stage === 'historical' ? 'historical_reference' : 'realtime_recognition'
  const [definitions, setDefinitions] = useState<RegimeGraphDefinition[]>([])
  const [runs, setRuns] = useState<RegimeFormalRun[]>([])
  const [runsFailed, setRunsFailed] = useState(false)
  const [references, setReferences] = useState<HistoricalReference[]>([])
  const [reports, setReports] = useState<RegimeReliabilityCatalogItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    let active = true
    setLoading(true)
    void Promise.allSettled([
      listRegimeGraphDefinitions(controller.signal),
      listRegimeFormalRuns(undefined, controller.signal),
      listHistoricalReferences(controller.signal),
      stage === 'validation' ? listRegimeReliability(controller.signal) : Promise.resolve([] as RegimeReliabilityCatalogItem[]),
    ]).then(results => {
      if (!active) return
      if (results[0].status === 'fulfilled') setDefinitions(results[0].value)
      if (results[1].status === 'fulfilled') setRuns(results[1].value)
      setRunsFailed(results[1].status === 'rejected')
      if (results[2].status === 'fulfilled') setReferences(results[2].value)
      if (results[3].status === 'fulfilled') setReports(results[3].value)
      // 只有清单本身读不到才算错误：运行摘要、参考目录缺失时清单仍然可用，状态列退化为未知。
      setError(results[0].status === 'rejected'
        ? (results[0].reason instanceof Error ? results[0].reason.message : '已保存研究读取失败。')
        : '')
      setLoading(false)
    })
    return () => { active = false; controller.abort() }
  }, [stage, retry])

  const rows = useMemo<StudyRow[]>(() => {
    const runsByDefinition = new Map<string, RegimeFormalRun[]>()
    for (const run of runs) {
      if (!run.definition_id) continue
      runsByDefinition.set(run.definition_id, [...(runsByDefinition.get(run.definition_id) ?? []), run])
    }
    const reportsByDefinition = new Map<string, RegimeReliabilityCatalogItem[]>()
    for (const report of reports) {
      reportsByDefinition.set(report.definition_id, [...(reportsByDefinition.get(report.definition_id) ?? []), report])
    }
    return definitions
      .filter(item => item.id && !isManualEventDefinition(item) && studyBelongsToPurpose(item, purpose))
      .map(item => {
        const own = (runsByDefinition.get(item.id ?? '') ?? []).sort((left, right) => String(right.created_at).localeCompare(String(left.created_at)))
        const ownReports = (reportsByDefinition.get(item.id ?? '') ?? []).sort((left, right) => right.created_at.localeCompare(left.created_at))
        const key = referenceKey(item.study?.reference)
        return {
          definition: item,
          runCount: own.length,
          latestVersionRun: own.find(run => run.definition_revision === item.revision && run.mode === studyMode(purpose)),
          published: own.some(run => (run.publications ?? []).length > 0),
          reference: stage === 'historical'
            ? references.find(entry => entry.definition_id === item.id && entry.definition_revision === item.revision)
            : references.find(entry => referenceKey(entry) === key),
          report: ownReports[0],
        }
      })
      .sort((left, right) => String(right.definition.updated_at || right.definition.created_at || '')
        .localeCompare(String(left.definition.updated_at || left.definition.created_at || '')))
  }, [definitions, runs, references, reports, purpose, stage])

  const columns: TableColumn<StudyRow>[] = [
    {
      header: '名称',
      cell: row => <div className="min-w-64">
        <Link className="font-semibold text-accent-700 hover:underline" to={{ search: studyHref(row.definition) }}>{row.definition.name || '未命名研究'}</Link>
        <p className="mt-1 line-clamp-2 text-xs leading-5 text-slate-600">{row.definition.description || '尚未填写研究说明。'}</p>
      </div>,
    },
    { header: stage === 'validation' ? '验证状态' : '状态', nowrap: true, cell: row => statusBadge(stage, row) },
    stage === 'historical'
      ? { header: '研究结构', nowrap: true, cell: row => <span className="text-xs text-slate-600">{row.definition.states.length} 个状态 · {row.definition.graph.nodes.length} 个节点</span> }
      : {
        header: '历史参考',
        cell: row => row.reference
          ? <span className="text-xs text-slate-600">{row.reference.name} · v{row.reference.definition_revision}</span>
          : <span className="text-xs text-slate-600">{row.definition.study?.reference ? '绑定的参考不在可用目录中' : '未绑定参考'}</span>,
      },
    { header: '版本', numeric: true, nowrap: true, cell: row => <span className="inline-block min-w-6">r{row.definition.revision ?? 1}</span> },
    {
      header: 'PIT 日期', nowrap: true,
      cell: row => {
        if (runsFailed) return '读取失败'
        const run = row.latestVersionRun
        if (!run) return '当前版本尚未运行'
        if (run.as_of === undefined) return '未记录'
        if (!run.as_of) return '未设置'
        return <time dateTime={run.as_of}>{formatDate(run.as_of)}</time>
      },
    },
    { header: '正式运行', numeric: true, cell: row => <span className="inline-block min-w-12">{row.runCount}</span> },
    { header: '最近更新', nowrap: true, cell: row => formatDate(row.definition.updated_at || row.definition.created_at) },
    {
      header: '操作',
      nowrap: true,
      cell: row => <Link className={actionClass('secondary')} to={{ search: studyHref(row.definition) }}>{stage === 'validation' ? '去验证' : '继续研究'}</Link>,
    },
  ]

  return <section className="min-w-0 space-y-4" aria-label={copy.title} data-testid="regime-study-list">
    <SectionHeader
      title={copy.title}
      description={copy.description}
      actions={newHref && copy.action
        ? <Link className={actionClass('primary')} to={{ search: newHref }}>{copy.action}</Link>
        : undefined}
    />
    <p className="text-xs leading-5 text-slate-600">PIT 日期是当前版本最近一次正式运行的数据截止日，不是保存日期，也不代表已通过时点可用性验证。</p>
    {!loading && !error && runsFailed && <div role="alert" className="flex flex-wrap items-center gap-2 text-sm text-rose-800">
      <span>正式运行记录读取失败，暂时无法显示 PIT 日期。</span>
      <Button onClick={() => setRetry(value => value + 1)}>重试读取 PIT 日期</Button>
    </div>}
    {!loading && error ? <ErrorPanel onRetry={() => setRetry(value => value + 1)} retryLabel="重试读取" /> : !loading && !rows.length
      ? <EmptyState
        title={copy.empty}
        hint={copy.hint}
        action={newHref && copy.action ? <Link className={actionClass('primary')} to={{ search: newHref }}>{copy.action}</Link> : undefined}
      />
      : <DataTable
        caption={`${copy.title}的已保存研究`}
        minWidth="1000px"
        loading={loading ? '正在读取已保存研究…' : undefined}
        empty={copy.hint}
        columns={columns}
        rows={rows}
        rowKey={row => row.definition.id ?? row.definition.name}
      />}
  </section>
}
