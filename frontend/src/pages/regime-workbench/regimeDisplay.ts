import { humanizeIndicatorTechnicalText } from '../../utils/indicatorDiagnostics'
import type { RegimeGraphPortSchema, RegimeParameterSchema } from '../../services/regimeGraph'

const PARAMETER_LABELS: Record<string, string> = {
  rows: '内联数据记录', inline_rows: '内联数据记录', value_field: '数值字段', date_field: '观察日期字段',
  available_at_field: '可得日期字段', vintage_field: '数据版本字段', revision_field: '修订号字段', availability_mode: '数据可得性口径',
  name: '显示名称', artifact_id: '上传数据版本', checksum: '文件校验值', format: '文件格式', ts_code: '指数代码',
  source_api: '行情数据接口', field: '数值字段', frequency: '数据频率', start_date: '开始日期', end_date: '结束日期',
  snapshot_id: '数据快照', snapshot_generation: '快照代次', source_file: '来源文件', file_checksum: '来源文件校验值',
  indicator_id: '指标编号', indicator_revision: '指标修订版本', product_kind: '研究对象类型', product_id: '研究对象编号', period: '计算区间',
  data_fingerprint: '数据指纹', indicator_data_snapshot: '指标数据快照', numerator: '分子序列', denominator: '分母序列', transform: '相对强弱算法',
  dataset: '宏观数据集', series_id: '研究序列', code: '序列代码', value: '常量值', max_age_days: '允许的最长陈旧天数',
  aggregation: '聚合方法', every: '采样间隔', offset: '采样起点偏移', expression: '计算公式', variables: '公式变量映射',
  window: '计算窗口', periods: '比较期数', lower: '下限', upper: '上限', process_variance: '过程噪声方差', measurement_variance: '观测噪声方差',
  upper_enter: '进入上方状态阈值', upper_exit: '退出上方状态阈值', lower_enter: '进入下方状态阈值', lower_exit: '退出下方状态阈值',
  growth_threshold: '增长分界线', inflation_threshold: '通胀分界线', min_move: '最小有效涨跌幅', components: '状态数量',
  initial_train_size: '初始训练样本数', iterations: '最大迭代次数', initialization_strategy: '初始参数生成方法', random_seed: '随机种子',
  initial_means: '显式初始中心', threshold: '识别阈值', confirmation: '确认期数', weights: '候选模型权重', consensus_threshold: '共识门槛',
  min_duration: '最短持续期数', mapping: '模型分量到状态的映射', floor: '最低置信度', declaration: '隔离执行声明', adapter_id: '管理员模型适配器',
}

const PORT_LABELS: Record<string, string> = {
  value: '数值序列', left: '左侧序列', right: '右侧序列', anchor: '基准时间轴', feature: '待对齐特征',
  feature_1: '特征一', feature_2: '特征二', feature_3: '特征三', feature_4: '特征四', features: '特征矩阵',
  growth: '增长指标', inflation: '通胀指标', state: '状态序列', state_1: '候选状态一', state_2: '候选状态二',
  state_3: '候选状态三', state_4: '候选状态四', primary: '优先状态', secondary: '备选状态', score: '状态得分',
  confidence: '置信度', probabilities: '状态概率矩阵', recognition_index: '识别时点', effective_index: '生效时点', reason_code: '判定原因',
}

const TYPE_LABELS: Record<string, string> = {
  'series<float64>': '数值时间序列', 'series<bool>': '布尔时间序列', 'matrix<time,feature>': '时间 × 特征矩阵',
  'state_codes<int64>': '状态编码序列', 'regime_candidate<time,state>': '候选状态结果', 'regime_output<time,state>': '完整情景识别结果',
  'probabilities<time,state>': '时间 × 状态概率矩阵', 'confidence<time>': '置信度时间序列', 'index<time>': '时点索引序列',
  'reason_codes<int64>': '判定原因编码序列',
}

const ENUM_LABELS: Record<string, string> = {
  daily: '日频', weekly: '周频', monthly: '月频', quarterly: '季频', yearly: '年频', first: '区间首值', last: '区间末值',
  mean: '区间平均值', sum: '区间合计值', point_in_time: '时点可得口径', latest: '最新修订口径（仅事后研究）', ratio: '直接比值',
  log_ratio: '对数比值', quantile: '按分位数初始化', random: '按随机种子初始化', explicit: '使用显式初始中心', parquet: 'Parquet 数据文件',
  value: '数值', observation_date: '观察日期', available_at: '可得日期', vintage: '数据版本', revision: '修订号', index_daily: '中证/上证指数日行情',
  sw_daily: '申万指数日行情', ci_daily: '中信指数日行情', ths_daily: '同花顺指数日行情', dc_daily: '东方财富指数日行情', tdx_daily: '通达信指数日行情',
  index_global: '全球指数日行情', fut_index_daily: '股指期货日行情', open: '开盘点位', high: '最高点位', low: '最低点位', close: '收盘点位',
  pre_close: '前收盘点位', change: '涨跌点数', pct_chg: '涨跌幅', vol: '成交量', amount: '成交额', swing: '振幅', turnover_rate: '换手率',
  turnover_rate_f: '自由流通换手率', pe: '市盈率', pe_ttm: '滚动市盈率', pb: '市净率',
}

export function regimeParameterLabel(name: string, schema?: RegimeParameterSchema) {
  return schema?.label || schema?.title || PARAMETER_LABELS[name] || '节点参数'
}

export function regimeParameterHelp(name: string, schema?: RegimeParameterSchema) {
  return schema?.description || `该参数用于控制“${regimeParameterLabel(name, schema)}”的计算口径。修改后需要重新试算才能看到最新结果。`
}

export function regimeEnumLabel(value: unknown, schema?: RegimeParameterSchema, index = -1) {
  return (index >= 0 ? schema?.enum_labels?.[index] : undefined) || ENUM_LABELS[String(value)] || '可选值'
}

export function regimePortLabel(port: RegimeGraphPortSchema) {
  return port.label || PORT_LABELS[port.name || port.id] || '数据端口'
}

export function regimeTypeLabel(value: unknown, serverLabel?: string) {
  const raw = String(value || '')
  return serverLabel || TYPE_LABELS[raw] || humanizeIndicatorTechnicalText(raw) || '未定义数据类型'
}

export function regimePhaseLabel(value?: string) {
  return value === 'P0' ? '基础研究' : value === 'P1' ? '进阶验证' : value || '未标注阶段'
}
