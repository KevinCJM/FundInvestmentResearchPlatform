const DIAGNOSTIC_TITLES: Record<string, string> = {
  ARITY_MISMATCH: '参数数量不正确',
  AXIS_MISMATCH: '数据维度不匹配',
  COMBINATION_LIMIT_EXCEEDED: '计算数量超过上限',
  COMPUTE_BUDGET_EXCEEDED: '预计计算量过大',
  CONTEXT_KIND_MISMATCH: '指标与计算域不匹配',
  CONTEXT_VARIABLE_UNAVAILABLE: '计算所需变量不可用',
  DATA_NOT_FOUND: '未找到数据',
  DIVIDE_BY_ZERO: '计算中出现除零',
  DOMAIN_ERROR: '数值不在算子允许范围内',
  EMPTY_EXPRESSION: '还没有输入指标公式',
  FORMULA_TOO_COMPLEX: '公式结构过于复杂',
  FORMULA_TOO_LONG: '公式长度超过上限',
  ILLEGAL_AST: '公式包含不支持的语法',
  INDICATOR_CONTEXT_MISMATCH: '指标与当前计算域不匹配',
  INDICATOR_NOT_APPLICABLE: '指标不适用于当前产品',
  INPUT_COVERAGE_PARTIAL: '部分输入字段存在缺失',
  INDICATOR_NOT_FOUND: '未找到指标',
  INSUFFICIENT_COMMON_SAMPLE: '共同样本不足',
  INSUFFICIENT_SAMPLE: '样本不足',
  INVALID_ARGUMENT_SOURCE: '参数来源不受支持',
  INVALID_AS_OF: '截止日期不正确',
  INVALID_CONSTANT: '常数不正确',
  INVALID_CONTEXT_KIND: '计算域不正确',
  INVALID_EXPRESSION: '公式不正确',
  INVALID_LITERAL: '公式中的常量不受支持',
  INVALID_OUTPUT_CONTRACT: '指标结果要求不正确',
  INVALID_PARAMETER: '参数不正确',
  INVALID_PERIOD: '计算区间不正确',
  LATEX_PARSE_ERROR: '公式无法解析',
  MISSING_ARGUMENT: '缺少必填参数',
  NJIT_BATCH_COMPILE_FAILED: '批量计算计划编译失败',
  NJIT_BATCH_RESULT_MISSING: '批量计算结果不完整',
  NJIT_PLAN_COMPILE_FAILED: '高性能计算计划编译失败',
  NON_FINITE_INPUT: '输入包含无效数值',
  NON_FINITE_RESULT: '计算结果不是有限数值',
  NO_DATA_AS_OF: '截止日期之前没有可用数据',
  NO_DATA_FOR_PERIOD: '计算区间内没有可用数据',
  OPERATOR_EXECUTION_FAILED: '算子计算失败',
  OPERATOR_VERSION_MISMATCH: '算子版本不匹配',
  OUTPUT_CONTRACT_MISMATCH: '指标结果类型不符合要求',
  PRICE_BASIS_MISMATCH: '价格口径不一致',
  PRODUCT_DATA_NOT_FOUND: '未找到产品数据',
  REQUEST_VALIDATION_ERROR: '请求内容不完整',
  REVISION_CONFLICT: '指标已被其他操作更新',
  RUNTIME_SHAPE_MISMATCH: '实际数据规模与公式不匹配',
  RUNTIME_TYPE_MISMATCH: '实际数据类型与公式不匹配',
  SEMANTIC_DIMENSION_MISMATCH: '数值含义不兼容',
  SEMANTIC_NOTE: '数值含义提示',
  SEMANTIC_ROLE_MISMATCH: '变量用途不兼容',
  SEMANTIC_ROLE_WARNING: '变量用途需要确认',
  SHAPE_LIMIT_EXCEEDED: '数据规模超过上限',
  SHAPE_MISMATCH: '输入数据结构不匹配',
  SINGULAR_MATRIX: '矩阵无法求解',
  SOURCE_DATASET_MISSING: '缺少所需数据集',
  SOURCE_UNAVAILABLE_FOR_PRODUCT: '当前产品缺少对应数据源',
  SOURCE_FIELD_MISSING: '缺少所需数据字段',
  SOURCE_SCHEMA_MISMATCH: '数据格式不兼容',
  TYPE_MISMATCH: '输入数据类型不匹配',
  UNKNOWN_ARGUMENT: '包含未定义的参数',
  UNKNOWN_FUNCTION: '包含不支持的函数',
  UNKNOWN_OPERATOR: '包含不支持的算子',
  UNKNOWN_VARIABLE: '包含未定义的变量',
  UNRESOLVED_SHAPE: '无法确定输出数据结构',
  UNSUPPORTED_DSL_VERSION: '公式版本暂不支持',
  UNSUPPORTED_NUMERIC_KERNEL_VERSION: '计算内核版本暂不支持',
  UNSUPPORTED_OPERATOR_REGISTRY_VERSION: '算子版本暂不支持',
  VARIABLE_CONTEXT_MISMATCH: '变量不适用于当前计算域',
  VARIABLE_NOT_APPLICABLE: '变量不适用于当前产品',
  VARIABLE_NOT_IN_WINDOW: '计算区间内没有该变量',
  VARIABLE_NO_OBSERVATIONS: '该变量没有有效观察值',
  VARIABLE_UNAVAILABLE: '变量暂不可用',
  WEIGHTS_NOT_NORMALIZED: '权重合计不正确',
  WEIGHT_COUNT_MISMATCH: '权重数量与产品数量不一致',
  WEIGHT_SUM_INVALID: '权重合计不正确',
}

const stripDiagnosticPrefix = (value: string) => value
  .replace(/^\s*\[[A-Z][A-Z0-9_]+\]\s*/u, '')
  .replace(/^\s*[A-Z][A-Z0-9_]{3,}\s*[:：-]\s*/u, '')
  .trim()

export function humanizeIndicatorTechnicalText(value: unknown): string {
  if (value === null || value === undefined || value === '') return ''
  if (Array.isArray(value)) return value.map(humanizeIndicatorTechnicalText).filter(Boolean).join(' / ')
  if (typeof value === 'object') return '由输入数据类型决定'

  const raw = String(value)
  return raw
    .replace(/same\([^)]*\)/gi, '与输入相同的数据类型')
    .replace(/one_dimensional/gi, '一维数据')
    .replace(/mask<time\s*,\s*asset>(?:\[[^\]]*\])?/gi, '时间—资产布尔掩码')
    .replace(/mask<asset\s*,\s*time>(?:\[[^\]]*\])?/gi, '时间—资产布尔掩码')
    .replace(/mask<time>(?:\[[^\]]*\])?/gi, '时间序列布尔掩码')
    .replace(/mask<asset>(?:\[[^\]]*\])?/gi, '资产布尔掩码')
    .replace(/matrix<time\s*,\s*asset>(?:\[[^\]]*\])?/gi, '时间—资产矩阵')
    .replace(/matrix<asset\s*,\s*time>(?:\[[^\]]*\])?/gi, '时间—资产矩阵')
    .replace(/matrix<asset\s*,\s*asset>(?:\[[^\]]*\])?/gi, '资产方阵')
    .replace(/matrix<[^>]+>(?:\[[^\]]*\])?/gi, '矩阵')
    .replace(/series<[^>]+>(?:\[[^\]]*\])?/gi, '时间序列')
    .replace(/vector<[^>]+>(?:\[[^\]]*\])?/gi, '资产向量')
    .replace(/\bscalar\b/gi, '有限标量')
    .replace(/\bmask\b/gi, '布尔掩码')
    .replace(/\bseries\b/gi, '时间序列')
    .replace(/\bvector\b/gi, '资产向量')
    .replace(/\bmatrix\b/gi, '矩阵')
    .replace(/\btensor\b/gi, '数值数组')
    .replace(/\bfloat64\b/gi, '数值')
    .replace(/\bbool\b/gi, '布尔值')
    .replace(/\breturn_decimal\b/gi, '收益率')
    .replace(/\brate_decimal\b/gi, '利率')
    .replace(/\badjusted_nav\b/gi, '复权净值')
    .replace(/\breported_nav\b/gi, '披露净值')
    .replace(/\braw_market_price\b/gi, '市场价格')
    .replace(/\bcurrency_amount\b/gi, '金额')
    .replace(/\bdimensionless\b/gi, '无量纲数值')
    .replace(/\belementwise\b/gi, '逐元素计算')
    .replace(/\breduction\b/gi, '归约计算')
    .replace(/\bprice_basis\b/gi, '价格口径')
    .replace(/\bsemantic_dimension\b/gi, '数值含义')
    .replace(/\bshape\b/gi, '数据规模')
}

export function indicatorDiagnosticTitle(code: string): string {
  return DIAGNOSTIC_TITLES[code] ?? '需要处理的问题'
}

export function indicatorDiagnosticDetail(code: string, message: string): string {
  if (code === 'OUTPUT_CONTRACT_MISMATCH') {
    return '指标最终结果必须是单个有限数值。当前公式返回了一组数据，请继续使用求和、平均值、标准差等归约算子将其转换为单个数值。'
  }
  const withoutPrefix = stripDiagnosticPrefix(String(message || ''))
  const withoutRepeatedCode = code
    ? withoutPrefix.replace(new RegExp(`\\b${code.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'g'), indicatorDiagnosticTitle(code))
    : withoutPrefix
  return humanizeIndicatorTechnicalText(withoutRepeatedCode) || '请检查当前公式和输入数据。'
}

export function formatIndicatorDiagnostic(code: string, message: string): string {
  const title = indicatorDiagnosticTitle(code)
  const detail = indicatorDiagnosticDetail(code, message)
  return detail.startsWith(title) ? detail : `${title}：${detail}`
}

export function humanizeIndicatorMessage(message: unknown, fallback = '请检查当前公式和输入数据。'): string {
  const raw = String(message ?? '').trim()
  if (!raw) return fallback
  const code = raw.match(/^\s*\[([A-Z][A-Z0-9_]+)\]/)?.[1]
    ?? raw.match(/^\s*([A-Z][A-Z0-9_]{3,})\s*[:：-]/)?.[1]
  if (code) return formatIndicatorDiagnostic(code, raw)
  return humanizeIndicatorTechnicalText(stripDiagnosticPrefix(raw)) || fallback
}

export function evaluationStatusLabel(status: string): string {
  return ({
    ok: '计算正常',
    warning: '计算完成，但需注意',
    unavailable: '不可计算',
    error: '计算失败',
    ranked: '已排名',
    excluded: '不参与排名',
  } as Record<string, string>)[status] ?? '状态待确认'
}
