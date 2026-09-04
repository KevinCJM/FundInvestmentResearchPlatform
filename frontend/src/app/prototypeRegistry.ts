export interface PrototypeMetric {
  label: string
  value: string
  hint: string
}

export interface PrototypeLink {
  label: string
  description: string
  path: string
}

export interface PrototypeConfig {
  title: string
  description: string
  family: 'product' | 'allocation' | 'execution' | 'accounting' | 'post' | 'feedback' | 'settings'
  filterLabel: string
  filterOptions: string[]
  metrics: PrototypeMetric[]
  chartTitle: string
  chartPoints: Array<{ label: string; value: number }>
  columns: string[]
  rows: string[][]
  guidance: string[]
  tabs?: string[]
  links?: PrototypeLink[]
}

type ConfigOverrides = Partial<Omit<PrototypeConfig, 'title' | 'description' | 'family'>>

const familyDefaults: Record<PrototypeConfig['family'], Omit<PrototypeConfig, 'title' | 'description' | 'family'>> = {
  product: {
    filterLabel: '研究范围',
    filterOptions: ['全市场基金', '主动权益', '固收与固收+', '被动指数'],
    metrics: [
      { label: '研究样本', value: '1,286', hint: '示例产品数量' },
      { label: '可用历史', value: '8.4 年', hint: '示例中位数' },
      { label: '待复核项目', value: '12', hint: '示例研究事项' },
    ],
    chartTitle: '研究覆盖变化（示例数据）',
    chartPoints: [{ label: '4月', value: 42 }, { label: '5月', value: 56 }, { label: '6月', value: 61 }, { label: '7月', value: 73 }, { label: '8月', value: 86 }],
    columns: ['研究对象', '研究状态', '核心结论', '版本'],
    rows: [
      ['沪深300增强样本组', '已复核', '超额收益稳定性待持续观察', 'R3'],
      ['中短债基金样本组', '研究中', '回撤控制具有分层差异', 'R2'],
      ['红利低波指数样本组', '待复核', '风格暴露集中度较高', 'R1'],
    ],
    guidance: ['所有日期和数值均为预置示例。', '正式功能需绑定 PIT 数据版本与研究口径。', '研究结论与产品池版本应保持可追溯关系。'],
  },
  allocation: {
    filterLabel: '组合版本',
    filterOptions: ['稳健组合 · 草案03', '平衡组合 · 草案02', '进取组合 · 草案01'],
    metrics: [
      { label: '目标波动', value: '8.0%', hint: '示例约束' },
      { label: '权益中枢', value: '45%', hint: '示例 SAA 权重' },
      { label: '允许偏离', value: '±8%', hint: '示例 TAA 区间' },
    ],
    chartTitle: '资产权重研究（示例数据）',
    chartPoints: [{ label: '权益', value: 45 }, { label: '固收', value: 35 }, { label: '商品', value: 8 }, { label: '海外', value: 7 }, { label: '现金', value: 5 }],
    columns: ['研究维度', '基准值', '研究值', '约束状态'],
    rows: [
      ['权益类资产', '45%', '48%', '区间内'],
      ['固收类资产', '35%', '33%', '区间内'],
      ['商品类资产', '8%', '7%', '区间内'],
    ],
    guidance: ['SAA 负责长期配置中枢、区间和风险预算。', 'TAA 研究市场状态与相对 SAA 的偏离路径。', '页面结果仅用于展示研究过程，不构成投资建议。'],
  },
  execution: {
    filterLabel: '数据日期',
    filterOptions: ['2026-08-31', '2026-08-28', '2026-08-27'],
    metrics: [
      { label: '目标资金', value: '1.20 亿', hint: '示例组合规模' },
      { label: '预计现金占用', value: '7.6%', hint: '示例交收测算' },
      { label: '待核对记录', value: '4', hint: '示例数据项' },
    ],
    chartTitle: '目标与记录权重差异（示例数据）',
    chartPoints: [{ label: '权益', value: 3 }, { label: '固收', value: 2 }, { label: '商品', value: 1 }, { label: '海外', value: 4 }, { label: '现金', value: 2 }],
    columns: ['产品代码', '方向', '数量', '记录状态'],
    rows: [
      ['510300.SH', '买入', '120,000', '待核对'],
      ['000012.OF', '申购', '3,500,000', '交收中'],
      ['518880.SH', '卖出', '28,000', '已记录'],
    ],
    guidance: ['本平台仅承接计划测算、数据记录和研究核对。', '示例页面不连接交易系统，不产生真实指令。', '场外基金交收时滞与现金拖累应单独测算。'],
  },
  accounting: {
    filterLabel: '账务视角',
    filterOptions: ['组合/基金账 · PF-DEMO-01', '管理人公司账 · FM-DEMO', '双账勾稽视图'],
    metrics: [
      { label: '组合账套', value: '12', hint: '示例独立组合/基金账' },
      { label: '管理人账', value: '1', hint: '示例公司账映射' },
      { label: '待勾稽事项', value: '6', hint: '示例跨账差异' },
    ],
    chartTitle: '双账处理进度（示例数据）',
    chartPoints: [{ label: 'Booking', value: 100 }, { label: '组合账', value: 94 }, { label: '管理人账', value: 87 }, { label: '双账勾稽', value: 79 }, { label: '关账', value: 66 }],
    columns: ['核算主体', '账务范围', '处理状态', '会计版本'],
    rows: [
      ['PF-DEMO-01', '组合/基金账', '已过账', 'FUND-POL-R3'],
      ['FM-DEMO', '管理人公司账映射', '待复核', 'CORP-POL-R2'],
      ['BK-240831-FEE', '双账勾稽事项', '待匹配', 'MAP-R1'],
    ],
    guidance: ['本页为研究平台静态演示，不替代基金管理人、托管人或公司财务系统的法定账簿。', '每个组合/基金分别建账，管理人公司账独立核算；表内/表外是各账套下的确认属性。', '两类报表不得直接混加，跨账事项通过同一 Booking 来源号勾稽。'],
  },
  post: {
    filterLabel: '分析区间',
    filterOptions: ['近1年', '今年以来', '近3年', '成立以来'],
    metrics: [
      { label: '组合收益', value: '6.82%', hint: '示例区间收益' },
      { label: '最大回撤', value: '-4.36%', hint: '示例风险指标' },
      { label: '相对基准', value: '+1.24%', hint: '示例超额收益' },
    ],
    chartTitle: '组合研究序列（示例数据）',
    chartPoints: [{ label: '4月', value: 52 }, { label: '5月', value: 49 }, { label: '6月', value: 66 }, { label: '7月', value: 71 }, { label: '8月', value: 78 }],
    columns: ['分析维度', '组合结果', '基准结果', '研究解释'],
    rows: [
      ['权益配置', '+2.18%', '+1.42%', '主要正贡献'],
      ['固收配置', '+1.06%', '+0.98%', '表现接近基准'],
      ['产品选择', '+0.62%', '—', '增强策略贡献'],
    ],
    guidance: ['当前为单组合投后研究示例。', '正式页面需绑定真实持仓、现金流、费用和基准。', '归因结果应与投前研究版本和数据截止日关联。'],
  },
  feedback: {
    filterLabel: '复盘周期',
    filterOptions: ['2026年8月', '2026年二季度', '2026年上半年'],
    metrics: [
      { label: '复盘事项', value: '18', hint: '示例事项数量' },
      { label: '完成验证', value: '11', hint: '示例完成数' },
      { label: '进入下一版本', value: '5', hint: '示例迭代项' },
    ],
    chartTitle: '复盘事项进度（示例数据）',
    chartPoints: [{ label: '已确认', value: 82 }, { label: '验证中', value: 61 }, { label: '待补充', value: 34 }, { label: '已归档', value: 73 }],
    columns: ['复盘对象', '偏差来源', '验证状态', '关联版本'],
    rows: [
      ['权益风险预算', '波动状态切换', '验证中', 'SAA-R3'],
      ['中短债产品组', '信用利差变化', '已确认', 'POOL-R8'],
      ['月度再平衡规则', '交收时滞', '待补充', 'RULE-R2'],
    ],
    guidance: ['复盘基于实际结果与投前假设的差异。', '反馈可以触发产品复审、模型研究或流程修订。', '每次迭代保留原始版本，避免覆盖历史研究记录。'],
  },
  settings: {
    filterLabel: '能力状态',
    filterOptions: ['全部', '研究中', '已验证', '已发布'],
    metrics: [
      { label: '已发布版本', value: '24', hint: '示例公共配置' },
      { label: '待验证版本', value: '7', hint: '示例研究版本' },
      { label: '被流程引用', value: '16', hint: '示例绑定数量' },
    ],
    chartTitle: '公共能力版本分布（示例数据）',
    chartPoints: [{ label: '数据', value: 88 }, { label: '指标', value: 72 }, { label: '情景', value: 54 }, { label: '回测', value: 67 }, { label: '参数', value: 43 }],
    columns: ['能力名称', '版本', '状态', '引用范围'],
    rows: [
      ['流动性压力情景', 'SCN-R4', '已验证', 'TAA / 投后'],
      ['严格回测模板', 'BT-R7', '已发布', '投前 / 产品研究'],
      ['交易日历', 'CAL-2026', '已发布', '全平台'],
    ],
    guidance: ['公共能力按版本被研究流程引用。', '示例修改只在当前页面内生效。', '正式发布需要数据、公式和依赖关系校验。'],
  },
}

const makeConfig = (
  family: PrototypeConfig['family'],
  title: string,
  description: string,
  overrides: ConfigOverrides = {},
): PrototypeConfig => ({ title, description, family, ...familyDefaults[family], ...overrides })

const product = (title: string, description: string, overrides?: ConfigOverrides) => makeConfig('product', title, description, overrides)
const allocation = (title: string, description: string, overrides?: ConfigOverrides) => makeConfig('allocation', title, description, overrides)
const execution = (title: string, description: string, overrides?: ConfigOverrides) => makeConfig('execution', title, description, overrides)
const accounting = (title: string, description: string, overrides?: ConfigOverrides) => makeConfig('accounting', title, description, overrides)
const post = (title: string, description: string, overrides?: ConfigOverrides) => makeConfig('post', title, description, overrides)
const feedback = (title: string, description: string, overrides?: ConfigOverrides) => makeConfig('feedback', title, description, overrides)
const settings = (title: string, description: string, overrides?: ConfigOverrides) => makeConfig('settings', title, description, overrides)

export const prototypeConfigs: Record<string, PrototypeConfig> = {
  'holding-style': product('持仓穿透与风格研究', '区分真实披露持仓与估算风格暴露，研究底层集中度、行业风格、因子暴露和披露时滞。', {
    tabs: ['穿透概览', '披露持仓', '风格估算', '差异比较', '研究记录'],
    metrics: [
      { label: '披露覆盖率', value: '72.4%', hint: '示例底层资产覆盖' },
      { label: '披露时滞', value: '41 天', hint: '示例可得日差' },
      { label: '估算置信度', value: '0.81', hint: '净值回归示例' },
    ],
  }),
  'product-backtest': product('产品与信号回测', '研究单产品持有、择时信号和评价逻辑的历史表现。', {
    metrics: [
      { label: '样本外收益', value: '7.31%', hint: '示例年化结果' },
      { label: '信号胜率', value: '58.4%', hint: '示例统计' },
      { label: '换手率', value: '1.8 倍', hint: '示例年度换手' },
    ],
  }),
  'product-pools': product('产品池构建', '依据准入条件、评价结论和研究标签形成可复用产品池。'),
  'pool-lifecycle': product('产品池版本与生命周期管理', '管理产品准入、观察、限制、替换、排除和版本发布。'),

  objectives: allocation('投资目标与约束', '定义单个组合的收益目标、风险边界、基准、期限、流动性和投资范围。'),
  'product-pool-selection': allocation('选择产品池版本', '选择已生效产品池版本，锁定本次组合研究的产品边界。'),
  taa: allocation('战术资产配置（TAA）', '展示市场状态、情景算法和相对 SAA 中枢的权重偏离研究。', {
    metrics: [
      { label: '当前状态', value: '低波扩张', hint: '示例情景识别' },
      { label: '权益偏离', value: '+3%', hint: '相对 SAA 示例' },
      { label: '风险预算占用', value: '86%', hint: '示例测算' },
    ],
  }),
  'product-allocation-timing': allocation('产品配置与择时', '在大类预算内研究具体基金权重、替代关系和产品级择时规则。', {
    links: [
      { label: '打开已有产品组合构建', description: '运行真实产品组合研究并生成不可变研究快照。', path: '/pre-investment/product-allocation-timing/construction' },
      { label: '进入产品择时演示', description: '查看产品级信号、权重路径与替代关系的静态交互页面。', path: '/pre-investment/product-allocation-timing/timing' },
    ],
  }),
  'product-timing': allocation('产品择时研究', '研究同一大类内部具体产品的信号、替代顺序和动态权重路径。'),
  'portfolio-synthesis': allocation('目标组合合成与风险检查', '把 SAA、TAA 与产品配置合成为产品级目标组合，并检查风险预算、集中度和全部投资约束。', {
    tabs: ['组合总览', '权重穿透', '风险贡献', '约束检查', '结果记录'],
    metrics: [
      { label: '权重合计', value: '100.0%', hint: '大类与产品双层校验' },
      { label: '最大风险贡献', value: '31.6%', hint: '示例单一大类贡献' },
      { label: '硬约束', value: '12 / 12', hint: '示例全部通过' },
    ],
  }),
  validation: allocation('统一回测与稳健性验证', '通过样本内外、滚动验证、压力与成本模拟检验研究方案。', {
    tabs: ['验证概览', '样本划分', '敏感性分析', '版本比较', '结果记录'],
    links: [
      { label: '大类策略回测', description: '进入已有大类资产配置与策略回测工具。', path: '/pre-investment/saa/allocation-lab' },
      { label: '产品组合研究', description: '进入已有产品组合构建工具。', path: '/pre-investment/product-allocation-timing/construction' },
    ],
  }),
  approval: allocation('研究方案定稿与外部审批记录', '固化目标组合、产品池、模型版本、数据截止日，并记录平台外部审批结果。'),

  'trade-plan': execution('目标组合与交易计划测算', '结合目标权重、当前持仓、资金、容量和交收周期测算交易计划。'),
  'pre-trade-check': execution('交易前检查', '研究现金、申赎状态、容量、费用、流动性和投资限制。'),
  'cash-settlement': execution('现金与交收日历', '按产品交易与申赎规则测算资金可用日期、在途现金、资金缺口和现金拖累。', {
    tabs: ['资金概览', '交收日历', '在途资金', '现金拖累', '结果记录'],
    metrics: [
      { label: '在途资金', value: '860 万', hint: '示例待交收金额' },
      { label: '预计现金拖累', value: '0.12%', hint: '示例区间影响' },
      { label: '最大资金缺口', value: '320 万', hint: '示例峰值' },
    ],
  }),
  reconciliation: execution('持仓与资金核对', '核对交易记录、现金余额、基金份额、成本和目标组合差异。'),
  rebalancing: execution('再平衡测算与记录', '测算权重偏离、现金拖累和再平衡方案并记录研究过程。'),

  'accounting-policy': accounting('核算主体、账簿与记账规则', '关联组合中心已有核算主体与账簿，分别定义组合/基金账和管理人公司账的科目、借贷模板、确认计量、估值及报表口径。', {
    tabs: ['主体与账簿', '科目表', '借贷规则模板', '确认与计量', '生效版本'],
  }),
  'portfolio-ledger': accounting('组合/基金总账与明细账', '按每个组合或基金独立复核和过账，维护凭证、总账、明细账、试算平衡、持仓、现金、净资产和表外备查簿。', {
    tabs: ['账套概览', '会计凭证', '总账与明细账', '试算平衡', '表外备查簿'],
    metrics: [
      { label: '当前账套', value: 'PF-DEMO-01', hint: '示例组合账' },
      { label: '借贷差额', value: '0.00', hint: '示例试算平衡' },
      { label: '待复核分录', value: '3', hint: '示例异常数量' },
    ],
  }),
  'manager-ledger': accounting('管理人公司账映射', '记录管理费收入与应收款、自有资金跟投等管理人相关事项，并生成面向公司财务系统的账务映射。', {
    tabs: ['映射概览', '管理费与应收', '自有资金跟投', '公司科目映射', '接口边界'],
    metrics: [
      { label: '管理费应收', value: '286 万', hint: '示例公司账余额' },
      { label: '自有资金跟投', value: '1,200 万', hint: '示例投资资产' },
      { label: '待财务确认', value: '4', hint: '不自动写入公司总账' },
    ],
  }),
  'cross-book-reconciliation': accounting('双账勾稽、差错与关账', '通过 Booking 来源号核对组合/基金账与管理人公司账，并分别完成托管、银行对账和账期冻结。', {
    tabs: ['勾稽看板', '管理费双边匹配', '自有资金跟投', '差错与调账', '关账版本'],
    metrics: [
      { label: '跨账事项', value: '18', hint: '示例待勾稽业务' },
      { label: '自动匹配', value: '16', hint: '来源号与金额一致' },
      { label: '差异事项', value: '2', hint: '不得静默抵销' },
    ],
  }),
  'accounting-valuation': accounting('估值、应计与净资产核算', '依据冻结价格、汇率和会计政策处理公允价值、利息、费用、税费、份额与净资产。', {
    tabs: ['估值概览', '价格与汇率', '收益费用应计', '净资产核算', '估值调整'],
  }),
  'performance-dataset': accounting('绩效核算数据集（PBOR）', '从冻结会计版本输出估值快照、外部资金流、内部交易、费用、持仓与基准映射，供组合层指标计算。', {
    tabs: ['数据集概览', '外部资金流', '内部交易', '估值快照', '指标引用'],
    metrics: [
      { label: '估值覆盖', value: '100%', hint: '示例日度覆盖率' },
      { label: '外部资金流', value: '3 笔', hint: '示例客户入金/出金' },
      { label: '待分类交易', value: '1 笔', hint: '不得进入正式指标' },
    ],
  }),

  'portfolio-data': post('实际组合数据与三账核对', '围绕所选真实组合核对 IBOR、ABOR、PBOR 的交易、持仓、现金、估值、外部资金流和基准口径。'),
  performance: post('绩效计量', '基于冻结估值、外部资金流、费用和基准计算单组合 TWR、MWR/XIRR、风险与相对表现。', {
    links: [{ label: '打开已有研究组合诊断', description: '基于不可变研究快照查看真实计算结果。', path: '/post-investment/research-diagnosis' }],
  }),
  'return-attribution': post('收益归因', '展示配置、择时、产品选择、交互和执行偏差的归因结构。'),
  'risk-attribution': post('风险归因', '展示大类、产品、行业风格和风险因子的风险贡献结构。'),
  monitoring: post('持仓产品与管理人监控', '跟踪持仓产品业绩、风格漂移、规模、流动性和团队变化。'),
  scenarios: post('情景分析与压力测试', '展示历史情景、自定义冲击和流动性压力下的组合研究结果。', {
    links: [{ label: '打开已有历史情景工具', description: '在研究组合诊断中运行真实历史区间情景。', path: '/post-investment/research-diagnosis' }],
  }),
  conclusions: post('投后研究结论', '固化基于业绩、归因、监控和情景分析形成的可审计研究结论。'),
  reports: post('投资报告与归档', '展示报告版本、研究附件、数据截止日和归档记录。'),

  results: feedback('投资结果反馈', '汇集实际收益、风险、归因、数据偏差和投后事件。'),
  'hypothesis-review': feedback('研究假设复盘', '比较投前假设与实际结果，识别差异及证据来源。'),
  'product-review': feedback('产品复审与产品池更新', '重新评价相关产品并形成新的产品池研究版本。'),
  'model-update': feedback('配置与模型更新', '复核战略配置、TAA 信号、风险预算和再平衡参数。'),
  'rolling-validation': feedback('滚动复验与模型比较', '通过 Walk-forward、样本外和敏感性分析比较研究版本。'),
  'process-improvement': feedback('投资流程改进', '把已验证的复盘结论固化为新研究流程、规则和控制要求。'),

  'pit-snapshots': settings('PIT 时点快照', '管理历史时点切片、数据血缘、可用版本和被研究任务引用的关系。'),
  'research-parameters': settings('研究口径与参数中心', '维护全平台可复用、版本化的研究参数模板，供产品研究、资产配置、组合回测和投后分析按版本引用。', {
    filterLabel: '适用范围',
    filterOptions: ['全部研究环节', '产品研究', '资产配置与组合回测', '投后分析'],
    tabs: ['模板概览', '基准与利率', '数据与计算口径', '适用范围', '版本记录'],
    metrics: [
      { label: '已发布模板', value: '6', hint: '示例有效版本' },
      { label: '待复核模板', value: '2', hint: '示例候选版本' },
      { label: '当前引用', value: '21', hint: '示例研究任务' },
    ],
    chartTitle: '参数模板引用分布（示例数据）',
    chartPoints: [{ label: '产品研究', value: 8 }, { label: 'SAA', value: 4 }, { label: 'TAA', value: 3 }, { label: '组合回测', value: 4 }, { label: '投后分析', value: 2 }],
    columns: ['模板名称', '版本', '核心口径', '适用范围'],
    rows: [
      ['人民币多资产标准口径', 'RP-R4', '中债/权益复合基准；R007 无风险利率', 'SAA / TAA / 组合回测'],
      ['公募基金评价口径', 'RP-R6', '同类基准；周频；复权净值', '产品研究 / 产品池'],
      ['真实组合绩效口径', 'RP-R3', 'TWR + XIRR；外部现金流分类', '投后分析'],
    ],
    guidance: ['业务页面只选择并引用模板，不重复维护公共口径。', '研究任务启动时冻结模板及其依赖版本，后续模板变更不改写历史结果。', 'PIT 页面负责实际数据快照，指标中心负责公式；本中心负责组装研究口径。'],
  }),
  'scenario-algorithms': settings('情景算法中心', '研究、验证、保存和版本化情景算法，供各研究环节按版本引用。', {
    tabs: ['算法列表', '参数研究', '版本比较', '应用关系'],
  }),
  'backtest-center': settings('统一回测中心', '集中展示回测引擎、规则、模板、任务和不可变结果档案。', {
    tabs: ['引擎管理', '规则管理', '模板管理', '运行任务', '结果档案'],
  }),
  'system-parameters': settings('系统参数', '管理枚举字典、时区、精度、功能开关和运行环境等技术参数。'),
}

export const getPrototypeConfig = (pageKey: string) => {
  const config = prototypeConfigs[pageKey]
  if (!config) throw new Error(`Unknown prototype page: ${pageKey}`)
  return config
}
