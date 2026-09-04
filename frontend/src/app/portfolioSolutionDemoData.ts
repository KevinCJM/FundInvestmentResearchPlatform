export interface PortfolioSolutionDemo {
  id: string
  name: string
  subtitle: string
  riskLevel: string
  horizon: string
  status: string
  benchmark: string
  version: string
  effectiveDate: string
  annualizedReturn: string
  volatility: string
  maxDrawdown: string
  sharpe: string
  allocation: Array<{ label: string; value: number; color: string }>
}

export const portfolioSolutions: PortfolioSolutionDemo[] = [
  {
    id: 'steady-multi-asset',
    name: '稳健多资产方案',
    subtitle: '以控制回撤为首要目标，通过多资产分散与规则化再平衡获取中长期稳健回报。',
    riskLevel: '中低风险',
    horizon: '建议持有 3 年以上',
    status: '展示中',
    benchmark: '中债综合财富指数 70% + 沪深300全收益指数 25% + 活期存款 5%',
    version: 'V3.2',
    effectiveDate: '2026-08-31',
    annualizedReturn: '8.42%',
    volatility: '7.61%',
    maxDrawdown: '-6.84%',
    sharpe: '0.92',
    allocation: [
      { label: '固收', value: 52, color: 'bg-indigo-500' },
      { label: '权益', value: 28, color: 'bg-fuchsia-500' },
      { label: '商品', value: 8, color: 'bg-amber-500' },
      { label: '海外', value: 7, color: 'bg-emerald-500' },
      { label: '现金', value: 5, color: 'bg-slate-400' },
    ],
  },
  {
    id: 'balanced-growth',
    name: '均衡增长方案',
    subtitle: '在权益增长与固收防御之间保持均衡，并以市场状态控制战术偏离。',
    riskLevel: '中风险',
    horizon: '建议持有 3—5 年',
    status: '待复核',
    benchmark: '沪深300全收益指数 50% + 中债综合财富指数 45% + 活期存款 5%',
    version: 'V2.6',
    effectiveDate: '2026-07-31',
    annualizedReturn: '10.68%',
    volatility: '11.42%',
    maxDrawdown: '-11.36%',
    sharpe: '0.86',
    allocation: [
      { label: '权益', value: 48, color: 'bg-fuchsia-500' },
      { label: '固收', value: 35, color: 'bg-indigo-500' },
      { label: '商品', value: 7, color: 'bg-amber-500' },
      { label: '海外', value: 6, color: 'bg-emerald-500' },
      { label: '现金', value: 4, color: 'bg-slate-400' },
    ],
  },
  {
    id: 'equity-enhanced',
    name: '权益增强方案',
    subtitle: '以权益风险溢价为主要收益来源，结合风格分散与产品级轮动控制集中度。',
    riskLevel: '中高风险',
    horizon: '建议持有 5 年以上',
    status: '草案',
    benchmark: '中证800全收益指数 85% + 中债综合财富指数 10% + 活期存款 5%',
    version: 'V1.4',
    effectiveDate: '2026-08-15',
    annualizedReturn: '14.12%',
    volatility: '17.83%',
    maxDrawdown: '-19.74%',
    sharpe: '0.71',
    allocation: [
      { label: '权益', value: 78, color: 'bg-fuchsia-500' },
      { label: '固收', value: 10, color: 'bg-indigo-500' },
      { label: '海外', value: 7, color: 'bg-emerald-500' },
      { label: '现金', value: 5, color: 'bg-slate-400' },
    ],
  },
]

export const performanceSeries = [
  { label: '2021', portfolio: 100, benchmark: 100 },
  { label: '2022', portfolio: 104, benchmark: 101 },
  { label: '2023', portfolio: 111, benchmark: 106 },
  { label: '2024', portfolio: 116, benchmark: 111 },
  { label: '2025', portfolio: 126, benchmark: 119 },
  { label: '2026', portfolio: 134, benchmark: 125 },
]

export const periodPerformance = [
  ['近1个月', '+0.62%', '+0.41%', '+0.21%'],
  ['近3个月', '+2.16%', '+1.72%', '+0.44%'],
  ['今年以来', '+6.84%', '+5.72%', '+1.12%'],
  ['近1年', '+8.21%', '+6.93%', '+1.28%'],
  ['近3年年化', '+7.92%', '+6.38%', '+1.54%'],
]

export const scenarioResults = [
  { id: 'recession', name: '经济衰退', description: '增长回落、权益承压、利率下行', portfolio: '-4.8%', benchmark: '-6.9%', recovery: '7 个月', confidence: '历史映射' },
  { id: 'inflation', name: '高通胀', description: '通胀上行、利率抬升、商品走强', portfolio: '-2.1%', benchmark: '-3.7%', recovery: '4 个月', confidence: '模型估算' },
  { id: 'liquidity', name: '流动性冲击', description: '风险资产同步下跌、信用利差走阔', portfolio: '-7.6%', benchmark: '-9.4%', recovery: '11 个月', confidence: '压力测试' },
  { id: 'recovery', name: '复苏扩张', description: '盈利改善、权益与商品相对占优', portfolio: '+9.3%', benchmark: '+7.8%', recovery: '—', confidence: '历史映射' },
]

export const algorithmRules = [
  ['战略配置', '长期风险预算 + 约束有效前沿', '年度复核', 'SAA-R6'],
  ['市场状态', '增长 / 通胀 / 波动 / 流动性四维识别', '月度识别', 'REGIME-R4'],
  ['战术偏离', '相对 SAA 在 ±5% 范围内调整', '月度或阈值触发', 'TAA-R3'],
  ['产品配置', '评分、相关性、容量与费用约束', '月度复核', 'SELECT-R8'],
  ['再平衡', '资产偏离 3% 或状态确认后触发', '周度检查', 'REB-R5'],
]

export const disclosureItems = [
  '实盘业绩与历史回测分段展示，不拼接为同一条未标识曲线',
  '组合与基准使用相同区间、频率和收益计算口径',
  '回测明确数据截止日、PIT 快照、费用、滑点和申赎时滞假设',
  '展示适用范围、风险等级、建议持有期及主要风险',
  '组合规则、指标、产品池和情景算法均绑定不可变版本',
  '历史表现和情景模拟不代表未来收益，也不构成投资建议',
]
