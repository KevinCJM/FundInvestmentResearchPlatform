export type ActualPortfolioStatus = '筹备中' | '待启用' | '运行中' | '暂停' | '清算中' | '已终止'

export type ExternalAccountKind = '执行通道账户' | '资金结算账户' | '托管证券账户' | 'TA 账户' | '管理人公司账户'
export type AccountSharingPolicy = '允许多组合共享执行' | '单一核算主体专用'

export interface ExternalAccount {
  accountId: string
  name: string
  kind: ExternalAccountKind
  institution: string
  legalOwnerEntityId: string
  currency: string
  sharingPolicy: AccountSharingPolicy
  status: '待启用' | '有效' | '冻结' | '已关闭'
}

export interface PortfolioAccountRelationship {
  relationshipId: string
  portfolioId: string
  accountId: string
  purpose: string
  virtualSubAccountId?: string
  effectiveFrom: string
  effectiveTo?: string
  status: '有效' | '待生效' | '已终止'
}

export interface PortfolioSleeve {
  sleeveId: string
  portfolioId: string
  name: string
  accountingTreatment: '辅助核算维度' | '独立核算主体'
}

export interface PortfolioManagerAssignment {
  assignmentId: string
  portfolioId: string
  portfolioManagerId: string
  portfolioManagerName: string
  managerEntityId: string
  role: '主基金经理' | '联席基金经理' | '投资经理'
  sleeveId: string
  effectiveFrom: string
  effectiveTo?: string
  status: '有效' | '待生效' | '已终止'
}

export interface PortfolioLifecycleEvent {
  date: string
  status: ActualPortfolioStatus
  event: string
  source: string
}

export interface ActualPortfolio {
  portfolioId: string
  name: string
  portfolioType: string
  status: ActualPortfolioStatus
  externalProductCode: string
  managerId: string
  managerName: string
  custodianEntityId: string
  custodianName: string
  baseCurrency: string
  inceptionDate: string
  activationDate: string
  benchmark: string
  approvalRecordId: string
  sourceResearchPlanId: string
  sourceResearchPlanVersion: string
  currentTargetVersion: string
  accountingEntityId: string
  primaryLedgerId: string
  accountingPolicyVersion: string
  lifecycle: PortfolioLifecycleEvent[]
}

export interface ApprovedResearchPlanDemo {
  planId: string
  name: string
  version: string
  approvalRecordId: string
  approvedAt: string
  targetVersion: string
}

export const actualPortfolioStatuses: ActualPortfolioStatus[] = ['筹备中', '待启用', '运行中', '暂停', '清算中', '已终止']

export const approvedResearchPlans: ApprovedResearchPlanDemo[] = [
  {
    planId: 'PLAN-STEADY-MA',
    name: '稳健多资产研究方案',
    version: 'R18',
    approvalRecordId: 'APR-2026-0188',
    approvedAt: '2026-08-25',
    targetVersion: 'TARGET-R18.1',
  },
  {
    planId: 'PLAN-BALANCED-GROWTH',
    name: '均衡增长研究方案',
    version: 'R11',
    approvalRecordId: 'APR-2026-0156',
    approvedAt: '2026-07-28',
    targetVersion: 'TARGET-R11.2',
  },
  {
    planId: 'PLAN-INCOME-PLUS',
    name: '固收增强研究方案',
    version: 'R7',
    approvalRecordId: 'APR-2026-0102',
    approvedAt: '2026-05-30',
    targetVersion: 'TARGET-R7.3',
  },
]

export const managerEntities = [
  { entityId: 'FM-DEMO', name: '示例基金管理有限公司' },
  { entityId: 'FM-DEMO-02', name: '示例资产管理有限公司' },
]

export const accountingPolicyVersions = ['FUND-GAAP-DEMO-v1', 'FUND-GAAP-DEMO-v2']

export const actualPortfolioDemoData: ActualPortfolio[] = [
  {
    portfolioId: 'PF-DEMO-01',
    name: '稳健多资产一号',
    portfolioType: '公募 FOF',
    status: '运行中',
    externalProductCode: '009901.OF',
    managerId: 'FM-DEMO',
    managerName: '示例基金管理有限公司',
    custodianEntityId: 'CUST-DEMO-01',
    custodianName: '示例托管银行',
    baseCurrency: 'CNY',
    inceptionDate: '2025-01-15',
    activationDate: '2025-01-16',
    benchmark: '中债综合财富指数 70% + 沪深300全收益指数 25% + 活期存款 5%',
    approvalRecordId: 'APR-2026-0188',
    sourceResearchPlanId: 'PLAN-STEADY-MA',
    sourceResearchPlanVersion: 'R18',
    currentTargetVersion: 'TARGET-R18.1',
    accountingEntityId: 'AE-PF-DEMO-01',
    primaryLedgerId: 'LEDGER-PF-DEMO-01',
    accountingPolicyVersion: 'FUND-GAAP-DEMO-v1',
    lifecycle: [
      { date: '2025-01-10', status: '筹备中', event: '完成平台主数据草案', source: 'ONBOARD-2025-001' },
      { date: '2025-01-15', status: '待启用', event: '登记外部成立与账户信息', source: 'EXT-EST-2025-015' },
      { date: '2025-01-16', status: '运行中', event: '组合主数据启用', source: 'ACT-2025-001' },
    ],
  },
  {
    portfolioId: 'PF-DEMO-02',
    name: '均衡配置二号',
    portfolioType: '基金专户',
    status: '暂停',
    externalProductCode: 'SMA-2025-008',
    managerId: 'FM-DEMO-02',
    managerName: '示例资产管理有限公司',
    custodianEntityId: 'CUST-DEMO-02',
    custodianName: '示例托管银行二',
    baseCurrency: 'CNY',
    inceptionDate: '2025-06-03',
    activationDate: '2025-06-05',
    benchmark: '沪深300全收益指数 50% + 中债综合财富指数 45% + 活期存款 5%',
    approvalRecordId: 'APR-2026-0156',
    sourceResearchPlanId: 'PLAN-BALANCED-GROWTH',
    sourceResearchPlanVersion: 'R11',
    currentTargetVersion: 'TARGET-R11.2',
    accountingEntityId: 'AE-PF-DEMO-02',
    primaryLedgerId: 'LEDGER-PF-DEMO-02',
    accountingPolicyVersion: 'FUND-GAAP-DEMO-v2',
    lifecycle: [
      { date: '2025-06-01', status: '筹备中', event: '登记组合主数据', source: 'ONBOARD-2025-008' },
      { date: '2025-06-05', status: '运行中', event: '组合主数据启用', source: 'ACT-2025-008' },
      { date: '2026-08-20', status: '暂停', event: '记录平台外暂停新增投资结果', source: 'EXT-STATUS-2026-021' },
    ],
  },
  {
    portfolioId: 'PF-DEMO-03',
    name: '固收增强三号',
    portfolioType: '公募 FOF',
    status: '清算中',
    externalProductCode: '008803.OF',
    managerId: 'FM-DEMO',
    managerName: '示例基金管理有限公司',
    custodianEntityId: 'CUST-DEMO-01',
    custodianName: '示例托管银行',
    baseCurrency: 'CNY',
    inceptionDate: '2023-03-20',
    activationDate: '2023-03-21',
    benchmark: '中债综合财富指数 90% + 沪深300全收益指数 10%',
    approvalRecordId: 'APR-2026-0102',
    sourceResearchPlanId: 'PLAN-INCOME-PLUS',
    sourceResearchPlanVersion: 'R7',
    currentTargetVersion: 'TARGET-R7.3',
    accountingEntityId: 'AE-PF-DEMO-03',
    primaryLedgerId: 'LEDGER-PF-DEMO-03',
    accountingPolicyVersion: 'FUND-GAAP-DEMO-v1',
    lifecycle: [
      { date: '2023-03-18', status: '待启用', event: '登记外部成立信息', source: 'EXT-EST-2023-006' },
      { date: '2023-03-21', status: '运行中', event: '组合主数据启用', source: 'ACT-2023-006' },
      { date: '2026-08-01', status: '清算中', event: '记录平台外清算状态', source: 'EXT-LIQ-2026-003' },
    ],
  },
]

export const getActualPortfolio = (portfolioId: string) => actualPortfolioDemoData.find((portfolio) => portfolio.portfolioId === portfolioId)

export const externalAccounts: ExternalAccount[] = [
  { accountId: 'EXEC-FM-DEMO-SSE', name: '示例管理人上交所集中交易通道', kind: '执行通道账户', institution: '示例证券公司', legalOwnerEntityId: 'FM-DEMO', currency: 'CNY', sharingPolicy: '允许多组合共享执行', status: '有效' },
  { accountId: 'EXEC-FM-DEMO-CIBM', name: '示例管理人银行间交易通道', kind: '执行通道账户', institution: '示例托管银行', legalOwnerEntityId: 'FM-DEMO', currency: 'CNY', sharingPolicy: '允许多组合共享执行', status: '有效' },
  { accountId: 'EXEC-FM-DEMO-02-SZSE', name: '示例资管深交所交易通道', kind: '执行通道账户', institution: '示例证券公司二', legalOwnerEntityId: 'FM-DEMO-02', currency: 'CNY', sharingPolicy: '允许多组合共享执行', status: '冻结' },
  { accountId: 'CASH-PF01-CNY', name: '稳健多资产一号托管资金账户', kind: '资金结算账户', institution: '示例托管银行', legalOwnerEntityId: 'AE-PF-DEMO-01', currency: 'CNY', sharingPolicy: '单一核算主体专用', status: '有效' },
  { accountId: 'CUST-PF01', name: '稳健多资产一号托管证券账户', kind: '托管证券账户', institution: '示例托管银行', legalOwnerEntityId: 'AE-PF-DEMO-01', currency: 'CNY', sharingPolicy: '单一核算主体专用', status: '有效' },
  { accountId: 'TA-PF01', name: '稳健多资产一号份额登记账户', kind: 'TA 账户', institution: '示例登记机构', legalOwnerEntityId: 'AE-PF-DEMO-01', currency: 'CNY', sharingPolicy: '单一核算主体专用', status: '有效' },
  { accountId: 'CASH-PF02-CNY', name: '均衡配置二号托管资金账户', kind: '资金结算账户', institution: '示例托管银行二', legalOwnerEntityId: 'AE-PF-DEMO-02', currency: 'CNY', sharingPolicy: '单一核算主体专用', status: '有效' },
  { accountId: 'CUST-PF02', name: '均衡配置二号托管证券账户', kind: '托管证券账户', institution: '示例托管银行二', legalOwnerEntityId: 'AE-PF-DEMO-02', currency: 'CNY', sharingPolicy: '单一核算主体专用', status: '有效' },
  { accountId: 'CASH-PF03-CNY', name: '固收增强三号托管资金账户', kind: '资金结算账户', institution: '示例托管银行', legalOwnerEntityId: 'AE-PF-DEMO-03', currency: 'CNY', sharingPolicy: '单一核算主体专用', status: '有效' },
  { accountId: 'CUST-PF03', name: '固收增强三号托管证券账户', kind: '托管证券账户', institution: '示例托管银行', legalOwnerEntityId: 'AE-PF-DEMO-03', currency: 'CNY', sharingPolicy: '单一核算主体专用', status: '有效' },
]

export const portfolioAccountRelationships: PortfolioAccountRelationship[] = [
  { relationshipId: 'REL-PF01-EXEC', portfolioId: 'PF-DEMO-01', accountId: 'EXEC-FM-DEMO-SSE', purpose: '场内集中交易', virtualSubAccountId: 'VSA-PF01-SSE', effectiveFrom: '2025-01-16', status: '有效' },
  { relationshipId: 'REL-PF01-CIBM', portfolioId: 'PF-DEMO-01', accountId: 'EXEC-FM-DEMO-CIBM', purpose: '银行间询价与成交', virtualSubAccountId: 'VSA-PF01-CIBM', effectiveFrom: '2025-01-16', status: '有效' },
  { relationshipId: 'REL-PF01-CASH', portfolioId: 'PF-DEMO-01', accountId: 'CASH-PF01-CNY', purpose: '现金收付与证券交收', effectiveFrom: '2025-01-16', status: '有效' },
  { relationshipId: 'REL-PF01-CUST', portfolioId: 'PF-DEMO-01', accountId: 'CUST-PF01', purpose: '证券托管与持仓核对', effectiveFrom: '2025-01-16', status: '有效' },
  { relationshipId: 'REL-PF01-TA', portfolioId: 'PF-DEMO-01', accountId: 'TA-PF01', purpose: '份额登记与申赎核对', effectiveFrom: '2025-01-16', status: '有效' },
  { relationshipId: 'REL-PF02-EXEC', portfolioId: 'PF-DEMO-02', accountId: 'EXEC-FM-DEMO-02-SZSE', purpose: '场内集中交易', virtualSubAccountId: 'VSA-PF02-SZSE', effectiveFrom: '2025-06-05', status: '有效' },
  { relationshipId: 'REL-PF02-CASH', portfolioId: 'PF-DEMO-02', accountId: 'CASH-PF02-CNY', purpose: '现金收付与证券交收', effectiveFrom: '2025-06-05', status: '有效' },
  { relationshipId: 'REL-PF02-CUST', portfolioId: 'PF-DEMO-02', accountId: 'CUST-PF02', purpose: '证券托管与持仓核对', effectiveFrom: '2025-06-05', status: '有效' },
  { relationshipId: 'REL-PF03-EXEC', portfolioId: 'PF-DEMO-03', accountId: 'EXEC-FM-DEMO-SSE', purpose: '场内集中交易', virtualSubAccountId: 'VSA-PF03-SSE', effectiveFrom: '2023-03-21', status: '有效' },
  { relationshipId: 'REL-PF03-CASH', portfolioId: 'PF-DEMO-03', accountId: 'CASH-PF03-CNY', purpose: '现金收付与证券交收', effectiveFrom: '2023-03-21', status: '有效' },
  { relationshipId: 'REL-PF03-CUST', portfolioId: 'PF-DEMO-03', accountId: 'CUST-PF03', purpose: '证券托管与持仓核对', effectiveFrom: '2023-03-21', status: '有效' },
]

export const portfolioSleeves: PortfolioSleeve[] = [
  { sleeveId: 'SLV-PF01-CORE', portfolioId: 'PF-DEMO-01', name: '长期配置单元', accountingTreatment: '辅助核算维度' },
  { sleeveId: 'SLV-PF01-TACTICAL', portfolioId: 'PF-DEMO-01', name: '战术增强单元', accountingTreatment: '辅助核算维度' },
  { sleeveId: 'SLV-PF02-MAIN', portfolioId: 'PF-DEMO-02', name: '主投资单元', accountingTreatment: '辅助核算维度' },
  { sleeveId: 'SLV-PF03-MAIN', portfolioId: 'PF-DEMO-03', name: '主投资单元', accountingTreatment: '辅助核算维度' },
]

export const portfolioManagerAssignments: PortfolioManagerAssignment[] = [
  { assignmentId: 'PMA-PF01-001', portfolioId: 'PF-DEMO-01', portfolioManagerId: 'PM-001', portfolioManagerName: '林岚', managerEntityId: 'FM-DEMO', role: '主基金经理', sleeveId: 'SLV-PF01-CORE', effectiveFrom: '2025-01-16', status: '有效' },
  { assignmentId: 'PMA-PF01-002', portfolioId: 'PF-DEMO-01', portfolioManagerId: 'PM-002', portfolioManagerName: '周衡', managerEntityId: 'FM-DEMO', role: '联席基金经理', sleeveId: 'SLV-PF01-TACTICAL', effectiveFrom: '2026-01-05', status: '有效' },
  { assignmentId: 'PMA-PF02-001', portfolioId: 'PF-DEMO-02', portfolioManagerId: 'PM-003', portfolioManagerName: '王宁', managerEntityId: 'FM-DEMO-02', role: '投资经理', sleeveId: 'SLV-PF02-MAIN', effectiveFrom: '2025-06-05', status: '有效' },
  { assignmentId: 'PMA-PF03-001', portfolioId: 'PF-DEMO-03', portfolioManagerId: 'PM-001', portfolioManagerName: '林岚', managerEntityId: 'FM-DEMO', role: '主基金经理', sleeveId: 'SLV-PF03-MAIN', effectiveFrom: '2023-03-21', status: '有效' },
]

export const getExternalAccount = (accountId: string) => externalAccounts.find((account) => account.accountId === accountId)

export const getAccountRelationshipsForPortfolio = (portfolioId: string) => portfolioAccountRelationships.filter((relationship) => relationship.portfolioId === portfolioId)

export const getPortfolioManagerAssignments = (portfolioId: string) => portfolioManagerAssignments.filter((assignment) => assignment.portfolioId === portfolioId)

export const getSettlementAccountForPortfolio = (portfolioId: string) => {
  const relationship = portfolioAccountRelationships.find((item) => item.portfolioId === portfolioId && getExternalAccount(item.accountId)?.kind === '资金结算账户' && item.status === '有效')
  return relationship ? getExternalAccount(relationship.accountId) : undefined
}
