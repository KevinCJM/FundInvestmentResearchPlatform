import { describe, expect, it } from 'vitest'
import {
  actualPortfolioDemoData,
  externalAccounts,
  getExternalAccount,
  portfolioAccountRelationships,
  portfolioManagerAssignments,
  portfolioSleeves,
} from './actualPortfolioDemoData'

describe('真实组合账户与责任关系示例', () => {
  it('共享执行通道可服务多个组合，但不共享基金账', () => {
    const relatedPortfolios = portfolioAccountRelationships
      .filter((relationship) => relationship.accountId === 'EXEC-FM-DEMO-SSE')
      .map((relationship) => relationship.portfolioId)

    expect(relatedPortfolios).toEqual(['PF-DEMO-01', 'PF-DEMO-03'])
    expect(getExternalAccount('EXEC-FM-DEMO-SSE')).toMatchObject({
      kind: '执行通道账户',
      sharingPolicy: '允许多组合共享执行',
    })
  })

  it('每个有效资金结算账户只服务一个核算主体', () => {
    const settlementAccounts = externalAccounts.filter((account) => account.kind === '资金结算账户')

    settlementAccounts.forEach((account) => {
      const relationships = portfolioAccountRelationships.filter((relationship) => relationship.accountId === account.accountId && relationship.status === '有效')
      expect(relationships).toHaveLength(1)
      const portfolio = actualPortfolioDemoData.find((item) => item.portfolioId === relationships[0].portfolioId)
      expect(account.sharingPolicy).toBe('单一核算主体专用')
      expect(account.legalOwnerEntityId).toBe(portfolio?.accountingEntityId)
    })
  })

  it('基金经理任职和内部投资单元必须属于同一组合', () => {
    portfolioManagerAssignments.forEach((assignment) => {
      const sleeve = portfolioSleeves.find((item) => item.sleeveId === assignment.sleeveId)
      expect(sleeve?.portfolioId).toBe(assignment.portfolioId)
    })
  })
})
