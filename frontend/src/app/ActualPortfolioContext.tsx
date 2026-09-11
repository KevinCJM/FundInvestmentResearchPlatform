import { createContext, useContext, useMemo, useState, type ReactNode } from 'react'
import { actualPortfolioDemoData, type ActualPortfolio } from './actualPortfolioDemoData'

interface ActualPortfolioContextValue {
  portfolios: ActualPortfolio[]
  selectedPortfolio: ActualPortfolio
  selectedPortfolioId: string
  selectPortfolio: (portfolioId: string) => void
}

const ActualPortfolioContext = createContext<ActualPortfolioContextValue | null>(null)

export function ActualPortfolioProvider({ children }: { children: ReactNode }) {
  const defaultPortfolio = actualPortfolioDemoData.find((portfolio) => portfolio.status === '运行中') ?? actualPortfolioDemoData[0]
  const [selectedPortfolioId, setSelectedPortfolioId] = useState(defaultPortfolio.portfolioId)
  const selectedPortfolio = useMemo(
    () => actualPortfolioDemoData.find((portfolio) => portfolio.portfolioId === selectedPortfolioId) ?? defaultPortfolio,
    [defaultPortfolio, selectedPortfolioId],
  )

  return (
    <ActualPortfolioContext.Provider value={{ portfolios: actualPortfolioDemoData, selectedPortfolio, selectedPortfolioId, selectPortfolio: setSelectedPortfolioId }}>
      {children}
    </ActualPortfolioContext.Provider>
  )
}

export function useActualPortfolio() {
  const context = useContext(ActualPortfolioContext)
  if (!context) throw new Error('useActualPortfolio must be used inside ActualPortfolioProvider')
  return context
}
