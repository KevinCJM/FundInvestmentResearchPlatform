import { useActualPortfolio } from '../app/ActualPortfolioContext'

export default function ActualPortfolioSelector({ compact = false }: { compact?: boolean }) {
  const { portfolios, selectedPortfolio, selectedPortfolioId, selectPortfolio } = useActualPortfolio()

  return (
    <label className={`block min-w-0 rounded-xl border border-accent-200 bg-accent-50 text-sm ${compact ? 'px-3 py-2 sm:flex sm:flex-wrap sm:items-center sm:gap-x-3' : 'px-4 py-3'}`}>
      <span className="font-semibold text-accent-950">当前真实组合</span>
      <select
        aria-label="当前真实组合"
        value={selectedPortfolioId}
        onChange={(event) => selectPortfolio(event.target.value)}
        className={`block min-h-10 w-full min-w-0 rounded-lg border border-accent-200 bg-white px-2 py-1 text-accent-950 ${compact ? 'mt-1 sm:mt-0 sm:w-64' : 'mt-2'}`}
      >
        {portfolios.map((portfolio) => (
          <option key={portfolio.portfolioId} value={portfolio.portfolioId}>
            {portfolio.name} · {portfolio.portfolioId} · {portfolio.status}
          </option>
        ))}
      </select>
      <span className={`mt-1 block text-xs text-accent-700 ${compact ? 'sm:mt-0' : ''}`}>主账簿 {selectedPortfolio.primaryLedgerId} · {selectedPortfolio.baseCurrency}</span>
    </label>
  )
}
