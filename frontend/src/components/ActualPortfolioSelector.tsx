import { useActualPortfolio } from '../app/ActualPortfolioContext'

export default function ActualPortfolioSelector({ compact = false }: { compact?: boolean }) {
  const { portfolios, selectedPortfolio, selectedPortfolioId, selectPortfolio } = useActualPortfolio()

  return (
    <label className="block rounded-xl border border-indigo-200 bg-indigo-50 px-4 py-3 text-sm">
      <span className="font-semibold text-indigo-950">当前真实组合</span>
      <select
        aria-label="当前真实组合"
        value={selectedPortfolioId}
        onChange={(event) => selectPortfolio(event.target.value)}
        className={`block rounded-lg border border-indigo-200 bg-white px-3 py-2 text-indigo-950 ${compact ? 'mt-2 w-full lg:min-w-72' : 'mt-2 w-full'}`}
      >
        {portfolios.map((portfolio) => (
          <option key={portfolio.portfolioId} value={portfolio.portfolioId}>
            {portfolio.name} · {portfolio.portfolioId} · {portfolio.status}
          </option>
        ))}
      </select>
      <span className="mt-1 block text-xs text-indigo-700">主账簿 {selectedPortfolio.primaryLedgerId} · {selectedPortfolio.baseCurrency}</span>
    </label>
  )
}
