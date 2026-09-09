import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { ActualPortfolioProvider } from './app/ActualPortfolioContext'
import { ResearchContextProvider } from './app/ResearchContext'
import PitSnapshots from './pages/PitSnapshots'
import Header from './components/Header'
import LocalizationProvider from './i18n/LocalizationProvider'
import LanguageTerminology from './pages/LanguageTerminology'
import LegacyRedirect from './components/LegacyRedirect'
import StageLayout from './layouts/StageLayout'
import AutoAssetClassification from './pages/AutoAssetClassification'
import ClassAllocation from './pages/ClassAllocation'
import AccountingBookingWorkspace from './pages/AccountingBookingWorkspace'
import AccountStatementAllocationWorkspace from './pages/AccountStatementAllocationWorkspace'
import FinancialStatementsWorkspace from './pages/FinancialStatementsWorkspace'
import Dashboard from './pages/Dashboard'
import DataManagement from './pages/DataManagement'
import DataModelCatalog from './pages/DataModelCatalog'
import DataSourceCenter from './pages/DataSourceCenter'
import EvaluationPlan from './pages/EvaluationPlan'
import HoldingDiagnosis from './pages/HoldingDiagnosis'
import HistoricalRegimeWorkbench from './pages/HistoricalRegimeWorkbench'
import DataQuality from './pages/IndexData'
import IndicatorStudio from './pages/IndicatorStudio'
import FactorResearchCenter from './pages/FactorResearchCenter'
import ManualConstruction from './pages/ManualConstruction'
import PortfolioConstruction from './pages/PortfolioConstruction'
import PortfolioCenterWorkspace from './pages/PortfolioCenterWorkspace'
import PortfolioOnboardingWorkspace from './pages/PortfolioOnboardingWorkspace'
import PortfolioSolutionCatalog from './pages/PortfolioSolutionCatalog'
import PortfolioSolutionShowcase from './pages/PortfolioSolutionShowcase'
import ProcessHome from './pages/ProcessHome'
import ProductCompare from './pages/ProductCompare'
import ProductDetail from './pages/ProductDetail'
import ProductResearch from './pages/ProductResearch'
import ProductPools from './pages/ProductPools'
import ProductPoolLifecycle from './pages/ProductPoolLifecycle'
import ProductPoolSelection from './pages/ProductPoolSelection'
import PrototypeWorkspace from './pages/PrototypeWorkspace'
import ResearchDataLab from './pages/ResearchDataLab'
import ScenarioCenters from './pages/ScenarioCenters'
import StageOverview from './pages/StageOverview'
import ToolHubPage from './pages/ToolHubPage'
import TradeAllocationWorkspace from './pages/TradeAllocationWorkspace'
import TacticalAllocationWorkspace from './pages/TacticalAllocationWorkspace'

const prototypePage = (pageKey: string) => <PrototypeWorkspace key={pageKey} pageKey={pageKey} />

const saaHub = (
  <ToolHubPage
    title="战略资产配置（SAA）"
    description="研究长期大类资产中枢、权重区间、风险预算、有效前沿与不同固定配置的历史表现。"
    tools={[
      { label: '大类资产构建', description: '定义大类及 ETF/公募基金代理，完成拟合、相关性与配置保存。', path: '/pre-investment/saa/asset-classes' },
      { label: '大类资产配置与策略回测', description: '研究有效前沿、风险预算、目标权重和配置回测。', path: '/pre-investment/saa/allocation-lab' },
      { label: '自动构建大类', description: '按收益相关性、风险画像或主成分自动划分大类，并给出代表产品、类内权重与分类诊断。', path: '/pre-investment/saa/auto-classification' },
    ]}
  />
)

export default function App() {
  return (
    <LocalizationProvider><BrowserRouter>
      <ActualPortfolioProvider>
        <ResearchContextProvider>
        <div className="min-h-screen bg-gray-100">
          <Header />
          <main>
          <Routes>
            <Route path="/" element={<ProcessHome />} />

            <Route path="/product-research" element={<StageLayout stageId="product-research" />}>
              <Route index element={<StageOverview stageId="product-research" />} />
              <Route path="panorama" element={<Dashboard />} />
              <Route path="framework-data" element={<LegacyRedirect to="/settings/research-parameters" />} />
              <Route path="products" element={<ProductResearch />} />
              <Route path="products/:productId" element={<ProductDetail />} />
              <Route path="holding-style" element={prototypePage('holding-style')} />
              <Route path="compare" element={<ProductCompare />} />
              <Route path="evaluation" element={<EvaluationPlan />} />
              <Route path="product-backtest" element={prototypePage('product-backtest')} />
              <Route path="pools" element={<ProductPools />} />
              <Route path="pool-lifecycle" element={<ProductPoolLifecycle />} />
            </Route>

            <Route path="/pre-investment" element={<StageLayout stageId="pre-investment" />}>
              <Route index element={<StageOverview stageId="pre-investment" />} />
              <Route path="objectives" element={prototypePage('objectives')} />
              <Route path="product-pool" element={<ProductPoolSelection />} />
              <Route path="saa" element={saaHub} />
              <Route path="saa/asset-classes" element={<ManualConstruction />} />
              <Route path="saa/auto-classification" element={<AutoAssetClassification />} />
              <Route path="saa/allocation-lab" element={<ClassAllocation />} />
              <Route path="taa" element={<TacticalAllocationWorkspace />} />
              <Route path="product-allocation-timing" element={prototypePage('product-allocation-timing')} />
              <Route path="product-allocation-timing/construction" element={<PortfolioConstruction />} />
              <Route path="product-allocation-timing/timing" element={prototypePage('product-timing')} />
              <Route path="portfolio-synthesis" element={prototypePage('portfolio-synthesis')} />
              <Route path="validation" element={prototypePage('validation')} />
              <Route path="approval" element={prototypePage('approval')} />
            </Route>

            <Route path="/investment-execution" element={<StageLayout stageId="investment-execution" />}>
              <Route index element={<StageOverview stageId="investment-execution" />} />
              <Route path="onboarding" element={<PortfolioOnboardingWorkspace />} />
              <Route path="trade-plan" element={prototypePage('trade-plan')} />
              <Route path="pre-trade-check" element={prototypePage('pre-trade-check')} />
              <Route path="trade-allocation" element={<TradeAllocationWorkspace />} />
              <Route path="cash-settlement" element={prototypePage('cash-settlement')} />
              <Route path="execution" element={<LegacyRedirect to="/fund-accounting/booking" />} />
              <Route path="reconciliation" element={prototypePage('reconciliation')} />
              <Route path="rebalancing" element={prototypePage('rebalancing')} />
            </Route>

            <Route path="/portfolio-center" element={<StageLayout stageId="portfolio-center" />}>
              <Route index element={<StageOverview stageId="portfolio-center" />} />
              <Route path="register" element={<PortfolioCenterWorkspace view="register" />} />
              <Route path="master" element={<PortfolioCenterWorkspace view="master" />} />
              <Route path="accounts" element={<PortfolioCenterWorkspace view="relationships" />} />
              <Route path="relationships" element={<LegacyRedirect to="/portfolio-center/accounts" />} />
              <Route path="responsibilities" element={<PortfolioCenterWorkspace view="responsibilities" />} />
              <Route path="versions" element={<PortfolioCenterWorkspace view="versions" />} />
              <Route path="lifecycle" element={<PortfolioCenterWorkspace view="lifecycle" />} />
            </Route>

            <Route path="/fund-accounting" element={<StageLayout stageId="fund-accounting" />}>
              <Route index element={<StageOverview stageId="fund-accounting" />} />
              <Route path="policy" element={prototypePage('accounting-policy')} />
              <Route path="account-statements" element={<AccountStatementAllocationWorkspace />} />
              <Route path="booking" element={<AccountingBookingWorkspace />} />
              <Route path="portfolio-ledger" element={prototypePage('portfolio-ledger')} />
              <Route path="manager-ledger" element={prototypePage('manager-ledger')} />
              <Route path="valuation-nav" element={prototypePage('accounting-valuation')} />
              <Route path="cross-book-reconciliation" element={prototypePage('cross-book-reconciliation')} />
              <Route path="financial-statements" element={<FinancialStatementsWorkspace />} />
              <Route path="performance-dataset" element={prototypePage('performance-dataset')} />
              <Route path="ledger" element={<LegacyRedirect to="/fund-accounting/portfolio-ledger" />} />
              <Route path="reconciliation-close" element={<LegacyRedirect to="/fund-accounting/cross-book-reconciliation" />} />
            </Route>

            <Route path="/post-investment" element={<StageLayout stageId="post-investment" />}>
              <Route index element={<StageOverview stageId="post-investment" />} />
              <Route path="portfolio-data" element={prototypePage('portfolio-data')} />
              <Route path="performance" element={prototypePage('performance')} />
              <Route path="return-attribution" element={prototypePage('return-attribution')} />
              <Route path="risk-attribution" element={prototypePage('risk-attribution')} />
              <Route path="monitoring" element={prototypePage('monitoring')} />
              <Route path="scenarios" element={prototypePage('scenarios')} />
              <Route path="conclusions" element={prototypePage('conclusions')} />
              <Route path="reports" element={prototypePage('reports')} />
              <Route path="research-diagnosis" element={<HoldingDiagnosis />} />
              <Route path="actions" element={<LegacyRedirect to="/post-investment/conclusions" />} />
            </Route>

            <Route path="/feedback" element={<StageLayout stageId="feedback" />}>
              <Route index element={<StageOverview stageId="feedback" />} />
              <Route path="results" element={prototypePage('results')} />
              <Route path="hypothesis-review" element={prototypePage('hypothesis-review')} />
              <Route path="product-review" element={prototypePage('product-review')} />
              <Route path="model-update" element={prototypePage('model-update')} />
              <Route path="rolling-validation" element={prototypePage('rolling-validation')} />
              <Route path="process-improvement" element={prototypePage('process-improvement')} />
            </Route>

            <Route path="/portfolio-solutions" element={<StageLayout stageId="portfolio-solutions" />}>
              <Route index element={<StageOverview stageId="portfolio-solutions" />} />
              <Route path="catalog" element={<PortfolioSolutionCatalog />} />
              <Route path="profile" element={<PortfolioSolutionShowcase view="profile" />} />
              <Route path="performance" element={<PortfolioSolutionShowcase view="performance" />} />
              <Route path="backtest" element={<PortfolioSolutionShowcase view="backtest" />} />
              <Route path="scenarios" element={<PortfolioSolutionShowcase view="scenarios" />} />
              <Route path="rules" element={<PortfolioSolutionShowcase view="rules" />} />
              <Route path="publishing" element={<PortfolioSolutionShowcase view="publishing" />} />
            </Route>

            <Route path="/settings" element={<StageLayout stageId="settings" />}>
              <Route index element={<StageOverview stageId="settings" />} />
              <Route path="data-model" element={<DataModelCatalog />} />
              <Route path="source-center" element={<DataSourceCenter />} />
              <Route path="data-sources" element={<DataManagement />} />
              <Route path="data-refresh" element={<LegacyRedirect to="/settings/data-sources" />} />
              <Route path="data-quality" element={<DataQuality />} />
              <Route path="research-data-lab" element={<ResearchDataLab />} />
              <Route path="pit-snapshots" element={<PitSnapshots />} />
              <Route path="research-parameters" element={prototypePage('research-parameters')} />
              <Route path="indicators-models" element={<IndicatorStudio />} />
              <Route path="factor-research" element={<FactorResearchCenter />} />
              <Route path="scenario-algorithms" element={<ScenarioCenters />} />
              <Route path="scenario-algorithms/workbench" element={<HistoricalRegimeWorkbench />} />
              <Route path="backtest-center" element={prototypePage('backtest-center')} />
              <Route path="system-parameters" element={prototypePage('system-parameters')} />
              <Route path="language-terminology" element={<LanguageTerminology />} />
            </Route>

            <Route path="/research" element={<LegacyRedirect to="/product-research/products" />} />
            <Route path="/evaluation-plan" element={<LegacyRedirect to="/product-research/evaluation" />} />
            <Route path="/indicator-studio" element={<LegacyRedirect to="/settings/indicators-models" />} />
            <Route path="/product/:productId" element={<LegacyRedirect to={(params) => `/product-research/products/${encodeURIComponent(params.productId ?? '')}`} />} />
            <Route path="/product-compare" element={<LegacyRedirect to="/product-research/compare" />} />
            <Route path="/manual-construction" element={<LegacyRedirect to="/pre-investment/saa/asset-classes" />} />
            <Route path="/auto-classification" element={<LegacyRedirect to="/pre-investment/saa/auto-classification" />} />
            <Route path="/class-allocation" element={<LegacyRedirect to="/pre-investment/saa/allocation-lab" />} />
            <Route path="/portfolio-construction" element={<LegacyRedirect to="/pre-investment/product-allocation-timing/construction" />} />
            <Route path="/holding-diagnosis" element={<LegacyRedirect to="/post-investment/research-diagnosis" />} />
            <Route path="/index-data" element={<LegacyRedirect to="/settings/data-quality" />} />
            <Route path="*" element={<Navigate replace to="/" />} />
          </Routes>
          </main>
        </div>
        </ResearchContextProvider>
      </ActualPortfolioProvider>
    </BrowserRouter></LocalizationProvider>
  )
}
