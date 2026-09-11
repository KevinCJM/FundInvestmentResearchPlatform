import {
  ArrowPathIcon, ChartBarIcon, ChartPieIcon, CircleStackIcon, Cog6ToothIcon,
  CubeIcon, GlobeAltIcon, MagnifyingGlassIcon, PaperAirplaneIcon,
  PresentationChartLineIcon, RectangleGroupIcon, ScaleIcon, Squares2X2Icon,
} from '@heroicons/react/24/outline'
import type { HomeIcon as IconName } from './catalog'

const icons = {
  search: MagnifyingGlassIcon, chart: ChartBarIcon, send: PaperAirplaneIcon,
  pie: ChartPieIcon, repeat: ArrowPathIcon, database: CircleStackIcon,
  network: RectangleGroupIcon, market: PresentationChartLineIcon, cube: CubeIcon,
  grid: Squares2X2Icon, settings: Cog6ToothIcon, accounting: ScaleIcon, globe: GlobeAltIcon,
}
export function HomeIcon({ name, className = '' }: { name: IconName | 'globe'; className?: string }) {
  const Icon = icons[name]
  return <Icon aria-hidden="true" className={`home-icon ${className}`} />
}
