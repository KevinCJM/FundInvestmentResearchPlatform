import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';

const navItems = [
  { path: '/', label: '全景驾驶舱' },
  { path: '/research', label: '产品研究' },
  { path: '/evaluation-plan', label: '评价方案' },
  { path: '/indicator-studio', label: '指标中心' },
  { path: '/manual-construction', label: '手动构建大类' },
  { path: '/auto-classification', label: '自动构建大类' },
  { path: '/class-allocation', label: '大类资产配置' },
  { path: '/portfolio-construction', label: '产品组合构建' },
  { path: '/holding-diagnosis', label: '持仓诊断' },
];

export default function Header() {
  const [menuOpen, setMenuOpen] = useState(false);
  const baseStyle = 'whitespace-nowrap px-4 py-2 rounded-md text-sm font-medium';
  const activeStyle = 'bg-gray-900 text-white';
  const inactiveStyle = 'text-gray-500 hover:bg-gray-700 hover:text-white';

  return (
    <header className="sticky top-0 z-30 bg-gray-900 shadow" data-testid="site-header">
      <nav className="mx-auto max-w-7xl px-4 py-3 sm:px-6" aria-label="主导航">
        <div className="flex items-center justify-between gap-3">
          <NavLink to="/" className="shrink-0 text-sm font-bold tracking-wide text-white" aria-label="基金研究平台主页">基金研究平台</NavLink>
          <button type="button" className="rounded-md border border-gray-600 px-3 py-2 text-sm font-medium text-white focus:outline-none focus:ring-2 focus:ring-violet-300 lg:hidden" aria-expanded={menuOpen} aria-controls="mobile-navigation" onClick={() => setMenuOpen((open) => !open)}>菜单</button>
          <div className="hidden items-center space-x-1 lg:flex">
          {navItems.map((item) => (
            <NavLink
              key={item.label}
              to={item.path}
              end={item.path === '/'} // `end` 仅用于根路径，避免前缀匹配
              className={({ isActive }) => `${baseStyle} ${isActive ? activeStyle : inactiveStyle}`}
            >
              {item.label}
            </NavLink>
          ))}
          </div>
        </div>
        {menuOpen && <div id="mobile-navigation" className="mt-3 grid gap-1 border-t border-gray-700 pt-3 lg:hidden">
          {navItems.map((item) => (
            <NavLink key={item.label} to={item.path} end={item.path === '/'} onClick={() => setMenuOpen(false)} className={({ isActive }) => `${baseStyle} text-left ${isActive ? activeStyle : inactiveStyle}`}>{item.label}</NavLink>
          ))}
        </div>}
      </nav>
    </header>
  );
}
