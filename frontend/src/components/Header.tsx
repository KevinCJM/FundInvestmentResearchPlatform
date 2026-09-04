import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';

const navItems = [
  { path: '/', label: '首页' },
  { path: '/product-research', label: '产品研究' },
  { path: '/pre-investment', label: '投前决策' },
  { path: '/portfolio-center', label: '组合中心' },
  { path: '/investment-execution', label: '投中执行' },
  { path: '/fund-accounting', label: '基金会计' },
  { path: '/post-investment', label: '投后管理' },
  { path: '/feedback', label: '反馈与迭代' },
  { path: '/portfolio-solutions', label: '方案展示' },
  { path: '/settings', label: '设置' },
];

export default function Header() {
  const [menuOpen, setMenuOpen] = useState(false);
  const baseStyle = 'whitespace-nowrap px-3 py-2 rounded-md text-sm font-medium';
  const activeStyle = 'bg-gray-900 text-white';
  const inactiveStyle = 'text-gray-500 hover:bg-gray-700 hover:text-white';

  return (
    <header className="sticky top-0 z-30 border-b border-slate-800 bg-slate-950 shadow" data-testid="site-header">
      <nav className="mx-auto max-w-[1600px] px-4 py-3 sm:px-6 lg:px-8" aria-label="主导航">
        <div className="flex items-center justify-between gap-3">
          <NavLink to="/" className="shrink-0 text-sm font-bold tracking-wide text-white" aria-label="基金量化投研平台主页">基金量化投研平台</NavLink>
          <button type="button" className="rounded-md border border-slate-600 px-3 py-2 text-sm font-medium text-white focus:outline-none focus:ring-2 focus:ring-sky-300 xl:hidden" aria-expanded={menuOpen} aria-controls="mobile-navigation" onClick={() => setMenuOpen((open) => !open)}>菜单</button>
          <div className="hidden items-center space-x-1 xl:flex">
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
        {menuOpen && <div id="mobile-navigation" className="mt-3 grid gap-1 border-t border-slate-700 pt-3 xl:hidden">
          {navItems.map((item) => (
            <NavLink key={item.label} to={item.path} end={item.path === '/'} onClick={() => setMenuOpen(false)} className={({ isActive }) => `${baseStyle} text-left ${isActive ? activeStyle : inactiveStyle}`}>{item.label}</NavLink>
          ))}
        </div>}
      </nav>
    </header>
  );
}
