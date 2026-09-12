import React, { useEffect, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { recordRecentVisit } from '../homepage/history';
import PitBadge from './PitBadge';
import { availableLanguages, chooseLocale, useI18n } from '../i18n/runtime';
import { routeTranslationKey, type Locale } from '../i18n/catalogs';

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
  const { s, locale } = useI18n();
  const [menuOpen, setMenuOpen] = useState(false);
  const { pathname } = useLocation();
  useEffect(() => { recordRecentVisit(pathname); setMenuOpen(false); }, [pathname]);
  const baseStyle = 'whitespace-nowrap px-3 py-2 rounded-lg text-sm font-medium';
  const activeStyle = 'bg-slate-900 text-white';
  const inactiveStyle = 'text-slate-300 hover:bg-slate-700 hover:text-white';

  // The landing page owns its light navigation; workspaces keep the existing shell.
  if (pathname === '/') return null;

  return (
    <header className="sticky top-0 z-[60] border-b border-slate-800 bg-slate-950 shadow" data-testid="site-header">
      <nav className="mx-auto max-w-[1600px] px-4 py-3 sm:px-6 lg:px-8" aria-label={s('navigation.main')}>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <NavLink to="/" className="shrink-0 text-sm font-bold tracking-wide text-white" aria-label={s('app.home')}>{s('app.title')}</NavLink>
          <label className="shrink-0"><span className="sr-only">{s('i18n.language')}</span><select aria-label={s('i18n.language')} value={locale} onChange={event => void chooseLocale(event.target.value as Locale)} className="min-h-10 max-w-[100px] rounded-lg border border-slate-600 bg-slate-900 px-2 text-xs text-white">{availableLanguages().filter(item => item.enabled).map(item => <option key={item.id} value={item.id}>{item.id === 'zh-CN' ? '中文' : item.label}</option>)}</select></label>
          <button type="button" className="rounded-lg border border-slate-600 px-3 py-2 text-sm font-medium text-white focus:outline-none focus:ring-2 focus:ring-accent-500 xl:hidden" aria-expanded={menuOpen} aria-controls="mobile-navigation" onClick={() => setMenuOpen((open) => !open)}>{s('navigation.menu')}</button>
          <div className="hidden items-center space-x-1 xl:flex">
          <PitBadge />
          {navItems.map((item) => (
            <NavLink
              key={item.label}
              to={item.path}
              end={item.path === '/'} // `end` 仅用于根路径，避免前缀匹配
              className={({ isActive }) => `${baseStyle} ${isActive ? activeStyle : inactiveStyle}`}
            >
              {s(routeTranslationKey(item.path), {}, item.label)}
            </NavLink>
          ))}
          </div>
        </div>
        {menuOpen && <div id="mobile-navigation" className="mt-3 grid gap-1 border-t border-slate-700 pt-3 xl:hidden">
          <div className="pb-1"><PitBadge /></div>
          {navItems.map((item) => (
            <NavLink key={item.label} to={item.path} end={item.path === '/'} onClick={() => setMenuOpen(false)} className={({ isActive }) => `${baseStyle} text-left ${isActive ? activeStyle : inactiveStyle}`}>{s(routeTranslationKey(item.path), {}, item.label)}</NavLink>
          ))}
        </div>}
      </nav>
    </header>
  );
}
