import React from 'react';
import { NavLink } from 'react-router-dom';

const navItems = [
  { path: '/', label: '主界面' },
  { path: '/research', label: '产品研究' },
  { path: '/manual-construction', label: '手动构建大类' },
  { path: '/auto-classification', label: '自动构建大类' },
  { path: '/class-allocation', label: '大类资产配置' },
  { path: '/portfolio-construction', label: '产品组合构建' },
];

export default function Header() {
  const baseStyle = 'px-4 py-2 rounded-md text-sm font-medium';
  const activeStyle = 'bg-gray-900 text-white';
  const inactiveStyle = 'text-gray-500 hover:bg-gray-700 hover:text-white';

  return (
    <header className="bg-gray-800 shadow">
      <nav className="mx-auto max-w-5xl px-6 py-3">
        <div className="flex items-center space-x-4">
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
      </nav>
    </header>
  );
}
