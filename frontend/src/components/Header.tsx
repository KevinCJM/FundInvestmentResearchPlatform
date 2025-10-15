import React from 'react';
import { NavLink } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const navItems = [
  { path: '/', label: '主界面', roles: ['basic', 'advanced', 'admin'] },
  { path: '/research', label: '产品研究', roles: ['basic', 'advanced', 'admin'] },
  { path: '/manual-construction', label: '手动构建大类', roles: ['advanced', 'admin'] },
  { path: '/auto-classification', label: '自动构建大类', roles: ['advanced', 'admin'] },
  { path: '/class-allocation', label: '大类资产配置', roles: ['advanced', 'admin'] },
  { path: '/portfolio-construction', label: '产品组合构建', roles: ['advanced', 'admin'] },
  { path: '/parameter-settings', label: '参数配置', roles: ['admin'] },
];

const roleLabels: Record<string, string> = {
  admin: '管理员',
  advanced: '高级用户',
  basic: '普通用户',
};

export default function Header() {
  const { user, logout } = useAuth();

  if (!user) {
    return null;
  }

  const baseStyle = 'px-4 py-2 rounded-md text-sm font-medium';
  const activeStyle = 'bg-gray-900 text-white';
  const inactiveStyle = 'text-gray-300 hover:bg-gray-700 hover:text-white';

  const allowedItems = navItems.filter((item) => item.roles.includes(user.role));

  return (
    <header className="bg-gray-800 shadow">
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
        <div className="flex items-center space-x-3">
          <span className="text-lg font-semibold text-white">基金投研平台</span>
          <div className="flex items-center space-x-2">
            {allowedItems.map((item) => (
              <NavLink
                key={item.label}
                to={item.path}
                end={item.path === '/'}
                className={({ isActive }) => `${baseStyle} ${isActive ? activeStyle : inactiveStyle}`}
              >
                {item.label}
              </NavLink>
            ))}
          </div>
        </div>
        <div className="flex items-center space-x-3 text-sm text-gray-200">
          <div className="rounded-full bg-gray-700 px-3 py-1 text-xs font-medium text-gray-200">
            {roleLabels[user.role] ?? user.role}
          </div>
          <span className="font-medium text-white">{user.displayName}</span>
          <button
            onClick={logout}
            className="rounded-md border border-gray-500 px-3 py-1 text-xs font-medium text-gray-200 transition hover:border-white hover:text-white"
          >
            退出登录
          </button>
        </div>
      </nav>
    </header>
  );
}
