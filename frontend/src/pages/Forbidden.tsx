import React from 'react';
import { useAuth } from '../context/AuthContext';

const Forbidden: React.FC = () => {
  const { user } = useAuth();

  return (
    <div className="mx-auto max-w-3xl px-4 py-16">
      <div className="rounded-xl bg-white p-8 text-center shadow">
        <h1 className="text-2xl font-semibold text-gray-900">抱歉，您没有访问权限</h1>
        <p className="mt-4 text-gray-600">
          当前账号权限等级：
          <span className="ml-2 rounded bg-gray-100 px-2 py-1 text-sm font-medium text-gray-800">
            {user?.role === 'admin' && '管理员'}
            {user?.role === 'advanced' && '高级用户'}
            {user?.role === 'basic' && '普通用户'}
          </span>
        </p>
        <p className="mt-2 text-gray-500">如需使用该功能，请联系管理员升级权限。</p>
      </div>
    </div>
  );
};

export default Forbidden;
