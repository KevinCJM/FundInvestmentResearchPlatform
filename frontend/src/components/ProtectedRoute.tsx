import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import Forbidden from '../pages/Forbidden';
import type { UserRole } from '../context/AuthContext';

interface ProtectedRouteProps {
  allow?: UserRole[];
}

const defaultRoles: UserRole[] = ['basic', 'advanced', 'admin'];

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ allow = defaultRoles }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-100">
        <div className="rounded-lg bg-white px-6 py-4 shadow">正在加载权限...</div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  if (!allow.includes(user.role)) {
    return <Forbidden />;
  }

  return <Outlet />;
};

export default ProtectedRoute;
