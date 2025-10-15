import React from 'react';
import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import ManualConstruction from './pages/ManualConstruction';
import ClassAllocation from './pages/ClassAllocation';
import Placeholder from './pages/Placeholder';
import ProductResearch from './pages/ProductResearch';
import ProductDetail from './pages/ProductDetail';
import Login from './pages/Login';
import Forbidden from './pages/Forbidden';
import AdminSettings from './pages/AdminSettings';
import ProtectedRoute from './components/ProtectedRoute';
import { AuthProvider, useAuth } from './context/AuthContext';

const AppShell: React.FC = () => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gray-100">
        <div className="rounded-lg bg-white px-6 py-4 shadow">正在初始化会话...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      {user && <Header />}
      <main>
        <Outlet />
      </main>
    </div>
  );
};

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route element={<AppShell />}>
            <Route element={<ProtectedRoute allow={['basic', 'advanced', 'admin']} />}>
              <Route index element={<Dashboard />} />
              <Route path="research" element={<ProductResearch />} />
              <Route path="product/:productId" element={<ProductDetail />} />
            </Route>
            <Route element={<ProtectedRoute allow={['advanced', 'admin']} />}>
              <Route path="manual-construction" element={<ManualConstruction />} />
              <Route path="auto-classification" element={<Placeholder title="自动构建大类" />} />
              <Route path="class-allocation" element={<ClassAllocation />} />
              <Route path="portfolio-construction" element={<Placeholder title="产品组合构建" />} />
            </Route>
            <Route element={<ProtectedRoute allow={['admin']} />}>
              <Route path="parameter-settings" element={<AdminSettings />} />
            </Route>
            <Route path="forbidden" element={<Forbidden />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
