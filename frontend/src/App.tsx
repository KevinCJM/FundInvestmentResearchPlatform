import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import ManualConstruction from './pages/ManualConstruction';
import ClassAllocation from './pages/ClassAllocation';
import Placeholder from './pages/Placeholder';
import ProductResearch from './pages/ProductResearch';
import ProductDetail from './pages/ProductDetail';

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-100">
        <Header />
        <main>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/research" element={<ProductResearch />} />
            <Route path="/product/:productId" element={<ProductDetail />} />
            <Route path="/manual-construction" element={<ManualConstruction />} />
            <Route path="/auto-classification" element={<Placeholder title="自动构建大类" />} />
            <Route path="/class-allocation" element={<ClassAllocation />} />
            <Route path="/portfolio-construction" element={<Placeholder title="产品组合构建" />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
