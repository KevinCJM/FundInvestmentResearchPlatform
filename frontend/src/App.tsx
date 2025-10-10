import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import ManualConstruction from './pages/ManualConstruction';
import ClassAllocation from './pages/ClassAllocation';
import Placeholder from './pages/Placeholder';

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-100">
        <Header />
        <main>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/manual-construction" element={<ManualConstruction />} />
            <Route path="/kmeans-construction" element={<Placeholder title="K-mean分层构建大类" />} />
            <Route path="/class-allocation" element={<ClassAllocation />} />
            <Route path="/auto-classification" element={<Placeholder title="自动ETF分类" />} />
            <Route path="/research" element={<Placeholder title="ETF产品研究" />} />
            <Route path="/portfolio-construction" element={<Placeholder title="ETF组合构建" />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}