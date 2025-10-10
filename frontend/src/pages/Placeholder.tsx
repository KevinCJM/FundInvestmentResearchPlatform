import React from 'react';

export default function Placeholder({ title }: { title: string }) {
  return (
    <div className="mx-auto max-w-5xl p-6">
      <h1 className="text-2xl font-semibold">{title}</h1>
      <p className="mt-4 text-gray-600">此功能正在开发中，敬请期待。</p>
    </div>
  );
}