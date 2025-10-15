import React, { useEffect, useState } from 'react';
import { useAuthToken, useAuth } from '../context/AuthContext';

interface ParameterState {
  risk_tolerance: number;
  max_positions: number;
  rebalance_frequency: string;
}

const defaultState: ParameterState = {
  risk_tolerance: 0.45,
  max_positions: 20,
  rebalance_frequency: 'monthly',
};

const AdminSettings: React.FC = () => {
  const token = useAuthToken();
  const { refresh } = useAuth();
  const [form, setForm] = useState<ParameterState>(defaultState);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!token) {
      setLoading(false);
      return;
    }
    fetch('/api/settings', {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
      .then(async (resp) => {
        if (!resp.ok) {
          throw new Error('无法加载参数配置');
        }
        return resp.json();
      })
      .then((data) => {
        setForm(data.parameters);
      })
      .catch((err: Error) => {
        setError(err.message);
      })
      .finally(() => setLoading(false));
  }, [token]);

  const handleChange = (field: keyof ParameterState, value: string) => {
    setForm((prev) => ({ ...prev, [field]: field === 'rebalance_frequency' ? value : Number(value) }));
  };

  const handleSubmit = async (evt: React.FormEvent) => {
    evt.preventDefault();
    if (!token) return;
    setSaving(true);
    setMessage(null);
    setError(null);
    try {
      const resp = await fetch('/api/settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(form),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: '保存失败' }));
        throw new Error(err.detail || '保存失败');
      }
      const data = await resp.json();
      setForm(data.parameters);
      setMessage('参数已更新');
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-16">
        <div className="rounded-xl bg-white p-6 text-center shadow">正在加载参数配置...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-16">
        <div className="rounded-xl bg-red-50 p-6 text-center text-red-600 shadow">{error}</div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-10">
      <div className="rounded-2xl bg-white p-8 shadow-lg">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-gray-900">参数配置中心</h1>
            <p className="mt-1 text-sm text-gray-500">仅管理员可修改核心参数，调整后立即影响相关模型</p>
          </div>
        </div>
        <form className="space-y-6" onSubmit={handleSubmit}>
          <div>
            <label className="block text-sm font-medium text-gray-700">风险容忍度</label>
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={form.risk_tolerance}
              onChange={(e) => handleChange('risk_tolerance', e.target.value)}
              className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
            />
            <p className="mt-1 text-xs text-gray-500">取值范围 0-1，越大表示策略接受更高波动</p>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">最大持仓数量</label>
            <input
              type="number"
              min={1}
              max={200}
              value={form.max_positions}
              onChange={(e) => handleChange('max_positions', e.target.value)}
              className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">再平衡频率</label>
            <select
              value={form.rebalance_frequency}
              onChange={(e) => handleChange('rebalance_frequency', e.target.value)}
              className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
            >
              <option value="monthly">每月</option>
              <option value="quarterly">每季度</option>
              <option value="semiannual">半年</option>
              <option value="annual">每年</option>
            </select>
          </div>
          <div className="flex items-center justify-between">
            <div>
              {message && <span className="text-sm text-green-600">{message}</span>}
              {!message && <span className="text-sm text-gray-400">提交后立即生效</span>}
            </div>
            <button
              type="submit"
              disabled={saving}
              className="rounded-lg bg-indigo-600 px-6 py-2 text-white shadow transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-indigo-400"
            >
              {saving ? '保存中...' : '保存修改'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AdminSettings;
