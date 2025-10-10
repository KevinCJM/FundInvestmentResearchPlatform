import React, { useEffect, useState } from 'react';
import ReactECharts from 'echarts-for-react';

interface ETFManagerInfo {
  management?: string;
}

export default function Dashboard() {
  const [chartOption, setChartOption] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const resp = await fetch('/api/etf/search?page_size=200');
        const data = await resp.json();
        const items: ETFManagerInfo[] = data.items || [];

        const managerCounts = items.reduce((acc, etf) => {
          const manager = etf.management || '其他';
          acc[manager] = (acc[manager] || 0) + 1;
          return acc;
        }, {} as Record<string, number>);

        const sortedManagers = Object.entries(managerCounts)
          .sort(([, countA], [, countB]) => countB - countA)
          .slice(0, 15); // Display top 15

        setChartOption({
          title: {
            text: 'ETF基金公司产品数量统计 (Top 15)',
            left: 'center'
          },
          tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' }
          },
          grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
          },
          xAxis: {
            type: 'value',
            boundaryGap: [0, 0.01]
          },
          yAxis: {
            type: 'category',
            data: sortedManagers.map(([name]) => name).reverse(),
            axisLabel: { interval: 0, rotate: 0 }
          },
          series: [
            {
              name: '产品数量',
              type: 'bar',
              data: sortedManagers.map(([, count]) => count).reverse()
            }
          ]
        });
      } catch (error) {
        console.error("Failed to fetch ETF data for dashboard:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return <div className="mx-auto max-w-5xl p-6">正在加载图表数据...</div>;
  }

  return (
    <div className="mx-auto max-w-5xl p-6">
      <h1 className="text-2xl font-semibold">主界面</h1>
      <p className="text-sm text-gray-500 mt-1">展示当前市场ETF的概览统计信息</p>
      <div className="mt-5 rounded-2xl border border-gray-200 bg-white p-4">
        <ReactECharts option={chartOption} style={{ height: '500px', width: '100%' }} />
      </div>
    </div>
  );
}