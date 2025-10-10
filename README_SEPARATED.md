前后端分离版（React 前端 + FastAPI 后端）

目录结构
- backend/：Python FastAPI 服务
- frontend/：React + Vite + Tailwind 前端

一、启动后端（Python）
1) 建议 Python 3.9+
2) 安装依赖：
   cd backend
   pip install -r requirements.txt
3) 运行服务（默认端口 8000）：
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
4) 健康检查：浏览器打开 http://127.0.0.1:8000/api/health 应返回 {"ok": true}

接口说明
- POST /api/risk-parity/solve
  请求体：
  {
    "assetClassId": "...",
    "riskMetric": "vol|var|es",
    "maxLeverage": 0.0,
    "etfs": [{"code":"SPY","name":"...","riskContribution":60}, ...]
  }
  返回：{ "weights": [80.0, 20.0, ...] }
  说明：当前为“简化版”风险平价（按风险贡献按比例分配，支持线性杠杆）。如需真实风险平价（基于协方差矩阵），我可以接入 data/ 下数据实现。

二、启动前端（React）
1) 安装依赖：
   cd frontend
   npm install  # 或 pnpm install / yarn
2) 开发模式启动（默认端口 5173）：
   npm run dev
3) 打开浏览器访问：http://127.0.0.1:5173

联调说明
- 前端开发服务器已在 vite.config.ts 中配置代理：`/api` -> `http://127.0.0.1:8000`。
- 确保后端已启动后，再在前端页面点击“计算反推资金权重”。

三、与旧版的关系
- 之前的 Streamlit 单文件版（app.py）仍保留，方便对比或离线演示。
- 正式使用建议采用本前后端分离结构。

四、下一步可选增强
- 用真实风险平价算法替换简化求解器：读取行情，估协方差矩阵，数值求解使边际风险贡献相等。
- 接入持久化（保存/加载资产大类配置）。
- 生产部署（Nginx 反代+前端静态构建+Uvicorn/Gunicorn 多进程）。

五、只用 Python 一键启动（生产/演示）
- 思路：前端先构建为静态资源（frontend/dist），FastAPI 挂载为静态站点并提供 API，同源部署；再由 Python 自动打开浏览器。
- 一次性构建（需要 Node 环境）：
  cd frontend && npm install && npm run build
- 之后每次启动仅需：
  python backend/run.py
- 打开地址：http://127.0.0.1:8000 （页面与 API 同域，不再需要前端 dev 服务）
