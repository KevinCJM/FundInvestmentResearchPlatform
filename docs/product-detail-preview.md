# 产品详情页预览说明

为了快速验证「统计分析」模块的交互与视觉效果，可以按照以下步骤在本地生成预览页面：

1. 在 `frontend` 目录安装依赖并启动 Vite 开发服务器：
   ```bash
   cd frontend
   npm install
   npm run dev -- --host 0.0.0.0 --port 4173
   ```
2. 在浏览器中访问 `http://localhost:4173/product/demo-etf`。当后端接口缺失时，页面会自动注入带有 180 个交易日的模拟行情数据。
3. 页面顶部展示价格 K 线与可选叠加指标，下方的「统计分析」模块通过指标卡、日收益率折线与分布柱状图展示关键统计信息，可用于演示场景。

如需重新生成预览截图，可使用 Playwright/Chromium 打开上述地址并截取整页画面，例如：

```bash
python - <<'PY'
import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={'width': 1440, 'height': 900})
        await page.goto('http://127.0.0.1:4173/product/demo-etf', wait_until='networkidle')
        await page.wait_for_timeout(3000)
        await page.screenshot(path='docs/assets/product-detail-preview.png', full_page=True)
        await browser.close()

asyncio.run(main())
PY
```

运行后可在 `docs/assets/product-detail-preview.png` 找到示例预览图，便于在文档或演示材料中引用。
