# -*- encoding: utf-8 -*-
"""集中管理本地开发配置."""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv(env_path: Path) -> None:
    """Minimal .env loader so secrets stay outside版本控制."""

    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


# 尝试加载项目根目录的 .env（若存在）
_load_dotenv(Path(__file__).resolve().parent / ".env")

# 公开配置项 ---------------------------------------------------------------
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")


def require_tushare_token() -> str:
    """获取 Tushare 令牌，未设置时抛出解释性错误."""

    if not TUSHARE_TOKEN:
        raise RuntimeError(
            "未检测到 TUSHARE_TOKEN 环境变量，请在 shell 中导出或在 .env 文件中配置."
        )
    return TUSHARE_TOKEN
