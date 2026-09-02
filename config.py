# -*- encoding: utf-8 -*-
"""集中管理本地开发配置和本机 Tushare 凭据。"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TUSHARE_CREDENTIAL_PATH = PROJECT_ROOT / "data" / ".tushare_token"


def read_tushare_token(path: Path | None = None) -> str:
    """读取前端写入的本机凭据文件，不读取进程环境变量。"""

    credential_path = (path or TUSHARE_CREDENTIAL_PATH).expanduser()
    if credential_path.is_symlink():
        raise RuntimeError("Tushare Token 凭据文件不能是符号链接。")
    if not credential_path.is_file():
        return ""
    try:
        return credential_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError("无法读取本机 Tushare Token 凭据文件。") from exc


def require_tushare_token() -> str:
    """获取前端配置的 Tushare 令牌，未设置时给出用户可执行的提示。"""

    token = read_tushare_token()
    if not token:
        raise RuntimeError(
            "尚未配置 Tushare Token，请在主界面的“数据管理”中先保存 Token。"
        )
    return token
