"""Explicit code normalization, shared by sources rather than guessed names."""
from __future__ import annotations
import re


def normalize_external_key(value: object, transform: str) -> str:
    text = str(value).strip()
    if transform == "none":
        return text
    if transform == "cn_etf_code":
        if re.fullmatch(r"5\d{5}(?:\.SH)?", text):
            return text[:6] + ".SH"
        if re.fullmatch(r"1\d{5}(?:\.SZ)?", text):
            return text[:6] + ".SZ"
        raise ValueError("ETF 代码不能确定交易所；请提供有效 SH/SZ 代码或配置人工对照")
    if transform == "cn_fund_code" and re.fullmatch(r"\d{6}(?:\.OF)?", text):
        return text[:6] + ".OF"
    raise ValueError("外部代码不符合所选代码转换规则")
