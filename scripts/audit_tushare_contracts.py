"""Read-only comparison of downloader declarations with an interface catalog."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "T01_get_data.py").read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    apis = set(constants.get("API_ROW_LIMITS", {})) | {"trade_cal", "stock_basic", "fund_company", "fund_basic", "etf_basic", "index_basic", "etf_index"}
    catalog = json.loads(args.catalog.read_text())
    for item in catalog["interfaces"]:
        if item["api"] in apis:
            print(json.dumps({"api": item["api"], "doc_id": item["doc_id"], "limits": item.get("limits"), "access": item.get("availability_with_10000_points"), "legacy_cap": constants.get("API_ROW_LIMITS", {}).get(item["api"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
