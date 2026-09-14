"""M1 browser fixture: all mutable state is in the existing temporary test root."""
import json
import pandas as pd
from backend.tests.strategic_allocation_app import app, root

path = root / 'asset_alloc_info.parquet'
frame = pd.read_parquet(path)
frame['universe_snapshot_id'] = 'm1-browser-domain'
frame.to_parquet(path, index=False)
(root / 'product_pools.json').write_text(json.dumps({'pools': [], 'versions': [], 'universe_snapshots': [
    {'id': 'm1-browser-domain', 'name': '离线产品域', 'research_date': '2020-01-01', 'immutable': True,
     'members': [{'kind': 'etf', 'product_id': code, 'eligible': True} for code in ['510300.SH', '511010.SH']]}
]}))
