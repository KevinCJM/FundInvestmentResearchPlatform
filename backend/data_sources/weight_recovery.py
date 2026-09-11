"""Import only weight parts proven by complete, immutable query receipts."""
import hashlib
import json
import re
from datetime import datetime

import pandas as pd

from .constituent_recovery import require, _same_frame
from .task_workspace import clone_file


def import_weights(journal, source, target, step, catalog, progress):
    import T01_get_data as script
    require(source.is_dir() and not source.is_symlink(), '权重检查点目录无效。')
    codes = set(catalog.loc[catalog.quote_source_api.eq('index_daily') &
                catalog.get('status', pd.Series('active', index=catalog.index)).fillna('active').eq('active'), 'ts_code'].dropna().astype(str))
    end = pd.Timestamp(step.params['end_date'])
    lower = max(pd.Timestamp(step.params['start_date']), end - pd.Timedelta(days=119))
    query_dir = source / 'queries'
    require(query_dir.is_dir() and not query_dir.is_symlink(), '权重缺少原始查询回执，不能直接复用完成标记。')
    files = list(query_dir.iterdir())
    require(all(p.is_file() and not p.is_symlink() and re.fullmatch(r'[a-f0-9]{64}\.(json|parquet)', p.name) for p in files), '权重查询含未知文件。')
    receipts = {p.stem for p in files if p.suffix == '.json'}
    require(receipts == {p.stem for p in files if p.suffix == '.parquet'}, '权重原始文件与回执不成对。')
    grouped, evidence = {}, []
    target.mkdir()
    (target / 'queries').mkdir()
    def copy(original, dest):
        artifact = journal.artifact(original)
        clone_file(original, dest)
        imported = journal.artifact(dest)
        require(imported['checksum'] == artifact['checksum'], '权重复制后校验失败。')
        evidence.append({'source': artifact, 'imported': imported})
    for key in sorted(receipts):
        receipt, path = query_dir / (key + '.json'), query_dir / (key + '.parquet')
        meta = json.loads(receipt.read_text())
        identity = meta.get('request', {})
        params = identity.get('params', {})
        require(identity.get('version') == 1 and identity.get('api') == 'index_weight'
                and set(params) == {'index_code', 'start_date', 'end_date'}
                and hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == key,
                '权重请求身份或文件名不一致。')
        require(params['index_code'] in codes and lower <= pd.Timestamp(params['start_date']) <= pd.Timestamp(params['end_date']) <= end,
                '权重回执超出冻结代码/120天范围。')
        require(isinstance(meta.get('collected_at'), str) and datetime.fromisoformat(meta['collected_at']).tzinfo is not None,
                '权重回执缺少采集时间。')
        require(journal.artifact(path)['checksum'] == meta.get('sha256'), '权重查询校验和不一致。')
        frame = pd.read_parquet(path)
        require(len(frame) == meta.get('rows'), '权重查询行数不一致。')
        script.validate_constituent_frame(frame, 'index_weight', params, set())
        grouped.setdefault(params['index_code'], []).append((params, path))
        copy(receipt, target / 'queries' / receipt.name)
        copy(path, target / 'queries' / path.name)
        if len(evidence) % 500 == 0:
            progress(f'已核验 {len(evidence) // 2} 个完整权重查询；未重新请求供应商。')
    parts = [p for p in source.iterdir() if p.name != 'queries']
    require(all(p.is_file() and not p.is_symlink() and p.suffix in {'.parquet', '.empty'} and p.stem in codes for p in parts), '权重成品检查点含未知代码或文件。')
    require(len(parts) == len({p.stem for p in parts}), '权重检查点同时存在数据与空标记。')
    for part in parts:
        queries = grouped.get(part.stem, [])
        require(bool(queries), '权重完成标记缺少原始证据。')
        frames = [pd.read_parquet(path) for _, path in queries]
        nonempty = [frame for frame in frames if not frame.empty]
        if part.suffix == '.empty':
            require(not nonempty and part.read_text() == 'no data\n', '空权重标记与查询不一致。')
            cursor = lower
        else:
            require(bool(nonempty), '非空权重缺少原始数据。')
            rebuilt = script._normalise_member_frame(pd.concat(nonempty, ignore_index=True), 'index_weight', part.stem)
            cursor = rebuilt.trade_date.max()
            rebuilt = rebuilt.loc[rebuilt.trade_date.eq(cursor)].drop_duplicates(['source_api', 'index_code', 'con_code'], keep='last')
            _same_frame(pd.read_parquet(part), rebuilt)
        # Every date after the selected latest observation must be covered;
        # otherwise a newer, still-missing month could change the latest result.
        for params, _ in sorted(queries, key=lambda q: q[0]['start_date']):
            start, stop = pd.Timestamp(params['start_date']), pd.Timestamp(params['end_date'])
            if start <= cursor <= stop:
                cursor = stop + pd.Timedelta(days=1)
        require(cursor > end, '权重查询区间不完整，不能证明为最新权重。')
        copy(part, target / part.name)
    return {'copied': len(parts), 'queries': len(receipts), 'files': evidence}
