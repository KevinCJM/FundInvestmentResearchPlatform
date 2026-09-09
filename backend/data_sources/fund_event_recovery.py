"""Verified import of current v4 fund-event receipts, not a second downloader."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime

import pandas as pd
import pyarrow.parquet as pq

from .fund_events import fund_inceptions
from .models import CenterError
from .task_workspace import clone_file


def _require(value, message):
    if not value:
        raise CenterError('ETL_MIGRATION_BLOCKED', message, 409)


def _day(value):
    return value.date() if isinstance(value, datetime) else value


def import_dividend_receipts(journal, parts, dest, step, progress=lambda _: None):
    """Reuse exact announcement contracts; never infer completeness from files.

    This bridge accepts market-day COMPLETE/EMPTY receipts only. A SPLIT or
    individual-fund receipt requires a separate review, not implicit refetch.
    """
    from T01_get_data import FUND_DIVIDEND_FIELDS

    universe_path = dest.parent / 'fund_info_df.parquet'
    _require(universe_path.is_file() and not universe_path.is_symlink(), '缺少冻结基金目录。')
    universe = pd.read_parquet(universe_path, columns=['ts_code', 'found_date'])
    expected = dict(version=4, strategy='announcement', range_field='ann_date', api='fund_div',
                    fields=FUND_DIVIDEND_FIELDS, smoke=False,
                    dates=pd.date_range(step.params['start_date'], step.params['end_date']).strftime('%Y%m%d').tolist(),
                    inceptions=fund_inceptions(universe))
    name = 'events_v4_' + hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()[:20]
    directories = sorted(parts.glob('events_v4_*'))
    _require(len(directories) == 1 and directories[0].name == name, '分红检查点合同与冻结范围不一致。')
    source = directories[0]
    contract = source / 'contract.json'
    _require(not source.is_symlink() and contract.is_file() and not contract.is_symlink(), '分红合同路径无效。')
    contract_checksum = journal.artifact(contract)['checksum']
    _require(json.loads(contract.read_text()) == expected, '分红字段、范围或基金目录合同已改变。')
    target = dest / name
    target.mkdir()
    evidence, complete, empty = [], 0, 0

    def copy_evidence(path, checksum):
        before = journal.artifact(path)
        _require(before['checksum'] == checksum, '分红证据在核验期间改变。')
        clone_file(path, target / path.name)
        after = journal.artifact(target / path.name)
        _require(after['checksum'] == checksum, '分红证据复制校验失败。')
        evidence.append({'source': before, 'imported': after})

    copy_evidence(contract, contract_checksum)
    required = set(FUND_DIVIDEND_FIELDS) | {'available_at', 'source_api', 'observation_date', 'availability_status', 'ingested_at'}
    dates = set(expected['dates'])
    for receipt in sorted(source.glob('*.json')):
        if receipt.name in {'contract.json', 'failure.json'}:
            continue
        _require(not receipt.is_symlink(), '分红回执不得为符号链接。')
        checksum = journal.artifact(receipt)['checksum']
        record = json.loads(receipt.read_text())
        match = re.fullmatch(r'(\d{8})_market', receipt.stem)
        _require(match is not None and match[1] in dates and record.get('date') == match[1]
                 and record.get('code') is None, '分红回执请求身份不匹配或为未支持的拆分回执。')
        status = record.get('status')
        _require(status in {'COMPLETE', 'EMPTY'}, '分红只允许导入已完成日期，不能把拆分当作完成。')
        if status == 'EMPTY':
            _require(record.get('confirmations') == 2, '分红空响应未独立复核。')
            empty += 1
        else:
            part = receipt.with_suffix('.parquet')
            _require(part.is_file() and not part.is_symlink(), '分红分片缺失或为符号链接。')
            _require(journal.artifact(part)['checksum'] == record.get('sha256'), '分红分片校验和不符。')
            parquet = pq.ParquetFile(part)
            _require(type(record.get('rows')) is int and record['rows'] >= 0
                     and parquet.metadata.num_rows == record['rows']
                     and required <= set(parquet.schema_arrow.names), '分红分片行数或字段不完整。')
            day = datetime.strptime(match[1], '%Y%m%d').date()
            for batch in parquet.iter_batches(batch_size=16384):
                for row in batch.to_pylist():
                    ex, observed = row['ex_date'], row['observation_date']
                    _require(_day(row['ann_date']) == _day(row['available_at']) == day
                             and row['ts_code'] in expected['inceptions']
                             and row['source_api'] == 'fund_div' and row['availability_status'] == 'announced_date'
                             and isinstance(row['ingested_at'], str) and bool(row['ingested_at'])
                             and ((ex is None and observed is None) or
                                  (isinstance(ex, (date, datetime)) and _day(ex) == _day(observed))),
                             '分红分片基金、日期或来源时点不一致。')
            copy_evidence(part, record['sha256'])
            complete += 1
        copy_evidence(receipt, checksum)
        if (complete + empty) % 250 == 0:
            progress(f'已核验 {complete + empty} 个分红日期回执；不发起下载请求。')
    return {'files': evidence, 'complete': complete, 'empty': empty}


def import_history_receipts(journal, parts, dest, step, progress=lambda _: None):
    """Pin the exact data contract; copy only checksummed, decoded evidence.

    Empty receipts require two successful responses. Split receipts prove no
    completeness by themselves: the collector still visits every child range.
    Temporary/merged outputs, failure logs and old protocol versions stay put.
    """
    from T01_get_data import FUND_PORTFOLIO_FIELDS

    directories = sorted(parts.glob('events_v4_*'))
    if not directories:
        return {'files': [], 'complete': 0, 'empty': 0, 'split': 0}
    universe_file = dest.parent / 'fund_info_df.parquet'
    _require(universe_file.is_file() and not universe_file.is_symlink(), '缺少冻结基金目录，不能复用区间分片。')
    universe = pd.read_parquet(universe_file, columns=['ts_code', 'found_date'])
    expected = dict(version=4, strategy='fund_announcement_history', range_field='ann_date',
                    api='fund_portfolio', fields=FUND_PORTFOLIO_FIELDS, smoke=False,
                    dates=pd.date_range(step.params['start_date'], step.params['end_date']).strftime('%Y%m%d').tolist(),
                    inceptions=fund_inceptions(universe))
    digest = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()[:20]
    name = 'events_v4_' + digest
    _require(len(directories) == 1 and directories[0].name == name, '区间检查点与冻结范围、基金目录或字段合同不一致。')
    source = directories[0]
    contract_path = source / 'contract.json'
    _require(not source.is_symlink() and contract_path.is_file() and not contract_path.is_symlink(), '区间合同路径无效。')
    contract_checksum = journal.artifact(contract_path)['checksum']
    _require(json.loads(contract_path.read_text()) == expected, '区间检查点合同已变化。')
    target = dest / name
    target.mkdir()
    evidence = []
    counts = {'complete': 0, 'empty': 0, 'split': 0, 'requery': 0}

    def copy_evidence(path, expected_checksum, *, audit_only=False):
        before = journal.artifact(path)
        _require(before['checksum'] == expected_checksum, '已验证分片在复制前发生变化。')
        destination = target / 'requery_evidence' / path.name if audit_only else target / path.name
        destination.parent.mkdir(exist_ok=True)
        clone_file(path, destination)
        after = journal.artifact(destination)
        _require(before['checksum'] == after['checksum'], '区间分片复制后校验和不一致。')
        evidence.append({'source': before, 'imported': after})

    copy_evidence(contract_path, contract_checksum)
    for receipt in sorted(source.glob('*.json')):
        if receipt.name in {'contract.json', 'failure.json'}:
            continue
        _require(not receipt.is_symlink(), '区间回执不得为符号链接。')
        receipt_checksum = journal.artifact(receipt)['checksum']
        record = json.loads(receipt.read_text())
        match = re.fullmatch(r'(\d{8})-(\d{8})_([A-Za-z0-9][A-Za-z0-9._-]{0,63})', receipt.stem)
        # Imported daily receipts are already represented by verified_day_imports.
        if not match and re.fullmatch(r'\d{8}_market', receipt.stem):
            continue
        _require(match is not None, '区间回执名称无效。')
        start, end, code = match.groups()
        _require(step.params['start_date'] <= start <= end <= step.params['end_date']
                 and code in expected['inceptions'] and record.get('code') == code
                 and record.get('date') == start + '-' + end, '区间回执身份或范围不一致。')
        left, right = datetime.strptime(start, '%Y%m%d').date(), datetime.strptime(end, '%Y%m%d').date()
        status = record.get('status')
        _require(status in {'COMPLETE', 'EMPTY', 'SPLIT'}, '未知区间回执状态。')
        if status == 'COMPLETE':
            part = receipt.with_suffix('.parquet')
            _require(part.is_file() and not part.is_symlink(), '区间数据文件缺失或为符号链接。')
            _require(journal.artifact(part)['checksum'] == record.get('sha256'), '区间数据校验和不一致。')
            parquet = pq.ParquetFile(part)
            required = set(FUND_PORTFOLIO_FIELDS) | {'available_at', 'source_api', 'observation_date', 'availability_status', 'ingested_at'}
            _require(parquet.metadata.num_rows > 0 and parquet.metadata.num_rows == record.get('rows')
                     and required <= set(parquet.schema_arrow.names), '区间数据行数或字段不完整。')
            for batch in parquet.iter_batches(batch_size=16384):
                for row in batch.to_pylist():
                    ann, available, report = map(_day, (row['ann_date'], row['available_at'], row['end_date']))
                    _require(isinstance(ann, date) and left <= ann <= right and ann == available
                             and isinstance(report, date) and _day(row['observation_date']) == report
                             and row['ts_code'] == code and isinstance(row['symbol'], str) and row['symbol'].strip()
                             and row['source_api'] == 'fund_portfolio' and row['availability_status'] == 'announced_date'
                             and row['ingested_at'] is not None, '区间数据业务键、来源或时点口径不一致。')
            copy_evidence(part, record['sha256'])
        elif status == 'EMPTY':
            _require(record.get('confirmations') == 2, '空区间未独立复核，不能复用。')
        elif start == end:
            # Preserve the failed receipt as immutable evidence, outside the
            # working cache. Replacing an imported receipt after a successful
            # page fetch would invalidate future recovery artifact checks.
            copy_evidence(receipt, receipt_checksum, audit_only=True)
            counts['requery'] += 1
            continue
        copy_evidence(receipt, receipt_checksum)
        counts[status.lower()] += 1
        if sum(counts.values()) % 250 == 0:
            progress(f'已核验 {sum(counts.values())} 个区间回执；不发起下载请求。')
    return {'files': evidence, **counts}
