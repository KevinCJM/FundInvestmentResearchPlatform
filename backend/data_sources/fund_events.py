"""Bounded fund-event I/O and resumable request receipts.

Dates below are announcement dates. Current-contract inception/liquidation
dates do not bound predecessor history. This module does no financial math.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd

from backend.services.refresh_runtime import atomic_write_json, read_json_object
from .models import CenterError


def fund_inceptions(universe: pd.DataFrame) -> dict[str, str | None]:
    """Consistent current-contract dates are split hints, never history cutoffs."""
    dates: dict[str, set[str | None]] = {}
    for row in universe.to_dict('records'):
        if pd.isna(row.get('ts_code')):
            continue
        code = str(row['ts_code']).strip()
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', code):
            raise CenterError('FUND_EVENT_UNIVERSE', '基金目录包含非法代码。')
        value = row.get('found_date')
        # Missing/conflicting dates disable the split hint, not the fund.
        parsed = pd.to_datetime(str(value), errors='coerce') if pd.notna(value) else pd.NaT
        date = None if pd.isna(parsed) else parsed.strftime('%Y%m%d')
        dates.setdefault(code, set()).add(date)
    return {code: next(iter(values)) if len(values) == 1 else None
            for code, values in sorted(dates.items())}


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _page_evidence(frame):
    """Hash raw I/O records without rounding floats or treating NaN as a value."""
    if frame.empty:
        return hashlib.sha256(b'empty-page').digest(), {}
    records = frame.astype(object).where(frame.notna(), None).to_dict('records')
    digest, identities = hashlib.sha256(), {}
    for record in records:
        encoded = json.dumps(record, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()
        digest.update(encoded + b'\n')
        key = tuple(record[name] for name in ('ts_code', 'ann_date', 'end_date', 'symbol'))
        signature = hashlib.sha256(encoded).digest()
        identities.setdefault(key, set()).add(signature)
    return digest.digest(), identities


class FundEventDownload:
    def __init__(self, *, directory, dates, universe, api_name, fields, smoke,
                 max_requests=100_000, idle_timeout=180, max_runtime=86_400, strategy='announcement'):
        self.dates, self.api_name, self.fields, self.smoke = dates, api_name, fields, smoke
        self.contract_strategy = strategy
        self.market_paged = False
        self.inceptions = fund_inceptions(universe)
        if not self.inceptions:
            raise CenterError('FUND_EVENT_UNIVERSE', '基金目录为空，不能声明下载完整。')
        contract = dict(version=4, strategy=strategy, range_field='ann_date', api=api_name, fields=fields, dates=dates,
                        inceptions=self.inceptions, smoke=smoke)
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        self.directory = directory / ('events_v4_' + digest[:20])
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.directory / 'contract.json', contract)
        imported = directory / 'verified_day_imports.json'
        manifest = read_json_object(imported) if imported.exists() else {}
        if imported.is_symlink() or (manifest and manifest.get('version') != 1):
            raise CenterError('FUND_EVENT_CHECKPOINT', '旧分片导入回执无效。')
        self.imports = manifest.get('days', {})
        self.stop = threading.Event()
        self.failure = None
        self.failed_date = None
        self.mutex = threading.Lock()
        self.started = self.activity = time.monotonic()
        self.calls = 0
        self.reused = 0
        self.acknowledgements = []
        self.conflicts = {}
        self.max_requests = 1 if smoke else max_requests
        self.idle_timeout, self.max_runtime = idle_timeout, max_runtime

    def check(self):
        if self.stop.is_set():
            if self.failure is not None:
                raise self.failure
            raise CenterError('FUND_EVENT_STOPPED', '已停止派发，已完成请求检查点保留。')
        now = time.monotonic()
        if now - self.started >= self.max_runtime:
            raise CenterError('FUND_EVENT_RUNTIME', '基金披露任务达到最大运行时限，检查点保留。')
        if now - self.activity >= self.idle_timeout:
            raise CenterError('FUND_EVENT_IDLE', '基金披露任务长时间没有响应或检查点，已停止派发。')

    def before_request(self):
        self.check()
        with self.mutex:
            if self.calls >= self.max_requests:
                raise CenterError('FUND_EVENT_BUDGET', '达到基金披露请求预算，检查点保留；请核查范围后续跑。')
            self.calls += 1

    def pause(self, seconds):
        if self.stop.wait(seconds):
            self.check()
        self.check()

    def _paths(self, date, code):
        key = date + ('_' + code if code else '_market')
        return self.directory / (key + '.parquet'), self.directory / (key + '.json')

    def _cached(self, date, code):
        part, receipt = self._paths(date, code)
        if not receipt.exists():
            return None
        record = read_json_object(receipt)
        if record.get('date') != date or record.get('code') != code:
            raise CenterError('FUND_EVENT_CHECKPOINT', '检查点请求身份不匹配。')
        if record.get('status') == 'COMPLETE':
            if not part.is_file() or _checksum(part) != record.get('sha256'):
                raise CenterError('FUND_EVENT_CHECKPOINT', '基金披露检查点校验失败，禁止当作成功跳过。')
            if record.get('quality_status') == 'CONFLICTED' and not record.get('conflict_evidence'):
                raise CenterError('FUND_EVENT_CHECKPOINT', '冲突检查点缺少原始证据。')
            if record.get('conflict_evidence'):
                from .fund_event_conflicts import checked_part, validate_placeholders
                raw = checked_part(self.directory, record['conflict_evidence'])
                validate_placeholders(part, raw)
                self.conflicts[(date, code)] = record['conflict_evidence']
        elif record.get('status') == 'EMPTY':
            if record.get('confirmations', 0) < (1 if self.smoke else 2):
                raise CenterError('FUND_EVENT_CHECKPOINT', '空响应未复核，禁止当作成功跳过。')
        elif record.get('status') != 'SPLIT':
            raise CenterError('FUND_EVENT_CHECKPOINT', '未知检查点状态。')
        return record

    def _record(self, date, code, status, **values):
        _, receipt = self._paths(date, code)
        if status == 'COMPLETE' and (date, code) in self.conflicts:
            values.update(quality_status='CONFLICTED', conflict_evidence=self.conflicts[(date, code)])
        atomic_write_json(receipt, dict(date=date, code=code, status=status, **values))
        with self.mutex:
            self.activity = time.monotonic()

    def _prepare(self, frame, date, code, prepare):
        if self.api_name != 'fund_portfolio' or frame.empty:
            return prepare(frame)
        from .fund_event_conflicts import segregate, write_part, STATUS
        clean, evidence = segregate(frame)
        mask = clean.pop('_source_conflict')
        # Apply lineage before sorting/deduplication; never infer conflict rows
        # from their position in a reordered frame.
        clean['availability_status'] = mask.map({True: STATUS, False: 'announced_date'})
        result = prepare(clean)
        if not evidence.empty:
            with self.mutex:
                self.conflicts[(date, code)] = write_part(self.directory, date, code, evidence)
            print(f'[WARN] {date} {code or "全市场"} 持仓冲突已隔离；保留全部原始值，标准值留空。')
        return result

    def _confirm_unpaged_conflicts(self, frame, fetch):
        if self.api_name != 'fund_portfolio' or frame.empty:
            return
        signature, identities = _page_evidence(frame)
        if any(len(values) > 1 for values in identities.values()):
            other = fetch()
            if _page_evidence(other)[0] != signature:
                raise CenterError('FUND_EVENT_PAGINATION', '持仓冲突复核时供应商数据发生变化，不能声明采集完整。')

    def _validate(self, frame, date, code):
        if frame.empty:
            return
        required = {'ts_code', 'ann_date'}
        if self.api_name == 'fund_portfolio':
            required.update({'end_date', 'symbol'})
        if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
            raise CenterError('FUND_EVENT_RESPONSE', '基金披露响应缺少业务键，拒绝静默丢行。')
        announcements = pd.to_datetime(frame['ann_date'].astype(str), format='%Y%m%d', errors='coerce')
        if announcements.isna().any() or (date and not frame['ann_date'].astype(str).eq(date).all()):
            raise CenterError('FUND_EVENT_RESPONSE', '供应商返回其他公告日，拒绝错误分片或忽略参数的响应。')
        if code and not frame['ts_code'].astype(str).eq(code).all():
            raise CenterError('FUND_EVENT_RESPONSE', '供应商返回其他基金，拒绝错误分片。')
        if self.api_name == 'fund_portfolio':
            ends = pd.to_datetime(frame['end_date'].astype(str), format='%Y%m%d', errors='coerce')
            if ends.isna().any() or frame['symbol'].astype(str).str.strip().eq('').any():
                raise CenterError('FUND_EVENT_RESPONSE', '持仓报告期或证券代码无效。')

    def _request(self, date, code, fetch, prepare, save):
        cached = self._cached(date, code)
        if cached is not None:
            with self.mutex:
                self.activity = time.monotonic()
                self.reused += 1
            return cached['status']
        if code is None and date in self.imports:
            return self._import_date(date, prepare, save)
        confirmations = 1 if self.smoke else 2
        for attempt in range(confirmations):
            self.check()
            frame = fetch(date, code)
            with self.mutex:
                self.activity = time.monotonic()
            self._validate(frame, date, code)
            if not frame.empty:
                break
            if attempt + 1 < confirmations:
                print(f'[INFO] {self.api_name} {date} {code or "全市场"} 空响应，独立复核。')
                self.pause(0.25)
        if frame.empty:
            self._record(date, code, 'EMPTY', confirmations=confirmations)
            return 'EMPTY'
        if not self.market_paged:
            self._confirm_unpaged_conflicts(frame, lambda: fetch(date, code))
        # An irrelevant nonempty market response is not an upstream empty result.
        frame = self._prepare(frame[frame['ts_code'].astype(str).isin(self.inceptions)].copy(), date, code, prepare)
        part, _ = self._paths(date, code)
        save(frame, part, quiet=True)
        self._record(date, code, 'COMPLETE', rows=len(frame), sha256=_checksum(part))
        return 'COMPLETE'

    def _import_date(self, date, prepare, save):
        """Only explicit migration receipts authorize old nonempty shard reuse."""
        record = self.imports[date]
        source = self.directory.parent / (date + '.parquet')
        if source.is_symlink() or not source.is_file() or _checksum(source) != record.get('sha256'):
            raise CenterError('FUND_EVENT_CHECKPOINT', '已导入日期分片校验失败。')
        frame = pd.read_parquet(source)
        required = {'ann_date', 'available_at', 'source_api', 'ts_code', 'end_date'}
        if frame.empty or len(frame) != record.get('rows') or not required.issubset(frame.columns):
            raise CenterError('FUND_EVENT_CHECKPOINT', '已导入日期分片缺少行数或字段证据。')
        if (not frame.source_api.eq(self.api_name).all()
                or not pd.to_datetime(frame.available_at).eq(pd.Timestamp(date)).all()
                or not frame.ts_code.isin(self.inceptions).all()):
            raise CenterError('FUND_EVENT_CHECKPOINT', '已导入日期分片来源、基金范围或可得日期不符。')
        raw = frame.drop(columns=['available_at'])
        for name in ('ann_date', 'end_date'):
            raw[name] = pd.to_datetime(raw[name]).dt.strftime('%Y%m%d')
        self._validate(raw, date, None)
        part, _ = self._paths(date, None)
        save(prepare(raw), part, quiet=True)
        self._record(date, None, 'COMPLETE', rows=len(frame), sha256=_checksum(part),
                     imported_sha256=record['sha256'])
        return 'COMPLETE'

    def _date(self, date, fetch, prepare, save, cap_error):
        try:
            return self._download_date(date, fetch, prepare, save, cap_error)
        except Exception as exc:
            with self.mutex:
                if self.failure is None:
                    self.failure = exc
                    self.failed_date = date
                self.stop.set()
            raise

    def _download_date(self, date, fetch, prepare, save, cap_error):
        self.check()
        eligible = list(self.inceptions)
        try:
            status = self._request(date, None, fetch, prepare, save)
        except cap_error:
            if self.smoke or self.market_paged:
                raise
            self._record(date, None, 'SPLIT')
            status = 'SPLIT'
        if status == 'SPLIT':
            return self._split_date(date, eligible, fetch, prepare, save, cap_error)
        return self._paths(date, None)[0] if status == 'COMPLETE' else None

    def _split_date(self, date, eligible, fetch, prepare, save, cap_error):
        print(f'[INFO] {self.api_name} {date} 触顶补抓：候选 {len(eligible)} 只；'
              '保留基金转型前历史，逐只保存检查点。')
        paths = []
        last_log = 0.0
        for index, code in enumerate(eligible, 1):
            self.check()
            try:
                status = self._request(date, code, fetch, prepare, save)
            except cap_error as exc:
                raise CenterError('FUND_EVENT_UNSPLITTABLE',
                                  f'{self.api_name} {date} {code} 仍触顶，未核定更细分片；停止而非提交残缺结果。') from exc
            if status == 'COMPLETE':
                paths.append(self._paths(date, code)[0])
            if time.monotonic() - last_log >= 5 or index == len(eligible):
                print(f'[INFO] {self.api_name} {date} 基金补抓 {index}/{len(eligible)}，'
                      f'当前 {code}，本次请求 {self.calls}/{self.max_requests}（含重试/空复核）。')
                last_log = time.monotonic()
        # Only one date is assembled at a time; large full-history output is streamed later.
        frame = prepare(pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)
                        if paths else pd.DataFrame(columns=self.fields))
        part, _ = self._paths(date, None)
        save(frame, part, quiet=True)
        self._record(date, None, 'COMPLETE', rows=len(frame), sha256=_checksum(part))
        return part

    def run(self, *, fetch, prepare, save, cap_error, max_workers):
        print(f'[STAGE] {self.api_name} 公告日下载；保留基金转型前历史；'
              f'请求预算 {self.max_requests}，空闲上限 {self.idle_timeout}s。')
        operation = lambda date: self._date(date, fetch, prepare, save, cap_error)
        results = self._run_units(self.dates, operation, max_workers, '日期')
        return [results[date] for date in self.dates if results[date] is not None]

    def _run_units(self, units, operation, max_workers, label):
        dates, results, futures = iter(units), {}, {}
        executor = ThreadPoolExecutor(max_workers=max(1, max_workers))
        failure = None
        try:
            self._submit(executor, futures, dates, operation, max_workers)
            while futures:
                self.check()
                done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    date = futures.pop(future)
                    results[date] = future.result()
                    print(f'[INFO] {self.api_name} {label}进度 {len(results)}/{len(units)}。')
                self._submit(executor, futures, dates, operation, len(done))
        except Exception as exc:
            failure = exc
            self.stop.set()
            for future in futures:
                future.cancel()
            atomic_write_json(self.directory / 'failure.json', dict(
                code=getattr(exc, 'code', type(exc).__name__), calls=self.calls,
                message=exc.message if isinstance(exc, CenterError) else '下载失败，检查点保留。',
                failed_date=self.failed_date, completed_dates=len(results), pending_dates=list(futures.values())))
            print(f'[ERROR] {self.api_name} 已停止派发：{getattr(exc, "code", type(exc).__name__)}；'
                  '等待有限数量的在途请求结束，已下载检查点保留。')
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        if failure is not None:
            print(f'[ERROR] {getattr(failure, "code", type(failure).__name__)}：'
                  f'{failure.message if isinstance(failure, CenterError) else "下载失败，检查点保留。"}')
            raise failure
        return results

    def _submit(self, executor, futures, dates, operation, count):
        for _ in range(count):
            date = next(dates, None)
            if date is None:
                return
            self.check()
            futures[executor.submit(operation, date)] = date

    def _history_leaf(self, code, start, end, fetch, prepare, save, cap_error):
        key = start + '-' + end
        cached = self._cached(key, code)
        if cached is not None:
            with self.mutex:
                self.activity = time.monotonic()
            if cached['status'] == 'COMPLETE':
                return [self._paths(key, code)[0]]
            if cached['status'] == 'EMPTY':
                return []
        if cached is None or start == end:
            try:
                return self._fetch_history_leaf(code, start, end, fetch, prepare, save)
            except cap_error:
                self._record(key, code, 'SPLIT')
        if start == end:
            raise CenterError('FUND_EVENT_UNSPLITTABLE', f'{code} 公告日 {start} 仍触顶，停止而非提交残缺结果。')
        with self.mutex:
            self.acknowledgements.append(dict(ts_code=code, start_date=start, end_date=end))
        left, right = pd.Timestamp(start), pd.Timestamp(end)
        middle = left + (right - left) // 2
        # A split hint, NEVER a lower boundary: the earlier half is also fetched.
        found = self.inceptions[code]
        if found and start < found < end:
            middle = max(middle, pd.Timestamp(found))
        middle = middle.normalize()
        following = (middle + pd.Timedelta(days=1)).strftime('%Y%m%d')
        print(f'[INFO] {code} 公告区间 {start}—{end} 触顶，拆分于 {middle:%Y%m%d}；本次请求 {self.calls}/{self.max_requests}。')
        return (self._history_leaf(code, start, middle.strftime('%Y%m%d'), fetch, prepare, save, cap_error)
                + self._history_leaf(code, following, end, fetch, prepare, save, cap_error))

    def _fetch_history_leaf(self, code, start, end, fetch, prepare, save):
        key = start + '-' + end
        for attempt in range(2):
            self.check()
            frame = fetch(code, start, end)
            with self.mutex:
                self.activity = time.monotonic()
            self._validate(frame, None, code)
            if not frame.empty:
                break
            if attempt == 0:
                print(f'[INFO] {code} 公告区间 {start}—{end} 空响应，独立复核。')
                self.pause(0.25)
        if frame.empty:
            self._record(key, code, 'EMPTY', confirmations=2)
            return []
        # Provider observation (2026-09-08): start/end filter ann_date, despite
        # doc 121 calling them report dates. Lock this contract; never guess a
        # different date axis per response or silently discard out-of-range rows.
        announcements = frame.ann_date.astype(str)
        if not (announcements.ge(start) & announcements.le(end)).all():
            raise CenterError('FUND_EVENT_RESPONSE',
                              f'{code} 公告区间 {start}—{end} 返回公告日期 '
                              f'{announcements.min()}—{announcements.max()}，拒绝越界分片。')
        if start != end or not self.market_paged:
            self._confirm_unpaged_conflicts(frame, lambda: fetch(code, start, end))
        frame = self._prepare(frame, key, code, prepare)
        path, _ = self._paths(key, code)
        save(frame, path, quiet=True)
        self._record(key, code, 'COMPLETE', rows=len(frame), sha256=_checksum(path))
        return [path]

    def announcement_pages(self, code, date, fetch, *, page_size, max_pages, confirm_first_empty=True):
        """Bounded raw pages; duplicates/conflicts require a stable full reread."""
        def read_page(page):
            self.check()
            frame = fetch(page * page_size, page_size)
            # The outer request owns the first-page empty confirmation. Only
            # terminal empty pages after data must be confirmed here as well.
            if frame.empty and (page > 0 or confirm_first_empty) and not self.smoke:
                self.pause(0.25)
                frame = fetch(page * page_size, page_size)
            with self.mutex:
                self.activity = time.monotonic()
            self._validate(frame, date, code)
            if len(frame) > page_size:
                raise CenterError('FUND_EVENT_PAGINATION', '供应商未遵守分页大小，拒绝残缺或重复结果。')
            return frame

        frames, seen, pages = [], {}, []
        overlap_count = 0
        for page in range(max_pages):
            frame = read_page(page)
            raw_count = len(frame)
            signature, identities = _page_evidence(frame)
            pages.append(signature)
            if not frame.empty:
                overlapping = identities.keys() & seen.keys()
                if overlapping and len(overlapping) == len(identities):
                    raise CenterError('FUND_EVENT_PAGINATION', f'公告日 {date} 第 {page + 1} 页无新业务键，疑似整页重放；未提交数据。')
                overlap_count += len(overlapping) + sum(len(values) > 1 for values in identities.values())
                if overlapping:
                    print(f'[INFO] 公告日 {date} 第 {page + 1} 页有 {len(overlapping)} 个跨页业务键，取得末页后复核；数值冲突不会任意去重。')
                for key, signatures in identities.items():
                    seen.setdefault(key, set()).update(signatures)
                frames.append(frame.drop_duplicates())
                repeated = raw_count - len(frame.drop_duplicates())
                if repeated:
                    print(f'[INFO] 公告日 {date} 本页合并 {repeated} 条完全相同的供应商重复行。')
            print(f'[INFO] {code or "全市场"} 公告日 {date} 分页 {page + 1}/{max_pages}，'
                  f'本页原始 {raw_count} 行，已接收 {len(seen)} 个唯一业务键。')
            # Never terminate using the deduplicated count.
            if raw_count < page_size:
                if overlap_count:
                    print(f'[STAGE] 公告日 {date} 跨页重复稳定性复核：{len(pages)} 页，共享原请求预算。')
                    for number, expected in enumerate(pages):
                        actual, _ = _page_evidence(read_page(number))
                        if actual != expected:
                            raise CenterError('FUND_EVENT_PAGINATION', f'公告日 {date} 第 {number + 1} 页复核不一致，供应商数据或排序已变化；未提交数据。')
                        print(f'[INFO] 公告日 {date} 页面复核 {number + 1}/{len(pages)}。')
                    print(f'[INFO] 公告日 {date} 全部页面复核一致；相同记录合并，冲突记录保留并隔离。')
                return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame(columns=self.fields)
        raise CenterError('FUND_EVENT_PAGINATION', '持仓达到最大分页数但未取得末页，检查点保留；未提交残缺结果。')

    def run_history(self, *, fetch, prepare, save, cap_error, max_workers, sort_columns):
        from .fund_event_merge import merge_event_parts
        print(f'[STAGE] 持仓全历史按基金/公告区间下载：{len(self.inceptions)} 只，'
              f'不逐日遍历全部基金；保留转型前历史。请求预算 {self.max_requests}。')
        def operation(code):
            try:
                return self._history_leaf(code, self.dates[0], self.dates[-1], fetch, prepare, save, cap_error)
            except Exception as exc:
                with self.mutex:
                    if self.failure is None:
                        self.failure, self.failed_date = exc, code
                    self.stop.set()
                raise
        results = self._run_units(list(self.inceptions), operation, max_workers, '基金')
        imported = []
        for date in sorted(self.imports):
            if date in self.dates:
                self._import_date(date, prepare, save)
                imported.append(self._paths(date, None)[0])
        paths = imported + [path for code in self.inceptions for path in results[code]]
        print('[STAGE] 持仓报告分片已下载，正在有界归并与校验。')
        def merge_check():
            with self.mutex:
                self.activity = time.monotonic()
            self.check()
        merged = merge_event_parts(paths, self.directory / 'history_merged.parquet', sort_columns, merge_check)
        return [merged] if merged else []
