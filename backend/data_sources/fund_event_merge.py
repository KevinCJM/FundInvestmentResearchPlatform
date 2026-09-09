"""Bounded external merge of sorted Parquet records; data I/O, no analytics."""
from __future__ import annotations

import heapq
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _rows(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512, use_threads=False):
        yield from batch.to_pylist()


def _key(row, columns):
    return tuple((0, '') if row.get(name) is None else (1, str(row[name])) for name in columns)


def _merge_group(paths, target, columns, check):
    schema = pa.unify_schemas([pq.read_schema(p).remove_metadata() for p in paths])
    streams = [iter(_rows(path)) for path in paths]
    heap, buffer, previous, previous_key = [], [], None, None
    for index, stream in enumerate(streams):
        row = next(stream, None)
        if row is not None:
            heapq.heappush(heap, (_key(row, columns), index, row))
    with pq.ParquetWriter(target, schema, compression='snappy') as writer:
        while heap:
            key, index, row = heapq.heappop(heap)
            if previous is not None and key != previous_key:
                buffer.append(previous)
            previous, previous_key = row, key  # Stable later-part precedence.
            successor = next(streams[index], None)
            if successor is not None:
                heapq.heappush(heap, (_key(successor, columns), index, successor))
            if len(buffer) >= 512:
                check()
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()
        if previous is not None:
            buffer.append(previous)
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))


def merge_event_parts(paths, target, columns, check):
    """At most 32 x 512 decoded rows plus one output batch per merge group."""
    if not paths:
        return None
    with tempfile.TemporaryDirectory(prefix='.event-merge-', dir=target.parent) as scratch:
        current, generation = list(paths), 0
        while True:
            merged = []
            for offset in range(0, len(current), 32):
                check()
                output = Path(scratch) / f'{generation}_{offset}.parquet'
                _merge_group(current[offset:offset + 32], output, columns, check)
                merged.append(output)
                print(f'[INFO] 合并持仓分片 {offset + len(current[offset:offset + 32])}/{len(current)}。')
            if generation:
                for path in current:
                    path.unlink()  # Only this invocation's explicit scratch files.
            if len(merged) == 1:
                check()
                os.replace(merged[0], target)
                return target
            current, generation = merged, generation + 1
