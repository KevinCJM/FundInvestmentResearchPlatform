"""Read-only task cards: recovery lineage is metadata, not another task to do."""
import re
import json
import threading
import weakref


_records = weakref.WeakKeyDictionary()
_records_lock = threading.Lock()


def read_run_records(store):
    """Cache unchanged journals, not live state. Poll the small revision columns.

    Frozen contracts make journals large; rereading every ancestor on each
    progress poll holds read locks and overloads a removable disk. Single-flight
    initialization prevents multiple browser tabs from repeating that scan.
    """
    with _records_lock:
        if store not in _records:
            # The revision columns follow the large JSON body in the SQLite
            # record. A covering index avoids walking its overflow pages just
            # to check whether it changed, especially on an external disk.
            store.database_operation(lambda db: db.execute(
                'CREATE INDEX IF NOT EXISTS etl_run_view_revision ON etl_run(id,updated_at,cancel_requested)'))
        cache = _records.setdefault(store, {})
        with store.connection() as db:
            revisions = db.execute('SELECT id,updated_at,cancel_requested FROM etl_run').fetchall()
        active = set()
        for identifier, updated_at, cancelled in revisions:
            active.add(identifier)
            signature = (updated_at, cancelled)
            if identifier not in cache or cache[identifier][0] != signature:
                with store.connection() as db:
                    row = db.execute('SELECT body,updated_at,cancel_requested FROM etl_run WHERE id=?', (identifier,)).fetchone()
                if row is None:
                    active.discard(identifier)
                    continue
                cache[identifier] = ((row[1], row[2]), {**json.loads(row[0]), 'cancel_requested': bool(row[2])})
        for identifier in set(cache) - active:
            del cache[identifier]
        return [cache[identifier][1] for identifier, _, _ in revisions if identifier in cache]


def current_run_views(records):
    """Keep independent roots/branches separate; never infer identity from names.

    Group before limiting so a chain older than 30 runs still has its origin.
    Corrupt cycles and active ancestors remain visible instead of being hidden.
    """
    indexed = {r['run_id']: r for r in records}
    parents = {r['run_id']: r.get('recovered_from') for r in records}
    ancestors = {value for value in parents.values() if value in indexed}
    visible, covered = [], set()

    def card(run):
        chain, seen = [], set()
        cursor = run['run_id']
        while cursor in indexed and cursor not in seen:
            seen.add(cursor)
            chain.append(indexed[cursor])
            cursor = parents.get(cursor)
        origin = chain[-1]
        # Old names accumulated a machine suffix; leave independent user names alone.
        name = origin['name']
        if origin.get('recovered_from'):
            name = re.sub(r'(?:\s*（恢复）)+$', '', name)
        history = [{
            'run_id': item['run_id'], 'status': item['status'],
            'created_at': item.get('created_at'), 'finished_at': item.get('finished_at'),
            'attempt': item.get('attempt', 1), 'error': item.get('error'),
            'failed_step': next((s['name'] for s in item.get('steps', []) if s['status'] == 'FAILED'), None),
        } for item in chain[1:]]
        metadata = {'root_run_id': origin['run_id'], 'display_name': name,
                    'records': history, 'resume_count': sum(max(0, r.get('attempt', 1) - 1) for r in chain) + len(history),
                    'lineage_warning': '历史关联不完整或成环，原记录仍保留。' if cursor else None}
        covered.update(seen)
        visible.append((run, metadata, chain))

    for run in records:
        if run['run_id'] not in ancestors or run['status'] == 'RUNNING':
            card(run)
    for run in records:
        if run['run_id'] not in covered:
            card(run)
    return sorted(visible, key=lambda item: (item[0].get('updated_at', ''), item[0]['run_id']), reverse=True)[:30]
