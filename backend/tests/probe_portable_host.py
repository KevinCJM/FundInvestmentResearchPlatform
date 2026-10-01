"""Compare the same warm business lookup directly and through authenticated HTTP receipts."""
import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time

import pytest
from test_portable_agent_host import host, context, tool
from research_access.contracts import PageContext
from research_access.tools import execute_business


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    threshold = {'http_p95_seconds': 2, 'overhead_p95_seconds': 1}
    with tempfile.TemporaryDirectory(prefix='portable-business-benchmark-') as folder, pytest.MonkeyPatch.context() as patch:
        fixture = host.__wrapped__(Path(folder), patch)
        client, bridge, service, _ = next(fixture)
        try:
            ctx = context(client)
            page = PageContext.model_validate(bridge.store.context(ctx['ref'], subject='alice', workspace='lab')['page_context'])
            arguments = {'kind': 'indicators', 'query': '累计', 'limit': 5}
            direct, http = [], []
            for i in range(53):
                start = time.perf_counter()
                expected = execute_business('metrics.lookup', arguments, authoring={'scope': 'indicator_center'}, page_context=page, service=service)
                direct_time = time.perf_counter()-start
                start = time.perf_counter()
                actual, _ = tool(client, ctx, 'metrics_lookup', arguments)
                http_time = time.perf_counter()-start
                assert actual['status'] == 'succeeded' and actual['model']['result'] == expected['result']
                if i >= 3: direct.append(direct_time); http.append(http_time)
            assert not bridge.tasks
            percentile = lambda values: sorted(values)[int((len(values)-1)*.95)]
            report = {'scope': 'same real indicator catalog; 3 warmups then 50 paired samples', 'thresholds': threshold,
                'direct_p50_seconds': statistics.median(direct), 'direct_p95_seconds': percentile(direct),
                'http_p50_seconds': statistics.median(http), 'http_p95_seconds': percentile(http),
                'overhead_p95_seconds': percentile([b-a for a,b in zip(direct,http)]), 'pending_tasks': len(bridge.tasks)}
            report['status'] = 'passed' if all(report[key] <= value for key,value in threshold.items()) else 'failed'
            args.output.write_text(json.dumps(report,indent=2))
            assert report['status'] == 'passed', report
        finally:
            fixture.close()


if __name__ == '__main__':
    main()
