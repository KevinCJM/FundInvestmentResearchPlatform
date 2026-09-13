"""Trusted RPC adapter. Candidate code executes only in this disposable process."""
import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
from types import SimpleNamespace


def execute(root, operation, payload):
    sys.path.insert(0, root + '/scripts')
    if operation == 'call':
        module = importlib.import_module(payload['module'])
        target = module
        for part in payload['function'].split('.'):
            target = getattr(target, part)
        return target(*payload.get('args', []), **payload.get('kwargs', {}))
    gate = importlib.import_module('check_ai_review')
    if operation == 'published_records':
        gh = gate.GitHub()
        def pages(path, key):
            if key == 'workflow_runs':
                return payload['runs']
            if key == 'jobs':
                return payload['jobs']
            return payload.get('fake_checks', [])
        gh.pages = pages
        subprocess.run = lambda *args, **kwargs: SimpleNamespace(returncode=payload.get('log_code', 0), stdout=payload['log'])
        return gh.published_records(payload['policy'])
    if operation == 'reviews':
        gh = gate.GitHub()
        gh.pages = lambda *args: payload['records']
        pages = iter(payload['pages'])
        gh.api = lambda *args: next(pages)
        return gh.reviews(11)
    if operation == 'current_pr':
        values = iter(payload['responses'])
        gh = SimpleNamespace(api=lambda path: next(values))
        return gate.current_pr(gh, 11)
    if operation == 'request':
        values = iter(payload['responses'])
        gate.GitHub = lambda: SimpleNamespace(api=lambda path: next(values))
        sys.argv = ['check_ai_review.py', '--pr', '11', '--request-body',
                    '--expected-head', payload['head'], '--expected-base', payload['base']]
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            code = gate.main()
        return {'code': code, 'body': output.getvalue()}
    publisher = importlib.import_module('check_submission')
    policy = importlib.import_module('submission_policy')
    if operation == 'quality':
        class Evidence:
            def pages(self, path, key):
                return payload['jobs'] if key == 'jobs' else payload['runs']

            def api(self, path):
                sha = payload['base_blob'] if ('ref=' + payload['pr']['base']['sha']) in path else payload['run_blob']
                return {'type': 'file', 'sha': sha}
        return publisher.quality_evidence(Evidence(), payload['pr'])
    if operation == 'plan':
        planner = importlib.import_module('ci_quality')
        calls = []
        values = iter(payload['responses'])
        def api(path):
            calls.append(path)
            return next(values)
        planner.GitHub = lambda: SimpleNamespace(api=api)
        os.environ.update(PR_NUMBER='11', PR_HEAD=payload['head'], PR_BASE=payload['base'],
                          GITHUB_OUTPUT=payload['output_path'])
        with contextlib.redirect_stdout(io.StringIO()):
            planner.main()
        return {'calls': calls, 'output': open(payload['output_path']).read()}
    if operation == 'verify':
        policies = iter(payload.get('policy_reads', [payload['final_policy']] * 2))
        class Evidence:
            def api(self, path):
                if '/git/ref/' in path:
                    return {'object': {'sha': next(policies)}}
                return {'type': 'file', 'sha': payload['remote_blob']}

            def published_records(self, policy_sha):
                return {11: payload['record']} if payload['record'] else {}

            def collect(self, pr):
                return payload['snapshot']

            def pages(self, path, key):
                return payload['checks']
        gh = Evidence()
        snapshots = iter(payload['versions'])
        publisher.current_pr = lambda *args: next(snapshots)
        publisher.evaluate = lambda *args: payload['ai']
        publisher.quality_evidence = lambda *args: payload['quality']
        subprocess.run = lambda *args, **kwargs: SimpleNamespace(stdout=payload['local_blob'] + '\n')
        return publisher.verify_submission(gh, 11, payload['head'], payload['base'])
    if operation == 'publish':
        writes = []
        reads = 0
        class Evidence:
            def api(self, path, *args, **kwargs):
                nonlocal reads
                if '/git/ref/heads/main' in path:
                    reads += 1
                    return {'object': {'sha': payload['new_policy'] if reads >= payload['advance_at'] else payload['policy']}}
                raise ValueError('Unexpected API request')

            def pages(self, *args):
                return [payload['pr']]

            def context(self, *args, **kwargs):
                return payload['context']

            def collect(self, *args):
                return payload['snapshot']

            def publish(self, context, result, *args, **kwargs):
                writes.append(result)
                return len(writes)
        publisher.GitHub = Evidence
        publisher.current_pr = lambda *args: [payload['pr'], payload['base'], payload['head']]
        publisher.evaluate = lambda *args: {'state': 'success'}
        publisher.quality_evidence = lambda *args: ['success', 'passed', {'id': 1, 'html_url': 'quality'}]
        os.environ.update(GITHUB_ACTIONS='true', GITHUB_REPOSITORY=policy.REPOSITORY,
                          GITHUB_EVENT_NAME='workflow_dispatch', GITHUB_REF='refs/heads/main',
                          GH_TOKEN='fixture-only', SUBMISSION_POLICY_SHA=payload['policy'])
        sys.argv = ['check_submission.py', '--publish']
        with contextlib.redirect_stdout(io.StringIO()):
            code = publisher.main()
        return {'code': code, 'writes': writes}
    raise ValueError('Unknown protected operation')


if __name__ == '__main__':
    request = json.loads(sys.stdin.read())
    try:
        value = execute(sys.argv[1], request['operation'], request['payload'])
        result = {'value': value}
    except (Exception, SystemExit) as error:
        result = {'error': type(error).__name__}
    print(json.dumps(result))
