"""Base-owned assertions. Never import candidate Python into this interpreter."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(os.environ.get('FIRP_GATE_ROOT', Path(__file__).parents[2])).resolve()
# Read the trusted worker before any candidate process can start.
WORKER = Path(__file__).with_name('worker.py').read_text()
REPO = 'KevinCJM/FundInvestmentResearchPlatform'
HEAD, BASE, POLICY = 'a' * 40, 'b' * 40, 'c' * 40
START, END = '2026-09-11T10:00:00Z', '2026-09-11T10:10:00Z'
BOT = {'id': 199175422, 'login': 'chatgpt-codex-connector[bot]', 'type': 'Bot'}
JOBS = ['prepare', 'governance', 'policy-tests', 'protected-policy-tests', 'frontend', 'backend', 'e2e', 'quality-result']
CHECKS = ['branch-policy', 'ai-review', 'quality-gate']
RULESET = json.loads((ROOT / '.github/branch-ruleset.json').read_text())


def rpc(operation, payload, root=ROOT, timeout=20):
    result = subprocess.run([sys.executable, '-I', '-c', WORKER, str(root)],
        input=json.dumps({'operation': operation, 'payload': payload}), capture_output=True,
        text=True, cwd=root, timeout=timeout)
    if result.returncode or len(result.stdout) > 2_000_000:
        raise RuntimeError('Candidate process failed')
    response = json.loads(result.stdout)
    if not isinstance(response, dict) or set(response) not in ({'value'}, {'error'}):
        raise ValueError('Invalid candidate response')
    return response


def call(module, function, *args, **kwargs):
    result = rpc('call', {'module': module, 'function': function, 'args': args, 'kwargs': kwargs})
    if 'error' in result:
        raise RuntimeError(result['error'])
    return result['value']


def pr():
    return {'number': 11, 'state': 'open', 'draft': False,
        'head': {'sha': HEAD, 'ref': 'codex/task', 'repo': {'full_name': REPO}},
        'base': {'sha': BASE, 'ref': 'Dev', 'repo': {'full_name': REPO}}}


def request_body(candidate):
    pair = {'repository': REPO, 'pr_number': 11, 'head_sha': candidate['head']['sha'],
            'base_sha': candidate['base']['sha'], 'base_ref': candidate['base']['ref']}
    return ('@codex review\n\nPlease review the complete PR diff against branch_submission_rules.md, '
        'including P0/P1/P2 findings. Review exactly the HEAD/base recorded below; '
        'if either changes, report INCOMPLETE. Do not modify code.\n\n'
        '<!-- codex-review-request/v1 ' + json.dumps(pair, sort_keys=True) + ' -->')


def native():
    candidate = pr()
    context = {'schema': 'codex-review/v1', 'pr_number': 11, 'base_ref': 'Dev',
               'head_sha': HEAD, 'base_sha': BASE, 'observed_at': START}
    snapshot = {'pr': candidate, 'base_is_ancestor': True, 'other_prs_with_same_head': [],
        'reviews': [], 'findings': [], 'resolved_commits': {'aaaaaaa': HEAD},
        'comments': [{'user': BOT.copy(), 'performed_via_github_app': {'id': 1144995},
            'updated_at': END, 'html_url': 'summary', 'body': '<!-- codex-pull-request-review-summary -->\n'
            '| 📝 **Code Review** | ✅ **Completed** <relative-time datetime="' + END + '">done</relative-time> | `aaaaaaa` | Manual request |'},
            {'id': 2, 'body': request_body(candidate) + '\n', 'created_at': START, 'updated_at': START}],
        'reactions': [{'user': BOT.copy(), 'content': '+1', 'created_at': END, 'request_comment_id': 2}]}
    return snapshot, context


def quality():
    candidate = pr()
    run = {'id': 10, 'run_attempt': 1, 'repository': {'full_name': REPO},
        'path': '.github/workflows/quality-gate.yml', 'event': 'pull_request', 'head_sha': HEAD,
        'display_title': f'quality PR 11 head:{HEAD} base:{BASE}', 'pull_requests': [candidate],
        'status': 'completed', 'conclusion': 'success', 'workflow_verified': True}
    jobs = [{'name': name, 'status': 'completed', 'conclusion': 'success', 'run_attempt': 1} for name in JOBS]
    return candidate, run, jobs


class ProtectedGateContracts(unittest.TestCase):
    def test_00_ruleset_invariants(self):
        config=RULESET
        self.assertEqual(config['target'],'branch');self.assertEqual(config['enforcement'],'active')
        self.assertEqual(config['bypass_actors'],[])
        self.assertEqual(config['conditions'],{'ref_name':{'include':['refs/heads/main','refs/heads/Dev'],'exclude':[]}})
        rules={rule['type']:rule for rule in config['rules']}
        self.assertEqual(len(rules),len(config['rules']))
        self.assertEqual(set(rules),{'deletion','non_fast_forward','pull_request','required_status_checks'})
        params=rules['pull_request']['parameters']
        self.assertEqual(params['required_approving_review_count'],0)
        self.assertEqual(params['required_reviewers'],[]);self.assertEqual(params['allowed_merge_methods'],['merge'])
        for key in ['dismiss_stale_reviews_on_push','required_review_thread_resolution','require_extra_approval_for_unattributed_changes']:
            self.assertIs(params[key],True)
        for key in ['require_code_owner_review','require_last_push_approval']:
            self.assertIs(params[key],False)
        checks=rules['required_status_checks']['parameters']
        self.assertIs(checks['strict_required_status_checks_policy'],True)
        self.assertIs(checks['do_not_enforce_on_create'],False)
        self.assertEqual(checks['required_status_checks'],[{'context':name,'integration_id':15368} for name in CHECKS])

    def test_branch_matrix(self):
        candidate = pr()
        options = {'base_is_ancestor': True, 'latest_base': BASE, 'latest_source': HEAD}
        def state(value, extra=None):
            return call('submission_policy', 'branch_decision', value, **(options | (extra or {})))[0]
        self.assertEqual(state(candidate), 'success')
        release = copy.deepcopy(candidate); release['head']['ref'] = 'Dev'; release['base']['ref'] = 'main'
        self.assertEqual(state(release), 'success')
        for key, value in [('state', 'closed'), ('draft', True)]:
            changed = copy.deepcopy(candidate); changed[key] = value
            self.assertEqual(state(changed), 'failure')
        for side, key, value in [('base', 'ref', 'main'), ('head', 'ref', 'main'),
                ('head', 'ref', 'release/test'), ('head', 'repo', {'full_name': 'fork/repo'})]:
            changed = copy.deepcopy(candidate); changed[side][key] = value
            self.assertEqual(state(changed), 'failure')
        for extra in [{'base_is_ancestor': False}, {'latest_base': POLICY},
                      {'latest_source': POLICY}, {'duplicate_heads': [12]}]:
            self.assertEqual(state(candidate, extra), 'failure')
        candidate['head']['ref'] = 'codex/sync-main-test'
        self.assertEqual(state(candidate), 'failure')
        self.assertEqual(state(candidate, {'latest_main': POLICY, 'main_is_ancestor': True}), 'success')

    def test_native_review_matrix(self):
        snapshot, context = native()
        def state(s, c=context):
            return call('check_ai_review', 'evaluate', s, c)['state']
        self.assertEqual(state(snapshot), 'success')
        mutations = [lambda s: s['pr'].update(draft=True), lambda s: s['pr']['head'].update(sha=POLICY),
            lambda s: s.update(base_is_ancestor=False), lambda s: s.update(other_prs_with_same_head=[12]),
            lambda s: s['comments'][0]['user'].update(id=1),
            lambda s: s['comments'][0]['performed_via_github_app'].update(id=1),
            lambda s: s['comments'][1].update(updated_at=END), lambda s: s.update(reactions=[]),
            lambda s: s['reactions'][0].update(request_comment_id=99),
            lambda s: s['reactions'][0].update(created_at='2026-09-10T10:00:00Z'),
            lambda s: s.update(findings=[{'resolved': False, 'url': 'finding', 'updated_at': END}]),
            lambda s: s.update(findings=[{'resolved': True, 'url': 'finding', 'updated_at': '2026-09-11T10:11:00Z'}]),
            lambda s: s['comments'][0].update(body=s['comments'][0]['body'].replace('Completed', 'Running')),
            lambda s: s.update(reviews=[{'user': BOT, 'state': 'COMMENTED', 'commit_id': HEAD,
                'submitted_at': START, 'updated_at': '2026-09-11T10:11:00Z', 'body': 'new finding'}])]
        for mutate in mutations:
            changed = copy.deepcopy(snapshot); mutate(changed)
            self.assertNotEqual(state(changed), 'success')
        self.assertNotEqual(state(snapshot, context | {'base_sha': POLICY}), 'success')
        for ending in ["You're on a roll.", "Can't wait for the next one!"]:
            changed = copy.deepcopy(snapshot); changed['reactions'] = []
            changed['comments'].append({'user': BOT, 'performed_via_github_app': {'id': 1144995},
                'created_at': END, 'updated_at': END, 'html_url': 'native',
                'body': "Codex Review: Didn't find any major issues. " + ending + '\n\n**Reviewed commit:** `aaaaaaa`'})
            self.assertEqual(state(changed), 'success')
        report = context | {'repository': REPO, 'conclusion': 'PASS', 'reviewed_scope': 'README only'}
        changed = copy.deepcopy(snapshot)
        changed['reviews'] = [{'user': BOT, 'state': 'COMMENTED', 'commit_id': HEAD, 'submitted_at': END,
            'updated_at': END, 'html_url': 'partial', 'body': '```json\n' + json.dumps(report) + '\n```'}]
        self.assertNotEqual(state(changed), 'success')

    def test_quality_and_source_matrix(self):
        candidate, run, jobs = quality()
        payload = {'pr': candidate, 'runs': [run], 'jobs': jobs, 'base_blob': 'trusted', 'run_blob': 'trusted'}
        self.assertEqual(rpc('quality', payload)['value'][0], 'success')
        for field, value in [('path', 'fake.yml'), ('event', 'push'), ('head_sha', POLICY),
                             ('display_title', 'fake'), ('conclusion', 'failure')]:
            changed = copy.deepcopy(payload); changed['runs'][0][field] = value
            self.assertNotEqual(rpc('quality', changed)['value'][0], 'success')
        for changed in [payload | {'run_blob': 'untrusted'}, payload | {'jobs': jobs[:-1]},
                        payload | {'jobs': [{**job, 'run_attempt': 2} for job in jobs]}]:
            self.assertNotEqual(rpc('quality', changed)['value'][0], 'success')
        for status in ['failure', 'cancelled', 'skipped', 'neutral']:
            for name in JOBS:
                changed = copy.deepcopy(payload)
                next(job for job in changed['jobs'] if job['name'] == name)['conclusion'] = status
                actual = rpc('quality', changed)['value'][0]
                if status == 'skipped' and name in {'frontend', 'backend', 'e2e'}:
                    self.assertEqual(actual, 'success')
                else:
                    self.assertNotEqual(actual, 'success')
        dispatched=copy.deepcopy(payload)
        dispatched['runs'][0].update(event='workflow_dispatch',head_branch='Dev',head_sha=BASE,pull_requests=[])
        self.assertEqual(rpc('quality',dispatched)['value'][0],'success')
        dispatched['runs'][0]['head_branch']='codex/task'
        self.assertNotEqual(rpc('quality',dispatched)['value'][0],'success')
        changed=copy.deepcopy(payload);changed['runs'][0]['run_attempt']=2
        self.assertNotEqual(rpc('quality',changed)['value'][0],'success')
        pending = copy.deepcopy(payload); pending['runs'][0]['status'] = 'queued'
        self.assertNotEqual(rpc('quality', pending)['value'][0], 'success')

    def test_plan_selection_and_publisher_log_origin(self):
        for paths,wanted in [(['docs/a.md'],[False,False,False]),
                (['frontend/src/a.tsx'],[True,False,True]),(['backend/a.py'],[False,True,True]),
                (['unknown.py'],[True,True,True]),(['.github/requirements-ci.txt'],[False,True,True])]:
            value=call('submission_policy','quality_plan',paths,'Dev')
            self.assertEqual([value[key] for key in ['frontend','backend','e2e']],wanted)
        _,context=native();context['policy_sha']=POLICY
        record={'pr':11,'context':context,'results':{name:{'state':'success'} for name in CHECKS}}
        run={'id':42,'path':'.github/workflows/ai-review.yml','event':'push','head_branch':'main',
             'head_sha':POLICY,'status':'completed','repository':{'full_name':REPO},'run_attempt':1}
        payload={'policy':POLICY,'runs':[run],'jobs':[{'id':9,'name':'publish','conclusion':'success'}],
                 'log':'prefix '+json.dumps(record)+'\n','fake_checks':[{'output':{'text':json.dumps(record)}}]}
        self.assertEqual(rpc('published_records',payload)['value']['11'],record)
        for key,value in [('path','fake.yml'),('event','pull_request'),('head_branch','codex/task'),('head_sha',HEAD)]:
            changed=copy.deepcopy(payload);changed['runs'][0][key]=value
            self.assertEqual(rpc('published_records',changed)['value'],{})
        self.assertIn('error',rpc('published_records',payload|{'log_code':1}))
        self.assertEqual(rpc('published_records',payload|{'log':'{}'})['value'],{})

    def test_immutable_plan_and_truncation(self):
        for files, wanted in [([{'filename': 'docs/test.md'}], False),
                ([{'filename': '.github/requirements-ci.txt'}], True),
                ([{'filename': f'docs/{n}.md'} for n in range(300)], True)]:
            with tempfile.TemporaryDirectory() as directory:
                payload = {'head': HEAD, 'base': BASE, 'output_path': directory + '/plan',
                    'responses': [pr(), {'object': {'sha': BASE}},
                                  {'merge_base_commit': {'sha': BASE}, 'files': files}]}
                value = rpc('plan', payload)['value']
                self.assertEqual(value['calls'][-1], f'repos/{REPO}/compare/{BASE}...{HEAD}')
                self.assertIn('backend=' + str(wanted).lower(), value['output'])
                payload['responses'][1]['object']['sha'] = POLICY
                self.assertIn('error', rpc('plan', payload))

    def test_review_activity_and_pr_identity(self):
        record = {'id': 1, 'user': BOT, 'state': 'COMMENTED', 'submitted_at': START,
                  'body': 'review', 'html_url': 'review-url', 'commit_id': HEAD}
        node = {'databaseId': 1, 'updatedAt': END, 'submittedAt': START, 'body': 'review',
                'state': 'COMMENTED', 'url': 'review-url', 'commit': {'oid': HEAD}}
        def page(nodes, more=False):
            return {'data': {'repository': {'pullRequest': {'reviews': {
                'pageInfo': {'hasNextPage': more, 'endCursor': 'next'}, 'nodes': nodes}}}}}
        payload = {'records': [record], 'pages': [page([], True), page([node])]}
        self.assertEqual(rpc('reviews', payload)['value'][0]['updated_at'], END)
        for nodes in [[], [node, node], [node, node | {'databaseId': 2}],
                      [node | {'state': 'CHANGES_REQUESTED'}], [node | {'body': 'edited'}],
                      [node | {'updatedAt': None}]]:
            self.assertIn('error', rpc('reviews', {'records': [record], 'pages': [page(nodes)]}))
        self.assertIn('error', rpc('reviews', {'records': [record | {'state': 'PENDING'}], 'pages': [page([node])]}))
        responses = [pr(), {'object': {'sha': BASE}}, {'object': {'sha': HEAD}}]
        responses[0]['base']['sha']=POLICY
        actual=rpc('current_pr', {'responses': responses})['value']
        self.assertEqual(actual[1:], [BASE, HEAD]); self.assertEqual(actual[0]['base']['sha'],BASE)
        request = {'responses': responses, 'head': HEAD, 'base': BASE}
        self.assertEqual(rpc('request', request)['value']['body'].rstrip('\n'), request_body(pr()))
        request['responses'][2]['object']['sha'] = POLICY
        self.assertIn('error', rpc('request', request))

    def test_terminal_verification_requires_all_live_evidence(self):
        snapshot, context = native(); context.update(policy_sha=POLICY)
        payload = {'head': HEAD, 'base': BASE, 'final_policy': POLICY, 'local_blob': 'blob', 'remote_blob': 'blob',
            'record': {'context': context, 'results': {name: {'state': 'success'} for name in CHECKS}},
            'snapshot': snapshot, 'checks': [{'id': i, 'name': name, 'app': {'id': 15368},
                'status': 'completed', 'conclusion': 'success'} for i, name in enumerate(CHECKS)],
            'versions': [[pr(), BASE, HEAD], [pr(), BASE, HEAD]],
            'ai': {'state': 'success', 'reason': 'review'}, 'quality': ['success', 'quality', {}]}
        self.assertEqual(rpc('verify', payload)['value']['state'], 'success')
        variants = [payload | {'record': None}, payload | {'local_blob': 'candidate'},
            payload | {'final_policy': 'd' * 40}, payload | {'policy_reads': [POLICY, 'd' * 40]}, payload | {'head': POLICY}, payload | {'base': POLICY},
            payload | {'ai': {'state': 'failure', 'reason': 'review'}}, payload | {'quality': ['failure', 'bad', None]},
            payload | {'versions': [[pr(), BASE, HEAD], [pr(), POLICY, HEAD]]}]
        for name in CHECKS:
            changed = copy.deepcopy(payload)
            next(check for check in changed['checks'] if check['name'] == name)['conclusion'] = 'failure'
            variants.append(changed)
        for changed in variants:
            self.assertIn('error', rpc('verify', changed))

    def test_obsolete_publisher_stops_writing(self):
        snapshot, context = native()
        payload = {'head': HEAD, 'base': BASE, 'policy': POLICY, 'new_policy': 'd' * 40,
                   'pr': pr(), 'snapshot': snapshot, 'context': context, 'advance_at': 99}
        self.assertEqual(rpc('publish', payload)['value']['code'], 0)
        payload['advance_at'] = 3
        result = rpc('publish', payload)['value']
        self.assertEqual(result['code'], 1)
        self.assertEqual(len(result['writes']), 3)
        self.assertTrue(all(write['state'] == 'pending' for write in result['writes']))

    def test_candidate_cannot_replace_parent_assertions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'scripts').mkdir()
            (root / 'scripts/submission_policy.py').write_text(
                'import unittest\nunittest.TestCase.assertEqual=lambda *a,**k:None\n'
                'unittest.TestSuite.countTestCases=lambda *a:99\n'
                'def branch_decision(*a,**k): return ["success", "fabricated"]\n')
            value = rpc('call', {'module': 'submission_policy', 'function': 'branch_decision', 'args': [pr()]}, root)['value']
            with self.assertRaises(AssertionError):
                self.assertEqual(value[0], 'failure')
            for code in ['import os;os._exit(0)', 'print("not-json");import os;os._exit(0)',
                         'print("{}\\n{}");import os;os._exit(0)']:
                (root / 'scripts/submission_policy.py').write_text(code)
                with self.assertRaises((ValueError, RuntimeError)):
                    rpc('call', {'module': 'submission_policy', 'function': 'branch_decision'}, root)
            (root/'scripts/submission_policy.py').write_text('import time;time.sleep(10)')
            with self.assertRaises(subprocess.TimeoutExpired):
                rpc('call',{'module':'submission_policy','function':'branch_decision'},root,timeout=0.05)


if __name__ == '__main__':
    unittest.main()
