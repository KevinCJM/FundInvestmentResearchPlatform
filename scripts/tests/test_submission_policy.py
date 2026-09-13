"""Negative and positive cases for real branch and trusted-CI requirements."""
import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1]))
import submission_policy as policy
import check_submission as publisher
import ci_quality

HEAD, BASE = 'a' * 40, 'b' * 40


def pr_fixture():
    repo = {'full_name': policy.REPOSITORY}
    return {'number': 11, 'state': 'open', 'draft': False,
            'head': {'sha': HEAD, 'ref': 'codex/task', 'repo': repo},
            'base': {'sha': BASE, 'ref': 'Dev', 'repo': repo}}


def run_fixture(pr):
    run = {'id': 10, 'run_attempt': 1, 'repository': {'full_name': policy.REPOSITORY},
           'path': policy.QUALITY_WORKFLOW, 'display_title': policy.quality_title(11, HEAD, BASE),
           'event': 'pull_request', 'head_sha': HEAD, 'pull_requests': [copy.deepcopy(pr)],
           'status': 'completed', 'conclusion': 'success', 'workflow_verified': True}
    jobs = [{'name': name, 'status': 'completed', 'conclusion': 'success', 'run_attempt': 1} for name in policy.QUALITY_JOBS]
    return run, jobs


class BranchPolicyTests(unittest.TestCase):
    def decision(self, pr, **overrides):
        args = {'base_is_ancestor': True, 'latest_base': BASE, 'latest_source': HEAD}
        return policy.branch_decision(pr, **(args | overrides))[0]

    def test_valid_development_and_release(self):
        pr = pr_fixture(); self.assertEqual(self.decision(pr), 'success')
        pr['base']['ref'] = 'main'; pr['head']['ref'] = 'Dev'
        self.assertEqual(self.decision(pr), 'success')

    def test_wrong_direction_fork_draft_closed_release_rejected(self):
        for side, field, value in [('base','ref','main'), ('head','ref','main'), ('head','ref','release/test')]:
            pr = pr_fixture(); pr[side][field] = value
            self.assertEqual(self.decision(pr), 'failure')
        pr = pr_fixture(); pr['head']['repo'] = {'full_name': 'fork/repo'}
        self.assertEqual(self.decision(pr), 'failure')
        for key, value in [('draft',True), ('state','closed')]:
            pr = pr_fixture(); pr[key] = value
            self.assertEqual(self.decision(pr), 'failure')

    def test_stale_source_base_ancestry_and_shared_head_rejected(self):
        for override in [{'latest_base':'c'*40}, {'latest_source':'c'*40}, {'base_is_ancestor':False}, {'duplicate_heads':[12]}]:
            self.assertEqual(self.decision(pr_fixture(), **override), 'failure')

    def test_invalid_identity_rejected(self):
        self.assertEqual(self.decision(pr_fixture(), latest_source='a'), 'failure')


class QualityDecisionTests(unittest.TestCase):
    def test_success_and_documented_optional_children(self):
        pr=pr_fixture(); run,jobs=run_fixture(pr)
        self.assertEqual(policy.quality_decision([run], {10:jobs}, pr)[0], 'success')
        for job in jobs:
            if job['name'] in {'frontend','backend','e2e'}: job['conclusion']='skipped'
        self.assertEqual(policy.quality_decision([run], {10:jobs}, pr)[0], 'success')

    def test_no_run_pending_and_untrusted_workflow_never_passes(self):
        pr=pr_fixture(); run,jobs=run_fixture(pr)
        self.assertEqual(policy.quality_decision([], {}, pr)[0], 'pending')
        for key,value in [('workflow_verified',False), ('path','.github/workflows/fake.yml'), ('event','push'),
                          ('head_sha','c'*40), ('display_title',policy.quality_title(11,HEAD,'c'*40))]:
            changed={**run,key:value}
            self.assertNotEqual(policy.quality_decision([changed], {10:jobs}, pr)[0], 'success')

    def test_dispatch_must_run_from_actual_protected_target(self):
        pr=pr_fixture(); run,jobs=run_fixture(pr)
        run.update(event='workflow_dispatch',head_branch='Dev',head_sha=BASE,pull_requests=[])
        self.assertEqual(policy.quality_decision([run], {10:jobs}, pr)[0], 'success')
        run['head_branch']='codex/task'
        self.assertNotEqual(policy.quality_decision([run], {10:jobs}, pr)[0], 'success')

    def test_failed_cancelled_skipped_and_missing_children_block(self):
        pr=pr_fixture(); run,jobs=run_fixture(pr)
        for bad in ['failure','cancelled','skipped','neutral','timed_out']:
            self.assertEqual(policy.quality_decision([{**run,'conclusion':bad}], {10:jobs}, pr)[0], 'failure')
        for changed in [jobs[:-1], jobs+[jobs[0]], [{**j,'run_attempt':2} for j in jobs]]:
            self.assertEqual(policy.quality_decision([run], {10:changed}, pr)[0], 'failure')
        for job in jobs:
            altered=copy.deepcopy(jobs)
            next(j for j in altered if j['name']==job['name'])['conclusion']='failure'
            self.assertEqual(policy.quality_decision([run], {10:altered}, pr)[0], 'failure')

    def test_newer_run_or_attempt_invalidates_previous_success(self):
        pr=pr_fixture(); run,jobs=run_fixture(pr)
        for new in [{**run,'id':11,'status':'queued'}, {**run,'run_attempt':2,'status':'in_progress'}]:
            self.assertEqual(policy.quality_decision([run,new], {10:jobs}, pr)[0], 'pending')

    def test_plan_docs_governance_frontend_backend_and_unknown(self):
        self.assertEqual(policy.quality_plan(['docs/design.md','scripts/check_submission.py'], 'Dev'),
                         {'frontend':False,'backend':False,'e2e':False})
        self.assertEqual(policy.quality_plan(['frontend/src/App.tsx'], 'Dev'),
                         {'frontend':True,'backend':False,'e2e':True})
        self.assertEqual(policy.quality_plan(['backend/app.py'], 'Dev'),
                         {'frontend':False,'backend':True,'e2e':True})
        self.assertEqual(policy.quality_plan(['.github/requirements-ci.txt'], 'Dev'),
                         {'frontend':False,'backend':True,'e2e':True})
        for paths,base in [(['backend/app.py'],'main'), (['config.py'],'Dev'), (['docs/runtime-config.json'],'Dev')]:
            self.assertTrue(all(policy.quality_plan(paths,base).values()))
        with self.assertRaises(ValueError): policy.quality_plan(['../escape.py'],'Dev')


class WorkflowContractTests(unittest.TestCase):
    def test_test_workflow_has_no_publisher_credentials_or_privileged_pr_trigger(self):
        import yaml
        root=Path(__file__).parents[2]
        quality=yaml.load((root/'.github/workflows/quality-gate.yml').read_text(), Loader=yaml.BaseLoader)
        publisher=yaml.load((root/'.github/workflows/ai-review.yml').read_text(), Loader=yaml.BaseLoader)
        self.assertIn('pull_request',quality['on']); self.assertNotIn('pull_request_target',quality['on'])
        self.assertEqual(set(quality['jobs']),policy.QUALITY_JOBS)
        self.assertTrue(all(value=='read' for value in quality['permissions'].values()))
        self.assertNotIn('secrets.',json.dumps(quality))
        self.assertTrue(all('environment' not in job for job in quality['jobs'].values()))
        self.assertEqual(publisher['jobs']['publish']['environment'],'ai-review-publisher')
        self.assertNotIn('pull_request.head',json.dumps(publisher['jobs']))
        self.assertIn('github.workflow_sha',json.dumps(publisher['jobs']))
        self.assertIn("github.ref == 'refs/heads/main'", publisher['jobs']['publish']['if'])
        self.assertNotIn('environment', publisher['jobs']['dispatch'])
        governance=json.dumps(quality['jobs']['governance'])
        self.assertIn('../trusted/skills/',governance)
        self.assertNotIn('unittest',governance)
        self.assertIn('unittest',json.dumps(quality['jobs']['policy-tests']))
        for workflow in [quality,publisher]:
            for job in workflow['jobs'].values():
                for step in job.get('steps',[]):
                    if 'uses' in step:
                        self.assertRegex(step['uses'],r'@[a-f0-9]{40}$')
                        if step['uses'].startswith('actions/checkout@'):
                            self.assertEqual(step['with']['persist-credentials'],'false')

    def test_aggregate_script_rejects_unexpected_skip_failure_and_cancellation(self):
        import yaml
        path=Path(__file__).parents[2]/'.github/workflows/quality-gate.yml'
        workflow=yaml.load(path.read_text(),Loader=yaml.BaseLoader)
        script=workflow['jobs']['quality-result']['steps'][0]['run'].split("python3 - <<'PY'\n",1)[1].rsplit('\nPY',1)[0]
        jobs={name:{'result':'success'} for name in policy.QUALITY_JOBS-{'quality-result'}}
        jobs['prepare']['outputs']={key:'true' for key in ['frontend','backend','e2e']}
        for state in ['success','failure','cancelled','skipped']:
            changed=copy.deepcopy(jobs);changed['backend']['result']=state
            run=subprocess.run([sys.executable,'-c',script],env=os.environ|{'NEEDS':json.dumps(changed)},capture_output=True)
            self.assertEqual(run.returncode==0,state=='success')


class EvidenceCollectionTests(unittest.TestCase):
    def test_immutable_diff_plan_renames_and_truncation(self):
        for files, expected in [
            ([{'filename':'docs/readme.md'}],False),
            ([{'filename':'docs/retired.md','previous_filename':'backend/retired.py'}],True),
            ([{'filename':f'docs/{i}.md'} for i in range(300)],True),
        ]:
            gh=Mock()
            gh.api.side_effect=[pr_fixture(),{'object':{'sha':BASE}},
                                {'merge_base_commit':{'sha':BASE},'files':files}]
            with tempfile.TemporaryDirectory() as directory:
                output=Path(directory)/'outputs'
                with patch.dict(os.environ,{'PR_NUMBER':'11','PR_HEAD':HEAD,'PR_BASE':BASE,'GITHUB_OUTPUT':str(output)}), \
                        patch.object(ci_quality,'GitHub',return_value=gh),redirect_stdout(io.StringIO()):
                    ci_quality.main()
                self.assertIn(f'backend={str(expected).lower()}',output.read_text())
            self.assertEqual(gh.api.call_args.args[0],f'repos/{policy.REPOSITORY}/compare/{BASE}...{HEAD}')

    def test_base_advance_rejects_test_plan(self):
        gh=Mock();gh.api.side_effect=[pr_fixture(),{'object':{'sha':'c'*40}}]
        with patch.dict(os.environ,{'PR_NUMBER':'11','PR_HEAD':HEAD,'PR_BASE':BASE}), \
                patch.object(ci_quality,'GitHub',return_value=gh),self.assertRaises(ValueError):
            ci_quality.main()

    def test_obsolete_policy_stops_without_final_success_or_failure(self):
        gh=Mock();pr=pr_fixture();gh.pages.return_value=[pr]
        context={'head_sha':HEAD,'base_sha':BASE,'base_ref':'Dev'}
        gh.context.return_value=context
        gh.collect.return_value={'base_is_ancestor':True}
        gh.publish.side_effect=[1,2,3]
        env={'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':policy.REPOSITORY,
             'GITHUB_EVENT_NAME':'workflow_dispatch','GITHUB_REF':'refs/heads/main',
             'AI_REVIEW_CHECKS_TOKEN':'test-only','AI_REVIEW_APP_ID':'123','SUBMISSION_POLICY_SHA':BASE}
        with patch.dict(os.environ,env),patch.object(sys,'argv',['check_submission.py','--publish']), \
                patch.object(publisher,'GitHub',return_value=gh), \
                patch.object(publisher,'current_pr',return_value=(pr,BASE,HEAD)), \
                patch.object(publisher,'quality_evidence',return_value=('success','Tests passed',{'id':1,'html_url':'https://github.com/example/run'})), \
                patch.object(publisher,'evaluate',return_value={'state':'success'}), \
                patch.object(publisher,'ensure_current_policy',side_effect=[None,None,publisher.PolicyChanged()]), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(publisher.main(),1)
        self.assertEqual(gh.publish.call_count,3)
        self.assertTrue(all(call.args[1]['state']=='pending' for call in gh.publish.call_args_list))


if __name__=='__main__': unittest.main()
