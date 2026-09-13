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

ROOT = Path(os.environ.get("FIRP_GATE_ROOT", Path(__file__).parents[2]))
sys.path.insert(0, str(ROOT / "scripts"))
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

    def test_sync_branch_must_include_current_main(self):
        pr = pr_fixture(); pr['head']['ref'] = 'codex/sync-main-release'
        self.assertEqual(self.decision(pr), 'failure')
        self.assertEqual(self.decision(pr, latest_main=BASE, main_is_ancestor=False), 'failure')
        self.assertEqual(self.decision(pr, latest_main=BASE, main_is_ancestor=True), 'success')



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
        root=ROOT
        quality=yaml.load((root/'.github/workflows/quality-gate.yml').read_text(), Loader=yaml.BaseLoader)
        publisher=yaml.load((root/'.github/workflows/ai-review.yml').read_text(), Loader=yaml.BaseLoader)
        self.assertIn('pull_request',quality['on']); self.assertNotIn('pull_request_target',quality['on'])
        self.assertEqual(set(quality['jobs']),policy.QUALITY_JOBS)
        self.assertTrue(all(value=='read' for value in quality['permissions'].values()))
        self.assertNotIn('secrets.',json.dumps(quality))
        self.assertTrue(all('environment' not in job for job in quality['jobs'].values()))
        self.assertNotIn('environment',publisher['jobs']['publish'])
        self.assertEqual(publisher['jobs']['publish']['permissions']['checks'],'write')
        self.assertNotIn('secrets.',json.dumps(publisher))
        self.assertNotIn('create-github-app-token',json.dumps(publisher))
        self.assertNotIn('pull_request.head',json.dumps(publisher['jobs']))
        self.assertIn('github.workflow_sha',json.dumps(publisher['jobs']))
        self.assertIn("github.ref == 'refs/heads/main'", publisher['jobs']['publish']['if'])
        self.assertNotIn('environment', publisher['jobs']['dispatch'])
        governance=json.dumps(quality['jobs']['governance'])
        self.assertIn('../trusted/skills/',governance)
        self.assertNotIn('unittest',governance)
        self.assertIn('unittest',json.dumps(quality['jobs']['policy-tests']))
        protected=json.dumps(quality['jobs']['protected-policy-tests'])
        self.assertIn('trusted/scripts/tests',protected)
        self.assertIn('FIRP_GATE_ROOT',protected)
        self.assertIn('countTestCases',protected)
        candidate=json.dumps(quality['jobs']['policy-tests'])
        for command in ['compileall -q skills','validate_ai_routing.py','route_task.py','evolve_ai_routing.py']:
            self.assertIn(command,candidate)
        for workflow in [quality,publisher]:
            for job in workflow['jobs'].values():
                for step in job.get('steps',[]):
                    if 'uses' in step:
                        self.assertRegex(step['uses'],r'@[a-f0-9]{40}$')
                        if step['uses'].startswith('actions/checkout@'):
                            self.assertEqual(step['with']['persist-credentials'],'false')

    def test_review_event_relay_has_no_privileges_or_candidate_execution(self):
        import yaml
        relay=yaml.load((ROOT/'.github/workflows/review-events.yml').read_text(),Loader=yaml.BaseLoader)
        publisher=yaml.load((ROOT/'.github/workflows/ai-review.yml').read_text(),Loader=yaml.BaseLoader)
        self.assertEqual(relay['on'],{'pull_request_review':{'types':['submitted','edited','dismissed']},
            'pull_request_review_comment':{'types':['created','edited','deleted']}})
        self.assertEqual(relay['permissions'],{})
        self.assertEqual(relay['jobs']['signal']['steps'],[{'run':"echo 'Review evidence changed; protected main will revalidate.'"}])
        self.assertNotIn('environment',relay['jobs']['signal'])
        self.assertNotIn('permissions',relay['jobs']['signal'])
        self.assertIn(relay['name'],publisher['on']['workflow_run']['workflows'])
        self.assertEqual(publisher['on']['workflow_run']['types'],['completed'])

    def test_integrity_is_checked_before_candidate_imports_and_routing_before_tests(self):
        import yaml
        workflow=yaml.load((ROOT/'.github/workflows/quality-gate.yml').read_text(),Loader=yaml.BaseLoader)
        steps=workflow['jobs']['protected-policy-tests']['steps']
        integrity=next(i for i,step in enumerate(steps) if step.get('name')=='Verify workflow integrity before importing candidate code')
        candidate_import=next(i for i,step in enumerate(steps) if step.get('name')=='Run protected contracts against candidate gate modules')
        self.assertLess(integrity,candidate_import)
        steps=workflow['jobs']['policy-tests']['steps']
        routing=next(i for i,step in enumerate(steps) if step.get('name')=='Exercise candidate routing executables in their isolated runner')
        tests=next(i for i,step in enumerate(steps) if step.get('name')=='Run nonempty candidate test suite')
        self.assertLess(routing,tests)

    def test_deployable_ruleset_preserves_submission_requirements(self):
        config=json.loads((ROOT/'.github/branch-ruleset.json').read_text())
        self.assertEqual(config['target'],'branch')
        self.assertEqual(config['enforcement'],'active')
        self.assertEqual(config['bypass_actors'],[])
        self.assertEqual(config['conditions'],{'ref_name':{'include':['refs/heads/main','refs/heads/Dev'],'exclude':[]}})
        rules={rule['type']:rule for rule in config['rules']}
        self.assertEqual(len(rules),len(config['rules']))
        self.assertEqual(set(rules),{'deletion','non_fast_forward','pull_request','required_status_checks'})
        pr=rules['pull_request']['parameters']
        self.assertEqual(pr['required_approving_review_count'],0)
        self.assertEqual(pr['required_reviewers'],[])
        self.assertEqual(pr['allowed_merge_methods'],['merge'])
        for name in ['dismiss_stale_reviews_on_push','required_review_thread_resolution','require_extra_approval_for_unattributed_changes']:
            self.assertIs(pr[name],True)
        for name in ['require_code_owner_review','require_last_push_approval']:
            self.assertIs(pr[name],False)
        checks=rules['required_status_checks']['parameters']
        self.assertIs(checks['strict_required_status_checks_policy'],True)
        self.assertIs(checks['do_not_enforce_on_create'],False)
        self.assertEqual(checks['required_status_checks'],[
            {'context':name,'integration_id':15368} for name in ['branch-policy','ai-review','quality-gate']])

    def test_test_controllers_reject_early_exit_skips_and_expected_failures(self):
        import yaml
        workflow=yaml.load((ROOT/'.github/workflows/quality-gate.yml').read_text(),Loader=yaml.BaseLoader)
        cases=[("import os; os._exit(0)\n",False),
            ("import unittest\nclass T(unittest.TestCase):\n def test_ok(self): pass\n",True),
            ("import unittest\n@unittest.skip('skip')\nclass T(unittest.TestCase):\n def test_ok(self): pass\n",False),
            ("import unittest\nclass T(unittest.TestCase):\n @unittest.expectedFailure\n def test_bad(self): self.fail()\n",False)]
        for job,source in [('policy-tests','scripts/tests'),('protected-policy-tests','trusted/scripts/tests')]:
            step=workflow['jobs'][job]['steps'][-1]['run']
            self.assertIn('python3 -I',step)
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory); tests=root/source;tests.mkdir(parents=True)
                for code,passed in cases:
                    (tests/'test_candidate.py').write_text(code)
                    run=subprocess.run(['bash','-ec',step],cwd=root,capture_output=True,text=True)
                    self.assertEqual(run.returncode==0,passed,(job,code,run.stderr))

    def test_actual_integrity_step_rejects_configuration_changes_and_invalid_next_contract(self):
        import shutil, yaml
        workflow=yaml.load((ROOT/'.github/workflows/quality-gate.yml').read_text(),Loader=yaml.BaseLoader)
        script=next(step['run'] for step in workflow['jobs']['protected-policy-tests']['steps']
                    if step.get('name')=='Verify workflow integrity before importing candidate code')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for owner in ['trusted','candidate']:
                shutil.copytree(ROOT/'.github',root/owner/'.github')
            def run():
                return subprocess.run(['bash','-ec',script],cwd=root,capture_output=True,text=True).returncode
            self.assertEqual(run(),0)
            path=root/'candidate/.github/workflows/quality-gate.yml'; original=path.read_text()
            for job,field in [('frontend','run'),('backend','run'),('e2e','run'),
                              ('protected-policy-tests','if'),('protected-policy-tests','continue-on-error'),('governance','if')]:
                altered=yaml.load(original,Loader=yaml.BaseLoader)
                step=next(step for step in altered['jobs'][job]['steps'] if 'run' in step)
                step[field]='true' if field in {'run','continue-on-error'} else 'false'
                path.write_text(yaml.safe_dump(altered,sort_keys=False))
                self.assertNotEqual(run(),0,(job,field))
            path.write_text(original+'\n# Formatting alone is not a behavior change.\n')
            self.assertEqual(run(),0)
            rules_path=root/'candidate/.github/branch-ruleset.json'; original_rules=rules_path.read_text()
            for field,value in [('enforcement','disabled'),('bypass_actors',[{'actor_id':1,'bypass_mode':'always'}]),('rules',[])]:
                config=json.loads(original_rules);config[field]=value;rules_path.write_text(json.dumps(config))
                self.assertNotEqual(run(),0,field)
            rules_path.write_text(original_rules)
            (root/'candidate/.github/workflow-contracts.json').write_text('{}')
            self.assertNotEqual(run(),0)

    def test_aggregate_script_rejects_unexpected_skip_failure_and_cancellation(self):
        import yaml
        path=ROOT/'.github/workflows/quality-gate.yml'
        workflow=yaml.load(path.read_text(),Loader=yaml.BaseLoader)
        script=workflow['jobs']['quality-result']['steps'][0]['run'].split("python3 - <<'PY'\n",1)[1].rsplit('\nPY',1)[0]
        jobs={name:{'result':'success'} for name in policy.QUALITY_JOBS-{'quality-result'}}
        jobs['prepare']['outputs']={key:'true' for key in ['frontend','backend','e2e']}
        for state in ['success','failure','cancelled','skipped']:
            changed=copy.deepcopy(jobs);changed['backend']['result']=state
            run=subprocess.run([sys.executable,'-c',script],env=os.environ|{'NEEDS':json.dumps(changed)},capture_output=True)
            self.assertEqual(run.returncode==0,state=='success')

    def test_protected_contract_step_detects_broken_candidate_with_no_candidate_tests(self):
        import yaml
        workflow=yaml.load((ROOT/'.github/workflows/quality-gate.yml').read_text(),Loader=yaml.BaseLoader)
        script=workflow['jobs']['protected-policy-tests']['steps'][-1]['run']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); tests=root/'trusted/scripts/tests'; tests.mkdir(parents=True)
            candidate=root/'candidate'; candidate.mkdir()
            contract=tests/'test_contract.py'
            contract.write_text("import os, unittest\nfrom pathlib import Path\nclass Contract(unittest.TestCase):\n def test_behavior(self):\n  self.assertEqual((Path(os.environ['FIRP_GATE_ROOT'])/'result.txt').read_text(), 'correct')\n")
            for behavior in ['correct','broken']:
                (candidate/'result.txt').write_text(behavior)
                run=subprocess.run(['bash','-ec',script],cwd=root,env=os.environ|{'FIRP_GATE_ROOT':str(candidate)},capture_output=True,text=True)
                self.assertEqual(run.returncode==0,behavior=='correct')
            contract.unlink()
            run=subprocess.run(['bash','-ec',script],cwd=root,env=os.environ|{'FIRP_GATE_ROOT':str(candidate)},capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
            self.assertIn('Test suite is empty',run.stderr)

    def test_candidate_step_rejects_empty_test_suite(self):
        import yaml
        workflow=yaml.load((ROOT/'.github/workflows/quality-gate.yml').read_text(),Loader=yaml.BaseLoader)
        script=next(step['run'] for step in workflow['jobs']['policy-tests']['steps']
                    if step.get('name')=='Run nonempty candidate test suite')
        with tempfile.TemporaryDirectory() as directory:
            tests=Path(directory)/'scripts/tests'; tests.mkdir(parents=True)
            run=subprocess.run(['bash','-ec',script],cwd=directory,capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
            self.assertIn('Test suite is empty',run.stderr)
            (tests/'test_basic.py').write_text('import unittest\nclass Basic(unittest.TestCase):\n def test_ok(self): self.assertTrue(True)\n')
            self.assertEqual(subprocess.run(['bash','-ec',script],cwd=directory,capture_output=True).returncode,0)

    def test_candidate_routing_syntax_error_fails_the_actual_workflow_step(self):
        import yaml
        workflow=yaml.load((ROOT/'.github/workflows/quality-gate.yml').read_text(),Loader=yaml.BaseLoader)
        command=next(step['run'] for step in workflow['jobs']['policy-tests']['steps'] if step.get('name')=='Exercise candidate routing executables in their isolated runner').splitlines()[0]
        with tempfile.TemporaryDirectory() as directory:
            scripts=Path(directory)/'skills'; scripts.mkdir()
            (scripts/'route_task.py').write_text('def broken(:\n')
            result=subprocess.run(['bash','-ec',command],cwd=directory,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('SyntaxError',result.stdout)


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
             'GH_TOKEN':'test-only','SUBMISSION_POLICY_SHA':BASE}
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


class TerminalVerificationTests(unittest.TestCase):
    def setup_evidence(self):
        gh = Mock(); pr = pr_fixture()
        context = {'pr_number': 11, 'head_sha': HEAD, 'base_sha': BASE, 'base_ref': 'Dev',
                   'policy_sha': BASE, 'observed_at': '2026-09-13T00:00:00Z'}
        gh.api.side_effect = lambda path: {'object': {'sha': BASE}} if '/git/ref/' in path else {'type': 'file', 'sha': 'blob'}
        gh.published_records.return_value = {11: {'context': context, 'results': {
            name: {'state': 'success'} for name in ('branch-policy', 'ai-review', 'quality-gate')}}}
        gh.collect.return_value = {'base_is_ancestor': True, 'other_prs_with_same_head': []}
        gh.pages.return_value = [{'id': i, 'name': name, 'app': {'id': publisher.ACTIONS_APP_ID},
            'status': 'completed', 'conclusion': 'success'} for i, name in enumerate(('branch-policy', 'ai-review', 'quality-gate'))]
        return gh, pr

    def verify(self, gh, pr, ai_state='success'):
        with patch('subprocess.run', return_value=Mock(stdout='blob\n')), \
                patch.object(publisher, 'current_pr', return_value=(pr, BASE, HEAD)), \
                patch.object(publisher, 'evaluate', return_value={'state': ai_state, 'reason': 'review'}), \
                patch.object(publisher, 'quality_evidence', return_value=('success', 'quality', None)):
            return publisher.verify_submission(gh, 11, HEAD, BASE)

    def test_green_checks_alone_cannot_replace_authentic_publisher_log(self):
        gh, pr = self.setup_evidence(); gh.published_records.return_value = {}
        with self.assertRaisesRegex(ValueError, 'authentic publisher'):
            self.verify(gh, pr)

    def test_current_evidence_is_recomputed_after_real_success(self):
        gh, pr = self.setup_evidence()
        self.assertEqual(self.verify(gh, pr)['state'], 'success')
        with self.assertRaisesRegex(ValueError, 'Fresh evidence failed'):
            self.verify(gh, pr, 'failure')

    def test_modified_local_verifier_and_changed_version_are_rejected(self):
        gh, pr = self.setup_evidence()
        with patch('subprocess.run', return_value=Mock(stdout='different\n')):
            with self.assertRaisesRegex(ValueError, 'unmodified verifier'):
                publisher.verify_submission(gh, 11, HEAD, BASE)
        gh.published_records.return_value[11]['context']['base_sha'] = 'c' * 40
        with self.assertRaisesRegex(ValueError, 'authentic publisher'):
            self.verify(gh, pr)

    def test_publisher_pending_and_new_check_failure_block_merge(self):
        gh, pr = self.setup_evidence()
        gh.published_records.return_value[11]['results']['ai-review']['state'] = 'pending'
        with self.assertRaisesRegex(ValueError, 'not passed'):
            self.verify(gh, pr)
        gh, pr = self.setup_evidence()
        gh.pages.return_value.append({'id': 99, 'name': 'ai-review', 'app': {'id': publisher.ACTIONS_APP_ID},
                                     'status': 'completed', 'conclusion': 'failure'})
        with self.assertRaisesRegex(ValueError, 'Required check'):
            self.verify(gh, pr)


if __name__=='__main__': unittest.main()
