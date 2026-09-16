import copy
import pytest
from historical_regimes.reliability.consumer import calibrated_output
from historical_regimes.reliability import kernels
from historical_regimes.taa import run_taa_backtest
from test_historical_regime_taa import _synthetic_run, _taa_request
from test_tactical_allocation_service import workspace


def test_new_unresolved_study_saa_legacy_deterministic_replay():
    run=_synthetic_run()
    before=copy.deepcopy(run)
    old=run_taa_backtest(run,_taa_request(),{"passed":True})
    assert not old["weights"][1]["fallback_to_base"]
    assert run==before
    run["definition"]={"study":{"purpose":"realtime_recognition"}}
    result=run_taa_backtest(run,_taa_request(),{"passed":True})
    assert all(p["fallback_to_base"] for p in result["weights"])
    assert result["weights"][1]["fallback_reason"]=="calibration_not_resolved"


def _eligible_fixture():
    # Unit-only trusted read model; service-generated reports remain diagnostic.
    run=_synthetic_run()
    run["definition"]={"study":{"purpose":"realtime_recognition"}}
    states=["bull","bear"]
    run["_reliability"]={"error":None,"artifact":{"created_at":"2024-01-01T00:00:00Z","report":{
        "states":run["states"],"lineage":{"state_mapping":{s:s for s in states}},
        "calibration":{"deployment_eligible":True,"available_from":"2024-01-01","expires_on":"2024-03-31",
                       "confidence_floor":.1,"method":"class_frequency",
                       "parameters":{"counts":[[2.,8.],[3.,7.]],"temperature":None}}}}}
    return run


def test_eligible_output_final_chosen_state_not_max_and_dates():
    kernels.warm()
    run=_eligible_fixture();point=run["series"][0]
    q,confidence,reason=calibrated_output(run,point,"2024-01-02",["bull","bear"])
    assert reason is None and confidence==.2
    assert q=={"bull":.2,"bear":.8}
    for as_of,expected in [("2023-12-31","calibration_not_yet_available"),("2024-04-01","calibration_expired")]:
        assert calibrated_output(run,point,as_of,["bull","bear"])[2]==expected
    assert calibrated_output(run,point,"2024-01-02",["bear","bull"])[2]=="calibration_state_axis_mismatch"
    unknown={**point,"state_id":"unclassified"}
    assert calibrated_output(run,unknown,"2024-01-02",["bull","bear"])[1] is None
    run["_reliability"]["artifact"]["report"]["calibration"]["deployment_eligible"]=False
    assert calibrated_output(run,point,"2024-01-02",["bull","bear"])[2]=="calibration_not_deployment_eligible"


def test_tactical_consumer_new_study_falls_back_with_reason(workspace):
    service,_,request=workspace
    service.warm()
    run={"states":[{"id":"risk_on"}],"definition":{"study":{"purpose":"realtime_recognition"}},
         "series":[{"state_id":"risk_on","observation_date":str(request.start_date),
                    "recognized_at":str(request.start_date),"effective_date":str(request.start_date),
                    "probabilities":{"risk_on":1.},"confidence":1.}]}
    service.regime_resolver=lambda _: (run,{"passed":True})
    body=request.model_copy(update={"signal_mode":"regime","regime_run_id":"new-study",
                                   "state_tilts":{"risk_on":{"股票":.1,"债券":-.1}},"max_signal_age_days":3650})
    result=service.preview(body)
    assert all(row["weights"]=={"股票":.6,"债券":.4} for row in result["weight_path"])
    assert all(row["fallback_reason"]=="calibration_not_resolved" for row in result["audit"]["signal"]["signal_timing"])


@pytest.mark.parametrize('mapped', [False, True])
def test_tactical_eligible_read_model_uses_same_warmed_calibrator(workspace, mapped):
    from datetime import timedelta
    service,_,request=workspace
    service.warm();kernels.warm()
    run=_eligible_fixture()
    start=str(request.start_date)
    run["series"]= [{**run["series"][0],"observation_date":start,"recognized_at":start,"effective_date":start}]
    artifact=run["_reliability"]["artifact"]
    artifact["created_at"]=str(request.start_date-timedelta(days=1))+"T00:00:00Z"
    artifact["report"]["calibration"].update(available_from=start,expires_on=str(request.as_of+timedelta(days=1)))
    tilts = {"bull":{"股票":.1,"债券":-.1},"bear":{"股票":-.1,"债券":.1}}
    if mapped:
        axis = [{"id": "BullRef"}, {"id": "BearRef"}]
        run['_reliability']['states'] = axis
        artifact['report']['states'] = axis
        artifact['report']['lineage']['state_mapping'] = {'bull': 'BullRef', 'bear': 'BearRef'}
        tilts = {artifact['report']['lineage']['state_mapping'][key]: value for key, value in tilts.items()}
    service.regime_resolver=lambda _: (run,{"passed":True})
    body=request.model_copy(update={"signal_mode":"regime","regime_run_id":"calibrated-unit-fixture",
             "state_tilts":tilts,
             "max_signal_age_days":3650,"confidence_floor":.1})
    result=service.preview(body)
    assert all(row["confidence"]==.2 for row in result["audit"]["signal"]["signal_timing"])
    assert all(row["fallback_reason"] is None for row in result["audit"]["signal"]["signal_timing"])


def test_partial_forward_qualification_only_authorizes_qualified_state():
    kernels.warm()
    run = _eligible_fixture()
    context = run["_reliability"]
    context["artifact"]["report"]["calibration"]["deployment_eligible"] = False
    context["qualification"] = {"id": "forward-q", "status": "qualified",
        "qualified_states": ["bull"], "fallback_states": ["bear"],
        "available_from": "2025-01-01T12:00:00+00:00", "expires_at": "2025-02-01T00:00:00+00:00"}
    context["verified_at"] = "2025-01-10T12:00:00+00:00"
    run["definition"]["study"]["qualification_id"] = "forward-q"
    bear = {**run["series"][0], "state_id": "bear", "observation_date": "2025-01-02",
            "recognized_at": "2025-01-02", "effective_date": "2025-01-02"}
    assert calibrated_output(run, bear, "2025-01-02", ["bull", "bear"])[2] == "calibration_state_not_qualified"
    bull = {**bear, "state_id": "bull"}
    assert calibrated_output(run, bull, "2025-01-02", ["bull", "bear"])[2] is None


def test_both_taa_consumers_reject_an_explicit_failed_gate(workspace):
    from custom_indicators.errors import ValidationError
    run = _eligible_fixture()
    with pytest.raises(ValidationError, match='发布'):
        run_taa_backtest(run, _taa_request(), {'passed': False})
    service, _, request = workspace
    service.regime_resolver = lambda _: (run, {'passed': False})
    from backend.custom_indicators.errors import ValidationError as TacticalValidationError
    with pytest.raises(TacticalValidationError) as caught:
        service._signals(request.model_copy(update={'signal_mode': 'regime', 'regime_run_id': 'unpublished'}),
                         {'returns': __import__('numpy').empty((0, 2)), 'period_starts': []}, ['股票', '债券'])
    assert caught.value.code == 'TAA_RUN_NOT_PUBLISHED'


def test_nonidentity_mapping_uses_reference_axis_in_calibration_and_taa():
    kernels.warm()
    run = _eligible_fixture()
    states = [{'id': 'BullRef', 'label': '参考正常'}, {'id': 'BearRef', 'label': '参考压力'}]
    run['_reliability']['states'] = states
    report = run['_reliability']['artifact']['report']
    report['states'] = states
    report['lineage']['state_mapping'] = {'bull': 'BullRef', 'bear': 'BearRef'}
    probs, confidence, reason = calibrated_output(run, run['series'][0], '2024-01-02', ['BullRef', 'BearRef'])
    assert reason is None and probs == {'BullRef': .2, 'BearRef': .8}
    assert confidence == .2
    request = _taa_request()
    request['confidence_floor'] = .1
    request['state_tilts'] = {report['lineage']['state_mapping'][key]: value for key, value in request['state_tilts'].items()}
    result = run_taa_backtest(run, request, {'passed': True})
    assert not result['weights'][1]['fallback_to_base']


@pytest.mark.parametrize('error_name', ['ConflictError', 'NotFoundError'])
def test_qualification_domain_errors_are_caught_and_degrade(monkeypatch, error_name):
    from types import SimpleNamespace
    from custom_indicators import errors
    from historical_regimes.reliability import consumer
    run = _eligible_fixture()
    artifact = run['_reliability']['artifact']
    reference = {'run_id': 'reference', 'publication_id': 'publication', 'content_hash': 'hash'}
    run['definition']['study'].update(reference=reference, calibration_id='calibration', qualification_id='qualification')
    artifact['request'] = {'definition_id': run['definition_id'], 'reference': reference}
    monkeypatch.setattr(consumer, 'resolve_reference', lambda *a, **k: ({'states': run['states']}, {}))
    monkeypatch.setattr(consumer, 'parse_definition_v2', lambda value: value)
    monkeypatch.setattr(consumer, 'model_binding_hash', lambda value: 'binding')
    def verify(*args):
        raise getattr(errors, error_name)('QUALIFICATION_UNAVAILABLE', '资格不可用')
    service = SimpleNamespace(get=lambda identity: artifact, graph=SimpleNamespace(prospective=SimpleNamespace(verify_qualification=verify)))
    attached = consumer.attach_calibration(run, service)
    assert attached['_reliability']['error'] == 'calibration_qualification_unavailable'
    assert consumer.calibrated_output(attached, run['series'][0], '2024-01-02', ['bull', 'bear']) == (None, None, 'calibration_qualification_unavailable')
