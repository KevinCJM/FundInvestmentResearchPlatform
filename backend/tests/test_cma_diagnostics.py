"""Counterexamples to overstrong review claims, plus real diagnostic-service coverage."""
import json
from datetime import date, timedelta

import numpy as np
import pytest
from scipy.stats import lognorm

from backend.custom_indicators.errors import ValidationError
from backend.strategic_allocation import cma_statistical_kernels as numeric, cma_model_kernels as models, reference_evidence_kernels
from backend.strategic_allocation.contracts import CmaRequest
from backend.tests.test_strategic_allocation import workspace, warm
from backend.tests.test_ltcma_statistics import request


@pytest.fixture(scope="module", autouse=True)
def ready():
    models.warm(); reference_evidence_kernels.warm(); numeric.warm()


@pytest.mark.parametrize("sign", [1., -1.])
def test_diagonal_shrinkage_does_not_always_reduce_long_only_risk(sign):
    series = np.sin(np.arange(80))*.01
    returns = np.column_stack((series, sign*series))
    sample = numeric.historical_estimate(returns,0.)[1]
    shrunk = numeric.historical_estimate(returns,.1)[1]
    w=np.array([.5,.5])
    change=w @ (shrunk-sample) @ w
    assert change == pytest.approx(-2*.1*w[0]*w[1]*sample[0,1])
    assert change*sign < 0


def test_ml_mixture_and_unbiased_total_have_different_denominators():
    returns=np.random.default_rng(3).normal(size=(100,3))*.01
    states=np.array([0]*40+[1]*60,dtype=np.int64)
    returns[states==1] += .005
    counts,means,covariances=numeric.conditional_state_moments(returns,states,2,0.)
    p=counts/counts.sum()
    mean,total,_,between=models.mixture_moments_kernel(p,means,covariances,False)
    np.testing.assert_allclose(total,np.cov(returns,rowvar=False,ddof=0),atol=1e-15)
    within_unbiased=covariances*(counts/(counts-1))[:,None,None]
    wrong_total=models.mixture_moments_kernel(p,means,within_unbiased,False)[1]
    assert not np.allclose(wrong_total,np.cov(returns,rowvar=False,ddof=1),rtol=1e-5,atol=0)
    combined=(np.einsum('s,sij->ij',counts-1,within_unbiased)+len(returns)*between)/(len(returns)-1)
    np.testing.assert_allclose(combined,np.cov(returns,rowvar=False,ddof=1),atol=1e-15)


def test_persistent_volatility_regimes_need_not_have_return_autocorrelation():
    # Conditional independent, zero-mean innovations are a mathematical counterexample.
    p=np.array([.5,.5]); transition=np.array([[.95,.05],[.05,.95]])
    conditional_means=np.array([[0.],[0.]])
    covariances=np.array([[[.0001]],[[.001]]])
    mean,risk,_,_=models.mixture_moments_kernel(p,conditional_means,covariances,False)
    for lag in (1,5,100):
        gamma=np.sum(p[:,None]*np.linalg.matrix_power(transition,lag)*conditional_means*conditional_means.T)-mean[0]**2
        assert gamma==0.
    assert numeric.annualize_moments(mean,risk,252)[1][0,0]==pytest.approx(252*.00055)


def test_equal_moments_do_not_order_success_probabilities():
    # Gross outcomes 0.6/1.1 with probabilities .2/.8: E=1, variance=.04.
    outcomes=np.array([.6,1.1]); probabilities=np.array([.2,.8])
    assert probabilities @ outcomes==pytest.approx(1.)
    variance=probabilities @ (outcomes-1.)**2
    sigma=np.sqrt(np.log1p(variance)); distribution=lognorm(s=sigma,scale=np.exp(-.5*sigma*sigma))
    for threshold,proxy_is_higher in ((.7,True),(1.05,False)):
        mixture_success=probabilities[outcomes>=threshold].sum()
        assert (distribution.sf(threshold)>mixture_success)==proxy_is_higher


def test_transition_counts_stationary_and_readonly_stride():
    owner=np.repeat(np.array([0,0,1,1,0,1,1,0],dtype=np.int64),2)
    states=owner[::2];states.flags.writeable=False
    before=owner.copy()
    counts,p,duration,stationary,status=numeric.regime_transition_diagnostics_kernel(states,2)
    np.testing.assert_array_equal(counts,[[1,2],[2,2]])
    np.testing.assert_allclose(stationary,[3/7,4/7],atol=1e-14)
    np.testing.assert_allclose(stationary @ p,stationary,atol=1e-14)
    np.testing.assert_allclose(duration,[1.5,2.])
    assert status==0 and np.shares_memory(states,owner)
    np.testing.assert_array_equal(owner,before)


def test_unknown_labels_break_transitions_and_unidentified_rows_stay_missing():
    states=np.array([0,0,-1,1,1],dtype=np.int64)
    counts,p,duration,stationary,status=numeric.regime_transition_diagnostics_kernel(states,2)
    np.testing.assert_array_equal(counts,np.eye(2,dtype=np.int64))
    assert status==2 and np.isnan(stationary).all() and np.isnan(duration).all()
    missing=numeric.regime_transition_diagnostics_kernel(states,3)
    assert missing[-1]==1 and np.isnan(missing[1][2]).all()
    single=numeric.regime_transition_diagnostics_kernel(np.zeros(30,dtype=np.int64),1)
    assert single[-1]==0 and single[3].tolist()==[1.] and np.isnan(single[2][0])
    with pytest.raises(ValueError):numeric.regime_transition_diagnostics_kernel(np.array([-2],dtype=np.int64),2)


def test_short_window_is_visible_not_an_arbitrary_new_hard_gate(workspace):
    service,_=workspace
    result=service.preview_cma(request())
    audit=result["model_result"]["model_audit"]
    assert audit["evidence"]["sample_horizon"]["short_window_review"]
    assert audit["evidence"]["sample_horizon"]["observation_years"]==pytest.approx(160/252)
    assert len(audit["mean_standard_error"])==2
    np.testing.assert_allclose(np.square(audit["mean_standard_error"]),np.diag(result["model_result"]["mean_estimation_covariance"]))
    assert any("不是统计有效性硬门槛" in warning for warning in result["warnings"])


@pytest.mark.parametrize("future_field", [None,"recognized_at","available_at"])
def test_regime_service_persists_diagnostics_but_rejects_future_labels(workspace,future_field):
    service,days=workspace
    from backend.historical_regimes.v2_service import _stored_run_snapshot_hash
    service.cma.evidence.prepare_regime_reader()
    root=service.cma.evidence.regime_root;root.mkdir(parents=True,exist_ok=True)
    raw={"id":"regime-cma-review","name":"离线状态证据","schema_version":"2.0","immutable":True,
        "mode":"retrospective","frequency":"daily","as_of":str(date.today()),"states":[{"id":"up"},{"id":"down"}],
        "application_bindings":[],"publications":[],
        "series":[{"observation_date":str(day.date()),"available_at":str(day.date()),"recognized_at":str(day.date()),"state_id":"up" if i%2 else "down"} for i,day in enumerate(days)]}
    if future_field:raw["series"][0][future_field]=str(date.today()+timedelta(days=1))
    raw["content_hash"]=_stored_run_snapshot_hash(raw)
    path=root/"historical_regime_runs.json";path.write_text(json.dumps({"items":[raw]}))
    original=path.read_bytes()
    query=request("historical_regime_occupancy",run_ref={"id":raw["id"],"content_hash":raw["content_hash"]})
    if future_field:
        with pytest.raises(ValidationError,match="研究日之后"):
            service.preview_cma(query)
    else:
        result=service.preview_cma(query)
        evidence=result["model_result"]["model_audit"]["transition_diagnostics"]
        assert evidence["matrix"]==[[0.,1.],[1.,0.]]
        assert evidence["stationary_probabilities"]==[.5,.5]
        assert evidence["forecast_used"] is False
    assert path.read_bytes()==original
