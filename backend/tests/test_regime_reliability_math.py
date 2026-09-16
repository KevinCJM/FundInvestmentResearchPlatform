import numpy as np
import pytest
from historical_regimes.reliability import kernels as k


@pytest.fixture(scope="module", autouse=True)
def warm():
    k.warm()


def test_confusion_abstentions_unknown_and_iou():
    y = np.array([0,0,1,1,-1], dtype=np.int64)
    p = np.array([0,-1,0,1,1], dtype=np.int64)
    cm, per, summary = k.classification_kernel(y,p,2)
    np.testing.assert_array_equal(cm, [[1,0,1],[1,1,0]])
    np.testing.assert_allclose(summary, [.5,.5,(.5+2/3)/2,.75,1/3])
    np.testing.assert_allclose(per[:,4], [1/3,.5])
    empty = k.classification_kernel(y[:0],p[:0],2)
    assert np.isnan(empty[2]).all()


def test_probability_scores_true_formulas_and_chosen_state():
    y = np.array([0,1],np.int64)
    p = np.array([1,1],np.int64)
    q = np.array([[.8,.2],[.4,.6]])
    scores, bins = k.probability_kernel(y,p,q,2)
    assert scores[0] == 2
    assert scores[1] == pytest.approx(np.mean(np.sum((q-np.eye(2))**2,axis=1)))
    assert scores[2] == pytest.approx(-np.log([.8,.6]).mean())
    assert scores[3] == pytest.approx(.3)
    assert bins[0,1] == .2  # chosen state, not max
    bad = q.copy(); bad[0,0] = np.inf
    assert k.probability_kernel(y,p,bad,2)[0][0] == 1


def test_calibration_never_learns_from_test_labels():
    y = np.array([0,1,0,1,1,0],np.int64)
    p = np.array([0,0,1,1,0,1],np.int64)
    q = np.array([[.8,.2],[.7,.3],[.4,.6],[.3,.7],[.9,.1],[.1,.9]])
    for method in (0,1):
        before = k.calibrate_kernel(y,p,q,4,method)
        changed = y.copy(); changed[4:] = 1-changed[4:]
        after = k.calibrate_kernel(changed,p,q,4,method)
        for a,b in zip(before,after):
            np.testing.assert_allclose(a,b,equal_nan=True)
    np.testing.assert_allclose(k.calibrate_kernel(y,p,q,4,0)[0], .5)


def test_interval_transition_matching_and_gaps():
    y = np.array([0,0,1,1,0,0,1,1],np.int64)
    p = np.array([0,0,0,1,0,0,0,1],np.int64)
    counts,stats = k.events_kernel(y,p,1)
    assert counts[8] == 3
    assert counts[9] == counts[10] == 0
    assert stats[1] == 1
    assert counts[5] == 2
    counts,_ = k.events_kernel(np.array([0,-1,1],np.int64),np.array([0,-1,1],np.int64),1)
    assert counts[6] == 0


def test_readonly_stride_zero_copy_and_fixed_signatures():
    base = np.arange(12,dtype=np.int64)
    view = base[::2]; view.flags.writeable = False
    assert np.shares_memory(view,base)
    np.testing.assert_array_equal(k.align_kernel(view,view),np.arange(6))
    pbase = np.tile([.4,.6],(12,1)); pv = pbase[::2]; pv.flags.writeable=False
    labels = np.zeros(12,np.int64)[::2]; labels.flags.writeable=False
    k.probability_kernel(labels,labels,pv,2)
    assert np.shares_memory(pv,pbase)
    assert k.audit()["python_fallback"] == 0
    with pytest.raises(TypeError):
        k.align_kernel(view.astype(np.int32),view)
    assert all(len(fn.signatures)==1 for fn,_ in k.KERNELS)


def test_mapping_real_reference_axis_and_invalid_mass():
    q=np.array([[.2,.3,.5],[np.nan,0,1.]])
    out=k.map_probability_kernel(q,np.array([1,0,1],np.int64),2)
    np.testing.assert_allclose(out[0],[.3,.7])
    assert np.isnan(out[1]).all()


def test_independent_intervals_and_no_false_events_in_unknown_reference():
    y=np.array([-1,-1,0,0,1,1],np.int64)
    p=np.array([0,1,0,0,1,1],np.int64)
    summary,_=k.events_kernel(y,p,1)
    assert summary[10]==0
    assert summary[7]==1
    # Multiple predicted events cannot match the same reference transition.
    y=np.array([0,0,0,1,1,1,1,1],np.int64)
    p=np.array([0,0,1,0,1,0,1,1],np.int64)
    summary,_=k.events_kernel(y,p,3)
    assert summary[8]==1
    assert summary[10]==4


def test_no_calibration_class_null_and_readiness_closed(monkeypatch):
    y=np.array([0,0,1],np.int64);p=np.array([0,0,1],np.int64)
    raw=np.array([[1.,0],[1.,0],[0,1.]])
    q,_,_,_=k.calibrate_kernel(y,p,raw,2,0)
    assert np.isnan(q[2]).all()
    with monkeypatch.context() as m:
        m.setattr(k,"_READY",False)
        with pytest.raises(RuntimeError,match="NOT_READY"):
            k.audit()


def test_numeric_boundary_guards_and_no_input_mutation():
    y=np.array([0,1],np.int64);p=y.copy();q=np.eye(2)
    with pytest.raises(ValueError):
        k.classification_kernel(y,p[:1],2)
    with pytest.raises(ValueError):
        k.calibrate_kernel(y,p,q,3,0)
    with pytest.raises(ValueError):
        k.map_probability_kernel(q,np.array([0,2],np.int64),2)
    originals=[x.copy() for x in (y,p,q)]
    k.calibrate_kernel(y,p,q,1,0)
    for original,current in zip(originals,(y,p,q)):
        np.testing.assert_array_equal(original,current)


def test_calibration_support_counts_only_usable_past_pairs():
    y = np.array([0, 1, 0, 1, 0, 1], np.int64)
    pred = y.copy()
    raw = np.tile([.4, .6], (6, 1))
    pairs, support, sufficient = k.calibration_support_kernel(y, pred, raw, 4, 0, 4, 2)
    assert pairs == 4 and sufficient
    np.testing.assert_array_equal(support, [[2, 2], [2, 2]])
    pred[4:] = -1  # Future changes cannot change calibration sufficiency.
    raw[4:] = np.nan
    after = k.calibration_support_kernel(y, pred, raw, 4, 1, 4, 2)
    assert after[0] == pairs and after[2]
    np.testing.assert_array_equal(after[1], support)
    raw[1] = np.nan
    assert not k.calibration_support_kernel(y, pred, raw, 4, 1, 4, 2)[2]
    assert k.calibration_support_kernel(y, pred, raw, 4, 0, 4, 2)[2]
    pred[:4] = 0  # Many reference classes do not imply calibrated output classes.
    assert not k.calibration_support_kernel(y, pred, raw, 4, 0, 4, 1)[2]
    for array in (y, pred, raw):
        array.flags.writeable = False
    assert not k.calibration_support_kernel(y[::2], pred[::2], raw[::2], 2, 0, 2, 1)[2]
    assert np.shares_memory(raw[::2], raw)
    with pytest.raises(ValueError):
        k.calibration_support_kernel(y, pred, raw, 7, 0, 2, 1)


def test_each_worker_requires_its_own_startup_warmup(monkeypatch):
    monkeypatch.setattr(k,"_WARMED_PID",-1)
    with pytest.raises(RuntimeError,match="NOT_READY"):
        k.audit()
    k.warm()
    assert k.audit()["complete"]


def test_state_evidence_separates_rare_episode_from_failure():
    from historical_regimes.reliability.contracts import Policy
    from historical_regimes.reliability.report import state_verification

    y = np.array([0,0,1,1,0,0,1,1,0,0,2,2,1,1,0,0,1,1,0,0], np.int64)
    accepted = y.copy()
    counts, metrics = k.state_evidence_kernel(y, accepted, 3)
    assert counts[2, 3] == 1  # one independent complete sideways episode
    assert metrics[2, 0] == pytest.approx(1.0)
    policy = Policy(calibration_end='2020-01-01', minimum_state_episodes=2,
                    minimum_state_predictions=2, minimum_state_precision=.65)
    result = state_verification(y, accepted, ['bull','sideways','bear'], policy, .3, .6, 'test')
    by_state = {row['state_id']: row for row in result['states']}
    assert by_state['bear']['status'] == 'insufficient_evidence'
    assert 'insufficient_independent_state_episodes' in by_state['bear']['reasons']
    assert by_state['bear']['precision'] == pytest.approx(1.0)
    assert result['status'] == 'partially_verified'
    assert result['recognition_ready'] is True
    assert result['purpose'] == 'recognition_state_evidence'
    assert result['unverified_state_policy'] == 'do_not_authorize_unverified_states'
    assert 'bear' in result['fallback_states']


def test_state_with_enough_episodes_but_low_precision_is_failed():
    from historical_regimes.reliability.contracts import Policy
    from historical_regimes.reliability.report import state_verification

    y = np.array([0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1], np.int64)
    accepted = np.array([1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1], np.int64)
    policy = Policy(calibration_end='2020-01-01', minimum_state_episodes=2,
                    minimum_state_predictions=2, minimum_state_precision=.8)
    result = state_verification(y, accepted, ['bull','bear'], policy, .3, .6, 'test')
    assert any(row['status'] == 'failed' for row in result['states'])
    assert any('accepted_state_precision_below_policy' in row['reasons'] for row in result['states'])
