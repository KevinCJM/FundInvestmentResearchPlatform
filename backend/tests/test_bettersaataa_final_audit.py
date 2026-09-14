"""Final integration regressions, using temporary research data only."""
from datetime import date, timedelta

import numpy as np
import pytest

from backend.tests.test_strategic_allocation import workspace, warm
from backend.tests.test_cma_model_integration import model_request, save, mandate, policy
from backend.tactical_allocation import numeric


@pytest.mark.parametrize("row", [[.2, -.2], [1., -1.], [.1, .1], [1.1, -1.1], [np.nan, 0.], [np.inf, -np.inf]])
def test_matrix_tilt_validation_reuses_vector_contract_without_python_dispatch(row):
    owner = np.zeros((6, 4), dtype=np.float64)
    view = owner[::2, ::2]
    view[:] = row
    owner.setflags(write=False)
    view.setflags(write=False)
    before = owner.copy()
    expected = numeric._weight_vector_validation_kernel(view[0], 0., 1e-8, -1., 1.)[1]
    assert numeric.tilt_rows_status_kernel(view, 1e-8) == expected
    assert np.shares_memory(view, owner)
    np.testing.assert_array_equal(owner, before)
    assert len(numeric.tilt_rows_status_kernel.signatures) == 1
    assert not numeric.tilt_rows_status_kernel._can_compile


def test_matrix_tilt_validation_is_in_real_warmup_and_used_once(monkeypatch):
    numeric.warm_tactical_allocation_kernels()
    assert 'tilt_rows_status_kernel' in numeric.execution_audit()['kernel_signatures']
    calls = []
    original = numeric.tilt_rows_status_kernel

    def counted(values, tolerance):
        calls.append(values)
        return original(values, tolerance)

    monkeypatch.setattr(numeric, 'tilt_rows_status_kernel', counted)
    values = np.zeros((40, 2))
    direct = np.broadcast_to(np.array([.1, -.1]), values.shape)
    result = numeric.evaluate_candidates(values, np.ones((40, 1)), np.ones(40, dtype=np.uint8),
        np.array([.5, .5]), np.zeros((1, 2)), np.zeros(2), np.ones(2), np.ones(2), 20,
        direct_tilts=direct)
    assert result['selected_path'].shape[0] == 20
    assert len(calls) == 1
    assert np.shares_memory(calls[0], direct)


def test_expired_policy_explains_disabled_application_and_keeps_historical_research(workspace, monkeypatch):
    service, _ = workspace
    service.warm()
    cma = save(service, model_request())
    target = mandate(service)
    request, initial = policy(service, target, cma)

    class LaterDate(date):
        @classmethod
        def today(cls):
            return date.today() + timedelta(days=120)

    monkeypatch.setattr('backend.strategic_allocation.service.date', LaterDate)
    expired = service.preview_policy(request)
    assert expired['current_application_eligible'] is False
    assert any('政策已到复核日' in reason for reason in expired['application_blockers'])
    assert expired['candidates'] == initial['candidates']
