import copy
import pytest
from historical_regimes.v2_contracts import parse_definition_v2, definition_content_hash
from historical_regimes.v2_service import RegimeGraphV2Service
from historical_regimes.reliability.contracts import Policy, Study
from test_historical_regime_v2 import _definition


def test_old_hash_and_study_serialization():
    old = parse_definition_v2(_definition())
    payload = old.model_dump(mode="json")
    assert "study" not in payload
    assert definition_content_hash(old) == definition_content_hash(parse_definition_v2(payload))
    payload["study"] = {"purpose": "historical_reference", "family": "market_trend"}
    current = parse_definition_v2(payload)
    assert current.default_mode == "retrospective"
    with pytest.raises(Exception, match="研究用途"):
        RegimeGraphV2Service._validate_realtime_graph(current, "realtime")
    assert definition_content_hash(current) != definition_content_hash(old)


def test_no_recursive_reference_or_order_mapping():
    with pytest.raises(ValueError):
        Study(purpose="historical_reference", family="custom", calibration_id="x")
    payload = _definition()
    payload["study"] = {"purpose": "realtime_recognition", "family": "risk", "reference": {
        "run_id": "r", "publication_id": "p", "content_hash": "a" * 64}, "state_mapping": {"wrong": "a"}}
    with pytest.raises(Exception):
        parse_definition_v2(payload)


def test_policy_time_blocks_and_bounds():
    with pytest.raises(ValueError):
        Policy(calibration_end="2020-01-02", test_end="2020-01-01")
    with pytest.raises(ValueError):
        Policy(calibration_end="2020-01-02", bins=100)
