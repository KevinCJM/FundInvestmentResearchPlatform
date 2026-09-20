"""Candidate stress reuses actual published exposure and scenario math."""

import pytest
from backend.tests.test_published_risk_models import (
    environment,
    warmed,
    publish_model,
    direct_scenario,
    impact_request,
)
from backend.custom_indicators.errors import ValidationError
from backend.pre_investment.scenarios import ScenarioAdapter


def test_candidate_identity_weights_and_uncovered_holdings(environment):
    env = environment
    _, exposure = publish_model(env.model, env.fields)
    _, scenario = direct_scenario(env)
    target = {
        "kind": "implementation_candidate",
        "candidate_hash": "a" * 64,
        "holdings": [
            {"key": "fund:000001.OF", "weight": 0.6},
            {"key": "fund:000002.OF", "weight": 0.4},
        ],
    }
    result = env.scenarios.impact(
        impact_request(env, exposure, scenario, target=target)
    )
    assert result["target"]["candidate_hash"] == "a" * 64
    assert result["summary"]["terminal_return"] == pytest.approx(
        -0.1 * (0.6 * 0.7 + 0.4 * 0.3), abs=2e-4
    )
    assert result["transient"] and env.scenarios.impacts.list("impact") == []
    options = ScenarioAdapter(env.scenarios).options()
    assert options["scenarios"][0]["id"] == scenario["id"]
    target["holdings"][1]["key"] = "fund:000003.OF"
    with pytest.raises(ValidationError, match="全部产品"):
        env.scenarios.impact(impact_request(env, exposure, scenario, target=target))
