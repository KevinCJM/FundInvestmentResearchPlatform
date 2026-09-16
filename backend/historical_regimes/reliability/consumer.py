"""One calibrated-confidence gate shared by both TAA consumers.

The attachment is a transient validated read model, never written into old runs.
"""
from datetime import date, datetime, timedelta, timezone
import numpy as np
from custom_indicators.errors import IndicatorDomainError
from ..v2_contracts import parse_definition_v2
from .execution import model_binding_hash
from .references import resolve_reference
from . import kernels


def attach_calibration(run, service, *, publication_eligible=True):
    study = (run.get("definition") or {}).get("study")
    if not study:
        return run
    result = dict(run)  # Only attach a new context; frozen arrays/series stay read-only.
    context = {"error":None,"artifact":None,"qualification":None,"states":run["states"]}
    result["_reliability"] = context
    if study.get("purpose") != "realtime_recognition":
        context["error"] = "calibration_requires_realtime_study"
        return result
    reference = None
    if study.get("reference"):
        try:
            reference,_ = resolve_reference(service.graph,study["reference"],hydrate=False)
            context["states"] = reference["states"]
        except IndicatorDomainError as exc:
            context["error"] = "calibration_"+exc.code.lower()
            return result
    if not publication_eligible:
        context["error"] = "calibration_run_not_formally_published"
    elif not study.get("reference") or not study.get("calibration_id"):
        context["error"] = "calibration_missing"
    else:
        try:
            artifact = service.get(study["calibration_id"])
            report = artifact["report"]
            definition = parse_definition_v2(run["definition"])
            mapping = study.get("state_mapping") or {s["id"]:s["id"] for s in run["states"]}
            if (artifact["request"]["definition_id"] != run["definition_id"]
                    or artifact["request"]["reference"] != study["reference"]
                    or (not study.get("qualification_id")
                        and report["lineage"]["model_binding_hash"] != model_binding_hash(definition))
                    or report["lineage"]["state_mapping"] != mapping
                    or report["states"] != reference["states"]):
                context["error"] = "calibration_lineage_mismatch"
            elif study.get("qualification_id"):
                # Authenticate/re-score the forward journal once per run resolution,
                # never execute a full model again for each historical TAA date.
                qualification = service.graph.prospective.verify_qualification(
                    study["qualification_id"], study["calibration_id"], model_binding_hash(definition))
                context["qualification"] = qualification
                context["verified_at"] = service.graph.prospective._now().isoformat()
                context["artifact"] = artifact
            elif not report["calibration"]["deployment_eligible"]:
                context["error"] = "calibration_not_deployment_eligible"
            else:
                context["artifact"] = artifact
        except IndicatorDomainError as exc:
            context["error"] = "calibration_"+exc.code.lower()
    return result


def consumer_states(run):
    rows = run.get("_reliability",{}).get("states",run.get("states") or [])
    return [str(s.get("id")) for s in rows if s.get("id")]


def allocation_probabilities(run, probabilities):
    """Gate tilt weights only; unverified probability mass stays in SAA.

    Call after calibrated_output has authenticated the qualification. The full
    probability distribution remains available for display and confidence.
    """
    if probabilities is None:
        return None
    qualification = run.get("_reliability", {}).get("qualification")
    qualified = qualification.get("qualified_states") if qualification else None
    if qualified is None:
        return dict(probabilities)
    return {state: value if state in qualified else 0.0
            for state, value in probabilities.items()}


def calibrated_output(run, point, as_of, states):
    """Return (probabilities, chosen confidence, reason); explicit old contract."""
    from ..taa import _validated_probabilities
    study = (run.get("definition") or {}).get("study")
    if not study:
        probabilities,reason = _validated_probabilities(point.get("probabilities"),states)
        return probabilities,point.get("confidence"),reason
    context = run.get("_reliability")
    if not context:
        return None,None,"calibration_not_resolved"
    if context["error"]:
        return None,None,context["error"]
    artifact = context.get("artifact")
    qualification = context.get("qualification")
    if not artifact or (not qualification and artifact["report"]["calibration"]["deployment_eligible"] is not True):
        return None,None,"calibration_not_deployment_eligible"
    calibration = artifact["report"]["calibration"]
    if qualification:
        if qualification.get("status") != "qualified" or qualification.get("id") != study.get("qualification_id"):
            return None,None,"calibration_qualification_mismatch"
        decision = datetime.combine(date.fromisoformat(as_of), datetime.min.time(), timezone.utc)
        available = datetime.fromisoformat(qualification["available_from"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(qualification["expires_at"].replace("Z", "+00:00"))
        verified = datetime.fromisoformat(context["verified_at"].replace("Z", "+00:00"))
        if decision <= available:
            return None,None,"calibration_not_yet_available"
        if decision >= expires:
            return None,None,"calibration_expired"
        if decision > verified:
            return None,None,"calibration_future_decision"
        # Forward evidence authorizes a new validity window; it does not rewrite
        # the old candidate's expired diagnostic-only availability fields.
    else:
        if as_of < max(calibration["available_from"],(date.fromisoformat(artifact["created_at"][:10])+timedelta(days=1)).isoformat()):
            return None,None,"calibration_not_yet_available"
        if as_of > calibration["expires_on"]:
            return None,None,"calibration_expired"
    times = [point.get(key) for key in ("observation_date", "recognized_at", "effective_date")]
    if any(not isinstance(value, str) or len(value) != 10 for value in times):
        return None,None,"calibration_signal_time_missing"
    try:
        for value in times:
            date.fromisoformat(value)
    except ValueError:
        return None,None,"calibration_signal_time_invalid"
    if max(times) > as_of:
        return None,None,"calibration_signal_not_available"
    axis = [s["id"] for s in artifact["report"]["states"]]
    if states != axis:
        return None,None,"calibration_state_axis_mismatch"
    mapping = artifact["report"]["lineage"]["state_mapping"]
    chosen = mapping.get(point.get("state_id"))
    if chosen not in states:
        return None,None,"calibration_prediction_abstained"
    if qualification and qualification.get("qualified_states") is not None and chosen not in qualification.get("qualified_states", []):
        return None,None,"calibration_state_not_qualified"
    model_states = [s["id"] for s in run["states"]]
    source,reason = _validated_probabilities(point.get("probabilities"),model_states)
    if reason:
        return None,None,reason
    kernels.audit()
    raw = kernels.map_probability_kernel(np.asarray([[source[s] for s in model_states]],np.float64),
                                         np.asarray([states.index(mapping[s]) for s in model_states],np.int64),len(states))
    params = calibration["parameters"]
    q = kernels.apply_kernel(np.asarray([states.index(chosen)],np.int64),raw,
                             np.asarray(params["counts"],np.float64),
                             float(params["temperature"] or 1.0),1 if calibration["method"]=="temperature" else 0)
    if not np.isfinite(q).all():
        return None,None,"calibration_unavailable_for_predicted_class"
    confidence = float(q[0,states.index(chosen)])
    if confidence < calibration["confidence_floor"]:
        return None,confidence,"calibration_confidence_below_floor"
    return {s:float(q[0,i]) for i,s in enumerate(states)},confidence,None
