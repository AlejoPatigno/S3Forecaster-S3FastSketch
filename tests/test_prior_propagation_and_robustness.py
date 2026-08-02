import numpy as np
import pandas as pd
import pytest

from s3paper.baselines import TorchLinearBaseline, build_baseline
from s3paper.kaggle_workflow import NotebookRunConfig
from s3paper.fixed_baseline_aliases import normalize_fixed_baseline
from s3paper.multi_prior_robustness import CallableAutoregressivePrior, default_prior_factories, evaluate_prior_only
from s3paper.priors import (
    PRIOR_REGISTRY,
    RollingMeanPrior,
    HPO_PRIOR_NAMES,
    ChronosPrior,
    TimesFMPrior,
    override_only_prior,
    prior_params_from_namespace,
    suggest_prior_params,
    validate_prior_names,
)
from s3paper import single_series_transfer_hpo as transfer
from s3paper.s3_forecaster_experiment import optimize_s3_forecaster_holdout
from s3paper.s3_fastsketch_experiment import optimize_fastsketch_holdout


class ChoiceTrial:
    def __init__(self, choice):
        self.choice = choice

    def suggest_categorical(self, name, choices):
        return self.choice if name == "prior_name" else choices[-1]

    def suggest_int(self, name, low, high):
        return low


class Study:
    best_params = {"prior_name": "chronos"}


@pytest.fixture
def series_map():
    return {
        "a": pd.Series(np.arange(40.0)),
        "b": pd.Series(np.arange(40.0) + 1),
    }


def test_notebook_default_exposes_all_hpo_priors():
    assert NotebookRunConfig().prior_names == HPO_PRIOR_NAMES


def test_prior_names_reach_point_optimizer_and_metadata(monkeypatch, series_map):
    seen = {}

    def point(train, calibration, **kwargs):
        seen["prior_names"] = kwargs["prior_names"]
        return Study()

    def uq(*args, **kwargs):
        return type("UQ", (), {"best_params": {}})()

    def evaluate(train, test, params, **kwargs):
        frame = pd.DataFrame({"pred": test, "lower": test, "upper": test}, index=test.index)
        return {"forecast": frame, "metrics": {"mape": 0.0}}

    monkeypatch.setattr(transfer, "_model_functions", lambda _: (point, uq, evaluate))
    chosen = ("chronos", "timesfm")
    result = transfer.run_single_series_hpo_transfer_experiment(
        series_map,
        model_name="S3Forecaster",
        dataset_name="x",
        development_series_id="a",
        prior_names=chosen,
        git_commit="test",
    )
    assert seen["prior_names"] == chosen
    assert result["metadata"]["prior_names"] == list(chosen)



@pytest.mark.parametrize("optimizer", [optimize_s3_forecaster_holdout, optimize_fastsketch_holdout])
@pytest.mark.parametrize("prior_name", ["chronos", "timesfm"])
def test_foundation_priors_reach_both_holdout_objectives(monkeypatch, optimizer, prior_name):
    monkeypatch.setitem(PRIOR_REGISTRY, prior_name, lambda **kwargs: RollingMeanPrior(window_size=3))
    values = pd.Series(np.linspace(1.0, 5.0, 42))
    study = optimizer(values.iloc[:36], values.iloc[36:], n_trials=1, prior_names=(prior_name,))
    trial = study.trials[0]
    assert trial.user_attrs["prior_name"] == prior_name
    assert "error" not in trial.user_attrs
    assert study.user_attrs["prior_names"] == [prior_name]

def test_unknown_prior_fails_before_hpo():
    with pytest.raises(ValueError, match="Unknown priors"):
        validate_prior_names(["not-a-prior"])


def test_timesfm_uses_its_own_context_namespace():
    params = suggest_prior_params(ChoiceTrial("timesfm"), train_length=100, prior_names=["timesfm"])
    assert params["prior__timesfm_max_context"] == 1024
    assert "foundation_window" not in params
    assert prior_params_from_namespace(params)["max_context"] == 1024


def test_override_only_prior_preserves_model_params_and_removes_stale_prior_params():
    original = {"ar_lags": 4, "ridge_alpha": 0.1, "foundation_window": 6, "prior_name": "ets", "prior__ets_damped": True}
    changed = override_only_prior(original, {"prior_name": "timesfm", "prior__timesfm_max_context": 512})
    assert changed == {"ar_lags": 4, "ridge_alpha": 0.1, "prior_name": "timesfm", "prior__timesfm_max_context": 512}


def test_foundation_priors_use_causal_adapter_not_fit_predict(monkeypatch):
    predictor = lambda history, horizon: np.repeat(float(history.iloc[-1]), horizon)
    for prior in (ChronosPrior(predict_fn=predictor), TimesFMPrior(predict_fn=predictor)):
        adapter = prior._adapter()
        assert isinstance(adapter, CallableAutoregressivePrior)
        monkeypatch.setattr(adapter, "fit_predict", lambda *args: (_ for _ in ()).throw(AssertionError("fit_predict called")))
        output = evaluate_prior_only(adapter, pd.Series(np.arange(30.0)), pd.Series(np.arange(3.0)))
        assert len(output["forecast"]) == 3


def test_default_robustness_prior_set_is_complete():
    assert {"causal_rolling_mean", "seasonal_naive", "ets", "theta", "prophet", "chronos", "timesfm"}.issubset(default_prior_factories())


def test_robustness_collection_keeps_expected_cohort_and_counts_failures():
    train = {"a": pd.Series(np.arange(20.0)), "b": pd.Series(np.arange(20.0))}
    test = {"a": pd.Series(np.arange(2.0)), "b": pd.Series(np.arange(2.0))}

    class SelectivePrior:
        calls = 0

        def fit_predict(self, history, horizon):
            type(self).calls += 1
            if type(self).calls == 2:
                raise RuntimeError("expected failure")
            return history.copy(), pd.Series(np.repeat(history.iloc[-1], horizon))

    from s3paper.multi_prior_robustness import run_multi_prior_robustness
    result = run_multi_prior_robustness(train, test, {"selective": SelectivePrior})
    row = result["summary"].iloc[0]
    assert row["n_expected"] == 2
    assert row["n_success"] == 1
    assert row["n_failed"] == 1
    assert not row["complete_cohort"]
    assert set(result["details"]["series_id"]) == {"a", "b"}
    assert result["failures"].iloc[0]["exception_type"] == "RuntimeError"


def test_dlinear_factory_does_not_regress_to_nlinear():
    name, params = normalize_fixed_baseline("EasyTSF", {"model_type": "DLinear", "window_size": 8})
    model = build_baseline(name, params)
    assert isinstance(model, TorchLinearBaseline)
    assert model.architecture == "DLinear"


def test_default_linear_baselines_use_supported_window_parameter():
    from s3paper.kaggle_workflow import default_baseline_parameters

    parameters = default_baseline_parameters(seasonal_period=12)

    assert parameters["NLinear"]["window_size"] == 12
    assert parameters["DLinear"]["window_size"] == 12
    assert "input_window" not in parameters["NLinear"]
    assert "input_window" not in parameters["DLinear"]
    assert isinstance(
        build_baseline("NLinear", parameters["NLinear"]),
        TorchLinearBaseline,
    )
    assert isinstance(
        build_baseline("DLinear", parameters["DLinear"]),
        TorchLinearBaseline,
    )
