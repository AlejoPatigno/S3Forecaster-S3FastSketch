import inspect

import numpy as np
import pandas as pd

from s3paper import single_series_transfer_hpo as hpo


class FakeStudy:
    def __init__(self, params):
        self.best_params = dict(params)

    def trials_dataframe(self):
        return pd.DataFrame([self.best_params])


def synthetic_series_map(n=36):
    index = pd.RangeIndex(n)
    return {
        "S1": pd.Series(np.linspace(1.0, 10.0, n), index=index),
        "S2": pd.Series(np.linspace(2.0, 11.0, n), index=index),
        "S3": pd.Series(np.linspace(3.0, 12.0, n), index=index),
    }


def test_select_development_series_is_reproducible():
    series_map = synthetic_series_map()
    id_1 = hpo.select_development_series(series_map, seed=42)[0]
    id_2 = hpo.select_development_series(series_map, seed=42)[0]
    assert id_1 == id_2


def test_temporal_train_cal_test_split_preserves_order():
    series = pd.Series(np.arange(50.0), index=pd.RangeIndex(100, 150))
    train, calibration, test = hpo.temporal_train_cal_test_split(series)

    assert len(train) + len(calibration) + len(test) == len(series)
    assert train.index.max() < calibration.index.min()
    assert calibration.index.max() < test.index.min()
    assert train.index.equals(series.index[: len(train)])
    assert test.index.equals(series.index[-len(test) :])


def test_main_protocol_does_not_import_temporal_cv_folds():
    source = inspect.getsource(hpo)
    assert "make_expanding_window_folds" not in source


def test_transfer_protocol_calls_hpo_once_and_excludes_development(monkeypatch):
    calls = {"point": 0, "uq": 0, "evaluated_ids": []}
    point_params = {"ar_lags": 2, "prior_name": "causal_rolling_mean", "prior__window": 2}
    uq_params = {"aci_step_size": 0.05, "interval_scale": 1.0, "min_width_factor": 0.1}

    def fake_point(train, calibration, **kwargs):
        calls["point"] += 1
        assert len(train) > 0
        assert len(calibration) > 0
        return FakeStudy(point_params)

    def fake_uq(train, calibration, frozen_point_params, **kwargs):
        calls["uq"] += 1
        assert frozen_point_params == point_params
        return FakeStudy(uq_params)

    def fake_eval(train, test, frozen_point_params, *, uq_params=None, **kwargs):
        assert frozen_point_params == point_params
        assert uq_params == uq_params_expected
        forecast = pd.DataFrame(
            {
                "pred": test.to_numpy(dtype=float),
                "lower": test.to_numpy(dtype=float) - 1.0,
                "upper": test.to_numpy(dtype=float) + 1.0,
            },
            index=test.index,
        )
        return {
            "forecast": forecast,
            "metrics": {"mape": 0.0, "mase": 0.0, "msis": 1.0, "ecp": 1.0},
        }

    uq_params_expected = uq_params

    monkeypatch.setattr(hpo, "_model_functions", lambda model_name: (fake_point, fake_uq, fake_eval))

    result = hpo.run_single_series_hpo_transfer_experiment(
        synthetic_series_map(),
        model_name="S3Forecaster",
        dataset_name="synthetic",
        n_point_trials=2,
        n_uq_trials=2,
        seed=42,
        git_commit="test",
    )

    assert calls == {"point": 1, "uq": 1, "evaluated_ids": []}
    assert result["development_series_id"] not in result["evaluation_results"]["aggregate_series_ids"]
    assert len(result["evaluation_results"]["aggregate_series_ids"]) == 2
    assert result["metadata"]["cross_validation"] is False
    assert result["metadata"]["one_hpo_series_per_dataset"] is True
    assert result["metadata"]["hpo_repeated_per_series"] is False
